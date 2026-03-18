"""
Training Router
================
REST endpoints for triggering model fine-tuning, checking status,
and previewing training data derived from accumulated human feedback.
"""
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from services.feedback_trainer import FeedbackTrainer


router = APIRouter()

# Singleton trainer — shared across requests
_trainer = FeedbackTrainer()


class TrainRequest(BaseModel):
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4


# ──────────────────────────────────────────────
# Status & preview endpoints
# ──────────────────────────────────────────────

@router.get("/status")
async def training_status():
    """Return training status for all models (version, last trained, is_training, etc.)."""
    return JSONResponse(_trainer.get_status())


@router.get("/feedback/stats")
async def feedback_stats():
    """Return statistics about the accumulated feedback dataset."""
    return JSONResponse(_trainer.get_feedback_stats())


@router.get("/feedback/preview")
async def feedback_preview(n: int = 5):
    """Return sample training pairs for human inspection before training."""
    return JSONResponse(_trainer.preview_training_pairs(n=n))


# ──────────────────────────────────────────────
# Training endpoints (background tasks)
# ──────────────────────────────────────────────

@router.post("/nli")
async def train_nli(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger NLI LoRA fine-tuning in the background.
    Returns immediately; poll /training/status for progress.
    """
    status = _trainer.get_status()
    if status["nli"]["is_training"]:
        raise HTTPException(status_code=409, detail="NLI training already in progress.")

    stats = _trainer.get_feedback_stats()
    if stats["nli_training_pairs"] < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough training pairs ({stats['nli_training_pairs']}). Submit more feedback first."
        )

    def run_training():
        try:
            _trainer.fine_tune_nli(
                epochs=req.epochs,
                lr=req.learning_rate,
                batch_size=req.batch_size,
            )
            # Auto-reload into running services via global service references
            _auto_reload_nli()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"NLI training failed: {e}", exc_info=True)
            _trainer._training_status["nli"]["is_training"] = False
            _trainer._training_status["nli"]["log"].append(f"ERROR: {e}")

    background_tasks.add_task(run_training)
    return JSONResponse({"status": "started", "model": "nli", "config": req.model_dump()})


@router.post("/embeddings")
async def train_embeddings(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger embedding contrastive fine-tuning in the background.
    """
    status = _trainer.get_status()
    if status["embeddings"]["is_training"]:
        raise HTTPException(status_code=409, detail="Embedding training already in progress.")

    stats = _trainer.get_feedback_stats()
    if stats["embedding_triplets"] < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough triplets ({stats['embedding_triplets']}). Need feedback with both high and low relevance scores."
        )

    def run_training():
        try:
            # Grab the live model already loaded in memory — avoids slow re-init
            live_emb_model = None
            try:
                import main as app_main
                live_emb_model = app_main.verify_controller.embedding_service.model
            except Exception:
                pass

            _trainer.fine_tune_embeddings(
                epochs=req.epochs,
                lr=req.learning_rate,
                batch_size=req.batch_size,
                live_model=live_emb_model,
            )
            _auto_reload_embeddings()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Embedding training failed: {e}", exc_info=True)
            _trainer._training_status["embeddings"]["is_training"] = False
            _trainer._training_status["embeddings"]["log"].append(f"ERROR: {e}")

    background_tasks.add_task(run_training)
    return JSONResponse({"status": "started", "model": "embeddings", "config": req.model_dump()})


@router.post("/reload/nli")
async def reload_nli():
    """Manually hot-reload the NLI model from the latest saved adapter."""
    success = _auto_reload_nli()
    if not success:
        raise HTTPException(status_code=404, detail="No trained NLI adapter found. Train first.")
    return JSONResponse({"status": "reloaded", "model": "nli"})


@router.post("/reload/embeddings")
async def reload_embeddings():
    """Manually hot-reload the embedding model from the latest saved checkpoint."""
    success = _auto_reload_embeddings()
    if not success:
        raise HTTPException(status_code=404, detail="No trained embedding model found. Train first.")
    return JSONResponse({"status": "reloaded", "model": "embeddings"})


# ──────────────────────────────────────────────
# Internal: access global service singletons for hot-reload
# ──────────────────────────────────────────────

def _auto_reload_nli() -> bool:
    """Reload NLI adapter into the global verify_controller's NLI service."""
    try:
        import main as app_main
        nli_service = app_main.verify_controller.nli_service
        return _trainer.reload_nli_into_service(nli_service)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"NLI hot-reload skipped: {e}")
        return False


def _auto_reload_embeddings() -> bool:
    """Reload embedding model into the global verify_controller's embedding service."""
    try:
        import main as app_main
        emb_service = app_main.verify_controller.embedding_service
        return _trainer.reload_embeddings_into_service(emb_service)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Embedding hot-reload skipped: {e}")
        return False


def get_trainer() -> FeedbackTrainer:
    """Return the singleton trainer (used by main.py for startup auto-reload)."""
    return _trainer
