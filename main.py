import json
import os
import logging
from datetime import datetime
from pathlib import Path

from fastapi import Request, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from core import server

app = server.app

logger = logging.getLogger(__name__)

# Setup templates
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

# ──────────────────────────────────────────────
# Controller (shared across simulation + hot-reload)
# ──────────────────────────────────────────────
from controllers.v1.verify_controller import VerifyController

verify_controller = VerifyController()

# ──────────────────────────────────────────────
# Auto-reload trained model adapters on startup
# ──────────────────────────────────────────────

@app.on_event("startup")
async def auto_reload_adapters():
    """On startup, reload any previously trained model adapters."""
    try:
        from routers.v1.training_router import get_trainer
        trainer = get_trainer()

        nli_ok = trainer.reload_nli_into_service(verify_controller.nli_service)
        emb_ok = trainer.reload_embeddings_into_service(verify_controller.embedding_service)

        if nli_ok:
            logger.info("✅ NLI adapter auto-loaded on startup.")
        if emb_ok:
            logger.info("✅ Embedding model auto-loaded on startup.")
    except Exception as e:
        logger.warning(f"Startup adapter auto-load skipped: {e}")


# ──────────────────────────────────────────────
# Simulation / HITL web interface
# ──────────────────────────────────────────────

@app.get("/simulation", response_class=HTMLResponse)
async def simulation_page(request: Request):
    return templates.TemplateResponse("simulation.html", {"request": request})


@app.post("/simulation/verify")
async def simulation_verify(data: dict):
    claim = data.get("claim", "")
    if not claim:
        return JSONResponse({"evidences": [], "overall_verdict": None, "is_negated": False})

    raw_limit = data.get("aggregation_limit")
    aggregation_limit = None
    if raw_limit is not None and raw_limit != "":
        try:
            aggregation_limit = int(raw_limit)
        except (ValueError, TypeError):
            aggregation_limit = None

    result = await verify_controller.verify_claim(claim, aggregation_limit=aggregation_limit)
    claim_entities = verify_controller.extract_entities(claim)

    evidences = []

    def art_to_dict(art, skipped=False):
        nli_result = getattr(art, "nli_result", None)
        if nli_result and hasattr(nli_result, "model_dump"):
            nli_result = nli_result.model_dump()
        return {
            "claim": claim,
            "entities": claim_entities,
            "text": art.content,
            "verdict": art.verdict if art.verdict is not None else None,
            "link": art.url,
            "nli_result": nli_result,
            "found_claim": getattr(art, "found_claim", None),
            "found_verdict": getattr(art, "found_verdict", None),
            "similarity_score": getattr(art, "similarity_score", None),
            "entity_match_score": getattr(art, "entity_match_score", None),
            "combined_relevance_score": getattr(art, "combined_relevance_score", None),
            "source": getattr(art, "source", None),
            "source_type": getattr(art, "source_type", None),
            "source_bias": getattr(art, "source_bias", None),
            "chunk_texts": getattr(art, "chunk_texts", None),
            "publish_date": getattr(art, "publish_date", None),
            "skip_reason": getattr(art, "skip_reason", []),
            "skipped": skipped,
            "is_aggregated": getattr(art, "is_aggregated", True),
        }

    for art in result.get("results", []):
        evidences.append(art_to_dict(art, skipped=False))

    # Include skipped results so users can grade them as negative training examples
    for art in result.get("skipped", []):
        evidences.append(art_to_dict(art, skipped=True))

    return JSONResponse({
        "evidences": evidences,
        "overall_verdict": result.get("overall_verdict"),
        "truth_confidence_score": result.get("truth_confidence_score"),
        "is_negated": result.get("is_negated", False),
    })


# ──────────────────────────────────────────────
# Feedback storage
# ──────────────────────────────────────────────

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback.json")


@app.post("/simulation/feedback")
async def simulation_feedback(data: dict):
    feedback = data.get("feedback", [])
    evidences = data.get("evidences", [])
    expected_entities = data.get("expected_entities", "")
    user_verdict = data.get("user_verdict", None)  # NEW: overall human label

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "feedback": feedback,
        "evidences": evidences,
        "expected_entities": expected_entities,
        "user_verdict": user_verdict,
    }
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                all_feedback = json.load(f)
        else:
            all_feedback = []
        all_feedback.append(entry)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(all_feedback, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
    return JSONResponse({"status": "success"})


# ──────────────────────────────────────────────
# Threshold configuration endpoints
# ──────────────────────────────────────────────

@app.get("/simulation/thresholds")
async def get_thresholds():
    """Return current relevance threshold values from the running controller."""
    return JSONResponse({
        "relevance_threshold": verify_controller.RELEVANCE_THRESHOLD,
        "entity_threshold": verify_controller.ENTITY_THRESHOLD,
        "combined_threshold": verify_controller.COMBINED_THRESHOLD,
        "max_deep_analysis": verify_controller.MAX_DEEP_ANALYSIS,
        "aggregation_limit": verify_controller.AGGREGATION_LIMIT,
    })


@app.post("/simulation/thresholds")
async def set_thresholds(data: dict):
    """
    Update relevance thresholds on the live controller instance.
    All keys are optional — only provided keys are updated.
    """
    updated = {}
    if "relevance_threshold" in data:
        verify_controller.RELEVANCE_THRESHOLD = float(data["relevance_threshold"])
        updated["relevance_threshold"] = verify_controller.RELEVANCE_THRESHOLD
    if "entity_threshold" in data:
        verify_controller.ENTITY_THRESHOLD = float(data["entity_threshold"])
        updated["entity_threshold"] = verify_controller.ENTITY_THRESHOLD
    if "combined_threshold" in data:
        verify_controller.COMBINED_THRESHOLD = float(data["combined_threshold"])
        updated["combined_threshold"] = verify_controller.COMBINED_THRESHOLD
    if "max_deep_analysis" in data:
        verify_controller.MAX_DEEP_ANALYSIS = int(data["max_deep_analysis"])
        updated["max_deep_analysis"] = verify_controller.MAX_DEEP_ANALYSIS
    if "aggregation_limit" in data:
        val = data["aggregation_limit"]
        verify_controller.AGGREGATION_LIMIT = int(val) if val else None
        updated["aggregation_limit"] = verify_controller.AGGREGATION_LIMIT
    return JSONResponse({"status": "updated", "values": updated})
