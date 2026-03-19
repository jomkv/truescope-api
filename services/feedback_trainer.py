"""
FeedbackTrainer Service
========================
Converts human feedback (from feedback.json) into model training data and
fine-tunes the NLI model (via LoRA) and the embedding model (via triplet loss).

GPU Strategy (11.9 GB VRAM):
  - NLI LoRA adapter: ~400 MB on GPU, rest of memory handles batch
  - Embedding triplet: MiniLM is tiny, fits easily
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

logger = logging.getLogger(__name__)

FEEDBACK_FILE = Path("feedback.json")
ADAPTERS_DIR = Path("data/model_adapters")
TRAINING_LOG = Path("data/training_log.json")
NLI_ADAPTERS_DIR = ADAPTERS_DIR / "nli"
EMB_ADAPTERS_DIR = ADAPTERS_DIR / "embeddings"

# Feedback-to-label thresholds
SUPPORT_GRADE_THRESH = 0.4
REFUTE_GRADE_THRESH = -0.4
MIN_RELATED = 0.5
POSITIVE_RELATED_THRESH = 0.7
NEGATIVE_RELATED_THRESH = 0.3

NLI_LABEL_MAP = {"REFUTE": 0, "NEUTRAL": 1, "SUPPORT": 2}
NLI_BASE_MODEL = "joeddav/xlm-roberta-large-xnli"
EMB_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FeedbackTrainer:
    """Manages feedback-driven fine-tuning for NLI and Embedding models."""

    def __init__(self):
        ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        NLI_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        EMB_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._training_status: dict[str, Any] = {
            "nli": {"is_training": False, "last_trained": None, "version": 0,
                    "num_pairs": 0, "last_loss": None, "log": []},
            "embeddings": {"is_training": False, "last_trained": None, "version": 0,
                          "num_pairs": 0, "last_loss": None, "log": []},
        }
        self._load_training_log()

    # ──────────────────────────────────────────────
    # Feedback loading & pair building
    # ──────────────────────────────────────────────

    def load_feedback(self) -> list[dict]:
        """Load and return all feedback sessions from feedback.json."""
        if not FEEDBACK_FILE.exists():
            return []
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_feedback_stats(self) -> dict:
        """Return statistics about the current feedback dataset."""
        sessions = self.load_feedback()
        nli_pairs = self.build_nli_training_pairs()
        emb_pairs = self.build_embedding_training_pairs()
        label_dist = {"SUPPORT": 0, "NEUTRAL": 0, "REFUTE": 0}
        for _, _, label in nli_pairs:
            label_dist[label] += 1
        return {
            "total_sessions": len(sessions),
            "total_evidence_items": sum(len(s.get("evidences", [])) for s in sessions),
            "nli_training_pairs": len(nli_pairs),
            "embedding_triplets": len(emb_pairs),
            "label_distribution": label_dist,
        }

    def build_nli_training_pairs(
        self,
    ) -> list[tuple[str, str, str]]:
        """
        Convert feedback sessions into NLI training pairs.
        Returns list of (premise=claim, hypothesis=evidence_text, label_str).
        """
        sessions = self.load_feedback()
        pairs: list[tuple[str, str, str]] = []

        for session in sessions:
            feedbacks = session.get("feedback", [])
            evidences = session.get("evidences", [])

            for fb, ev in zip(feedbacks, evidences):
                claim = ev.get("claim", "")
                text = ev.get("text", "")
                if not claim or not text:
                    continue

                related = float(fb.get("related", 0.5))
                grade = float(fb.get("grade", 0.0))

                if grade >= SUPPORT_GRADE_THRESH and related >= MIN_RELATED:
                    label = "SUPPORT"
                elif grade <= REFUTE_GRADE_THRESH and related >= MIN_RELATED:
                    label = "REFUTE"
                else:
                    label = "NEUTRAL"

                # Use the full text as hypothesis (truncate to first 300 chars for NLI)
                hypothesis = text[:300].strip()
                pairs.append((claim, hypothesis, label))

        # Balance classes to prevent severe overfitting, but allow a soft ratio (e.g. 3:1)
        # to utilize more of the available data.
        supports = [p for p in pairs if p[2] == "SUPPORT"]
        refutes = [p for p in pairs if p[2] == "REFUTE"]
        neutrals = [p for p in pairs if p[2] == "NEUTRAL"]
        
        # Determine the size of the minority class
        sizes = [len(supports), len(refutes), len(neutrals)]
        min_size = min([s for s in sizes if s > 0] or [0])
        
        if min_size > 0:
            import random
            
            # Allow majority class to be only slightly larger than minority class (1.2x)
            # to prevent NEUTRAL from becoming overly aggressive.
            max_size = int(min_size * 1.2)
            
            if len(supports) > max_size:
                supports = random.sample(supports, max_size)
            if len(refutes) > max_size:
                refutes = random.sample(refutes, max_size)
            if len(neutrals) > max_size:
                neutrals = random.sample(neutrals, max_size)
                
            pairs = supports + refutes + neutrals
            random.shuffle(pairs)

        return pairs

    def build_embedding_training_pairs(
        self,
    ) -> list[tuple[str, str, str]]:
        """
        Build triplet (anchor, positive, negative) pairs for contrastive training.
        Uses the claim as anchor; high-related evidence as positive, low-related as negative.
        """
        sessions = self.load_feedback()
        # Group by claim
        claim_evidences: dict[str, dict[str, list]] = {}

        for session in sessions:
            feedbacks = session.get("feedback", [])
            evidences = session.get("evidences", [])

            for fb, ev in zip(feedbacks, evidences):
                claim = ev.get("claim", "")
                text = ev.get("text", "")
                if not claim or not text:
                    continue
                related = float(fb.get("related", 0.5))

                if claim not in claim_evidences:
                    claim_evidences[claim] = {"positive": [], "negative": []}

                if related >= POSITIVE_RELATED_THRESH:
                    claim_evidences[claim]["positive"].append(text[:300])
                elif related <= NEGATIVE_RELATED_THRESH:
                    claim_evidences[claim]["negative"].append(text[:300])

        triplets: list[tuple[str, str, str]] = []
        for claim, groups in claim_evidences.items():
            for pos in groups["positive"]:
                for neg in groups["negative"]:
                    triplets.append((claim, pos, neg))

        return triplets

    def preview_training_pairs(self, n: int = 5) -> dict:
        """Return sample training pairs for inspection."""
        nli = self.build_nli_training_pairs()
        emb = self.build_embedding_training_pairs()
        return {
            "nli_sample": [
                {"claim": p, "evidence": h[:120], "label": l}
                for p, h, l in nli[:n]
            ],
            "embedding_sample": [
                {"anchor": a[:80], "positive": p[:80], "negative": n_[:80]}
                for a, p, n_ in emb[:n]
            ],
        }

    # ──────────────────────────────────────────────
    # NLI fine-tuning via LoRA
    # ──────────────────────────────────────────────

    def fine_tune_nli(
        self,
        epochs: int = 3,
        lr: float = 2e-4,
        batch_size: int = 4,
        on_log=None,
        max_pairs: int = 500000,
    ) -> str:
        """
        Fine-tune the NLI model using LoRA on accumulated feedback.
        Saves adapter to data/model_adapters/nli/v{N}/.
        Uses only the most recent `max_pairs` examples so training time stays
        bounded as feedback grows — old knowledge is baked into previous adapters.
        Returns the path to the saved adapter.
        """
        all_pairs = self.build_nli_training_pairs()
        if len(all_pairs) < 2:
            raise ValueError(
                f"Not enough training pairs ({len(all_pairs)}). Need at least 2. "
                "Submit more feedback first."
            )

        pairs = all_pairs[-max_pairs:] if len(all_pairs) > max_pairs else all_pairs

        with self._lock:
            self._training_status["nli"]["is_training"] = True
            self._training_status["nli"]["log"] = []

        try:
            def _log(msg: str):
                logger.info(f"[NLI Training] {msg}")
                self._training_status["nli"]["log"].append(msg)
                if on_log:
                    on_log(msg)

            _log(f"Building dataset from {len(pairs)} pairs…")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _log(f"Using device: {device}")

            tokenizer = AutoTokenizer.from_pretrained(NLI_BASE_MODEL)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                NLI_BASE_MODEL, num_labels=3
            )

            # Check for existing adapter to continue from — merge weights into
            # base_model so the next LoRA round starts from the previous best state.
            current_adapter = self._get_current_adapter_path("nli")
            if current_adapter and (current_adapter / "adapter_config.json").exists():
                _log(f"Continuing from previous adapter: {current_adapter}")
                peft_model = PeftModel.from_pretrained(base_model, str(current_adapter))
                base_model = peft_model.merge_and_unload()  # ← keep merged weights
                _log("Previous adapter merged — applying fresh LoRA on top")
            else:
                _log(f"No previous adapter found — training from base model {NLI_BASE_MODEL}")

            # Apply LoRA
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(base_model, lora_config)
            trainable, total = model.get_nb_trainable_parameters()
            _log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            model = model.to(device)

            # Build a simple torch Dataset
            texts_a = [p[0] for p in pairs]
            texts_b = [p[1] for p in pairs]
            labels = [NLI_LABEL_MAP[p[2]] for p in pairs]

            class NLIDataset(TorchDataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item

            encodings = tokenizer(
                texts_b, texts_a,  # Premise=text, Hypothesis=claim
                truncation=True, max_length=256, padding=True
            )
            hf_dataset = NLIDataset(encodings, labels)

            # Determine output path
            version = self._next_version("nli")
            out_path = NLI_ADAPTERS_DIR / f"v{version}"
            out_path.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(out_path),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=lr,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                fp16=(device == "cuda"),
                dataloader_pin_memory=(device == "cuda"),
                no_cuda=(device == "cpu"),
            )

            class LoggingTrainer(Trainer):
                def log(self_inner, logs: dict, *args, **kwargs):
                    super().log(logs, *args, **kwargs)
                    if "loss" in logs:
                        msg = f"Epoch {logs.get('epoch', '?'):.1f} — loss: {logs['loss']:.4f}"
                        _log(msg)
                        self._training_status["nli"]["last_loss"] = logs["loss"]

            trainer = LoggingTrainer(
                model=model,
                args=training_args,
                train_dataset=hf_dataset,
                data_collator=default_data_collator,
            )

            _log("Starting LoRA training…")
            trainer.train()

            # Save only the LoRA adapter (not full model)
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))
            _log(f"Adapter saved to {out_path}")

            # Update metadata
            self._set_current_adapter("nli", version)
            now = datetime.now(timezone.utc).isoformat()
            self._training_status["nli"].update({
                "is_training": False,
                "last_trained": now,
                "version": version,
                "num_pairs": len(pairs),
            })
            self._save_training_log("nli", version, len(pairs), now)
            _log(f"Training complete — adapter v{version}")
            return str(out_path)

        except Exception:
            self._training_status["nli"]["is_training"] = False
            raise

    # ──────────────────────────────────────────────
    # Embedding fine-tuning via triplet loss
    # ──────────────────────────────────────────────

    def fine_tune_embeddings(
        self,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 16,
        on_log=None,
        live_model=None,
        max_triplets: int = 500000,
    ) -> str:
        """
        Fine-tune the embedding model using triplet margin loss.
        Saves the full model to data/model_adapters/embeddings/v{N}/.
        Uses only the most recent `max_triplets` examples so training time
        stays bounded as feedback grows — old knowledge is already in the weights.
        """
        all_triplets = self.build_embedding_training_pairs()
        if len(all_triplets) < 2:
            raise ValueError(
                f"Not enough triplets ({len(all_triplets)}). Need at least 2. "
                "Submit feedback with both high and low relevance scores."
            )

        # Use only the most recent N triplets (feedback.json is append-only,
        # so the last entries are the newest)
        triplets = all_triplets[-max_triplets:] if len(all_triplets) > max_triplets else all_triplets

        with self._lock:
            self._training_status["embeddings"]["is_training"] = True
            self._training_status["embeddings"]["log"] = []

        try:
            def _log(msg: str):
                logger.info(f"[Embedding Training] {msg}")
                self._training_status["embeddings"]["log"].append(msg)
                if on_log:
                    on_log(msg)

            skipped = len(all_triplets) - len(triplets)
            if skipped > 0:
                _log(f"Dataset: {len(all_triplets)} total triplets → using newest {len(triplets)} (skipped {skipped} older ones already baked into weights)")
            else:
                _log(f"Building triplet dataset from {len(triplets)} triplets…")

            # Reuse the already-loaded model from the running service if supplied —
            # avoids re-downloading / re-initialising SentenceTransformer (~10s+ startup).
            if live_model is not None and isinstance(live_model, SentenceTransformer):
                _log("Reusing live embedding model from service (no reload needed)")
                model = live_model
            else:
                current_path = self._get_current_adapter_path("embeddings")
                if current_path and current_path.exists():
                    _log(f"Loading from existing checkpoint: {current_path}")
                    model = SentenceTransformer(str(current_path))
                else:
                    _log(f"Loading base model: {EMB_BASE_MODEL}")
                    model = SentenceTransformer(EMB_BASE_MODEL)

            examples = list(triplets)  # list of (anchor, positive, negative)
            version = self._next_version("embeddings")
            out_path = EMB_ADAPTERS_DIR / f"v{version}"
            out_path.mkdir(parents=True, exist_ok=True)

            _log(f"Training for {epochs} epochs on {len(examples)} triplets (lr={lr})…")

            # ── Optimized PyTorch triplet training with DataLoader & tqdm ──
            from torch.utils.data import DataLoader, Dataset
            from tqdm import tqdm
            
            backbone = model[0].auto_model   # transformer backbone
            tokenizer_emb = model.tokenizer
            device = next(backbone.parameters()).device
            backbone.train()

            optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr)
            loss_fn = torch.nn.TripletMarginLoss(margin=1.0)

            def mean_pool(model_output, attention_mask):
                token_emb = model_output.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            # Pre-tokenize everything at once instead of inside the training loop
            _log(f"Tokenizing {len(examples)} triplets... (this may take a minute)")
            
            anchors = [t[0] for t in examples]
            positives = [t[1] for t in examples]
            negatives = [t[2] for t in examples]
            
            class TripletDataset(Dataset):
                def __init__(self, anchors, positives, negatives, tokenizer):
                    self.anchors = anchors
                    self.positives = positives
                    self.negatives = negatives
                    self.tokenizer = tokenizer
                    
                def __len__(self):
                    return len(self.anchors)
                    
                def __getitem__(self, idx):
                    return {
                        "anchor": self.anchors[idx],
                        "positive": self.positives[idx],
                        "negative": self.negatives[idx]
                    }

            dataset = TripletDataset(anchors, positives, negatives, tokenizer_emb)
            
            # Use dataloader with a custom collate function to handle the batching efficiently
            def collate_fn(batch):
                batch_anchors = [item["anchor"] for item in batch]
                batch_positives = [item["positive"] for item in batch]
                batch_negatives = [item["negative"] for item in batch]
                
                # Tokenize exactly what is needed for this batch
                enc_a = tokenizer_emb(batch_anchors, padding=True, truncation=True, max_length=128, return_tensors="pt")
                enc_p = tokenizer_emb(batch_positives, padding=True, truncation=True, max_length=128, return_tensors="pt")
                enc_n = tokenizer_emb(batch_negatives, padding=True, truncation=True, max_length=128, return_tensors="pt")
                
                return enc_a, enc_p, enc_n

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

            _log(f"Starting Training: {len(dataloader)} batches per epoch.")
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                steps = 0
                
                # Wrap the dataloader in a tqdm progress bar
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

                for enc_a, enc_p, enc_n in progress_bar:
                    # Move batch to GPU/CPU
                    enc_a = {k: v.to(device) for k, v in enc_a.items()}
                    enc_p = {k: v.to(device) for k, v in enc_p.items()}
                    enc_n = {k: v.to(device) for k, v in enc_n.items()}

                    optimizer.zero_grad()
                    
                    # Forward pass
                    out_a = backbone(**enc_a)
                    out_p = backbone(**enc_p)
                    out_n = backbone(**enc_n)
                    
                    a_emb = mean_pool(out_a, enc_a["attention_mask"])
                    p_emb = mean_pool(out_p, enc_p["attention_mask"])
                    n_emb = mean_pool(out_n, enc_n["attention_mask"])

                    loss = loss_fn(a_emb, p_emb, n_emb)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    steps += 1
                    
                    # Update progress bar with the current loss
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                avg = epoch_loss / max(steps, 1)
                _log(f"Epoch {epoch}/{epochs} complete — Avg loss: {avg:.4f}")
                self._training_status["embeddings"]["last_loss"] = avg

            backbone.eval()
            model.save(str(out_path))
            _log(f"Model saved → {out_path}")

            self._set_current_adapter("embeddings", version)
            now = datetime.now(timezone.utc).isoformat()
            final_loss = self._training_status["embeddings"].get("last_loss")
            self._training_status["embeddings"].update({
                "is_training": False,
                "last_trained": now,
                "version": version,
                "num_pairs": len(triplets),
            })
            self._save_training_log("embeddings", version, len(triplets), now, last_loss=final_loss)
            _log(f"Training complete — model v{version} saved to {out_path}")
            return str(out_path)

        except Exception:
            self._training_status["embeddings"]["is_training"] = False
            raise

    # ──────────────────────────────────────────────
    # Hot-reload trained models into running services
    # ──────────────────────────────────────────────

    def reload_nli_into_service(self, nli_service) -> bool:
        """
        Hot-swap the LoRA-adapted NLI model into a running NLIService instance.
        Returns True if a new adapter was loaded, False if none exists yet.
        """
        adapter_path = self._get_current_adapter_path("nli")
        if not adapter_path or not (adapter_path / "adapter_config.json").exists():
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            NLI_BASE_MODEL, num_labels=3
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model = model.to(device)
        model.eval()

        # Swap in-place (thread-safe thanks to GIL for object reference swap)
        nli_service.tokenizer = tokenizer
        nli_service.model = model
        logger.info(f"NLI model hot-reloaded from {adapter_path}")
        return True

    def reload_embeddings_into_service(self, embedding_service) -> bool:
        """
        Hot-swap the fine-tuned embedding model into a running EmbeddingService instance.
        """
        model_path = self._get_current_adapter_path("embeddings")
        if not model_path or not model_path.exists():
            return False

        new_model = SentenceTransformer(str(model_path))
        embedding_service.model = new_model
        logger.info(f"Embedding model hot-reloaded from {model_path}")
        return True

    # ──────────────────────────────────────────────
    # Status & versioning helpers
    # ──────────────────────────────────────────────

    def get_status(self) -> dict:
        self._load_training_log()
        return {
            "nli": {
                **self._training_status["nli"],
                "base_model": NLI_BASE_MODEL,
                "log": self._training_status["nli"]["log"][-20:]
            },
            "embeddings": {
                **self._training_status["embeddings"],
                "base_model": EMB_BASE_MODEL,
                "log": self._training_status["embeddings"]["log"][-20:]
            },
        }

    def _next_version(self, model_type: str) -> int:
        meta = self._get_meta(model_type)
        return meta.get("version", 0) + 1

    def _set_current_adapter(self, model_type: str, version: int):
        meta_path = ADAPTERS_DIR / f"{model_type}_meta.json"
        meta = {"version": version, "current_path": str(ADAPTERS_DIR / model_type / f"v{version}")}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _get_current_adapter_path(self, model_type: str) -> Path | None:
        meta_path = ADAPTERS_DIR / f"{model_type}_meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        return Path(meta["current_path"])

    def _get_meta(self, model_type: str) -> dict:
        meta_path = ADAPTERS_DIR / f"{model_type}_meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def _save_training_log(self, model_type: str, version: int, num_pairs: int, timestamp: str, last_loss: float = None):
        log: list = []
        if TRAINING_LOG.exists():
            with open(TRAINING_LOG) as f:
                log = json.load(f)
        log.append({
            "model": model_type,
            "version": version,
            "num_pairs": num_pairs,
            "timestamp": timestamp,
            "last_loss": last_loss,
        })
        with open(TRAINING_LOG, "w") as f:
            json.dump(log, f, indent=2)

    def _load_training_log(self):
        """
        Restore last-known status from training_log.json on startup.
        Also forces is_training=False — if the server reloaded mid-training,
        the background thread is gone so 'training in progress' is always stale.
        """
        for mt in self._training_status:
            self._training_status[mt]["is_training"] = False
            self._training_status[mt]["version"] = 0
            self._training_status[mt]["last_trained"] = None
            self._training_status[mt]["num_pairs"] = 0
            self._training_status[mt]["last_loss"] = None

        if not TRAINING_LOG.exists():
            return
        with open(TRAINING_LOG) as f:
            log = json.load(f)
        for entry in reversed(log):
            mt = entry.get("model")
            if mt in self._training_status and self._training_status[mt]["version"] == 0:
                self._training_status[mt]["version"] = entry.get("version", 0)
                self._training_status[mt]["last_trained"] = entry.get("timestamp")
                self._training_status[mt]["num_pairs"] = entry.get("num_pairs", 0)
                self._training_status[mt]["last_loss"] = entry.get("last_loss")
                self._training_status[mt]["log"] = [
                    f"[Restored] v{entry.get('version')} trained on "
                    f"{entry.get('num_pairs',0)} pairs at {entry.get('timestamp','?')[:19]}"
                ]
