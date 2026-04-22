import json
import logging
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import Tensor
from constants.enums import NLILabel
from pathlib import Path


logger = logging.getLogger(__name__)


class NLIService:
    @staticmethod
    def _get_env_int(name: str, default: int, min_value: int = 1) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
            if value < min_value:
                logger.warning(
                    "%s=%s is below minimum %s, using default %s",
                    name,
                    raw,
                    min_value,
                    default,
                )
                return default
            return value
        except ValueError:
            logger.warning(
                "%s=%s is invalid, expected integer. Using default %s",
                name,
                raw,
                default,
            )
            return default

    @staticmethod
    def _resolve_meta_path(raw_path: str) -> Path:
        return Path(raw_path.replace("\\", "/").strip())

    def __init__(self):
        # Set to 1 so each NLI worker thread uses exactly 1 CPU.
        # With NLI_MAX_THREADS=2 in the controller, this gives us 2 truly
        # parallel NLI jobs, each pinned to 1 vCPU — no thread contention.
        # The old default of 2 here with NLI_MAX_THREADS=1 used the same
        # 2 CPUs but serialized requests rather than parallelising them.
        self.torch_num_threads = self._get_env_int("NLI_TORCH_NUM_THREADS", 1)
        torch.set_num_threads(self.torch_num_threads)

        model_name = "joeddav/xlm-roberta-large-xnli"
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()

        # Check for latest trained adapter
        try:
            meta_path = Path("data/model_adapters/nli_meta.json")
            if not meta_path.exists():
                logger.warning(
                    "NLI meta file not found at %s; using base model.", meta_path
                )
            else:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                raw_path = str(meta.get("current_path", "")).strip()
                if not raw_path:
                    logger.warning("NLI meta has empty current_path; using base model.")
                else:
                    adapter_path = self._resolve_meta_path(raw_path)
                    if not adapter_path.exists():
                        logger.warning(
                            "NLI adapter path not found; using base model. raw_path='%s' normalized='%s' cwd='%s'",
                            raw_path,
                            adapter_path,
                            Path.cwd(),
                        )
                    elif not self.load_adapter(adapter_path):
                        logger.warning(
                            "NLI adapter load failed from %s; using base model.",
                            adapter_path,
                        )
                    else:
                        logger.warning("NLI adapter loaded from %s", adapter_path)
        except Exception as e:
            logger.error("Error checking for NLI adapter: %s", e)

        self.LABEL_MAP: dict[int, NLILabel] = {
            0: NLILabel.REFUTE,
            1: NLILabel.NEUTRAL,
            2: NLILabel.SUPPORT,
        }

    def load_adapter(self, adapter_path: str | Path) -> bool:
        """Hot-swap a LoRA-adapted model into this service."""
        try:
            from peft import PeftModel

            adapter_path = Path(adapter_path)
            if not (adapter_path / "adapter_config.json").exists():
                return False
            device = "cuda" if torch.cuda.is_available() else "cpu"
            base = AutoModelForSequenceClassification.from_pretrained(
                self._model_name, num_labels=3
            )
            model = PeftModel.from_pretrained(base, str(adapter_path))
            model = model.to(device)
            model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
            self.model = model
            return True
        except Exception as e:
            logger.error("Failed to load NLI adapter: %s", e)
            return False

    @staticmethod
    def _get_nli_uncertainty(probs_tensor: Tensor) -> float:
        """Shannon entropy — higher = more uncertain. Max is log(3) ≈ 1.0986 for 3 classes."""
        eps = 1e-9
        entropy = -torch.sum(probs_tensor * torch.log(probs_tensor + eps))
        return entropy.item()

    def classify_nli(
        self, premise: str, hypothesis: str
    ) -> tuple[NLILabel, float, float]:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].detach().clone()
        del outputs, inputs
        label_id = int(torch.argmax(probs).item())
        result = (
            self.LABEL_MAP[label_id],
            probs[label_id].item(),
            self._get_nli_uncertainty(probs),
        )
        del probs
        return result

    def classify_nli_batch(
        self, premises: list[str], hypothesis: str
    ) -> list[tuple[NLILabel, float, float]]:
        """Sequential per-pair inference (preserves numerical consistency vs batched padding)."""
        return [self.classify_nli(premise, hypothesis) for premise in premises]
