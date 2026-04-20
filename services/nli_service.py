import json
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import Tensor
from constants.enums import NLILabel
from pathlib import Path


logger = logging.getLogger(__name__)


class NLIService:
    @staticmethod
    def _resolve_meta_path(raw_path: str) -> Path:
        # Normalize Windows-style separators so Linux containers can resolve paths.
        normalized = raw_path.replace("\\", "/").strip()
        return Path(normalized)

    def __init__(self):
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
        """
        Hot-swap a LoRA-adapted model into this service.
        Used by FeedbackTrainer.reload_nli_into_service().
        Returns True if loaded successfully.
        """
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
        """
        Calculate classification uncertainty using Shannon Entropy.
        Higher entropy = flatter distribution = more uncertain.
        For 3 classes, max entropy is log(3) approx 1.0986.
        """
        eps = 1e-9
        entropy = -torch.sum(probs_tensor * torch.log(probs_tensor + eps))
        return entropy.item()

    def classify_nli(
        self, premise: str, hypothesis: str
    ) -> tuple[NLILabel, float, float]:
        torch.set_num_threads(2)
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].detach().clone()
        # Explicitly delete intermediate tensors to help GC under concurrent load
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
        """
        Run single-pair inference for each premise sequentially.

        We intentionally do NOT pad-batch here. xlm-roberta-large produces
        slightly different logits per sample when batched with padding vs run
        alone, which is enough to flip borderline verdicts and degrade accuracy.
        Looping classify_nli() is numerically identical to the original
        sequential path while still consolidating the call site.
        """
        return [self.classify_nli(premise, hypothesis) for premise in premises]
