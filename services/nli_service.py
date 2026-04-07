from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import Tensor
from constants.enums import NLILabel
from pathlib import Path


class NLIService:
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
            if meta_path.exists():
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    adapter_path = Path(meta.get("current_path", ""))
                    if adapter_path.exists():
                        import logging
                        logging.getLogger(__name__).info(f"Auto-loading trained NLI adapter: {adapter_path}")
                        self.load_adapter(adapter_path)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error checking for NLI adapter: {e}")

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
            import logging
            logging.getLogger(__name__).error(f"Failed to load NLI adapter: {e}")
            return False

    @staticmethod
    def _get_nli_uncertainty(probs_tensor: Tensor) -> float:
        """
        Calculate classification uncertainty using Shannon Entropy.
        Higher entropy = flatter distribution = more uncertain.
        For 3 classes, max entropy is log(3) approx 1.0986.
        """
        # Add epsilon to avoid log(0)
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
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        label_id = int(torch.argmax(probs).item())
        return (
            self.LABEL_MAP[label_id],
            probs[label_id].item(),
            self._get_nli_uncertainty(probs),
        )

    def classify_nli_batch(
        self, premises: list[str] | str, hypotheses: list[str] | str
    ) -> list[tuple[NLILabel, float, float]]:
        """
        Batch classify NLI for multiple pairs.
        Supports:
        - One premise vs many hypotheses
        - Many premises vs one hypothesis
        """
        # Universal list conversion
        if isinstance(premises, str):
            premises = [premises]
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]

        if not hypotheses or not premises:
            return []

        # Broadcast if one side is single and other is multiple
        if len(premises) == 1 and len(hypotheses) > 1:
            premises = premises * len(hypotheses)
        elif len(hypotheses) == 1 and len(premises) > 1:
            hypotheses = hypotheses * len(premises)

        if len(premises) != len(hypotheses):
            raise ValueError(f"Batch mismatch: {len(premises)} premises vs {len(hypotheses)} hypotheses.")

        device = next(self.model.parameters()).device

        # Tokenize all pairs at once
        inputs = self.tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            all_probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for probs in all_probs:
            label_id = int(torch.argmax(probs).item())
            results.append(
                (
                    self.LABEL_MAP[label_id],
                    probs[label_id].item(),
                    self._get_nli_uncertainty(probs),
                )
            )

        return results
