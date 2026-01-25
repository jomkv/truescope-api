from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import torch, Tensor


class NLIService:
    def __init__(self):
        model_name = "joeddav/xlm-roberta-large-xnli"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}

    @staticmethod
    def _get_average_probability(probs_tensor: Tensor) -> float:
        total_probs = (
            probs_tensor[0].item() + probs_tensor[1].item() + probs_tensor[2].item()
        )

        return total_probs / 3

    def classify_nli(self, premise: str, hypothesis: str) -> tuple[str, float, float]:
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        label_id = torch.argmax(probs).item()
        return (
            self.LABEL_MAP[label_id],
            probs[label_id].item(),
            self._get_average_probability(probs),
        )
