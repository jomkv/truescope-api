from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class NLIService:
  def __init__(self):
    model_name = "joeddav/xlm-roberta-large-xnli"

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    self.model.eval()

    self.LABEL_MAP = {
        0: "contradiction",
        1: "neutral",
        2: "entailment"
    }

  def classify_nli(self, premise: str, hypothesis: str):
    inputs = self.tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    label_id = torch.argmax(probs).item()
    return self.LABEL_MAP[label_id], probs[label_id].item()