from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class RemarksGenerationService:
    def __init__(self):
        # Base Model for now, we can use other version if we want to improve quality over speed later
        model_name = "google/flan-t5-large"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate_remarks(
        self, input_text: str, claim: str, verdict_score: float, max_length: int = 128
    ) -> str:
        prompt = (
            f"It was said in this article that: {input_text}\n"
            f"Which {'contradicts' if 'not' in claim.lower() or 'false' in claim.lower() else 'supports or relates to'} the user claim: {claim}.\n"
            f"Final verdict score for this match: {verdict_score} (Scale: -1 = strongly refute, 0 = neutral, 1 = strongly support/true).\n"
            f"Summarize this relationship in a concise English remark. At the end, add a sentence explaining what the verdict score means in this context."
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
            )

        generated_ids = outputs[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text
