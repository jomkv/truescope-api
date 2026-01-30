from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class TranslationService:
    def __init__(self):
        model_name = (
            "Helsinki-NLP/opus-mt-mul-en"  # Multilingual to English translation model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def translate_to_english(self, input_text: str, max_length: int = 512) -> str:
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
            )

        generated_ids = outputs[0]
        translated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return translated_text
