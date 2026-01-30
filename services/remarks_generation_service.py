from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import langdetect
from services.translation_service import TranslationService


class RemarksGenerationService:
    def __init__(self):

        model_name = "facebook/bart-large-cnn"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def generate_remarks(
        self, input_text: str, claim: str, verdict_score: float, max_length: int = 128
    ) -> str:
        # Detect language of input_text and translate if not in English
        try:
            lang = langdetect.detect(input_text)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = "unknown"

        if lang != "en":
            translator = TranslationService()
            input_text = translator.translate_to_english(input_text)

        prompt = (
            f"Article summary: {input_text}\n"
            f"User claim: {claim}\n"
            f"Verdict score: {verdict_score} (Scale: -1 = strongly refute, 0 = neutral, 1 = strongly support/true).\n"
            "Instructions:\n"
            "1. Write at least 3 clear, concise sentences summarizing how the article content relates to the user claim.\n"
            "2. Reference specific details from the article if possible.\n"
            "3. At the end, add a separate sentence that explains what the verdict score means in this context.\n"
            "Respond in fluent, natural translated English."
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
