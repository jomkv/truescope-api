from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import langdetect
import re


class TranslationService:
    def __init__(self):

        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

        self.tgt_lang = "eng_Latn"  # English target

    def protect_storm_names(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Protect Filipino storm names (e.g., "bagyong Ineng") from mistranslation.

        Returns:
            (protected_text, placeholder_map)
        """
        placeholder_map: dict[str, str] = {}

        def replace_match(match: re.Match) -> str:
            bagyo_word = match.group(1)
            quote = match.group(2) or ""
            name = match.group(3)

            placeholder = f"STORMNAME_{name.upper()}"
            placeholder_map[placeholder] = name
            return f"{bagyo_word} {quote}{placeholder}{quote}"

        protected_text = re.sub(
            r"\b(bagyo(?:ng)?)\s+([\"'“”]?)([A-Z][A-Za-z]+)\2",
            replace_match,
            text,
        )

        return protected_text, placeholder_map

    def restore_storm_names(self, text: str, placeholder_map: dict[str, str]) -> str:
        """
        Restore protected storm names after translation.
        """
        restored = text
        for placeholder, name in placeholder_map.items():
            restored = restored.replace(placeholder, name)
        return restored

    def detect_language(self, text: str) -> str:
        """
        Detect source language and return appropriate NLLB language code.
        """
        try:
            detected_lang = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            detected_lang = "unknown"

        # Map langdetect codes to NLLB language codes
        lang_map = {
            "en": "eng_Latn",  # English
            "tl": "tgl_Latn",  # Tagalog
            "fil": "fil_Latn",  # Filipino
            "es": "spa_Latn",  # Spanish
            "la": "lat_Latn",  # Latin
        }

        return lang_map.get(detected_lang, "fil_Latn")

    def translate_to_english(
        self, input_text: str, src_lang: str, max_length: int = 512
    ) -> str:
        """
        Translate text to English using NLLB-200.
        Handles Filipino/Tagalog and other languages with auto-detection.
        """
        protected_text, placeholder_map = self.protect_storm_names(input_text)

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            protected_text, return_tensors="pt", truncation=True, max_length=512
        )

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        generated_ids = outputs[0]
        translated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        translated_text = self.restore_storm_names(translated_text, placeholder_map)

        # If translation seems to have failed (still contains Filipino words) try it with tgl_Latn
        if src_lang == "fil_Latn" and self.has_untranslated_tagalog(translated_text):
            return self.translate_with_fallback(input_text, "tgl_Latn", max_length)

        return translated_text

    def translate_with_fallback(
        self, input_text: str, fallback_lang: str, max_length: int = 512
    ) -> str:
        """
        Retry translation with a different source language code.
        """
        self.tokenizer.src_lang = fallback_lang

        protected_text, placeholder_map = self.protect_storm_names(input_text)

        inputs = self.tokenizer(
            protected_text, return_tensors="pt", truncation=True, max_length=512
        )

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        generated_ids = outputs[0]
        translated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        translated_text = self.restore_storm_names(translated_text, placeholder_map)

        return translated_text

    def has_untranslated_tagalog(self, text: str) -> bool:
        """
        Check if text still contains common Tagalog words that should have been translated.
        """
        tagalog_markers = [
            "bagyo",
            "bagyong",
            "ang",
            "na",
            "sa",
            "mga",
            "ng",
            "ay",
            "pa",
            "rin",
            "din",
            "lang",
            "lamang",
            "ayon",
            "nitong",
            "noong",
        ]

        text_lower = text.lower()
        # Check for Tagalog function words that shouldn't appear in English
        return any(f" {word} " in f" {text_lower} " for word in tagalog_markers)
