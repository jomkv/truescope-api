from pathlib import Path

import spacy


class EntityExtractionService:
    def __init__(self):
        self.primary_extractor = self._load_model()

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        if not self.primary_extractor:
            return []

        doc = self.primary_extractor(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        if not entities:
            return []

        return entities

    def _load_model(self) -> spacy.language.Language | None:

        model_dir = (
            Path(__file__).resolve().parents[1]
            / "trained_model"
            / "entity_extractor_model"
        )
        if model_dir.exists():
            return spacy.load(model_dir)

        print(
            f"[entity_extraction] Model not found at {model_dir}. Please run train_scripts/entity_extractor_train.py to train the model."
        )
        return None
