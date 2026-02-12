import spacy


class EntityExtractionService:
    def __init__(self):
        model_name = "xx_ent_wiki_sm"
        self.entity_extractor = spacy.load(model_name)

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        doc = self.entity_extractor(text)
        entities = []

        for ent in doc.ents:
            entities.append((ent.text, ent.label_))

        return entities
