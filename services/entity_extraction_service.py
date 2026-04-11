import spacy


class EntityExtractionService:
    def __init__(self):
        self.entity_extractor = spacy.load("xx_ent_wiki_sm")

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        doc = self.entity_extractor(text)
        entities = []

        for ent in doc.ents:
            entities.append((ent.text, ent.label_))

        return entities
