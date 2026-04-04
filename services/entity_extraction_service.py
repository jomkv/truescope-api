import spacy
from constants.tokens import COMMON_STOPWORDS, ENTITY_GENERIC_TOKENS


class EntityExtractionService:
    def __init__(self):
        self.entity_extractor = spacy.load("xx_ent_wiki_sm")

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """
        Extracts named entities using the spacy model.
        Includes a secondary 'Title Case' extraction pass if the input is predominantly lowercase,
        as smaller spaCy models often fail to recognize lowercase entities.
        """
        if not text:
            return []

        # 0. Preparation
        stop_lower = {s.lower() for s in COMMON_STOPWORDS}
        generic_lower = {g.lower() for g in ENTITY_GENERIC_TOKENS}

        # 1. Primary Extraction
        doc = self.entity_extractor(text)
        entities = {ent.text.strip(): ent.label_ for ent in doc.ents}

        # 2. Secondary Extraction (Re-Case Augmentation)
        # If the input is primarily lowercase, we try again with a Title-Cased version.
        is_mostly_lower = sum(1 for c in text if c.islower()) > (len(text) * 0.4)
        if is_mostly_lower:
            # Selective Title-Case: Only capitalize words that aren't stopwords or generic descriptors,
            # and avoid common verbs like "is" or "caused".
            words = text.split()
            title_words = []
            for w in words:
                w_low = w.lower()
                if w_low in stop_lower or w_low in {"is", "caused", "be", "was", "has"} or len(w) <= 2:
                    title_words.append(w_low)
                else:
                    title_words.append(w.capitalize())
            
            text_augmented = " ".join(title_words)
            doc_title = self.entity_extractor(text_augmented)
            for ent in doc_title.ents:
                ent_text = ent.text.strip()
                # Use a case-insensitive check to avoid duplicates
                if not any(ent_text.lower() == existing.lower() for existing in entities):
                    entities[ent_text] = ent.label_

        # 3. Filter & Cleanup
        # Remove entities that are just generic tokens, single stopwords, or phrases 
        # made entirely of generic words (e.g. "super typhoon").
        filtered_entities = []
        for ent_text, label in entities.items():
            ent_low = ent_text.lower().strip()
            tokens = ent_low.split()
            
            # 1. Skip if the entire text is a single stopword or generic token
            if ent_low in stop_lower or ent_low in generic_lower:
                continue
                
            # 2. Skip if it's too short (garbage tokens like "I" or "X")
            if len(ent_text) <= 1:
                continue

            # 3. Filter if EVERY word in a multi-word phrase is generic/stopword
            # This catches "Super Typhoon" but keeps "Super Typhoon Uwan"
            is_all_generic = all(t in generic_lower or t in stop_lower for t in tokens)
            if is_all_generic:
                continue
                
            filtered_entities.append((ent_text, label))

        # 4. Fallback: If no entities found, use the whole text as a pseudo-entity
        if not filtered_entities and len(text.strip()) > 3:
            # We Title-Case the fallback to help with downstream matching
            fallback_text = text.strip().title()
            filtered_entities.append((fallback_text, "MISC"))

        return filtered_entities
