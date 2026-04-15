import spacy
import ahocorasick
import pickle
from typing import List, Tuple


def is_word_boundary(text, start, end):
    # Check if the character before and after the match is not a letter or digit
    before = (start == 0) or (not text[start - 1].isalnum())
    after = (end == len(text) - 1) or (not text[end + 1].isalnum())
    return before and after


class EntityExtractionService:
    def __init__(self, use_gazette=True, gazette_pkl_path="constants/gazetteer.pkl"):
        self.entity_extractor = spacy.load("xx_ent_wiki_sm")
        self.use_gazette = use_gazette

        if self.use_gazette:
            # Load gazetteer from pickle file
            with open(gazette_pkl_path, "rb") as f:
                gazetteer = pickle.load(f)

            # Build Aho-Corasick automaton with lowercased keywords
            self.automaton = ahocorasick.Automaton()
            for keyword in gazetteer:
                self.automaton.add_word(keyword, keyword)
            self.automaton.make_automaton()

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        doc = self.entity_extractor(text)
        entities = set()

        # Add spaCy NER results and record their spans
        ner_spans = set()
        for ent in doc.ents:
            entities.add((ent.text, ent.label_))
            ner_spans.update(range(ent.start_char, ent.end_char))

        if self.use_gazette:
            lowered_text = text.lower()

            matches = []
            for end_idx, keyword in self.automaton.iter(lowered_text):
                start_idx = end_idx - len(keyword) + 1
                if is_word_boundary(lowered_text, start_idx, end_idx):
                    matches.append(
                        (start_idx, end_idx, text[start_idx : end_idx + 1], keyword)
                    )

            # Sort by length (longest first), then by start index
            matches.sort(key=lambda x: (-(x[1] - x[0] + 1), x[0]))

            # Select non-overlapping matches
            occupied = set()
            for start_idx, end_idx, matched_text, keyword in matches:
                # Skip if overlaps with any NER span
                if any(i in ner_spans for i in range(start_idx, end_idx + 1)):
                    continue
                # Skip if overlaps with already selected gazetteer matches
                if any(i in occupied for i in range(start_idx, end_idx + 1)):
                    continue
                entities.add((matched_text, "GAZ"))
                occupied.update(range(start_idx, end_idx + 1))

        return list(entities)


# Used for unit testing this service on a sample claim
if __name__ == "__main__":
    service = EntityExtractionService()
    sample_claim = "Donald Trump is against marriage equality."
    print(service.extract_entities(sample_claim))
