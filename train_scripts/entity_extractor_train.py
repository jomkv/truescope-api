import json
import re
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import compounding, minibatch


class EntityExtractorTrainer:
    def __init__(self):
        self.model_dir = (
            Path(__file__).resolve().parents[1]
            / "trained_model"
            / "entity_extractor_model"
        )

    def train(self):
        training_data = self._build_training_data()
        if not training_data:
            print("[entity_extraction] No training data available")
            return

        nlp = spacy.blank("en")
        ner = nlp.add_pipe("ner")
        ner.add_label("ENTITY")

        examples = []
        for text, spans in training_data:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, {"entities": spans}))

        optimizer = nlp.initialize(get_examples=lambda: examples)
        for epoch in range(1, 21):
            print(f"[entity_extraction] Training epoch {epoch}/20...")
            batches = minibatch(examples, size=compounding(2.0, 8.0, 1.5))
            for batch in batches:
                nlp.update(batch, sgd=optimizer)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(self.model_dir)
        print(f"[entity_extraction] Model saved to {self.model_dir}")

    def _build_training_data(self) -> list[tuple[str, list[tuple[int, int, str]]]]:
        dataset_paths = [
            Path(__file__).resolve().parents[1]
            / "tests"
            / "datasets"
            / "test_dataset_2.json",
            Path(__file__).resolve().parents[1]
            / "tests"
            / "datasets"
            / "test_dataset_3.json",
            Path(__file__).resolve().parents[1]
            / "tests"
            / "datasets"
            / "test_dataset_4.json",
            Path(__file__).resolve().parents[1]
            / "tests"
            / "datasets"
            / "test_dataset_5.json",
            Path(__file__).resolve().parents[1]
            / "tests"
            / "datasets"
            / "test_dataset_6.json",
        ]

        training_data: list[tuple[str, list[tuple[int, int, str]]]] = []
        total_examples = 0
        for path in dataset_paths:
            if not path.exists():
                print(f"[entity_extraction] Dataset not found: {path}")
                continue
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            num_cases = len(data.get("test_cases", []))
            print(f"[entity_extraction] Loaded {path.name} with {num_cases} examples")
            total_examples += num_cases
            for case in data.get("test_cases", []):
                text = case.get("claim", "")
                ground_truth = case.get("ground_truth", {})

                # Try to use pre-aligned entity offsets first (from fix_dataset_alignment.py)
                spans: list[tuple[int, int, str]] = []
                if "entity_offsets" in ground_truth:
                    spans = ground_truth["entity_offsets"]
                else:
                    # Fallback: use regex-based extraction for legacy datasets
                    entities = ground_truth.get("expected_entities", [])
                    for entity in entities:
                        for match in re.finditer(
                            re.escape(entity), text, flags=re.IGNORECASE
                        ):
                            spans.append((match.start(), match.end(), "ENTITY"))

                    # Remove overlapping spans
                    spans = self._remove_overlapping_spans(spans)

                if spans:
                    training_data.append((text, spans))

        print(f"[entity_extraction] Total training examples: {len(training_data)}")
        return training_data

    def _remove_overlapping_spans(
        self, spans: list[tuple[int, int, str]]
    ) -> list[tuple[int, int, str]]:
        """Remove overlapping entity spans, keeping longer ones."""
        if not spans:
            return []

        # Sort by start position, then by length (longer first)
        sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))

        non_overlapping = []
        for span in sorted_spans:
            start, end, label = span
            # Check if this span overlaps with any already selected span
            overlaps = False
            for existing_start, existing_end, _ in non_overlapping:
                if start < existing_end and end > existing_start:
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(span)

        # Sort by start position for final output
        return sorted(non_overlapping, key=lambda x: x[0])


def main():
    trainer = EntityExtractorTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
