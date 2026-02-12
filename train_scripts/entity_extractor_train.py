import json
import re
from pathlib import Path
import random

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

        # Create blank English model (training from scratch)
        nlp = spacy.blank("xx")

        # Add NER component
        ner = nlp.add_pipe("ner")
        ner.add_label("ENTITY")

        # Convert to spaCy Examples
        examples = []
        for text, spans in training_data:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, {"entities": spans}))

        # Initialize model
        optimizer = nlp.initialize(get_examples=lambda: examples)

        # Shuffle and split into train/dev
        random.shuffle(examples)
        split = int(len(examples) * 0.8)
        train_examples = examples[:split]
        dev_examples = examples[split:]

        print(
            f"[entity_extraction] Training with {len(train_examples)} examples, "
            f"validating on {len(dev_examples)} examples..."
        )

        best_dev_loss = float("inf")
        patience = 200
        patience_counter = 0
        max_epochs = 200

        for epoch in range(1, max_epochs + 1):
            print(f"[entity_extraction] Epoch {epoch}/{max_epochs}")

            # Shuffle training data
            random.shuffle(train_examples)

            # Train
            train_losses = {}
            batches = minibatch(
                train_examples,
                size=compounding(4.0, 32.0, 1.5),
            )

            for batch in batches:
                nlp.update(
                    batch,
                    sgd=optimizer,
                    drop=0.2,
                    losses=train_losses,
                )

            train_loss = train_losses.get("ner", 0.0)

            # Validate (NO weight updates)
            dev_losses = {}
            with nlp.select_pipes(enable=["ner"]):
                for batch in minibatch(dev_examples, size=8):
                    nlp.update(
                        batch,
                        sgd=None,  # 🚨 Prevent weight updates
                        drop=0.0,
                        losses=dev_losses,
                    )

            dev_loss = dev_losses.get("ner", 0.0)

            print(f"  Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f}")

            # Early stopping check
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                patience_counter = 0

                # Save best model checkpoint
                self.model_dir.mkdir(parents=True, exist_ok=True)
                nlp.to_disk(self.model_dir)
                print(f"  ✅ New best model saved (Dev Loss: {best_dev_loss:.4f})")

            else:
                patience_counter += 1
                print(f"  ⚠ No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"[entity_extraction] Early stopping at epoch {epoch}")
                break

        print(f"[entity_extraction] Training complete.")
        print(f"[entity_extraction] Best validation loss: {best_dev_loss:.4f}")

    def _build_training_data(self):
        dataset_paths = [
            Path(__file__).resolve().parents[1]
            / "tests"
            / "datasets"
            / f"train_dataset_{i}.json"
            for i in range(2, 7)
        ]

        training_data = []

        for path in dataset_paths:
            if not path.exists():
                print(f"[entity_extraction] Dataset not found: {path}")
                continue

            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            num_cases = len(data.get("test_cases", []))
            print(
                f"[entity_extraction] Loaded {path.name} " f"with {num_cases} examples"
            )

            for case in data.get("test_cases", []):
                text = case.get("claim", "")
                ground_truth = case.get("ground_truth", {})

                spans = []

                # Use aligned offsets if available
                if "entity_offsets" in ground_truth:
                    spans = ground_truth["entity_offsets"]
                else:
                    # Fallback: regex match
                    entities = ground_truth.get("expected_entities", [])
                    for entity in entities:
                        for match in re.finditer(
                            re.escape(entity),
                            text,
                            flags=re.IGNORECASE,
                        ):
                            spans.append((match.start(), match.end(), "ENTITY"))

                    spans = self._remove_overlapping_spans(spans)

                if spans:
                    training_data.append((text, spans))

        print(f"[entity_extraction] Total training examples: " f"{len(training_data)}")

        return training_data

    def _remove_overlapping_spans(self, spans):
        if not spans:
            return []

        sorted_spans = sorted(
            spans,
            key=lambda x: (x[0], -(x[1] - x[0])),
        )

        non_overlapping = []
        for span in sorted_spans:
            start, end, label = span
            overlaps = False

            for existing_start, existing_end, _ in non_overlapping:
                if start < existing_end and end > existing_start:
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(span)

        return sorted(non_overlapping, key=lambda x: x[0])


def main():
    trainer = EntityExtractorTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
