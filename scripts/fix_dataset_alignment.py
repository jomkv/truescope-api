"""
Fix entity alignment in datasets by using spaCy tokenizer.
This addresses the W030 warnings and ensures entities are properly aligned.
"""

import json
import spacy
from pathlib import Path
from typing import List, Tuple

# Load spaCy model (blank English model)
nlp = spacy.blank("en")


def find_entity_offsets(text: str, entity: str) -> List[Tuple[int, int]]:
    """
    Find all possible character offsets for an entity in text.
    Returns list of (start, end) tuples.
    """
    offsets = []
    text_lower = text.lower()
    entity_lower = entity.lower()

    start = 0
    while True:
        pos = text_lower.find(entity_lower, start)
        if pos == -1:
            break
        end = pos + len(entity)
        offsets.append((pos, end))
        start = pos + 1

    return offsets


def validate_entity_alignment(text: str, start: int, end: int) -> bool:
    """Check if entity span aligns with spaCy tokenization."""
    doc = nlp.make_doc(text)

    # Get token boundaries
    token_starts = {token.idx for token in doc}
    token_ends = {token.idx + len(token.text) for token in doc}

    # Check if start and end align with token boundaries
    return start in token_starts and end in token_ends


def build_training_entities(
    text: str, expected_entities: List[str]
) -> List[Tuple[int, int, str]]:
    """
    Build properly aligned entities for training.
    Returns list of (start, end, label) tuples.
    """
    entities = []
    covered_ranges = set()

    # Sort entities by length (longest first) to handle overlaps better
    sorted_entities = sorted(expected_entities, key=len, reverse=True)

    for entity in sorted_entities:
        offsets = find_entity_offsets(text, entity)

        for start, end in offsets:
            # Check if this range overlaps with already covered entities
            if any(
                covered_start <= start < covered_end
                or covered_start < end <= covered_end
                for covered_start, covered_end in covered_ranges
            ):
                continue

            # Try to align with token boundaries
            doc = nlp.make_doc(text)
            aligned_start = start
            aligned_end = end

            # Find nearest token boundaries
            for token in doc:
                if token.idx <= start < token.idx + len(token.text):
                    aligned_start = token.idx
                if token.idx < end <= token.idx + len(token.text):
                    aligned_end = token.idx + len(token.text)

            # Verify the alignment makes sense (not expanding too much)
            if aligned_end - aligned_start <= len(entity) + 10:  # Allow small tolerance
                entities.append((aligned_start, aligned_end, "ENTITY"))
                covered_ranges.add((aligned_start, aligned_end))
                break  # Use first valid offset for this entity

    # Sort by start position
    entities.sort(key=lambda x: x[0])

    # Remove overlaps
    cleaned_entities = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:
            cleaned_entities.append((start, end, label))
            last_end = end

    return cleaned_entities


def fix_dataset(dataset_path: Path) -> dict:
    """Load dataset and fix entity alignments."""
    print(f"Loading {dataset_path.name}...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_cases = len(data["test_cases"])
    fixed_count = 0

    # Fix each test case
    for i, case in enumerate(data["test_cases"]):
        if (i + 1) % 500 == 0:
            print(f"  Processing case {i + 1}/{total_cases}...")

        claim = case["claim"]
        expected_entities = case["ground_truth"]["expected_entities"]

        # Build training entities with proper alignment
        entities = build_training_entities(claim, expected_entities)

        # Store the aligned entity offsets for training
        case["ground_truth"]["entity_offsets"] = entities

        # Keep original expected_entities for reference
        case["ground_truth"]["expected_entities"] = expected_entities

        fixed_count += 1

    print(f"Fixed {fixed_count} cases in {dataset_path.name}")
    return data


def main():
    """Fix all datasets."""
    dataset_dir = Path("tests/datasets")
    dataset_files = [
        "test_dataset_2.json",
        "test_dataset_3.json",
        "test_dataset_4.json",
        "test_dataset_5.json",
        "test_dataset_6.json",
    ]

    for dataset_file in dataset_files:
        dataset_path = dataset_dir / dataset_file
        if not dataset_path.exists():
            print(f"⚠️  {dataset_file} not found, skipping...")
            continue

        # Fix the dataset
        fixed_data = fix_dataset(dataset_path)

        # Save the fixed dataset
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(fixed_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved fixed {dataset_file}\n")

    print("✅ All datasets fixed with proper entity alignment!")


if __name__ == "__main__":
    main()
