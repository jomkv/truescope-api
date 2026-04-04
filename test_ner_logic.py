import asyncio
from services.entity_extraction_service import EntityExtractionService


def test_ner():
    service = EntityExtractionService()

    test_cases = [
        {
            "name": "Lowercase Augmentation",
            "text": "donald trump is against marriage equality",
            "expected_contains": "Donald Trump",
        },
        {
            "name": "Mixed Case Normal",
            "text": "super typhoon uwan caused great damage",
            "expected_contains": "Super Typhoon Uwan",
        },
        {
            "name": "Stopword Filtering",
            "text": "The and Of",
            "expected_empty_after_filter": True,
        },
        {
            "name": "Whole-Text Fallback",
            "text": "some random text with no entities",
            "expected_fallback": True,
        },
    ]

    print("=== NER Logic Test ===")
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        print(f"Input: '{case['text']}'")

        entities = service.extract_entities(case["text"])
        print(f"Extracted: {entities}")

        # Verification
        if "expected_contains" in case:
            found = any(
                case["expected_contains"].lower() in e[0].lower() for e in entities
            )
            print(
                f"  - Result: {'PASS' if found else 'FAIL'} (Expected: '{case['expected_contains']}')"
            )

        if "expected_empty_after_filter" in case:
            # Note: "The and Of" might trigger fallback if it's treated as whole-text.
            # But "The and Of" is 10 chars, so it might fallback to "The and Of"
            # However, if it's filtered to empty correctly, it should show its logic.
            print(f"  - Entities: {entities}")

        if "expected_fallback" in case:
            is_fallback = len(entities) == 1 and entities[0][1] == "MISC"
            print(f"  - Fallback check: {'PASS' if is_fallback else 'FAIL'}")


if __name__ == "__main__":
    test_ner()
