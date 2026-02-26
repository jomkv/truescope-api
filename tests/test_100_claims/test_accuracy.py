import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from controllers.v1.verify_controller import VerifyController

DATASET_PATH = os.path.join(os.path.dirname(__file__), "test_dataset_generated.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "test_accuracy_results.json")


def get_score_label(score: float) -> str:
    """
    Normalize a score in [-1, 1] to 'TRUE', 'NEUTRAL', or 'FALSE'.
    """
    if score > 0.33:
        return "TRUE"
    elif score < -0.33:
        return "FALSE"
    else:
        return "NEUTRAL"


def normalize_verdict_label(verdict: str) -> str:
    """
    Normalize a fine-grained verdict label to 'TRUE', 'NEUTRAL', or 'FALSE'.
    """
    true_labels = {"TRUE", "MOSTLY_TRUE", "HALF_TRUE"}
    neutral_labels = {"UNPROVEN", "UNKNOWN", "MISSING_CONTEXT", "OUTDATED", "SATIRE"}
    false_labels = {"MISLEADING", "MOSTLY_FALSE", "FALSE"}

    verdict_upper = verdict.upper()
    if verdict_upper in true_labels:
        return "TRUE"
    elif verdict_upper in false_labels:
        return "FALSE"
    else:
        return "NEUTRAL"


async def main():
    vc = VerifyController()
    with open(DATASET_PATH, encoding="utf-8") as f:
        claims = json.load(f)

    correct = 0
    total = 0
    skipped = 0
    results_list = []

    for entry in claims:
        claim_text = entry["claim"]
        ground_truth = entry["Ground_truth"]
        doc_id = entry["docID"]
        result = await vc.verify_claim(
            claim_text, exclude_doc_ids=[doc_id], exclude_articles=True
        )
        # Handle skipped evidence
        if not result["results"]:
            score_label = "NEUTRAL"
            system_score = None
            skipped += 1
        else:
            system_score = result["overall_verdict"]
            score_label = get_score_label(system_score)
        ground_truth_norm = normalize_verdict_label(ground_truth)
        is_correct = score_label == ground_truth_norm
        if is_correct:
            correct += 1
        total += 1
        results_list.append(
            {
                "claim": claim_text,
                "ground_truth": ground_truth_norm,
                "predicted_label": score_label,
                "system_score": system_score,
                "docID": doc_id,
                "is_correct": is_correct,
                "skipped": not result["results"],
            }
        )
        print(
            f"Claim: {claim_text}\nTrue: {ground_truth_norm}, Pred: {score_label} (score: {system_score})\n"
        )

    accuracy = correct / total if total else 0
    output_data = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "results": results_list,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
