import json
import os
import datetime
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from controllers.v1.verify_controller import VerifyController

from tests.test_100_claims.test_cases import true_50, false_50

# Dynamically generate results file name with month, day, hour
parser = argparse.ArgumentParser()
parser.add_argument(
    "file_name",
    help="Suffix for the results file (e.g., 'mytest' for test_n_mytest.json)",
)
args = parser.parse_args()

now = datetime.datetime.now()
results_dir = os.path.join(os.path.dirname(__file__), "accuracy_results")
os.makedirs(results_dir, exist_ok=True)
RESULTS_PATH = os.path.join(results_dir, f"test_n_{args.file_name}.json")


def get_score_label(score: float) -> str:
    """
    Normalize a score in [-1, 1] to 'TRUE', 'NEUTRAL', or 'FALSE'.
    """
    if score > 0:
        return "TRUE"
    elif score == 0:
        return "NEUTRAL"
    else:
        return "FALSE"


def normalize_verdict_label(verdict: str) -> str:
    """
    Normalize a fine-grained verdict label to 'TRUE', 'NEUTRAL', or 'FALSE'.
    """
    true_labels = {"TRUE", "MOSTLY-TRUE", "HALF-TRUE"}
    neutral_labels = {"UNPROVEN", "UNKNOWN", "MISSING-CONTEXT", "OUTDATED", "SATIRE"}
    false_labels = {"MISLEADING", "MOSTLY-FALSE", "FALSE"}

    verdict_upper = verdict.upper()
    if verdict_upper in true_labels:
        return "TRUE"
    elif verdict_upper in false_labels:
        return "FALSE"
    else:
        return "NEUTRAL"


async def main():
    vc = VerifyController()

    combined_test_set = [*true_50, *false_50]

    correct = 0
    total = 0
    skipped = 0
    results_list = []

    # Confusion matrix counters
    TP = TN = FP = FN = 0

    # Count TRUE and FALSE expected labels, and check for duplicates
    expected_true = 0
    expected_false = 0
    seen_doc_ids = set()
    duplicate_doc_ids = set()
    for entry in combined_test_set:
        label = entry["expected_verdict"]
        doc_id = entry["doc_id"]
        if label == "TRUE":
            expected_true += 1
        elif label == "FALSE":
            expected_false += 1
        # Check for duplicates
        if doc_id in seen_doc_ids:
            duplicate_doc_ids.add(doc_id)
        else:
            seen_doc_ids.add(doc_id)

    for entry in combined_test_set:
        claim_text = entry["claim"]
        ground_truth = entry["expected_verdict"]
        doc_id = entry["doc_id"]
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
        is_correct = score_label == ground_truth
        if is_correct:
            correct += 1
        total += 1
        results_list.append(
            {
                "claim": claim_text,
                "ground_truth": ground_truth,
                "predicted_label": score_label,
                "system_score": system_score,
                "docID": doc_id,
                "is_correct": is_correct,
                "skipped": not result["results"],
                "result": result,
            }
        )
        print(
            f"Claim: {claim_text}\nTrue: {ground_truth}, Pred: {score_label} (score: {system_score})\n"
        )

        # Confusion matrix logic (only for TRUE/FALSE, skip NEUTRAL)
        if ground_truth == "TRUE" and score_label == "TRUE":
            TP += 1
        elif ground_truth == "FALSE" and score_label == "FALSE":
            TN += 1
        elif ground_truth == "FALSE" and score_label == "TRUE":
            FP += 1
        elif ground_truth == "TRUE" and score_label == "FALSE":
            FN += 1
        # NEUTRAL predictions are ignored for precision/recall/F1

    accuracy = correct / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    output_data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "expected_true": expected_true,
        "expected_false": expected_false,
        "duplicate_doc_ids": list(duplicate_doc_ids),
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "results": results_list,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1: {f1:.2%}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
