import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from constants.weights import score_to_label
from controllers.v1.verify_controller import VerifyController

DATASET_PATH = os.path.join(os.path.dirname(__file__), "test_dataset_generated.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "test_accuracy_results.json")


async def main():
    vc = VerifyController()
    with open(DATASET_PATH, encoding="utf-8") as f:
        claims = json.load(f)

    correct = 0
    total = 0
    results_list = []

    for entry in claims:
        claim_text = entry["claim"]
        ground_truth = entry["Ground_truth"]
        doc_id = entry["docID"]
        # Exclude evidence from the same doc_id
        result = await vc.verify_claim(claim_text, exclude_doc_ids=[doc_id])
        system_score = result["overall_verdict"]
        system_label = (
            score_to_label(system_score)
            if isinstance(system_score, (int, float))
            else str(system_score)
        )
        is_correct = system_label == ground_truth
        if is_correct:
            correct += 1
        total += 1
        results_list.append(
            {
                "claim": claim_text,
                "ground_truth": ground_truth,
                "predicted_label": system_label,
                "system_score": system_score,
                "docID": doc_id,
                "is_correct": is_correct,
            }
        )
        print(
            f"Claim: {claim_text}\nTrue: {ground_truth}, Pred: {system_label} (score: {system_score})\n"
        )

    accuracy = correct / total if total else 0
    output_data = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results_list,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
