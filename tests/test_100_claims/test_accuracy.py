import json
import os
import datetime
import sys
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from controllers.v1.verify_controller import VerifyController

DATASET_PATH = os.path.join(os.path.dirname(__file__), "negatedClaims.json")
# Add timestamp to results to avoid overwriting previous runs
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), f"test_accuracy_results_{_timestamp}.json"
)


def get_score_label(score: float) -> str:
    """
    Normalize a score in [-1, 1] to 'TRUE' or 'FALSE'.
    """
    if score >= 0:
        return "TRUE"
    else:
        return "FALSE"


def normalize_verdict_label(verdict: str) -> str:
    """
    Normalize a fine-grained verdict label to 'TRUE' or 'FALSE'.
    """
    true_labels = {"TRUE", "MOSTLY_TRUE", "HALF_TRUE"}
    # The user requested everything else to fall under FALSE, as neutral is no longer evaluated.
    verdict_upper = verdict.upper()
    if verdict_upper in true_labels:
        return "TRUE"
    else:
        return "FALSE"


async def main():
    print("Initializing VerifyController...", flush=True)
    vc = VerifyController()
    print("VerifyController initialized.", flush=True)

    # Auto-load latest HITL models for testing
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    log_path = os.path.join(project_root, "data", "training_log.json")
    if os.path.exists(log_path):

        with open(log_path) as f:
            logs = json.load(f)
            # Load NLI adapter
            nli_log = next((l for l in reversed(logs) if l["model"] == "nli"), None)
            if nli_log and nli_log["version"] > 0:
                v = nli_log["version"]
                nli_path = os.path.join(project_root, f"data/model_adapters/nli/v{v}")
                if os.path.exists(nli_path):
                    print(f"Loading NLI adapter v{v} for testing...")
                    vc.nli_service.load_adapter(nli_path)

            # Load Embedding model
            emb_log = next(
                (l for l in reversed(logs) if l["model"] == "embeddings"), None
            )
            if emb_log and emb_log["version"] > 0:
                v = emb_log["version"]
                emb_path = os.path.join(
                    project_root, f"data/model_adapters/embeddings/v{v}"
                )
                if os.path.exists(emb_path):
                    print(f"Loading Embedding model v{v} for testing...")
                    vc.embedding_service.reload_model(emb_path)

    print(f"Loading dataset from {DATASET_PATH}...", flush=True)
    with open(DATASET_PATH, encoding="utf-8") as f:
        claims = json.load(f)
    print(f"Dataset loaded. Total claims: {len(claims)}", flush=True)

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
        # Handle various field names for ground truth in different test versions
        ground_truth = entry.get(
            "expected_verdict", entry.get("ground_truth", entry.get("Ground_truth"))
        )
        claim_id = entry["index"]
        # If the test case specifies a doc_id to exclude, pass it to the search
        exclude_docs = [entry["doc_id"]] if "doc_id" in entry else []

        # Perform a full search identical to the live /simulation/verify endpoint
        result = await vc.verify_claim(claim_text, exclude_doc_ids=exclude_docs)
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
        # Only include in accuracy if system_score is not None
        if system_score is not None:
            total += 1
            if is_correct:
                correct += 1
        results_list.append(
            {
                "claim": claim_text,
                "ground_truth": ground_truth,
                "predicted_label": score_label,
                "system_score": system_score,
                "id": claim_id,
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
    metrics = compute_metrics(results_list)
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
        "metrics": metrics,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1: {f1:.2%}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to {RESULTS_PATH}")
    print("Class metrics:")
    for cls, vals in metrics.items():
        print(
            f"{cls}: Precision={vals['precision']}, Recall={vals['recall']}, F1-Score={vals['f1-score']}"
        )

    # Compute metrics
    metrics = compute_metrics(results_list)
    print(f"Metrics: {metrics}")


def compute_metrics(results_list):
    # Map normalized labels to class names
    label_map = {"TRUE": "Support", "FALSE": "Refute"}
    classes = ["Support", "Refute"]
    # Initialize counters
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for entry in results_list:
        true_label = label_map.get(entry["ground_truth"], entry["ground_truth"])
        pred_label = label_map.get(entry["predicted_label"], entry["predicted_label"])
        for cls in classes:
            if pred_label == cls and true_label == cls:
                tp[cls] += 1
            elif pred_label == cls and true_label != cls:
                fp[cls] += 1
            elif pred_label != cls and true_label == cls:
                fn[cls] += 1
    metrics = {}
    for cls in classes:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[cls] = {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1-score": round(f1, 2),
        }
    return metrics


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
