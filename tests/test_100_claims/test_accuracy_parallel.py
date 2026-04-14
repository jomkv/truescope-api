import asyncio
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from controllers.v1.verify_controller import VerifyController

DATASET_FILES = ["jomTestCases.json", "negatedClaims.json"]
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_save_files = os.getenv("ACCURACY_SAVE_FILES", "1") == "1"
_quiet_logs = os.getenv("ACCURACY_QUIET", "0") == "1"


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def get_score_label(score: float) -> str:
    return "TRUE" if score >= 0 else "FALSE"


def normalize_verdict_label(verdict: str) -> str:
    true_labels = {"TRUE", "MOSTLY_TRUE", "HALF_TRUE"}
    return "TRUE" if verdict.upper() in true_labels else "FALSE"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results_list: list[dict]) -> dict:
    label_map = {"TRUE": "Support", "FALSE": "Refute"}
    classes = ["Support", "Refute"]
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

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
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "tp": tp[cls],
            "fp": fp[cls],
            "fn": fn[cls],
        }
    return metrics


def compute_error_breakdown(results_list: list[dict]) -> dict:
    """
    Categorises wrong predictions to help identify where the system loses points.
    - false_positive: predicted TRUE but ground truth is FALSE
    - false_negative: predicted FALSE but ground truth is TRUE
    - skipped_wrong: claim was skipped (no evidence found) and the skip was wrong
    - skipped_correct: claim was skipped but would have been correct if guessed TRUE
    """
    false_positives = []
    false_negatives = []
    skipped_wrong = []
    skipped_correct = []

    for e in results_list:
        if e["skipped"]:
            # Skipped items are excluded from accuracy but we still categorise them
            if e["ground_truth"] == "TRUE":
                skipped_correct.append(e)  # would have defaulted to TRUE anyway
            else:
                skipped_wrong.append(e)  # we missed a FALSE claim
            continue
        if not e["is_correct"]:
            if e["predicted_label"] == "TRUE":
                false_positives.append(e)
            else:
                false_negatives.append(e)

    def slim(entries: list[dict]) -> list[dict]:
        return [
            {"id": e["id"], "claim": e["claim"], "score": e["system_score"]}
            for e in entries
        ]

    return {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "skipped_wrong_count": len(skipped_wrong),
        "skipped_correct_count": len(skipped_correct),
        "false_positives": slim(false_positives),
        "false_negatives": slim(false_negatives),
        "skipped_wrong": slim(skipped_wrong),
        "skipped_correct": slim(skipped_correct),
    }


def compute_score_distribution(results_list: list[dict]) -> dict:
    """Bucket system_score values to understand confidence distribution."""
    buckets = {
        "strong_true  (>= 0.5)": 0,
        "weak_true    (0.0–0.5)": 0,
        "weak_false   (-0.5–0.0)": 0,
        "strong_false (<= -0.5)": 0,
        "skipped": 0,
    }
    for e in results_list:
        s = e["system_score"]
        if s is None:
            buckets["skipped"] += 1
        elif s >= 0.5:
            buckets["strong_true  (>= 0.5)"] += 1
        elif s >= 0.0:
            buckets["weak_true    (0.0–0.5)"] += 1
        elif s >= -0.5:
            buckets["weak_false   (-0.5–0.0)"] += 1
        else:
            buckets["strong_false (<= -0.5)"] += 1
    return buckets


# ---------------------------------------------------------------------------
# Model loader helper
# ---------------------------------------------------------------------------


def load_hitl_models(vc: VerifyController, dataset_name: str) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    log_path = os.path.join(project_root, "data", "training_log.json")
    if not os.path.exists(log_path):
        return
    with open(log_path) as f:
        logs = json.load(f)

    nli_log = next((l for l in reversed(logs) if l["model"] == "nli"), None)
    if nli_log and nli_log["version"] > 0:
        v = nli_log["version"]
        nli_path = os.path.join(project_root, f"data/model_adapters/nli/v{v}")
        if os.path.exists(nli_path):
            print(f"[{dataset_name}] Loading NLI adapter v{v}...")
            vc.nli_service.load_adapter(nli_path)

    emb_log = next((l for l in reversed(logs) if l["model"] == "embeddings"), None)
    if emb_log and emb_log["version"] > 0:
        v = emb_log["version"]
        emb_path = os.path.join(project_root, f"data/model_adapters/embeddings/v{v}")
        if os.path.exists(emb_path):
            print(f"[{dataset_name}] Loading Embedding model v{v}...")
            vc.embedding_service.reload_model(emb_path)


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


async def run_dataset(vc: VerifyController, dataset_filename: str) -> dict:
    dataset_path = os.path.join(os.path.dirname(__file__), dataset_filename)
    dataset_name = os.path.splitext(dataset_filename)[0]
    results_path = os.path.join(
        os.path.dirname(__file__),
        f"test_accuracy_results_{dataset_name}_{_timestamp}.json",
    )

    print(f"[{dataset_name}] Loading dataset from {dataset_path}...", flush=True)
    with open(dataset_path, encoding="utf-8") as f:
        claims = json.load(f)
    print(f"[{dataset_name}] {len(claims)} claims loaded.", flush=True)

    correct = 0
    total = 0
    skipped = 0
    results_list: list[dict] = []

    for entry in claims:
        claim_text = entry["claim"]
        ground_truth = entry.get(
            "expected_verdict", entry.get("ground_truth", entry.get("Ground_truth"))
        )
        claim_id = entry["index"]
        exclude_docs = [entry["doc_id"]] if "doc_id" in entry else []

        result = await vc.verify_claim(claim_text, exclude_doc_ids=exclude_docs)

        if not result["results"]:
            score_label = "NEUTRAL"
            system_score = None
            skipped += 1
        else:
            system_score = result["overall_verdict"]
            score_label = get_score_label(system_score)

        ground_truth_norm = normalize_verdict_label(ground_truth)
        is_correct = score_label == ground_truth_norm

        if system_score is not None:
            total += 1
            if is_correct:
                correct += 1

        # Capture richer per-result diagnostics when file output is enabled.
        top_evidence = []
        if _save_files:
            top_results = result.get("results", [])
            top_evidence = [
                {
                    "source": r.source if hasattr(r, "source") else r.get("source"),
                    "source_type": (
                        r.source_type
                        if hasattr(r, "source_type")
                        else r.get("source_type")
                    ),
                    "similarity": (
                        r.similarity_score
                        if hasattr(r, "similarity_score")
                        else r.get("similarity_score")
                    ),
                    "entity_match": (
                        r.entity_match_score
                        if hasattr(r, "entity_match_score")
                        else r.get("entity_match_score")
                    ),
                    "combined": (
                        r.combined_relevance_score
                        if hasattr(r, "combined_relevance_score")
                        else r.get("combined_relevance_score")
                    ),
                    "verdict": r.verdict if hasattr(r, "verdict") else r.get("verdict"),
                    "nli_label": (
                        r.nli_result.relationship.value
                        if hasattr(r, "nli_result") and r.nli_result
                        else (r.get("nli_result") or {}).get("relationship")
                    ),
                    "nli_confidence": (
                        r.nli_result.relationship_confidence
                        if hasattr(r, "nli_result") and r.nli_result
                        else (r.get("nli_result") or {}).get("relationship_confidence")
                    ),
                }
                for r in top_results[:3]  # top 3 only to keep file size manageable
            ]

        results_list.append(
            {
                "id": claim_id,
                "claim": claim_text,
                "ground_truth": ground_truth_norm,
                "predicted_label": score_label,
                "system_score": system_score,
                "is_correct": is_correct,
                "skipped": not result["results"],
                "is_negated": result.get("is_negated", False),
                "truth_confidence": result.get("truth_confidence_score"),
                "top_evidence": top_evidence,
            }
        )

        if not _quiet_logs:
            print(
                f"[{dataset_name}] #{claim_id:03d} {'✓' if is_correct else '✗'}  "
                f"GT={ground_truth_norm:<5} Pred={score_label:<5} Score={str(round(system_score, 4)) if system_score is not None else 'SKIP':<8}  "
                f"{claim_text[:80]}",
                flush=True,
            )

    accuracy = correct / total if total else 0.0
    metrics = compute_metrics(results_list)
    error_breakdown = compute_error_breakdown(results_list)
    score_dist = compute_score_distribution(results_list)

    output_data = {
        "dataset": dataset_name,
        "timestamp": _timestamp,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "metrics": metrics,
        "error_breakdown": error_breakdown,
        "score_distribution": score_dist,
        "results": results_list,
    }

    if _save_files:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy : {correct}/{total} = {accuracy:.2%}")
    print(f"  Skipped  : {skipped}")
    print(
        f"  FP       : {error_breakdown['false_positive_count']}  (predicted TRUE, actually FALSE)"
    )
    print(
        f"  FN       : {error_breakdown['false_negative_count']}  (predicted FALSE, actually TRUE)"
    )
    print(f"\n  Per-class metrics:")
    for cls, vals in metrics.items():
        print(
            f"    {cls:<8}  P={vals['precision']:.2f}  R={vals['recall']:.2f}  F1={vals['f1-score']:.2f}"
            f"  (TP={vals['tp']} FP={vals['fp']} FN={vals['fn']})"
        )
    print(f"\n  Score distribution:")
    for bucket, count in score_dist.items():
        print(f"    {bucket}: {count}")
    if _save_files:
        print(f"\n  Results saved to {results_path}")
    print(f"{'='*60}\n")

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "metrics": metrics,
        "error_breakdown": error_breakdown,
    }


# ---------------------------------------------------------------------------
# Combined summary report
# ---------------------------------------------------------------------------


def write_combined_summary(all_results: list[dict]) -> dict:
    summary_path = os.path.join(
        os.path.dirname(__file__),
        f"test_accuracy_SUMMARY_{_timestamp}.json",
    )

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)

    combined = {}
    for r in all_results:
        name = r["dataset"]
        combined[name] = {
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "skipped": r["skipped"],
            "false_positives": r["error_breakdown"]["false_positive_count"],
            "false_negatives": r["error_breakdown"]["false_negative_count"],
            "support_f1": r["metrics"]["Support"]["f1-score"],
            "refute_f1": r["metrics"]["Refute"]["f1-score"],
        }
        print(
            f"  {name:<25} {r['correct']}/{r['total']} = {r['accuracy']:.2%}  "
            f"(FP={r['error_breakdown']['false_positive_count']}  FN={r['error_breakdown']['false_negative_count']}  "
            f"Skipped={r['skipped']})"
        )

    # Goal check
    print("\n  Goal check (target: both ≥80%, ideally normal > negated):")
    for name, vals in combined.items():
        status = "✓" if vals["accuracy"] >= 0.80 else "✗"
        print(f"    {status} {name}: {vals['accuracy']:.2%}")

    summary_payload = {"timestamp": _timestamp, "datasets": combined}

    if _save_files:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        print(f"\n  Summary saved to {summary_path}")

    print("ACCURACY_SUMMARY_JSON=" + json.dumps(summary_payload, ensure_ascii=False))
    print("=" * 60)
    return summary_payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> dict:
    print("[shared] Initializing VerifyController once...", flush=True)
    vc = VerifyController()
    load_hitl_models(vc, "shared")

    # Run datasets sequentially to avoid loading multiple heavy model instances.
    all_results = []
    for dataset_file in DATASET_FILES:
        all_results.append(await run_dataset(vc, dataset_file))

    return write_combined_summary(list(all_results))


if __name__ == "__main__":
    summary = asyncio.run(main())
    print("ACCURACY_SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=False))
