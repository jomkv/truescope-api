import csv
import asyncio
from controllers.v1.verify_controller import VerifyController
from .liar_label_map import LIAR_LABEL_TO_VERDICT, score_to_verdict

LIAR_TSV_PATH = "tests/LIAR/test.tsv"


# Adjust indices based on LIAR format (claim_text, label)
def parse_liar_tsv(path):
    claims = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            claim_text = row[2]
            label = row[1]
            claims.append((claim_text, label))
    return claims


async def main():
    vc = VerifyController()
    claims = parse_liar_tsv(LIAR_TSV_PATH)

    correct = 0
    total = 0
    results_list = []

    for claim_text, label in claims:
        result = await vc.verify_claim(claim_text)
        # If system output is a score, map it to Verdict
        system_score = result["overall_verdict"]
        system_verdict = (
            score_to_verdict(system_score)
            if isinstance(system_score, (int, float))
            else system_score
        )
        true_verdict = LIAR_LABEL_TO_VERDICT.get(label.lower())
        is_correct = system_verdict == true_verdict
        if is_correct:
            correct += 1
        total += 1
        results_list.append(
            {
                "claim": claim_text,
                "true_label": label,
                "true_verdict": str(true_verdict),
                "predicted_label": str(system_verdict),
                "system_score": system_score,
                "is_correct": is_correct,
            }
        )
        print(
            f"Claim: {claim_text}\nTrue: {label} ({true_verdict}), Pred: {system_verdict} (score: {system_score})\n"
        )

    import json
    import json
    import os
    from datetime import datetime

    results_dir = "tests/results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"liar_test_results_{timestamp}.json")
    output_data = {
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
        "results": results_list,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
