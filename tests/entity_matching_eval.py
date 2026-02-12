import json
from datetime import datetime
from typing import List, Dict
from controllers.v1.verify_controller import VerifyController
import statistics


class EntityMatchingEvaluator:
    def __init__(self, dataset_path: str = "tests/datasets/test_dataset.json"):
        self.dataset_path = dataset_path
        self.results = {
            "timestamp": str(datetime.now()),
            "config": {
                "timeout": 30,
            },
            "aggregate_metrics": {},
            "case_results": [],
        }
        self.verify_controller = VerifyController()

    def load_dataset(self) -> List[Dict]:
        with open(self.dataset_path) as f:
            data = json.load(f)
        return data["test_cases"]

    def calculate_recall(self, extracted: list[str], expected: list[str]) -> float:
        if not expected:
            return 1.0

        extracted_lower = [e.lower() for e in extracted]
        expected_lower = [e.lower() for e in expected]

        matches = sum(
            1
            for exp in expected_lower
            if any(exp in ext or ext in exp for ext in extracted_lower)
        )
        return matches / len(expected)

    def calculate_precision(self, extracted: list[str], expected: list[str]) -> float:
        if not extracted:
            return 1.0 if not expected else 0.0

        extracted_lower = [e.lower() for e in extracted]
        expected_lower = [e.lower() for e in expected]

        matches = sum(
            1
            for ext in extracted_lower
            if any(ext in exp or exp in ext for exp in expected_lower)
        )
        return matches / len(extracted)

    def calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate(self) -> Dict:
        test_cases = self.load_dataset()

        recalls = []
        precisions = []
        f1_scores = []

        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case["id"]
            claim = test_case["claim"]
            expected = test_case["ground_truth"]["expected_entities"]

            extracted = self.verify_controller.extract_entities(claim)

            recall = self.calculate_recall(extracted, expected)
            precision = self.calculate_precision(extracted, expected)
            f1 = self.calculate_f1(precision, recall)

            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)

            case_result = {
                "test_id": test_id,
                "claim": claim,
                "expected_entities": expected,
                "extracted_entities": extracted,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1_score": round(f1, 4),
                "entities_match": len(extracted) > 0,
            }

            self.results["case_results"].append(case_result)

        self.results["aggregate_metrics"] = {
            "total_cases": len(test_cases),
            "avg_recall": round(statistics.mean(recalls), 4),
            "avg_precision": round(statistics.mean(precisions), 4),
            "avg_f1": round(statistics.mean(f1_scores), 4),
            "recall_stdev": (
                round(statistics.stdev(recalls), 4) if len(recalls) > 1 else 0
            ),
            "precision_stdev": (
                round(statistics.stdev(precisions), 4) if len(precisions) > 1 else 0
            ),
            "perfect_f1_count": sum(1 for f in f1_scores if f == 1.0),
            "good_f1_count": sum(1 for f in f1_scores if f >= 0.7),
            "fair_f1_count": sum(1 for f in f1_scores if 0.4 <= f < 0.7),
            "poor_f1_count": sum(1 for f in f1_scores if f < 0.4),
        }
        return self.results

    def save_results(self, output_path: str | None = None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"tests/results/entity_eval_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)


def main():
    evaluator = EntityMatchingEvaluator()
    evaluator.evaluate()
    evaluator.save_results()


if __name__ == "__main__":
    main()
