import json
from datetime import datetime
from typing import List, Dict
import requests
import statistics


class ArticleRetrievalEvaluator:
    def __init__(self, dataset_path: str = "tests/test_dataset.json"):
        self.dataset_path = dataset_path
        # Quality thresholds for article qualification
        self.quality_thresholds = {
            "similarity_min": 0.5,
            "entity_match_min": 0.4,
            "combined_min": 0.5,
        }
        self.results = {
            "timestamp": str(datetime.now()),
            "config": {
                "api_endpoint": "http://127.0.0.1:8000/v1/verify",
                "timeout": 30,
                "evaluate_all_articles": True,  # Evaluate all retrieved articles
                "quality_thresholds": self.quality_thresholds,
            },
            "aggregate_metrics": {},
            "case_results": [],
        }

    def load_dataset(self) -> List[Dict]:
        with open(self.dataset_path) as f:
            data = json.load(f)
        return data["test_cases"]

    def get_retrieved_articles(self, claim: str) -> List[Dict]:
        """Retrieve ALL articles for a claim"""
        try:
            response = requests.post(
                "http://127.0.0.1:8000/v1/verify", json={"claim": claim}, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            scores = result.get("scores", [])
            articles = []

            # Get ALL articles, not just top K
            for article in scores:
                articles.append(
                    {
                        "doc_id": article.get("doc_id", "")[:16],
                        "title": article.get("title", ""),
                        "similarity_score": article.get("similarity_score", 0),
                        "entity_match_score": article.get("entity_match_score", 0),
                        "combined_relevance_score": article.get(
                            "combined_relevance_score", 0
                        ),
                        "source_type": article.get("source_type", ""),
                        "has_nli": article.get("nli_result") is not None,
                        "skip_reason": article.get("skip_reason", []),
                    }
                )

            return articles
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            return []

    def check_entity_coverage(
        self, articles: List[Dict], expected_entities: List[str]
    ) -> Dict:
        """Check if retrieved articles have good entity match scores"""
        if not articles:
            return {
                "avg_entity_match": 0.0,
                "articles_with_entities": 0,
                "perfect_entity_match_count": 0,
            }

        entity_scores = [
            a["entity_match_score"] for a in articles if not a["skip_reason"]
        ]

        return {
            "avg_entity_match": (
                statistics.mean(entity_scores) if entity_scores else 0.0
            ),
            "articles_with_entities": sum(
                1 for a in articles if a["entity_match_score"] > 0.4
            ),
            "perfect_entity_match_count": sum(
                1 for a in articles if a["entity_match_score"] == 1.0
            ),
        }

    def evaluate_article_quality(self, article: Dict) -> Dict:
        """Evaluate if a single article qualifies as good evidence"""
        similarity = article["similarity_score"]
        entity_match = article["entity_match_score"]
        combined = article["combined_relevance_score"]

        # Check against thresholds
        meets_similarity = similarity >= self.quality_thresholds["similarity_min"]
        meets_entity = entity_match >= self.quality_thresholds["entity_match_min"]
        meets_combined = combined >= self.quality_thresholds["combined_min"]

        # Article qualifies if it meets at least 2 out of 3 criteria
        qualifies = sum([meets_similarity, meets_entity, meets_combined]) >= 2

        return {
            "qualifies": qualifies,
            "meets_similarity": meets_similarity,
            "meets_entity": meets_entity,
            "meets_combined": meets_combined,
            "quality_score": (similarity + entity_match + combined) / 3,
        }

    def evaluate(self) -> Dict:
        test_cases = self.load_dataset()

        all_similarity_scores = []
        all_entity_scores = []
        all_combined_scores = []
        all_quality_scores = []
        all_qualification_rates = []

        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case["id"]
            claim = test_case["claim"]
            expected_entities = test_case["ground_truth"]["expected_entities"]

            articles = self.get_retrieved_articles(claim)

            if not articles:
                case_result = {
                    "test_id": test_id,
                    "claim": claim,
                    "articles_retrieved": 0,
                    "qualified_articles": 0,
                    "qualification_rate": 0.0,
                    "avg_similarity": 0.0,
                    "avg_entity_match": 0.0,
                    "avg_combined_score": 0.0,
                    "avg_quality_score": 0.0,
                    "articles": [],
                }
                self.results["case_results"].append(case_result)
                continue

            # Evaluate each article individually
            evaluated_articles = []
            qualified_count = 0

            for article in articles:
                quality_eval = self.evaluate_article_quality(article)

                evaluated_articles.append(
                    {
                        "title": article["title"],
                        "similarity": article["similarity_score"],
                        "entity_match": article["entity_match_score"],
                        "combined": article["combined_relevance_score"],
                        "quality_score": quality_eval["quality_score"],
                        "qualifies": quality_eval["qualifies"],
                        "meets_similarity": quality_eval["meets_similarity"],
                        "meets_entity": quality_eval["meets_entity"],
                        "meets_combined": quality_eval["meets_combined"],
                        "has_nli": article["has_nli"],
                        "skipped": len(article["skip_reason"]) > 0,
                    }
                )

                if quality_eval["qualifies"]:
                    qualified_count += 1

            # Calculate per-claim averages
            similarity_scores = [a["similarity"] for a in evaluated_articles]
            entity_scores = [a["entity_match"] for a in evaluated_articles]
            combined_scores = [a["combined"] for a in evaluated_articles]
            quality_scores = [a["quality_score"] for a in evaluated_articles]

            avg_similarity = statistics.mean(similarity_scores)
            avg_entity = statistics.mean(entity_scores)
            avg_combined = statistics.mean(combined_scores)
            avg_quality = statistics.mean(quality_scores)
            qualification_rate = qualified_count / len(articles) if articles else 0

            # Add to overall tracking
            all_similarity_scores.append(avg_similarity)
            all_entity_scores.append(avg_entity)
            all_combined_scores.append(avg_combined)
            all_quality_scores.append(avg_quality)
            all_qualification_rates.append(qualification_rate)

            # Store case result
            case_result = {
                "test_id": test_id,
                "claim": claim,
                "expected_entities": expected_entities,
                "articles_retrieved": len(articles),
                "qualified_articles": qualified_count,
                "qualification_rate": round(qualification_rate, 4),
                "avg_similarity": round(avg_similarity, 4),
                "avg_entity_match": round(avg_entity, 4),
                "avg_combined_score": round(avg_combined, 4),
                "avg_quality_score": round(avg_quality, 4),
                "articles": evaluated_articles,
            }
            self.results["case_results"].append(case_result)

        # Calculate aggregate metrics (overall averages across all claims)
        self.results["aggregate_metrics"] = {
            "total_cases": len(test_cases),
            "quality_thresholds": self.quality_thresholds,
            "overall_avg_similarity": (
                round(statistics.mean(all_similarity_scores), 4)
                if all_similarity_scores
                else 0
            ),
            "overall_avg_entity_match": (
                round(statistics.mean(all_entity_scores), 4) if all_entity_scores else 0
            ),
            "overall_avg_combined": (
                round(statistics.mean(all_combined_scores), 4)
                if all_combined_scores
                else 0
            ),
            "overall_avg_quality_score": (
                round(statistics.mean(all_quality_scores), 4)
                if all_quality_scores
                else 0
            ),
            "overall_avg_qualification_rate": (
                round(statistics.mean(all_qualification_rates), 4)
                if all_qualification_rates
                else 0
            ),
            "similarity_stdev": (
                round(statistics.stdev(all_similarity_scores), 4)
                if len(all_similarity_scores) > 1
                else 0
            ),
            "entity_match_stdev": (
                round(statistics.stdev(all_entity_scores), 4)
                if len(all_entity_scores) > 1
                else 0
            ),
            "excellent_claims": sum(
                1
                for c in self.results["case_results"]
                if c.get("qualification_rate", 0) >= 0.8
            ),
            "good_claims": sum(
                1
                for c in self.results["case_results"]
                if 0.6 <= c.get("qualification_rate", 0) < 0.8
            ),
            "fair_claims": sum(
                1
                for c in self.results["case_results"]
                if 0.4 <= c.get("qualification_rate", 0) < 0.6
            ),
            "poor_claims": sum(
                1
                for c in self.results["case_results"]
                if c.get("qualification_rate", 0) < 0.4
            ),
        }

        return self.results

    def save_results(self, output_path: str = None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"tests/results/retrieval_eval_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)


def main():
    evaluator = ArticleRetrievalEvaluator()
    evaluator.evaluate()
    evaluator.save_results()


if __name__ == "__main__":
    main()
