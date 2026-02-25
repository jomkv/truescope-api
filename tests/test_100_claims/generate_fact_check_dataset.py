import random
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from databases.verify import VerifyDatabase
from schemas.claim_schema import Claim
from schemas.article_schema import Article
from sqlalchemy import func

# Initialize DB
verify_db = VerifyDatabase()


# Query all claims with a verdict (excluding UNKNOWN)
def get_fact_checks_with_verdict(limit=100):
    session = verify_db.session
    target_sources = ["rappler", "verafiles", "tsek"]

    claims = (
        session.query(Claim, Article)
        .join(Article, Claim.doc_id == Article.doc_id)
        .filter(Claim.verdict != "UNKNOWN")
        .filter(func.lower(Article.source).in_([s.lower() for s in target_sources]))
        .all()
    )
    if len(claims) < limit:
        limit = len(claims)
    selected_claims = random.sample(claims, limit)
    dataset = []
    for claim, article in selected_claims:
        dataset.append(
            {
                "docID": claim.doc_id,
                "claim": claim.claim_text,
                "Ground_truth": claim.verdict,
                "id": str(claim.id),
                "source": article.source,
            }
        )
    return dataset


def main():
    data = get_fact_checks_with_verdict(100)
    out_path = os.path.join(os.path.dirname(__file__), "test_dataset_generated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(data)} samples in {out_path}")


if __name__ == "__main__":
    main()
