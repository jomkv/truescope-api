from typing import Annotated

from fastapi import APIRouter, Body, Query

from controllers.v1.verify_controller import VerifyController
from models.article_model import ArticleModel
from core.db import Session


router = APIRouter()
controller = VerifyController()


@router.post("/")
async def verify_claim(
    claim: Annotated[str, Body()],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    use_fallback: Annotated[bool, Query(description="Use titles/content when no explicit claims")] = True,
    relevance_threshold: Annotated[float, Query(ge=0.0, le=1.0, description="Min similarity score for NLI (0-1)")] = 0.3,
    entity_threshold: Annotated[float, Query(ge=0.0, le=1.0, description="Min entity match score for inclusion (0-1)")] = 0.4
):
    """
    Verify a claim by finding similar articles and checking if they support or refute it.
    
    Process:
    1. Embed the claim
    2. Do similarity search in the database
    3. Extract key entities (organizations, people, locations) from claim
    4. Filter by dual criteria:
       a. Semantic relevance (similarity score)
       b. Entity matching (key actors/organizations must appear in article)
    5. For relevant articles, use NLI to check if they support/refutes/neutral
    6. Uses fallback strategies: explicit claims > titles > extracted sentences
    
    Query Parameters:
    - limit: Number of articles to retrieve (1-50, default: 20)
    - use_fallback: Whether to analyze titles/content when no explicit claim (default: true)
    - relevance_threshold: Minimum similarity score (0-1) for NLI analysis (default: 0.3)
      Higher values = stricter semantic relevance filtering
    - entity_threshold: Minimum entity match score (0-1) for article inclusion (default: 0.4)
      Ensures key entities from claim (e.g., "INC") appear in article
      Higher values = stricter entity matching
    """
    with Session() as session:
        results = controller.verify_claim_with_articles(
            session, claim, limit, use_fallback, relevance_threshold, entity_threshold
        )

    # Calculate summary statistics
    analyzed_count = sum(1 for r in results if r.get("nli_result"))
    skipped_count = sum(1 for r in results if r.get("skip_reason"))
    supports_count = sum(1 for r in results if r.get("nli_result") and r["nli_result"]["relation"] == "supports")
    refutes_count = sum(1 for r in results if r.get("nli_result") and r["nli_result"]["relation"] == "refutes")
    neutral_count = sum(1 for r in results if r.get("nli_result") and r["nli_result"]["relation"] == "neutral")
    
    # Breakdown by claim source
    source_breakdown = {}
    for r in results:
        if r.get("nli_result"):
            source = r["nli_result"].get("claim_source", "unknown")
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
    
    # Calculate average scores for analyzed vs skipped articles
    analyzed_results = [r for r in results if r.get("nli_result")]
    skipped_results = [r for r in results if r.get("skip_reason")]
    
    analyzed_avg_sim = sum(r["similarity_score"] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
    analyzed_avg_entity = sum(r["entity_match_score"] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
    analyzed_avg_combined = sum(r["combined_relevance_score"] for r in analyzed_results) / len(analyzed_results) if analyzed_results else 0
    
    skipped_avg_sim = sum(r["similarity_score"] for r in skipped_results) / len(skipped_results) if skipped_results else 0
    skipped_avg_entity = sum(r["entity_match_score"] for r in skipped_results if r["entity_match_score"] is not None) / len([r for r in skipped_results if r["entity_match_score"] is not None]) if any(r["entity_match_score"] is not None for r in skipped_results) else 0
    skipped_avg_combined = sum(r["combined_relevance_score"] for r in skipped_results if r["combined_relevance_score"] is not None) / len([r for r in skipped_results if r["combined_relevance_score"] is not None]) if any(r["combined_relevance_score"] is not None for r in skipped_results) else 0
    
    # Count skip reasons
    skip_reasons_count = {}
    for r in skipped_results:
        for reason in r.get("skip_reason", []):
            reason_key = "low_similarity" if "similarity" in reason else "missing_entities"
            skip_reasons_count[reason_key] = skip_reasons_count.get(reason_key, 0) + 1

    return {
        "user_claim": claim,
        "summary": {
            "total_articles": len(results),
            "articles_analyzed": analyzed_count,
            "articles_skipped": skipped_count,
            "avg_similarity_analyzed": round(analyzed_avg_sim, 4),
            "avg_similarity_skipped": round(skipped_avg_sim, 4),
            "avg_entity_match_analyzed": round(analyzed_avg_entity, 4),
            "avg_entity_match_skipped": round(skipped_avg_entity, 4),
            "avg_combined_relevance_analyzed": round(analyzed_avg_combined, 4),
            "avg_combined_relevance_skipped": round(skipped_avg_combined, 4),
            "skip_reasons": skip_reasons_count,
            "supports": supports_count,
            "refutes": refutes_count,
            "neutral": neutral_count,
            "claim_sources": source_breakdown
        },
        "results": results
    }
