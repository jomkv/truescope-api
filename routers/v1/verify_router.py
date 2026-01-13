from typing import Annotated

from fastapi import APIRouter, Body, Query

from controllers.v1.verify_controller import VerifyController
from models.article_model import ArticleModel
from core.db import Session


router = APIRouter()
controller = VerifyController()

# Fixed thresholds for article filtering
RELEVANCE_THRESHOLD = 0.3  
ENTITY_THRESHOLD = 0.4     


@router.post("/")
async def verify_claim(
    claim: Annotated[str, Body()],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    use_fallback: Annotated[bool, Query(description="Use titles/content when no explicit claims")] = True
):
    """
    Verify a claim by finding similar articles and checking if they support or refute it.
    
    Process:
    1. Embed the claim
    2. Do similarity search in the database
    3. Extract key entities (organizations, people, locations) from claim
    4. Filter by dual criteria:
       a. Semantic relevance (similarity score >= 0.3)
       b. Entity matching (key actors/organizations must appear in article >= 0.4)
    5. For relevant articles, use NLI to check if they support/refutes/neutral
    6. Uses fallback strategies: explicit claims > titles > extracted sentences
    
    Query Parameters:
    - limit: Number of articles to retrieve (1-50, default: 20)
    - use_fallback: Whether to analyze titles/content when no explicit claim (default: true)
    """
    with Session() as session:
        results = controller.verify_claim_with_articles(
            session, claim, limit, use_fallback, RELEVANCE_THRESHOLD, ENTITY_THRESHOLD
        )
    
    summary = controller.calculate_summary_statistics(results)

    return {
        "user_claim": claim,
        "summary": summary,
        "results": results
    }
