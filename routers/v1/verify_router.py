from fastapi import APIRouter

from controllers.v1.verify_controller import VerifyController
from models.verify_claim_model import VerifyClaimModel
from core.db import Session


router = APIRouter()
controller = VerifyController()


@router.post("/")
async def verify_claim(verify: VerifyClaimModel):
    """
    Verify a claim by finding similar articles and checking if they support or refute it.

    Process:
    1. Embed the claim
    2. Do similarity search in the database
    3. Extract key entities (organizations, people, locations) from claim
    4. Filter by dual criteria:
       a. Semantic relevance
       b. Entity matching
    5. For relevant articles, use NLI to check if they support/refutes/neutral
    6. Compute final score, weighing in other relevant factors (source bias, dataset verdict, nli result)

    Query Parameters:
    - limit: Number of articles to retrieve (1-50, default: 20)
    - use_fallback: Whether to analyze titles/content when no explicit claim (default: true)
    """
    with Session() as session:
        interfence_scores = await controller.verify_claim(session, verify.claim)

    entities = controller.extract_entities(verify.claim)
    timeframe = controller.extract_claim_timeframe(verify.claim)

    return {"entities": entities, "timeframe": timeframe, "scores": interfence_scores}
