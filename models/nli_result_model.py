from pydantic import BaseModel
from constants.enums import NLILabel


class NLIResultModel(BaseModel):
    relationship: NLILabel
    relationship_confidence: float
    relationship_uncertainty: float
    claim_source: str
    analyzed_text: str
