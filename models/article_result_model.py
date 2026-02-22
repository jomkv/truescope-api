from pydantic import BaseModel
from constants.enums import SourceBias
from models.nli_result_model import NLIResultModel


class ArticleResultModel(BaseModel):
    doc_id: str
    title: str
    content: str
    found_claim: str | None
    found_verdict: str | None
    publish_date: str
    url: str
    similarity_score: float
    entity_match_score: float
    combined_relevance_score: float
    nli_result: None | NLIResultModel
    verdict: None | float
    skip_reason: list[str]
    source: str
    source_type: str
    source_bias: SourceBias
    chunk_texts: str
