from pydantic import BaseModel


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
    nli_result: None | dict
    verdict: None | float
    skip_reason: list[str]
    source: str
    source_type: str
    source_bias: str | None
    remarks: None | str
