from pydantic import BaseModel, conlist
from typing import Annotated


class ClaimModel(BaseModel):
    id: str
    doc_id: str
    claim_text: str
    verdict: str | None = None
    embedding: Annotated[list[float], conlist(float, min_length=384, max_length=384)]
