from pydantic import BaseModel, conlist
from typing import Annotated
from uuid import UUID


class ArticleVectorModel(BaseModel):
    id: UUID
    chunk_id: str
    doc_id: str
    embedding: Annotated[list[float], conlist(float, min_length=384, max_length=384)]
    source: str
    type: str
    source: str
