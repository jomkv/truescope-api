from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ArticleModel(BaseModel):
    doc_id: str
    title: str
    content: str
    claim: Optional[str] = None
    verdict: Optional[str] = None
    publish_date: datetime
    url: str
