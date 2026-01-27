from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ArticleModel(BaseModel):
    doc_id: str
    title: str
    content: str
    publish_date: datetime
    url: str
    source = str
    type = str
    source_bias = str
