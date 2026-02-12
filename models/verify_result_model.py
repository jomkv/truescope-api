from models.article_result_model import ArticleResultModel
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


class VerifyResultModel(BaseModel):
    entities: list[str]
    timeframe: list
    skipped: list[ArticleResultModel]
    results: list[ArticleResultModel]
    overall_verdict: float
