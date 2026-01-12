from typing import Annotated

from fastapi import APIRouter, Body

from controllers.v1.verify_controller import VerifyController
from models.article_model import ArticleModel
from core.db import Session


router = APIRouter()
controller = VerifyController()


@router.post("/")
async def verify_claim(claim: Annotated[str, Body()]):
    # 1. Embed claim
    # 2. Do similarity search at DB
    #   2.1. IF nothing retrieved, immediately return
    # 3. Populate articles of retrieved
    # 4. Do semantic similarity on retrieved vs. claim
    # 5. Do calculations, format result

    # 1, 2, 3 (no 2.1 yet)
    with Session() as session:
        articles: list[ArticleModel] = controller.get_relevant_articles(
            session, claim, 10
        )

    # 4
    # TODO

    # 5
    # TODO

    return articles
