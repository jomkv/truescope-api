from schemas.claim_schema import Claim
from schemas.article_schema import Article
from schemas.article_chunk_schema import ArticleChunk
from core.db import Session
import logging
from sqlalchemy.orm import Session as SessionType

logger = logging.getLogger(__name__)


class VerifyDatabase:
    def __init__(self) -> None:
        self.session: SessionType = Session()

    @staticmethod
    def claim_distance_col(embedding: list[float]):
        return Claim.embedding.cosine_distance(embedding)  # type: ignore

    @staticmethod
    def chunk_distance_col(embedding: list[float]):
        return ArticleChunk.embedding.cosine_distance(embedding)  # type: ignore

    def find_similar_claims(
        self, embedding: list[float], limit: int
    ) -> list[tuple[Claim, float]]:
        try:
            distance_col = self.claim_distance_col(embedding)

            return (
                self.session.query(Claim, distance_col)
                .filter(Claim.verdict != "UNKNOWN")
                .order_by(distance_col)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Error while finding similar claims: {e}")
            raise e

    def find_similar_chunks(
        self, embedding: list[float], limit: int
    ) -> list[tuple[ArticleChunk, float]]:
        try:
            distance_col = self.chunk_distance_col(embedding)

            return (
                self.session.query(ArticleChunk, distance_col)
                .order_by(distance_col)
                .limit(limit)
                .all()
            )
        except Exception as e:
            logger.error(f"Error while finding similar chunks")
            raise e

    def find_similar_chunks_from_doc_ids(
        self, embedding: list[float], doc_ids: set[str]
    ) -> list[tuple[ArticleChunk, float]]:
        """
        For each doc_id in doc_ids, find the most relevant chunks.

        Args:
            embedding (list[float]): Embedded user claim
            doc_ids (list[str]): Document IDs of articles on which we will run the search on

        Returns:
            chunks_with_distance (list[tuple[ArticleChunk, float]]): List of tuples containing the chunk and its distance.
        """
        try:
            distance_col = self.chunk_distance_col(embedding)

            return (
                self.session.query(ArticleChunk, distance_col)
                .filter(ArticleChunk.doc_id.in_(doc_ids))
                .all()
            )
        except Exception as e:
            logger.error(f"Error while finding similar article chunks: {e}")
            raise e

    def find_articles_from_doc_ids(self, doc_ids: set[str]) -> list[Article]:
        try:
            return self.session.query(Article).filter(Article.doc_id.in_(doc_ids)).all()
        except Exception as e:
            logger.error(f"Error while finding articles from doc_ids: {e}")
            raise e
