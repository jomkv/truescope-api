from schemas.claim_schema import Claim
from schemas.article_schema import Article
from schemas.article_chunk_schema import ArticleChunk
from core.db import Session
import logging
from sqlalchemy.orm import Session as SessionType

logger = logging.getLogger(__name__)


class VerifyDatabase:
    def __init__(self) -> None:
        self.session_factory = Session

    @staticmethod
    def claim_distance_col(embedding: list[float]):
        return Claim.embedding.cosine_distance(embedding)  # type: ignore

    @staticmethod
    def chunk_distance_col(embedding: list[float]):
        return ArticleChunk.embedding.cosine_distance(embedding)  # type: ignore

    def find_similar_claims(
        self, embedding: list[float], limit: int
    ) -> list[tuple[Claim, float]]:
        """
        Retrieves the most similar claims to the provided embedding using HNSW vector search,
        excluding claims with an 'UNKNOWN' verdict.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            limit (int): Maximum number of similar claims to retrieve.

        Returns:
            list[tuple[Claim, float]]: List of tuples containing the claim and its distance to the input embedding.
        """
        try:
            distance_col = self.claim_distance_col(embedding)
            with self.session_factory() as session:
                return (
                    session.query(Claim, distance_col)
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
        """
        Retrieves the most similar article chunks to the provided embedding using HNSW vector search.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            limit (int): Maximum number of similar chunks to retrieve.

        Returns:
            list[tuple[ArticleChunk, float]]: List of tuples containing the article chunk and its distance to the input embedding.
        """
        try:
            distance_col = self.chunk_distance_col(embedding)
            with self.session_factory() as session:
                return (
                    session.query(ArticleChunk, distance_col)
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
        Retrieves the most relevant article chunks for each doc_id in doc_ids using vector search.

        Args:
            embedding (list[float]): Embedding vector for the user claim.
            doc_ids (set[str]): Set of document IDs to restrict the search.

        Returns:
            list[tuple[ArticleChunk, float]]: List of tuples containing the article chunk and its distance to the input embedding.
        """
        try:
            distance_col = self.chunk_distance_col(embedding)
            with self.session_factory() as session:
                return (
                    session.query(ArticleChunk, distance_col)
                    .filter(ArticleChunk.doc_id.in_(doc_ids))
                    .all()
                )
        except Exception as e:
            logger.error(f"Error while finding similar article chunks: {e}")
            raise e

    def find_articles_from_doc_ids(self, doc_ids: set[str]) -> list[Article]:
        """
        Retrieves articles from the database for the given set of document IDs.

        Args:
            doc_ids (set[str]): Set of document IDs.

        Returns:
            list[Article]: List of Article objects matching the provided doc_ids.
        """
        try:
            with self.session_factory() as session:
                return session.query(Article).filter(Article.doc_id.in_(doc_ids)).all()
        except Exception as e:
            logger.error(f"Error while finding articles from doc_ids: {e}")
            raise e
