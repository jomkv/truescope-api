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
        """
        Retrieves similar claims using a standard distance scan.
        For a 48,000 row table, this is extremely fast and 100% reliable,
        avoiding Turso's parser issues entirely.
        """
        distance_col = self.claim_distance_col(embedding)
        with Session() as session:
            return (
                session.query(Claim, distance_col)
                .options(defer(Claim.embedding))
                .order_by(distance_col)
                .limit(limit)
                .all()
            )

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
            with Session() as session:
                return (
                    session.query(ArticleChunk, distance_col)
                    .options(defer(ArticleChunk.embedding))
                    .order_by(distance_col)
                    .limit(limit)
                    .all()
                )

        # --- Turso High-Performance Search ---
        try:
            import struct

            vec_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            vec_blob = struct.pack("<384f", *vec_list)

            with Session() as session:
                # Use vector index for performance; map rowid back to UUID
                top_k_query = text(
                    f"""
                    SELECT c.id 
                    FROM vector_top_k('chunks_vec_idx', CAST(:v AS BLOB), {limit}) v
                    JOIN article_chunks c ON v.id = c.rowid
                """
                )

                rows = session.execute(top_k_query, {"v": vec_blob}).fetchall()

                # Fallback to brute-force scan if index returns no results (e.g. index lag)
                if not rows:
                    logger.info(
                        "Vector index empty; falling back to high-accuracy scan."
                    )
                    distance_col = self.chunk_distance_col(embedding)
                    return (
                        session.query(ArticleChunk, distance_col)
                        .options(defer(ArticleChunk.embedding))
                        .order_by(distance_col)
                        .limit(limit)
                        .all()
                    )

                chunk_ids = [r[0] for r in rows]
                distance_col = self.chunk_distance_col(embedding)

                chunks_with_dist = (
                    session.query(ArticleChunk, distance_col)
                    .filter(ArticleChunk.id.in_(chunk_ids))
                    .options(defer(ArticleChunk.embedding))
                    .all()
                )

                # Maintain index order
                chunks_with_dist.sort(
                    key=lambda x: next(
                        (i for i, cid in enumerate(chunk_ids) if cid == x[0].id), 999
                    )
                )

                # If doc_ids are provided, we optionally filter *after* retrieval
                # (but in practice, VerifyController won't restrict it anymore)
                if doc_ids:
                    chunks_with_dist = [
                        c for c in chunks_with_dist if c[0].doc_id in doc_ids
                    ]

                return chunks_with_dist

        except Exception as e:
            logger.warning(
                f"Native index (chunks_idx) failed or not found, falling back to high-accuracy scan: {e}"
            )
            # FAIL-SAFE: Revert to the 100% reliable scan pattern if indexer is busy or missing
            distance_col = self.chunk_distance_col(embedding)
            with Session() as session:
                return (
                    session.query(ArticleChunk, distance_col)
                    .options(defer(ArticleChunk.embedding))
                    .order_by(distance_col)
                    .limit(limit)
                    .all()
                )

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

            return (
                self.session.query(ArticleChunk, distance_col)
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
            return self.session.query(Article).filter(Article.doc_id.in_(doc_ids)).all()
        except Exception as e:
            logger.error(f"Error while finding articles from doc_ids: {e}")
            raise e
