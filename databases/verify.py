import logging
from schemas.claim_schema import Claim
from schemas.article_schema import Article
from schemas.article_chunk_schema import ArticleChunk
from sqlalchemy import func, text, column
from sqlalchemy.orm import Session as SessionType, defer
from core.db import Session, engine

logger = logging.getLogger(__name__)


class VerifyDatabase:
    def __init__(self) -> None:
        pass

    @staticmethod
    def claim_distance_col(embedding: list[float]):
        if engine.dialect.name == "postgresql":
            return Claim.embedding.cosine_distance(embedding)  # type: ignore

        import struct

        # Ensure it's a list
        vec_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        # pack as binary f32 blob (384 floats)
        vec_blob = struct.pack("<384f", *vec_list)
        return func.vector_distance_cos(Claim.embedding, vec_blob)

    @staticmethod
    def chunk_distance_col(embedding: list[float]):
        if engine.dialect.name == "postgresql":
            return ArticleChunk.embedding.cosine_distance(embedding)  # type: ignore

        import struct

        # Ensure it's a list
        vec_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
        # pack as binary f32 blob (384 floats)
        vec_blob = struct.pack("<384f", *vec_list)
        return func.vector_distance_cos(ArticleChunk.embedding, vec_blob)

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
        self, embedding: list[float], limit: int, doc_ids: set[str] = None
    ) -> list[tuple[ArticleChunk, float]]:
        """
        Retrieves similar article chunks using the clean native Turso index (chunks_vec_idx).
        Searches ALL 350,000 chunks for the best possible evidence, with zero scanning overhead.
        """
        # --- Postgres High-Performance Search ---
        if engine.dialect.name == "postgresql":
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
        Retrieves article chunks for doc_ids with vector distance.
        """
        try:
            distance_col = self.chunk_distance_col(embedding)

            with Session() as session:
                return (
                    session.query(ArticleChunk, distance_col)
                    .options(defer(ArticleChunk.embedding))
                    .filter(ArticleChunk.doc_id.in_(doc_ids))
                    .all()
                )
        except Exception as e:
            logger.error(f"Error while finding similar article chunks: {e}")
            raise e

    def find_articles_from_doc_ids(self, doc_ids: set[str]) -> list[Article]:
        """
        Retrieves articles from the database for the given set of document IDs.
        Defers Article.content only if necessary? No, we often need content.
        But we definitely don't need Article embeddings (if they exist).
        Wait, Article doesn't have an embedding column in the schema I saw.
        """
        try:
            with Session() as session:
                return session.query(Article).filter(Article.doc_id.in_(doc_ids)).all()
        except Exception as e:
            logger.error(f"Error while finding articles from doc_ids: {e}")
            raise e

    def find_chunks_by_doc_ids(self, doc_ids: set[str]) -> list[ArticleChunk]:
        """
        Retrieves all article chunks for the given set of document IDs in a single query.
        """
        try:
            with Session() as session:
                return (
                    session.query(ArticleChunk)
                    .filter(ArticleChunk.doc_id.in_(doc_ids))
                    .all()
                )
        except Exception as e:
            logger.error(f"Error while finding chunks from doc_ids: {e}")
            raise e
