from core.types import get_vector_type
from core.db import Base, engine
from sqlalchemy import UUID, Index, Column, String, ForeignKey
import uuid

class Claim(Base):
    __tablename__ = "claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(String, ForeignKey("articles.doc_id"), nullable=False)
    embedding = Column(get_vector_type(384), nullable=False)
    claim_text = Column(String, nullable=False)
    verdict = Column(String)

    # Dialect-aware table args
    if engine.dialect.name == "postgresql":
        __table_args__ = (
            Index(
                "hnsw_cosine_claims_idx",
                "embedding",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )
    else:
        # SQLite / Turso usually manages its own vector indexing separately,
        # often via specialized FTS or native vector indexes that are not easily
        # represented in standard SQLAlchemy indices without custom DDL.
        __table_args__ = ()

