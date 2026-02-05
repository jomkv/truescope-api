import uuid
from core.db import Base
from sqlalchemy import UUID, Index, Column, String, ForeignKey
from pgvector.sqlalchemy import VECTOR


class Claim(Base):
    __tablename__ = "claims"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(String, ForeignKey("articles.doc_id"), nullable=False)
    embedding = Column(VECTOR(384), nullable=False)
    claim_text = Column(String, nullable=False)
    verdict = Column(String)

    __table_args__ = (
        Index(
            "hnsw_cosine_claims_idx",
            "embedding",
            postgresql_using="hnsw",
            # Experimental
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
