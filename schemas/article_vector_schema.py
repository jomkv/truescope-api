from sqlalchemy import Index
import uuid
from core.db import Base
from sqlalchemy import UUID, Column, String, ForeignKey
from pgvector.sqlalchemy import VECTOR


class ArticleVector(Base):
    __tablename__ = "article_vectors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String, unique=True, nullable=False)
    chunk_content = Column(String, nullable=False)
    doc_id = Column(String, ForeignKey("articles.doc_id"), nullable=False)
    embedding = Column(VECTOR(384), nullable=False)
    source = Column(String)
    type = Column(String)
    source_bias = Column(String)

        # Improves performance and potentially the accuracy of results
    __table_args__ = (
        Index(
            "hnsw_cosine_idx",
            "embedding",
            postgresql_using="hnsw",
            # Experimental
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
