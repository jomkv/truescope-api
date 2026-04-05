from core.types import get_vector_type
from core.db import Base, engine
from sqlalchemy import UUID, Index, Column, String, ForeignKey
import uuid

class ArticleChunk(Base):
    __tablename__ = "article_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String, unique=True, nullable=False)
    chunk_content = Column(String, nullable=False)
    doc_id = Column(String, ForeignKey("articles.doc_id"), nullable=False)
    embedding = Column(get_vector_type(384), nullable=False)
    
    # Relationships
    from sqlalchemy.orm import relationship
    article = relationship("Article", back_populates="chunks", lazy="selectin")

    # Dialect-aware table args
    if engine.dialect.name == "postgresql":
        __table_args__ = (
            Index(
                "hnsw_cosine_article_chunks_idx",
                "embedding",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )
    else:
        # SQLite / Turso managed independently
        __table_args__ = ()

