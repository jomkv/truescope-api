"""Add indices and adjust table cols

Revision ID: 0ce9cc815b17
Revises:
Create Date: 2026-01-22 23:24:46.681498

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision: str = "0ce9cc815b17"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "hnsw_cosine_claims_idx",
        "claims",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "hnsw_cosine_article_chunks_idx",
        "article_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "hnsw_cosine_article_chunks_idx",
        table_name="article_chunks",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.drop_index(
        "hnsw_cosine_claims_idx",
        table_name="claims",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
