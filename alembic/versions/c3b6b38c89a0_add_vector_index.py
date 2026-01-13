"""add_vector_index

Revision ID: c3b6b38c89a0
Revises:
Create Date: 2026-01-11 11:42:17.234143

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision: str = "c3b6b38c89a0"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "hnsw_cosine_idx",  # Name of the index
        "article_vectors",  # Table name
        ["embedding"],  # Column name
        postgresql_using="hnsw",  # Use HNSW algorithm
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},  # For semantic similarity
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("hnsw_cosine_idx", table_name="article_vectors")
