import struct
from sqlalchemy.types import TypeDecorator, BLOB

try:
    from pgvector.sqlalchemy import VECTOR as PG_VECTOR
except ImportError:
    PG_VECTOR = None


class VectorType(TypeDecorator):
    """
    Handles vector storage for Libsql/Turso using binary F32_BLOB format.
    Translates Python lists to binary blobs using struct.pack.
    """

    impl = BLOB
    cache_ok = True

    def __init__(self, dimensions=384):
        super().__init__()
        self.dimensions = dimensions
        self.format = f"<{dimensions}f"

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, list):
            if len(value) != self.dimensions:
                raise ValueError(
                    f"Vector length must be {self.dimensions}, got {len(value)}"
                )
            return struct.pack(self.format, *value)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, bytes):
            return list(struct.unpack(self.format, value))
        return value


def get_vector_type(dimensions=384):
    """
    Returns the appropriate vector type.
    Prioritizes VectorType (binary F32_BLOB) for Turso compatibility.
    """
    # Force VectorType to avoid conflicts with pgvector in the environment
    # when connecting to Turso.
    return VectorType(dimensions)
