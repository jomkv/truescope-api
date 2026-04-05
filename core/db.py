import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core.config import DATABASE_URI, TURSO_DATABASE_URL, TURSO_AUTH_TOKEN, USE_LOCAL_SQLITE, LOCAL_DB_PATH

logger = logging.getLogger(__name__)

# Standard SQLAlchemy sqlite dialect Base
Base = declarative_base()

if USE_LOCAL_SQLITE or (TURSO_DATABASE_URL and TURSO_AUTH_TOKEN):
    try:
        import libsql

        class LibSQLCursorWrapper:
            def __init__(self, cursor, sanitizer):
                self.cursor = cursor
                self.sanitizer = sanitizer

            def __getattr__(self, name):
                return getattr(self.cursor, name)

            def execute(self, sql, params=None):
                if params is not None:
                    params = self.sanitizer(params)
                return self.cursor.execute(sql, params)

            def executemany(self, sql, param_list):
                new_param_list = [self.sanitizer(p) for p in param_list]
                return self.cursor.executemany(sql, new_param_list)

        class LibSQLConnectionWrapper:
            def __init__(self, conn):
                self.conn = conn

            def __getattr__(self, name):
                return getattr(self.conn, name)

            def cursor(self):
                return LibSQLCursorWrapper(self.conn.cursor(), self._sanitize_params)

            def _sanitize_params(self, params):
                if isinstance(params, (list, tuple)):
                    return [self._sanitize_params(p) for p in params]
                if isinstance(params, dict):
                    return {k: self._sanitize_params(v) for k, v in params.items()}
                if isinstance(params, memoryview):
                    return params.tobytes()
                return params

            def create_function(self, *args, **kwargs):
                pass

        def get_libsql_connection():
            if USE_LOCAL_SQLITE:
                conn = libsql.connect(LOCAL_DB_PATH)
            else:
                url = TURSO_DATABASE_URL
                if url.startswith("libsql://"):
                    url = url.replace("libsql://", "https://", 1)
                conn = libsql.connect(url, auth_token=TURSO_AUTH_TOKEN)
            return LibSQLConnectionWrapper(conn)

        engine = create_engine(
            "sqlite://",
            creator=get_libsql_connection,
            pool_pre_ping=True
        )
    except ImportError:
        logger.error("libsql package missing. Vector features will not work.")
        engine = create_engine(DATABASE_URI, pool_pre_ping=True)
else:
    engine = create_engine(DATABASE_URI, pool_pre_ping=True)

Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = Session()
    try:
        yield db
    finally:
        db.close()

def drop_tables():
    """Drop all tables one-by-one with retries to avoid cloud timeouts."""
    import time
    tables = ["claims", "article_chunks", "articles"]

    with engine.connect() as conn:
        for table in tables:
            print(f"  - Dropping {table}...")
            for attempt in range(3):
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                    conn.commit()
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"    ⚠️ Failed to drop {table}: {e}")
                    else:
                        print(f"    🔄 Retrying {table} drop...")
                        time.sleep(2)

def create_tables():
    """Create all tables defined in models and establish vector indexes"""
    # Import models so they are registered with Base before create_all.
    # Note: Using singular filenames as they exist on disk.
    import schemas.article_schema  # noqa: F401
    import schemas.article_chunk_schema  # noqa: F401
    import schemas.claim_schema  # noqa: F401

    Base.metadata.create_all(engine)

    if TURSO_DATABASE_URL:
        # Note: Vector indexes (DiskANN) should be created manually after upload is complete.
        print("💡 Skipping automatic vector index creation (will create after upload).")
        pass

if __name__ == "__main__":
    if TURSO_DATABASE_URL:
        print(f"Targeting Turso Cloud: {TURSO_DATABASE_URL}")
        create_tables()
    else:
        print("Targeting Local sqlite/Postgres")
        create_tables()
    print("Database initialization complete.")

