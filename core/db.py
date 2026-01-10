from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core.config import DATABASE_URI

engine = create_engine(DATABASE_URI, pool_pre_ping=True)
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _init_extensions():
    """Initialize required database extensions"""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def create_tables():
    """Create all tables defined in models"""
    _init_extensions()
    Base.metadata.create_all(engine)
