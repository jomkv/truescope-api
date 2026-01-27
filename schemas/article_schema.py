from core.db import Base
from sqlalchemy import Column, String, DateTime


class Article(Base):
    __tablename__ = "articles"

    doc_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    publish_date = Column(DateTime, nullable=False)
    url = Column(String, nullable=False)
    source = Column(String)
    type = Column(String)
    source_bias = Column(String)
