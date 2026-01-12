from sentence_transformers import SentenceTransformer
from sqlalchemy import select, text
from schemas.article_vector_schema import ArticleVector
from schemas.article_schema import Article
from models.article_vector_model import ArticleVectorModel
from models.article_model import ArticleModel


class VerifyController:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.model = "joeddav/xlm-roberta-large-xnli"

    def embed_claim(self, claim: str) -> list[float]:
        return self.embedding_model.encode(claim)

    @staticmethod
    def get_articles_by_doc_ids(session, doc_ids: list[str]) -> list[Article]:
        """
        Retrieve articles by a list of doc_ids.
        """
        stmt = select(Article).where(Article.doc_id.in_(doc_ids))
        articles = session.execute(stmt).scalars().all()

        return articles

    @staticmethod
    def find_similar_embeddings(
        session, embedding, top_k=5
    ) -> list[ArticleVectorModel]:
        """
        Search for top_k most similar embeddings in the database.
        """
        session.execute(text("SET hnsw.ef_search = 120"))

        stmt = (
            select(ArticleVector)
            .order_by(ArticleVector.embedding.cosine_distance(embedding))
            .limit(top_k)
        )
        article_vectors = session.execute(stmt).scalars().all()

        return article_vectors

    def get_relevant_articles(
        self, session, claim: str, limit: int = 5
    ) -> list[ArticleModel]:
        """
        Get all relevant articles from claim
        """
        embedding = self.embed_claim(claim)
        similar_embeddings = self.find_similar_embeddings(session, embedding, limit)
        doc_ids = [v.doc_id for v in similar_embeddings]
        relevant_articles = self.get_articles_by_doc_ids(session, doc_ids)

        return relevant_articles
