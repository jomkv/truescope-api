from sentence_transformers import SentenceTransformer

class EmbeddingService:
  def __init__(self):
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    self.model = SentenceTransformer(model_name)

  def embed_text(self, text: str) -> list[float]:
    return self.model.encode(text)