from sentence_transformers import SentenceTransformer
from pathlib import Path


class EmbeddingService:
    def __init__(self):
        # Default base model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Check for latest trained version
        try:
            meta_path = Path("data/model_adapters/embeddings_meta.json")
            if meta_path.exists():
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    trained_path = Path(meta.get("current_path", ""))
                    if trained_path.exists():
                        model_name = str(trained_path)
                        import logging
                        logging.getLogger(__name__).info(f"Auto-loading trained embedding model: {model_name}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error checking for trained model: {e}")

        self._model_name = model_name
        self.model = SentenceTransformer(model_name)

    def reload_model(self, model_path: str | Path) -> bool:
        """
        Hot-swap the embedding model from a fine-tuned checkpoint.
        Used by FeedbackTrainer.reload_embeddings_into_service().
        Returns True if loaded successfully.
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return False
            self.model = SentenceTransformer(str(model_path))
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to reload embedding model: {e}")
            return False

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text)