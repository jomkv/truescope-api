import json
import logging
from sentence_transformers import SentenceTransformer
from shared.helpers import resolve_meta_path
from pathlib import Path


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        # Default base model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Check for latest trained version
        try:
            meta_path = Path("data/model_adapters/embeddings_meta.json")
            if not meta_path.exists():
                logger.warning(
                    "Embedding meta file not found at %s; using base model.",
                    meta_path,
                )
            else:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                raw_path = str(meta.get("current_path", "")).strip()
                if not raw_path:
                    logger.warning(
                        "Embedding meta has empty current_path; using base model."
                    )
                else:
                    trained_path = resolve_meta_path(raw_path)
                    if not trained_path.exists():
                        logger.warning(
                            "Embedding adapter path not found; using base model. raw_path='%s' normalized='%s' cwd='%s'",
                            raw_path,
                            trained_path,
                            Path.cwd(),
                        )
                    else:
                        model_name = str(trained_path)
                        logger.warning(
                            "Embedding model loaded from %s",
                            model_name,
                        )
        except Exception as e:
            logger.error("Error checking for trained model: %s", e)

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
            logger.error("Failed to reload embedding model: %s", e)
            return False

    def embed_text(self, text: str) -> list[float]:
        return self.model.encode(text)
