import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value < min_value:
            logger.warning(
                "%s=%s is below minimum %s, using default %s",
                name,
                raw,
                min_value,
                default,
            )
            return default
        return value
    except ValueError:
        logger.warning(
            "%s=%s is invalid, expected integer. Using default %s",
            name,
            raw,
            default,
        )
        return default


def resolve_meta_path(raw_path: str) -> Path:
    return Path(raw_path.replace("\\", "/").strip())
