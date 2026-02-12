import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import v1_router

LOGGER = logging.getLogger(__name__)


def init_routers(_app: FastAPI) -> None:
    _app.include_router(v1_router)


def init_cors(_app: FastAPI) -> None:
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_app() -> FastAPI:
    try:
        LOGGER.info("Initiliase fast-API app")
        _app = FastAPI(
            title="TrueScope API",
        )
        init_routers(_app=_app)
        init_cors(_app=_app)
        return _app
    except Exception as e:
        LOGGER.error(f"Error in fast-API app initialisation => {e}")


app: FastAPI = create_app()
