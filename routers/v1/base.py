from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi import APIRouter

from core.config import API_NAME, API_VERSION, ENVIRONMENT
from routers.v1 import verify_router
from routers.v1 import training_router

router = APIRouter(prefix=f"/{API_VERSION}")

router.include_router(verify_router.router, prefix="/verify", tags=["Verify"])
router.include_router(training_router.router, prefix="/training", tags=["Training"])


@router.get("/health-check")
def health_check():
    """
    Health checkpoint of current app status and version
    """
    response = {
        "status": "Online",
        "api_name": API_NAME,
        "environment": ENVIRONMENT,
        "version": API_VERSION,
    }
    return response


@router.get("/openapi")
async def openapi_endpoint():
    openapi_schema = get_openapi(
        title="API Documentation",
        version=API_VERSION,
        description="This is the API documentation for TrueScope.",
        routes=router.routes,
    )
    return JSONResponse(openapi_schema)
