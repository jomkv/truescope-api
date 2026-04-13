from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from constants.enums import StreamEventType
from models.verify_claim_model import VerifyClaimModel
from models.verify_result_model import VerifyResultModel
from models.article_result_model import ArticleResultModel

router = APIRouter()


def _get_controller():
    """
    Return the shared VerifyController singleton from main.py.
    Using a lazy getter avoids circular import issues at module load time
    and ensures all endpoints (REST, WebSocket, simulation) share the same
    instance — including any live threshold changes applied via the dashboard.
    """
    import main

    return main.verify_controller


@router.post("/", response_model=VerifyResultModel)
async def verify_claim(verify: VerifyClaimModel):
    """
    Verify a claim by finding similar articles and checking if they support or refute it.
    """
    controller = _get_controller()
    config = verify.config.model_dump(exclude_none=True) if verify.config else None
    results = await controller.verify_claim(
        verify.claim,
        config=config,
    )
    entities = controller.extract_entities(verify.claim)
    timeframe = controller.extract_claim_timeframe(verify.claim)
    return {"entities": entities, "timeframe": timeframe, **results}


@router.post("/calculate")
async def calculate_score(evidences: list[ArticleResultModel]):
    controller = _get_controller()
    stats = controller.calculate_stats(evidences)

    return stats


@router.websocket("/ws")
async def websocket_verify_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time verification streaming with stats.

    Client sends:
    {"claim": "Your claim to verify here"}

    Server sends:
    1. {"entities": [...], "timeframe": null, "results": []} (initial metadata)
    2. {"type": "result", "data": {...}} (each article)
    3. {"type": "complete", "total_results": ..., "stats": {...}} (final stats)
    """
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        try:
            payload = VerifyClaimModel.model_validate(data)
        except ValidationError as e:
            await websocket.send_json(
                {
                    "type": StreamEventType.ERROR,
                    "message": "Invalid payload",
                    "details": e.errors(),
                }
            )
            await websocket.close()

            # Prevent error from propagating to parent try-catch
            return

        if not payload.claim:
            await websocket.send_json(
                {"type": StreamEventType.ERROR, "message": "No claim provided"}
            )
            await websocket.close()
            return

        claim = payload.claim
        config = (
            payload.config.model_dump(exclude_none=True) if payload.config else None
        )

        controller = _get_controller()

        entities = controller.extract_entities(claim)
        await websocket.send_json({"entities": entities, "results": []})

        async for message in controller.verify_claim_stream_with_stats(claim, config):
            await websocket.send_json(message)

    except WebSocketDisconnect:
        print("Client disconnected from verify WebSocket")
    except Exception as e:
        try:
            await websocket.send_json(
                {"type": StreamEventType.ERROR, "message": str(e)}
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
