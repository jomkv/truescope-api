from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from constants.enums import StreamEventType
from models.verify_claim_model import VerifyClaimModel
from models.verify_result_model import VerifyResultModel

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


@router.post("/")
async def verify_claim(verify: VerifyClaimModel):
    """
    Verify a claim by finding similar articles and checking if they support or refute it.
    """
    controller = _get_controller()
    exclude_doc_ids = [verify.exclude_id] if verify.exclude_id is not None else []
    results = await controller.verify_claim(verify.claim, exclude_doc_ids=exclude_doc_ids)
    entities = controller.extract_entities(verify.claim)
    return results


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
        claim = data.get("claim")

        if not claim:
            await websocket.send_json(
                {"type": StreamEventType.ERROR, "message": "No claim provided"}
            )
            await websocket.close()
            return

        controller = _get_controller()

        entities = controller.extract_entities(claim)
        await websocket.send_json({"entities": entities, "results": []})

        async for message in controller.verify_claim_stream_with_stats(claim):
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
