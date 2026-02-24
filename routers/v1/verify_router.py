import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from controllers.v1.verify_controller import VerifyController
from constants.enums import StreamEventType
from models.verify_claim_model import VerifyClaimModel
from models.verify_result_model import VerifyResultModel

router = APIRouter()
controller = VerifyController()

STREAM_CAPTURE_DIR = Path("data/ws_stream_captures")


def _capture_stream_event(
    events: list[dict], direction: str, payload: dict | str
) -> None:
    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "payload": payload,
        }
    )


def _persist_stream_capture(session_id: str, events: list[dict]) -> None:
    STREAM_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = STREAM_CAPTURE_DIR / f"ws_stream_{session_id}.json"
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(events, file, ensure_ascii=False, indent=2)


@router.post("/", response_model=VerifyResultModel)
async def verify_claim(verify: VerifyClaimModel):
    """
    Verify a claim by finding similar articles and checking if they support or refute it.

    Process:
    1. Embed the claim
    2. Do similarity search in the database
    3. Extract key entities (organizations, people, locations) from claim
    4. Filter by dual criteria:
        a. Semantic relevance
        b. Entity matching
    5. For relevant articles, use NLI to check if they support/refutes/neutral
    6. Compute final score, weighing in other relevant factors (source bias, dataset verdict, nli result)

    Query Parameters:
    - limit: Number of articles to retrieve (1-50, default: 20)
    - use_fallback: Whether to analyze titles/content when no explicit claim (default: true)
    """
    results = await controller.verify_claim(verify.claim)

    entities = controller.extract_entities(verify.claim)
    timeframe = controller.extract_claim_timeframe(verify.claim)

    return {"entities": entities, "timeframe": timeframe, **results}


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
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stream_events: list[dict] = []

    async def send_and_capture(message: dict):
        _capture_stream_event(stream_events, "outbound", message)
        await websocket.send_json(message)

    try:
        # Receive claim from client
        data = await websocket.receive_json()
        _capture_stream_event(stream_events, "inbound", data)
        claim = data.get("claim")

        if not claim:
            await send_and_capture(
                {"type": StreamEventType.ERROR, "message": "No claim provided"}
            )
            await websocket.close()
            return

        # Extract metadata
        entities = controller.extract_entities(claim)

        # Send initial response
        await send_and_capture({"entities": entities, "results": []})

        # Stream results from controller
        async for message in controller.verify_claim_stream_with_stats(claim):
            await send_and_capture(message)

    except WebSocketDisconnect:
        _capture_stream_event(stream_events, "meta", "client_disconnected")
        print("Client disconnected from verify WebSocket")
    except Exception as e:
        _capture_stream_event(stream_events, "meta", {"unhandled_error": str(e)})
        try:
            await send_and_capture({"type": StreamEventType.ERROR, "message": str(e)})
        except Exception:
            pass
    finally:
        _capture_stream_event(stream_events, "meta", "connection_closed")
        _persist_stream_capture(session_id, stream_events)
        try:
            await websocket.close()
        except RuntimeError:
            pass
