from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from controllers.v1.verify_controller import VerifyController
from models.verify_claim_model import VerifyClaimModel
from models.verify_result_model import VerifyResultModel


router = APIRouter()
controller = VerifyController()


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

    try:
        # Receive claim from client
        data = await websocket.receive_json()
        claim = data.get("claim")

        if not claim:
            await websocket.send_json({"type": "error", "message": "No claim provided"})
            await websocket.close()
            return

        # Extract metadata
        entities = controller.extract_entities(claim)
        timeframe = controller.extract_claim_timeframe(claim)

        # Send initial response
        await websocket.send_json(
            {"entities": entities, "timeframe": timeframe, "results": []}
        )

        # Stream results with stats from controller
        async for message in controller.verify_claim_stream_with_stats(claim):
            if message["type"] == "result":
                await websocket.send_json({"type": "result", "data": message["data"]})
            else:
                # Send complete message with final stats
                await websocket.send_json(message)

    except WebSocketDisconnect:
        print("Client disconnected from verify WebSocket")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await websocket.close()
