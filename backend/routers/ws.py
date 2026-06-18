from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.websocket.manager import manager
import logging

router = APIRouter(prefix="/ws", tags=["websocket"])
logger = logging.getLogger(__name__)

@router.websocket("/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    await manager.connect(websocket, channel)
    logger.info(f"WebSocket connected to channel: {channel}")
    try:
        while True:
            # We don't expect messages from the client right now, but we must receive to keep the connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
        logger.info(f"WebSocket disconnected from channel: {channel}")
