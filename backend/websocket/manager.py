from fastapi import WebSocket
from typing import Dict, List
import json

class WebSocketManager:
    def __init__(self):
        # Dictionary to hold active connections per channel
        # Channels: 'devices', 'rounds', 'metrics', 'events', 'all'
        self.active_connections: Dict[str, List[WebSocket]] = {
            "devices": [],
            "rounds": [],
            "metrics": [],
            "events": [],
            "all": []
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)
        else:
            self.active_connections[channel] = [websocket]

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections and websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, channel: str, message: dict):
        # Broadcast to the specific channel
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception:
                    # Ignore failing connections (they'll be cleaned up on disconnect)
                    pass
        
        # Broadcast to the 'all' channel as well
        if channel != "all" and "all" in self.active_connections:
            for connection in self.active_connections["all"]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = WebSocketManager()
