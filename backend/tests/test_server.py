import asyncio
import websockets

async def test_ws_connection():
    uri = "ws://backend:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send("ping")
        response = await websocket.recv()
        assert response == "pong" 