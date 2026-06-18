from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings

from backend.routers import (
    devices_router, rounds_router, metrics_router,
    events_router, dashboard_router, models_router, privacy_router
)
from backend.routers.ws import router as ws_router
from backend.middleware import HFAuthMiddleware

app = FastAPI(title="FusionNet Backend", version="0.1.0")

app.add_middleware(HFAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(devices_router)
app.include_router(rounds_router)
app.include_router(metrics_router)
app.include_router(events_router)
app.include_router(dashboard_router)
app.include_router(models_router)
app.include_router(privacy_router)
app.include_router(ws_router)

@app.get("/")
async def root():
    return {"message": "FusionNet Backend API"}
