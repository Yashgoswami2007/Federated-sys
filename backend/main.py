import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
import logging
from backend.database import init_db

from backend.routers.ws import router as ws_router
from backend.middleware import HFAuthMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(title="FusionNet Backend", version="0.1.0")

app.add_middleware(HFAuthMiddleware)
# CORS_ORIGINS env var: comma-separated list of allowed origins.
# Example: CORS_ORIGINS=https://fusionnet.vercel.app,http://localhost:3000
_cors_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"}
    )

from backend.routers import (
    devices_router, rounds_router, metrics_router,
    events_router, dashboard_router, models_router, privacy_router
)

app.include_router(devices_router)
app.include_router(rounds_router)
app.include_router(metrics_router)
app.include_router(events_router)
app.include_router(dashboard_router)
app.include_router(models_router)
app.include_router(privacy_router)
app.include_router(ws_router)

@app.on_event("startup")
async def startup_event():
    if settings.BACKEND_AUTO_CREATE_TABLES:
        logger.warning("BACKEND_AUTO_CREATE_TABLES is enabled; creating missing database tables.")
        await init_db()

@app.get("/")
async def root():
    return {"message": "FusionNet Backend API"}
