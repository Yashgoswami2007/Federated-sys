from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from huggingface_hub import HfApi
import logging
from cachetools import TTLCache
from backend.config import settings

logger = logging.getLogger(__name__)

class HFAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, bypass_paths=None):
        super().__init__(app)
        self.bypass_paths = bypass_paths or ["/docs", "/openapi.json", "/ws", "/api/dashboard/kpi"]
        # Max 1000 tokens, expire after 1 hour (3600 seconds)
        self._token_cache = TTLCache(maxsize=1000, ttl=3600)

    async def dispatch(self, request: Request, call_next):
        if settings.BACKEND_AUTH_DISABLED:
            request.state.user = {"name": "local-dev"}
            return await call_next(request)

        if any(request.url.path.startswith(path) for path in self.bypass_paths):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid token"},
            )

        token = auth_header.replace("Bearer ", "")
        
        if token not in self._token_cache:
            try:
                api = HfApi(token=token)
                info = api.whoami()
                self._token_cache[token] = info
                logger.info(f"Authenticated user: {info.get('name')}")
            except Exception as e:
                logger.warning(f"HF Authentication failed: {str(e)}")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid token"},
                )

        request.state.user = self._token_cache[token]
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
