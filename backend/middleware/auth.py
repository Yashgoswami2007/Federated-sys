from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from huggingface_hub import HfApi
import logging

logger = logging.getLogger(__name__)

class HFAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, bypass_paths=None):
        super().__init__(app)
        self.bypass_paths = bypass_paths or ["/docs", "/openapi.json", "/ws", "/api/dashboard/kpi"]
        self._token_cache = {}

    async def dispatch(self, request: Request, call_next):
        if any(request.url.path.startswith(path) for path in self.bypass_paths):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            # For hackathon/demo, we allow unauthenticated requests, but in production this should block
            # return await call_next(request)  # Uncomment this to bypass auth entirely
            raise HTTPException(status_code=401, detail="Missing or invalid token")

        token = auth_header.replace("Bearer ", "")
        
        if token not in self._token_cache:
            try:
                api = HfApi(token=token)
                info = api.whoami()
                self._token_cache[token] = info
                logger.info(f"Authenticated user: {info.get('name')}")
            except Exception as e:
                logger.warning(f"HF Authentication failed: {str(e)}")
                raise HTTPException(status_code=401, detail="Invalid token")

        request.state.user = self._token_cache[token]
        response = await call_next(request)
        return response
