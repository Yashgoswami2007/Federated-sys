# ── FusionNet Backend (FastAPI + PostgreSQL/SQLite) ────────────────────────────
# Build context should be the repo root so we can access .env
# Usage: docker build -f backend/Dockerfile .

FROM python:3.11-slim

WORKDIR /app

# System deps for asyncpg / psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install backend Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source as a package
COPY backend/ ./backend/

# Copy repo-root .env (will be overridden by env_file in docker-compose)
# .env is injected via docker-compose env_file at runtime

EXPOSE 8000

# Environment defaults (overridable via docker-compose / .env)
ENV BACKEND_AUTH_DISABLED=true \
    BACKEND_AUTO_CREATE_TABLES=true \
    DATABASE_URL=sqlite+aiosqlite:///./fusionnet.db \
    CORS_ORIGINS=*

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
