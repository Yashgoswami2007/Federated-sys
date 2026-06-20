import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/fusionnet")
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    PORT: int = int(os.getenv("PORT", "8000"))
    BACKEND_AUTH_DISABLED: bool = os.getenv("BACKEND_AUTH_DISABLED", "false").lower() in {"1", "true", "yes"}
    BACKEND_AUTO_CREATE_TABLES: bool = os.getenv("BACKEND_AUTO_CREATE_TABLES", "false").lower() in {"1", "true", "yes"}
    BACKEND_IN_MEMORY: bool = os.getenv("BACKEND_IN_MEMORY", "false").lower() in {"1", "true", "yes"}

settings = Settings()
