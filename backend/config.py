import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/fusionnet")
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
