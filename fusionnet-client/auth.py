import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load from .env in repo root (one level up from fusionnet-client/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def get_token() -> str:
    """Returns HF_TOKEN from environment. Raises if not set."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Add it to your .env file.")
    return token


def hf_login():
    """Authenticate with Hugging Face using HF_TOKEN from .env."""
    token = get_token()
    login(token=token)
    print("Logged in to Hugging Face successfully.")


if __name__ == "__main__":
    hf_login()
