import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in .env file")

login(token=token)
print("Logged in to Hugging Face successfully.")
