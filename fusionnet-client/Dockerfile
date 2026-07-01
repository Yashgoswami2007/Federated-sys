# ── FusionNet FL Client (Federated Learning Edge Node) ────────────────────────
# Build context should be the repo root.
#
# For AMD ROCm GPU, swap base image to: rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2
# For NVIDIA CUDA, swap base image to: pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (swap for ROCm/CUDA wheel in production)
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining client deps
COPY fusionnet-client/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy FL client source
COPY fusionnet-client/ ./

# Copy fusionnet core library (needed for aggregator, comms imports)
COPY fusionnet/ ./fusionnet/

# Copy .env from repo root
# .env is injected via docker-compose env_file at runtime

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--client-id", "0", "--rounds", "3", "--num-clients", "3"]
