# FusionNet Project File Summary

## fusionnet-client summary

- `README.md`
  - Explains what the local client does.
  - Shows quickstart steps, CLI arguments, and describes the folder layout.

- `requirements.txt`
  - Lists Python packages needed to run the client.
  - Includes `torch`, `transformers`, `bitsandbytes`, `opacus`, `datasets`, `pyyaml`, `tqdm`, `numpy`, and `python-dotenv`.

- `auth.py`
  - Loads `HF_TOKEN` from the `.env` file in the repo root using `python-dotenv`.
  - Calls `huggingface_hub.login()` to authenticate before model downloads.
  - Run once before starting any training or model loading: `python fusionnet-client/auth.py`.

- `config.yaml`
  - Stores default settings for model, federation, dataset, privacy, and device profiles.
  - `model.name` is fixed to `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` — the federation-wide model. `models/loader.py` enforces this regardless of what is set here.
  - `federation.hub.repo_id` points to `yash-goswami/fusionnet-coordinator` (private HF Dataset).
  - LoRA rank and batch size are overridden per device profile at runtime.

- `main.py`
  - CLI entry point to start the client.
  - Accepts `--client-id` (int), `--num-clients` (int), and `--rounds` (int) arguments.
  - Authenticates with HF at startup via `auth.hf_login()`.
  - Runs a full FL round loop: pull global A → local train → push local A update to HF Hub.

- `client.py`
  - Main client class `FusionNetClient(config_path, client_id)`.
  - `client_id` is passed into the constructor and forwarded to `FederatedClient` as `"client_{id}"`, ensuring each node uploads to a unique file on the HF Hub.
  - `train(num_clients)` handles dataset loading with the correct Dirichlet shard and runs local training epochs.

- `training/engine.py`
  - Defines training utilities.
  - Builds the DataLoader and optimizer for AFLoRA A, B and Lambda parameters.
  - Runs one local training epoch with correct gradient lifecycle: `zero_grad → forward → backward → step → zero_grad`.
  - Both Opacus and CustomPrivacyEngine paths call `zero_grad()` after `step()` to prevent gradient accumulation across batches.
  - **Updated:** Clones input IDs to initialize target labels (`batch['labels'] = batch['input_ids'].clone()`) to prevent shape mismatch issues on target causal text generation tasks like Banking77.

- `models/loader.py`
  - Detects hardware profile and loads the federation model (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
  - Any `model.name` value in config is overridden to the federation constant with a warning — prevents aggregation crashes from shape mismatches.
  - Uses 4-bit NF4 quantization on GPU (bitsandbytes); falls back to FP32 on CPU (`device_map=None`).
  - **Updated:** Uses `torch_dtype` instead of the invalid `dtype` parameter when calling `from_pretrained()` to prevent runtime errors.
  - Freezes all base model weights so only AFLoRA A, B and Lambda matrices train.

- `aflora/layer.py`
  - Defines the `AFLoRALayer`.
  - Implements the adapter math `ΔW = A × Λ × B`.
  - Trains A locally alongside B/Λ to capture and export the local learning signal.
  - **Updated:** Casts AFLoRA weights (`A`, `B`, and `Lambda`) to the input tensor's `device` and `dtype` dynamically during `forward()`, avoiding CPU/CUDA device and precision mismatches.

- `aflora/injection.py`
  - Finds target modules in the model and replaces them with AFLoRA layers.
  - Provides helper functions to get AFLoRA parameters and layers.

- `federation/hf_hub.py`
  - `HFParameterServer` — wraps HF Hub as the federated parameter server.
  - Authenticates via `auth.get_token()` (reads `HF_TOKEN` from `.env`).
  - `upload_local_A_matrices(round_num, client_id, updates)` — pushes `round_N/client_K.pt` to the private repo.
  - `download_global_A_matrices(round_num)` — fetches `global/Global_A_round_N.pt`; returns `None` if not yet available.
  - Repo: `yash-goswami/fusionnet-coordinator` (private HF Dataset).

- `federation/privacy.py`
  - Manages differential privacy.
  - Tries to use Opacus DP-SGD; if that fails, uses a custom DP fallback.
  - Also converts tensors to/from base64 for safe JSON transfer.

- `fl_datasets/__init__.py`
  - Marks `fl_datasets/` as a Python package so relative imports work correctly.

- `fl_datasets/loader.py`
  - Loads datasets like Banking77, SST-2, IMDB, or AG News.
  - Tokenizes text and prepares PyTorch tensors.
  - Calls `dirichlet_partition()` after tokenising, passing `device_tier`, `client_id`, and `num_clients`.
  - **Updated:** Disables Hugging Face dataset caching with `hf_datasets.disable_caching()` before running the map tokenization function. This resolves parallel write lock collisions when running multiple local clients on Windows.
  - Prints a per-node partition report (label distribution bar chart) at startup for demo visibility.

- `fl_datasets/partitioner.py`
  - **New file.** Implements device-tier-aware Dirichlet Non-IID data partitioning.
  - `dirichlet_partition(dataset, device_tier, client_id, num_clients)` — carves a shard whose size and label skewness both scale with the hardware tier.
  - `TIER_PARTITION_CONFIG` maps each device tier to a Dirichlet α and a data fraction cap:
    - `MI300X`:      α = 10.0 (balanced),      100% of corpus
    - `RX_7900_XTX`: α = 2.0  (mild skew),     50% of corpus
    - `Steam_Deck`:  α = 0.5  (heavy skew),    20% of corpus
    - `CPU_only`:    α = 0.1  (extreme skew),  8% of corpus
  - `print_partition_report()` — prints an ASCII bar chart of the shard's top-5 label distribution, visible to judges without extra tooling.
  - `describe_partition()` — returns a summary dict for programmatic access to shard statistics.

- `scripts/example_train.py`
  - Example script that runs local training from the client.
  - **Updated:** accepts `--client-id` and `--num-clients` CLI arguments.
  - Shows how to use `FusionNetClient` with the config and Dirichlet partitioning.

- `scripts/example_federated_round.py`
  - Example script that simulates one federated round.
  - Saves local adapter state, exports updates, and receives a simulated coordinator broadcast.

---

## fusionnet folder summary

- `__init__.py`
  - Marks `fusionnet` as a Python package.

- `models/llama_loader.py`
  - Loads a 4-bit Llama model and tokenizer.
  - Prepares the model for k-bit training and injects LoRA adapters using PEFT.
  - Returns the LoRA-enabled model and tokenizer.
  - **Note:** hardcodes 4-bit quantization; unsuitable for CPU-only nodes. Use `model_selector.py` to pick the right model first.

- `models/model_selector.py`
  - Selects a model for a **standalone single-device** deployment based on local VRAM or RAM.
  - **⚠️ NOT compatible with federated use as-is.** In a federated system, the model is a federation-wide constant decided by the coordinator — not a per-device choice. If different devices load different architectures (e.g. Llama-3-8B on GPU, TinyLlama on CPU), the AFLoRA `A` matrices have different shapes and FedAvg aggregation crashes. This file is only valid for single-node inference or for a future cohort-based coordinator that runs separate FedAvg rounds per architecture group.
  - For the hackathon demo, the federation model must be a single agreed-upon model (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) hardcoded in `config.yaml`. This file should not be called by `fusionnet-client` until cohort-based aggregation is implemented.

- `core/fl_coordinator.py`
  - Coordinates federated learning rounds on the client side.
  - Applies global weights, executes local training, extracts LoRA adapter weights, and returns updated weights and metrics.

- `core/dp_sgd.py`
  - Sets up Opacus DP-SGD for the model.
  - Provides helper functions for privacy metrics and legacy DP noise stubs.

- `core/lora_trainer.py`
  - Handles local LoRA training logic.
  - Detects hardware profile, configures batch size and rank, enables DP, and trains for epochs.

- `core/hardware_utils.py`
  - Detects available GPU or CPU hardware (supports both ROCm and CUDA via `torch.cuda`).
  - Returns a hardware profile with device type, batch size, LoRA rank, contribution weight, and backend.
  - On Windows, adds `System32` to the DLL search path before accessing `torch.cuda`.

- `core/aggregator.py`
  - Implements FedAvg weighted model aggregation.
  - Computes an averaged state dict from multiple clients.
  - Includes a stub for secure aggregation.

- `comms/rccl_backend.py`
  - Placeholder for RCCL/distributed initialization.
  - Intended for multi-GPU communication but currently not implemented.

- `kernels/dp_noise.hip`
  - Custom HIP kernel source for DP noise generation (on AMD ROCm).
  - Used by the project for privacy-preserving gradient noise if deployed on AMD hardware.

---

## docs folder summary

- `amd_integration.md`
  - Explains how the project uses AMD ROCm, PyTorch, bitsandbytes, and Opacus.
  - Describes planned RCCL secure aggregation and fallback DP handling for 4-bit modules.

- `architecture.md`
  - High-level architecture documentation.
  - Covers hardware abstraction, dynamic device detection, model selection, and communication of AFLoRA `A` matrices.
  - Documents the Dirichlet Non-IID partitioning strategy and the device-tier to data-distribution mapping.
  - Documents the model architecture constraint: all clients in the same federation round must share identical model architecture for AFLoRA aggregation to work.

- `hackathon_qa_prep.md`
  - Contains anticipated hackathon judge questions and model answers.
  - Covers: weak-device contributions, weighted aggregation, privacy tradeoffs, data heterogeneity proof, and model choice rationale for CPU-only demo nodes.

- `privacy_proof.md`
  - Describes differential privacy guarantees and mathematical formulas.
  - Explains the Opacus-based DP-SGD workflow and the custom DP fallback formula.

---

## experiments folder summary

- `benchmarks/plot_convergence.py`
  - Placeholder script for plotting training convergence metrics.
  - Intended to visualize benchmark results for model training.

- `mvp_sentiment/run_mvp.py`
  - Minimal stub (currently just a print statement).
  - Intended to be the end-to-end MVP demo script for the sentiment analysis federated round.

---

## scripts folder summary

- `hf_coordinator.py`
  - Serverless FL coordinator using HF Hub as parameter server.
  - Authenticates via `HF_TOKEN` from `.env`.
  - Polls `list_repo_files()` until all expected client `.pt` files appear for the round.
  - Downloads all client A matrices, runs FedAvg layer-by-layer, and uploads `global/Global_A_round_N.pt`.
  - Args: `--repo-id` (default: `yash-goswami/fusionnet-coordinator`), `--num-clients`, `--rounds`.
  - Run in parallel with client nodes: `python scripts/hf_coordinator.py --num-clients 3 --rounds 3`.

- `setup_cuda.sh`
  - Script to install NVIDIA CUDA-compatible PyTorch and dependencies.
  - Verifies CUDA availability and prints GPU information after setup.

- `setup_rocm.sh`
  - Script to install AMD ROCm-compatible PyTorch and dependencies.
  - Verifies ROCm GPU availability and prints device information after setup.

- `test_local_training.py`
  - Python test script for local training and federated pipeline logic.
  - Loads a model via `select_model_for_hardware()` (hardware-aware), simulates a local FL round, and tests FedAvg aggregation.
  - Falls back to a dummy `torch.nn.Linear` model if bitsandbytes/ROCm is not configured, allowing logic testing on CPU-only machines.
