# AutoValidate — Training pipeline for embedding model (Airflow)

## Summary
- This project orchestrates fetching the latest embedding model from Hugging Face Hub, gathering user interaction training examples from a Supabase table, fine-tuning a sentence-transformer embedding model, and optionally uploading the finetuned artifact back to the Hub. Airflow is used to schedule and connect tasks; training runs inside Airflow tasks and uses sentence-transformers + PyTorch.

## Table of contents
- Overview
- Architecture & key components
- Prerequisites
- Setup (build & run)
- Required Airflow Variables / Secrets
- How it works (task flow)
- File layout
- Development & troubleshooting notes

## Overview
- Purpose: continually improve an embedding model using user interactions collected in the application. The pipeline downloads the latest base model from HF, fetches training pairs (typo → corrected) from Supabase, fine-tunes using MultipleNegativesRankingLoss, validates the model, and can push the updated model back to Hugging Face.
- Designed for reproducibility: Python dependencies installed at image build time; DAGs and utils are packaged into the image for deterministic runs.

## Architecture & key components
- **Airflow DAG**: `embedding_model_training_dag.py` — orchestrates the load → fetch → train → evaluate → (zip/upload) steps.
- **[`utils/hf_utils.py`](utils/hf_utils.py )**
  - Downloads the latest model artifact from a specified HF repo (uses HF token).
  - Returns a model directory path via XCom.
  - Zips and uploads finetuned model to HF (requires token).
- **[`utils/data_utils.py`](utils/data_utils.py )**
  - Reads training examples from Supabase table `typo_training_dataset`.
  - Converts to pandas DataFrame, renames columns, writes a temp CSV, and pushes the CSV path via XCom.
- **[`utils/training_utils.py`](utils/training_utils.py )**
  - Loads the base model from the extracted path, creates InputExample pairs, builds DataLoader, and fine-tunes the SentenceTransformer model (PyTorch).
  - Pushes the resulting trained_model_path via XCom and runs a simple evaluation step.
- **Dockerfile**
  - Based on `apache/airflow:2.7.1`; installs Python dependencies during image build for deterministic runtime.

## Prerequisites
- Docker Desktop (Windows) with at least 8 GB RAM (recommended 12+ GB) and multiple CPUs.
- HF account with a write token (if uploading).
- Supabase project and a table `typo_training_dataset` containing records with columns at least: `typo`, `corrected`, `domain`.
- Optional: GPU-enabled build if you will train on CUDA (this README assumes CPU training unless you modify the Dockerfile and torch install).

## Required Airflow Variables / environment
- **HF_TOKEN** — Hugging Face token (read + write if uploading). Provide one of:
  - Airflow Variable `HF_TOKEN` (UI or CLI), or
  - Container environment var `HF_TOKEN`.
- **SUPABASE_URL** — Supabase project URL (Airflow Variable).
- **SUPABASE_ANON_KEY** — Supabase anon/public key (Airflow Variable).

How to set in the running container (example):
- PowerShell:
  - `docker exec -it autovalidate airflow variables set HF_TOKEN "<token>"`
  - `docker exec -it autovalidate airflow variables set SUPABASE_URL "<https://xyz.supabase.co>"`
  - `docker exec -it autovalidate airflow variables set SUPABASE_ANON_KEY "<anon-key>"`

## Build & run (Windows PowerShell)
1.  Build Images
    - `docker build --progress=plain -t autovalidate-airflow:latest .`
2. Run container
   - `docker run -d --rm -p 8080:8080 --name autovalidate autovalidate-airflow:latest`
3. Follow logs / check UI
   - `docker logs -f autovalidate`
   - Open http://localhost:8080 (default admin/admin created by startup script)

## How it works (task flow)
1. **load_model_from_hf**
   - Uses HF_TOKEN to query the Hugging Face repo, lists model artifacts, picks the latest finetuned zip, downloads and extracts to a temp directory.
   - Pushes `model_dir` XCom (path to extracted content).
2. **fetch_training_data**
   - Connects to Supabase using SUPABASE_URL + SUPABASE_ANON_KEY, queries `typo_training_dataset`, converts results to pandas DataFrame, renames fields to {query, correction}, writes a temp CSV and pushes `training_data_path` XCom.
3. **train_model**
   - Pulls `model_dir` and `training_data_path` from XCom, reads CSV into DataFrame, constructs sentence-transformers InputExample pairs (both query→correction and reversed), creates DataLoader and fine-tunes for configured epochs. Saves output to temp directory and pushes `trained_model_path` XCom.
4. **evaluate_model**
   - Loads the trained model and runs basic encode calls to sanity-check model functionality.
5. **zip_and_upload_model** (optional)
   - Zips `trained_model_path` and uploads to HF using HfApi with HF_TOKEN.
   - FastAPI loads the latest finetuned model.

## File layout
- [`Dockerfile`](Dockerfile )
- [`requirements.txt`](requirements.txt )
- [`dags`](dags )
  - `embedding_model_training_dag.py`
- [`utils`](utils )
  - `data_utils.py`
  - `hf_utils.py`
  - `training_utils.py`
  - `cleanup_utils.py`
- [`README.md`](README.md )

## Development & troubleshooting notes
- **HF_TOKEN KeyError in Airflow**: set HF_TOKEN either as an Airflow Variable (UI/CLI) or pass as container env var (`-e HF_TOKEN="..."`).
- **Docker build long/pip time and failures**:
  - sentence-transformers, torch and datasets are large: prefer installing torch from the official PyTorch wheel index in the Dockerfile (pin CPU vs GPU), upgrade pip before heavy installs, and allocate more Docker Desktop RAM/CPUs.
  - For debug output: `$env:DOCKER_BUILDKIT=0; docker build --progress=plain -t autovalidate-airflow:latest .`
  - If build fails with RPC/EOF, retry or run interactive pip installs inside a temporary container to iterate quickly.
- **NameError referencing `datasets` inside sentence-transformers.fit**:
  - Ensure `datasets` is present in [`requirements.txt`](requirements.txt ) and installed at image build time.
- **db init deprecation**:
  - `airflow db init` is deprecated in newer Airflow; consider switching to `airflow db migrate` + `airflow connections create-default-connections` for production pipelines.
- **Secrets**: do not add tokens to the image. Use Airflow Variables, Docker secrets, environment variables, or a proper secrets backend.

## Best practices & tips
- Build-time installation of Python dependencies gives reproducible runtime; keep requirements pinned in requirements.txt.
- For iterative DAG development, mount your host dags folder:
  - `docker run -v S:\Projects\AutoValidate-Training-Airflow\dags:/opt/airflow/dags:ro ...`
- Use small batch sizes and CPU-optimized torch wheel for local development. For production training, use a GPU node and update the Dockerfile to install CUDA-capable torch.
- Monitor resource usage during training; PyTorch may require substantial memory.

## License & contacts
- No license file included. Treat repository as private by default. Add LICENSE as needed.
