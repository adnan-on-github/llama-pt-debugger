# PT Test Failure Debugger

Fine-tuned `LLaMA-3.1-8B-Instruct` on PyTorch training test failure reports (JUnit XML) for three debuggability workloads: **Summarization**, **Q&A**, and **Chatbot**.

---

## Project layout

```
llama-pt-debugger/
├── data/
│   ├── raw/                          # Drop your XML files here (or symlink)
│   ├── processed/parsed_failures.jsonl
│   └── datasets/
│       ├── raw_*.jsonl               # Seeded datasets (from create_datasets.py)
│       ├── augmented_*.jsonl         # LLM-augmented datasets
│       ├── train.jsonl               # Final training split
│       ├── val.jsonl                 # Final validation split
│       └── dataset_stats.json
├── scripts/
│   ├── parse_xml.py                  # Phase 1 — JUnit XML → JSONL
│   ├── create_datasets.py            # Phase 1 — Build 3-task datasets
│   ├── augment_data.py               # Phase 1 — Synthetic augmentation via LLM
│   └── validate_datasets.py          # Phase 1 — Dedup, filter, split
├── training/
│   ├── train.py                      # Phase 2 — Main training entrypoint
│   ├── configs/
│   │   ├── lora_config.yaml          # LoRA + training hyperparameters
│   │   ├── ds_zero3_nvidia.json      # DeepSpeed ZeRO-3 for NVIDIA
│   │   └── ds_zero3_gaudi2.json      # DeepSpeed ZeRO-3 for Gaudi2
│   └── launch/
│       ├── run_nvidia.sh             # Multi-node NVIDIA launcher (torchrun)
│       └── run_gaudi2.sh             # Multi-node Gaudi2 launcher (gaudi_spawn / mpirun)
├── inference/
│   ├── infer.py                      # Hardware-aware model loader + generate()
│   ├── summarize.py                  # Summarization task module
│   ├── qa.py                         # Q&A task module
│   └── chatbot.py                    # Stateful multi-turn chatbot module
├── serving/
│   ├── app.py                        # Gradio 3-tab UI
│   ├── api.py                        # FastAPI REST API
│   └── k8s/
│       ├── configmap.yaml
│       ├── deployment.yaml           # Gaudi2 HPU deployment
│       └── service.yaml
├── eval/
│   ├── evaluate.py                   # ROUGE / F1 / Perplexity evaluation
│   └── results/
└── requirements/
    ├── requirements-nvidia.txt
    ├── requirements-gaudi2.txt
    └── requirements-serve.txt
```

---

## Quick-start

### 1 — Install dependencies

**NVIDIA GPU (A100/H100):**
```bash
pip install -r requirements/requirements-nvidia.txt
```

**Gaudi2 HPU** (install Habana vault packages first per the [Habana docs](https://docs.habana.ai/en/latest/Installation_Guide/)):
```bash
pip install -r requirements/requirements-gaudi2.txt
```

---

### 2 — Prepare data

```bash
# Step 1: Parse your JUnit XML test reports
python scripts/parse_xml.py \
    --xml_dir data/raw \
    --output data/processed/parsed_failures.jsonl \
    --failures_only

# Step 2: Build seed datasets for 3 tasks
python scripts/create_datasets.py

# Step 3: Augment with LLM-generated completions
export LLM_API_KEY=sk-...                         # or EMPTY for local vLLM
export LLM_BASE_URL=https://api.openai.com/v1     # or your local endpoint
export LLM_MODEL=gpt-4o
python scripts/augment_data.py --tasks summarize qa chatbot

# Step 4: Validate, deduplicate, split → train.jsonl + val.jsonl
python scripts/validate_datasets.py
```

---

### 3 — Fine-tune

**Single-node NVIDIA (8 GPUs):**
```bash
bash training/launch/run_nvidia.sh
```

**Single-node Gaudi2 (8 HPU cards):**
```bash
bash training/launch/run_gaudi2.sh
```

**Multi-node NVIDIA:**
```bash
MASTER_ADDR=10.0.0.1 NUM_NODES=2 GPUS_PER_NODE=8 \
    bash training/launch/run_nvidia.sh
```

**Multi-node Gaudi2:**
```bash
# Create training/launch/hostfile first:
# node01 slots=8
# node02 slots=8
NUM_NODES=2 GAUDI_PER_NODE=8 MASTER_ADDR=10.0.0.1 \
    bash training/launch/run_gaudi2.sh
```

The fine-tuned adapter is saved to `outputs/llama-pt-debugger/`.

---

### 4 — Evaluate

```bash
python eval/evaluate.py \
    --val_file data/datasets/val.jsonl \
    --adapter_path outputs/llama-pt-debugger \
    --device auto \
    --max_samples 500
```

Results are written to `eval/results/eval_report.json`.

---

### 5 — Serve

**Gradio UI (port 7860):**
```bash
pip install -r requirements/requirements-serve.txt
python serving/app.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --adapter_path outputs/llama-pt-debugger \
    --device auto
```

**FastAPI REST API (port 8000):**
```bash
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct \
ADAPTER_PATH=outputs/llama-pt-debugger \
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

**Kubernetes (Gaudi2):**
```bash
kubectl create namespace pt-debugger
kubectl apply -f serving/k8s/configmap.yaml
kubectl apply -f serving/k8s/deployment.yaml
kubectl apply -f serving/k8s/service.yaml
```

---

## REST API reference

| Method | Endpoint     | Description                                |
|--------|-------------|---------------------------------------------|
| GET    | `/health`   | Server health check                         |
| GET    | `/ready`    | Readiness — 503 until model is loaded       |
| POST   | `/summarize`| Summarize a test failure (XML or text)      |
| POST   | `/qa`       | Answer a question about a failure           |
| POST   | `/chat`     | Multi-turn chatbot (stateless, pass history)|

**Summarize example:**
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"failure_input": "<testsuite>...<failure>AssertionError</failure></testsuite>"}'
```

**Q&A example:**
```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"context": "RuntimeError: NCCL timeout...", "question": "What is the root cause?"}'
```

**Chat example:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My test failed with RuntimeError: NCCL timeout. Help me debug."}
    ]
  }'
```

---

## Task prompt format

All three tasks share a task-prefix convention:

| Task | Prefix |
|---|---|
| Summarization | `[TASK: SUMMARIZE]` |
| Q&A | `[TASK: QA]` |
| Chatbot | LLaMA-3 ChatML with system prompt |

This allows a single fine-tuned model to handle all three workloads at inference time by routing on the task prefix.

---

## Hardware notes

| Setting | NVIDIA (A100/H100) | Gaudi2 |
|---|---|---|
| Trainer | `trl.SFTTrainer` | `optimum-habana.GaudiSFTTrainer` |
| Precision | bf16 | bf16 |
| Attention | Flash-Attention 2 | Standard (HPU kernel) |
| Lazy mode | N/A | `PT_HPU_LAZY_MODE=1` |
| Distributed | NCCL | HCCL |
| DeepSpeed | `ds_zero3_nvidia.json` | `ds_zero3_gaudi2.json` |

The training script (`training/train.py`) auto-detects the hardware and selects the correct path.
