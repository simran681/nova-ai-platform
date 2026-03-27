# NOVA AI Support & Personalization Platform

> AI Engineer Assessment — Full implementation of NOVA's AI Support Platform across 5 tasks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

## Shareable Links

| Resource | Link |
|----------|------|
| Task 1 — Prompt Engineering Colab | [Open Notebook](https://drive.google.com/file/d/1qxJ9Ey83wrnnKMqA2iCtFzE0KQTbCVuP/view?usp=sharing) |
| Task 3 — RAG Pipeline Colab | [Open Notebook](https://drive.google.com/file/d/1jv0lvRguqb2xm89tfQq1H_5gCYL40uSH/view?usp=sharing) |
| Task 4 — Fine-tuning Colab | [Open Notebook](https://drive.google.com/file/d/1mHuW9k8IGT0gDWN0OPrcqeuIaZzA-kwJ/view?usp=sharing) |
| Task 5 — Multi-Agent Platform Colab | [Open Notebook](https://drive.google.com/file/d/1y0rLpfhZTi2gQIfRaKlAd7cMDkQucXKg/view?usp=sharing) |
| W&B Training Dashboard | [View Run](https://wandb.ai/chandanisimran51-none/nova-brand-voice/runs/gsju51v5) |
| Fine-tuned Model (HF Hub) | [simran681/nova-brand-voice-tinyllama](https://huggingface.co/simran681/nova-brand-voice-tinyllama) |
| GitHub Repository | [simran681/nova-ai-platform](https://github.com/simran681/nova-ai-platform) |

---

## What Was Built

NOVA's AI Support & Personalization Platform — a complete multi-agent system that handles routine customer support autonomously, surfaces personalised product recommendations, and escalates complex cases to humans with full audit trails.

| Capability | Task | Runs On | Status |
|-----------|------|---------|--------|
| Prompt Engineering (COSTAR + CoT) | Task 1 | Colab (CPU) | ✅ Complete |
| MCP Backend Tools (5 tools + audit) | Task 2 | Local | ✅ Complete |
| RAG Pipeline (hybrid search + RAGAS) | Task 3 | Colab (CPU) | ✅ Complete |
| QLoRA Fine-tuning (TinyLlama brand voice) | Task 4 | Colab (T4 GPU) | ✅ Complete |
| LangGraph Multi-Agent Platform | Task 5 | Colab (T4 GPU) | ✅ Complete |

---

## Architecture

```
Customer Query
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 1: Prompt Engineering (COSTAR + CoT Intent Router)    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Task 5: LangGraph Multi-Agent Orchestrator                 │
│                                                             │
│  ┌──────────────┐   order/return   ┌──────────────────┐    │
│  │ TicketRouter │ ───────────────► │  SupportAgent    │    │
│  │              │   product/size   │  (MCP Task 2)    │    │
│  │ (Task 1 CoT) │ ───────────────► ├──────────────────┤    │
│  │              │   recommendation │  RAGAgent         │    │
│  │              │ ───────────────► │  (Task 3 RAG)    │    │
│  │              │   escalate       ├──────────────────┤    │
│  └──────────────┘ ───────────────► │  PersonalizAgent │    │
│                                    │  (MCP Task 2)    │    │
│                                    ├──────────────────┤    │
│                                    │  EscalationAgent │    │
│                                    │  (Human HITL)    │    │
│                                    └────────┬─────────┘    │
│                                             │              │
│                                    ┌────────▼─────────┐    │
│                                    │ BrandVoiceAgent  │    │
│                                    │ (Task 4 TinyLlama│    │
│                                    └────────┬─────────┘    │
│                                             │              │
│                                    ┌────────▼─────────┐    │
│                                    │   AuditLogger    │    │
│                                    │ nova_traces.json │    │
│                                    └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
nova-ai-platform/
├── README.md                          ← Setup instructions + all shareable links
├── requirements.txt                   ← All pinned dependencies
├── .env.example                       ← Environment variable template (no real keys)
├── nova_mock_db.json                  ← Synthetic order/customer/product data (200 products)
│
├── prompts/
│   ├── system_prompt_v1.txt           ← COSTAR framework system prompt
│   ├── intent_classifier_v1.txt       ← Chain-of-Thought intent classifier
│   └── escalation_detector_v1.txt     ← Frustration/escalation detector
│
├── task1_prompt_engineering.ipynb     ← Task 1: Prompt engineering (Colab)
├── task2_mcp/                         ← Task 2: MCP server + client + demo
│   ├── server.py
│   ├── client.py
│   └── demo.py
├── task3_rag_pipeline.ipynb           ← Task 3: RAG pipeline + RAGAS eval (Colab)
├── rag_module.py                      ← Importable NOVARAGPipeline (used by Task 5)
├── task4_finetune.ipynb               ← Task 4: QLoRA fine-tuning (Colab T4 GPU)
├── task5_nova_platform.py             ← Task 5: LangGraph multi-agent system
├── task5_nova_platform.ipynb          ← Task 5: Multi-agent platform (Colab T4 GPU)
├── task5_demo.py                      ← Task 5: 3-scenario demo script
│
├── nova_agent_graph.png               ← LangGraph agent workflow visualization
├── evaluation_report.json             ← RAGAS evaluation results (Task 3)
├── audit_log.jsonl                    ← MCP tool call audit log (Task 2)
└── nova_traces.json                   ← Agent session audit trails (Task 5)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/simran681/nova-ai-platform.git
cd nova-ai-platform
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required API keys:
- **GROQ_API_KEY** — Free at [console.groq.com](https://console.groq.com) (no credit card needed)
- **HF_TOKEN** — Free at [huggingface.co](https://huggingface.co/settings/tokens) (for Task 4)
- **WANDB_API_KEY** — Free at [wandb.ai](https://wandb.ai) (for Task 4 tracking)

---

## Running Each Task

### Task 1 — Prompt Engineering

**In Colab**: Open the shared notebook → add `GROQ_API_KEY` to Colab Secrets → Run All

**What it demonstrates**:
- COSTAR system prompt for NOVA support agent
- Chain-of-Thought intent classification (12 test cases)
- Escalation detection with frustration scoring (1-10)
- Prompt injection defense

---

### Task 2 — MCP Server

**Option A: Local client (no server needed)**
```bash
python3 task2_mcp/demo.py
```

**Option B: Full HTTP server**
```bash
# Terminal 1
uvicorn task2_mcp.server:app --reload --port 8001

# Terminal 2
python3 task2_mcp/demo.py --server
```

**What it demonstrates**:
- 5 backend tools: get_order_status, process_return, get_product_info, get_customer_history, recommend_products
- Compound scenario (4 tools in sequence for one customer enquiry)
- Full audit logging to `audit_log.jsonl`

---

### Task 3 — RAG Pipeline

**In Colab** (recommended — CPU only, no GPU needed):
1. Open the shared notebook
2. Add `GROQ_API_KEY` to Colab Secrets (🔑 icon) and enable notebook access
3. Run All

**What it demonstrates**:
- BM25 + ChromaDB hybrid search with RRF fusion
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)
- RAGAS evaluation on 20 Q&A pairs → `evaluation_report.json`

---

### Task 4 — Fine-Tuning (Requires GPU)

**In Colab** (T4 GPU required):
1. Runtime → Change runtime type → **T4 GPU**
2. Open the shared notebook
3. Add secrets: `GROQ_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`
4. Run All (~15-20 minutes)

**Architecture**:
- Base: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Method: QLoRA (rank=16, NF4 4-bit quantization)
- Dataset: 200 synthetic brand voice training pairs
- Trainer: TRL `SFTTrainer`
- Tracking: Weights & Biases

**Actual results**:
- Train loss: 1.1398 | Eval loss: 0.8827
- Brand voice score: **0.860 / 1.0** (empathy 0.95, warmth 0.98)
- Training time: ~15 minutes on T4
- Fine-tuned model: [simran681/nova-brand-voice-tinyllama](https://huggingface.co/simran681/nova-brand-voice-tinyllama)
- W&B run: [tinyllama-qlora-1774596482](https://wandb.ai/chandanisimran51-none/nova-brand-voice/runs/gsju51v5)

---

### Task 5 — Multi-Agent Platform

**Runs locally** (no Colab needed):
```bash
# Run all 3 demo scenarios
python3 task5_demo.py --all

# Run a specific scenario
python3 task5_demo.py --scenario 1
python3 task5_demo.py --scenario 2
python3 task5_demo.py --scenario 3
```

**Or use programmatically**:
```python
from task5_nova_platform import NOVAPlatform, NOVAPlatformConfig
import os
from dotenv import load_dotenv
load_dotenv()

config = NOVAPlatformConfig(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    mock_db_path="nova_mock_db.json"
)
platform = NOVAPlatform(config)

result = platform.process_ticket(
    "My order ORD-1042 hasn't arrived!",
    customer_id="CUST-1010"
)
print(result['final_response'])
```

**What it demonstrates**:
- LangGraph state machine with 7 nodes
- Conditional routing based on intent (Task 1 CoT)
- MCP tool calls for order/return/recommendations (Task 2)
- Hybrid RAG for product knowledge (Task 3)
- Brand voice polishing via fine-tuned model (Task 4)
- Human-in-the-loop escalation (HITL)
- Full audit trail per session → `nova_traces.json`

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Inference LLM | Groq (llama-3.1-8b-instant) | Free, fast, no GPU needed |
| Fine-tuning base | TinyLlama-1.1B-Chat | Fits Colab Free T4 confidently |
| Fine-tuning method | QLoRA (4-bit NF4) | ~80% memory reduction vs full fine-tune |
| Training framework | TRL SFTTrainer | HuggingFace native, Colab-tested |
| Experiment tracking | Weights & Biases | Required, free tier available |
| Embeddings | all-MiniLM-L6-v2 | Fast, free, high quality |
| Vector DB | ChromaDB | Local, no cloud account needed |
| Sparse search | rank_bm25 | Lightweight BM25 implementation |
| Re-ranker | ms-marco-MiniLM-L-6-v2 | Free cross-encoder from HF |
| RAG evaluation | RAGAS | Industry-standard RAG metrics |
| Multi-agent | LangGraph | Required, excellent HITL support |
| Synthetic data | Python Faker | Structured, reproducible |

---

## Evaluation Results

### Task 3 — RAG (RAGAS scores on 20 Q&A pairs)
Results saved in `evaluation_report.json`.

| Metric | Score |
|--------|-------|
| Faithfulness | 0.765 ✅ |
| Context Precision | 0.617 ⚠️ |
| Context Recall | 0.905 ✅ |
| Answer Correctness | 0.773 ✅ |

### Task 4 — Fine-tuning
Results logged to [W&B](https://wandb.ai/chandanisimran51-none/nova-brand-voice/runs/gsju51v5). Brand voice score on 20 held-out queries.

| Metric | Score |
|--------|-------|
| Train Loss (final) | 1.1398 |
| Eval Loss | 0.8827 |
| Brand Voice Score | **0.860 / 1.0** ✅ |
| Empathy Score | 0.950 ✅ |
| Warmth Score | 0.984 ✅ |
| Action Score | 0.425 ⚠️ |

---

## Business Impact Mapping

| Business Goal | AI Solution | Task |
|--------------|-------------|------|
| Reduce support costs by 40% | Automate 60% of tickets autonomously | Task 5 |
| Handle order/return queries at scale | MCP tools with real-time DB access | Task 2 |
| Answer product knowledge questions | RAG with ingredient + sizing guides | Task 3 |
| Increase repeat purchase 25% | Personalized recommendations | Tasks 2+5 |
| Consistent brand voice | Fine-tuned TinyLlama (QLoRA) | Task 4 |
| Legal compliance / audit trails | Full decision logging per session | Tasks 2+5 |
| Improve NPS for complex cases | Human escalation with full context | Task 5 |

---

## Notes

- Tasks 1, 3, 4 run on **Google Colab** (CPU or T4 GPU)
- Tasks 2 and 5 run **locally** using `uv run`
- LLM: **Groq** (`llama-3.1-8b-instant`) — free, no credit card required
- Synthetic data generated with Python Faker (no real customer data)
- Prompts versioned in `prompts/` directory (easily updatable)
- `rag_module.py` is importable and reused by both Task 3 notebook and Task 5
- Audit trail format is JSONL for easy streaming and compliance review

---

*Built by: Simran | NOVA AI Engineer Assessment | 2026*
