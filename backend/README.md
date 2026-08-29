# JanSaathi Backend & AI Engine 🇮🇳

The core intelligence and multi-agent backend powering **JanSaathi**. This service exposes asynchronous REST APIs, orchestrates the LangGraph state machine, hosts the fine-tuned `InLegalBERT` Small Language Model (SLM), and executes the 4-stage Hybrid RAG retrieval pipeline.

---

## ⚡ Core Architecture Highlights

1. **Local Fine-Tuned InLegalBERT (10-Class Intent SLM):**
   - Base model: `law-ai/InLegalBERT`
   - High-precision intent classification across 10 specialized categories: `Cheque_Bounce`, `Civic_Scheme_Info`, `Consumer_Dispute`, `Criminal_FIR`, `Cybercrime`, `Legal_Notice_Contract`, `RERA_RealEstate`, `RTI`, `Tenant_Landlord`, and `Workplace_Labour`.
   - Evaluated on a strictly held-out test split with **78.81% Test Accuracy** and **78.75% Macro F1** (92.3% on Cheque Bounce, 83.0% on Civic Schemes).
   - Graceful fallback to Groq LLM when deployed in lightweight cloud environments.

2. **Deterministic Legal Knowledge Graph (`knowledge/legal_graph.py`):**
   - 4-layer directed NetworkX ontology encoding Acts, Sections, Statutory Authorities/Forums, and Remedies.
   - Automatically traverses legal hierarchies and injects hard truth constraints into LLM prompts.

3. **Zero-LLM Pecuniary Jurisdiction Engine (`knowledge/jurisdiction_engine.py`):**
   - Pure deterministic algorithmic calculator that maps claim amounts, case types, and state jurisdictions to exact statutory courts (DCDRC / SCDRC / NCDRC, RERA authorities, PIO portals), computing filing fees and limitation periods with zero hallucination.

4. **4-Stage Hybrid RAG Pipeline (`rag/pipeline.py`):**
   - **Query Expansion:** 3 parallel legal rephrasings.
   - **Dual Search:** Dense vector search (`BAAI/bge-small-en-v1.5` over ChromaDB) + Sparse exact keyword search (`rank_bm25`).
   - **Fusion & Rerank:** Reciprocal Rank Fusion (RRF) followed by `cross-encoder/ms-marco-MiniLM-L-12-v2` scoring.

5. **Reflexion Self-Correction Loop (`agents/verifier.py`):**
   - Adversarial critic evaluating 8 legal compliance parameters. Automatically triggers re-drafting if the output score is below 7.5/10.

6. **Contract & Document Analyzer (`agents/analyzer.py`):**
   - PyMuPDF text parsing with OCR.Space API fallback for scanned tenancy agreements, employment bonds, and builder contracts.

---

## 🛠️ Tech Stack

- **Framework:** FastAPI, Uvicorn (ASGI)
- **Agent Orchestration:** LangGraph, LangChain Core
- **ML & NLP:** PyTorch, Hugging Face `transformers`, `scikit-learn`, `law-ai/InLegalBERT`
- **Embeddings & Vector Store:** ChromaDB, `BAAI/bge-small-en-v1.5`, `rank_bm25`, `ms-marco-MiniLM-L-12-v2`
- **Database & Auth:** SQLAlchemy (Async), SQLite / PostgreSQL, JWT, Passlib (bcrypt)
- **Document Processing:** PyMuPDF, PyPDF2, OCR.Space API, Jinja2

---

## 🚀 Running the Backend

### 1. Virtual Environment Setup
```powershell
# Inside /backend directory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:
```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_super_secret_jwt_key
```

### 3. Start Development Server
```powershell
uvicorn main:app --reload --port 8000
```
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.
