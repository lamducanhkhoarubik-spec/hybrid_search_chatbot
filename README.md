# 📄 PDF RAG Chatbot — Hybrid Search Edition

> **AIO2026 Project 1.2 Upgrade** · Built with `pypdf`, `chromadb`, `ollama`, and a from-scratch BM25 implementation.

A document Q&A chatbot powered by **Retrieval-Augmented Generation (RAG)**. Upload any PDF, ask questions, and get answers grounded in the document — no hallucination, no guessing.

This version replaces pure vector search with **Hybrid Search**: combining BM25 keyword matching (implemented from scratch) with semantic vector similarity for more accurate retrieval.

---

## ✨ What's New in This Version

| Feature | Original | This Version |
|---|---|---|
| Retrieval method | Vector-only (cosine similarity) | **Hybrid: BM25 + Vector** |
| BM25 | ❌ | ✅ From scratch, no external library |
| Score fusion | — | Reciprocal Rank Fusion (RRF) |
| UI | Streamlit | Streamlit (unchanged) |
| Deployment | Local / Colab | **Streamlit Cloud** (public link) |

---

## 🏗️ Architecture

```
PDF File
   │
   ▼
[pypdf] ──► Extract text
   │
   ▼
[chunk_text()] ──► Chunks (size=1000, overlap=200)
   │
   ├──────────────────────────┐
   ▼                          ▼
[BM25Index]             [ollama embed]
  (inverted index)        (bge-m3 vectors)
   │                          │
   ▼                          ▼
[BM25 scores]        [ChromaDB query]
   │                          │
   └──────────┬───────────────┘
              ▼
        [RRF Fusion]
              │
              ▼
      Top-k relevant chunks
              │
              ▼
    [Vicuna LLM via ollama]
              │
              ▼
           Answer
```

---

## 🔍 How Hybrid Search Works

### BM25 (from scratch)

BM25 (Best Match 25) is a keyword-based ranking function. Given a query, it scores each chunk based on term frequency, document frequency, and document length normalization.

The scoring formula for a query `q` and document `d`:

```
score(d, q) = Σ IDF(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

Where:
- `tf(t, d)` — how many times term `t` appears in document `d`
- `IDF(t)` — inverse document frequency (how rare the term is across all chunks)
- `|d|` — length of the document (in tokens)
- `avgdl` — average document length across the corpus
- `k1 = 1.5`, `b = 0.75` — tuning parameters

The implementation in `bm25.py` uses only the Python standard library. No `rank_bm25`, no `sklearn`.

### Vector Search

Each chunk is embedded with `bge-m3` (via Ollama) and stored in ChromaDB. At query time, the question is embedded and ChromaDB returns the top-k nearest chunks by cosine similarity.

### Reciprocal Rank Fusion (RRF)

Both methods return ranked lists. RRF merges them into a single ranking without needing to normalize scores across different scales:

```
RRF_score(chunk) = 1/(k + rank_bm25) + 1/(k + rank_vector)
```

Where `k = 60` (a smoothing constant). Chunks appearing high in both lists get the highest combined score.

---

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py                  # Streamlit app entry point
├── bm25.py                 # BM25 implementation from scratch
├── rag.py                  # Core RAG pipeline (chunk, embed, retrieve, generate)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # (local only) API keys / model config
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+ (managed via `uv`)
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create virtual environment with `uv`

```bash
# Install uv if you haven't
pip install uv

# Create and activate virtual environment
uv venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

`requirements.txt`:
```
pypdf
chromadb
ollama
streamlit
```

### 4. Pull Ollama models

```bash
# LLM for answer generation
ollama pull vicuna:7b-v1.5-q5_1

# Embedding model (multilingual, supports Vietnamese)
ollama pull bge-m3
```

> **Note:** First-time pull downloads ~4–5 GB per model. This only happens once; subsequent runs reuse the cached model.

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🌐 Deployment on Streamlit Cloud

The app is deployed publicly via [Streamlit Community Cloud](https://streamlit.io/cloud).

> **Live demo:** `<your-streamlit-link-here>`  
> *(Link will be shared separately after deployment.)*

### How to deploy your own instance

1. Push your code to a **public GitHub repository**.

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **"New app"** → select your repo, branch (`main`), and main file (`app.py`).

4. Under **Advanced settings → Secrets**, add:
   ```toml
   OLLAMA_BASE_URL = "https://your-ollama-endpoint"
   LLM_MODEL = "vicuna:7b-v1.5-q5_1"
   EMBED_MODEL = "bge-m3"
   ```

5. Click **Deploy**. Streamlit Cloud installs dependencies from `requirements.txt` automatically.

> **Important:** Streamlit Cloud does **not** run Ollama natively. You need an external Ollama endpoint (e.g., a VPS, Colab with ngrok/cloudflared, or a cloud VM). Point `OLLAMA_BASE_URL` to that endpoint.

### Running Ollama on Google Colab as a backend

```python
# Install and start Ollama
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time, os
os.environ['OLLAMA_FLASH_ATTENTION'] = 'false'
subprocess.Popen(["ollama", "serve"])
time.sleep(30)
!ollama pull vicuna:7b-v1.5-q5_1
!ollama pull bge-m3

# Expose via cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:11434
# Copy the https://....trycloudflare.com link → paste into Streamlit secrets as OLLAMA_BASE_URL
```

---

## 🧩 Key Implementation Notes

### BM25 (`bm25.py`)

- Tokenization: lowercase + whitespace split (no external tokenizer)
- IDF formula uses a log-smoothed variant to avoid negative scores on very common terms
- The index is built once per PDF upload and stored in `st.session_state`
- For Vietnamese text, BM25 still works reasonably on word-level tokens since Vietnamese is largely syllable-space-separated

### Chunking

- Default: `size=1000` characters, `overlap=200` characters
- Overlap prevents context loss at chunk boundaries (the last 200 chars of each chunk are repeated at the start of the next)
- Paragraphs are the natural split unit; extra-long paragraphs are force-split

### Prompt design

The system prompt includes the instruction *"Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa"* ("If the context has no information, say you don't know — don't make things up"). This is the single most important guardrail against hallucination in a RAG system.

### Temperature = 0

The LLM is called with `temperature=0` so answers are deterministic and factual. Higher temperature values are appropriate for creative writing but hurt Q&A accuracy.

### Session state

Streamlit reruns the entire script on every user interaction. All stateful objects — the BM25 index, the ChromaDB collection, and the chat history — are stored in `st.session_state` to survive reruns.

---

## 🚀 Suggested Further Improvements

- **Better Vietnamese tokenization** — integrate `underthesea` for word segmentation before BM25 indexing
- **Re-ranking** — add a cross-encoder model to re-score the top-k retrieved chunks before passing to the LLM
- **Persistent vector store** — replace `chromadb.Client()` with `chromadb.PersistentClient(path="./chroma_db")` so the index survives restarts
- **Multi-file support** — allow uploading multiple PDFs into the same collection
- **Streaming responses** — use `ollama.chat(..., stream=True)` with `st.write_stream()` for real-time token output

---

## 📚 References

- [RAG paper — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)
- [BM25 — Robertson & Zaragoza, 2009](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf)
- [Reciprocal Rank Fusion — Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [ChromaDB docs](https://docs.trychroma.com)
- [Ollama docs](https://ollama.com/docs)
- [BGE-M3 model](https://huggingface.co/BAAI/bge-m3)

---

