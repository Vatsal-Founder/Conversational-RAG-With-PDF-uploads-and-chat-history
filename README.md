# Conversational RAG — PDF Uploads & Chat History

End‑to‑end **Q\&A RAG** application that remembers prior turns and answers with context from your uploaded **PDFs**.

* **Groq** for fast, low‑latency generation (e.g., Llama‑3 family)
* **OpenAI embeddings** for retrieval quality
* **Chroma** as the vector store (persistent)
* **LangChain** for orchestration + **LangSmith** for tracing
* **Conversation memory** so follow‑ups “just work”

> Repo: `Vatsal-Founder/Conversational-RAG-With-PDF-uploads-and-chat-history`

Link: 
---

## Features

* 📤 **Multi‑PDF upload**: drag & drop any number of PDFs; incremental indexing.
* 🧠 **Conversational memory**: replies consider previous messages within the same session.
* ⚡ **Groq LLM**: low‑latency chat models via `GROQ_API_KEY`.
* 🔎 **OpenAI embeddings**: strong recall with `text-embedding-3-*`.
* 🗂️ **Chroma**: local persistent store you can mount/backup.
* 🧪 **LangSmith**: opt‑in traces for runs, latencies, and chains.

---

## Architecture

```
[Browser UI]
   ↓ (WebSocket/HTTP)
[App (Streamlit or FastAPI)]
   ├─ Session Manager (per user/session_id)
   ├─ Memory (ChatHistory / Summary)
   ├─ Ingestion (PDF → chunks → embeddings → Chroma)
   ├─ Retrieval (k docs from Chroma)
   └─ Generator (Groq LLM)
            ↓
     [Answer + cited sources]
```

---

## Quick Start

### 1) Setup

```bash
git clone https://github.com/Vatsal-Founder/Conversational-RAG-With-PDF-uploads-and-chat-history.git
cd Conversational-RAG-With-PDF-uploads-and-chat-history
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # create if missing
```

### 2) Environment variables (minimum)

```ini
# LLM (Groq)
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b  # example

# Embeddings (OpenAI)
OPENAI_API_KEY=your_openai_key
EMBEDDINGS_MODEL=text-embedding-3-small


# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=conv-rag
```

> Defaults favor local development: in‑memory chat history and a local Chroma folder.

### 3) Run

Choose the entrypoint that matches your `app.py`:

**Streamlit UI**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Upload PDFs, start a chat, and ask follow‑ups.


---

## Using it

1. **Upload PDFs** (one at a time). The app chunks the files and updates the Chroma index.
2. **Ask a question.** The retriever pulls relevant chunks from the combined corpus.
3. **Follow up naturally.** Memory keeps prior turns in context for the same session.
4. **Inspect traces** in LangSmith (if enabled).

> Tip: Give your chat a **session name/ID** so you can come back later. Memory TTL controls how long we keep past turns.

---

## Pre‑loading documents (optional)


Re‑run ingestion whenever PDFs change.

---

## Deployment

### Docker (recommended)

**Dockerfile (example)**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Streamlit by default
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
# For FastAPI instead:
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & run**

```bash
docker build -t conv-rag .
docker run -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/.chroma:/app/.chroma \
  -v $(pwd)/uploads:/app/uploads \
  conv-rag
```

---

## Configuration Tips

* **Embeddings**: start with `text-embedding-3-small` for speed; upgrade to `text-embedding-3-large` for best recall.
* **Chunking**: \~1000 tokens with 150 overlap is a good default; adjust for short PDFs.
* **Memory**: `inmemory` is simplest; use `redis` for multi‑instance deployments.
* **Sources**: surface the top‑k chunks and metadata in your UI for transparency.

---

