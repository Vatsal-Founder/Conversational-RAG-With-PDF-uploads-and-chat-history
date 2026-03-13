# Conversational RAG — PDF Chat with Voice

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![Whisper](https://img.shields.io/badge/Voice-OpenAI%20Whisper-412991?logo=openai)](https://openai.com/research/whisper)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)](https://langchain.com)

Multi-document **conversational RAG** with **voice input/output**, session memory, and per-query performance tracking. Upload PDFs, ask questions by text or voice, and get answers grounded in your documents.

> **Key features:** Whisper speech-to-text input, gTTS voice responses, multi-turn memory, per-query latency & retrieval tracking.

---

## Features

- 📤 **Multi-PDF upload**: drag & drop any number of PDFs, auto-chunked and indexed
- 🧠 **Conversational memory**: follow-up questions use full chat history
- 🎙️ **Voice input**: ask questions by speaking (OpenAI Whisper STT)
- 🔊 **Voice output**: hear answers read aloud (gTTS text-to-speech)
- ⚡ **Free demo tier**: 10 questions/session, no API key needed for visitors
- 📊 **Per-query eval tracking**: latency, chunks retrieved, and response length logged in sidebar

---

## Architecture

```
                          ┌──────────────────────────┐
                          │     Streamlit UI          │
                          │  Text Input / Voice Input │
                          └────────┬─────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            [Text Question]              [Audio Recording]
                    │                             │
                    │                    [Whisper STT API]
                    │                             │
                    └──────────┬──────────────────┘
                               ▼
                    [History-Aware Retriever]
                    (reformulate with chat history)
                               │
                               ▼
                    [FAISS Vector Store]
                    (retrieve top-k chunks)
                               │
                               ▼
                    [Groq LLM — Llama 3.3 70B]
                    (generate grounded answer)
                               │
                    ┌──────────┴──────────────┐
                    ▼                         ▼
             [Text Response]          [gTTS Voice Response]
                    │                         │
                    └──────────┬──────────────┘
                               ▼
                    [📊 Log: latency, chunks, length]
```

---

## Per-Query Performance Tracking

Every question is tracked with real-time metrics visible in the sidebar:

| Metric | What It Shows |
|--------|--------------|
| ⏱️ **Latency** | End-to-end response time in milliseconds |
| 📄 **Chunks retrieved** | Number of document chunks used for context |
| 📏 **Response length** | Character count of the generated answer |

This helps identify slow queries, insufficient retrieval, or overly verbose responses — key signals for tuning your RAG pipeline.

---

## Project Structure

```
.
├── app.py                # Main Streamlit app (chat + voice + tracking)
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick Start

### 1) Install

```bash
git clone https://github.com/Vatsal-Founder/Conversational-RAG-With-PDF-uploads-and-chat-history.git
cd Conversational-RAG-With-PDF-uploads-and-chat-history
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure

```bash
cp .env.example .env
# Fill in your API keys
```

| Key | Required For | Free Tier |
|-----|-------------|-----------|
| `GROQ_API_KEY` | LLM generation | Yes |
| `OPENAI_API_KEY` | Voice input (Whisper) | Paid (voice is optional) |

Voice features are **optional** — the app works fully with text-only if no OpenAI key is provided.

### 3) Run

```bash
streamlit run app.py
```

1. Upload PDFs in the sidebar → click **Index Documents**
2. Ask questions via the chat input or the voice recorder
3. Toggle **voice responses** in the sidebar to hear answers
4. Check per-query metrics in the sidebar under **Query Performance**

---

## Voice Features

### Voice Input (Speech-to-Text)
Uses **OpenAI Whisper API** via Streamlit's native `st.audio_input` widget. Click the microphone, speak your question, and it gets transcribed and sent to the RAG pipeline. Supports multiple languages.

### Voice Output (Text-to-Speech)
Uses **gTTS** (Google Text-to-Speech) — free, no API key required. Toggle it on in the sidebar. Responses auto-play in the browser after generation.

### Graceful Degradation
If `OPENAI_API_KEY` is not set, voice input is simply hidden and the app works as a text-only chat. No errors, no broken UI.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS (in-memory) |
| Voice Input | OpenAI Whisper API |
| Voice Output | gTTS |
| Orchestration | LangChain |
| UI | Streamlit |

---

## Configuration Tips

- **Chunk size**: 1000 tokens with 200 overlap works well for general documents
- **top_k**: default is 4 chunks; increase to 6–8 for longer documents
- **Voice**: Whisper works best with clear audio and minimal background noise
- **Memory**: in-memory session history resets on app restart; suitable for demos

---

## License

GPL-3.0 © Vatsal Kansara
