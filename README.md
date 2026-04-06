# RAG Assistant — Personal Learning Project

A conversational AI assistant that answers questions about your PDF documents using Retrieval-Augmented Generation (RAG). Built as a personal project to apply concepts from the Google ML Engineer certification course.

## Overview

Upload one or multiple PDFs and ask questions about them. The agent decides whether to search your documents or the web to find the best answer.

## Architecture

User → Streamlit UI → LangChain Agent
├── Tool 1: PDF Retriever (ChromaDB)
└── Tool 2: Web Search (DuckDuckGo)
↓
OpenRouter LLM

**Key design decisions:**
- **Sliding window memory** — only the last 10 conversation turns are sent to the LLM to avoid token overflow
- **Chunking strategy** — documents split into 512-token chunks with 50-token overlap for optimal retrieval precision
- **Agent over chain** — uses a tool-calling agent so the model decides when to use the PDF retriever vs web search

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Interface | Streamlit | |
| Orchestration | LangChain | |
| LLM | OpenRouter (stepfun/step-3.5-flash) | |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Runs locally, no API key needed |
| Vector Store | ChromaDB | Local |
| Web Search | DuckDuckGo | No API key needed |
| Containerization | Docker | |

## Model Decisions

Started with **Gemini Pro + Google textembedding-gecko** via Google AI Studio. Migrated to:
- **OpenRouter** for the LLM — more model options and no rate limits on free tier
- **HuggingFace embeddings (all-MiniLM-L6-v2)** — runs locally, zero rate limits, no API key required

## Getting Started

### Prerequisites
- Python 3.11
- Docker (optional)
- OpenRouter API key — get one free at openrouter.ai

### Run locally
```bash or VS terminal
# Clone the repo
git clone https://github.com/yourusername/rag-assistant.git
cd rag-assistant

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo OPENAI_API_KEY=your_key_here > .env 
Or if you are into VS you can just create the .env file manually and then enter your API keys.

# Run
streamlit run app.py
```

### Run with Docker
```bash or VS terminal
docker build -t rag-assistant .
docker run -p 8080:8080 --env-file .env rag-assistant
```

Then open `http://localhost:8080`

## Configuration

You can adjust these parameters in `app.py`:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 512 | Size of each document chunk in tokens |
| `chunk_overlap` | 50 | Overlap between chunks to preserve context |
| `k` | 4 | Number of chunks retrieved per query |
| `window` | 20 | Messages kept in memory (10 turns) |
| `max_iterations` | 5 | Max agent steps before stopping |
| `temperature` | 0.2 | LLM creativity — lower is more deterministic |

## Technical Challenges

- **Chunking strategy** — initial chunk size of 1000 tokens produced generic answers. Reduced to 512 with 50-token overlap for better retrieval precision
- **Token overflow** — storing full conversation history caused context limit errors. Solved with a 10-turn sliding window
- **Hallucination** — LLM sometimes answered from training data instead of documents. Fixed with explicit system prompt instructions and low temperature
- **Library breaking changes** — LangChain 1.0 moved several modules to `langchain-classic`. Updated imports accordingly
- **Streaming issues** — free tier models on OpenRouter don't reliably support streaming. Disabled streaming to stabilize responses

## Deployment

Containerized with Docker and ready to deploy on **Google Cloud Run**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rag-assistant
gcloud run deploy rag-assistant \
  --image gcr.io/YOUR_PROJECT_ID/rag-assistant \
  --platform managed \
  --region us-central1
```

## What I learned

- How RAG pipelines work end to end — from document ingestion to retrieval to generation
- How to build and orchestrate LangChain agents with multiple tools
- How to manage token limits with conversation windowing
- How to containerize and deploy AI applications with Docker and Cloud Run
- Practical debugging of LLM systems — hallucination, retrieval quality, prompt engineering
