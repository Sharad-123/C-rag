# CRAG — Corrective RAG, live

A visual Corrective RAG (CRAG) app. Upload PDFs, ask a question, and watch the
CRAG pipeline run step by step on an animated architecture diagram.

## How it works

1. **Retrieve** – top-4 chunks from a FAISS index built from your PDFs.
2. **Evaluate** – an LLM scores each chunk's relevance (0.0–1.0).
3. **Verdict**
   - `CORRECT` – at least one chunk scores > 0.7 → use internal documents.
   - `INCORRECT` – all chunks score < 0.3 → discard and use web search.
   - `AMBIGUOUS` – otherwise → combine internal documents and web search.
4. **Rewrite query + Web search** (Tavily) when the verdict is not `CORRECT`.
5. **Refine** – split context into sentences and keep only the relevant ones.
6. **Generate** – answer using only the refined context.

Stage events are streamed to the browser over Server-Sent Events (SSE).

## Tech stack

FastAPI · LangGraph · LangChain · FAISS · Google Gemini embeddings ·
Groq (Llama 3.3 70B) · Tavily search · vanilla HTML/CSS/JS frontend

## Setup

```bash
git clone https://github.com/<your-username>/CRAG.git
cd CRAG

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then add your API keys
```

### API keys needed

| Variable         | Used for                  | Get it from                         |
| ---------------- | ------------------------- | ----------------------------------- |
| `GOOGLE_API_KEY` | Embeddings                | https://aistudio.google.com/apikey  |
| `GROQ_API_KEY`   | LLM                       | https://console.groq.com/keys       |
| `TAVILY_API_KEY` | Web search fallback       | https://app.tavily.com              |

## Run

```bash
uvicorn server:app --reload --port 8000
```

Open http://localhost:8000, upload one or more PDFs, and ask a question.

## Project structure

```
CRAG/
├── server.py          # FastAPI backend + LangGraph CRAG pipeline
├── static/
│   └── index.html     # Frontend with live architecture diagram
├── requirements.txt
├── .env.example
└── LICENSE
```

## Notes

- Sessions and FAISS indexes are stored in memory and are lost on restart.
- Thresholds (`UPPER_TH = 0.7`, `LOWER_TH = 0.3`) can be changed at the top of `server.py`.
- Based on the paper [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884).

## License

MIT
