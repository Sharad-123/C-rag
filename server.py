"""
CRAG Visual RAG App — backend
-----------------------------
FastAPI server that:
  1. Accepts PDF uploads and builds a FAISS index per session
  2. Runs the CRAG LangGraph pipeline for a question
  3. Streams live stage events (SSE) so the frontend can animate
     the CRAG architecture diagram in real time.

Run:
  uvicorn server:app --reload --port 8000
Then open http://localhost:8000
"""

import json
import os
import queue
import re
import shutil
import tempfile
import threading
import uuid
from typing import List, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch

load_dotenv()

# =====================================================
# Globals
# =====================================================
UPPER_TH = 0.7
LOWER_TH = 0.3

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tavily = TavilySearch(max_results=5)

# session_id -> {"retriever": ..., "queue": Queue, "files": [names]}
SESSIONS: dict = {}


def emit(session_id: str, event: dict):
    """Push a stage event to this session's SSE queue."""
    s = SESSIONS.get(session_id)
    if s and s.get("queue") is not None:
        s["queue"].put(event)


# =====================================================
# State
# =====================================================
class State(TypedDict):
    session_id: str
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    web_docs: List[Document]

    answer: str


# =====================================================
# Nodes (same CRAG logic as the notebook, + emit calls)
# =====================================================
def retrieve_node(state: State) -> State:
    sid = state["session_id"]
    emit(sid, {"stage": "retrieve", "status": "start"})

    retriever = SESSIONS[sid]["retriever"]
    docs = retriever.invoke(state["question"])

    emit(sid, {
        "stage": "retrieve", "status": "done",
        "n_docs": len(docs),
        "previews": [d.page_content[:120] for d in docs],
    })
    return {"docs": docs}


class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            'Output JSON only, with EXACTLY these two keys: "score" (float) and "reason" (string). '
            "Do not use any other key names.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)
doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore, method="json_mode")


def eval_each_doc_node(state: State) -> State:
    sid = state["session_id"]
    emit(sid, {"stage": "evaluate", "status": "start", "n_docs": len(state["docs"])})

    q = state["question"]
    scores: List[float] = []
    good: List[Document] = []

    for i, d in enumerate(state["docs"]):
        out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
        scores.append(out.score)
        if out.score > LOWER_TH:
            good.append(d)
        emit(sid, {
            "stage": "evaluate", "status": "doc_score",
            "index": i, "score": out.score, "reason": out.reason,
        })

    if any(s > UPPER_TH for s in scores):
        verdict, reason, good_docs = "CORRECT", f"At least one retrieved chunk scored > {UPPER_TH}.", good
    elif len(scores) > 0 and all(s < LOWER_TH for s in scores):
        verdict, reason, good_docs = "INCORRECT", f"All retrieved chunks scored < {LOWER_TH}.", []
    else:
        verdict, reason, good_docs = "AMBIGUOUS", f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.", good

    emit(sid, {"stage": "evaluate", "status": "done", "verdict": verdict, "reason": reason, "scores": scores})
    return {"good_docs": good_docs, "verdict": verdict, "reason": reason}


def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            'Use ONLY the sentence. Output JSON only, with EXACTLY one key: "keep" (boolean true/false). '
            "Do not use any other key name.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)
filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop, method="json_mode")


def refine(state: State) -> State:
    sid = state["session_id"]
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
        source = "internal"
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]
        source = "web"
    else:  # AMBIGUOUS
        docs_to_use = state["good_docs"] + state["web_docs"]
        source = "internal+web"

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentences(context)

    emit(sid, {"stage": "refine", "status": "start", "n_strips": len(strips), "source": source})

    kept: List[str] = []
    for i, s in enumerate(strips):
        keep = filter_chain.invoke({"question": q, "sentence": s}).keep
        if keep:
            kept.append(s)
        emit(sid, {
            "stage": "refine", "status": "strip",
            "index": i, "total": len(strips), "kept": len(kept), "keep": keep,
        })

    refined_context = "\n".join(kept).strip()
    emit(sid, {"stage": "refine", "status": "done", "n_strips": len(strips), "n_kept": len(kept)})

    return {"strips": strips, "kept_strips": kept, "refined_context": refined_context}


class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            '- Return JSON with EXACTLY one key: "query" (string). Do not use any other key name.',
        ),
        ("human", "Question: {question}"),
    ]
)
rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery, method="json_mode")


def rewrite_query_node(state: State) -> State:
    sid = state["session_id"]
    emit(sid, {"stage": "rewrite", "status": "start"})
    out = rewrite_chain.invoke({"question": state["question"]})
    emit(sid, {"stage": "rewrite", "status": "done", "web_query": out.query})
    return {"web_query": out.query}


def web_search_node(state: State) -> State:
    sid = state["session_id"]
    q = state.get("web_query") or state["question"]
    emit(sid, {"stage": "web_search", "status": "start", "query": q})

    results = tavily.invoke({"query": q})

    web_docs: List[Document] = []
    titles = []
    for r in results.get("results", []):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))
        titles.append(title)

    emit(sid, {"stage": "web_search", "status": "done", "n_results": len(web_docs), "titles": titles})
    return {"web_docs": web_docs}


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, say: 'I don't know.'",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)


def generate(state: State) -> State:
    sid = state["session_id"]
    emit(sid, {"stage": "generate", "status": "start", "verdict": state.get("verdict", "")})
    out = (answer_prompt | llm).invoke({"question": state["question"], "context": state["refined_context"]})
    emit(sid, {"stage": "generate", "status": "done"})
    return {"answer": out.content}


def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    else:
        return "rewrite_query"


# =====================================================
# Build graph
# =====================================================
g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search", web_search_node)
g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")
g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "rewrite_query": "rewrite_query"},
)
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search", "refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)

crag = g.compile()

# =====================================================
# FastAPI app
# =====================================================
app = FastAPI(title="CRAG Visual RAG")


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Accept one or more PDFs, build a FAISS index, return a session id."""
    session_id = str(uuid.uuid4())[:8]

    all_docs = []
    names = []
    tmpdir = tempfile.mkdtemp(prefix="crag_")
    try:
        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                continue
            path = os.path.join(tmpdir, f.filename)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            all_docs.extend(PyPDFLoader(path).load())
            names.append(f.filename)

        if not all_docs:
            return {"error": "No readable PDF files were uploaded."}

        chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(all_docs)
        for d in chunks:
            d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        SESSIONS[session_id] = {"retriever": retriever, "queue": None, "files": names}
        return {"session_id": session_id, "files": names, "n_chunks": len(chunks)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/ask")
def ask(session_id: str, question: str):
    """Run the CRAG graph and stream stage events as SSE."""
    if session_id not in SESSIONS:
        def err():
            yield f"data: {json.dumps({'stage': 'error', 'message': 'Unknown session. Upload a document first.'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    q: queue.Queue = queue.Queue()
    SESSIONS[session_id]["queue"] = q

    def run_graph():
        try:
            res = crag.invoke({
                "session_id": session_id,
                "question": question,
                "docs": [], "good_docs": [],
                "verdict": "", "reason": "",
                "strips": [], "kept_strips": [], "refined_context": "",
                "web_query": "", "web_docs": [],
                "answer": "",
            })
            q.put({
                "stage": "complete",
                "answer": res["answer"],
                "verdict": res["verdict"],
                "reason": res["reason"],
                "web_query": res.get("web_query", ""),
                "n_kept": len(res.get("kept_strips", [])),
                "n_strips": len(res.get("strips", [])),
            })
        except Exception as e:
            q.put({"stage": "error", "message": str(e)})
        finally:
            q.put(None)  # sentinel: stream ends

    threading.Thread(target=run_graph, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
