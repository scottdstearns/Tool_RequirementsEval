# apps/req-eval/app/utils/retriever.py
import os
from typing import List, Tuple
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://litellm:4000/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-1234")
EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "azure-embedding-large")
QDRANT_URL      = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
KB_COLLECTION   = os.getenv("KB_COLLECTION", "open-webui_knowledge")
KB_NAME         = os.getenv("KB_NAME", "").strip().lower()  # friendly OWUI KB name

def _embed(texts: List[str]) -> List[List[float]]:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _payload_text(payload: dict) -> str:
    return (
        (payload or {}).get("text")
        or payload.get("page_content")
        or payload.get("content")
        or payload.get("chunk")
        or ""
    )

def _contains_kb_name(payload: dict) -> bool:
    if not KB_NAME:
        return True
    for v in (payload or {}).values():
        if isinstance(v, str) and KB_NAME in v.lower():
            return True
        if isinstance(v, list) and any(isinstance(x, str) and KB_NAME in x.lower() for x in v):
            return True
    return False

def retrieve_topk(query: str, top_k: int = 4) -> List[Tuple[str, float]]:
    try:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
        qdrant.get_collection(collection_name=KB_COLLECTION)  # existence check
    except Exception as e:
        print(f"[KB] Collection '{KB_COLLECTION}' not available: {e}. Continuing without KB.")
        return []

    try:
        vec = _embed([query])[0]
    except Exception as e:
        print(f"[KB] Embedding failed: {e}. Continuing without KB.")
        return []

    try:
        # Pull a few extra, then filter to the specific KB by friendly name if provided
        result = qdrant.search(
            collection_name=KB_COLLECTION,
            query_vector=vec,
            limit=max(20, top_k),
            with_payload=True,
        )
    except ResponseHandlingException as e:
        print(f"[KB] Qdrant search error: {e}. Continuing without KB.")
        return []

    chunks: List[Tuple[str, float]] = []
    for p in result:
        if not _contains_kb_name(p.payload or {}):
            continue
        text = _payload_text(p.payload or {})
        if text:
            chunks.append((text, p.score))
    return chunks[:top_k]

def join_context(chunks: List[Tuple[str, float]]) -> str:
    if not chunks:
        return ""
    return "\n\n".join(f"- {t}" for t, _ in chunks)

