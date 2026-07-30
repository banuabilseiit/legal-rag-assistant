"""
RAG-пайплайн: поиск релевантных чанков в ChromaDB + генерация ответа
через Hugging Face Inference API в двух режимах (официальный / простой).
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "legal_docs"
CHUNKS_JSONL_PATH = "data/processed/chunks.jsonl"

SYSTEM_OFFICIAL = """Ты — официальный представитель государственного органа.
Отвечай формально, ссылаясь ТОЛЬКО на предоставленные фрагменты законов и НПА.
Обязательно указывай источник (закон/НПА, статья или пункт) для каждого утверждения.

ВАЖНО: сформируй ответ ОДНИМ связным текстом (2-5 абзацев), а не списком цитат
или перечислением фрагментов. Объедини информацию из всех релевантных источников
в цельное объяснение, как будто отвечаешь на официальный запрос гражданина.
Если в контексте нет ответа на вопрос — прямо скажи, что информация отсутствует,
не выдумывай нормы."""

SYSTEM_SIMPLE = """Объясни ответ простым, разговорным языком, как для человека без
юридического образования.

ВАЖНО: сформируй ответ ОДНИМ связным текстом (короткий, по сути), без списков,
без ссылок на статьи и номера пунктов, без цитирования формулировок закона.
Просто объясни суть и практический вывод, как будто говоришь с другом."""


def load_embed_model() -> SentenceTransformer:
    """Загружает модель эмбеддингов (без токена — модель публичная)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_collection():
    """Открывает существующую персистентную векторную базу ChromaDB (для локальной работы)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def build_index_from_jsonl(
    embed_model: SentenceTransformer,
    jsonl_path: str = CHUNKS_JSONL_PATH,
):
    """
    Строит векторный индекс В ПАМЯТИ (EphemeralClient) из файла chunks.jsonl.

    Используется для деплоя на Streamlit Community Cloud: там нет смысла
    хранить готовую бинарную базу ChromaDB в git-репозитории — вместо этого
    храним лёгкий текстовый chunks.jsonl, а индекс пересчитывается при
    старте приложения (один раз, благодаря st.cache_resource в app.py).
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {jsonl_path} не найден. Убедитесь, что он лежит в репозитории "
            "по пути data/processed/chunks.jsonl"
        )

    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    client = chromadb.EphemeralClient()  # in-memory, не пишет на диск
    collection = client.get_or_create_collection(COLLECTION_NAME)

    texts_for_embedding = [f"passage: {c['text']}" for c in chunks]
    embeddings = embed_model.encode(
        texts_for_embedding,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    ids = [f"{c['source']}__{c.get('doc_part') or 'single'}__{i}" for i, c in enumerate(chunks)]
    metadatas = [
        {
            "source": c["source"],
            "doc_type": c["doc_type"],
            "article": c.get("article") or "",
            "doc_part": c.get("doc_part") or "",
        }
        for c in chunks
    ]
    documents = [c["text"] for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def retrieve(query: str, embed_model: SentenceTransformer, collection, top_k: int = 5) -> list[dict]:
    """Ищет top_k наиболее релевантных чанков в векторной базе."""
    query_embedding = embed_model.encode([f"query: {query}"], normalize_embeddings=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", ""),
            "doc_type": results["metadatas"][0][i].get("doc_type", ""),
            "article": results["metadatas"][0][i].get("article", ""),
            "doc_part": results["metadatas"][0][i].get("doc_part", ""),
            "distance": results["distances"][0][i],
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Собирает найденные чанки в единый текстовый блок с указанием источника."""
    parts = []
    for c in chunks:
        part_label = f", {c['doc_part']}" if c["doc_part"] else ""
        label = f"[Источник: {c['source']}{part_label}, {c['article']}]"
        parts.append(f"{label}\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    context: str,
    mode: str,
    hf_token: str,
    model_name: str = LLM_MODEL_NAME,
) -> str:
    """Отправляет запрос в LLM через Hugging Face Inference API."""
    llm_client = InferenceClient(model=model_name, token=hf_token)
    system_prompt = SYSTEM_OFFICIAL if mode == "official" else SYSTEM_SIMPLE

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"},
    ]

    response = llm_client.chat_completion(
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
    )
    return response.choices[0].message.content


def answer(
    query: str,
    embed_model: SentenceTransformer,
    collection,
    hf_token: str,
    mode: str = "official",
    top_k: int = 5,
    model_name: str = LLM_MODEL_NAME,
) -> dict:
    """Полный цикл: retrieval + генерация ответа."""
    chunks = retrieve(query, embed_model, collection, top_k=top_k)
    context = build_context(chunks)
    answer_text = generate_answer(query, context, mode, hf_token, model_name=model_name)

    return {
        "answer": answer_text,
        "sources": chunks,
    }
