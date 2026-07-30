"""
RAG-пайплайн: поиск релевантных чанков в ChromaDB и генерация ответа
через Hugging Face Inference Providers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Пути и настройки моделей
# ---------------------------------------------------------------------------

# rag_pipeline.py находится в папке src/, поэтому parent.parent — корень проекта.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "legal_docs"
CHUNKS_JSONL_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


# ---------------------------------------------------------------------------
# Системные инструкции для языковой модели
# ---------------------------------------------------------------------------

SYSTEM_OFFICIAL = """Ты — официальный представитель государственного органа.
Отвечай формально и основывайся ТОЛЬКО на предоставленных фрагментах законов,
НПА и внутренних документов.

Для каждого существенного утверждения указывай источник, статью или пункт,
если они присутствуют в контексте.

Сформируй один связный ответ из 2–5 абзацев. Не превращай ответ в набор
разрозненных цитат. Объедини найденную информацию в цельное объяснение.

Если предоставленного контекста недостаточно, прямо сообщи, что в найденных
документах нет достаточной информации. Не придумывай нормы и факты."""

SYSTEM_SIMPLE = """Объясни ответ простым и понятным языком для человека без
юридического образования.

Основывайся ТОЛЬКО на предоставленном контексте. Дай короткий связный ответ
без сложных юридических формулировок, без длинного списка цитат и без
выдуманных норм.

Если информации недостаточно, прямо скажи об этом."""


# ---------------------------------------------------------------------------
# Загрузка embedding-модели
# ---------------------------------------------------------------------------

def load_embed_model() -> SentenceTransformer:
    """Загружает публичную multilingual-e5-base модель для embeddings."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Работа с ChromaDB
# ---------------------------------------------------------------------------

def load_collection():
    """Открывает локальную постоянную коллекцию ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _read_chunks(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Читает и проверяет чанки из JSONL-файла."""
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Файл с чанками не найден: {path}. "
            "Убедитесь, что в репозитории существует "
            "data/processed/chunks.jsonl."
        )

    chunks: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Некорректный JSON в {path}, строка {line_number}: {exc}"
                ) from exc

            text = str(chunk.get("text", "")).strip()
            source = str(chunk.get("source", "")).strip()
            doc_type = str(chunk.get("doc_type", "")).strip()

            if not text:
                continue

            if not source:
                source = "Неизвестный источник"

            if not doc_type:
                doc_type = "unknown"

            chunks.append(
                {
                    "text": text,
                    "source": source,
                    "doc_type": doc_type,
                    "article": str(chunk.get("article") or ""),
                    "doc_part": str(chunk.get("doc_part") or ""),
                }
            )

    if not chunks:
        raise ValueError(
            f"Файл {path} существует, но не содержит пригодных для индексации чанков."
        )

    return chunks


def build_index_from_jsonl(
    embed_model: SentenceTransformer,
    jsonl_path: str | Path = CHUNKS_JSONL_PATH,
):
    """
    Создаёт временную коллекцию ChromaDB в памяти из chunks.jsonl.

    Этот вариант подходит для Streamlit Community Cloud: бинарную папку
    chroma_db не нужно хранить в GitHub, потому что индекс создаётся при старте.
    """
    chunks = _read_chunks(jsonl_path)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Для E5 документы должны иметь префикс "passage:".
    texts_for_embedding = [f"passage: {chunk['text']}" for chunk in chunks]

    embeddings = embed_model.encode(
        texts_for_embedding,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    ids = [f"chunk_{index}" for index in range(len(chunks))]

    metadatas = [
        {
            "source": chunk["source"],
            "doc_type": chunk["doc_type"],
            "article": chunk["article"],
            "doc_part": chunk["doc_part"],
        }
        for chunk in chunks
    ]

    documents = [chunk["text"] for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return collection


# ---------------------------------------------------------------------------
# Поиск релевантных фрагментов
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    embed_model: SentenceTransformer,
    collection,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Находит наиболее релевантные чанки для вопроса пользователя."""
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Вопрос пользователя пустой.")

    collection_size = collection.count()
    if collection_size == 0:
        raise ValueError("Векторная база пуста: в ней нет чанков для поиска.")

    actual_top_k = min(max(int(top_k), 1), collection_size)

    # Для E5 вопрос должен иметь префикс "query:".
    query_embedding = embed_model.encode(
        [f"query: {cleaned_query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=actual_top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents or not documents[0]:
        return []

    chunks: list[dict[str, Any]] = []

    for index, text in enumerate(documents[0]):
        metadata = metadatas[0][index] if metadatas and metadatas[0] else {}
        distance = (
            distances[0][index]
            if distances and distances[0] and index < len(distances[0])
            else 1.0
        )

        chunks.append(
            {
                "text": text,
                "source": metadata.get("source", "Неизвестный источник"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "article": metadata.get("article", ""),
                "doc_part": metadata.get("doc_part", ""),
                "distance": float(distance),
            }
        )

    return chunks


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Собирает найденные чанки в единый контекст для LLM."""
    context_parts: list[str] = []

    for number, chunk in enumerate(chunks, start=1):
        labels = [f"Источник: {chunk.get('source', 'Неизвестный источник')}"]

        doc_part = str(chunk.get("doc_part") or "").strip()
        article = str(chunk.get("article") or "").strip()

        if doc_part:
            labels.append(f"часть: {doc_part}")
        if article:
            labels.append(f"статья/пункт: {article}")

        header = "; ".join(labels)
        text = str(chunk.get("text") or "").strip()

        context_parts.append(f"[Фрагмент {number}. {header}]\n{text}")

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Генерация ответа через Hugging Face Inference Providers
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    context: str,
    mode: str,
    hf_token: str,
    model_name: str = LLM_MODEL_NAME,
) -> str:
    """Отправляет вопрос и найденный контекст в языковую модель."""
    cleaned_token = hf_token.strip()
    cleaned_model_name = model_name.strip()

    if not cleaned_token:
        raise ValueError("Не указан Hugging Face токен.")

    if not cleaned_model_name:
        raise ValueError("Не указано название языковой модели.")

    if not context.strip():
        return "В загруженных документах не найдено информации по этому вопросу."

    system_prompt = SYSTEM_OFFICIAL if mode == "official" else SYSTEM_SIMPLE

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Ниже приведены фрагменты документов, найденные системой.\n\n"
                f"{context}\n\n"
                f"Вопрос пользователя: {query.strip()}"
            ),
        },
    ]

    # provider="auto" позволяет Hugging Face выбрать доступного провайдера.
    llm_client = InferenceClient(
        provider="auto",
        api_key=cleaned_token,
        timeout=120,
    )

    try:
        response = llm_client.chat_completion(
            model=cleaned_model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )
    except Exception as exc:
        raise RuntimeError(
            "Не удалось получить ответ от Hugging Face Inference Providers. "
            "Проверьте интернет-соединение Streamlit, права HF-токена, "
            "доступность выбранной модели и баланс/лимиты Inference Providers. "
            f"Техническая ошибка: {exc}"
        ) from exc

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Hugging Face вернул ответ в неожиданном формате."
        ) from exc

    if not content or not str(content).strip():
        raise RuntimeError("Языковая модель вернула пустой ответ.")

    return str(content).strip()


# ---------------------------------------------------------------------------
# Полный RAG-цикл
# ---------------------------------------------------------------------------

def answer(
    query: str,
    embed_model: SentenceTransformer,
    collection,
    hf_token: str,
    mode: str = "official",
    top_k: int = 5,
    model_name: str = LLM_MODEL_NAME,
) -> dict[str, Any]:
    """Выполняет поиск по документам и генерирует итоговый ответ."""
    chunks = retrieve(
        query=query,
        embed_model=embed_model,
        collection=collection,
        top_k=top_k,
    )

    if not chunks:
        return {
            "answer": "В загруженных документах не найдено информации по этому вопросу.",
            "sources": [],
        }

    context = build_context(chunks)

    answer_text = generate_answer(
        query=query,
        context=context,
        mode=mode,
        hf_token=hf_token,
        model_name=model_name,
    )

    return {
        "answer": answer_text,
        "sources": chunks,
    }
