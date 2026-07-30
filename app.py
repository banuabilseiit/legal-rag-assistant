"""
Streamlit-приложение: онлайн-консультант по законодательству на базе RAG.

Запуск:
    streamlit run app.py

Требует наличия папки chroma_db/ рядом с этим файлом — векторной базы,
построенной заранее (см. ноутбук подготовки данных). Если папки нет,
приложение покажет инструкцию, как её получить.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.rag_pipeline import (
    load_embed_model,
    build_index_from_jsonl,
    answer,
    LLM_MODEL_NAME,
    CHUNKS_JSONL_PATH,
)

st.set_page_config(page_title="⚖️ Юридический AI-консультант", layout="wide")
st.title("⚖️ AI-консультант по законодательству")
st.caption(
    "Задайте вопрос по загруженным законам и НПА. "
    "Выберите формат ответа: официальный (со ссылками) или простой (без ссылок)."
)

# ---------------------------------------------------------------------------
# Проверка, что данные на месте
# ---------------------------------------------------------------------------
if not Path(CHUNKS_JSONL_PATH).exists():
    st.error(
        f"Файл `{CHUNKS_JSONL_PATH}` не найден рядом с приложением.\n\n"
        "Это подготовленные чанки законов/НПА. Убедитесь, что файл лежит в репозитории "
        "по пути data/processed/chunks.jsonl (сформируйте его через app_ingest.py)."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Загрузка модели и построение индекса (кешируется — выполняется один раз при старте)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Загружаю модель эмбеддингов (может занять минуту при первом запуске)...")
def get_embed_model():
    return load_embed_model()


@st.cache_resource(show_spinner="Строю векторный индекс из документов...")
def get_collection(_embed_model):
    return build_index_from_jsonl(_embed_model)


embed_model = get_embed_model()
collection = get_collection(embed_model)

# ---------------------------------------------------------------------------
# Боковая панель — настройки
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Настройки")

    hf_token = st.text_input(
        "Hugging Face токен",
        type="password",
        help="Токен не сохраняется — используется только для запросов в текущей сессии. "
             "Получить: https://huggingface.co/settings/tokens",
    )

    model_name = st.text_input("Модель для генерации", value=LLM_MODEL_NAME)

    mode_label = st.radio(
        "Формат ответа",
        ["Официальный (со ссылками на закон)", "Простой (без ссылок)"],
    )
    mode = "official" if "Официальный" in mode_label else "simple"


# ---------------------------------------------------------------------------
# Основная область — вопрос и ответ
# ---------------------------------------------------------------------------
query = st.text_area("Ваш вопрос:", height=100, placeholder="Например: какие требования к участникам платежной системы?")

col_btn, _ = st.columns([1, 4])
with col_btn:
    submit = st.button("Получить ответ", type="primary", use_container_width=True)

if submit:
    if not hf_token.strip():
        st.warning("Введите Hugging Face токен в боковой панели, чтобы получить ответ.")
        st.stop()
    if not query.strip():
        st.warning("Введите вопрос.")
        st.stop()

    with st.spinner("Ищу релевантные документы и формирую ответ..."):
        try:
            result = answer(
                query=query,
                embed_model=embed_model,
                collection=collection,
                hf_token=hf_token.strip(),
                mode=mode,
                top_k=top_k,
                model_name=model_name.strip(),
            )
        except Exception as e:
            st.error(f"Ошибка при обращении к модели: {e}")
            st.stop()

    st.subheader("Ответ")
    st.write(result["answer"])

    st.subheader("Использованные источники")
    for s in result["sources"]:
        part_label = f" — {s['doc_part']}" if s["doc_part"] else ""
        header = f"{s['source']}{part_label}, {s['article'] or 'без номера'} (сходство: {1 - s['distance']:.2f})"
        with st.expander(header):
            st.write(s["text"])
