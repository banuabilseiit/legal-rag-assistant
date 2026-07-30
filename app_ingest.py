"""
Streamlit-страница: загрузка законов, НПА и внутренних ответов подразделения.

Запуск:
    streamlit run app_ingest.py

Что делает:
1. Позволяет загрузить несколько файлов (.txt, .pdf, .docx) через браузер.
2. Для каждой группы файлов нужно указать тип документа
   (закон / НПА / внутренний ответ подразделения).
3. Извлекает текст, чистит его и разбивает на чанки по статьям.
4. Показывает превью чанков и статистику.
5. Даёт скачать результат как .jsonl и/или сохранить в data/processed/chunks.jsonl,
   откуда его дальше заберёт скрипт построения векторного индекса (embed_store.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.document_parser import extract_text, UnsupportedFileType
from src.ingest import clean_text, split_by_articles, chunks_to_jsonl, save_chunks, load_chunks

PROCESSED_PATH = Path("data/processed/chunks.jsonl")

DOC_TYPE_LABELS = {
    "law": "Закон",
    "npa": "НПА (нормативно-правовой акт)",
    "internal_answer": "Внутренний ответ подразделения",
}

st.set_page_config(page_title="Загрузка документов | Юридический RAG", layout="wide")
st.title("📥 Загрузка документов")
st.caption(
    "Загрузите законы, НПА и ответы подразделения. Каждый файл будет очищен "
    "и разбит на чанки (по статьям, если они есть, иначе по абзацам)."
)

# ---------------------------------------------------------------------------
# 1. Загрузка файлов
# ---------------------------------------------------------------------------
st.subheader("1. Выберите файлы")

uploaded_files = st.file_uploader(
    "Перетащите или выберите файлы (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.markdown("**Укажите тип документа для каждого файла:**")
    doc_types: dict[str, str] = {}

    cols_per_row = 2
    for i in range(0, len(uploaded_files), cols_per_row):
        row_files = uploaded_files[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, f in zip(cols, row_files):
            with col:
                choice = st.selectbox(
                    f.name,
                    options=list(DOC_TYPE_LABELS.keys()),
                    format_func=lambda k: DOC_TYPE_LABELS[k],
                    key=f"type_{f.name}_{f.size}",
                )
                doc_types[f.name] = choice

    # -----------------------------------------------------------------------
    # 2. Обработка
    # -----------------------------------------------------------------------
    st.subheader("2. Обработка")

    if st.button("Обработать файлы", type="primary"):
        all_chunks = []
        errors = []
        progress = st.progress(0.0, text="Обработка файлов...")

        for idx, f in enumerate(uploaded_files):
            try:
                raw_text, name = extract_text(f)
                cleaned = clean_text(raw_text)

                if not cleaned.strip():
                    errors.append(f"⚠️ {f.name}: не удалось извлечь текст (файл пустой или это скан без OCR)")
                    continue

                doc_type = doc_types[f.name]
                chunks = split_by_articles(cleaned, source=name, doc_type=doc_type)
                all_chunks.extend(chunks)

            except UnsupportedFileType as e:
                errors.append(f"❌ {f.name}: {e}")
            except Exception as e:
                errors.append(f"❌ {f.name}: ошибка обработки — {e}")

            progress.progress((idx + 1) / len(uploaded_files), text=f"Обработано {idx + 1}/{len(uploaded_files)}")

        progress.empty()

        if errors:
            for err in errors:
                st.warning(err)

        if all_chunks:
            st.session_state["last_chunks"] = all_chunks
            st.success(f"Готово! Извлечено {len(all_chunks)} чанков из {len(uploaded_files) - len(errors)} файлов.")
        else:
            st.error("Не удалось извлечь ни одного чанка. Проверьте файлы.")

# ---------------------------------------------------------------------------
# 3. Превью и статистика
# ---------------------------------------------------------------------------
if "last_chunks" in st.session_state:
    chunks = st.session_state["last_chunks"]

    st.subheader("3. Превью и статистика")

    df = pd.DataFrame([{
        "source": c.source,
        "doc_type": DOC_TYPE_LABELS[c.doc_type],
        "article": c.article or "—",
        "length": len(c.text),
        "text": c.text,
    } for c in chunks])

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего чанков", len(df))
    col2.metric("Файлов обработано", df["source"].nunique())
    col3.metric("Средняя длина чанка", f"{int(df['length'].mean())} симв.")

    st.bar_chart(df.groupby("doc_type")["length"].count())

    st.dataframe(
        df[["source", "doc_type", "article", "length"]],
        use_container_width=True,
        height=300,
    )

    with st.expander("Посмотреть текст конкретного чанка"):
        selected_idx = st.number_input("Номер чанка", 0, len(df) - 1, 0)
        st.text(df.iloc[selected_idx]["text"])

    # -----------------------------------------------------------------------
    # 4. Сохранение
    # -----------------------------------------------------------------------
    st.subheader("4. Сохранение результата")

    jsonl_data = chunks_to_jsonl(chunks)

    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            "⬇️ Скачать chunks.jsonl",
            data=jsonl_data,
            file_name="chunks.jsonl",
            mime="application/json",
        )

    with col_b:
        append = st.checkbox("Дописать к существующему файлу (не перезаписывать)", value=True)
        if st.button("💾 Сохранить в data/processed/chunks.jsonl"):
            save_chunks(chunks, PROCESSED_PATH, append=append)
            st.success(f"Сохранено в {PROCESSED_PATH}. Теперь можно запускать src/embed_store.py")

# ---------------------------------------------------------------------------
# Что уже сохранено на диске
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Что уже накоплено в базе (data/processed/chunks.jsonl)")

existing = load_chunks(PROCESSED_PATH)
if existing:
    df_existing = pd.DataFrame(existing)
    df_existing["doc_type"] = df_existing["doc_type"].map(DOC_TYPE_LABELS)
    st.write(f"Всего сохранено: **{len(df_existing)}** чанков")
    st.dataframe(
        df_existing.groupby("doc_type").size().reset_index(name="количество чанков"),
        use_container_width=True,
    )
else:
    st.info("Пока ничего не сохранено. Загрузите и обработайте файлы выше.")
