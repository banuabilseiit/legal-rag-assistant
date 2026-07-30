from __future__ import annotations

from pathlib import Path
import html

import streamlit as st

from src.rag_pipeline import (
    CHUNKS_JSONL_PATH,
    LLM_MODEL_NAME,
    answer,
    build_index_from_jsonl,
    load_embed_model,
)


# ---------------------------------------------------------------------------
# 1. Настройки страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Юридический AI-консультант",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 2. Стили интерфейса
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Общий фон */
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.08), transparent 30%),
                #f6f8fc;
        }

        /* Ограничение ширины основного содержимого */
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Верхняя карточка */
        .hero-card {
            padding: 30px 34px;
            margin-bottom: 24px;
            border: 1px solid rgba(37, 99, 235, 0.15);
            border-radius: 22px;
            background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
        }

        .hero-badge {
            display: inline-block;
            padding: 7px 12px;
            margin-bottom: 12px;
            border-radius: 999px;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .hero-title {
            margin: 0;
            color: #0f172a;
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .hero-text {
            max-width: 760px;
            margin-top: 12px;
            margin-bottom: 0;
            color: #475569;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        /* Заголовок блока выбора режима */
        .section-title {
            margin-top: 4px;
            margin-bottom: 4px;
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 750;
        }

        .section-caption {
            margin-bottom: 12px;
            color: #64748b;
            font-size: 0.9rem;
        }

        /* Карточки radio */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            min-height: 108px;
            margin: 0;
            padding: 18px 20px;
            align-items: flex-start;
            border: 1px solid #dbe3ef;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
            cursor: pointer;
            transition: all 0.18s ease;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
            border-color: #93b4ef;
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.10);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
            border: 2px solid #2563eb;
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
            box-shadow: 0 10px 28px rgba(37, 99, 235, 0.14);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #0f172a;
            font-weight: 700;
        }

        /* Поле вопроса */
        div[data-testid="stTextArea"] textarea {
            min-height: 145px;
            padding: 16px;
            border: 1px solid #d7e0ec;
            border-radius: 14px;
            background: #ffffff;
            font-size: 1rem;
            line-height: 1.55;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10);
        }

        /* Основная кнопка */
        div.stButton > button[kind="primary"] {
            min-height: 48px;
            padding: 0 24px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #ffffff;
            font-weight: 750;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);
        }

        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            color: #ffffff;
            transform: translateY(-1px);
        }

        /* Карточка ответа */
        .answer-card {
            padding: 24px 26px;
            margin-top: 16px;
            border: 1px solid #dbe3ef;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .answer-label {
            margin-bottom: 10px;
            color: #1d4ed8;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] input {
            color: #0f172a !important;
            background: #ffffff !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="slider"] {
            padding-top: 8px;
        }

        /* Мобильная версия */
        @media (max-width: 760px) {
            .hero-card {
                padding: 24px 20px;
            }

            .hero-title {
                font-size: 1.8rem;
            }

            div[data-testid="stRadio"] > div[role="radiogroup"] {
                grid-template-columns: 1fr;
            }
        }

        /* Скрываем стандартный footer */
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 3. Проверка данных
# ---------------------------------------------------------------------------
if not Path(CHUNKS_JSONL_PATH).exists():
    st.error(
        f"Файл `{CHUNKS_JSONL_PATH}` не найден.\n\n"
        "Добавьте подготовленный файл в репозиторий по пути "
        "`data/processed/chunks.jsonl`."
    )
    st.stop()


# ---------------------------------------------------------------------------
# 4. Загрузка модели и индекса
# ---------------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Загружаю модель для поиска по документам..."
)
def get_embed_model():
    return load_embed_model()


@st.cache_resource(
    show_spinner="Создаю поисковый индекс..."
)
def get_collection(_embed_model):
    return build_index_from_jsonl(_embed_model)


embed_model = get_embed_model()
collection = get_collection(embed_model)


# ---------------------------------------------------------------------------
# 5. Боковая панель
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    st.caption("Параметры подключения и поиска")

    hf_token = st.text_input(
        "Hugging Face токен",
        type="password",
        placeholder="hf_...",
        help=(
            "Токен используется только для обращения к языковой модели "
            "в текущей сессии."
        ),
    )

    with st.expander("Дополнительные настройки"):
        model_name = st.text_input(
            "Модель для генерации",
            value=LLM_MODEL_NAME,
            help="Название модели на Hugging Face.",
        )

        top_k = st.slider(
            "Количество найденных фрагментов",
            min_value=1,
            max_value=10,
            value=5,
            help="Сколько наиболее релевантных чанков передавать модели.",
        )

    st.divider()
    st.metric(
        "Документов в поисковой базе",
        collection.count(),
    )

    st.info(
        "Ответ формируется только после поиска по загруженным "
        "законам и нормативным актам."
    )


# ---------------------------------------------------------------------------
# 6. Верхний блок
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">RAG · Юридический поиск</div>
        <h1 class="hero-title">⚖️ Юридический AI-консультант</h1>
        <p class="hero-text">
            Задайте вопрос по загруженным законам и нормативным актам.
            Система найдёт релевантные положения и подготовит ответ
            в выбранном вами формате.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 7. Красивый выбор формата ответа
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="section-title">Выберите формат ответа</div>
    <div class="section-caption">
        Поиск по документам одинаковый — меняется стиль объяснения результата.
    </div>
    """,
    unsafe_allow_html=True,
)

mode_label = st.radio(
    "Формат ответа",
    options=[
        "🏛️ Официальный ответ",
        "💬 Простое объяснение",
    ],
    captions=[
        "Деловой юридический стиль, название документа, статья и ссылки на найденные нормы.",
        "Понятное объяснение без сложных формулировок и перегруженных юридических ссылок.",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

mode = (
    "official"
    if mode_label == "🏛️ Официальный ответ"
    else "simple"
)


# ---------------------------------------------------------------------------
# 8. Поле вопроса
# ---------------------------------------------------------------------------
st.markdown("### Ваш вопрос")

st.markdown(
    """
    <style>
    div[data-testid="stTextArea"] textarea {
        color: #000000 !important;
        caret-color: #000000 !important;
        background-color: #ffffff !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

query = st.text_area(
    "Введите юридический вопрос",
    height=145,
    placeholder=(
        "Например: какие требования предъявляются "
        "к участникам платёжной системы?"
    ),
    label_visibility="collapsed",
)

col_button, col_hint = st.columns([1, 2.5])

with col_button:
    submit = st.button(
        "Найти ответ",
        type="primary",
        use_container_width=True,
    )

with col_hint:
    st.caption(
        "Чем точнее вопрос, тем точнее поиск по документам."
    )


# ---------------------------------------------------------------------------
# 9. Получение ответа
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .answer-card,
    .answer-card *,
    .answer-card div,
    .answer-card p,
    .answer-card span {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    .answer-card {
        background-color: #ffffff !important;
        border: 1px solid #dbe3ef;
        border-radius: 18px;
        padding: 24px 26px;
        margin-top: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }

    .answer-card .answer-label {
        color: #2563eb !important;
        -webkit-text-fill-color: #2563eb !important;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .answer-card .answer-text {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-size: 1rem;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if submit:
    if not hf_token.strip():
        st.warning("Введите Hugging Face токен в боковой панели.")
        st.stop()

    if not query.strip():
        st.warning("Введите вопрос.")
        st.stop()

    with st.spinner("Ищу подходящие нормы и формирую ответ..."):
        try:
            result = answer(
                query=query.strip(),
                embed_model=embed_model,
                collection=collection,
                hf_token=hf_token.strip(),
                mode=mode,
                top_k=top_k,
                model_name=model_name.strip(),
            )

        except Exception as error:
            st.error("Не удалось получить ответ от языковой модели.")

            with st.expander("Показать техническую ошибку"):
                st.code(str(error))

            st.stop()

    safe_answer = html.escape(
        str(result.get("answer", "Ответ не получен."))
    ).replace("\n", "<br>")

    answer_html = (
        '<div class="answer-card">'
        '<div class="answer-label">Ответ системы</div>'
        f'<div class="answer-text">{safe_answer}</div>'
        '</div>'
    )

    st.markdown(
        answer_html,
        unsafe_allow_html=True,
    )

    st.markdown("### Использованные источники")

    sources = result.get("sources", [])

    if not sources:
        st.info("Релевантные источники не найдены.")

    else:
        for number, source in enumerate(sources, start=1):
            doc_part = source.get("doc_part") or ""
            article = source.get("article") or "без номера статьи"
            source_name = source.get("source") or "Источник"
            distance = source.get("distance")

            if isinstance(distance, (int, float)):
                similarity = max(
                    0.0,
                    min(1.0, 1 - distance),
                )
                similarity_label = f"{similarity:.0%} совпадения"
            else:
                similarity_label = "релевантный фрагмент"

            part_label = f" · {doc_part}" if doc_part else ""

            source_title = (
                f"{number}. {source_name}{part_label} · "
                f"{article} · {similarity_label}"
            )

            with st.expander(source_title):
                st.write(
                    source.get(
                        "text",
                        "Текст источника отсутствует.",
                    )
                )
