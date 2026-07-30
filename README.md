# Юридический AI-консультант (RAG)

## Структура проекта

```
legal-rag-assistant/
├── app.py                    # главное приложение — вопрос/ответ по документам
├── app_ingest.py              # приложение для загрузки и чанкинга новых документов
├── src/
│   ├── document_parser.py     # извлечение текста из .txt/.pdf/.docx
│   ├── ingest.py               # очистка текста и разбиение на чанки
│   └── rag_pipeline.py         # retrieval + генерация ответа (2 режима)
├── data/
│   └── processed/
│       └── chunks.jsonl        # подготовленные чанки (кладём в репозиторий, это лёгкий текстовый файл)
├── requirements.txt
└── .gitignore
```

Важно: папка `chroma_db/` (бинарная векторная база) **не хранится в git** — при старте
приложения индекс строится в памяти прямо из `data/processed/chunks.jsonl`
(см. функцию `build_index_from_jsonl`). Так репозиторий остаётся лёгким и без
проблем с бинарными файлами в git.

## Локальный запуск

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Публикация на GitHub

```bash
cd legal-rag-assistant
git init
git add .
git commit -m "Initial commit: legal RAG assistant"

# создайте пустой репозиторий на github.com (без README/license — они уже есть локально)
git remote add origin https://github.com/ВАШ_ЛОГИН/legal-rag-assistant.git
git branch -M main
git push -u origin main
```

Обязательно убедитесь, что `data/processed/chunks.jsonl` реально закоммичен —
без него приложение не сможет построить индекс:

```bash
git status                       # chunks.jsonl должен быть в списке отслеживаемых
git add data/processed/chunks.jsonl
git commit -m "Add prepared chunks"
git push
```

## Деплой на Streamlit Community Cloud

1. Зайдите на https://share.streamlit.io и войдите через GitHub.
2. Нажмите **"New app"**.
3. Выберите ваш репозиторий `legal-rag-assistant`, ветку `main`, и файл `app.py`.
4. Нажмите **Deploy**.

Первый запуск займёт несколько минут — приложение скачает модель эмбеддингов
(`intfloat/multilingual-e5-large`, ~2GB) и построит индекс из `chunks.jsonl`.
Благодаря `st.cache_resource` это происходит один раз, пока приложение не "уснёт"
от неактивности.

### Hugging Face токен

Токен вводится прямо в интерфейсе (в боковой панели) — ничего дополнительно
настраивать в облаке не нужно. Токен не сохраняется на сервере, используется
только в рамках сессии браузера пользователя.

Если хотите не вводить токен каждый раз, а зашить его в облачные секреты
(тогда поле в интерфейсе можно убрать):
1. В настройках приложения на share.streamlit.io откройте **Settings → Secrets**.
2. Добавьте:
   ```toml
   HF_TOKEN = "hf_ваш_токен"
   ```
3. В `app.py` замените `st.text_input(...)` на `st.secrets["HF_TOKEN"]`.

## Обновление данных (новые документы)

1. Запустите `app_ingest.py` локально (`streamlit run app_ingest.py`), загрузите
   новые документы, сохраните результат — он допишется в `data/processed/chunks.jsonl`.
2. Закоммитьте обновлённый `chunks.jsonl` и запушьте:
   ```bash
   git add data/processed/chunks.jsonl
   git commit -m "Update documents"
   git push
   ```
3. Streamlit Cloud автоматически подхватит изменения и пересоберёт приложение
   (индекс пересчитается заново из обновлённого файла при следующем холодном старте).

## Модель для генерации

По умолчанию используется `Qwen/Qwen2.5-72B-Instruct` через Hugging Face Inference API.
Если модель недоступна на бесплатном тарифе, в интерфейсе можно ввести другую, например:
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
