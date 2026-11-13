# doc-orchestrator

Сервис оркестрации распознавания документов (FastAPI + Redis + asyncio) для сервера с двумя GPU по 8 ГБ.
Принимает файл, выбирает OCR/VLM-движок, выстраивает очередь к видеокартам, проверяет реквизиты
валидаторами и возвращает:

* **Markdown** — для Open WebUI (`PUT /process`);
* **строгий JSON с валидацией** — для 1С:ДО и Teamcenter (`POST /api/extract-structured`).

| Что пришло | Куда идёт |
|---|---|
| DOCX, XLSX, PPTX, HTML, MD, TXT, PDF с текстовым слоем | Docling (CPU) |
| Скан: PDF без текста, JPG, PNG, TIFF, BMP, WebP | MinerU (GPU 0); схемы из скана Qwen3-VL (GPU 1) переводит в Mermaid |
| Скан с печатями или сложными таблицами (`X-OCR-Engine: paddle`, `has_seals=true`, скан УПД) | PaddleOCR-VL (GPU 0) |
| УПД, счёт (`doc_type=upd`, `invoice`) | OCR → Qwen3-VL (JSON по схеме) → проверка ИНН, КПП, БИК и сумм |
| Чертёж (`doc_type=drawing`) | кропы штампа и техтребований → Qwen3-VL |
| Прочие текстовые файлы (CSV, JSON, XML, код…) | отдаются как есть, без движков |

Настоящих ML-движков в репозитории нет: на тестовом стенде их заменяет сервис `mock-engines`. Он
имитирует ограничения железа: `500 Simulated OOM`, если на GPU 0 оказались две модели, и `429`
на параллельные запросы к GPU 1.

## Запуск тестового стенда

```bash
docker compose up --build -d       # redis, mock-engines (:8001), doc-orchestrator (:8000)
curl http://localhost:8000/health  # {"status":"ok"}
```

Нужен Docker Compose 2.24 или новее. Описание API — <http://localhost:8000/docs>. Что загружено
в «видеопамять» мока и нарушались ли лимиты GPU — `curl http://localhost:8001/mock/state`.

## Примеры запросов

Для примеров подойдут любые свои файлы: мок-движки не смотрят на содержимое.

### Open WebUI: документ → Markdown

```bash
curl -X PUT http://localhost:8000/process \
  -H "X-Filename: scan.jpg" \
  -H "Content-Type: image/jpeg" \
  --data-binary @scan.jpg
```

```json
{
  "page_content": "# Скан scan.jpg (MinerU)\n\nРаспознанный текст.\n\n```mermaid\ngraph TD;\n  A[Вход] --> B[Обработка];\n```\n\nТекст после схемы.",
  "metadata": {"filename": "scan.jpg", "route": "SCAN_IMAGE", "engine_used": "mineru+qwen3-vl"}
}
```

Необязательные заголовки:

* `X-OCR-Engine: paddle` — распознать скан в PaddleOCR-VL (печати, сложные таблицы); по умолчанию `auto` — MinerU;
* `Authorization: Bearer <ключ>` — обязателен, если задан `API_KEY`.

Настройки на стороне Open WebUI (переменные окружения):

```bash
CONTENT_EXTRACTION_ENGINE=external
EXTERNAL_DOCUMENT_LOADER_URL=http://doc-orchestrator:8000
# Ключ обязателен: без него Open WebUI не включает внешний загрузчик.
# Если в оркестраторе задан API_KEY, значения должны совпадать.
EXTERNAL_DOCUMENT_LOADER_API_KEY=my-key
```

### 1С:ДО: УПД или счёт → JSON с валидацией

```bash
curl -X POST http://localhost:8000/api/extract-structured \
  -F "file=@upd.pdf" \
  -F "doc_type=upd"
```

Ответ (сокращён); схемы поля `data` лежат в `app/schemas/`:

```
{
  "status": "completed",
  "doc_type": "upd",
  "data": {
    "number": "128",
    "date": "2025-03-14",
    "seller": {"name": "ООО «Ромашка»", "inn": "7707083893", "kpp": "770701001", "address": "…"},
    "buyer": {…},
    "items": [
      {"name": "Подшипник 6205-2RS", "unit": "шт", "qty": 10.0, "price": 250.0, "amount_without_vat": 2500.0,
       "vat_rate": "20%", "vat_amount": 500.0, "row_total": 3000.0},
      …
    ],
    "total_with_vat": 34923.0,
    …
  },
  "validation": {"is_valid": true, "warnings": []}
}
```

Ошибки в данных — не сбой: сервис отвечает `200 OK`, отдаёт `data` как есть, а причины пишет в `warnings`,
чтобы документ ушёл оператору на ручную проверку. Мок возвращает «битый» УПД для файла с именем `bad_inn.pdf`:

```bash
curl -X POST http://localhost:8000/api/extract-structured \
  -F "file=@upd.pdf;filename=bad_inn.pdf" \
  -F "doc_type=upd"
```

```
"validation": {
  "is_valid": false,
  "warnings": [
    "ИНН продавца 7701234569 не прошёл контрольную сумму",
    "Сумма по строкам (1200.00) != Итого по документу (1500.00)"
  ]
}
```

Необязательные поля формы:

* `has_seals=true` — на скане печати или сложные таблицы: его прочитает PaddleOCR-VL (скан УПД — всегда);
* `async_mode=true` — обработка в фоне: ответ `202` с `task_id`, результат отдаёт `GET /api/tasks/{task_id}`.

### Teamcenter: чертёж → основная надпись и техтребования

```bash
curl -X POST http://localhost:8000/api/extract-structured \
  -F "file=@drawing.pdf" \
  -F "doc_type=drawing" \
  -F "async_mode=true"
# 202 Accepted: {"task_id": "a9131301-…", "status": "queued"}

curl http://localhost:8000/api/tasks/a9131301-…
```

```
{
  "task_id": "a9131301-…",
  "status": "completed",
  "doc_type": "drawing",
  "data": {
    "designation": "АБВГ.711141.001", "title": "Вал приводной", "material": "Сталь 45 ГОСТ 1050-2013",
    "mass": "12.5", "scale": "1:1", "organization": "АО КБ Маш",
    "requirements": ["1. Неуказанные предельные отклонения размеров H14, h14, +/-IT14/2.", "2. Острые кромки притупить R0,5."]
  },
  "validation": {"is_valid": true, "warnings": []}
}
```

Состояния задачи: `queued` → `processing` → `completed` или `failed` (причина — в поле `error`).
Состояние и результат хранятся час (`TASK_TTL_SEC`).

### Коды ошибок

Тело ответа с ошибкой — `{"detail": "описание"}`.

| Код | Когда |
|---|---|
| 401 | неверный ключ в `PUT /process` |
| 404 | задача не найдена или срок её хранения истёк |
| 413 | файл больше `MAX_FILE_SIZE_MB` |
| 415 | двоичный файл неизвестного формата |
| 422 | файл пуст, повреждён или защищён паролем; неверные параметры запроса |
| 502 | движок (OCR, VLM) недоступен или ответил ошибкой |
| 503 | видеокарта занята дольше `GPU_LOCK_WAIT_SEC`; недоступен Redis |

## Боевой сервер

Все параметры задаются переменными окружения или файлом `.env`: образец — `.env.example`, значения
по умолчанию — `app/config.py`. Адресов в коде нет: для перехода на настоящие движки достаточно
указать их URL и запустить стенд без заглушки:

```bash
cp .env.example .env   # указать DOCLING_URL, MINERU_URL, PADDLE_URL, OLLAMA_URL и при необходимости API_KEY
docker compose -f docker-compose.prod.yml up -d --build   # redis и doc-orchestrator (:8000)
```

Движок, запущенный на самом сервере, из контейнера доступен как `host.docker.internal`, например
`OLLAMA_URL=http://host.docker.internal:11434` (Ollama при этом должен слушать не только `127.0.0.1`:
`OLLAMA_HOST=0.0.0.0`). Модель задаётся в `VLM_MODEL` — по умолчанию `qwen3-vl:4b-instruct-q8_0`;
нужен instruct-вариант, не thinking.

От движков оркестратор ждёт такие HTTP-контракты (те же, что у `mock-engines`):

| Движок | Запрос | Ответ |
|---|---|---|
| Docling | `POST {DOCLING_URL}/convert`, multipart-поле `file` | `{"markdown": "…"}` |
| MinerU | `POST {MINERU_URL}/extract`, поле `file`; `POST …/load`, `POST …/unload` | `{"markdown": "… ![Схема](fig_1.png) …", "figures": {"fig_1.png": "<base64>"}}` |
| PaddleOCR-VL | `POST {PADDLE_URL}/extract`, поле `file`; `POST …/load`, `POST …/unload` | `{"markdown": "…"}` |
| Ollama (Qwen3-VL) | `POST {OLLAMA_URL}/api/chat` — штатный API Ollama; JSON Schema ответа — в поле `format` | `{"message": {"content": "…"}}` |

Контракт Ollama — родной. У штатных серверов остальных движков API другой (docling-serve —
`POST /v1/convert/file`, MinerU — `POST /v1/file_parse`, PaddleOCR-VL — `POST /layout-parsing`),
и команд `load` / `unload` у них нет: перед ними нужен тонкий адаптер под контракт из таблицы.

## Как это устроено

* **Очередь к GPU** (`app/core/gpu_manager.py`). Локи Redis `lock:gpu0` и `lock:gpu1` пропускают на каждую
  видеокарту по одной задаче. Перед сменой модели на GPU 0 прежняя выгружается из VRAM. Если PaddleOCR-VL
  простаивает дольше `PADDLE_IDLE_TTL_SEC`, фоновая проверка возвращает на GPU 0 основную модель — MinerU.
* **Картинки для Qwen3-VL** (`app/core/image_ops.py`) уменьшаются до `MAX_PIXELS`, стороны кратны 32 px.
  У чертежа вырезаются зоны штампа и техтребований — по-разному для книжного A4 и альбомных A3–A0.
* **Валидаторы** (`app/validators/`): контрольные цифры ИНН, формат КПП и БИК, сходимость сумм в строках
  и итога документа с допуском 0,05.
* **Фоновые задачи** (`async_mode=true`) выполняются внутри сервиса, одновременно — не больше
  `TASK_CONCURRENCY`. При перезапуске сервиса незавершённые задачи получают статус `failed`:
  документ нужно отправить повторно.

## Тесты

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

Docker и сеть тестам не нужны: `mock-engines` и Redis (fakeredis) работают внутри процесса,
тестовые PDF и PNG создаются на лету.

Запуск без Docker (нужен Redis на `localhost:6379`):

```bash
uvicorn main:app --app-dir mocks --port 8001 &
REDIS_URL=redis://localhost:6379/0 DOCLING_URL=http://localhost:8001/docling \
  MINERU_URL=http://localhost:8001/mineru PADDLE_URL=http://localhost:8001/paddle \
  OLLAMA_URL=http://localhost:8001 uvicorn app.main:app --port 8000
```
