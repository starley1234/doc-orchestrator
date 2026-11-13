# doc-orchestrator

Сервис оркестрации обработки документов (FastAPI + Redis). Принимает файл, выбирает OCR/VLM-движок,
выстраивает очередь к видеокартам, проверяет результат в DLP и валидаторами и возвращает:

* **Markdown** — для Open WebUI (`PUT /process`);
* **строгий JSON с валидацией** — для 1С и Teamcenter (`POST /api/extract-structured`).

| Что пришло | Куда идёт |
|---|---|
| DOCX, XLSX, PPTX, HTML, PDF с текстовым слоем | Docling (CPU) |
| Скан: PDF без текста, JPG, PNG, TIFF | MinerU (GPU 0); картинки из скана описывает Qwen3-VL (GPU 1) |
| Скан с таблицей или формой — в имени файла есть «упд», «счёт», «накладная», «таблица», `invoice`, `table`, `form`… | PaddleOCR-VL (GPU 0) |
| УПД, счёт, договор (`doc_type`) | Docling или PaddleOCR-VL → Qwen3-VL (JSON по схеме) → валидаторы |
| Чертёж (`doc_type=drawing`) | кропы штампа и техтребований → Qwen3-VL |

Настоящих ML-движков в репозитории нет: их заменяет сервис `mock-engines`. Он отвечает по тем же
HTTP-контрактам и имитирует ограничения железа: `500 Out Of Memory`, если на GPU 0 оказались две модели
или в Qwen3-VL пришла слишком большая картинка, и `429` на параллельные запросы к GPU 1.

## Запуск

```bash
docker compose up --build -d       # redis, mock-engines (:8001), doc-orchestrator (:8000)
curl http://localhost:8000/health  # {"status":"ok"}
```

Нужен Docker Compose 2.24 или новее. Описание API — <http://localhost:8000/docs>. Что загружено
в «видеопамять» мока и нарушались ли лимиты GPU — `curl http://localhost:8001/state`.

## Примеры запросов

### Open WebUI: документ → Markdown

```bash
curl -X PUT http://localhost:8000/process \
  -H "X-Filename: scan.pdf" \
  -H "Content-Type: application/pdf" \
  -H "Authorization: Bearer my-token" \
  --data-binary @tests/samples/scan.pdf
```

```json
{
  "page_content": "# Отчёт (MinerU)\n\nРаспознанный текст скана.\n\n```mermaid\ngraph TD; A-->B;\n```\n\nТекст после схемы.",
  "metadata": {"filename": "scan.pdf", "engine_used": "mineru+qwen3vl", "dlp_masked": false}
}
```

Заголовок `Authorization` проверяется, только если задан `OPENWEBUI_API_TOKEN`.

Настройки на стороне Open WebUI (переменные окружения):

```bash
CONTENT_EXTRACTION_ENGINE=external
EXTERNAL_DOCUMENT_LOADER_URL=http://doc-orchestrator:8000
# Ключ обязателен: без него Open WebUI не включает внешний загрузчик.
# Если задан OPENWEBUI_API_TOKEN, ключ должен с ним совпадать.
EXTERNAL_DOCUMENT_LOADER_API_KEY=my-token
```

В этом режиме Open WebUI отправляет в загрузчик файлы любых типов. Форматы, которых нет в таблице выше
(например, `.txt`, `.md`, `.csv`), сервис отклоняет с кодом 415.

### 1С: документ → JSON с валидацией

```bash
curl -X POST http://localhost:8000/api/extract-structured \
  -F "file=@tests/samples/scan.pdf" \
  -F "doc_type=upd"
```

`doc_type` — `upd`, `invoice`, `contract` или `drawing`; схемы поля `data` лежат в `app/schemas/`.
Ответ (сокращён):

```
{
  "status": "completed",
  "doc_type": "upd",
  "data": {
    "number": "128",
    "date": "2025-03-14",
    "seller": {"name": "ООО «Ромашка»", "inn": "7701234560", "kpp": "770101001", "address": "…"},
    "buyer": {…},
    "items": [
      {"name": "Подшипник 6205-2RS", "unit": "шт", "qty": 10.0, "price": 250.0, "amount_without_vat": 2500.0,
       "vat_rate": "20%", "vat_amount": 500.0, "row_total": 3000.0},
      …
    ],
    "document_total": 34923.0,
    …
  },
  "validation": {"is_valid": true, "warnings": []}
}
```

Ошибки в данных — не сбой: сервис отвечает `200 OK`, отдаёт `data` как есть, а причины пишет в `warnings`,
чтобы документ ушёл на ручную проверку. Мок возвращает «битый» УПД для файла с именем `bad_inn.pdf`:

```bash
curl -X POST http://localhost:8000/api/extract-structured \
  -F "file=@tests/samples/scan.pdf;filename=bad_inn.pdf" \
  -F "doc_type=upd"
```

```
"validation": {
  "is_valid": false,
  "warnings": [
    "ИНН продавца 7701234569 не прошёл контрольную сумму",
    "Сумма по строкам (34923.00) != Итого (35923.00)"
  ]
}
```

Асинхронный режим: задача ставится в очередь, результат забирают отдельным запросом.

```bash
curl -X POST http://localhost:8000/api/extract-structured \
  -F "file=@tests/samples/scan.pdf" \
  -F "doc_type=invoice" \
  -F "async_mode=true"
# 202 Accepted: {"task_id": "5cbb32e1-…", "status": "queued"}

curl http://localhost:8000/api/tasks/5cbb32e1-…
# {"task_id": "5cbb32e1-…", "status": "completed", "doc_type": "invoice", "data": {…}, "validation": {…}}
```

Состояния задачи: `queued` → `processing` → `completed` или `failed` (причина — в поле `error`).

### Teamcenter: чертёж → основная надпись и техтребования

```bash
curl -X POST http://localhost:8000/api/extract-structured \
  -F "file=@tests/samples/drawing.png" \
  -F "doc_type=drawing"
```

```
{
  "status": "completed",
  "doc_type": "drawing",
  "data": {
    "designation": "АБВГ.123456.001", "title": "Вал приводной", "material": "Сталь 45 ГОСТ 1050-2013",
    "mass": "12.5", "scale": null, "organization": null,
    "tech_requirements": ["1. 42...48 HRC.", …, "3. Острые кромки притупить."]
  },
  "validation": {"is_valid": true, "warnings": []}
}
```

### Коды ошибок

Тело ответа с ошибкой — `{"detail": "описание"}`.

| Код | Когда |
|---|---|
| 401 | неверный токен в `PUT /process` |
| 404 | задача не найдена или срок её хранения истёк |
| 413 | файл больше `MAX_FILE_SIZE_MB` |
| 415 | формат файла не поддерживается |
| 422 | файл пуст, повреждён или защищён паролем; неверные параметры запроса |
| 502 | движок (OCR, VLM, DLP) недоступен или ответил ошибкой |
| 503 | видеокарта занята дольше `GPU_WAIT_TIMEOUT_S`; недоступен Redis |

## Настройка и переход на боевые движки

Все параметры задаются переменными окружения или файлом `.env`: образец — `.env.example`, значения
по умолчанию — `app/config.py`. Адресов в коде нет, поэтому для перехода с заглушек на боевые движки
достаточно поменять URL (сервис `mock-engines` и его `depends_on` тогда можно убрать из `docker-compose.yml`):

```bash
# .env — базовые адреса; пути из таблицы ниже оркестратор добавляет сам
DOCLING_URL=http://docling.internal:5001
MINERU_URL=http://mineru.internal:8000
PADDLE_URL=http://paddle.internal:8000
VLM_GATEWAY_URL=http://gateway.internal:8080
DLP_URL=http://dlp.internal:8000
```

От движков оркестратор ждёт тот же HTTP-контракт, что у `mock-engines`. Если API боевого движка
отличается, перед ним ставится тонкий адаптер.

| Движок | Запрос | Ответ |
|---|---|---|
| Docling | `POST {DOCLING_URL}/convert`, multipart-поле `file` | `{"markdown": "…"}` |
| MinerU | `POST {MINERU_URL}/extract`, поле `file`; `POST …/load`, `POST …/unload` | `{"markdown": "… ![fig_1](fig_1.png) …", "images": [{"name": "fig_1.png", "data": "<base64>"}]}` |
| PaddleOCR-VL | `POST {PADDLE_URL}/extract`, поле `file`; `POST …/load`, `POST …/unload` | `{"markdown": "…"}` |
| Gateway (Qwen3-VL) | `POST {VLM_GATEWAY_URL}/v1/chat/completions` — чат в формате OpenAI; JSON Schema ответа — в поле `format` | `{"choices": [{"message": {"content": "…"}}]}` |
| DLP | `POST {DLP_URL}/inspect`, JSON `{"text": "…"}` | `{"text": "…", "masked": true}` |

## Как это устроено

* **Очередь к GPU** (`app/core/gpu_manager.py`). Локи Redis `lock:gpu0` и `lock:gpu1` пропускают на каждую
  видеокарту по одной задаче. Перед сменой модели на GPU 0 прежняя выгружается из VRAM. Если PaddleOCR-VL
  простаивает дольше `PADDLE_IDLE_TTL_S`, фоновая проверка возвращает на GPU 0 основную модель — MinerU.
* **DLP.** Промпт проверяется перед отправкой в Qwen3-VL, готовый результат — перед выдачей клиенту.
  О маскировании сообщает `dlp_masked` в Markdown-ответе, а в JSON-ответе — предупреждение и `is_valid=false`.
* **Валидаторы** (`app/validators/`): контрольные цифры ИНН, формат КПП и БИК, сходимость сумм в строках
  и итога документа с допуском 0,05.
* **Асинхронные задачи** (`app/core/tasks.py`): очередь на списках Redis, воркеры (`TASK_WORKERS`) работают
  внутри сервиса. Результат хранится `TASK_TTL_S` секунд. Задача, прерванная остановкой сервиса, после
  запуска возвращается в очередь. Очередь задач рассчитана на один экземпляр сервиса.

## Тесты

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

Docker и сеть тестам не нужны: `mock-engines` и Redis (fakeredis) работают внутри процесса.
Тестовые файлы в `tests/samples/` создаёт скрипт `python tests/make_samples.py`.

Запуск без Docker (нужен Redis на `localhost:6379`):

```bash
uvicorn main:app --app-dir mocks --port 8001 &
uvicorn app.main:app --port 8000
```
