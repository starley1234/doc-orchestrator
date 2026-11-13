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
на параллельные запросы к GPU 1. Как поставить настоящие движки и подключить к ним сервис —
в разделе [«Боевой сервер»](#боевой-сервер).

## Запуск тестового стенда

```bash
docker compose up --build -d       # redis, mock-engines (:8001), doc-orchestrator (:8000)
curl http://localhost:8000/health  # {"status":"ok"}
```

Настраивать ничего не нужно. Описание API — <http://localhost:8000/docs>. Что загружено
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

Сервер разворачивается по [`ИНСТРУКЦИЯ_ПО_УСТАНОВКЕ_OCR.md`](ИНСТРУКЦИЯ_ПО_УСТАНОВКЕ_OCR.md): Debian 13,
две видеокарты по 8 ГБ, драйвер NVIDIA 595, Docker, движки Docling, MinerU, PaddleOCR-VL и Ollama.
Шаги инструкции и подключение оркестратора выполняет скрипт:

```bash
git clone https://github.com/starley1234/doc-orchestrator.git && cd doc-orchestrator
sudo bash deploy/install.sh check   # ничего не меняет: что уже установлено и что делать дальше
sudo bash deploy/install.sh all     # всё по порядку; после установки драйвера попросит перезагрузку
```

| Команда | Что делает | Раздел инструкции |
|---|---|---|
| `check` | показывает ОС, диск, видеокарты и процессы на них, драйвер, Docker, контейнеры; называет следующий шаг | — |
| `driver` | заменяет драйвер из Debian на ветку 595 из репозитория NVIDIA; после него нужна перезагрузка | шаг 2 |
| `docker` | ставит Docker CE и NVIDIA Container Toolkit, проверяет, что из контейнера видны обе видеокарты | шаг 3 |
| `mineru` | собирает образ `mineru:4` — MinerU 4.2.1 с моделями внутри | шаг 4 |
| `stack` | создаёт сеть `doc-net`, кладёт `/opt/ai-doc-stack/docker-compose.yml` и запускает движки | шаги 5–6 |
| `models` | скачивает модели Qwen3-VL и создаёт профили `qwen3-vl-doc`, `qwen3-vl-hq` | шаг 6 |
| `verify` | отправляет каждому движку образец с русским текстом и проверяет распознанное | шаг 7 |
| `orchestrator` | создаёт `.env` с адресами и случайным ключом, запускает `doc-orchestrator` и шлюз движков | п. 9.3 |
| `status`, `gpu0 mineru\|paddle` | сводка по контейнерам и видеопамяти; ручное переключение GPU 0 (без оркестратора) | п. 9.1 |

Любой шаг можно запускать отдельно и повторно — сделанное он не переделывает. `--dry-run` показывает
команды, ничего не выполняя; `--yes` отвечает «да» на вопросы (удаление старого драйвера). Журнал —
`/var/log/ai-doc-stack-install.log`. Шаг 1 инструкции (остановить старый софт на видеокартах) остаётся
за администратором: занятые видеокарты покажет `check`.

Скрипт и шлюз проверены автотестами на имитациях сервера (apt, Docker, API движков), но на настоящем
сервере с GPU не запускались — начните с `check` и `--dry-run`.

### Что получается

```
Open WebUI, 1С:ДО, Teamcenter
        │  :8000
        ▼
 doc-orchestrator ──► doc-redis                       очередь к GPU, состояние задач
        │
        ├──► engine-gateway ──► docling-serve         CPU:   цифровые документы
        │         │        ├──► mineru-api            GPU 0: сканы
        │         │        └──► paddle-gpu0           GPU 0: сканы с печатями — вместо MinerU
        │         └──► сокет Docker                   запуск и остановка контейнеров GPU 0
        └──► ollama-vlm                               GPU 1: Qwen3-VL (схемы, JSON, чертежи)
```

Движки и Redis запускает `/opt/ai-doc-stack/docker-compose.yml` (исходник — `deploy/ai-doc-stack/`),
оркестратор и шлюз — `docker-compose.prod.yml`; все контейнеры видят друг друга по именам в сети `doc-net`.

### Шлюз движков

Оркестратор ждёт от движков те же HTTP-контракты, что у `mock-engines`. У штатных серверов API другой,
а команды «выгрузить модель из видеопамяти» нет вовсе, поэтому между ними стоит `engine-gateway`
(каталог `gateway/`, тот же образ, что у оркестратора):

| Оркестратор вызывает | Шлюз выполняет |
|---|---|
| `POST {DOCLING_URL}/convert`, поле `file` → `{"markdown": "…"}` | docling-serve v1.38.2: `POST /v1/convert/file` |
| `POST {MINERU_URL}/extract` → `{"markdown": "… ![](images/a.jpg) …", "figures": {"images/a.jpg": "<base64>"}}` | MinerU 4.2.1: `POST /v1/tasks`, загрузка файла, опрос, разбор ZIP с результатом |
| `POST {PADDLE_URL}/extract` → `{"markdown": "…"}` | paddleocr-pdf-api v0.5.0: `POST /ocr`, опрос, `GET /ocr/{id}/result`, удаление задачи |
| `POST {MINERU_URL}/load`, `POST {PADDLE_URL}/load` | запуск контейнера `mineru-api` или `paddle-gpu0` и ожидание, пока сервер ответит; сервер, который отвечает ошибкой, перезапускается |
| `POST …/unload` | остановка контейнера: только так у этих серверов освобождается видеопамять |
| `POST {OLLAMA_URL}/api/chat` | — (родной API Ollama, шлюз не нужен) |

Смена модели на GPU 0 — это перезапуск контейнера и занимает от десятков секунд до пары минут:
MinerU и PaddleOCR-VL в 8 ГБ вдвоём не помещаются. После простоя PaddleOCR-VL (`PADDLE_IDLE_TTL_SEC`)
оркестратор сам возвращает на GPU 0 MinerU. Пока оркестратор работает, контейнеры `mineru-api` и
`paddle-gpu0` вручную не трогайте: документ для PaddleOCR-VL отправляют с заголовком `X-OCR-Engine: paddle`.

Если движок упал посреди работы (например, не хватило видеопамяти на большой странице), запрос
получит ошибку `502`, а перед следующим запросом шлюз перезапустит контейнер. Что сейчас запущено
и сколько занято видеопамяти, показывает `sudo bash deploy/install.sh status`.

Из картинок, которые MinerU вырезал из скана, шлюз отдаёт на описание Qwen3-VL не больше 12 самых
крупных и не мельче 100 px (`MINERU_MAX_FIGURES`, `MINERU_MIN_FIGURE_SIDE`): каждая — несколько секунд GPU 1.

Доступ к сокету Docker равен правам root на сервере. Поэтому он выдан шлюзу, порт которого наружу
не публикуется, а не оркестратору, который разбирает чужие файлы.

### Настройки

`deploy/install.sh orchestrator` создаёт `.env` из `.env.example`, подставляя адреса боевого сервера:

```bash
REDIS_URL=redis://doc-redis:6379/0
DOCLING_URL=http://engine-gateway:8001/docling
MINERU_URL=http://engine-gateway:8001/mineru
PADDLE_URL=http://engine-gateway:8001/paddle
OLLAMA_URL=http://ollama-vlm:11434
VLM_MODEL=qwen3-vl-doc          # профиль qwen3-vl:4b-instruct-q8_0 с num_ctx 8192
API_KEY=<случайный ключ>        # его же указать в Open WebUI
```

Вручную то же самое: `cp .env.example .env`, заменить эти значения и выполнить
`docker compose -f docker-compose.prod.yml up -d --build`. Остальные переменные (таймауты, лимиты) описаны
в `.env.example`; настройки стека движков (уровень MinerU, адрес для портов) — в `/opt/ai-doc-stack/.env`.

В Open WebUI: `Admin Panel → Settings → Documents` → `Content Extraction Engine` = `External`, адрес
`http://<IP сервера>:8000`, `API Key` — значение `API_KEY`, в `Supported Media MIME Types` добавить `image/*`.

Проверка в любой момент: `sudo bash deploy/install.sh verify` — или напрямую
`python3 deploy/smoke_test.py --orchestrator http://localhost:8000 --api-key <ключ> --with-paddle`.

## Как это устроено

Каталоги: `app/` — оркестратор, `mocks/` — заглушка движков для тестового стенда, `gateway/` — шлюз
к настоящим движкам, `deploy/` — установка на сервер (скрипт, compose-файл стека, образцы для проверки).

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
тестовые PDF и PNG создаются на лету. Шлюз проверяется на имитациях API docling-serve, MinerU,
paddleocr-pdf-api и Docker (`tests/stock_engines.py`), а `deploy/install.sh` — на воображаемом сервере
с подменёнными apt, Docker и nvidia-smi (`tests/fake_server.py`; нужен `bash`).

Запуск без Docker (нужен Redis на `localhost:6379`):

```bash
uvicorn main:app --app-dir mocks --port 8001 &
REDIS_URL=redis://localhost:6379/0 DOCLING_URL=http://localhost:8001/docling \
  MINERU_URL=http://localhost:8001/mineru PADDLE_URL=http://localhost:8001/paddle \
  OLLAMA_URL=http://localhost:8001 uvicorn app.main:app --port 8000
```
