### Название проекта: **`doc-orchestrator`**

Ниже готовое техническое задание и пошаговый промпт-план, написанный специально для AI-агента младшего уровня: с жёсткой структурой файлов, точными контрактами API, алгоритмами без двусмысленностей и инструкцией по написанию заглушек (mocks). Можешь копировать и отдавать агенту целиком.

---

# ТЗ и план разработки сервиса `doc-orchestrator` (для AI-агента)

## 1. Контекст и твоя задача
Ты — Python Backend-разработчик. Твоя задача — с нуля написать сервис оркестрации обработки документов **`doc-orchestrator`** на базе **FastAPI + Redis + ARQ (или Celery)**. 

Сервис принимает документы из двух источников (Open WebUI и корпоративные системы 1С/Teamcenter), определяет тип файла, прогоняет его через нужные OCR/VLM-движки с учётом жёстких лимитов видеопамяти (GPU Mutex), проверяет данные через DLP и валидаторы контрольных сумм, и возвращает результат (Markdown или строгий JSON).

**ВАЖНОЕ ПРАВИЛО ДЛЯ РАЗРАБОТКИ:** Реальных тяжёлых ML-контейнеров (`Docling`, `MinerU`, `PaddleOCR`, `Qwen3-VL`, `DLP`) сейчас нет. Ты обязан написать:
1. Полноценный боевой код оркестратора `doc-orchestrator` (с реальными HTTP-вызовами через `httpx`).
2. Отдельный легковесный FastAPI-сервис **`mock-engines`**, который имитирует ответы всех внешних систем по HTTP, чтобы весь пайплайн запускался через `docker-compose up` и проходил интеграционные тесты.

---

## 2. Технологический стек (использовать строго его)
* **Python 3.11+**
* **FastAPI + Uvicorn** (веб-фреймворк)
* **Pydantic v2 + pydantic-settings** (схемы и конфигурация через `.env`)
* **Redis** (`redis.asyncio` — для очередей задач, хранения статусов и распределённых GPU-блокировок)
* **httpx** (асинхронные HTTP-запросы к движкам и мокам)
* **PyMuPDF (`fitz`)** (определение наличия текстового слоя в PDF, рендеринг страниц PDF в PNG)
* **Pillow (`PIL`)** (нарезка кропов чертежей и ресайз по `max_pixels`)
* **pytest + pytest-asyncio** (автотесты)

---

## 3. Жёсткая структура репозитория
Создай проект строго по этой структуре (не выдумывай лишних вложенностей):

```text
doc-orchestrator/
├── docker-compose.yml
├── .env.example
├── Dockerfile
├── pyproject.toml (или requirements.txt)
├── app/
│   ├── main.py                  # Точка входа FastAPI, lifespan (подключение к Redis)
│   ├── config.py                # Настройки (URL всех сервисов, таймауты, лимиты пикселей)
│   ├── api/
│   │   ├── openwebui.py         # PUT /process (контракт Open WebUI)
│   │   └── enterprise.py        # POST /api/extract-structured, GET /api/tasks/{task_id}
│   ├── core/
│   │   ├── classifier.py        # Детектор: цифровой документ / скан / таблица / чертёж
│   │   ├── gpu_manager.py       # Redis Lock (Mutex) для GPU 0 и GPU 1 + выгрузка моделей
│   │   └── image_ops.py         # Рендер PDF -> PNG, ресайз max_pixels, кроп штампа ГОСТ
│   ├── clients/                 # HTTP-клиенты к внешним сервисам
│   │   ├── docling.py
│   │   ├── mineru.py
│   │   ├── paddle.py
│   │   ├── vlm_gateway.py       # Вызовы Qwen3-VL через Gateway
│   │   └── dlp.py               # Pre-check и Post-check в Regex-DLP
│   ├── schemas/                 # Pydantic-схемы документов для 1С и Teamcenter
│   │   ├── common.py
│   │   ├── upd.py               # УПД / Счёт-фактура
│   │   ├── invoice.py           # Счёт на оплату
│   │   └── drawing_stamp.py     # Основная надпись чертежа (ГОСТ / ЕСКД)
│   ├── validators/              # Пост-валидация (защита от галлюцинаций)
│   │   ├── checksums.py         # Алгоритмы проверки ИНН (10/12), КПП, БИК
│   │   └── math_checks.py       # Проверка сходимости: Цена * Кол-во + НДС == Итого
│   └── pipelines/
│       ├── markdown_pipe.py     # Логика сборки Markdown для Open WebUI
│       └── structured_pipe.py   # Логика сборки и валидации JSON для 1С/Teamcenter
├── mocks/                       # Сервис-заглушка всех внешних систем
│   ├── Dockerfile
│   └── main.py                  # Мок-эндпоинты для Docling, MinerU, Paddle, VLM, DLP
└── tests/
    ├── samples/                 # Сгенерированные тестовые PDF и PNG
    ├── test_validators.py
    ├── test_gpu_manager.py
    └── test_pipelines.py
```

---

## 4. Детальная логика работы модулей (как именно писать код)

### 4.1. Входные API-эндпоинты (`app/api/`)

1. **`PUT /process` (Интеграция с Open WebUI External Document Loader)**
   * **Заголовки:** принимает `X-Filename` (имя файла), `Content-Type`, опционально `Authorization: Bearer <token>`.
   * **Тело запроса:** сырые байты файла (`request.body()`).
   * **Действие:** синхронно ждёт выполнения `markdown_pipe` (внутри пайплайна задача становится в очередь к GPU).
   * **Ответ (200 OK):**
     ```json
     {
       "page_content": "# Заголовок\nРаспознанный текст и таблицы в Markdown...",
       "metadata": {
         "filename": "contract.pdf",
         "engine_used": "mineru+qwen3vl",
         "dlp_masked": false
       }
     }
     ```

2. **`POST /api/extract-structured` (Интеграция с 1С:ДО и Teamcenter)**
   * **Формат:** `multipart/form-data`.
   * **Поля:**
     * `file`: загружаемый файл (`UploadFile`).
     * `doc_type`: строка `enum` (`"upd"`, `"invoice"`, `"contract"`, `"drawing"`).
     * `async_mode`: `bool = False` (если `False` — ждём и отдаём результат сразу; если `True` — возвращаем `{"task_id": "uuid", "status": "queued"}`).
   * **Ответ (при `async_mode=False` или в `GET /api/tasks/{task_id}`):**
     ```json
     {
       "status": "completed",
       "doc_type": "upd",
       "data": { ...строгий JSON по Pydantic-схеме документа... },
       "validation": {
         "is_valid": false,
         "warnings": ["ИНН продавца 7701234569 не прошёл контрольную сумму", "Сумма по строкам (1000.00) != Итого (1200.00)"]
       }
     }
     ```
   * *Важно:* если валидация не прошла, сервис **НЕ падает с ошибкой 500**, а возвращает `200 OK`, заполняет `data` как есть, ставит `validation.is_valid = False` и пишет причины в `warnings` (чтобы в 1С документ ушёл на ручную верификацию человеку).

---

### 4.2. Классификатор файлов (`app/core/classifier.py`)
Напиши функцию `classify_file(file_bytes: bytes, filename: str, doc_type_hint: str | None) -> RouteType`:
1. Если расширение `.docx`, `.xlsx`, `.pptx`, `.html` -> вернуть `DIGITAL_OFFICE` (пойдёт в `Docling` на CPU).
2. Если `doc_type_hint == "drawing"` -> вернуть `DRAWING` (пойдёт в кроппер + `Qwen3-VL`).
3. Если `.jpg`, `.jpeg`, `.png`, `.tiff` -> вернуть `SCAN_IMAGE`.
4. Если `.pdf`: открой через `fitz.open(stream=file_bytes, filetype="pdf")`.
   * Посчитай среднее количество символов текста (`page.get_text().strip()`) на страницу.
   * Если в среднем `>= 100` печатных символов на страницу И нет `doc_type_hint in ("upd", "invoice")` со сканами -> вернуть `DIGITAL_PDF` (`Docling` на CPU).
   * Иначе (текста мало — это скан) -> вернуть `SCAN_PDF`.

---

### 4.3. GPU Mutex и управление памятью (`app/core/gpu_manager.py`)
В системе две видеокарты по 8 ГБ:
* **GPU 0:** делят между собой `MinerU` (основной) и `PaddleOCR-VL` (тяжёлые таблицы/сканы). **Они НЕ МОГУТ работать в VRAM одновременно!**
* **GPU 1:** монопольно занят `Qwen3-VL` (`NUM_PARALLEL=1`).

Реализуй класс `GPUResourceManager` поверх Redis:
1. Ключ в Redis `gpu0:active_model` хранит имя загруженной модели (`"mineru"` или `"paddle"` или `"none"`).
2. Распределённый лок `redis.lock("lock:gpu0", timeout=300)` гарантирует, что на GPU 0 выполняется ровно 1 задача в единицу времени.
3. Алгоритм захвата GPU 0 `async with gpu_manager.acquire_gpu0(target_model="paddle")`:
   * Захватываем `lock:gpu0`.
   * Читаем `gpu0:active_model`.
   * Если там `"mineru"`, а нужен `"paddle"` -> отправляем `POST {MINERU_URL}/unload` (освободить VRAM), затем `POST {PADDLE_URL}/load`, записываем в Redis `"paddle"`.
   * Выполняем задачу.
   * Обновляем таймер неактивности (`gpu0:last_used_ts`). Фоновая корутина раз в 30 секунд проверяет: если на GPU 0 загружен `"paddle"` и он простаивает дольше `TTL=120` секунд -> захватывает лок, вызывает `POST {PADDLE_URL}/unload` и возвращает дефолтный `"mineru"`.
4. Для GPU 1 (`Qwen3-VL`) реализуй строгий семафор/лок `lock:gpu1` (`timeout=300`), чтобы запросы шли строго по одному (`1 поток`), иначе будет таймаут или OOM.

---

### 4.4. Препроцессинг картинок и чертежей (`app/core/image_ops.py`)
1. `render_pdf_to_images(pdf_bytes, dpi=200) -> list[bytes]` — конвертация страниц PDF в PNG через `PyMuPDF`.
2. `resize_max_pixels(img_bytes: bytes, max_pixels: int = 1_280_000) -> bytes`:
   * Если `width * height > max_pixels`, пропорционально уменьши картинку (`scale = sqrt(max_pixels / (width * height))`) через `Image.Resampling.LANCZOS`, чтобы `Qwen3-VL` не вылез за 8 ГБ VRAM.
3. `crop_drawing_zones(img_bytes: bytes) -> dict[str, bytes]` (для чертежей ЕСКД/ГОСТ):
   * Вырежи **штамп (основную надпись)**: правый нижний угол изображения — прямоугольник `x: от 55% до 100% ширины`, `y: от 75% до 100% высоты`.
   * Вырежи **зону технических требований** (над штампом): `x: от 55% до 100% ширины`, `y: от 35% до 78% высоты`.
   * Прогони оба кропа через `resize_max_pixels` и верни словарём `{"stamp": bytes, "tech_req": bytes}`.

---

### 4.5. Два боевых пайплайна (`app/pipelines/`)

#### Пайплайн А: `markdown_pipe.py` (для Open WebUI)
1. Классифицируем файл.
2. Если `DIGITAL_OFFICE` или `DIGITAL_PDF` -> отправляем в `Docling` (CPU, без GPU-локов) -> получаем Markdown.
3. Если `SCAN_PDF` / `SCAN_IMAGE`:
   * Если документ помечен как сложная таблица/форма -> берём `acquire_gpu0("paddle")` -> получаем Markdown-таблицы из `PaddleOCR-VL`.
   * Иначе -> берём `acquire_gpu0("mineru")` -> отправляем в `MinerU`. `MinerU` возвращает Markdown с плейсхолдерами картинок `![fig_1](fig_1.png)` и массив вырезанных картинок (base64).
   * Для каждой вырезанной картинки `fig_N`: делаем `DLP pre-check` промпта -> берём `acquire_gpu1()` -> отправляем картинку в `Qwen3-VL` с промптом *"Опиши схему текстом или переведи блок-схему в синтаксис Mermaid"* -> вставляем описание/Mermaid в Markdown вместо `![fig_1]`.
4. Прогоняем собранный Markdown через `DLP post-check` (`POST {DLP_URL}/inspect`). Если DLP замаскировал секреты/ПДн — сохраняем замаскированный текст и ставим `dlp_masked=True`.
5. Возвращаем результат.

#### Пайплайн Б: `structured_pipe.py` (для 1С и Teamcenter)
1. Если `doc_type == "drawing"`:
   * Рендерим 1-ю страницу чертежа в высоком разрешении (`dpi=300`), делаем `crop_drawing_zones`.
   * Через `acquire_gpu1()` отправляем кроп `stamp` в `Qwen3-VL` (передав JSON Schema модели `DrawingStampSchema` в поле `format`).
   * Вторым запросом отправляем кроп `tech_req` для извлечения списка техтребований.
   * Объединяем, прогоняем через `DLP post-check`, отдаём JSON.
2. Если `doc_type in ("upd", "invoice", "contract")`:
   * Шаг 1 (OCR): прогоняем документ через `PaddleOCR-VL` (если скан с таблицами, через `acquire_gpu0("paddle")`) или `Docling` (если цифровой), чтобы получить точный сырой текст и структуру таблиц.
   * Шаг 2 (Структурирование): отправляем сырой текст + ресайзнутое изображение 1-й страницы в `Qwen3-VL` (через `DLP pre-check` -> `Gateway` -> `acquire_gpu1()`), передав строгую JSON Schema (`UpdSchema` или `InvoiceSchema`).
   * Шаг 3 (Валидация): парсим ответ через Pydantic и прогоняем через `validators/checksums.py` и `validators/math_checks.py`. Все расхождения складываем в `validation.warnings`.
   * Шаг 4: `DLP post-check` и возврат ответа.

---

### 4.6. Пост-валидаторы (`app/validators/`) — написать реальные алгоритмы!
Никаких заглушек здесь, напиши чистую математику на Python:
1. **`validate_inn(inn: str) -> bool`**:
   * Проверка длины (10 цифр для юрлиц, 12 для ИП) и что все символы — цифры.
   * Расчёт контрольных цифр по официальным весовым коэффициентам ФНС РФ (`[2, 4, 10, 3, 5, 9, 4, 6, 8]` для 10-значного; две контрольные суммы для 12-значного).
2. **`validate_kpp(kpp: str) -> bool`** (9 символов, формат `\d{4}[\dA-Z]{2}\d{3}`).
3. **`validate_bik(bik: str) -> bool`** (9 цифр, начинается на `04`).
4. **`validate_invoice_math(data: UpdSchema | InvoiceSchema) -> list[str]`**:
   * Для каждой строки товара: `abs(qty * price - amount_without_vat) <= 0.05` (допуск на копейки округления).
   * `abs(amount_without_vat + vat_amount - row_total) <= 0.05`.
   * Сумма `row_total` всех строк должна сходиться с `document_total` (`<= 0.05`).
   * Возвращает список понятных строк с ошибками (или пустой список, если всё сошлось).

---

## 5. Спецификация сервиса заглушек (`mocks/main.py`)
Создай отдельное FastAPI-приложение на порту `8001`, которое имитирует все внешние сервисы (чтобы `doc-orchestrator` общался с ним по сети как с настоящими):

1. `POST /docling/convert` -> спит `0.1с`, возвращает `{"markdown": "# Документ (Docling)\nТекст цифрового документа."}`.
2. `POST /mineru/load` и `POST /mineru/unload` -> меняет внутреннюю переменную стейта мока `mineru_loaded: bool`. Если вызвать `/mineru/extract`, когда одновременно в моке `paddle_loaded == True`, мок **ДОЛЖЕН выбрасывать `HTTP 500 Out Of Memory (Simulated)`!** *(Это нужно, чтобы автотесты реально проверили, что твой `gpu_manager` не забывает выгружать модель перед запуском другой!)*
3. `POST /mineru/extract` -> возвращает Markdown с `![fig_1](fig_1.png)` и крошечный 1x1 PNG в base64.
4. `POST /paddle/load`, `POST /paddle/unload`, `POST /paddle/extract` -> аналогично проверяет отсутствие конфликта VRAM с `mineru` и возвращает Markdown-таблицу УПД.
5. `POST /gateway/v1/chat/completions` (эмуляция Gateway + Ollama/Qwen3-VL):
   * Проверяет, что параллельно не идёт второй запрос к нему (используй `asyncio.Lock`: если лок уже занят — кидай `HTTP 429 Too Many Concurrent Requests on GPU 1`, чтобы проверить семафор в оркестраторе!).
   * Смотрит на переданную JSON Schema или промпт:
     * если просят описание картинки -> возвращает блок кода ```` ```mermaid\ngraph TD; A-->B;\n``` ````;
     * если передана схема `UpdSchema` -> возвращает реалистичный JSON УПД (добавь возможность через спец-имя файла `bad_inn.pdf` возвращать битый ИНН и несходящуюся сумму для проверки валидаторов);
     * если схема `DrawingStampSchema` -> возвращает JSON штампа `{"designation": "АБВГ.123456.001", "title": "Вал приводной", "material": "Сталь 45 ГОСТ 1050-2013", "mass": "12.5"}`.
6. `POST /dlp/inspect` -> ищет регуляркой слово `SECRET_TOKEN` или тестовый номер паспорта, заменяет на `[REDACTED]` и возвращает `{"text": ..., "masked": true/false}`.

---

## 6. Пошаговый план выполнения работы (следуй строго по шагам)

Не пиши весь код одним куском. Двигайся итерациями и проверяй каждый шаг:

* **Шаг 1. Каркас и конфигурация.**
  Создай дерево директорий, `pyproject.toml`/`requirements.txt`, `.env.example`, `app/config.py`. Настрой `docker-compose.yml` с тремя контейнерами: `redis:7-alpine`, `mock-engines` (порт 8001), `doc-orchestrator` (порт 8000).
* **Шаг 2. Реализация `mock-engines`.**
  Напиши `mocks/main.py` со всеми эндпоинтами из раздела 5, включая симуляцию краша по OOM на GPU 0 и ошибки параллелизма 429 на GPU 1.
* **Шаг 3. Схемы и математические валидаторы.**
  Напиши Pydantic-схемы (`upd.py`, `invoice.py`, `drawing_stamp.py`) и валидаторы (`checksums.py`, `math_checks.py`). Сразу напиши `tests/test_validators.py` (проверь валидный/невалидный ИНН 10 и 12 цифр, расхождение копеек и НДС) и добейся зелёного `pytest`.
* **Шаг 4. Классификатор и обработка изображений.**
  Реализуй `classifier.py` (на `PyMuPDF`) и `image_ops.py` (рендер PDF, ресайз `max_pixels`, кроп штампа чертежа). Сгенерируй скриптом в `tests/samples/` 3 тестовых файла: PDF с текстом, PDF с пустой картинкой (скан) и большой PNG чертежа `3000x2000`. Проверь кропы и ресайз тестами.
* **Шаг 5. GPU Resource Manager на Redis.**
  Реализуй `gpu_manager.py` (мьютексы GPU 0 / GPU 1, переключение `mineru <-> paddle`, фоновый воркер возврата по TTL). Напиши HTTP-клиенты в `app/clients/`.
* **Шаг 6. Сборка пайплайнов и API.**
  Собери `markdown_pipe.py`, `structured_pipe.py` и подключи роуты `PUT /process` и `POST /api/extract-structured` (+ `GET /api/tasks/{task_id}`) в `app/main.py`.
* **Шаг 7. Интеграционное стресс-тестирование.**
  Напиши `tests/test_pipelines.py`, который:
  1. Отправляет запрос в `PUT /process` со сканом -> проверяет вклейку Mermaid и работу DLP.
  2. Отправляет `POST /api/extract-structured` с хорошим УПД (`is_valid=True`) и с битым УПД (`is_valid=False`, непустой `warnings`).
  3. Отправляет **5 одновременных запросов** (`asyncio.gather`) с разными типами задач (`mineru`, `paddle`, `drawing`), чтобы доказать, что `gpu_manager` корректно выстраивает очередь и ни разу не ловит `500 OOM` или `429` от мок-сервиса!

---

## 7. Критерии приёмки (Definition of Done)
Твоя работа считается выполненной только тогда, когда:
1. Команда `docker compose up --build -d` поднимает все 3 сервиса без ошибок.
2. Команда `pytest -v` проходит на **100% без падений и пропусков**.
3. В коде нет `TODO`, `pass` вместо логики или захардкоженных URL (всё берётся из `config.py` / переменных окружения, чтобы в проде мы просто поменяли URL с `http://mock-engines:8001/...` на реальные контейнеры).
4. Есть краткий `README.md` с примерами `curl`-запросов для Open WebUI и для 1С.