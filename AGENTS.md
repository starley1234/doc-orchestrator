### Название проекта: **`doc-orchestrator`**

Ниже готовое техническое задание и пошаговый промпт-план для AI-агента: с жёсткой структурой файлов, точными контрактами API, алгоритмами без двусмысленностей и инструкцией по написанию заглушек (mocks).

---

# ТЗ и план разработки сервиса `doc-orchestrator` (для AI-агента)

## 1. Контекст и твоя задача
Ты — Python Backend-разработчик. Твоя задача — с нуля написать сервис оркестрации распознавания документов **`doc-orchestrator`** на базе **FastAPI + Redis + asyncio**.

Сервис работает на выделенном сервере с 2× GPU по 8 ГБ VRAM, принимает документы из двух контуров (Open WebUI и корпоративные системы 1С:ДО / Teamcenter), определяет тип файла, прогоняет его через нужные OCR/VLM-движки с учётом жёстких лимитов видеопамяти (GPU Mutex), проверяет реквизиты математическими валидаторами и возвращает результат (обогащённый Markdown или строгий JSON). Без DLP и промежуточных API-шлюзов — фокус на скорости, стабильности VRAM и качестве распознавания.

**ВАЖНОЕ ПРАВИЛО ДЛЯ РАЗРАБОТКИ:** Реальных тяжёлых ML-контейнеров (`Docling-serve`, `MinerU`, `PaddleOCR-VL`, `Ollama / Qwen3-VL`) в тестовом окружении нет. Ты обязан написать:
1. Полноценный боевой код оркестратора `doc-orchestrator` (с реальными асинхронными HTTP-вызовами через `httpx`).
2. Отдельный легковесный FastAPI-сервис **`mock-engines`**, который имитирует ответы всех ML-движков по HTTP (включая симуляцию OOM и конкурентных ошибок GPU), чтобы весь пайплайн запускался одной командой `docker compose up` и проходил интеграционные стресс-тесты.

---

## 2. Технологический стек (использовать строго его)
* **Python 3.11+**
* **FastAPI + Uvicorn + python-multipart** (веб-фреймворк и загрузка файлов)
* **Pydantic v2 + pydantic-settings** (схемы документов и конфигурация через `.env`)
* **Redis** (`redis.asyncio` — для распределённых GPU-блокировок, хранения состояния активной модели и статусов фоновых задач)
* **httpx** (асинхронные HTTP-запросы к движкам с увеличенным таймаутом `timeout=300.0`)
* **PyMuPDF (`fitz`)** (определение наличия текстового слоя в PDF, рендеринг страниц PDF в PNG с нужным DPI)
* **Pillow (`PIL`)** (нарезка кропов чертежей ЕСКД и smart-resize кратно 32 px по `max_pixels`)
* **pytest + pytest-asyncio** (автотесты)

---

## 3. Жёсткая структура репозитория
Создай проект строго по этой структуре (не выдумывай лишних вложенностей):

```text
doc-orchestrator/
├── docker-compose.yml
├── .env.example
├── Dockerfile
├── requirements.txt
├── README.md
├── app/
│   ├── main.py                  # Точка входа FastAPI, lifespan (Redis, HTTP-клиент, фоновый TTL-воркер)
│   ├── config.py                # Настройки (URL движков, модель VLM, таймауты, лимиты пикселей, API-ключ)
│   ├── api/
│   │   ├── openwebui.py         # PUT /process (контракт Open WebUI External Document Loader)
│   │   └── enterprise.py        # POST /api/extract-structured, GET /api/tasks/{task_id}
│   ├── core/
│   │   ├── classifier.py        # Детектор: офисный файл / цифровой PDF / скан / чертёж
│   │   ├── gpu_manager.py       # Redis Lock (Mutex) для GPU 0 и GPU 1 + переключение и TTL моделей
│   │   └── image_ops.py         # Рендер PDF -> PNG, smart-resize (кратно 32 px), кроп штампа ЕСКД
│   ├── clients/                 # Асинхронные HTTP-клиенты к ML-движкам
│   │   ├── docling.py           # Вызовы Docling-serve (CPU)
│   │   ├── mineru.py            # Вызовы MinerU (GPU 0: load / unload / extract)
│   │   ├── paddle.py            # Вызовы PaddleOCR-VL-1.5 (GPU 0: load / unload / extract)
│   │   └── vlm.py               # Вызовы Ollama / Qwen3-VL Instruct (GPU 1: /api/chat)
│   ├── schemas/                 # Pydantic-схемы документов для 1С:ДО и Teamcenter
│   │   ├── common.py
│   │   ├── upd.py               # УПД / Счёт-фактура
│   │   ├── invoice.py           # Счёт на оплату
│   │   └── drawing_stamp.py     # Основная надпись чертежа (ГОСТ / ЕСКД) + ТТ
│   ├── validators/              # Пост-валидация (защита от галлюцинаций цифр)
│   │   ├── checksums.py         # Алгоритмы проверки ИНН (10/12), КПП, БИК
│   │   └── math_checks.py       # Проверка сходимости: Цена * Кол-во + НДС == Итого
│   └── pipelines/
│       ├── markdown_pipe.py     # Сборка обогащённого Markdown (текст + таблицы + Mermaid)
│       └── structured_pipe.py   # Двухшаговая сборка и валидация JSON для 1С/Teamcenter
├── mocks/                       # Сервис-заглушка всех ML-движков
│   ├── Dockerfile
│   └── main.py                  # Мок-эндпоинты для Docling, MinerU, PaddleOCR-VL, Ollama VLM
└── tests/
    ├── conftest.py              # Генерация тестовых PDF/PNG на лету и фикстуры
    ├── test_validators.py
    ├── test_image_ops.py
    └── test_pipelines.py
```

---

## 4. Детальная логика работы модулей (как именно писать код)

### 4.1. Входные API-эндпоинты (`app/api/`)

1. **`PUT /process` (Интеграция с Open WebUI External Document Loader)**
   * **Заголовки:**
     * `X-Filename` (имя файла, например `contract.pdf`; может быть URL-encoded),
     * `Content-Type`,
     * `Authorization: Bearer <token>` (если в `config.py` задан непустой `API_KEY`, проверять совпадение, иначе возвращать `401`),
     * опционально `X-OCR-Engine: auto | mineru | paddle` (по умолчанию `auto`; если `paddle` — принудительно использовать `PaddleOCR-VL` для тяжёлых сканов с печатями).
   * **Тело запроса:** сырые байты файла (`await request.body()`).
   * **Действие:** синхронно ждёт выполнения `markdown_pipe`.
   * **Ответ (200 OK):**
     ```json
     {
       "page_content": "# Заголовок\nРаспознанный текст, HTML-таблицы и блоки Mermaid...",
       "metadata": {
         "filename": "contract.pdf",
         "route": "SCAN_PDF",
         "engine_used": "mineru+qwen3-vl"
       }
     }
     ```

2. **`POST /api/extract-structured` (Интеграция с 1С:ДО и Teamcenter)**
   * **Формат:** `multipart/form-data`.
   * **Поля формы:**
     * `file`: загружаемый файл (`UploadFile`).
     * `doc_type`: строка `Literal["upd", "invoice", "drawing"]`.
     * `has_seals`: `bool = False` (если `True` на скане — использовать `PaddleOCR-VL` для чтения круглых печатей и сложных таблиц).
     * `async_mode`: `bool = False`.
   * **Логика `async_mode`:**
     * Если `async_mode=False` — ждём завершения `structured_pipe` и сразу отдаём итоговый JSON (`200 OK`).
     * Если `async_mode=True` — генерируем `task_id = str(uuid4())`, сохраняем в Redis ключ `task:{task_id}` со статусом `"queued"` (TTL 3600 сек), запускаем обработку через `asyncio.create_task(...)` (обязательно сохраняя ссылку на task в `set`, чтобы его не удалил GC!) и мгновенно возвращаем `202 Accepted`: `{"task_id": task_id, "status": "queued"}`.
   * **Эндпоинт `GET /api/tasks/{task_id}`:** читает `task:{task_id}` из Redis (`404`, если ключ не найден).
   * **Формат готового ответа:**
     ```json
     {
       "status": "completed",
       "doc_type": "upd",
       "data": { ...строгий JSON по Pydantic-схеме документа... },
       "validation": {
         "is_valid": false,
         "warnings": [
           "ИНН продавца 7701234569 не прошёл контрольную сумму",
           "Сумма по строкам (1000.00) != Итого по документу (1200.00)"
         ]
       }
     }
     ```
   * *Важно:* если математическая валидация или проверка ИНН не прошла, сервис **НЕ падает с ошибкой 422/500**, а возвращает `200 OK`, отдаёт распознанный `data`, ставит `validation.is_valid = False` и перечисляет причины в `warnings` (чтобы в 1С документ ушёл оператору на ручную проверку).

---

### 4.2. Классификатор файлов (`app/core/classifier.py`)
Реализуй `enum RouteType(str, Enum)`: `DIGITAL_OFFICE`, `DIGITAL_PDF`, `SCAN_PDF`, `SCAN_IMAGE`, `DRAWING`.
Напиши функцию `classify_file(file_bytes: bytes, filename: str, doc_type_hint: str | None = None) -> RouteType`:
1. Если `doc_type_hint == "drawing"` -> вернуть `DRAWING`.
2. Определи расширение `filename.lower()`:
   * `.docx`, `.xlsx`, `.pptx`, `.html`, `.md`, `.txt` -> вернуть `DIGITAL_OFFICE` (пойдёт в `Docling` на CPU).
   * `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.bmp`, `.webp` -> вернуть `SCAN_IMAGE`.
3. Если `.pdf`: открой через `fitz.open(stream=file_bytes, filetype="pdf")`.
   * Если страниц 0 -> выбросить `ValueError("Empty PDF")`.
   * Посчитай среднее количество непробельных символов (`len("".join(page.get_text().split()))`) на страницу.
   * Если в среднем `>= 100` символов на страницу -> вернуть `DIGITAL_PDF` (`Docling` на CPU).
   * Иначе (текста мало или нет — это скан) -> вернуть `SCAN_PDF`.

---

### 4.3. GPU Mutex и управление видеопамятью (`app/core/gpu_manager.py`)
В сервере две видеокарты RTX 2070 SUPER по 8 ГБ:
* **GPU 0:** делят между собой `MinerU` и `PaddleOCR-VL-1.5`. **Они НЕ МОГУТ находиться в VRAM одновременно!**
* **GPU 1:** монопольно занят `Qwen3-VL Instruct` в Ollama (`OLLAMA_NUM_PARALLEL=1`). Запросы к нему должны идти строго по одному.

Реализуй класс `GPUResourceManager` поверх `redis.asyncio`:
1. Ключи в Redis:
   * `gpu0:active_model` — текущая модель в VRAM (`"none"`, `"mineru"` или `"paddle"`). При старте, если ключа нет, считать `"none"`.
   * `gpu0:last_used_ts` — float unix timestamp последнего завершения задачи на GPU 0.
2. Распределённые блокировки (обязательно указывай `timeout=300, blocking_timeout=300`!):
   * `redis.lock("lock:gpu0", timeout=300, blocking_timeout=300)`
   * `redis.lock("lock:gpu1", timeout=300, blocking_timeout=300)`
3. Контекстный менеджер `async with gpu_manager.acquire_gpu0(target_model: Literal["mineru", "paddle"])`:
   * Захватывает `lock:gpu0`.
   * Читает `active = await redis.get("gpu0:active_model") or "none"`.
   * Если `active != target_model`:
     * Если `active == "mineru"` -> вызывает `await mineru_client.unload()`.
     * Если `active == "paddle"` -> вызывает `await paddle_client.unload()`.
     * Вызывает `load()` у `target_model` (`mineru_client.load()` или `paddle_client.load()`).
     * Записывает `await redis.set("gpu0:active_model", target_model)`.
   * В блоке `try ... finally` отдаёт управление вызывающему коду (`yield`), а при выходе обновляет `gpu0:last_used_ts = time.time()`.
4. Фоновая задача `idle_watcher` (запускается в `lifespan` FastAPI, интервал проверки настраиваемый через `GPU0_CHECK_INTERVAL_SEC`, по умолчанию `15` сек, а `PADDLE_IDLE_TTL_SEC` — `120` сек):
   * Если `gpu0:active_model == "paddle"` и `time.time() - last_used_ts > PADDLE_IDLE_TTL_SEC`:
   * Пытается неблокирующе захватить `lock:gpu0` (`blocking=False`). Если захватила — перепроверяет условие, вызывает `paddle_client.unload()`, затем `mineru_client.load()`, ставит `gpu0:active_model = "mineru"`.
5. Контекстный менеджер `async with gpu_manager.acquire_gpu1()`:
   * Захватывает `lock:gpu1` и гарантирует строго последовательное обращение к `Qwen3-VL`.

---

### 4.4. Препроцессинг картинок и чертежей (`app/core/image_ops.py`)
1. `render_pdf_to_images(pdf_bytes: bytes, dpi: int = 200, max_pages: int | None = None) -> list[bytes]` — рендерит страницы PDF в PNG-байты через `PyMuPDF` (`page.get_pixmap(dpi=dpi).tobytes("png")`).
2. `smart_resize(img_bytes: bytes, max_pixels: int = 1_280_000) -> bytes`:
   * Открывает картинку через `Pillow` и переводит в `RGB`.
   * Если `width * height > max_pixels`, вычисляет `scale = math.sqrt(max_pixels / (width * height))`, новые размеры `new_w = width * scale`, `new_h = height * scale`.
   * Округляет обе стороны до ближайшего числа, **кратного 32 px** (минимум `32` px: `max(32, int(new_w // 32) * 32)`), с ресемплингом `Image.Resampling.LANCZOS`. Это требование визуального энкодера `Qwen3-VL` для экономии VRAM на 8 ГБ картах.
   * Возвращает PNG-байты.
3. `crop_drawing_zones(img_bytes: bytes) -> dict[str, bytes]` (для чертежей ЕСКД / ГОСТ 2.104):
   * Открывает рендер страницы (сделанный в `300 DPI`).
   * Учитывает ориентацию листа (так как штамп 185×55 мм на книжном А4 занимает почти всю ширину листа, а на альбомных А3–А0 — правый угол):
     * **Если книжная ориентация (`width < height`, формат А4):**
       * `stamp`: `x` от `0%` до `100%`, `y` от `72%` до `100%`.
       * `tech_req` (зона ТТ над штампом): `x` от `0%` до `100%`, `y` от `35%` до `74%`.
     * **Если альбомная ориентация (`width >= height`, форматы А3–А0):**
       * `stamp`: `x` от `50%` до `100%`, `y` от `70%` до `100%`.
       * `tech_req`: `x` от `50%` до `100%`, `y` от `30%` до `72%`.
   * Прогоняет оба кропа через `smart_resize` и возвращает `{"stamp": bytes, "tech_req": bytes}`.

---

### 4.5. Контракт клиента `vlm.py` (совместимый с реальным Ollama)
В `config.py` задаются:
* `OLLAMA_URL = "http://mock-engines:8001"` (в проде поменяется на `http://ollama:11434`),
* `VLM_MODEL = "qwen3-vl:4b-instruct-q8_0"` (строго тег `-instruct`, не `thinking`!).

Клиент отправляет `POST {OLLAMA_URL}/api/chat` с телом:
```json
{
  "model": "qwen3-vl:4b-instruct-q8_0",
  "messages": [
    {
      "role": "user",
      "content": "Текст промпта...",
      "images": ["<base64_png_без_префикса_data_image>"]
    }
  ],
  "format": { ...опционально JSON Schema из Model.model_json_schema()... },
  "stream": false,
  "options": {
    "num_ctx": 8192,
    "temperature": 0.0
  }
}
```
Ответ читается из `response.json()["message"]["content"]`.

---

### 4.6. Два боевых пайплайна (`app/pipelines/`)

#### Пайплайн А: `markdown_pipe.py` (для Open WebUI)
1. Классифицируем файл через `classify_file`.
2. Если `DIGITAL_OFFICE` или `DIGITAL_PDF` -> вызываем `docling_client.convert(file_bytes, filename)` (CPU, без GPU-локов) -> возвращаем полученный Markdown.
3. Если `SCAN_PDF` или `SCAN_IMAGE`:
   * Если передан заголовок `X-OCR-Engine: paddle` -> захватываем `acquire_gpu0("paddle")` -> вызываем `paddle_client.extract(file_bytes, filename)` -> возвращаем Markdown с таблицами и текстом печатей.
   * По умолчанию -> захватываем `acquire_gpu0("mineru")` -> вызываем `mineru_client.extract(file_bytes, filename)`. `MinerU` возвращает `{"markdown": "...", "figures": {"fig_1.png": "<base64>"}}` (и с тем же успехом принимает картинки, предварительно отрендеренные или обёрнутые).
   * Освобождаем GPU 0! Затем для каждой вырезанной картинки `fig_name, fig_b64` из `figures`:
     * Прогоняем картинку через `smart_resize`.
     * Захватываем `acquire_gpu1()` -> отправляем картинку в `vlm_client` с промптом: *"Переведи схему на изображении в блок кода Mermaid или дай краткое техническое описание."*
     * Заменяем плейсхолдер `![...](fig_name)` в Markdown на полученный блок Mermaid/описание.
4. Возвращаем собранный Markdown и метаданные.

#### Пайплайн Б: `structured_pipe.py` (для 1С:ДО и Teamcenter)
1. **Если `doc_type == "drawing"` (Teamcenter):**
   * Если пришёл PDF — рендерим 1-ю страницу в `dpi=300`; если картинка — берём её байты.
   * Вызываем `crop_drawing_zones` -> получаем кропы `stamp` и `tech_req`.
   * Запрос 1 (`acquire_gpu1()`): отправляем кроп `stamp` в `vlm_client` с `format=DrawingStampCore.model_json_schema()` (обозначение, наименование, материал, масса, масштаб, организация).
   * Запрос 2 (`acquire_gpu1()`): отправляем кроп `tech_req` в `vlm_client` с `format=TechRequirementsList.model_json_schema()` (массив пунктов ТТ).
   * Собираем единый объект `DrawingStampSchema`, возвращаем со статусом `is_valid=True, warnings=[]`.
2. **Если `doc_type in ("upd", "invoice")` (1С:ДО) — двухшаговый конвейер:**
   * **Шаг 1 (Точный OCR текста и таблиц):**
     * Если `route in (DIGITAL_OFFICE, DIGITAL_PDF)` -> получаем сырой Markdown через `Docling` (CPU).
     * Если скан и `has_seals=True` (или `doc_type == "upd"`) -> через `acquire_gpu0("paddle")` получаем сырой Markdown таблиц и оттисков печатей из `PaddleOCR-VL`.
     * Иначе -> через `acquire_gpu0("mineru")` получаем сырой Markdown из `MinerU`.
   * **Шаг 2 (Сборка строгого JSON в Qwen3-VL на GPU 1):**
     * Готовим обзорное изображение 1-й страницы (рендер 150 DPI + `smart_resize`).
     * Захватываем `acquire_gpu1()` и отправляем в `vlm_client` промпт (включающий имя файла `filename` и распознанный на шаге 1 OCR-текст) + картинку 1-й страницы + `format=UpdSchema.model_json_schema()` (или `InvoiceSchema`).
   * **Шаг 3 (Пост-валидация):**
     * Парсим JSON через Pydantic (`model_validate_json`).
     * Прогоняем реквизиты через `validators/checksums.py` (ИНН продавца/покупателя, КПП, БИК) и суммы через `validators/math_checks.py`.
     * Если список ошибок пуст -> `is_valid = True`, иначе `is_valid = False` и записываем все расхождения в `warnings`.

---

### 4.7. Пост-валидаторы (`app/validators/`) — написать реальные алгоритмы!
Никаких заглушек, напиши чистую математику на Python:
1. **`validate_inn(inn: str | None) -> bool`**:
   * Если пусто, не цифры или длина не в `(10, 12)` -> `False`.
   * **Для 10 цифр (юрлицо):** коэффициенты `[2, 4, 10, 3, 5, 9, 4, 6, 8]`. Контрольное число `((sum(w * d) % 11) % 10)` должно равняться 10-й цифре (`inn[9]`).
   * **Для 12 цифр (ИП):**
     * 11-я цифра: коэффициенты `[7, 2, 4, 10, 3, 5, 9, 4, 6, 8]`, `((sum % 11) % 10) == int(inn[10])`.
     * 12-я цифра: коэффициенты `[3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]`, `((sum % 11) % 10) == int(inn[11])`.
2. **`validate_kpp(kpp: str | None) -> bool`**: длина 9, регулярка `^\d{4}[\dA-Z]{2}\d{3}$`.
3. **`validate_bik(bik: str | None) -> bool`**: 9 цифр, начинается на `"04"`.
4. **`validate_invoice_math(data: UpdSchema | InvoiceSchema) -> list[str]`**:
   * Для каждой позиции `i`: `abs(item.qty * item.price - item.amount_without_vat) <= 0.05`.
   * Для каждой позиции `i`: `abs(item.amount_without_vat + item.vat_amount - item.row_total) <= 0.05`.
   * По документу: `abs(sum(item.row_total for item in data.items) - data.total_with_vat) <= 0.05`.
   * Возвращает список читаемых ошибок на русском языке (или `[]`, если всё сошлось).

---

## 5. Спецификация сервиса заглушек (`mocks/main.py`)
Создай отдельное FastAPI-приложение на порту `8001`, которое имитирует все ML-движки, чтобы проверять сетевое взаимодействие и GPU-блокировки в бою:

1. **Состояние GPU 0 в памяти мока:** при старте `mineru_loaded: bool = False`, `paddle_loaded: bool = False`. Добавь эндпоинт `GET /mock/state` (возвращает текущие флаги для проверки в тестах).
2. `POST /docling/convert` -> спит `await asyncio.sleep(0.05)`, возвращает `{"markdown": f"# Цифровой документ ({filename})\nТаблица и текст из Docling."}`.
3. `POST /mineru/load` -> если `paddle_loaded == True`, выбрасывает **`HTTP 500 Simulated OOM: PaddleOCR is still in GPU 0 VRAM!`**, иначе ставит `mineru_loaded = True`.
4. `POST /mineru/unload` -> ставит `mineru_loaded = False`.
5. `POST /mineru/extract` -> если `mineru_loaded == False`, выбрасывает `HTTP 500 MinerU not loaded`; если `paddle_loaded == True`, выбрасывает `HTTP 500 Simulated OOM`. Спит `await asyncio.sleep(0.15)` (важно для теста параллелизма!), возвращает:
   * `markdown`: текст + `"![Схема](fig_1.png)"` (и упоминание имени файла),
   * `figures`: `{"fig_1.png": "<валидный base64 PNG 64x64>"}`.
6. `POST /paddle/load`, `POST /paddle/unload`, `POST /paddle/extract` -> зеркальная проверка конфликта VRAM с `mineru_loaded` (`HTTP 500 Simulated OOM`), спит `await asyncio.sleep(0.15)`, возвращает Markdown-таблицу УПД и текст круглой печати `"ПЕЧАТЬ: ООО Ромашка ОГРН 1027700132195"`.
7. `POST /api/chat` (эмуляция Ollama `Qwen3-VL` на GPU 1):
   * **Проверка одиночного потока GPU 1:** заведи глобальный `gpu1_lock = asyncio.Lock()`. В начале хэндлера проверь `if gpu1_lock.locked(): raise HTTPException(status_code=429, detail="GPU 1 Busy: Concurrent VLM request!")`. Затем зайди в `async with gpu1_lock: await asyncio.sleep(0.15)`.
   * Логика ответа по полю `"format"` в теле запроса:
     * Если `"format"` отсутствует (запрос на описание `figure`) -> вернуть `{"message": {"role": "assistant", "content": "```mermaid\ngraph TD;\n  A[Вход] --> B[Обработка];\n```"}}`.
     * Если в `"format"` есть поля штампа (`designation`) -> вернуть JSON основной надписи (`{"designation": "АБВГ.711141.001", "title": "Вал приводной", "material": "Сталь 45 ГОСТ 1050-2013", "mass": "12.5", "scale": "1:1", "organization": "АО КБ Маш"}`).
     * Если в `"format"` запрошены техтребования -> вернуть `{"requirements": ["1. Неуказанные предельные отклонения размеров H14, h14, +/-IT14/2.", "2. Острые кромки притупить R0,5."]}`.
     * Если в `"format"` схема УПД / Счёта:
       * Если в тексте промпта встречается подстрока `"bad_inn"` -> вернуть JSON с **битым ИНН** (`"7701234569"`) и **несходящейся суммой** (`qty=2, price=500, amount_without_vat=1000, vat_amount=200, row_total=1200`, но `total_with_vat=1500.00`), чтобы проверить срабатывание валидаторов!
       * Иначе -> вернуть полностью валидный JSON с настоящим проходящим контрольную сумму ИНН (например, `"7707083893"`), КПП `"770701001"`, БИК `"044525225"` и идеально сходящимися суммами.

---

## 6. Пошаговый план выполнения работы (следуй строго по шагам)

Двигайся итерациями и проверяй каждый шаг:

* **Шаг 1. Каркас и конфигурация.**
  Создай дерево директорий, `requirements.txt`, `.env.example`, `app/config.py`. Настрой `docker-compose.yml` с тремя контейнерами: `redis:7-alpine`, `mock-engines` (порт `8001`), `doc-orchestrator` (порт `8000`, `depends_on: [redis, mock-engines]`).
* **Шаг 2. Реализация `mock-engines`.**
  Напиши `mocks/main.py` со всеми эндпоинтами из раздела 5, включая жёсткую симуляцию `500 OOM` на GPU 0 и `429 Busy` на GPU 1.
* **Шаг 3. Схемы и математические валидаторы.**
  Напиши Pydantic-схемы (`upd.py`, `invoice.py`, `drawing_stamp.py`) и валидаторы (`checksums.py`, `math_checks.py`). Напиши `tests/test_validators.py` (валидный/невалидный ИНН на 10 и 12 цифр, КПП, БИК, расхождение копеек и НДС) и добейся зелёного `pytest`.
* **Шаг 4. Классификатор и обработка изображений.**
  Реализуй `classifier.py` и `image_ops.py`. В `tests/test_image_ops.py` программно сгенерируй через `PyMuPDF` и `Pillow` цифровой PDF (много текста), скан-PDF (картинка без текста), чертёж А4 (`2480x3508`) и чертёж А1 (`4000x2800`). Проверь, что `smart_resize` укладывается в `max_pixels` и выдаёт стороны строго кратные `32`, а кропы штампа работают для обеих ориентаций.
* **Шаг 5. GPU Resource Manager и HTTP-клиенты.**
  Реализуй `gpu_manager.py` (мьютексы GPU 0 / GPU 1, выгрузка при смене `mineru <-> paddle`, фоновый `idle_watcher` возврата по TTL) и клиенты в `app/clients/` с `httpx.AsyncClient(timeout=300.0)`.
* **Шаг 6. Сборка пайплайнов и API.**
  Собери `markdown_pipe.py`, `structured_pipe.py`, подключи `PUT /process`, `POST /api/extract-structured` и `GET /api/tasks/{task_id}` в `app/main.py`.
* **Шаг 7. Интеграционное и стресс-тестирование.**
  Напиши `tests/test_pipelines.py`, который проверяет:
  1. `PUT /process` с цифровым PDF (`Docling`) и со сканом (`MinerU` + вклейка блока `Mermaid` из `Qwen3-VL`).
  2. `POST /api/extract-structured` с валидным УПД (`is_valid=True, warnings=[]`) и с файлом `bad_inn.pdf` (`200 OK`, `is_valid=False`, непустой `warnings`).
  3. `POST /api/extract-structured` с `doc_type="drawing"` и `async_mode=True` -> поллинг `GET /api/tasks/{task_id}` до статуса `completed`.
  4. **Стресс-тест очереди GPU:** одновременный запуск **6 параллельных запросов** через `asyncio.gather` (2 скана `mineru`, 2 УПД с печатями `paddle`, 2 чертежа `drawing`), доказывающий, что `gpu_manager` выстраивает строгую очередь, корректно переключает модели в VRAM и ни разу не получает `500 OOM` или `429` от моков!

---

## 7. Критерии приёмки (Definition of Done)
Работа считается выполненной только тогда, когда:
1. Команда `docker compose up --build -d` поднимает все 3 контейнера без ошибок.
2. Команда `pytest -v` проходит на **100% без падений и пропусков**.
3. В коде нет `TODO`, заглушек `pass` вместо логики или захардкоженных адресов (все URL, имя модели VLM и лимиты берутся из `config.py` / `.env`, чтобы при переезде на боевой сервер достаточно было поменять URL моков на реальные контейнеры `docling-serve`, `mineru`, `paddle` и `ollama`).
4. Написан краткий `README.md` с командами запуска и рабочими примерами `curl` для Open WebUI (`PUT /process`), 1С:ДО (`upd`) и Teamcenter (`drawing`).
