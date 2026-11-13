# Полная инструкция по развёртыванию OCR/VLM-сервера (Debian 13 + CUDA 13.x + 2 × RTX 2070 SUPER)

> **Всё это делает скрипт.** Шаги 2–7 и подключение оркестратора (п. 9.3) автоматизирует `deploy/install.sh` из репозитория `doc-orchestrator`:
> ```bash
> sudo bash deploy/install.sh check   # ничего не меняет: что уже установлено и что делать дальше
> sudo bash deploy/install.sh all     # всё по порядку; после установки драйвера попросит перезагрузку
> ```
> Каждый шаг можно запускать отдельно и повторно (`driver`, `docker`, `mineru`, `stack`, `models`, `verify`, `orchestrator`; справка — `--help`). Ниже те же шаги вручную — чтобы было понятно, что именно делает скрипт. Ссылки, имена пакетов, теги образов и моделей сверены с первоисточниками 13.11.2026.
>
> Скрипт проверен автотестами на имитации сервера, на настоящем сервере с видеокартами он ещё не запускался. Начните с `check` и посмотрите план без выполнения: `sudo bash deploy/install.sh --dry-run all`.

---

## 0. Ревизия железа и важные поправки по софту

Ваша конфигурация:
* **ОС:** `Debian GNU/Linux 13 (trixie)` — актуальный стабильный релиз Debian.
* **GPU:** `2 × NVIDIA GeForce RTX 2070 SUPER` (архитектура **Turing `sm_75`**, по **8 ГБ VRAM** на каждую, без аппаратного `bf16`).
* **CPU / RAM / Диск:** `Intel Core i7-7700K` (4 ядра / 8 потоков), `32 ГБ RAM`, `500 ГБ NVMe SSD`.
* **Текущий драйвер:** `550.163.01` (из стандартного репозитория Debian, потолок — **CUDA 12.4**).

### Что критически важно учесть перед установкой:
1. **Драйвер NVIDIA нужно чисто обновить с `550` до ветки `595` (CUDA 13.2).**
   Актуальные контейнеры (`MinerU 4` на CUDA 13.0 и свежие сборки CUDA 13.x) требуют драйвер хоста **не ниже `580`–`595`** — на старом `550.163` они падают с `CUDA driver version is insufficient`. В репозитории NVIDIA для `debian13` уже лежат ветки `590`, `595` и `610–620`: мы жёстко закрепим проверенную ветку **R595**, которая официально поддерживает **Turing (`sm_75`)**. Перед её установкой обязательно вычищаются пакеты `550` из репозитория Debian, чтобы избежать конфликта `glx-alternative-nvidia`.
2. **Системный Python 3.13 не трогаем.**
   Debian 13 блокирует системный `pip` (`PEP 668`), а зависимости OCR-движков несовместимы между собой. **Весь стек работает изолированно в Docker.**
3. **Принцип автономности.**
   Все движки настраиваются так, чтобы **работать самостоятельно прямо сейчас** (через свои WebUI, REST API и штатные коннекторы Open WebUI), даже пока сервис `doc-orchestrator` ещё в разработке или отключён.

---

## 1. Подбор моделей под ваши задачи (RU + EN, техническая и административная документация)

На картах по **8 ГБ VRAM (`sm_75`)** два движка одновременно не помещаются, `vLLM` приходится ограничивать по памяти (MinerU делает это сам: на картах до 8 ГБ отдаёт ему 6 ГБ), и нельзя пулить дефолтные `thinking`-теги моделей. Ниже выверенный стек под русский/английский языки и потолок 8192 МБ:

| Задача и тип документов (RU / EN) | Сервис и точная модель | Где крутится и сколько ест | Автономная работа (без оркестратора) |
| :--- | :--- | :--- | :--- |
| **Цифровая админка и ТЗ:** `.docx`, `.xlsx`, `.pptx`, текстовые PDF (договоры, приказы, спецификации) | **`Docling Serve v1.38.2`** *(внутри `docling-ibm-models 4.0.4`, общие веса в памяти)* | **CPU** (4 потока под `i7-7700K`), ~3–5 ГБ RAM, **0 ГБ VRAM** | Веб-интерфейс на `:5001/ui`, Swagger `:5001/docs` + штатный коннектор в Open WebUI |
| **Сложные тех. сканы:** ГОСТы, ТУ, формулы, многоколоночные отчёты, таблицы | **`MinerU 4.2.1`** *(уровень `--tier standard`: текст читает VLM-модель `MinerU2.5-Pro-2605-1.2B`, она понимает кириллицу. Уровень `basic` (~4 ГБ, без VLM) русский текст **не распознаёт**: в словаре его OCR `PP-OCRv6` нет кириллицы, а выбора языка в MinerU 4 нет)* | **GPU 0** (`vLLM` резервирует **6 ГБ из 8**, остальное занимают малые модели: карта занята почти целиком — проверьте `nvidia-smi` после запуска) | Свой Gradio WebUI на `:7860` + REST API `:8002/v1` |
| **Тяжёлая первичка с круглыми печатями и штампами:** УПД, счета-фактуры, акты, мятые сканы | **`PaddleOCR-VL-1.6 (0.9B)` + `PP-DocLayoutV3`** *(111 языков вкл. RU/EN + `Seal Recognition`)* | **GPU 0** (автор образа указывает **~8.5 ГБ VRAM** при `DPI=200`; с `DPI=150` должно быть меньше, но цифру надо проверить на своей карте — запускается *вместо* MinerU!) | Готовый REST API `:8003` (сохраняет шапки, подвалы и оттиски печатей в Markdown) |
| **Чертежи (ЕСКД/ГОСТ), штампы, ТТ, схемы в Mermaid + сборка JSON для 1С** | **Ollama** + две `Instruct`-модели:<br>1) **`qwen3-vl:4b-instruct-q8_0`** *(основная рабочая, высокая чёткость и запас по памяти)*<br>2) **`qwen3-vl:8b-instruct-q4_K_M`** *(под особо мелкий шрифт)* | **GPU 1** монопольно (KV `q8_0`):<br>• `4b-q8_0` (`ctx=8192`) — **~6.2 ГБ VRAM**<br>• `8b-q4_K_M` (`ctx=4096`) — **~7.3 ГБ VRAM** | Прямое подключение к Open WebUI (`:11434`) — можно кидать кропы чертежей и сканы прямо в чат |

---

## 2. Шаг 1: Остановка старого софта и освобождение видеокарт

Остановите старые процессы Python/OCR, не удаляя их рабочие каталоги (для возможности отката):

```bash
# 1. Смотрим, какие процессы держат видеокарты
nvidia-smi
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')
[ -n "$PIDS" ] && ps -fp $PIDS

# 2. Ищем старые сервисы в автозагрузке
systemctl list-units --type=service --state=running | grep -iE "ocr|docling|ollama|vllm|mineru|paddle|python|ai|ml"

# 3. Отключаем найденные старые сервисы (подставьте ваши имена)
systemctl stop <имя_старого_сервиса>
systemctl disable <имя_старого_сервиса>
```
*(Если процессы запущены вручную в `tmux`/`screen`/`nohup` — завершите их через `kill <PID>`).*

Проверьте свободное место на NVMe (`df -h`) — нужно **не менее 90–100 ГБ** под образы и веса моделей.

---

## 3. Шаг 2: Чистое обновление драйвера NVIDIA до ветки 595 (CUDA 13.2) на Debian 13

Чтобы пакеты NVIDIA не столкнулись с системными пакетами Debian `550.163`, сначала удаляем старые пакеты драйвера, подключаем репозиторий `debian13` и закрепляем ветку **595**. В источниках APT должен быть включён раздел `contrib` (если драйвер ставили из репозитория Debian, он уже включён).

```bash
# 1. Ставим заголовки ядра и базовые утилиты
# (linux-headers-amd64 — метапакет: после обновления ядра модуль NVIDIA пересоберётся сам)
apt update
apt install -y linux-headers-$(uname -r) linux-headers-amd64 build-essential dkms ca-certificates wget curl gpg

# 2. Чисто удаляем старый драйвер 550 из репозитория Debian (во избежание конфликта альтернатив)
# (если NVIDIA Container Toolkit уже стоит, команда удалит и его — шаг 3 поставит заново)
apt purge -y '*nvidia*' 'libnvidia*'
apt autoremove -y

# 3. Подключаем официальный репозиторий NVIDIA CUDA для Debian 13 (trixie)
cd /tmp
rm -f cuda-keyring_1.1-1_all.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
apt install -y ./cuda-keyring_1.1-1_all.deb
apt update

# 4. Жёстко пиннингуем ветку 595 (чтобы apt не поставил 615/620) и ставим драйвер CUDA 13.2
apt install -y nvidia-driver-pinning-595
# cuda-drivers-595 — виртуальное имя (его «предоставляет» пакет cuda-drivers версий 595.x):
# ставим сам пакет cuda-drivers последней версии этой ветки явно
apt install -y "cuda-drivers=$(apt-cache madison cuda-drivers | awk -F'|' '{gsub(/ /,"",$2)} $2 ~ /^595[.]/ {print $2; exit}')"

# 5. Перезагружаем сервер
reboot
```

**Проверка после перезагрузки:**
Выполните `nvidia-smi`. В шапке таблицы должно быть:
* `Driver Version: 595.xx.xx`
* `CUDA Version: 13.2`
* Память на обеих `RTX 2070 SUPER`: `0–15 MiB / 8192 MiB`.

---

## 4. Шаг 3: Установка актуального Docker CE на Debian 13 + NVIDIA Container Toolkit

Ставим свежий Docker Engine из официального репозитория под `trixie` и настраиваем проброс GPU:

```bash
# 1. Убираем старые пакеты из архива Debian (если стояли)
apt remove -y $(dpkg --get-selections docker.io docker-compose docker-buildx docker-doc podman-docker containerd runc | cut -f1) 2>/dev/null || true

# 2. Подключаем репозиторий Docker (формат DEB822 для Debian 13)
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: $(. /etc/os-release && echo "$VERSION_CODENAME")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git

# 3. Ставим NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt update && apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 4. Контрольный тест проброса обеих карт в контейнер CUDA 13.0
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

---

## 5. Шаг 4: Сборка локального образа MinerU 4 (CUDA 13.0, модели запечены внутри)

Собираем официальный образ `mineru:4` (веса скачиваются с Hugging Face внутрь образа и дальше работают полностью офлайн):

```bash
mkdir -p /opt/ai-doc-stack/{ollama,paddle,redis}
cd /opt/ai-doc-stack

git clone --depth 1 --branch mineru-4.2.1-released https://github.com/opendatalab/MinerU.git mineru-src
cd mineru-src
# Dockerfile ставит «последний MinerU 4.x» — закрепляем проверенную версию 4.2.1
# (между 4.0 и 4.1 у MinerU менялся HTTP API, под 4.2.1 написан шлюз doc-orchestrator)
sed -i "s/'mineru\[torch\]>=4.0,<5'/'mineru[torch]==4.2.1'/" docker/global/Dockerfile
docker build -t mineru:4 -f docker/global/Dockerfile .
cd /opt/ai-doc-stack
```
*(Сборка скачивает около 20 ГБ — базовый образ vLLM и веса моделей — и занимает 15–40 минут в зависимости от канала интернета).*

---

## 6. Шаг 5: Единый `docker-compose.yml` всех сервисов

Создаём общую Docker-сеть (в неё же позже встанет `doc-orchestrator`):

```bash
docker network create doc-net
```

Создайте файл `/opt/ai-doc-stack/docker-compose.yml` (он же лежит в репозитории: `deploy/ai-doc-stack/docker-compose.yml`). Три настройки вынесены в файл `/opt/ai-doc-stack/.env` (образец — `deploy/ai-doc-stack/.env.example`; без файла работают значения по умолчанию): `MINERU_TIER` — уровень MinerU, `BIND_ADDR` — на каком адресе сервера открыть порты движков (авторизации у них нет), `PADDLE_OCR_DPI` — разрешение рендера для PaddleOCR-VL.

```yaml
services:
  # 1. Redis — очередь задач и GPU-семафоры doc-orchestrator
  redis:
    image: redis:7-alpine
    container_name: doc-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - ./redis:/data
    networks:
      - doc-net

  # 2. Docling Serve v1.38.2 (CPU) — цифровые PDF, DOCX, XLSX, PPTX (модели уже в образе)
  # WebUI: http://IP:5001/ui | Swagger: http://IP:5001/docs
  docling:
    image: quay.io/docling-project/docling-serve-cpu:v1.38.2
    container_name: docling-serve
    restart: unless-stopped
    ports:
      - "${BIND_ADDR:-0.0.0.0}:5001:5001"
    environment:
      - DOCLING_SERVE_ENABLE_UI=true
      - DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2
      - DOCLING_SERVE_ENG_LOC_SHARE_MODELS=true   # Не дублировать модели в RAM между потоками
      - DOCLING_SERVE_MAX_SYNC_WAIT=300           # Запас по таймауту для больших PDF на 4 ядрах
      - OMP_NUM_THREADS=4
      - MKL_NUM_THREADS=4
    cpus: 4.0
    mem_limit: 8g
    networks:
      - doc-net

  # 3. GPU 1 (Вторая RTX 2070S, 8 ГБ): Ollama для Qwen3-VL Instruct (чертежи, схемы, JSON)
  ollama-vlm:
    image: ollama/ollama:latest
    container_name: ollama-vlm
    restart: unless-stopped
    ports:
      - "${BIND_ADDR:-0.0.0.0}:11434:11434"
    environment:
      - OLLAMA_NUM_PARALLEL=1          # Строго 1 запрос одновременно для 8 ГБ VRAM
      - OLLAMA_MAX_LOADED_MODELS=1     # Только 1 VLM в видеопамяти
      - OLLAMA_FLASH_ATTENTION=1       # Обязателен для квантования KV-кэша
      - OLLAMA_KV_CACHE_TYPE=q8_0      # Вдвое сжимает KV-кэш без потери точности OCR
      - OLLAMA_KEEP_ALIVE=24h
    volumes:
      - ./ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']        # Монтирует строго вторую карту (GPU 1)
              capabilities: [gpu]
    networks:
      - doc-net

  # 4A. GPU 0 (Первая RTX 2070S, 8 ГБ): MinerU 4 API
  # Уровень задаёт MINERU_TIER (.env). standard: текст читает VLM-модель MinerU2.5-Pro — она
  # понимает кириллицу; vLLM резервирует под неё 6 ГБ VRAM, остальное занимают малые модели.
  # basic: только малые модели (~4 ГБ), но их OCR (PP-OCRv6) кириллицу НЕ распознаёт.
  mineru-api:
    image: mineru:4
    container_name: mineru-api
    restart: unless-stopped
    ports:
      - "${BIND_ADDR:-0.0.0.0}:8002:8000"
    environment:
      - MINERU_MODEL_SOURCE=local
    entrypoint: ["mineru-kit"]
    command:
      - api-server
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --tier
      - ${MINERU_TIER:-standard}
      - --concurrency
      - "1"
      - --preload-models               # Модели грузятся при старте: /v1/health отвечает, когда VRAM уже занята
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/v1/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s               # Пока грузятся модели, сервер не отвечает — это не сбой
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']        # Монтирует строго первую карту (GPU 0)
              capabilities: [gpu]
    networks:
      - doc-net

  # 4B. Опциональный браузерный интерфейс MinerU (:7860)
  # Своих моделей не грузит: отправляет задачи в mineru-api
  mineru-webui:
    image: mineru:4
    container_name: mineru-webui
    restart: unless-stopped
    profiles: ["webui"]
    ports:
      - "${BIND_ADDR:-0.0.0.0}:7860:7860"
    environment:
      - MINERU_MODEL_SOURCE=local
    entrypoint: ["mineru-kit"]
    command:
      - webui
      - --server-name
      - 0.0.0.0
      - --server-port
      - "7860"
      - --api-url
      - http://mineru-api:8000
    ipc: host
    networks:
      - doc-net

  # 5. GPU 0 (Слот под печати и тяжёлую первичку): PaddleOCR-VL-1.6 (0.9B) + PP-DocLayoutV3
  # ВНИМАНИЕ: Запускать ТОЛЬКО после `docker stop mineru-api`, чтобы уложиться в 8 ГБ на GPU 0!
  # Когда подключён doc-orchestrator, контейнеры GPU 0 переключает он сам (через engine-gateway).
  paddle-vl:
    image: edgaras0x4e/paddleocr-pdf-api:v0.5.0-vl-baked
    container_name: paddle-gpu0
    restart: unless-stopped            # После перезагрузки сервера поднимется тот движок GPU 0, что работал до неё
    profiles: ["manual-paddle"]
    ports:
      - "${BIND_ADDR:-0.0.0.0}:8003:8000"
    environment:
      - OCR_ENGINE=vl
      - OCR_DPI=${PADDLE_OCR_DPI:-150} # Меньше DPI — меньше пик видеопамяти (по умолчанию у образа 200)
      - OCR_MARKDOWN_IGNORE_LABELS=    # Пусто = НЕ вырезать шапки, подвалы, номера УПД и штампы!
    volumes:
      - ./paddle:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    networks:
      - doc-net

networks:
  doc-net:
    external: true
```

> **Удобная деталь в пункте 4B:** благодаря параметру `--api-url http://mineru-api:8000` веб-интерфейс `mineru-webui` вообще не грузит вторую копию моделей в видеопамять, а отправляет задачи в уже работающий `mineru-api`! Его можно держать включённым одновременно с API: `docker compose --profile webui up -d`.

---

## 7. Шаг 6: Запуск стека и настройка `Qwen3-VL Instruct` под 8 ГБ VRAM

Запускаем базовые контейнеры:

```bash
cd /opt/ai-doc-stack
docker compose --profile webui up -d
```

Скачиваем строго **`instruct`-теги** моделей `Qwen3-VL` (никогда не пулите короткие `qwen3-vl:4b` или `:8b` — они скачивают `thinking`-версии, забивающие память рассуждениями) и фиксируем безопасный контекст:

```bash
# 1. Скачиваем основную чёткую 4B Q8_0 (5.1 ГБ) и тяжёлую 8B Q4_K_M (6.2 ГБ)
docker exec -it ollama-vlm ollama pull qwen3-vl:4b-instruct-q8_0
docker exec -it ollama-vlm ollama pull qwen3-vl:8b-instruct-q4_K_M

# 2. Основной рабочий профиль qwen3-vl-doc (4B Q8_0, ctx=8192 -> ~6.2 ГБ из 8 ГБ, максимальная надёжность)
docker exec -it ollama-vlm sh -c 'cat <<EOF > /tmp/Modelfile.4b
FROM qwen3-vl:4b-instruct-q8_0
PARAMETER num_ctx 8192
PARAMETER temperature 0.1
EOF
ollama create qwen3-vl-doc -f /tmp/Modelfile.4b'

# 3. Профиль повышенной детализации qwen3-vl-hq (8B Q4_K_M, ctx=4096 -> ~7.3 ГБ из 8 ГБ, 100% на GPU)
docker exec -it ollama-vlm sh -c 'cat <<EOF > /tmp/Modelfile.8b
FROM qwen3-vl:8b-instruct-q4_K_M
PARAMETER num_ctx 4096
PARAMETER temperature 0.1
EOF
ollama create qwen3-vl-hq -f /tmp/Modelfile.8b'
```

---

## 8. Шаг 7: Проверка работоспособности каждого узла

Выполните по очереди 4 проверки:

```bash
# 1. Проверка Docling Serve на CPU (ожидаем {"status":"ok"})
curl -s http://localhost:5001/health && echo ""

# 2. Проверка MinerU 4 API и доступных уровней на GPU 0 (ожидаем "status":"ok" и уровни flash, basic, standard)
curl -s http://localhost:8002/v1/health && echo ""
curl -s http://localhost:8002/v1/tiers && echo ""

# 3. Прогрев и проверка Qwen3-VL Instruct на GPU 1
curl -s http://localhost:11434/api/generate \
  -d '{"model": "qwen3-vl-doc", "prompt": "Ответь одним словом: готов?", "stream": false}' && echo ""

# 4. Контроль размещения в VRAM (обязательно должно быть 100% GPU!)
docker exec -it ollama-vlm ollama ps
nvidia-smi
```
> **Проверка делом.** «Сервер отвечает» ещё не значит «распознаёт»: скрипт из репозитория отправляет каждому движку образец с русским текстом и проверяет результат — `python3 deploy/smoke_test.py --docling http://localhost:5001 --mineru http://localhost:8002 --ollama http://localhost:11434` (PaddleOCR-VL — `--paddle http://localhost:8003`, пока запущен `paddle-gpu0`). То же делает `deploy/install.sh verify`.

> **Важно в пункте 4:** в колонке `PROCESSOR` у модели обязательно должно стоять **`100% GPU`**. При отправке больших чертежей А1/А0 из кода оркестратора предварительно сжимайте целые листы до `≤ 1.3 МП` (размеры кратны `32 px`) или вырезайте кроп штампа ГОСТ / зоны ТТ — в самой Ollama нет автоматического ограничителя `max_pixels`.

---

## 9. Работа БЕЗ `doc-orchestrator` и последующее переключение

Все узлы уже готовы к продуктивной работе напрямую:

1. **Ручная конвертация документов через браузер (без кода):**
   * **Офисные файлы и цифровые PDF:** `http://<IP_СЕРВЕРА>:5001/ui` (Docling UI).
   * **Сложные тех. сканы, ГОСТы, таблицы и формулы:** `http://<IP_СЕРВЕРА>:7860` (MinerU WebUI, работает поверх `mineru-api`).
   * **Ручное переключение GPU 0 на PaddleOCR-VL-1.6 (под пачку тяжёлых УПД с печатями):**
     ```bash
     cd /opt/ai-doc-stack
     docker stop mineru-api
     docker compose --profile manual-paddle up -d paddle-vl
     # Возврат обратно на MinerU:
     docker stop paddle-gpu0 && docker start mineru-api
     ```
     То же одной командой: `sudo bash deploy/install.sh gpu0 paddle` и `… gpu0 mineru`. Когда подключён `doc-orchestrator`, GPU 0 переключает он сам — вручную контейнеры не трогайте: чтобы документ прочитал PaddleOCR-VL, его отправляют с заголовком `X-OCR-Engine: paddle`.

2. **Прямая привязка к Open WebUI (занимает 2 минуты):**
   * **Документы:** `Admin Panel → Settings → Documents → Content Extraction Engine` → выберите **`Docling`** и укажите `http://<IP_СЕРВЕРА>:5001`.
   * **Чертежи, штампы ГОСТ и сканы счетов в чате:** `Admin Panel → Settings → Connections → Ollama API` → укажите `http://<IP_СЕРВЕРА>:11434`. Сотрудники выбирают модель **`qwen3-vl-doc`** (или `qwen3-vl-hq`) и прикрепляют изображения/кропы прямо в диалог.

3. **Подключение `doc-orchestrator`:**
   Переустанавливать ничего не нужно. Из каталога репозитория `doc-orchestrator` на сервере:
   ```bash
   sudo bash deploy/install.sh orchestrator
   ```
   Команда создаёт файл `.env`, собирает образ и запускает в сети `doc-net` два контейнера (`docker-compose.prod.yml`): `doc-orchestrator` (порт `8000`) и `engine-gateway` — шлюз движков.

   Шлюз нужен потому, что оркестратор говорит с движками на одном языке (`/convert`, `/extract`, `/load`, `/unload` — так описано в `AGENTS.md`), а у штатных серверов API свой: docling-serve — `POST /v1/convert/file`, MinerU 4 — `POST /v1/tasks`, PaddleOCR-VL — `POST /ocr`, и команды «выгрузить модель из видеопамяти» у них нет. Шлюз переводит запросы, а GPU 0 переключает так же, как это делается вручную в п. 1: останавливает и запускает контейнеры `mineru-api` и `paddle-gpu0` (для этого ему проброшен сокет Docker; наружу порт шлюза не публикуется).

   Адреса в `.env` оркестратора:
   * `REDIS_URL=redis://doc-redis:6379/0`
   * `DOCLING_URL=http://engine-gateway:8001/docling`
   * `MINERU_URL=http://engine-gateway:8001/mineru`
   * `PADDLE_URL=http://engine-gateway:8001/paddle`
   * `OLLAMA_URL=http://ollama-vlm:11434` и `VLM_MODEL=qwen3-vl-doc` (JSON Schema ответа передаётся через поле `format`)

   Уровень и язык в запросах к MinerU не передаются: уровень задаётся при запуске сервера (`MINERU_TIER`), а параметра языка в API MinerU 4 нет — запрос с лишним полем сервер отклоняет с HTTP 400.

   При переключении Open WebUI на оркестратор (`Content Extraction Engine` = **`External`**) обязательно заполните **оба** поля (`URL` и непустой `API Key`) и добавьте `image/*` в `Supported Media MIME Types`. В случае сбоя оркестратора откат обратно на прямой `Docling` (`:5001`) в админке занимает 10 секунд.