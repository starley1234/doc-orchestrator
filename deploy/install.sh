#!/usr/bin/env bash
#
# Помощник администратора: ставит OCR/VLM-стек по «ИНСТРУКЦИЯ_ПО_УСТАНОВКЕ_OCR.md»
# (Debian 13, две видеокарты NVIDIA по 8 ГБ) и подключает к нему doc-orchestrator.
#
#   sudo bash deploy/install.sh check     # ничего не меняет: что уже есть и что делать дальше
#   sudo bash deploy/install.sh all       # всё по порядку; после установки драйвера попросит перезагрузку
#
# Каждый шаг можно запускать отдельно и повторно: сделанное он не переделывает.
# Полная справка: bash deploy/install.sh --help

set -Eeuo pipefail
umask 022   # ключи и списки репозиториев APT должен читать служебный пользователь _apt

# --- Настройки (переопределяются переменными окружения) --------------------------------------

STACK_DIR="${STACK_DIR:-/opt/ai-doc-stack}"      # каталог стека: docker-compose.yml, модели, данные
NVIDIA_BRANCH="${NVIDIA_BRANCH:-595}"            # ветка драйвера NVIDIA (CUDA 13.2, поддерживает Turing)
MINERU_VERSION="${MINERU_VERSION:-4.2.1}"        # версия MinerU в образе mineru:4 (под её API написан шлюз)
VLM_MAIN="${VLM_MAIN:-qwen3-vl:4b-instruct-q8_0}"     # основная модель  -> профиль qwen3-vl-doc
VLM_HQ="${VLM_HQ:-qwen3-vl:8b-instruct-q4_K_M}"       # для мелкого шрифта -> профиль qwen3-vl-hq
CUDA_TEST_IMAGE="${CUDA_TEST_IMAGE:-nvidia/cuda:13.0.0-base-ubuntu24.04}"
SYSROOT="${SYSROOT:-}"                           # только для автотестов: префикс системных путей

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
STACK_SOURCE="$SCRIPT_DIR/ai-doc-stack"          # docker-compose.yml и .env.example стека
LOG_FILE="$SYSROOT/var/log/ai-doc-stack-install.log"

ASSUME_YES=""        # --yes: не задавать вопросов
DRY_RUN=""           # --dry-run: показывать команды, но не выполнять
REBOOT_NEEDED=""     # шаг driver поставил драйвер: дальше — только после перезагрузки

export DEBIAN_FRONTEND=noninteractive

# --- Вывод ----------------------------------------------------------------------------------

if [ -t 1 ]; then
    BOLD=$'\033[1m' RED=$'\033[31m' GREEN=$'\033[32m' YELLOW=$'\033[33m' RESET=$'\033[0m'
else
    BOLD="" RED="" GREEN="" YELLOW="" RESET=""
fi

step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
info() { printf '      %s\n' "$*"; }
ok()   { printf '%s[ OK ]%s %s\n' "$GREEN" "$RESET" "$*"; }
did()  { [ -n "$DRY_RUN" ] || ok "$@"; }      # итог выполненного шага: при --dry-run не выводится
warn() { printf '%s[ !! ]%s %s\n' "$YELLOW" "$RESET" "$*"; }
die()  { printf '\n%sОШИБКА:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

trap 'die "команда завершилась с ошибкой (строка $LINENO): $BASH_COMMAND"' ERR

# Выполняет команду, изменяющую систему, и показывает её. С --dry-run — только показывает.
run() {
    printf '    $ %s\n' "$*"
    [ -z "$DRY_RUN" ] || return 0
    "$@" || die "команда не выполнена (код $?): $*"
}

# Записывает стандартный ввод в файл (с --dry-run — только сообщает об этом).
write_file() {
    printf '    > %s\n' "$1"
    if [ -n "$DRY_RUN" ]; then cat >/dev/null; else cat >"$1"; fi
}

confirm() {
    [ -z "$ASSUME_YES" ] || return 0
    [ -z "$DRY_RUN" ] || return 0
    [ -t 0 ] || die "нужно подтверждение: «$1». Запустите в терминале или добавьте --yes"
    local answer
    read -r -p "    $1 [y/N] " answer
    [[ "$answer" =~ ^[YyДд] ]]
}

have() { command -v "$1" >/dev/null 2>&1; }

# --- Проверки окружения -------------------------------------------------------------------------

need_root() {
    [ -z "$DRY_RUN" ] || return 0
    [ "$(id -u)" -eq 0 ] || die "нужны права root: sudo bash $0 $COMMAND"
}

need_debian13() {
    [ -r "$SYSROOT/etc/os-release" ] || die "не найден /etc/os-release: скрипт рассчитан на Debian 13"
    # shellcheck disable=SC1091
    . "$SYSROOT/etc/os-release"
    [ "${ID:-}" = "debian" ] && [ "${VERSION_ID:-}" = "13" ] ||
        die "скрипт рассчитан на Debian 13 (trixie), а здесь ${PRETTY_NAME:-неизвестная система}"
    [ "$(dpkg --print-architecture)" = "amd64" ] || die "нужна архитектура amd64"
}

docker_ready() { have docker && docker compose version >/dev/null 2>&1; }

need_docker() {
    docker_ready || [ -n "$DRY_RUN" ] || die "Docker не установлен: сначала sudo bash $0 docker"
}

need_stack() {
    [ -f "$STACK_DIR/docker-compose.yml" ] || [ -n "$DRY_RUN" ] ||
        die "в $STACK_DIR нет docker-compose.yml: сначала sudo bash $0 stack"
}

start_log() {
    [ -z "$DRY_RUN" ] || return 0
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null && touch "$LOG_FILE" 2>/dev/null || return 0
    exec > >(tee -a "$LOG_FILE") 2>&1
    printf '\n===== %s: install.sh %s =====\n' "$(date '+%F %T')" "$COMMAND"
}

# --- Состояние системы (только чтение) -----------------------------------------------------------

# Версия драйвера, с которым сейчас работает ядро; пусто — драйвер не загружен.
running_driver() {
    have nvidia-smi || return 0
    nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true
}

# Версия установленного пакета; пусто — пакет не установлен.
package_version() {
    dpkg-query -W -f='${db:Status-Abbrev}${Version}' "$1" 2>/dev/null | sed -n 's/^ii //p' || true
}

gpu_count() {
    have nvidia-smi || { echo 0; return 0; }
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c . || true
}

container_state() {     # running, exited, created ... или missing
    docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo missing
}

image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Проверки ниже сначала читают вывод целиком и лишь потом ищут в нём: `команда | grep -q`
# при set -o pipefail даёт ложный отказ, когда grep закрывает канал раньше, чем команда договорит.
nvidia_runtime_ready() { [[ "$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)" == *nvidia* ]]; }

secure_boot_on() { have mokutil && [[ "$(mokutil --sb-state 2>/dev/null || true)" == *"SecureBoot enabled"* ]]; }

contrib_enabled() {
    local sources
    sources="$(grep -rhsE '^(deb |Components:)' "$SYSROOT/etc/apt/sources.list" \
        "$SYSROOT/etc/apt/sources.list.d/" || true)"
    grep -qw contrib <<<"$sources"
}

ollama_models() { docker exec ollama-vlm ollama list 2>/dev/null | awk 'NR > 1 { print $1 }' || true; }

# Docker Compose для стека движков: run "${STACK_COMPOSE[@]}" up -d
STACK_COMPOSE=(docker compose --project-directory "$STACK_DIR" -f "$STACK_DIR/docker-compose.yml")

# Значение переменной из .env-файла (пусто — переменной нет).
env_value() {
    [ -f "$2" ] || return 0
    sed -n "s/^$1=//p" "$2" | tail -n 1
}

# Адрес, на котором открыты порты движков: BIND_ADDR из .env стека (0.0.0.0 -> локальный адрес).
stack_host() {
    local addr
    addr="$(env_value BIND_ADDR "$STACK_DIR/.env")"
    case "$addr" in "" | 0.0.0.0) echo 127.0.0.1 ;; *) echo "$addr" ;; esac
}

# Ждёт, пока сервер ответит 200. Не дождались или сервер ответил ошибкой — показывает хвост
# лога контейнера и останавливает установку.
wait_http() {
    local title="$1" url="$2" seconds="$3" container="$4" waited=0 code problem
    [ -z "$DRY_RUN" ] || { info "ожидание: $title ($url)"; return 0; }
    printf '      жду %s (до %s с) ' "$title" "$seconds"
    while :; do
        # Код 000 — сервер ещё не принимает соединения
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || true)"
        if [ "$code" = "200" ]; then
            printf ' готово\n'
            return 0
        elif [ "${code:-000}" != "000" ]; then
            # Сервер запустился, но отвечает ошибкой: так MinerU сообщает, что не загрузил модели.
            # Само это не пройдёт — ждать дальше незачем
            problem="сообщает о сбое (HTTP $code): $(curl -s --max-time 5 "$url" 2>/dev/null | head -c 300 || true)"
        elif [ "$waited" -ge "$seconds" ] || [ "$(container_state "$container")" != "running" ]; then
            problem="не отвечает ($url)"
        else
            sleep 5
            waited=$((waited + 5))
            printf '.'
            continue
        fi
        printf '\n'
        docker logs --tail 30 "$container" 2>&1 | sed 's/^/      | /' || true
        die "$title $problem. Выше — конец лога контейнера $container"
    done
}

# --- Шаг 2 инструкции: драйвер NVIDIA -------------------------------------------------------------

stage_driver() {
    need_root
    need_debian13
    step "Драйвер NVIDIA ветки $NVIDIA_BRANCH"

    local running installed
    running="$(running_driver)"
    installed="$(package_version nvidia-driver)"
    if [[ "$running" == "$NVIDIA_BRANCH".* ]]; then
        ok "драйвер $running уже работает"
        return 0
    fi
    if [[ "$installed" == "$NVIDIA_BRANCH".* ]]; then
        warn "драйвер $installed установлен, но ядро работает со старым (${running:-драйвер не загружен})"
        REBOOT_NEEDED=1
        return 0
    fi
    info "сейчас: драйвер ${running:-не загружен}, пакет nvidia-driver ${installed:-не установлен}"

    contrib_enabled ||
        die "в источниках APT не включён раздел contrib — он нужен пакетам драйвера NVIDIA.
       Допишите «contrib non-free non-free-firmware» в строку Components: файла
       /etc/apt/sources.list.d/debian.sources (или в строки deb файла /etc/apt/sources.list)
       и запустите шаг снова"

    if secure_boot_on; then
        warn "включён Secure Boot: модуль ядра NVIDIA заработает только после регистрации ключа DKMS:"
        info "mokutil --import /var/lib/dkms/mok.pub   # затем перезагрузка и подтверждение на консоли сервера"
        confirm "Продолжить установку драйвера?" || die "установка драйвера отменена"
    fi

    run apt-get update
    local kernel
    kernel="$(uname -r)"
    [ -n "$DRY_RUN" ] || apt-cache show "linux-headers-$kernel" >/dev/null 2>&1 ||
        die "для работающего ядра $kernel в репозитории нет заголовков: система давно не обновлялась.
       Выполните apt full-upgrade, перезагрузите сервер и запустите шаг снова"
    run apt-get install -y "linux-headers-$kernel" build-essential dkms ca-certificates wget curl gpg
    # Метапакет заголовков: после обновления ядра модуль NVIDIA пересоберётся сам
    local flavour="${kernel##*+deb13-}"
    if [ "$flavour" != "$kernel" ] && apt-cache show "linux-headers-$flavour" >/dev/null 2>&1; then
        run apt-get install -y "linux-headers-$flavour"
    fi

    # Старый драйвер из репозитория Debian удаляем целиком, иначе пакеты столкнутся.
    # NVIDIA Container Toolkit не трогаем: к драйверу он не относится.
    local old
    old="$(dpkg-query -W -f='${db:Status-Abbrev}${binary:Package}\n' 2>/dev/null |
        sed -n 's/^i[^ ]* //p' | grep -i nvidia |
        grep -Ev '^(nvidia-container|libnvidia-container|nvidia-driver-pinning)' || true)"
    if [ -n "$old" ]; then
        warn "будет удалён прежний драйвер NVIDIA: пакетов $(printf '%s\n' "$old" | grep -c .)"
        info "до перезагрузки видеокарты останутся без драйвера (шаг 1 инструкции: остановите старый софт)"
        confirm "Удалить прежний драйвер и поставить ветку $NVIDIA_BRANCH?" || die "установка драйвера отменена"
        # shellcheck disable=SC2086
        run apt-get purge -y $old
        run apt-get autoremove -y
    fi

    if [ -z "$(package_version cuda-keyring)" ]; then
        local deb="cuda-keyring_1.1-1_all.deb" tmp="/tmp"
        [ -n "$DRY_RUN" ] || tmp="$(mktemp -d)"
        run wget -q -O "$tmp/$deb" \
            "https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/$deb"
        run apt-get install -y "$tmp/$deb"
        run apt-get update
    fi

    # Пакет-закрепитель не даёт apt уйти на более новые ветки (610, 620)
    run apt-get install -y "nvidia-driver-pinning-$NVIDIA_BRANCH"
    # cuda-drivers-595 из инструкции — виртуальное имя: ставим настоящий пакет cuda-drivers
    # последней версии нужной ветки, чтобы результат не зависел от того, как apt разрешит имя
    local version
    version="$(apt-cache madison cuda-drivers 2>/dev/null |
        awk -F'|' -v branch="^${NVIDIA_BRANCH}[.]" '{ gsub(/ /, "", $2) } $2 ~ branch { print $2; exit }' || true)"
    if [ -z "$version" ]; then
        [ -n "$DRY_RUN" ] || die "в репозитории NVIDIA нет пакета cuda-drivers ветки $NVIDIA_BRANCH"
        version="<последняя $NVIDIA_BRANCH.x>"
    fi
    run apt-get install -y "cuda-drivers=$version"
    did "драйвер $version установлен"
    REBOOT_NEEDED=1
}

reboot_notice() {
    step "Нужна перезагрузка"
    info "Новый драйвер начнёт работать после перезагрузки сервера:"
    info "    reboot"
    info "Затем запустите ту же команду ещё раз — установка продолжится:"
    info "    sudo bash $0 $COMMAND"
}

# --- Шаг 3 инструкции: Docker и NVIDIA Container Toolkit -----------------------------------------

stage_docker() {
    need_root
    need_debian13
    step "Docker CE и NVIDIA Container Toolkit"
    [[ "$(running_driver)" == "$NVIDIA_BRANCH".* ]] || [ -n "$DRY_RUN" ] ||
        die "драйвер NVIDIA ветки $NVIDIA_BRANCH не работает: сначала sudo bash $0 driver и перезагрузка"

    if docker_ready; then
        ok "Docker уже установлен: $(docker --version)"
    else
        local conflicting
        conflicting="$(dpkg --get-selections docker.io docker-compose docker-buildx docker-doc podman-docker \
            containerd runc 2>/dev/null | cut -f1 || true)"
        # shellcheck disable=SC2086
        [ -z "$conflicting" ] || run apt-get remove -y $conflicting
        run apt-get update
        run apt-get install -y ca-certificates curl gpg
        run install -m 0755 -d "$SYSROOT/etc/apt/keyrings"
        run curl -fsSL https://download.docker.com/linux/debian/gpg -o "$SYSROOT/etc/apt/keyrings/docker.asc"
        run chmod a+r "$SYSROOT/etc/apt/keyrings/docker.asc"
        write_file "$SYSROOT/etc/apt/sources.list.d/docker.sources" <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: ${VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
        run apt-get update
        run apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
            docker-compose-plugin git
    fi

    if have nvidia-ctk; then
        ok "NVIDIA Container Toolkit уже установлен"
    else
        local keyring="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        local repo="https://nvidia.github.io/libnvidia-container"
        run install -m 0755 -d "$SYSROOT/usr/share/keyrings"
        printf '    $ curl %s/gpgkey | gpg --dearmor -o %s\n' "$repo" "$keyring"
        [ -n "$DRY_RUN" ] || curl -fsSL "$repo/gpgkey" | gpg --dearmor --yes -o "$SYSROOT$keyring"
        if [ -n "$DRY_RUN" ]; then
            : | write_file "$SYSROOT/etc/apt/sources.list.d/nvidia-container-toolkit.list"
        else
            curl -fsSL "$repo/stable/deb/nvidia-container-toolkit.list" |
                sed "s#deb https://#deb [signed-by=$keyring] https://#g" |
                write_file "$SYSROOT/etc/apt/sources.list.d/nvidia-container-toolkit.list"
        fi
        run apt-get update
        run apt-get install -y nvidia-container-toolkit
    fi

    if nvidia_runtime_ready; then
        ok "Docker уже умеет пробрасывать видеокарты (runtime nvidia)"
    else
        run nvidia-ctk runtime configure --runtime=docker
        run systemctl restart docker
    fi

    info "проверка: видны ли обе видеокарты из контейнера"
    if [ -n "$DRY_RUN" ]; then
        run docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi -L
        return 0
    fi
    local seen
    seen="$(docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi -L | grep -c '^GPU ' || true)"
    [ "$seen" -ge 2 ] || die "из контейнера видно видеокарт: $seen, а нужно 2. Проверьте nvidia-smi на сервере"
    did "из контейнера видны видеокарты: $seen"
}

# --- Шаг 4 инструкции: образ MinerU ----------------------------------------------------------------

stage_mineru() {
    need_root
    need_docker
    step "Образ mineru:4 (MinerU $MINERU_VERSION, модели внутри)"
    if image_exists mineru:4 && [ -z "${REBUILD:-}" ]; then
        ok "образ mineru:4 уже собран (пересобрать: REBUILD=1 bash $0 mineru)"
        return 0
    fi
    [[ "$STACK_DIR" == /* ]] || die "STACK_DIR должен быть абсолютным путём"
    local src="$STACK_DIR/mineru-src" tag="mineru-${MINERU_VERSION}-released"
    local dockerfile="$src/docker/global/Dockerfile"
    run mkdir -p "$STACK_DIR"
    run rm -rf "$src"
    run git clone --depth 1 --branch "$tag" https://github.com/opendatalab/MinerU.git "$src"
    # Dockerfile ставит «последний MinerU 4.x»: закрепляем версию, чтобы сборка через месяц дала тот же API
    if [ -n "$DRY_RUN" ] || grep -q "'mineru\[torch\]>=4.0,<5'" "$dockerfile"; then
        run sed -i "s/'mineru\[torch\]>=4.0,<5'/'mineru[torch]==$MINERU_VERSION'/" "$dockerfile"
    else
        warn "в Dockerfile MinerU нет ожидаемой строки установки: версия не закреплена"
    fi
    info "сборка скачивает около 20 ГБ (базовый образ vLLM и веса моделей): 15–40 минут"
    run docker build -t mineru:4 -f "$dockerfile" "$src"
    did "образ mineru:4 собран"
}

# --- Шаги 5–6 инструкции: сеть, docker-compose.yml, запуск ------------------------------------------

stage_stack() {
    need_root
    need_docker
    step "Стек движков в $STACK_DIR"
    image_exists mineru:4 || [ -n "$DRY_RUN" ] || die "образ mineru:4 не собран: сначала sudo bash $0 mineru"

    run mkdir -p "$STACK_DIR/ollama" "$STACK_DIR/paddle" "$STACK_DIR/redis"
    docker network inspect doc-net >/dev/null 2>&1 || run docker network create doc-net

    local target="$STACK_DIR/docker-compose.yml"
    if [ -f "$target" ] && ! cmp -s "$STACK_SOURCE/docker-compose.yml" "$target"; then
        local backup
        backup="$target.$(date +%Y%m%d-%H%M%S).bak"
        run cp -p "$target" "$backup"
        warn "docker-compose.yml отличался от версии из репозитория: прежний сохранён как $backup"
    fi
    run install -m 0644 "$STACK_SOURCE/docker-compose.yml" "$target"
    if [ -f "$STACK_DIR/.env" ]; then
        ok "настройки $STACK_DIR/.env сохранены как есть"
    else
        run install -m 0644 "$STACK_SOURCE/.env.example" "$STACK_DIR/.env"
    fi

    # GPU 0 делят MinerU и PaddleOCR-VL. Если сейчас работает PaddleOCR-VL, MinerU запускать нельзя:
    # вдвоём они в 8 ГБ не помещаются
    local services=()
    if [ "$(container_state paddle-gpu0)" = "running" ]; then
        warn "GPU 0 занят PaddleOCR-VL (paddle-gpu0): mineru-api сейчас не запускаю"
        services=(redis docling ollama-vlm mineru-webui)
    fi
    run "${STACK_COMPOSE[@]}" --profile webui up -d "${services[@]}"
    # Контейнер PaddleOCR-VL создаём (или обновляем по новым настройкам), но не запускаем:
    # его включают вместо MinerU
    [ "$(container_state paddle-gpu0)" = "running" ] ||
        run "${STACK_COMPOSE[@]}" --profile manual-paddle create paddle-vl

    local host
    host="$(stack_host)"
    wait_http "Docling" "http://$host:5001/health" 600 docling-serve
    wait_http "Ollama" "http://$host:11434/api/version" 120 ollama-vlm
    if [ ${#services[@]} -eq 0 ]; then
        wait_http "MinerU (загрузка моделей в видеопамять)" "http://$host:8002/v1/health" 900 mineru-api
    fi
    did "стек запущен"
}

# --- Шаг 6 инструкции: модели Qwen3-VL ---------------------------------------------------------------

ollama_profile() {      # профиль модели с безопасным контекстом: имя, базовая модель, num_ctx
    local name="$1" base="$2" ctx="$3"
    printf '    $ ollama create %s (FROM %s, num_ctx %s, temperature 0.1)\n' "$name" "$base" "$ctx"
    [ -z "$DRY_RUN" ] || return 0
    printf 'FROM %s\nPARAMETER num_ctx %s\nPARAMETER temperature 0.1\n' "$base" "$ctx" |
        docker exec -i ollama-vlm sh -c "cat > /tmp/Modelfile.$name && ollama create $name -f /tmp/Modelfile.$name" ||
        die "не удалось создать профиль модели $name"
}

stage_models() {
    need_root
    need_docker
    step "Модели Qwen3-VL в Ollama (GPU 1)"
    [ "$(container_state ollama-vlm)" = "running" ] || [ -n "$DRY_RUN" ] ||
        die "контейнер ollama-vlm не запущен: сначала sudo bash $0 stack"
    local model present
    present="$(ollama_models)"
    for model in "$VLM_MAIN" "$VLM_HQ"; do
        if grep -qxF "$model" <<<"$present"; then
            ok "модель $model уже скачана"
        else
            run docker exec ollama-vlm ollama pull "$model"
        fi
    done
    # Строго instruct-варианты: короткие теги qwen3-vl:4b и :8b — это thinking-версии
    ollama_profile qwen3-vl-doc "$VLM_MAIN" 8192
    ollama_profile qwen3-vl-hq "$VLM_HQ" 4096
    did "профили qwen3-vl-doc и qwen3-vl-hq готовы"
}

# --- Шаг 7 инструкции: проверка узлов --------------------------------------------------------------

smoke() {
    # Ключ оркестратора на экран и в журнал не выводим
    printf '    $ python3 deploy/smoke_test.py %s\n' "$(sed -E 's/(--api-key) [^ ]*/\1 ***/' <<<"$*")"
    [ -n "$DRY_RUN" ] || python3 "$SCRIPT_DIR/smoke_test.py" "$@"
}

# Ручное переключение GPU 0 (п. 9.1 инструкции): mineru или paddle.
switch_gpu0() {
    local host
    host="$(stack_host)"
    case "$1" in
        paddle)
            run docker stop mineru-api
            run "${STACK_COMPOSE[@]}" --profile manual-paddle up -d paddle-vl
            wait_http "PaddleOCR-VL" "http://$host:8003/jobs" 300 paddle-gpu0
            ;;
        mineru)
            [ "$(container_state paddle-gpu0)" = "missing" ] || run docker stop paddle-gpu0
            run docker start mineru-api
            wait_http "MinerU (загрузка моделей в видеопамять)" "http://$host:8002/v1/health" 900 mineru-api
            ;;
    esac
}

orchestrator_runs() { [ "$(container_state doc-orchestrator)" = "running" ]; }

stage_verify() {
    need_root
    need_docker
    need_stack
    step "Проверка движков на образце с русским текстом"
    have python3 || run apt-get install -y python3
    local host failed="" checks
    host="$(stack_host)"

    checks=(--docling "http://$host:5001" --ollama "http://$host:11434")
    if [ "$(container_state mineru-api)" = "running" ] || [ -n "$DRY_RUN" ]; then
        checks+=(--mineru "http://$host:8002")
    else
        warn "mineru-api не запущен (GPU 0 занят PaddleOCR-VL?): MinerU напрямую не проверяется"
    fi
    smoke "${checks[@]}" || failed=1

    if orchestrator_runs; then
        info "GPU 0 переключает оркестратор: PaddleOCR-VL проверяется через него"
        smoke --orchestrator http://127.0.0.1:8000 --api-key "$(env_value API_KEY "$REPO_DIR/.env")" \
            --with-paddle || failed=1
    elif [ -n "${SKIP_PADDLE:-}" ]; then
        warn "PaddleOCR-VL не проверяется (SKIP_PADDLE)"
    else
        info "PaddleOCR-VL: на время проверки он займёт GPU 0 вместо MinerU"
        switch_gpu0 paddle
        smoke --paddle "http://$host:8003" || failed=1
        switch_gpu0 mineru
    fi

    if [ -z "$DRY_RUN" ]; then
        info "видеопамять сейчас:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null |
            sed 's/^/      GPU /' || true
    fi
    [ -z "$failed" ] || die "не все проверки пройдены — причины в строках [FAIL] выше"
    did "все проверки пройдены"
}

# --- П. 9.3 инструкции: doc-orchestrator ------------------------------------------------------------

stage_orchestrator() {
    need_root
    need_docker
    need_stack
    step "doc-orchestrator и шлюз движков"
    [ "$(container_state doc-redis)" = "running" ] || [ -n "$DRY_RUN" ] ||
        die "стек движков не запущен (нет контейнера doc-redis): сначала sudo bash $0 stack"

    local env_file="$REPO_DIR/.env" key
    if [ -f "$env_file" ]; then
        ok "настройки $env_file сохранены как есть"
    else
        # Open WebUI включает внешний загрузчик только с непустым ключом — создаём случайный
        key="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        # umask 077: файл с ключом с первой секунды читает только root
        (
            umask 077
            sed -e "s|^REDIS_URL=.*|REDIS_URL=redis://doc-redis:6379/0|" \
                -e "s|^DOCLING_URL=.*|DOCLING_URL=http://engine-gateway:8001/docling|" \
                -e "s|^MINERU_URL=.*|MINERU_URL=http://engine-gateway:8001/mineru|" \
                -e "s|^PADDLE_URL=.*|PADDLE_URL=http://engine-gateway:8001/paddle|" \
                -e "s|^OLLAMA_URL=.*|OLLAMA_URL=http://ollama-vlm:11434|" \
                -e "s|^VLM_MODEL=.*|VLM_MODEL=qwen3-vl-doc|" \
                -e "s|^API_KEY=.*|API_KEY=$key|" \
                "$REPO_DIR/.env.example" | write_file "$env_file"
        )
    fi

    # Шлюз запускает и останавливает контейнеры GPU 0 по имени: оба должны существовать
    [ "$(container_state paddle-gpu0)" != "missing" ] ||
        run "${STACK_COMPOSE[@]}" --profile manual-paddle create paddle-vl

    run docker compose --project-directory "$REPO_DIR" -f "$REPO_DIR/docker-compose.prod.yml" up -d --build
    wait_http "doc-orchestrator" "http://127.0.0.1:8000/health" 180 doc-orchestrator
    did "doc-orchestrator запущен: http://<IP сервера>:8000 (описание API — /docs)"

    step "Подключение Open WebUI"
    info "Admin Panel → Settings → Documents:"
    info "    Content Extraction Engine  = External"
    info "    External Document Loader URL = http://<IP сервера>:8000"
    info "    API Key                    = значение API_KEY из файла $env_file"
    info "    Supported Media MIME Types  — добавьте image/*"
    info "Откат на прямой Docling: Content Extraction Engine = Docling, http://<IP сервера>:5001"
}

# --- Сводки --------------------------------------------------------------------------------------

stage_status() {
    step "Контейнеры"
    if have docker; then
        docker ps -a --format '{{.Names}}\t{{.Status}}' 2>/dev/null |
            grep -E '^(doc-redis|docling-serve|ollama-vlm|mineru-api|mineru-webui|paddle-gpu0|doc-orchestrator|engine-gateway)\b' |
            awk -F'\t' '{ printf "      %-17s %s\n", $1, $2 }' || info "контейнеров стека нет"
    else
        info "Docker не установлен"
    fi
    step "Видеокарты"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null |
        sed 's/^/      GPU /' || info "nvidia-smi недоступен"
    if have docker && [ "$(container_state ollama-vlm)" = "running" ]; then
        step "Модели в памяти Ollama"
        docker exec ollama-vlm ollama ps 2>/dev/null | sed 's/^/      /' || true
    fi
}

stage_check() {
    local next=""
    [ "$(id -u)" -eq 0 ] || warn "запущено без root: сведения о Docker и контейнерах могут быть неполными"
    step "Сервер"
    if [ -r "$SYSROOT/etc/os-release" ]; then
        # shellcheck disable=SC1091
        . "$SYSROOT/etc/os-release"
        if [ "${ID:-}" = "debian" ] && [ "${VERSION_ID:-}" = "13" ]; then
            ok "ОС: ${PRETTY_NAME:-Debian 13}"
        else
            warn "ОС: ${PRETTY_NAME:-неизвестна} — скрипт рассчитан на Debian 13"
        fi
    fi
    local ram_gb free_gb docker_root="$SYSROOT/var/lib/docker"
    ram_gb="$(awk '/^MemTotal:/ { printf "%d", $2 / 1048576 }' "$SYSROOT/proc/meminfo" 2>/dev/null || echo 0)"
    if [ "$ram_gb" -ge 30 ]; then ok "память: $ram_gb ГБ"; else warn "память: $ram_gb ГБ (в инструкции — 32 ГБ)"; fi
    while [ ! -d "$docker_root" ] && [ "$docker_root" != "/" ] && [ -n "$docker_root" ]; do
        docker_root="$(dirname "$docker_root")"
    done
    free_gb="$(df -BG --output=avail "$docker_root" 2>/dev/null | tail -n 1 | tr -dc '0-9' || true)"
    if [ "${free_gb:-0}" -ge 100 ]; then
        ok "свободно на диске под образы и модели: $free_gb ГБ"
    else
        warn "свободно на диске: ${free_gb:-?} ГБ — под образы и модели нужно около 100 ГБ"
    fi

    step "Видеокарты и драйвер"
    local running count
    running="$(running_driver)"
    count="$(gpu_count)"
    if [ -n "$running" ]; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null |
            sed 's/^/      GPU /' || true
        if [ "$count" -ge 2 ]; then ok "видеокарт: $count"; else warn "видеокарт: $count, а стек рассчитан на 2"; fi
        local busy
        busy="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
        if [ -n "$busy" ] && ! { have docker && [ "$(container_state doc-redis)" = "running" ]; }; then
            warn "видеокарты заняты процессами — остановите старый софт сами (шаг 1 инструкции):"
            printf '%s\n' "$busy" | sed 's/^/      /'
        fi
    fi
    if [[ "$running" == "$NVIDIA_BRANCH".* ]]; then
        ok "драйвер NVIDIA: $running"
    elif [[ "$(package_version nvidia-driver)" == "$NVIDIA_BRANCH".* ]]; then
        warn "драйвер ветки $NVIDIA_BRANCH установлен, но не загружен: перезагрузите сервер"
        next="reboot"
    else
        warn "драйвер NVIDIA: ${running:-не найден} — нужна ветка $NVIDIA_BRANCH"
        next="sudo bash $0 driver"
    fi
    if secure_boot_on; then
        warn "включён Secure Boot: модулю ядра NVIDIA понадобится регистрация ключа DKMS (mokutil --import)"
    fi

    step "Docker"
    if docker_ready; then
        ok "$(docker --version)"
        if nvidia_runtime_ready; then
            ok "проброс видеокарт в контейнеры настроен (runtime nvidia)"
        else
            warn "проброс видеокарт в контейнеры не настроен"
            next="${next:-sudo bash $0 docker}"
        fi
    else
        warn "Docker с плагином compose не установлен"
        next="${next:-sudo bash $0 docker}"
    fi

    if docker_ready; then
        step "Стек"
        if image_exists mineru:4; then
            ok "образ mineru:4 собран"
        else
            warn "образ mineru:4 не собран"
            next="${next:-sudo bash $0 mineru}"
        fi
        local name state
        for name in doc-redis docling-serve ollama-vlm; do
            state="$(container_state "$name")"
            if [ "$state" = "running" ]; then
                ok "контейнер $name работает"
            else
                warn "контейнер $name: $state"
                next="${next:-sudo bash $0 stack}"
            fi
        done
        local mineru paddle
        mineru="$(container_state mineru-api)"
        paddle="$(container_state paddle-gpu0)"
        case "$mineru/$paddle" in
            running/running) warn "mineru-api и paddle-gpu0 запущены одновременно: вдвоём они в 8 ГБ GPU 0 не помещаются" ;;
            running/*) ok "GPU 0: работает mineru-api (paddle-gpu0: $paddle)" ;;
            */running) ok "GPU 0: работает paddle-gpu0 (mineru-api: $mineru)" ;;
            *)
                warn "GPU 0: ни mineru-api ($mineru), ни paddle-gpu0 ($paddle) не запущены"
                next="${next:-sudo bash $0 stack}"
                ;;
        esac
        if [ "$(container_state ollama-vlm)" = "running" ]; then
            if grep -q '^qwen3-vl-doc' <<<"$(ollama_models)"; then
                ok "профиль модели qwen3-vl-doc создан"
            else
                warn "профиль модели qwen3-vl-doc не создан"
                next="${next:-sudo bash $0 models}"
            fi
        fi
        if orchestrator_runs && [ "$(container_state engine-gateway)" = "running" ]; then
            ok "doc-orchestrator и engine-gateway работают"
        else
            warn "doc-orchestrator: $(container_state doc-orchestrator), engine-gateway: $(container_state engine-gateway)"
            next="${next:-sudo bash $0 orchestrator}"
        fi
    fi

    step "Что дальше"
    if [ -n "$next" ]; then
        info "$next"
        info "или всё по порядку: sudo bash $0 all"
    else
        info "всё установлено. Проверка делом: sudo bash $0 verify"
    fi
}

stage_gpu0() {
    need_root
    need_docker
    need_stack
    case "${1:-}" in mineru | paddle) ;; *) die "укажите, кому отдать GPU 0: $0 gpu0 mineru | paddle" ;; esac
    ! orchestrator_runs || die "GPU 0 переключает doc-orchestrator: вручную нельзя — он не узнает о смене.
       Чтобы распознать документ в PaddleOCR-VL, отправьте его с заголовком X-OCR-Engine: paddle"
    step "GPU 0 -> $1"
    switch_gpu0 "$1"
    did "GPU 0 отдан: $1"
}

stage_all() {
    stage_driver
    if [ -n "$REBOOT_NEEDED" ]; then
        reboot_notice
        [ -n "$DRY_RUN" ] || return 0
        step "После перезагрузки установка продолжится так"    # пробный запуск показывает план целиком
    fi
    stage_docker
    stage_mineru
    stage_stack
    stage_models
    stage_verify
    stage_orchestrator
    step "Проверка всей цепочки через doc-orchestrator"
    smoke --orchestrator http://127.0.0.1:8000 --api-key "$(env_value API_KEY "$REPO_DIR/.env")" --with-paddle ||
        die "оркестратор установлен, но проверка не пройдена — причины в строках [FAIL] выше"
    did "установка завершена"
    step "Адреса"
    info "doc-orchestrator:  http://<IP сервера>:8000  (описание API — /docs)"
    info "Docling UI:        http://<IP сервера>:5001/ui"
    info "MinerU WebUI:      http://<IP сервера>:7860"
    info "Ollama:            http://<IP сервера>:11434  (модели qwen3-vl-doc, qwen3-vl-hq)"
    info "Состояние в любой момент: sudo bash $0 status"
}

usage() {
    cat <<EOF
Помощник администратора: OCR/VLM-стек по «ИНСТРУКЦИЯ_ПО_УСТАНОВКЕ_OCR.md» и doc-orchestrator.

Запуск:  sudo bash deploy/install.sh [--yes] [--dry-run] <команда>

Команды (в скобках — раздел инструкции):
  check          что уже установлено и что делать дальше; ничего не меняет
  driver         драйвер NVIDIA ветки $NVIDIA_BRANCH вместо драйвера из Debian (шаг 2); затем нужна перезагрузка
  docker         Docker CE, NVIDIA Container Toolkit, проверка проброса видеокарт (шаг 3)
  mineru         сборка образа mineru:4 — MinerU $MINERU_VERSION с моделями внутри (шаг 4)
  stack          сеть doc-net, $STACK_DIR/docker-compose.yml, запуск движков (шаги 5–6)
  models         модели Qwen3-VL и профили qwen3-vl-doc, qwen3-vl-hq в Ollama (шаг 6)
  verify         проверка каждого движка на образце с русским текстом (шаг 7)
  orchestrator   doc-orchestrator и шлюз движков в сети doc-net, ключ для Open WebUI (п. 9.3)
  all            всё перечисленное по порядку
  status         контейнеры, видеопамять, модели в памяти Ollama
  gpu0 mineru|paddle   вручную отдать GPU 0 одному из движков (п. 9.1; без оркестратора)

Ключи:
  --yes, -y      не спрашивать подтверждений (удаление старого драйвера и т. п.)
  --dry-run, -n  показать команды, ничего не выполняя

Переменные окружения:
  STACK_DIR=$STACK_DIR   NVIDIA_BRANCH=$NVIDIA_BRANCH   MINERU_VERSION=$MINERU_VERSION
  VLM_MAIN=$VLM_MAIN   VLM_HQ=$VLM_HQ
  REBUILD=1      пересобрать образ mineru:4 (команда mineru)
  SKIP_PADDLE=1  не проверять PaddleOCR-VL (команда verify)

Шаг 1 инструкции — остановку старого софта на видеокартах — скрипт не делает: что именно
останавливать, решает администратор (занятые видеокарты покажет команда check).
Журнал установки: /var/log/ai-doc-stack-install.log
EOF
}

# --- Разбор аргументов -------------------------------------------------------------------------------

COMMAND=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -y | --yes) ASSUME_YES=1 ;;
        -n | --dry-run) DRY_RUN=1 ;;
        -h | --help | help) COMMAND="help" ;;
        -*) die "неизвестный ключ $1 (справка: bash $0 --help)" ;;
        *) if [ -z "$COMMAND" ]; then COMMAND="$1"; else ARGS+=("$1"); fi ;;
    esac
    shift
done

case "$COMMAND" in
    "" | help) usage ;;
    check) stage_check ;;
    status) stage_status ;;
    driver | docker | mineru | stack | models | verify | orchestrator | all | gpu0)
        need_root
        start_log
        "stage_$COMMAND" "${ARGS[@]}"
        if [ "$COMMAND" = "driver" ] && [ -n "$REBOOT_NEEDED" ]; then reboot_notice; fi
        [ -z "$DRY_RUN" ] || step "Это был пробный запуск (--dry-run): ничего не выполнялось и не менялось"
        ;;
    *) die "неизвестная команда «$COMMAND» (справка: bash $0 --help)" ;;
esac
