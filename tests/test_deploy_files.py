"""Файлы развёртывания согласованы между собой: compose-файлы, настройки шлюза, .env и инструкция.

Имена контейнеров и порты записаны в нескольких местах (compose стека, значения по умолчанию
шлюза, .env оркестратора, инструкция). Разойдутся — на сервере перестанет работать, а узнают
об этом только там. Эти тесты ловят расхождение заранее.
"""

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from app.config import Settings
from gateway.config import GatewaySettings

ROOT = Path(__file__).parent.parent
STACK = yaml.safe_load((ROOT / "deploy/ai-doc-stack/docker-compose.yml").read_text())["services"]
PROD = yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text())
INSTALL_SH = (ROOT / "deploy/install.sh").read_text()
INSTRUCTION = (ROOT / "ИНСТРУКЦИЯ_ПО_УСТАНОВКЕ_OCR.md").read_text()


def env_names(path: Path, commented: bool = False) -> set[str]:
    """Имена переменных .env-файла; commented — вместе с закомментированными (#ИМЯ=значение)."""
    pattern = r"^#?([A-Z][A-Z0-9_]*)=" if commented else r"^([A-Z][A-Z0-9_]*)="
    return set(re.findall(pattern, path.read_text(), flags=re.MULTILINE))


def container_port(service: dict) -> int:
    """Порт внутри контейнера из записи вида «адрес:порт_сервера:порт_контейнера»."""
    return int(service["ports"][0].rsplit(":", 1)[1])


def gpu_of(service: dict) -> list[str]:
    return service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"]


# --- Стек движков и шлюз ----------------------------------------------------------------------------


def test_gateway_defaults_point_at_stack_containers():
    gateway = GatewaySettings()
    engines = {
        "docling": gateway.docling_serve_url,
        "mineru-api": gateway.mineru_api_url,
        "paddle-vl": gateway.paddle_api_url,
    }
    for service_name, url in engines.items():
        service, address = STACK[service_name], urlsplit(url)
        assert address.hostname == service["container_name"]  # в сети doc-net контейнер виден по имени
        assert address.port == container_port(service)
    assert gateway.mineru_container == STACK["mineru-api"]["container_name"]
    assert gateway.paddle_container == STACK["paddle-vl"]["container_name"]


def test_only_one_gpu0_engine_starts_by_default():
    assert gpu_of(STACK["mineru-api"]) == gpu_of(STACK["paddle-vl"]) == ["0"]
    assert gpu_of(STACK["ollama-vlm"]) == ["1"]
    # Вдвоём MinerU и PaddleOCR-VL в 8 ГБ не помещаются: второй запускается только по требованию
    assert "profiles" not in STACK["mineru-api"]
    assert STACK["paddle-vl"]["profiles"] == ["manual-paddle"]


def test_mineru_tier_reads_cyrillic_by_default():
    command = STACK["mineru-api"]["command"]
    # Уровень basic кириллицу не распознаёт (PP-OCRv6) — по умолчанию стоит standard
    assert command[command.index("--tier") + 1] == "${MINERU_TIER:-standard}"
    assert "--preload-models" in command  # /v1/health отвечает, когда модели уже в видеопамяти
    assert "MINERU_TIER=standard" in (ROOT / "deploy/ai-doc-stack/.env.example").read_text()


def test_stack_env_example_lists_exactly_compose_variables():
    compose_text = (ROOT / "deploy/ai-doc-stack/docker-compose.yml").read_text()
    used = set(re.findall(r"\$\{([A-Z_]+)", compose_text))

    assert (
        used
        == env_names(ROOT / "deploy/ai-doc-stack/.env.example")
        == {"MINERU_TIER", "BIND_ADDR", "PADDLE_OCR_DPI"}
    )


def test_install_script_waits_on_published_ports():
    published = {
        name: int(service["ports"][0].split(":")[-2]) for name, service in STACK.items() if "ports" in service
    }

    assert published == {
        "docling": 5001,
        "ollama-vlm": 11434,
        "mineru-api": 8002,
        "mineru-webui": 7860,
        "paddle-vl": 8003,
    }
    for url in (":5001/health", ":11434/api/version", ":8002/v1/health", ":8003/jobs"):
        assert url in INSTALL_SH


# --- Оркестратор на боевом сервере -------------------------------------------------------------------


def test_orchestrator_env_points_at_stack_and_gateway():
    """Адреса, которые install.sh подставляет в .env оркестратора, ведут в существующие контейнеры."""
    values = dict(re.findall(r'-e "s\|\^([A-Z_]+)=\.\*\|[A-Z_]+=([^|]+)\|"', INSTALL_SH))
    services = PROD["services"]
    gateway = services["engine-gateway"]
    gateway_port = int(gateway["command"][gateway["command"].index("--port") + 1])

    assert urlsplit(values["REDIS_URL"]).hostname == STACK["redis"]["container_name"]
    ollama = urlsplit(values["OLLAMA_URL"])
    assert (ollama.hostname, ollama.port) == (
        STACK["ollama-vlm"]["container_name"],
        container_port(STACK["ollama-vlm"]),
    )
    for engine in ("docling", "mineru", "paddle"):
        url = urlsplit(values[f"{engine.upper()}_URL"])
        assert (url.hostname, url.port, url.path) == (gateway["container_name"], gateway_port, f"/{engine}")
    assert values["VLM_MODEL"] == "qwen3-vl-doc"
    assert (
        'ollama_profile qwen3-vl-doc "$VLM_MAIN" 8192' in INSTALL_SH
    )  # профиль с этим именем создаёт шаг models
    assert set(values) | {"API_KEY"} <= {name.upper() for name in Settings.model_fields}


def test_prod_compose_joins_stack_network_and_gives_docker_socket_only_to_gateway():
    services = PROD["services"]
    orchestrator, gateway = services["doc-orchestrator"], services["engine-gateway"]

    assert PROD["networks"] == {"doc-net": {"external": True}}
    assert orchestrator["networks"] == gateway["networks"] == ["doc-net"]
    assert gateway["volumes"] == ["/var/run/docker.sock:/var/run/docker.sock"]
    assert GatewaySettings().docker_socket == "/var/run/docker.sock"
    assert "volumes" not in orchestrator  # сервис, разбирающий чужие файлы, к Docker доступа не имеет
    assert "ports" not in gateway  # шлюз нужен только оркестратору
    assert gateway["command"][:2] == ["uvicorn", "gateway.main:app"]
    # Один образ, разные команды. Собирает его один сервис: две сборки под одним именем мешают друг другу
    assert orchestrator["image"] == gateway["image"]
    assert "build" in orchestrator
    assert "build" not in gateway
    assert gateway["pull_policy"] == "never"  # иначе Compose сначала поищет образ в Docker Hub
    assert "COPY gateway ./gateway" in (ROOT / "Dockerfile").read_text()
    assert "!gateway/" in (ROOT / ".dockerignore").read_text()


def test_env_example_documents_every_gateway_setting():
    documented = env_names(ROOT / ".env.example", commented=True)

    assert {name.upper() for name in GatewaySettings.model_fields} <= documented


# --- Инструкция ----------------------------------------------------------------------------------------


def test_instruction_compose_block_is_the_file_from_repository():
    [block] = re.findall(r"```yaml\n(.*?)```", INSTRUCTION, flags=re.DOTALL)

    assert yaml.safe_load(block)["services"] == STACK


def test_instruction_names_match_what_orchestrator_reads():
    section = INSTRUCTION.split("Подключение `doc-orchestrator`")[1]

    assert "VLM_URL" not in INSTRUCTION  # оркестратор читает OLLAMA_URL
    for name in ("REDIS_URL", "DOCLING_URL", "MINERU_URL", "PADDLE_URL", "OLLAMA_URL", "VLM_MODEL"):
        assert name in section
        assert name.lower() in Settings.model_fields
    assert "deploy/install.sh" in INSTRUCTION
