"""deploy/smoke_test.py: проверка установленных движков — на стенде, поднятом на настоящих портах.

Скрипт запускается отдельным процессом, как на сервере, и ходит по HTTP в имитации штатных
серверов (stock_engines.py), а в последнем блоке — в настоящие оркестратор и шлюз перед ними.
"""

import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import httpx
import pymupdf
import pytest
import uvicorn
from fakeredis import FakeAsyncRedis
from stock_engines import MINERU, PADDLE, Stand, docker_app, docling_app, mineru_app, ollama_app, paddle_app

from app.config import Settings
from app.main import create_app as create_orchestrator
from gateway.config import GatewaySettings
from gateway.main import create_app as create_gateway
from mocks.main import create_app as create_mock_engines

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "deploy/smoke_test.py"
SAMPLES = ROOT / "deploy/samples"
DEAD = "http://127.0.0.1:9"  # порт, на котором никто не слушает


@contextmanager
def serve(app) -> Iterator[str]:
    """Поднимает приложение на свободном порту 127.0.0.1 в отдельном потоке; отдаёт его адрес."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "сервер стенда не запустился"
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


@pytest.fixture
def stand() -> Stand:
    return Stand()


@pytest.fixture
def engines(stand: Stand) -> Iterator[dict[str, str]]:
    """Адреса штатных серверов стенда: docling, mineru, paddle, ollama."""
    apps = {"docling": docling_app, "mineru": mineru_app, "paddle": paddle_app, "ollama": ollama_app}
    with ExitStack() as stack:
        yield {name: stack.enter_context(serve(build(stand))) for name, build in apps.items()}


def smoke(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--timeout", "60", *args], capture_output=True, text=True, timeout=120
    )


def lines(result: subprocess.CompletedProcess) -> list[str]:
    assert result.stderr == "", result.stderr
    return result.stdout.splitlines()


# --- Образцы ------------------------------------------------------------------------------------


def test_samples_match_what_the_script_looks_for():
    with pymupdf.open(SAMPLES / "ru-text.pdf") as doc:
        text = doc[0].get_text().lower().replace("ё", "е")
    with pymupdf.open(SAMPLES / "ru-scan.png") as scan:
        scan_text = scan[0].get_text()

    for word in ("подшипник", "поставщик", "покупатель", "ромашка", "оплат"):
        assert word in text  # слова, по которым скрипт узнаёт распознанный образец
    assert scan_text.strip() == ""  # скан — картинка без текстового слоя: его обязан прочитать OCR
    assert (SAMPLES / "ru-scan.png").stat().st_size < 100_000


# --- Движки по отдельности ---------------------------------------------------------------------------


def test_healthy_engines_pass(stand: Stand, engines: dict[str, str]):
    result = smoke(
        "--docling", engines["docling"], "--mineru", engines["mineru"],
        "--paddle", engines["paddle"], "--ollama", engines["ollama"],
    )  # fmt: skip
    output = lines(result)

    assert result.returncode == 0, result.stdout
    assert [line[:6] for line in output] == ["[ OK ]"] * 4
    assert output[0].startswith("[ OK ] Docling: русский текст распознан (5 из 5 слов образца)")
    assert output[1].startswith(
        "[ OK ] MinerU: русский текст распознан (5 из 5 слов образца), уровень standard"
    )
    assert output[2].startswith("[ OK ] PaddleOCR-VL: русский текст распознан")
    assert "Ollama (qwen3-vl-doc): русский текст распознан" in output[3]
    assert "модель целиком в видеопамяти (6.1 ГБ)" in output[3]
    # Движкам ушли образцы из репозитория; за собой скрипт убрал
    assert stand.docling_requests[0]["content"] == (SAMPLES / "ru-text.pdf").read_bytes()
    assert next(iter(stand.mineru_tasks.values()))["content"] == (SAMPLES / "ru-scan.png").read_bytes()
    assert stand.paddle_jobs == {}
    assert len(stand.ollama_requests[0]["messages"][0]["images"]) == 1


def test_mineru_basic_tier_is_caught_by_missing_cyrillic(stand: Stand, engines: dict[str, str]):
    stand.mineru_tiers = ("flash", "basic")  # сервер запущен с --tier basic, как в исходной инструкции
    result = smoke("--mineru", engines["mineru"])
    [line] = lines(result)

    assert result.returncode == 1
    assert line.startswith("[FAIL] MinerU: в ответе нет кириллицы")
    assert "Задачу выполнил уровень basic, сервер разрешает: flash, basic" in line
    assert "поставьте MINERU_TIER=standard" in line


def test_mineru_that_failed_to_load_models_is_reported(stand: Stand, engines: dict[str, str]):
    stand.mineru_preload_error = "CUDA out of memory"
    [line] = lines(smoke("--mineru", engines["mineru"]))

    assert line.startswith("[FAIL] MinerU: /v1/health -> HTTP 503")
    assert "CUDA out of memory" in line and "docker logs mineru-api" in line


@pytest.mark.parametrize(
    "pages, problem",
    [
        (["Lorem ipsum dolor sit amet 7707083893"], "в ответе нет кириллицы (37 символов без неё)"),
        (["Поставщик и ещё какой-то текст"], "текст образца не узнан: найдено слов 1 из 5"),
    ],
)
def test_wrong_recognition_is_not_accepted(stand: Stand, engines: dict[str, str], pages, problem: str):
    stand.paddle_pages = pages
    result = smoke("--paddle", engines["paddle"])
    [line] = lines(result)

    assert result.returncode == 1
    assert line.startswith(f"[FAIL] PaddleOCR-VL: {problem}")


def test_failed_engine_job_is_reported_and_removed(stand: Stand, engines: dict[str, str]):
    stand.paddle_error = "ResourceExhaustedError: Out of memory error on GPU 0"
    [line] = lines(smoke("--paddle", engines["paddle"]))

    assert line == f"[FAIL] PaddleOCR-VL: задача завершилась со статусом failed: {stand.paddle_error}"
    assert stand.paddle_jobs == {}


def test_ollama_model_partly_on_cpu_fails(stand: Stand, engines: dict[str, str]):
    stand.ollama_gpu_share = 0.8  # не поместилась в видеопамять: в `ollama ps` было бы «20%/80% CPU/GPU»
    result = smoke("--ollama", engines["ollama"])
    [line] = lines(result)

    assert result.returncode == 1
    assert "модель qwen3-vl-doc не поместилась в видеопамять: на GPU 80 %" in line


def test_missing_ollama_profile_is_reported(engines: dict[str, str]):
    [line] = lines(smoke("--ollama", engines["ollama"], "--model", "qwen3-vl-best"))

    assert line.startswith(
        "[FAIL] Ollama (qwen3-vl-best): модели qwen3-vl-best нет в Ollama (есть: qwen3-vl:4b"
    )
    assert line.endswith("выполните install.sh models")


def test_exit_code_counts_failed_checks(engines: dict[str, str]):
    result = smoke("--docling", DEAD, "--mineru", DEAD, "--ollama", engines["ollama"])
    output = lines(result)

    assert result.returncode == 2
    assert output[0] == "[FAIL] Docling: нет соединения с 127.0.0.1:9: [Errno 111] Connection refused"
    assert output[1].startswith("[FAIL] MinerU: нет соединения")
    assert output[2].startswith("[ OK ] Ollama")


def test_nothing_to_check_is_an_error():
    result = smoke()

    assert result.returncode == 2
    assert "не указано, что проверять" in result.stderr


# --- Вся боевая цепочка: скрипт -> оркестратор -> шлюз -> серверы стенда ---------------------------------


@pytest.fixture
def orchestrator_url(stand: Stand, engines: dict[str, str]) -> Iterator[str]:
    """Настоящие оркестратор и шлюз на портах 127.0.0.1; Docker и движки — стенд, Qwen3-VL — mock-engines."""
    with ExitStack() as stack:
        docker_url = stack.enter_context(serve(docker_app(stand)))
        vlm_url = stack.enter_context(serve(create_mock_engines(delay_scale=0.05)))
        gateway_settings = GatewaySettings(
            docling_serve_url=engines["docling"],
            mineru_api_url=engines["mineru"],
            paddle_api_url=engines["paddle"],
            poll_interval_sec=0.01,
        )
        # Сокет Docker заменён портом стенда; «контейнеры» движков там — просто записи о состоянии
        gateway = create_gateway(gateway_settings, docker=httpx.AsyncClient(base_url=docker_url))
        gateway_url = stack.enter_context(serve(gateway))
        settings = Settings(
            _env_file=None,
            docling_url=f"{gateway_url}/docling",
            mineru_url=f"{gateway_url}/mineru",
            paddle_url=f"{gateway_url}/paddle",
            ollama_url=vlm_url,
            api_key="s3cret",
        )
        yield stack.enter_context(serve(create_orchestrator(settings, redis=FakeAsyncRedis())))


def test_whole_chain_passes_through_real_orchestrator_and_gateway(stand: Stand, orchestrator_url: str):
    result = smoke("--orchestrator", orchestrator_url, "--api-key", "s3cret", "--with-paddle")
    [line] = lines(result)

    assert result.returncode == 0, result.stdout
    assert line.startswith(
        "[ OK ] doc-orchestrator: движки: docling, mineru+qwen3-vl, paddle; счёт разобран в JSON"
    )
    assert "ИНН продавца 7707083893" in line
    # GPU 0 переключён с MinerU на PaddleOCR-VL через «Docker», и вдвоём в VRAM они не оказывались
    assert stand.containers == {MINERU: "exited", PADDLE: "running"}
    assert stand.docker_calls()[-2:] == ["docker stop mineru-api t=30", "docker start paddle-gpu0"]
    assert stand.vram_conflicts == 0


def test_whole_chain_without_paddle_leaves_gpu0_to_mineru(stand: Stand, orchestrator_url: str):
    result = smoke("--orchestrator", orchestrator_url, "--api-key", "s3cret")

    assert result.returncode == 0, result.stdout
    assert stand.containers == {MINERU: "running", PADDLE: "created"}


def test_wrong_api_key_is_reported(orchestrator_url: str):
    result = smoke("--orchestrator", orchestrator_url, "--api-key", "wrong")
    [line] = lines(result)

    assert result.returncode == 1
    assert line.startswith("[FAIL] doc-orchestrator: PUT /process -> HTTP 401")


def test_engine_failure_behind_orchestrator_is_reported(stand: Stand, orchestrator_url: str):
    stand.mineru_tiers = ("flash", "basic")  # MinerU не читает кириллицу
    result = smoke("--orchestrator", orchestrator_url, "--api-key", "s3cret")
    [line] = lines(result)

    assert result.returncode == 1
    assert line.startswith("[FAIL] doc-orchestrator: в ответе нет кириллицы")
    assert "Файл ru-scan.png, движок mineru" in line
