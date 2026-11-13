"""deploy/install.sh: порядок шагов, повторные запуски и отказы — на воображаемом сервере.

Системные команды (apt, dpkg, docker, nvidia-smi...) подменены: см. fake_server.py.
Проверяется логика скрипта, а не сама установка: настоящих пакетов и контейнеров здесь нет.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from fake_server import FakeServer, installed_server, new_state

from app.config import Settings
from gateway.config import GatewaySettings

ROOT = Path(__file__).parent.parent
BASH = shutil.which("bash")
# С этих слов начинаются вызовы команд, меняющих систему
CHANGING = (
    "apt-get", "git", "wget", "systemctl", "nvidia-ctk",
    "docker run", "docker build", "docker stop", "docker start", "docker network create",
    "docker compose --project", "docker exec -i", "docker exec ollama-vlm ollama pull",
)  # fmt: skip


class Server(FakeServer):
    """Воображаемый сервер с копией репозитория: скрипт запускается из неё и пишет только в неё."""

    def __init__(self, root: Path, state: dict, **options) -> None:
        super().__init__(root, state, **options)
        self.repo = root / "doc-orchestrator"
        shutil.copytree(ROOT / "deploy", self.repo / "deploy")
        for name in (".env.example", "docker-compose.prod.yml"):
            shutil.copy(ROOT / name, self.repo / name)
        if "doc-redis" in state["containers"]:  # стек уже установлен: его файлы лежат на месте
            self.stack_dir.mkdir(parents=True)
            stack_source = ROOT / "deploy/ai-doc-stack"
            shutil.copy(stack_source / "docker-compose.yml", self.stack_dir / "docker-compose.yml")
            shutil.copy(stack_source / ".env.example", self.stack_dir / ".env")

    def install(self, *args: str, umask: int = 0o022, **env: str) -> subprocess.CompletedProcess:
        """Запускает install.sh; stdout и stderr — одним текстом в .stdout."""
        return subprocess.run(
            [BASH, str(self.repo / "deploy/install.sh"), *args],
            env=self.env() | env,
            umask=umask,  # маска прав новых файлов в оболочке администратора
            stdin=subprocess.DEVNULL,  # терминала нет: вопросы без --yes скрипт задать не может
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )

    def changes(self) -> list[str]:
        """Вызовы команд, меняющих систему."""
        return self.calls(*CHANGING)


@pytest.fixture
def make_server(tmp_path: Path):
    def make(state: dict | None = None, **options) -> Server:
        return Server(tmp_path, state or new_state(), **options)

    return make


def succeeded(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, result.stdout
    return result.stdout


def failed(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 1, result.stdout
    return result.stdout


# --- Справка, check, общие отказы ---------------------------------------------------------------


def test_help_needs_nothing(make_server):
    server = make_server()
    for args in ([], ["--help"]):
        output = succeeded(server.install(*args))
        assert "sudo bash deploy/install.sh [--yes] [--dry-run] <команда>" in output
    assert server.calls() == []


def test_unknown_command_is_rejected(make_server):
    assert "неизвестная команда «instal»" in failed(make_server().install("instal"))


def test_check_on_fresh_server_changes_nothing(make_server):
    server = make_server(new_state(gpu_processes=["4242, python3, 6120 MiB"]))
    output = succeeded(server.install("check"))

    assert "[ OK ] ОС: Debian GNU/Linux 13 (trixie)" in output
    assert "[ OK ] память: 31 ГБ" in output
    assert "[ OK ] свободно на диске под образы и модели: 250 ГБ" in output
    assert "[ !! ] драйвер NVIDIA: 550.163.01 — нужна ветка 595" in output
    assert "4242, python3, 6120 MiB" in output  # шаг 1 инструкции: кто занимает видеокарты
    assert "[ !! ] Docker с плагином compose не установлен" in output
    assert output.rstrip().endswith(
        "или всё по порядку: sudo bash %s all" % (server.repo / "deploy/install.sh")
    )
    assert "install.sh driver" in output
    assert server.changes() == []


def test_check_on_installed_server(make_server):
    state = installed_server(containers=installed_server()["containers"] | {"doc-orchestrator": "running"})
    state["containers"]["engine-gateway"] = "running"
    server = make_server(state)
    output = succeeded(server.install("check"))

    assert "[ !! ]" not in output
    assert "[ OK ] драйвер NVIDIA: 595.112.07" in output
    assert "[ OK ] GPU 0: работает mineru-api (paddle-gpu0: created)" in output
    assert "[ OK ] doc-orchestrator и engine-gateway работают" in output
    assert "всё установлено" in output
    assert server.changes() == []


NEXT_STEPS = {
    "reboot": lambda: new_state(packages={"nvidia-driver": "595.112.07-1"}),
    "install.sh docker": lambda: new_state(
        packages={"nvidia-driver": "595.112.07-1"}, running_driver="595.112.07"
    ),
    "install.sh mineru": lambda: installed_server(images=[], containers={}),
    "install.sh stack": lambda: installed_server(containers={}),
    "install.sh models": lambda: installed_server(models=[]),
    "install.sh orchestrator": lambda: installed_server(),
}


@pytest.mark.parametrize("expected", NEXT_STEPS)
def test_check_names_the_next_step(make_server, expected: str):
    output = succeeded(make_server(NEXT_STEPS[expected]()).install("check"))

    advice = output.split("==> Что дальше")[1].strip().splitlines()[0]
    assert advice.strip().endswith(expected), output


def test_both_gpu0_engines_running_is_flagged(make_server):
    state = installed_server()
    state["containers"]["paddle-gpu0"] = "running"
    output = succeeded(make_server(state).install("check"))

    assert "[ !! ] mineru-api и paddle-gpu0 запущены одновременно" in output


@pytest.mark.parametrize(
    "command", ["driver", "docker", "mineru", "stack", "models", "verify", "orchestrator", "all"]
)
def test_changing_commands_need_root(make_server, command: str):
    server = make_server(new_state(uid=1000))
    output = failed(server.install(command))

    assert "нужны права root: sudo bash" in output
    assert server.changes() == []


@pytest.mark.parametrize("command", ["driver", "docker"])
def test_other_system_is_rejected(make_server, command: str):
    ubuntu = 'PRETTY_NAME="Ubuntu 24.04.3 LTS"\nVERSION_ID="24.04"\nVERSION_CODENAME=noble\nID=ubuntu\n'
    server = make_server(os_release=ubuntu)
    output = failed(server.install("--yes", command))

    assert "скрипт рассчитан на Debian 13 (trixie), а здесь Ubuntu 24.04.3 LTS" in output
    assert server.changes() == []


# --- Шаг driver ---------------------------------------------------------------------------------


def test_driver_replaces_debian_driver_with_pinned_branch(make_server):
    state = new_state()
    state["packages"]["nvidia-container-toolkit"] = "1.20.1-1"  # остался от прежней установки
    server = make_server(state)
    output = succeeded(server.install("--yes", "driver"))

    apt = server.calls("apt-get")
    kernel = "6.12.48+deb13-amd64"
    assert apt == [
        "apt-get update",
        f"apt-get install -y linux-headers-{kernel} build-essential dkms ca-certificates wget curl gpg",
        "apt-get install -y linux-headers-amd64",  # метапакет: модуль пересоберётся после обновления ядра
        # Старый драйвер удалён целиком, Container Toolkit не тронут
        "apt-get purge -y nvidia-driver libnvidia-glcore:amd64 nvidia-kernel-dkms firmware-nvidia-gsp",
        "apt-get autoremove -y",
        apt[5],  # установка cuda-keyring из скачанного файла — проверена ниже
        "apt-get update",
        "apt-get install -y nvidia-driver-pinning-595",
        "apt-get install -y cuda-drivers=595.112.07-1",  # последняя версия ветки 595, а не 620
    ]
    keyring = apt[5].removeprefix("apt-get install -y ")
    assert keyring.endswith("/cuda-keyring_1.1-1_all.deb")
    repo = "https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64"
    assert server.calls("wget") == [f"wget -q -O {keyring} {repo}/cuda-keyring_1.1-1_all.deb"]
    assert "nvidia-container-toolkit" in server.read()["packages"]
    assert "==> Нужна перезагрузка" in output
    assert output.rstrip().endswith("install.sh driver")  # что запустить после перезагрузки


def test_driver_waits_for_reboot_without_reinstalling(make_server):
    server = make_server()
    succeeded(server.install("--yes", "driver"))
    server.clear_calls()

    output = succeeded(server.install("--yes", "driver"))  # перезагрузить забыли

    assert "драйвер 595.112.07-1 установлен, но ядро работает со старым (550.163.01)" in output
    assert "==> Нужна перезагрузка" in output
    assert server.changes() == []


def test_driver_is_left_alone_when_it_already_works(make_server):
    server = make_server()
    succeeded(server.install("--yes", "driver"))
    server.reboot()
    server.clear_calls()

    output = succeeded(server.install("driver"))

    assert "[ OK ] драйвер 595.112.07 уже работает" in output
    assert "Нужна перезагрузка" not in output
    assert server.changes() == []


def test_driver_asks_before_removing_old_one(make_server):
    server = make_server()
    output = failed(server.install("driver"))  # без --yes и без терминала

    assert "нужно подтверждение: «Удалить прежний драйвер и поставить ветку 595?»" in output
    assert not server.calls("apt-get purge")
    assert server.read()["packages"]["nvidia-driver"] == "550.163.01-2"


def test_driver_on_server_without_any_driver(make_server):
    server = make_server(new_state(packages={"coreutils": "9.7-3"}, running_driver=""))
    succeeded(server.install("driver"))  # удалять нечего — подтверждение не нужно

    assert not server.calls("apt-get purge", "apt-get autoremove")
    assert server.calls("apt-get")[-1] == "apt-get install -y cuda-drivers=595.112.07-1"


def test_driver_needs_contrib(make_server):
    server = make_server(contrib=False)
    output = failed(server.install("--yes", "driver"))

    assert "в источниках APT не включён раздел contrib" in output
    assert server.changes() == []


def test_driver_needs_headers_of_running_kernel(make_server):
    server = make_server(new_state(available=["linux-headers-amd64"]))  # ядро старее, чем в репозитории
    output = failed(server.install("--yes", "driver"))

    assert "для работающего ядра 6.12.48+deb13-amd64 в репозитории нет заголовков" in output
    assert server.calls("apt-get") == ["apt-get update"]


def test_driver_reports_branch_missing_in_repository(make_server):
    server = make_server(new_state(driver_versions=["620.70.06-1", "610.57.04-1"]))
    output = failed(server.install("--yes", "driver"))

    assert "в репозитории NVIDIA нет пакета cuda-drivers ветки 595" in output


def test_driver_warns_about_secure_boot(make_server):
    server = make_server(new_state(secure_boot=True))
    output = succeeded(server.install("--yes", "driver"))

    assert "включён Secure Boot" in output
    assert "mokutil --import /var/lib/dkms/mok.pub" in output


def test_failed_command_stops_the_script(make_server):
    state = new_state(packages={"coreutils": "9.7-3"}, running_driver="")
    state["driver_versions"] = []  # madison ниже ничего не вернёт...
    server = make_server(state)
    server.update(nvidia_repo=False)
    # ...а пакет-закрепитель без репозитория NVIDIA не ставится: wget «не скачал» файл
    (server.bin / "wget").write_text("#!/bin/sh\nexit 4\n")
    output = failed(server.install("--yes", "driver"))

    assert "ОШИБКА: команда не выполнена (код 4): wget -q -O" in output
    assert not server.calls("apt-get install -y nvidia-driver-pinning")


# --- Шаг docker ---------------------------------------------------------------------------------


def driver_ready(packages: dict | None = None, **changes) -> dict:
    """Драйвер 595 работает, Docker ещё не ставили. packages — что ещё установлено."""
    packages = {"nvidia-driver": "595.112.07-1"} | (packages or {})
    return new_state(packages=packages, running_driver="595.112.07", **changes)


def test_docker_installs_engine_toolkit_and_checks_gpus(make_server):
    server = make_server(driver_ready(packages={"docker.io": "26.1", "runc": "1.1"}))
    output = succeeded(server.install("docker"))

    assert server.calls("apt-get") == [
        "apt-get remove -y docker.io runc",  # пакеты Debian, конфликтующие с Docker CE
        "apt-get update",
        "apt-get install -y ca-certificates curl gpg",
        "apt-get update",
        "apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin "
        "docker-compose-plugin git",
        "apt-get update",
        "apt-get install -y nvidia-container-toolkit",
    ]
    apt_dir = server.sysroot / "etc/apt"
    assert (
        apt_dir / "keyrings/docker.asc"
    ).read_text() == "downloaded from https://download.docker.com/linux/debian/gpg\n"
    assert (apt_dir / "sources.list.d/docker.sources").read_text() == (
        "Types: deb\nURIs: https://download.docker.com/linux/debian\nSuites: trixie\nComponents: stable\n"
        "Architectures: amd64\nSigned-By: /etc/apt/keyrings/docker.asc\n"
    )
    keyring = "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    assert (server.sysroot / keyring.lstrip("/")).read_text().startswith("-----BEGIN PGP PUBLIC KEY BLOCK")
    assert (apt_dir / "sources.list.d/nvidia-container-toolkit.list").read_text() == (
        f"deb [signed-by={keyring}] https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /\n"
    )
    assert server.calls("nvidia-ctk", "systemctl") == [
        "nvidia-ctk runtime configure --runtime=docker",
        "systemctl restart docker",
    ]
    assert server.calls("docker run") == [
        "docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi -L"
    ]
    assert "[ OK ] из контейнера видны видеокарты: 2" in output


def test_docker_keeps_apt_files_readable_under_strict_umask(make_server):
    """При umask 077 (так настраивают защищённые серверы) ключи не прочитал бы пользователь _apt."""
    server = make_server(driver_ready())
    succeeded(server.install("docker", umask=0o077))

    for path in (
        "etc/apt/keyrings/docker.asc",
        "etc/apt/sources.list.d/docker.sources",
        "usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
        "etc/apt/sources.list.d/nvidia-container-toolkit.list",
    ):
        assert (server.sysroot / path).stat().st_mode & 0o777 == 0o644, path


def test_docker_step_repeated_changes_nothing(make_server):
    server = make_server(driver_ready())
    succeeded(server.install("docker"))
    server.clear_calls()

    output = succeeded(server.install("docker"))

    assert "[ OK ] Docker уже установлен: Docker version 29.0.1" in output
    assert "[ OK ] NVIDIA Container Toolkit уже установлен" in output
    # Docker не перезапускается зря: это остановило бы работающие контейнеры
    assert server.changes() == [
        "docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi -L"
    ]


def test_docker_reports_missing_gpu_in_container(make_server):
    server = make_server(driver_ready(container_gpus=1))
    output = failed(server.install("docker"))

    assert "из контейнера видно видеокарт: 1, а нужно 2" in output


def test_docker_needs_working_driver(make_server):
    server = make_server()  # драйвер 550 из Debian
    output = failed(server.install("docker"))

    assert "драйвер NVIDIA ветки 595 не работает" in output
    assert server.changes() == []


# --- Шаг mineru ---------------------------------------------------------------------------------


def test_mineru_image_is_built_from_pinned_release(make_server):
    server = make_server(installed_server(images=[]))
    succeeded(server.install("mineru"))

    src = server.stack_dir / "mineru-src"
    assert server.calls("git") == [
        f"git clone --depth 1 --branch mineru-4.2.1-released https://github.com/opendatalab/MinerU.git {src}"
    ]
    assert server.calls("docker build") == [
        f"docker build -t mineru:4 -f {src}/docker/global/Dockerfile {src}"
    ]
    # В образ попадает ровно та версия, под API которой написан шлюз, а не «последняя 4.x»
    assert (
        "pip install -U 'mineru[torch]==4.2.1' --break-system-packages" in server.read()["built_dockerfile"]
    )


def test_mineru_image_is_not_rebuilt_without_request(make_server):
    server = make_server(installed_server())
    output = succeeded(server.install("mineru"))
    assert "[ OK ] образ mineru:4 уже собран" in output
    assert server.changes() == []

    succeeded(server.install("mineru", REBUILD="1"))
    assert len(server.calls("docker build")) == 1


def test_mineru_version_can_be_chosen(make_server):
    server = make_server(installed_server(images=[]))
    succeeded(server.install("mineru", MINERU_VERSION="4.2.0"))

    assert "--branch mineru-4.2.0-released" in server.calls("git")[0]
    assert "'mineru[torch]==4.2.0'" in server.read()["built_dockerfile"]


# --- Шаг stack ----------------------------------------------------------------------------------


def stack_ready(**changes) -> dict:
    """Драйвер, Docker и образ MinerU готовы, стек ещё не запускали."""
    return installed_server(networks=[], containers={}, models=[], **changes)


def test_stack_installs_compose_file_and_starts_engines(make_server):
    server = make_server(stack_ready())
    output = succeeded(server.install("stack"))

    stack = server.stack_dir
    assert (stack / "docker-compose.yml").read_bytes() == (
        ROOT / "deploy/ai-doc-stack/docker-compose.yml"
    ).read_bytes()
    assert (stack / ".env").read_bytes() == (ROOT / "deploy/ai-doc-stack/.env.example").read_bytes()
    assert all((stack / folder).is_dir() for folder in ("ollama", "paddle", "redis"))
    compose = f"docker compose --project-directory {stack} -f {stack}/docker-compose.yml"
    assert server.calls("docker network create", "docker compose --project") == [
        "docker network create doc-net",
        f"{compose} --profile webui up -d",
        f"{compose} --profile manual-paddle create paddle-vl",  # создан, но не запущен
    ]
    assert server.read()["containers"] == {
        "doc-redis": "running",
        "docling-serve": "running",
        "ollama-vlm": "running",
        "mineru-api": "running",
        "mineru-webui": "running",
        "paddle-gpu0": "created",
    }
    assert "жду MinerU (загрузка моделей в видеопамять)" in output
    assert "[ OK ] стек запущен" in output


def test_stack_keeps_admin_settings_and_backs_up_changed_compose_file(make_server):
    server = make_server(stack_ready())
    server.stack_dir.mkdir(parents=True)
    (server.stack_dir / ".env").write_text("MINERU_TIER=basic\nBIND_ADDR=10.0.0.5\n")
    (server.stack_dir / "docker-compose.yml").write_text("services: {}\n")

    output = succeeded(server.install("stack"))

    assert (server.stack_dir / ".env").read_text() == "MINERU_TIER=basic\nBIND_ADDR=10.0.0.5\n"
    [backup] = server.stack_dir.glob("docker-compose.yml.*.bak")
    assert backup.read_text() == "services: {}\n"
    assert f"прежний сохранён как {backup}" in output
    # Порты открыты на адресе из BIND_ADDR — по нему же скрипт ждёт готовности движков
    assert any(call.endswith("--max-time 5 http://10.0.0.5:5001/health") for call in server.calls("curl"))


def test_stack_repeated_only_reapplies_compose_file(make_server):
    server = make_server(stack_ready())
    succeeded(server.install("stack"))
    server.clear_calls()

    succeeded(server.install("stack"))

    assert not list(server.stack_dir.glob("*.bak"))
    compose = (
        f"docker compose --project-directory {server.stack_dir} -f {server.stack_dir}/docker-compose.yml"
    )
    # Обе команды Docker Compose сам сводит к «ничего не менять», если настройки прежние
    assert server.changes() == [
        f"{compose} --profile webui up -d",
        f"{compose} --profile manual-paddle create paddle-vl",
    ]
    assert server.read()["containers"]["paddle-gpu0"] == "created"


def test_stack_does_not_start_mineru_while_paddle_holds_gpu0(make_server):
    state = installed_server()
    state["containers"].update({"mineru-api": "exited", "paddle-gpu0": "running"})  # так оставил оркестратор
    server = make_server(state)

    output = succeeded(server.install("stack"))

    assert "GPU 0 занят PaddleOCR-VL (paddle-gpu0): mineru-api сейчас не запускаю" in output
    assert server.calls("docker compose --project")[-1].endswith(
        "up -d redis docling ollama-vlm mineru-webui"
    )
    assert server.read()["containers"]["mineru-api"] == "exited"
    assert server.read()["vram_conflicts"] == 0


def test_stack_needs_mineru_image(make_server):
    server = make_server(stack_ready(images=[]))
    output = failed(server.install("stack"))

    assert "образ mineru:4 не собран: сначала sudo bash" in output
    assert server.changes() == []


def test_stack_shows_log_of_engine_that_does_not_start(make_server):
    server = make_server(stack_ready(never_ready=["mineru-api"]))
    output = failed(server.install("stack"))

    assert "| mineru-api: RuntimeError: CUDA out of memory" in output  # хвост docker logs
    assert "MinerU (загрузка моделей в видеопамять) не отвечает" in output


def test_stack_does_not_wait_for_engine_whose_container_died(make_server):
    """Процесс движка упал при старте: ждать 15 минут незачем — сразу лог контейнера."""
    server = make_server(stack_ready(crashing=["mineru-api"]))
    output = failed(server.install("stack"))

    assert "| mineru-api: RuntimeError: CUDA out of memory" in output
    assert "MinerU (загрузка моделей в видеопамять) не отвечает" in output
    assert len([call for call in server.calls("curl") if ":8002/v1/health" in call]) == 1


def test_stack_stops_at_once_when_engine_reports_failure(make_server):
    """MinerU, не загрузив модели, продолжает работать и отвечает 503: ждать 15 минут незачем."""
    server = make_server(stack_ready(broken=["mineru-api"]))
    output = failed(server.install("stack"))

    assert (
        "MinerU (загрузка моделей в видеопамять) сообщает о сбое (HTTP 503): "
        '{"error": {"code": "model_preload_failed", "message": "CUDA out of memory"}}'
    ) in output
    assert "| mineru-api: RuntimeError: CUDA out of memory" in output
    # Пока модели грузятся, сервер не отвечает (один опрос); получив 503, скрипт читает текст
    # ошибки и больше не опрашивает
    assert [call.split()[1:-1] for call in server.calls("curl") if ":8002/v1/health" in call] == [
        ["-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5"],
        ["-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5"],
        ["-s", "--max-time", "5"],
    ]


# --- Шаг models ---------------------------------------------------------------------------------


def test_models_are_pulled_and_profiles_created(make_server):
    server = make_server(installed_server(models=[]))
    succeeded(server.install("models"))

    assert server.calls("docker exec ollama-vlm ollama pull") == [
        "docker exec ollama-vlm ollama pull qwen3-vl:4b-instruct-q8_0",
        "docker exec ollama-vlm ollama pull qwen3-vl:8b-instruct-q4_K_M",
    ]
    settings = "PARAMETER num_ctx {}\nPARAMETER temperature 0.1\n"
    assert server.read()["modelfiles"] == {
        "qwen3-vl-doc": "FROM qwen3-vl:4b-instruct-q8_0\n" + settings.format(8192),
        "qwen3-vl-hq": "FROM qwen3-vl:8b-instruct-q4_K_M\n" + settings.format(4096),
    }


def test_models_already_present_are_not_pulled_again(make_server):
    server = make_server(installed_server())
    output = succeeded(server.install("models"))

    assert "[ OK ] модель qwen3-vl:4b-instruct-q8_0 уже скачана" in output
    assert not server.calls("docker exec ollama-vlm ollama pull")


def test_models_need_running_ollama(make_server):
    server = make_server(stack_ready())
    assert "контейнер ollama-vlm не запущен" in failed(server.install("models"))


# --- Шаг verify ---------------------------------------------------------------------------------


def smoke_calls(server: Server) -> list[str]:
    script = str(server.repo / "deploy/smoke_test.py")
    return [call.removeprefix(f"python3 {script} ") for call in server.calls("python3")]


def test_verify_checks_engines_and_lends_gpu0_to_paddle(make_server):
    server = make_server(installed_server())
    output = succeeded(server.install("verify"))

    assert smoke_calls(server) == [
        "--docling http://127.0.0.1:5001 --ollama http://127.0.0.1:11434 --mineru http://127.0.0.1:8002",
        "--paddle http://127.0.0.1:8003",
    ]
    stack = server.stack_dir
    compose = f"docker compose --project-directory {stack} -f {stack}/docker-compose.yml"
    assert server.calls("docker stop", "docker start", "docker compose --project", "python3") == [
        server.calls("python3")[0],
        "docker stop mineru-api",  # сначала освобождаем GPU 0...
        f"{compose} --profile manual-paddle up -d paddle-vl",
        server.calls("python3")[1],
        "docker stop paddle-gpu0",  # ...а после проверки возвращаем его MinerU
        "docker start mineru-api",
    ]
    state = server.read()
    assert (state["containers"]["mineru-api"], state["containers"]["paddle-gpu0"]) == ("running", "exited")
    assert state["vram_conflicts"] == 0
    assert "[ OK ] все проверки пройдены" in output


@pytest.mark.parametrize("broken", ["--mineru", "--paddle"])
def test_verify_fails_but_returns_gpu0_to_mineru(make_server, broken: str):
    server = make_server(installed_server(smoke_fails=[broken]))
    output = failed(server.install("verify"))

    assert "не все проверки пройдены" in output
    assert len(smoke_calls(server)) == 2  # отказ одного движка не отменяет проверку остальных
    assert server.read()["containers"]["mineru-api"] == "running"
    assert server.read()["containers"]["paddle-gpu0"] == "exited"


def test_verify_can_skip_paddle(make_server):
    server = make_server(installed_server())
    succeeded(server.install("verify", SKIP_PADDLE="1"))

    assert len(smoke_calls(server)) == 1
    assert not server.calls("docker stop")


def test_verify_goes_through_orchestrator_when_it_owns_gpu0(make_server):
    state = installed_server()
    state["containers"].update({"doc-orchestrator": "running", "engine-gateway": "running"})
    server = make_server(state)
    (server.repo / ".env").write_text("REDIS_URL=redis://doc-redis:6379/0\nAPI_KEY=s3cret\n")

    succeeded(server.install("verify"))

    assert smoke_calls(server)[1] == "--orchestrator http://127.0.0.1:8000 --api-key s3cret --with-paddle"
    assert not server.calls("docker stop", "docker start")  # контейнеры GPU 0 переключает только оркестратор


# --- Шаг orchestrator ---------------------------------------------------------------------------


def test_orchestrator_gets_settings_for_the_stack_and_starts(make_server):
    state = installed_server()
    del state["containers"][
        "paddle-gpu0"
    ]  # стек ставили вручную по инструкции: PaddleOCR-VL ещё не создавали
    server = make_server(state)
    output = succeeded(server.install("orchestrator"))

    env_file = server.repo / ".env"
    env = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines() if line and not line.startswith("#")
    )
    key = env.pop("API_KEY")
    assert len(key) == 48 and set(key) <= set("0123456789abcdef")
    assert key not in output  # ключ не попадает ни на экран, ни в журнал установки
    assert {
        name: env[name] for name in ("REDIS_URL", "DOCLING_URL", "MINERU_URL", "PADDLE_URL", "OLLAMA_URL")
    } == {
        "REDIS_URL": "redis://doc-redis:6379/0",
        "DOCLING_URL": "http://engine-gateway:8001/docling",
        "MINERU_URL": "http://engine-gateway:8001/mineru",
        "PADDLE_URL": "http://engine-gateway:8001/paddle",
        "OLLAMA_URL": "http://ollama-vlm:11434",
    }
    assert env["VLM_MODEL"] == "qwen3-vl-doc"
    assert env["HTTP_TIMEOUT_SEC"] == "300"  # остальные настройки — как в образце
    assert env_file.stat().st_mode & 0o777 == 0o600
    # Файл читается и оркестратором, и шлюзом (у шлюза все адреса — по умолчанию)
    assert Settings(_env_file=env_file).mineru_url == "http://engine-gateway:8001/mineru"
    assert GatewaySettings(_env_file=env_file) == GatewaySettings()

    stack, repo = server.stack_dir, server.repo
    assert server.calls("docker compose --project") == [
        f"docker compose --project-directory {stack} -f {stack}/docker-compose.yml "
        "--profile manual-paddle create paddle-vl",
        f"docker compose --project-directory {repo} -f {repo}/docker-compose.prod.yml up -d --build",
    ]
    containers = server.read()["containers"]
    assert (containers["doc-orchestrator"], containers["engine-gateway"]) == ("running", "running")
    assert containers["paddle-gpu0"] == "created"  # шлюзу нужен готовый контейнер, но запускает его он сам
    assert f"API Key                    = значение API_KEY из файла {env_file}" in output


def test_orchestrator_keeps_existing_settings(make_server):
    server = make_server(installed_server())
    (server.repo / ".env").write_text("API_KEY=мой-ключ\n")

    succeeded(server.install("orchestrator"))

    assert (server.repo / ".env").read_text() == "API_KEY=мой-ключ\n"


def test_orchestrator_needs_the_stack(make_server):
    server = make_server(stack_ready())
    output = failed(server.install("orchestrator"))

    assert f"в {server.stack_dir} нет docker-compose.yml: сначала sudo bash" in output
    assert not (server.repo / ".env").exists()


def test_orchestrator_needs_running_redis(make_server):
    state = installed_server()
    state["containers"]["doc-redis"] = "exited"
    server = make_server(state)
    output = failed(server.install("orchestrator"))

    assert "стек движков не запущен (нет контейнера doc-redis)" in output
    assert not (server.repo / ".env").exists()


# --- Ручное переключение GPU 0 ---------------------------------------------------------------------


def test_gpu0_is_switched_by_hand(make_server):
    server = make_server(installed_server())

    succeeded(server.install("gpu0", "paddle"))
    containers = server.read()["containers"]
    assert (containers["mineru-api"], containers["paddle-gpu0"]) == ("exited", "running")

    succeeded(server.install("gpu0", "mineru"))
    containers = server.read()["containers"]
    assert (containers["mineru-api"], containers["paddle-gpu0"]) == ("running", "exited")
    assert server.read()["vram_conflicts"] == 0


def test_gpu0_is_not_switched_behind_orchestrator(make_server):
    state = installed_server()
    state["containers"]["doc-orchestrator"] = "running"
    server = make_server(state)
    output = failed(server.install("gpu0", "paddle"))

    assert "GPU 0 переключает doc-orchestrator: вручную нельзя" in output
    assert server.changes() == []


def test_gpu0_needs_engine_name(make_server):
    assert "укажите, кому отдать GPU 0" in failed(make_server(installed_server()).install("gpu0"))


# --- Всё по порядку ----------------------------------------------------------------------------------


def test_all_pauses_for_reboot_then_finishes_and_is_repeatable(make_server):
    server = make_server()

    # 1-й запуск: только драйвер, дальше — после перезагрузки
    output = succeeded(server.install("--yes", "all"))
    assert "==> Нужна перезагрузка" in output
    assert output.rstrip().endswith("install.sh all")
    assert not server.calls("apt-get install -y docker-ce", "docker")

    # 2-й запуск, после перезагрузки: всё остальное
    server.reboot()
    server.clear_calls()
    output = succeeded(server.install("--yes", "all"))
    assert "[ OK ] установка завершена" in output
    state = server.read()
    assert state["containers"] == {
        "doc-redis": "running",
        "docling-serve": "running",
        "ollama-vlm": "running",
        "mineru-api": "running",
        "mineru-webui": "running",
        "paddle-gpu0": "exited",  # проверен и остановлен: GPU 0 возвращён MinerU
        "engine-gateway": "running",
        "doc-orchestrator": "running",
    }
    assert state["vram_conflicts"] == 0
    assert set(state["modelfiles"]) == {"qwen3-vl-doc", "qwen3-vl-hq"}
    key = dict(line.split("=", 1) for line in (server.repo / ".env").read_text().splitlines() if "=" in line)[
        "API_KEY"
    ]
    assert smoke_calls(server)[-1] == f"--orchestrator http://127.0.0.1:8000 --api-key {key} --with-paddle"
    log = (server.sysroot / "var/log/ai-doc-stack-install.log").read_text()
    assert "install.sh all" in log and "установка завершена" in log
    assert "--api-key *** --with-paddle" in output
    assert key not in output and key not in log  # ключ не попадает ни на экран, ни в журнал
    assert "http://<IP сервера>:5001/ui" in output  # в конце — адреса интерфейсов

    # 3-й запуск: ничего не ставится и не пересобирается заново
    server.clear_calls()
    succeeded(server.install("--yes", "all"))
    assert not server.calls("apt-get", "git", "docker build", "docker network create", "systemctl")
    assert not server.calls("docker exec ollama-vlm ollama pull")
    assert server.read()["vram_conflicts"] == 0


def test_dry_run_shows_commands_and_changes_nothing(make_server):
    server = make_server(new_state(uid=1000))  # пробный запуск не требует root
    before = server.read()
    output = succeeded(server.install("--dry-run", "all"))

    assert (
        "    $ apt-get purge -y nvidia-driver libnvidia-glcore:amd64 nvidia-kernel-dkms firmware-nvidia-gsp"
        in output
    )
    assert "    $ apt-get install -y nvidia-driver-pinning-595" in output
    # План показан целиком — и та часть, что выполнится после перезагрузки
    assert output.index("Нужна перезагрузка") < output.index("После перезагрузки установка продолжится так")
    for later_step in (
        "$ docker build -t mineru:4",
        "docker-compose.prod.yml up -d --build",
        "smoke_test.py --orchestrator http://127.0.0.1:8000 --api-key *** --with-paddle",  # ключа ещё нет
    ):
        assert later_step in output
    assert "[ OK ]" not in output  # о несделанном как о сделанном не сообщается
    assert "видеопамять сейчас" not in output  # и ничего не измеряется
    assert "Это был пробный запуск (--dry-run): ничего не выполнялось и не менялось" in output
    assert server.changes() == []
    assert server.read() == before
    assert not server.stack_dir.exists()
    assert not (server.sysroot / "var/log").exists()


def test_dry_run_of_later_steps_shows_the_whole_plan(make_server):
    server = make_server(driver_ready())
    output = succeeded(server.install("--dry-run", "all"))

    for expected in (
        "$ apt-get install -y docker-ce docker-ce-cli",
        "$ docker build -t mineru:4",
        "docker-compose.yml --profile webui up -d",
        "$ docker exec ollama-vlm ollama pull qwen3-vl:4b-instruct-q8_0",
        "$ python3 deploy/smoke_test.py --paddle http://127.0.0.1:8003",
        "docker-compose.prod.yml up -d --build",
    ):
        assert expected in output
    assert server.changes() == []
    assert not (server.repo / ".env").exists()


def test_status_lists_containers(make_server):
    output = succeeded(make_server(installed_server()).install("status"))

    assert "      mineru-api        running" in output
    assert "      paddle-gpu0       created" in output
    assert "qwen3-vl-doc:latest" in output and "100% GPU" in output
