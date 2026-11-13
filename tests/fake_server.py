"""Имитация сервера для тестов deploy/install.sh: подменяет системные команды.

Скрипт установки запускается с PATH, в котором лежат только обёртки над этим файлом и настоящие
текстовые утилиты (sed, awk, grep...). Каждая «команда» записывает свой вызов в журнал и меняет
состояние воображаемого сервера (JSON-файл): пакеты, драйвер, образы и контейнеры Docker, модели
Ollama. Так проверяется логика скрипта — порядок шагов, повторный запуск, отказы, — а не сами apt и Docker.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

# Настоящие текстовые утилиты, которыми пользуется скрипт; всё остальное в PATH — подмены
REAL_TOOLS = (
    "awk", "cat", "chmod", "cmp", "cp", "cut", "date", "dirname", "grep", "head",
    "install", "mkdir", "mktemp", "od", "rm", "sed", "tail", "tee", "touch", "tr",
)  # fmt: skip
ALWAYS = (
    "apt-get", "apt-cache", "dpkg", "dpkg-query", "id", "uname", "df",
    "wget", "curl", "gpg", "git", "systemctl", "python3", "sleep",
)  # fmt: skip

KERNEL = "6.12.48+deb13-amd64"
OS_RELEASE = (
    'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nVERSION_ID="13"\nVERSION_CODENAME=trixie\nID=debian\n'
)
APT_SOURCES = "Types: deb\nURIs: http://deb.debian.org/debian\nSuites: trixie\nComponents: main{extra}\n"
MINERU_DOCKERFILE = (
    "FROM vllm/vllm-openai:v0.21.0\n"
    "RUN python3 -m pip install -U 'mineru[torch]>=4.0,<5' --break-system-packages\n"
)
PORTS = {
    5001: "docling-serve",
    11434: "ollama-vlm",
    8002: "mineru-api",
    8003: "paddle-gpu0",
    8000: "doc-orchestrator",
}
GPU0 = ("mineru-api", "paddle-gpu0")
# Так MinerU отвечает на /v1/health, когда не смог загрузить модели: сервер работает, но бесполезен
BROKEN_REPLY = '{"error": {"code": "model_preload_failed", "message": "CUDA out of memory"}}'


def new_state(**changes) -> dict:
    """Сервер «из коробки»: Debian 13 со старым драйвером NVIDIA из репозитория Debian, без Docker."""
    state = {
        "uid": 0,
        "packages": {
            "nvidia-driver": "550.163.01-2",
            "libnvidia-glcore:amd64": "550.163.01-2",
            "nvidia-kernel-dkms": "550.163.01-2",
            "firmware-nvidia-gsp": "550.163.01-2",
            "coreutils": "9.7-3",
        },
        "available": [f"linux-headers-{KERNEL}", "linux-headers-amd64"],  # что есть в репозиториях
        "nvidia_repo": False,  # подключён ли репозиторий NVIDIA (после установки cuda-keyring)
        "driver_versions": ["620.70.06-1", "595.112.07-1", "595.91.07-1", "590.48.01-1"],
        "running_driver": "550.163.01",  # с этим драйвером работает ядро; меняется только «перезагрузкой»
        "gpus": 2,
        "gpu_processes": [],
        "secure_boot": False,
        "nvidia_runtime": False,  # настроен ли Docker на проброс видеокарт
        "container_gpus": 2,  # сколько видеокарт видно из контейнера
        "images": [],
        "networks": [],
        "containers": {},  # имя -> running / exited / created
        "never_ready": [],  # контейнеры, чей сервер так и не начинает отвечать
        "broken": [],  # контейнеры, чей сервер запустился, но отвечает 503 (модели не загрузились)
        "crashing": [],  # контейнеры, чей процесс падает сразу после запуска
        "boot_polls": {},  # сколько ещё опросов сервер контейнера «грузит модель»
        "models": [],  # модели в Ollama
        "modelfiles": {},  # профиль -> текст Modelfile
        "built_dockerfile": "",  # Dockerfile, с которым собран mineru:4
        "smoke_fails": [],  # ключи smoke_test.py, при которых он завершается ошибкой
        "vram_conflicts": 0,  # сколько раз оба движка GPU 0 были запущены одновременно
    }
    state.update(changes)
    return state


def installed_server(**changes) -> dict:
    """Сервер после успешной установки всего стека (кроме оркестратора)."""
    state = new_state(
        packages={
            "nvidia-driver": "595.112.07-1",
            "cuda-keyring": "1.1-1",
            "docker-ce": "29.0.1",
            "nvidia-container-toolkit": "1.20.1-1",
        },
        nvidia_repo=True,
        running_driver="595.112.07",
        nvidia_runtime=True,
        images=["mineru:4"],
        networks=["doc-net"],
        containers={
            "doc-redis": "running",
            "docling-serve": "running",
            "ollama-vlm": "running",
            "mineru-api": "running",
            "mineru-webui": "running",
            "paddle-gpu0": "created",
        },
        models=[
            "qwen3-vl:4b-instruct-q8_0",
            "qwen3-vl:8b-instruct-q4_K_M",
            "qwen3-vl-doc:latest",
            "qwen3-vl-hq:latest",
        ],
    )
    state.update(changes)
    return state


class FakeServer:
    """Каталог с подменёнными командами и состоянием сервера. Тесты запускают в нём install.sh."""

    def __init__(self, root: Path, state: dict, contrib: bool = True, os_release: str = OS_RELEASE) -> None:
        self.root = root
        self.bin = root / "bin"
        self.sysroot = root / "sysroot"
        self.stack_dir = root / "opt" / "ai-doc-stack"
        self.state_file = root / "state.json"
        self.log_file = root / "calls.log"
        self.bin.mkdir()
        (self.sysroot / "etc/apt/sources.list.d").mkdir(parents=True)
        (self.sysroot / "proc").mkdir()
        (self.sysroot / "etc/os-release").write_text(os_release)
        extra = " contrib non-free non-free-firmware" if contrib else ""
        (self.sysroot / "etc/apt/sources.list.d/debian.sources").write_text(APT_SOURCES.format(extra=extra))
        (self.sysroot / "proc/meminfo").write_text("MemTotal:       32780000 kB\n")
        self.log_file.write_text("")
        for tool in REAL_TOOLS:
            (self.bin / tool).symlink_to(shutil.which(tool))
        for name in ALWAYS:
            add_command(self.bin, name)
        (self.bin / "sleep").write_text("#!/bin/sh\nexit 0\n")  # ждать в тестах незачем
        self.write(state)
        sync_optional_commands(self.bin, state)

    def read(self) -> dict:
        return json.loads(self.state_file.read_text())

    def write(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1))

    def update(self, **changes) -> None:
        state = self.read() | changes
        self.write(state)
        sync_optional_commands(self.bin, state)

    def reboot(self) -> None:
        """Перезагрузка: ядро начинает работать с установленным драйвером; контейнеры поднимаются снова."""
        state = self.read()
        state["running_driver"] = state["packages"].get("nvidia-driver", "").split("-")[0]
        self.write(state)
        sync_optional_commands(self.bin, state)

    def calls(self, *prefixes: str) -> list[str]:
        """Журнал вызовов команд; prefixes оставляет только начинающиеся с этих слов."""
        lines = self.log_file.read_text().splitlines()
        return [line for line in lines if not prefixes or line.startswith(prefixes)]

    def clear_calls(self) -> None:
        self.log_file.write_text("")

    def env(self) -> dict[str, str]:
        return {
            "PATH": str(self.bin),
            "SYSROOT": str(self.sysroot),
            "STACK_DIR": str(self.stack_dir),
            "STUB_STATE": str(self.state_file),
            "STUB_LOG": str(self.log_file),
            "STUB_BIN": str(self.bin),
            "LANG": "C.UTF-8",
        }


def add_command(bin_dir: Path, name: str) -> None:
    wrapper = bin_dir / name
    wrapper.unlink(missing_ok=True)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{__file__}" {name} "$@"\n')
    wrapper.chmod(0o755)


def sync_optional_commands(bin_dir: Path, state: dict) -> None:
    """Команды, которых на сервере может не быть: появляются вместе со своими пакетами."""
    packages = state["packages"]
    present = {
        "docker": "docker-ce" in packages,
        "nvidia-ctk": "nvidia-container-toolkit" in packages,
        "nvidia-smi": "nvidia-driver" in packages,
        "mokutil": state["secure_boot"],
    }
    for name, installed in present.items():
        if installed:
            add_command(bin_dir, name)
        else:
            (bin_dir / name).unlink(missing_ok=True)


# --- Подменённые команды (выполняются в отдельном процессе) ---------------------------------------


def fail(message: str, code: int = 1):
    print(message, file=sys.stderr)
    sys.exit(code)


def start(state: dict, name: str) -> None:
    state["containers"][name] = "running"
    state["boot_polls"][name] = 1
    if all(state["containers"].get(engine) == "running" for engine in GPU0):
        state["vram_conflicts"] += 1


def compose(state: dict, args: list[str]) -> None:
    """docker compose [--project-directory D] [-f FILE] [--profile P] up|create|version ..."""
    profiles, file, rest = set(), None, []
    items = iter(args)
    for arg in items:
        if arg == "--profile":
            profiles.add(next(items))
        elif arg == "-f":
            file = next(items)
        elif arg == "--project-directory":
            next(items)
        else:
            rest.append(arg)
    if rest[0] == "version":
        print("Docker Compose version v2.40.0")
        return
    services = yaml.safe_load(Path(file).read_text())["services"]
    chosen = [arg for arg in rest[1:] if not arg.startswith("-")]
    if not chosen:  # без списка — все сервисы включённых профилей
        chosen = [
            name
            for name, spec in services.items()
            if not spec.get("profiles") or profiles & set(spec["profiles"])
        ]
    for name in chosen:
        spec = services[name]
        container = spec.get("container_name", name)
        image = spec.get("image", name)
        if "build" not in spec and image.split(":")[0] == "mineru" and image not in state["images"]:
            fail(f"Error response from daemon: pull access denied for {image}")
        if image not in state["images"]:
            state["images"].append(image)  # образ скачан или собран
        if rest[0] == "up":
            start(state, container)
        elif rest[0] == "create":
            state["containers"].setdefault(container, "created")


def docker(state: dict, args: list[str]) -> None:
    command, rest = args[0], args[1:]
    if command == "--version":
        print("Docker version 29.0.1, build eedd969")
    elif command == "compose":
        compose(state, rest)
    elif command == "info":
        runtimes = {"runc": {}} | ({"nvidia": {}} if state["nvidia_runtime"] else {})
        print(json.dumps(runtimes))
    elif command == "image":  # image inspect NAME
        if rest[1] not in state["images"]:
            fail(f"Error response from daemon: No such image: {rest[1]}")
        print("[{}]")
    elif command == "inspect":  # inspect -f FORMAT NAME
        if rest[-1] not in state["containers"]:
            fail(f"Error: No such object: {rest[-1]}")
        print(state["containers"][rest[-1]])
    elif command == "network":
        if rest[0] == "inspect" and rest[1] not in state["networks"]:
            fail(f"Error response from daemon: network {rest[1]} not found")
        if rest[0] == "create":
            state["networks"].append(rest[1])
    elif command == "build":  # build -t TAG -f DOCKERFILE CONTEXT
        state["built_dockerfile"] = Path(rest[rest.index("-f") + 1]).read_text()
        state["images"].append(rest[rest.index("-t") + 1])
    elif command == "run":  # run --rm --gpus all IMAGE nvidia-smi -L
        if not state["nvidia_runtime"]:
            fail(
                "docker: Error response from daemon: could not select device driver with capabilities: gpu",
                125,
            )
        for index in range(state["container_gpus"]):
            print(f"GPU {index}: NVIDIA GeForce RTX 2070 SUPER (UUID: GPU-{index})")
    elif command in ("stop", "start"):
        if rest[0] not in state["containers"]:
            fail(f"Error response from daemon: No such container: {rest[0]}")
        if command == "start":
            start(state, rest[0])
        else:
            state["containers"][rest[0]] = "exited"
        print(rest[0])
    elif command == "exec":
        ollama(state, [arg for arg in rest if arg != "-i"][1:])
    elif command == "logs":
        print(f"{rest[-1]}: RuntimeError: CUDA out of memory")
    elif command == "ps":
        for name, status in state["containers"].items():
            print(f"{name}\t{status}")
    else:
        fail(f"docker: неизвестная команда в тесте: {args}")


def ollama(state: dict, args: list[str]) -> None:
    """Команды в контейнере ollama-vlm: ollama list|pull|ps и sh -c 'cat > F && ollama create NAME -f F'."""
    if args[0] == "sh":
        name = args[2].split("ollama create ")[1].split()[0]
        state["modelfiles"][name] = sys.stdin.read()
        if f"{name}:latest" not in state["models"]:
            state["models"].append(f"{name}:latest")
    elif args[1] == "list":
        print("NAME    ID    SIZE    MODIFIED")
        for model in state["models"]:
            print(f"{model}    c6aab124f79d    5.1 GB    2 minutes ago")
    elif args[1] == "pull":
        state["models"].append(args[2])
    elif args[1] == "ps":
        print("NAME    ID    SIZE    PROCESSOR    UNTIL")
        print("qwen3-vl-doc:latest    c6aa    6.2 GB    100% GPU    24 hours from now")


def apt_get(state: dict, args: list[str]) -> None:
    command = args[0]
    names = [arg for arg in args[1:] if not arg.startswith("-")]
    packages = state["packages"]
    if command in ("purge", "remove"):
        for name in names:
            packages.pop(name, None)
    elif command == "install":
        for name in names:
            if "cuda-keyring" in name:  # локальный .deb: подключает репозиторий NVIDIA
                if not Path(name).is_file():
                    fail(f"E: Unsupported file {name} given on commandline", 100)
                packages["cuda-keyring"] = "1.1-1"
                state["nvidia_repo"] = True
            elif name.startswith("cuda-drivers="):
                version = name.split("=")[1]
                if not state["nvidia_repo"] or version not in state["driver_versions"]:
                    fail(f"E: Version '{version}' for 'cuda-drivers' was not found", 100)
                packages["cuda-drivers"] = packages["nvidia-driver"] = version
            elif name.startswith("nvidia-driver-pinning-") and not state["nvidia_repo"]:
                fail(f"E: Unable to locate package {name}", 100)
            else:
                packages[name] = "1.0"


def nvidia_smi(state: dict, args: list[str]) -> None:
    if not state["running_driver"]:
        fail("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", 9)
    query = next((arg for arg in args if arg.startswith("--query")), "")
    if query == "--query-gpu=driver_version":
        print("\n".join([state["running_driver"]] * state["gpus"]))
    elif query == "--query-gpu=index":
        print("\n".join(str(index) for index in range(state["gpus"])))
    elif query.startswith("--query-gpu="):
        for index in range(state["gpus"]):
            print(f"{index}, NVIDIA GeForce RTX 2070 SUPER, 8192 MiB, 12 MiB")
    elif query.startswith("--query-compute-apps"):
        print("\n".join(state["gpu_processes"]), end="")


def curl(state: dict, args: list[str]) -> None:
    url = next(arg for arg in args if arg.startswith("http"))
    if "--max-time" in args:  # опрос сервера из wait_http: с ключом -w — код ответа, без него — тело
        port = int(url.split(":")[2].split("/")[0])
        container = PORTS[port]
        if container in state["crashing"] and state["containers"].get(container) == "running":
            state["containers"][container] = "exited"
        status = "200"
        if state["containers"].get(container) != "running" or container in state["never_ready"]:
            status = "000"
        elif state["boot_polls"].get(container, 0) > 0:
            state["boot_polls"][container] -= 1
            status = "000"
        elif container in state["broken"]:
            status = "503"
        if "-w" in args:
            print(status, end="")
        elif status == "503":
            print(BROKEN_REPLY, end="")
        if status == "000":  # соединение не установлено
            sys.exit(7)
    elif "-o" in args:
        Path(args[args.index("-o") + 1]).write_text(f"downloaded from {url}\n")
    elif url.endswith(".list"):
        print("deb https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /")
    else:
        print("-----BEGIN PGP PUBLIC KEY BLOCK-----")


def run_command(state: dict, name: str, args: list[str]) -> None:
    if name == "id":
        print(state["uid"])
    elif name == "uname":
        print(KERNEL)
    elif name == "df":
        print("Avail\n 250G")
    elif name == "dpkg":
        if args[0] == "--print-architecture":
            print("amd64")
        else:  # --get-selections ИМЕНА
            for package in args[1:]:
                if package in state["packages"]:
                    print(f"{package}\tinstall")
    elif name == "dpkg-query":  # -W -f=ФОРМАТ [ПАКЕТ]
        if len(args) == 3:
            if args[2] not in state["packages"]:
                fail(f"dpkg-query: no packages found matching {args[2]}")
            print(f"ii {state['packages'][args[2]]}", end="")
        else:
            for package in state["packages"]:
                print(f"ii {package}")
    elif name == "apt-cache":
        if args[0] == "show" and args[1] not in state["available"]:
            fail(f"E: No packages found: {args[1]}", 100)
        if args[0] == "madison" and state["nvidia_repo"]:
            for version in state["driver_versions"]:
                print(
                    f"cuda-drivers | {version} | https://developer.download.nvidia.com/compute/cuda Packages"
                )
    elif name == "apt-get":
        apt_get(state, args)
    elif name == "nvidia-smi":
        nvidia_smi(state, args)
    elif name == "mokutil":
        print("SecureBoot enabled" if state["secure_boot"] else "SecureBoot disabled")
    elif name == "wget":  # wget -q -O ФАЙЛ URL
        Path(args[args.index("-O") + 1]).write_text("deb\n")
    elif name == "curl":
        curl(state, args)
    elif name == "gpg":  # gpg --dearmor --yes -o ФАЙЛ
        Path(args[args.index("-o") + 1]).write_text(sys.stdin.read())
    elif name == "git":  # git clone --depth 1 --branch TAG URL DIR
        dockerfile = Path(args[-1]) / "docker/global/Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text(MINERU_DOCKERFILE)
    elif name == "nvidia-ctk":
        state["nvidia_runtime"] = True
    elif name == "python3":  # python3 .../smoke_test.py КЛЮЧИ
        if any(key in args for key in state["smoke_fails"]):
            print("[FAIL] проверка не пройдена")
            sys.exit(1)
        print("[ OK ] проверка пройдена")
    elif name == "docker":
        docker(state, args)


def main() -> None:
    name, args = sys.argv[1], sys.argv[2:]
    state_file = Path(os.environ["STUB_STATE"])
    state = json.loads(state_file.read_text())
    if name != "sleep":
        with open(os.environ["STUB_LOG"], "a") as log:
            log.write(" ".join([name, *args]) + "\n")
    try:
        run_command(state, name, args)
    finally:  # состояние сохраняется и тогда, когда команда завершилась ошибкой
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1))
        sync_optional_commands(Path(os.environ["STUB_BIN"]), state)


if __name__ == "__main__":
    main()
