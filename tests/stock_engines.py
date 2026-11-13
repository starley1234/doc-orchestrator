"""Имитации штатных серверов движков и Docker Engine — для тестов шлюза и скриптов развёртывания.

В отличие от mock-engines (условный контракт AGENTS.md) эти серверы отвечают так же, как настоящие:
API воспроизведены по исходникам docling-serve v1.38.2, MinerU 4.2.1 (Tasks API), paddleocr-pdf-api v0.5.0
и по справочнику Docker Engine API. Движок GPU 0 «слушает порт», только пока запущен его контейнер.
"""

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

MINERU, PADDLE, DOCLING, OLLAMA = "mineru-api", "paddle-gpu0", "docling-serve", "ollama-vlm"

# Текст образца deploy/samples: его «распознают» движки стенда
SAMPLE_TEXT = (
    "Счёт на оплату № 128. Поставщик: ООО «Ромашка», ИНН 7707083893. Покупатель: АО «Вектор». "
    "Подшипник 6205-2RS."
)
# Так тот же текст читает OCR без кириллицы в словаре (MinerU 4, уровень basic)
SAMPLE_TEXT_NO_CYRILLIC = "Cyet Ha onnaty No 128 ot 14.03.2025. NHH 7707083893. 6205-2RS."

MINERU_TASK_FIELDS = {"files", "tier", "ocr_mode", "page_range", "output_formats"}
MINERU_FILE_FIELDS = {"file_name", "bytes", "url", "sha256sum", "page_range"}
DOCLING_TIMEOUT = (
    "Conversion is taking too long. The maximum wait time is configure as DOCLING_SERVE_MAX_SYNC_WAIT=300."
)
PADDLE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def picture(width: int, height: int, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 200, 200)).save(buffer, fmt)
    return buffer.getvalue()


@dataclass
class Stand:
    """Состояние стенда: контейнеры Docker и поведение серверов. Тесты меняют поля и читают журнал."""

    containers: dict[str, str] = field(default_factory=lambda: {MINERU: "running", PADDLE: "created"})
    boot_probes: int = 0  # столько обращений сервер «грузит модель» после запуска контейнера
    crash_on_start: set[str] = field(default_factory=set)  # контейнеры, падающие сразу после запуска
    booting: dict[str, int] = field(default_factory=dict)
    calls: list[str] = field(
        default_factory=list
    )  # журнал: «docker start mineru-api», «mineru POST /v1/tasks»
    vram_conflicts: int = 0  # сколько раз оба движка GPU 0 оказались запущены одновременно

    # MinerU
    mineru_tiers: tuple[str, ...] = ("flash", "basic", "standard")  # сервер запущен с --tier standard
    mineru_polls: int = 2  # на каком по счёту опросе задача завершается
    mineru_error: str = ""  # непустая — задача завершается ошибкой
    mineru_preload_error: str = ""  # непустая — модели не загрузились: /v1/health отвечает 503
    # Непустая — движок внутри сервера упал: сервер работает, но отвечает 503, пока контейнер не перезапустят
    mineru_engine_dead: str = ""
    mineru_markdown: str = f"# Счёт\n\n{SAMPLE_TEXT}\n\n![](images/scheme.jpg)\n\nПодпись."
    mineru_images: dict[str, bytes] = field(default_factory=lambda: {"images/scheme.jpg": picture(640, 480)})
    mineru_archive: bytes | None = None  # задан — сервер отдаёт эти байты вместо ZIP с результатом
    mineru_public_url: str = ""  # задан — ссылки в ответах сервер строит от этого адреса (он за прокси)
    mineru_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # PaddleOCR-VL
    paddle_polls: int = 2
    paddle_error: str = ""
    paddle_pages: list[str] = field(
        default_factory=lambda: [f"| Товар |\n|---|\n| {SAMPLE_TEXT} |", "ПЕЧАТЬ"]
    )
    paddle_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Docling
    docling_status: str = "success"
    docling_http_error: int = 0  # ненулевой — сервер отвечает этим кодом (504 — не уложился в ожидание)
    docling_requests: list[dict[str, Any]] = field(default_factory=list)

    # Ollama
    ollama_gpu_share: float = 1.0  # доля модели в видеопамяти: 1.0 — «100% GPU»
    ollama_requests: list[dict[str, Any]] = field(default_factory=list)

    def is_listening(self, host: str) -> bool:
        """Отвечает ли сервер: его контейнер запущен и модель уже загружена."""
        if host not in self.containers:
            return True  # сервер без контейнера в этом стенде: Docling, Ollama
        if host in self.crash_on_start and self.containers[host] == "running":
            self.containers[host] = "exited"
        if self.containers[host] != "running":
            return False
        if self.booting.get(host, 0) > 0:
            self.booting[host] -= 1
            return False
        return True

    def docker_calls(self) -> list[str]:
        return [call for call in self.calls if call.startswith("docker")]


# --- Docker Engine API ----------------------------------------------------------------------


def docker_app(stand: Stand) -> FastAPI:
    app = FastAPI()

    def missing(name: str) -> JSONResponse:
        return JSONResponse({"message": f"No such container: {name}"}, status_code=404)

    @app.post("/containers/{name}/start")
    async def start(name: str) -> Response:
        stand.calls.append(f"docker start {name}")
        if name not in stand.containers:
            return missing(name)
        if stand.containers[name] == "running":
            return Response(status_code=304)
        stand.containers[name] = "running"
        stand.booting[name] = stand.boot_probes
        if name == MINERU:
            stand.mineru_engine_dead = ""  # новый процесс — новый движок
        if stand.containers.get(MINERU) == stand.containers.get(PADDLE) == "running":
            stand.vram_conflicts += 1  # две модели в 8 ГБ не помещаются
        return Response(status_code=204)

    @app.post("/containers/{name}/stop")
    async def stop(name: str, t: int = 10) -> Response:
        stand.calls.append(f"docker stop {name} t={t}")
        if name not in stand.containers:
            return missing(name)
        if stand.containers[name] != "running":
            return Response(status_code=304)
        stand.containers[name] = "exited"
        return Response(status_code=204)

    @app.get("/containers/{name}/json")
    async def inspect(name: str) -> Response:
        if name not in stand.containers:
            return missing(name)
        return JSONResponse({"Id": "0" * 64, "Name": f"/{name}", "State": {"Status": stand.containers[name]}})

    return app


# --- MinerU 4: Tasks API --------------------------------------------------------------------


def mineru_app(stand: Stand) -> FastAPI:
    app = FastAPI()

    def error(status: int, code: str, message: str, param: str | None = None) -> JSONResponse:
        body = {"code": code, "message": message} | ({"param": param} if param else {})
        return JSONResponse({"error": body}, status_code=status)

    def file_view(task: dict[str, Any], request: Request) -> dict[str, Any]:
        base_url = stand.mineru_public_url or str(request.base_url)  # настоящий сервер смотрит на Host
        view = {"file_id": task["file_id"], "file_name": task["file_name"], "status": task["status"]}
        if task["status"] == "waiting-file":
            view["upload"] = {
                "url": f"{base_url}v1/tasks/{task['id']}/uploads/{task['file_id']}",
                "headers": {"Content-Type": "application/octet-stream"},
                "expires_at": 1_800_003_600,
            }
        if task["status"] == "completed":
            view["parse"] = {"tier": task["tier"], "duration_ms": 800, "parser_version": "4.2.1"}
            view["result_url"] = f"{base_url}v1/files/{task['file_id']}/content"
        if task["status"] == "failed":
            view["error"] = {"code": "parse_failed", "message": stand.mineru_error}
        return view

    @app.get("/v1/health")
    async def health() -> Response:
        if stand.mineru_engine_dead:
            return error(503, "engine_dead", stand.mineru_engine_dead)
        if stand.mineru_preload_error:
            return error(503, "model_preload_failed", stand.mineru_preload_error)
        return JSONResponse({"status": "ok", "version": "4.2.1", "runtime": {"device": "gpu"}})

    @app.get("/v1/tiers")
    async def tiers() -> dict[str, Any]:
        data = [
            {"id": tier, "description": "…", "current_model": "mineru-4.2.1"} for tier in stand.mineru_tiers
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/tasks")
    async def create_task(request: Request) -> Response:
        stand.calls.append("mineru POST /v1/tasks")
        if stand.mineru_engine_dead:
            return error(503, "engine_dead", stand.mineru_engine_dead)
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            return error(415, "invalid_request", "Use application/json")
        body = await request.json()
        for extra in sorted(set(body) - MINERU_TASK_FIELDS):  # лишние поля сервер отклоняет
            return error(
                400, "invalid_request", f"Invalid request: {extra}: Extra inputs are not permitted", extra
            )
        files = body.get("files")
        if not isinstance(files, list) or len(files) != 1:
            return error(400, "invalid_request", "Invalid request: files", "files")
        item = files[0]
        name = item.get("file_name", "")
        if set(item) - MINERU_FILE_FIELDS or not name or any(char in name for char in "/\\:"):
            return error(400, "invalid_request", "Invalid file entry", "files.0.file_name")
        tier = body.get("tier") or next(t for t in ("standard", "basic", "flash") if t in stand.mineru_tiers)
        if tier not in stand.mineru_tiers:
            return error(400, "invalid_request", f"Unsupported tier {tier}", "tier")
        number = len(stand.mineru_tasks) + 1
        task = {
            "id": f"job_{number}",
            "file_id": f"file-{number}",
            "file_name": name,
            "bytes": item.get("bytes"),
            "tier": tier,
            "status": "waiting-file",
            "polls": 0,
            "content": b"",
        }
        stand.mineru_tasks[task["id"]] = task
        view = {"task_id": task["id"], "status": task["status"], "files": [file_view(task, request)]}
        return JSONResponse(view, status_code=202)

    @app.put("/v1/tasks/{task_id}/uploads/{file_id}")
    async def upload(task_id: str, file_id: str, request: Request) -> Response:
        stand.calls.append(f"mineru PUT upload {request.headers.get('content-type')}")
        task = stand.mineru_tasks.get(task_id)
        if task is None or task["file_id"] != file_id:
            return error(404, "upload_not_found", "No such upload")
        task["content"] = await request.body()
        if len(task["content"]) != task["bytes"]:
            return error(400, "upload_size_mismatch", "Uploaded size differs from the manifest")
        task["status"] = "pending"
        return JSONResponse({"task_id": task_id, "file_id": file_id, "status": "pending"})

    @app.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str, request: Request) -> Response:
        task = stand.mineru_tasks.get(task_id)
        if task is None:
            return error(404, "task_not_found", "Task not found")
        if task["status"] in ("pending", "running"):
            task["polls"] += 1
            task["status"] = "running"
            if task["polls"] >= stand.mineru_polls:
                task["status"] = "failed" if stand.mineru_error else "completed"
        return JSONResponse(
            {"task_id": task_id, "status": task["status"], "files": [file_view(task, request)]}
        )

    @app.post("/v1/tasks/{task_id}/cancel")
    async def cancel(task_id: str) -> Response:
        stand.calls.append("mineru POST cancel")
        task = stand.mineru_tasks.get(task_id)
        if task is None:
            return error(404, "task_not_found", "Task not found")
        task["status"] = "canceled"
        return JSONResponse({"task_id": task_id, "status": "canceled", "files": []})

    @app.get("/v1/files/{file_id}/content")
    async def result(file_id: str) -> Response:
        task = next((t for t in stand.mineru_tasks.values() if t["file_id"] == file_id), None)
        if task is None or task["status"] != "completed":
            return error(404, "file_not_found", "File not found")
        if stand.mineru_archive is not None:
            return Response(stand.mineru_archive, media_type="application/zip")
        # Уровень basic читает текст моделью PP-OCRv6, в словаре которой нет кириллицы
        markdown = SAMPLE_TEXT_NO_CYRILLIC if task["tier"] == "basic" else stand.mineru_markdown
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("markdown.md", markdown)
            archive.writestr("middle_json.json", "{}")
            archive.writestr("structured_content.json", "{}")
            archive.writestr("images/", b"")
            for name, data in stand.mineru_images.items():
                archive.writestr(name, data)
        return Response(buffer.getvalue(), media_type="application/zip")

    return app


# --- paddleocr-pdf-api ----------------------------------------------------------------------


def paddle_app(stand: Stand) -> FastAPI:
    app = FastAPI()

    def job_view(job: dict[str, Any]) -> dict[str, Any]:
        pages = len(stand.paddle_pages)
        done = pages if job["status"] == "completed" else 0
        return {
            "job_id": job["id"],
            "filename": job["filename"],
            "status": job["status"],
            "total_pages": pages,
            "processed_pages": done,
            "error": stand.paddle_error if job["status"] == "failed" else None,
        }

    def find(job_id: str) -> dict[str, Any]:
        if job_id not in stand.paddle_jobs:
            raise HTTPException(404, "Job not found")
        return stand.paddle_jobs[job_id]

    @app.post("/ocr")
    async def submit(file: UploadFile) -> dict[str, Any]:
        stand.calls.append(f"paddle POST /ocr {file.filename}")
        if PurePath(file.filename or "").suffix.lower() not in PADDLE_SUFFIXES:
            raise HTTPException(
                400, "Only PDF and image files (PNG, JPG, JPEG, BMP, TIFF, WEBP) are supported"
            )
        job_id = f"{len(stand.paddle_jobs) + 1:032x}"
        content = await file.read()
        stand.paddle_jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "queued",
            "polls": 0,
            "content": content,
        }
        return {"job_id": job_id, "filename": file.filename, "status": "queued"}

    @app.get("/ocr/{job_id}")
    async def status(job_id: str) -> dict[str, Any]:
        job = find(job_id)
        if job["status"] in ("queued", "processing"):
            job["polls"] += 1
            job["status"] = "processing"
            if job["polls"] >= stand.paddle_polls:
                job["status"] = "failed" if stand.paddle_error else "completed"
        return job_view(job)

    @app.get("/ocr/{job_id}/result")
    async def result(job_id: str) -> dict[str, Any]:
        job = find(job_id)
        pages = [{"page_num": number, "markdown": text} for number, text in enumerate(stand.paddle_pages, 1)]
        # Порядок страниц в ответе настоящего сервера задаёт база данных: клиент обязан сортировать сам
        return job_view(job) | {"pages": pages[::-1] if job["status"] == "completed" else []}

    @app.post("/ocr/{job_id}/cancel")
    async def cancel(job_id: str) -> dict[str, Any]:
        stand.calls.append("paddle POST cancel")
        job = find(job_id)
        job["status"] = "cancelled"
        return {"job_id": job_id, "status": "cancelling"}

    @app.delete("/ocr/{job_id}")
    async def delete(job_id: str) -> dict[str, str]:
        stand.calls.append("paddle DELETE job")
        find(job_id)
        del stand.paddle_jobs[job_id]
        return {"status": "deleted"}

    @app.get("/jobs")
    async def jobs() -> dict[str, Any]:
        return {"jobs": [job_view(job) for job in stand.paddle_jobs.values()]}

    return app


# --- docling-serve ---------------------------------------------------------------------------


def docling_app(stand: Stand) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/convert/file")
    async def convert(request: Request) -> Response:
        form = await request.form()
        upload = form["files"]
        content = await upload.read()
        options = {key: value for key, value in form.items() if key != "files"}
        stand.docling_requests.append({"filename": upload.filename, "content": content, "options": options})
        if stand.docling_http_error:
            return JSONResponse({"detail": DOCLING_TIMEOUT}, status_code=stand.docling_http_error)
        markdown: str | None = f"## Документ\n\n{SAMPLE_TEXT}"
        errors: list[dict[str, str]] = []
        status = stand.docling_status
        if PurePath(upload.filename).suffix in (".md", ".txt"):
            try:
                markdown = content.decode("utf-8")  # Docling читает текстовые файлы только в UTF-8
            except UnicodeDecodeError:
                status = "failure"
        if status == "failure":
            markdown = None
            errors = [
                {
                    "component_type": "document_backend",
                    "module_name": "Backend",
                    "error_message": "Input document is not valid.",
                }
            ]
        document = {
            "filename": upload.filename,
            "md_content": markdown,
            "json_content": None,
            "html_content": None,
            "text_content": None,
            "doctags_content": None,
        }
        reply = {
            "document": document,
            "status": status,
            "errors": errors,
            "processing_time": 0.4,
            "timings": {},
        }
        return JSONResponse(reply)

    return app


# --- Ollama -----------------------------------------------------------------------------------


def ollama_app(stand: Stand) -> FastAPI:
    app = FastAPI()
    size = 6_500_000_000

    @app.get("/api/version")
    async def version() -> dict[str, str]:
        return {"version": "0.40.16"}

    @app.get("/api/tags")
    async def tags() -> dict[str, Any]:
        names = [
            "qwen3-vl:4b-instruct-q8_0",
            "qwen3-vl:8b-instruct-q4_K_M",
            "qwen3-vl-doc:latest",
            "qwen3-vl-hq:latest",
        ]
        return {"models": [{"name": name, "model": name, "size": size} for name in names]}

    @app.post("/api/chat")
    async def chat(request: Request) -> dict[str, Any]:
        body = json.loads(await request.body())
        stand.ollama_requests.append(body)
        return {
            "model": body["model"],
            "message": {"role": "assistant", "content": SAMPLE_TEXT},
            "done": True,
        }

    @app.get("/api/ps")
    async def ps() -> dict[str, Any]:
        model = {"name": "qwen3-vl-doc:latest", "size": size, "size_vram": int(size * stand.ollama_gpu_share)}
        return {"models": [model] if stand.ollama_requests else []}

    return app


# --- Сборка стенда ----------------------------------------------------------------------------


def stand_apps(stand: Stand) -> dict[str, FastAPI]:
    """Серверы стенда по именам хостов — как в сети doc-net из инструкции по установке."""
    return {
        MINERU: mineru_app(stand),
        PADDLE: paddle_app(stand),
        DOCLING: docling_app(stand),
        OLLAMA: ollama_app(stand),
    }


class StandTransport(httpx.AsyncBaseTransport):
    """Доставляет запросы серверам стенда по имени хоста. Остановленный контейнер «не слушает порт»."""

    def __init__(self, stand: Stand, extra: dict[str, Any] | None = None) -> None:
        self._stand = stand
        apps = stand_apps(stand) | (extra or {})
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if not self._stand.is_listening(host):
            raise httpx.ConnectError("All connection attempts failed", request=request)
        return await self._transports[host].handle_async_request(request)


def engines_client(stand: Stand, extra: dict[str, Any] | None = None) -> httpx.AsyncClient:
    """HTTP-клиент, запросы которого попадают в серверы стенда (extra — дополнительные хосты)."""
    return httpx.AsyncClient(transport=StandTransport(stand, extra))


def docker_client(stand: Stand) -> httpx.AsyncClient:
    """HTTP-клиент «сокета Docker» стенда."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=docker_app(stand)), base_url="http://docker")
