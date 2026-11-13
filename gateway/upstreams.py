"""Клиенты штатных серверов движков: docling-serve, MinerU 4 и paddleocr-pdf-api.

Каждый переводит вызов «файл -> Markdown» в родной API своего сервера. API сверены с исходниками
docling-serve v1.38.2, MinerU 4.2.1 и paddleocr-pdf-api v0.5.0 — их и ставит install.sh.
"""

import asyncio
import codecs
import io
import mimetypes
import re
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import PurePath
from typing import Any, ClassVar
from urllib.parse import quote, urlsplit

import httpx
from PIL import Image

from gateway.config import GatewaySettings
from gateway.errors import GatewayError

IMAGE_REF = re.compile(r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)")  # картинка в Markdown: ![подпись](путь)


class EngineApi:
    """Общая часть клиентов: запрос к серверу движка, единая обработка сбоев, предел времени."""

    title: ClassVar[str] = "Движок"
    suffixes: ClassVar[frozenset[str]] = frozenset()  # расширения файлов, которые принимает сервер
    renamed: ClassVar[dict[str, str]] = {}  # расширения, которые сервер знает под другим именем
    ready_path: ClassVar[str] = "/health"  # сервер готов к работе, когда GET сюда отвечает 200

    def __init__(self, http: httpx.AsyncClient, base_url: str, settings: GatewaySettings) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._settings = settings

    async def probe(self) -> httpx.Response | None:
        """Ответ сервера на проверку готовности. None — сервер не отвечает: остановлен или запускается."""
        try:
            return await self._http.get(self._base_url + self.ready_path, timeout=10)
        except httpx.HTTPError:
            return None

    async def is_ready(self) -> bool:
        response = await self.probe()
        return response is not None and response.status_code == 200

    async def _request(self, method: str, url: str, **request: Any) -> httpx.Response:
        """Запрос к серверу движка. url — путь либо полный адрес, который сервер сам вернул в ответе."""
        if "://" in url:
            # Такие ссылки сервер строит по заголовку Host: путь берём из ответа, а хост — свой
            base, link = urlsplit(self._base_url), urlsplit(url)
            url = f"{base.scheme}://{base.netloc}{link.path}" + (f"?{link.query}" if link.query else "")
        else:
            url = self._base_url + url
        try:
            response = await self._http.request(method, url, **request)
        except httpx.HTTPError as exc:
            raise GatewayError(502, f"{self.title} недоступен: {type(exc).__name__} {exc}".rstrip()) from exc
        if not response.is_success:
            path = urlsplit(url).path
            raise GatewayError(
                502, f"{self.title}: {method} {path} -> HTTP {response.status_code}: {response.text[:300]}"
            )
        return response

    def _upload_name(self, filename: str) -> str:
        """Имя, под которым файл уходит на сервер: нейтральная основа и расширение.

        По расширению сервер узнаёт формат. Исходное имя не передаётся: к пробелам, кириллице
        и двоеточиям в именах серверы относятся по-разному.
        """
        suffix = PurePath(filename).suffix.lower()
        suffix = self.renamed.get(suffix, suffix)
        if suffix not in self.suffixes:
            raise GatewayError(422, f"{self.title} не принимает файлы «{suffix or filename}»")
        return "document" + suffix

    @asynccontextmanager
    async def _within_deadline(self, cancel: Callable[[], Awaitable[Any]]) -> AsyncIterator[None]:
        """Ограничивает время распознавания. Не уложились — отменяем задачу на сервере, отвечаем 504."""
        deadline = self._settings.deadline_sec
        try:
            async with asyncio.timeout(deadline):
                yield
        except TimeoutError:
            with suppress(GatewayError):  # отмена — из вежливости: ответ уже определён
                await cancel()
            raise GatewayError(504, f"{self.title} не уложился в {deadline:g} с") from None

    @contextmanager
    def _reading_reply(self) -> Iterator[None]:
        """Разбор ответа сервера: нет ожидаемых полей или это не JSON — ошибка шлюза, а не падение."""
        try:
            yield
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            detail = f"{self.title} вернул неожиданный ответ: {type(exc).__name__} {exc}"
            raise GatewayError(502, detail) from exc


class DoclingServe(EngineApi):
    """docling-serve: POST /v1/convert/file — multipart с полем files и параметрами конвертации."""

    title = "Docling"
    suffixes = frozenset({".pdf", ".docx", ".xlsx", ".pptx", ".html", ".md", ".txt"})

    async def convert(self, file_bytes: bytes, filename: str) -> str:
        name = self._upload_name(filename)
        if name.endswith((".md", ".txt")):
            file_bytes = _as_utf8(file_bytes)
        options = {
            "to_formats": "md",
            "image_export_mode": "placeholder",  # картинки в Markdown не встраиваются
            "include_images": "false",
            "do_ocr": str(self._settings.docling_do_ocr).lower(),
        }
        response = await self._request(
            "POST",
            "/v1/convert/file",
            files={"files": (name, file_bytes, _mime(name))},
            data=options,
            timeout=self._settings.deadline_sec,  # ответ приходит, когда документ готов
        )
        with self._reading_reply():
            reply = response.json()
            status, markdown = reply["status"], reply["document"]["md_content"]
        if status not in ("success", "partial_success") or markdown is None:
            errors = str(reply.get("errors") or "причина не указана")[:300]
            raise GatewayError(502, f"Docling не смог преобразовать документ (status={status}): {errors}")
        return markdown


class MineruApi(EngineApi):
    """MinerU 4 (mineru-kit api-server), Tasks API: создать задачу, загрузить файл, дождаться, скачать ZIP."""

    title = "MinerU"
    suffixes = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".jp2"})
    renamed: ClassVar[dict[str, str]] = {".tif": ".tiff"}
    ready_path = "/v1/health"

    async def extract(self, file_bytes: bytes, filename: str) -> tuple[str, dict[str, bytes]]:
        """Скан -> Markdown и картинки, на которые он ссылается (ключ — путь, как в Markdown)."""
        # Языка распознавания в API нет, уровень (tier) сервер выбирает сам: старший из разрешённых
        # при запуске. Лишние поля в запросе он отклоняет с HTTP 400.
        manifest = {"files": [{"file_name": self._upload_name(filename), "bytes": len(file_bytes)}]}
        response = await self._request("POST", "/v1/tasks", json=manifest)
        with self._reading_reply():
            task = response.json()
            task_url = "/v1/tasks/" + quote(task["task_id"], safe="")
            upload = task["files"][0].get("upload")

        async with self._within_deadline(lambda: self._request("POST", task_url + "/cancel")):
            if upload:
                with self._reading_reply():
                    upload_url, headers = upload["url"], dict(upload.get("headers") or {})
                await self._request("PUT", upload_url, content=file_bytes, headers=headers)
            while True:
                response = await self._request("GET", task_url)
                with self._reading_reply():
                    file = response.json()["files"][0]
                    status = file["status"]
                    result_url = file["result_url"] if status == "completed" else ""
                if status == "completed":
                    break
                if status in ("failed", "canceled"):
                    error = file.get("error") or {}
                    reason = error.get("message") or error.get("code") or status
                    raise GatewayError(502, f"MinerU не распознал документ: {reason}")
                await asyncio.sleep(self._settings.poll_interval_sec)
            archive = (await self._request("GET", result_url)).content

        markdown, images = _read_mineru_archive(archive)
        return _pick_figures(markdown, images, self._settings)


class PaddleApi(EngineApi):
    """paddleocr-pdf-api (PaddleOCR-VL): POST /ocr создаёт задачу, GET /ocr/{id} и /ocr/{id}/result — итог."""

    title = "PaddleOCR-VL"
    suffixes = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
    ready_path = "/jobs"  # отдельной проверки живости у сервера нет

    async def extract(self, file_bytes: bytes, filename: str) -> str:
        """Скан -> Markdown: страницы склеиваются по порядку."""
        name = self._upload_name(filename)
        response = await self._request("POST", "/ocr", files={"file": (name, file_bytes, _mime(name))})
        with self._reading_reply():
            job_url = "/ocr/" + quote(response.json()["job_id"], safe="")

        async with self._within_deadline(lambda: self._request("POST", job_url + "/cancel")):
            while True:
                response = await self._request("GET", job_url)
                with self._reading_reply():
                    job = response.json()
                    status = job["status"]
                if status in ("completed", "failed", "cancelled"):
                    break
                await asyncio.sleep(self._settings.poll_interval_sec)
            try:
                if status != "completed":
                    raise GatewayError(
                        502, f"PaddleOCR-VL не распознал документ: {job.get('error') or status}"
                    )
                response = await self._request("GET", job_url + "/result")
            finally:
                # Сервер хранит задачи и загруженные файлы бессрочно: завершённую удаляем сами
                with suppress(GatewayError):
                    await self._request("DELETE", job_url)
        with self._reading_reply():
            pages = sorted(response.json()["pages"], key=lambda page: page["page_num"])
            return "\n\n".join(page["markdown"] for page in pages)


def _mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _as_utf8(data: bytes) -> bytes:
    """Текстовый файл в UTF-8: Docling читает только её, а выгрузки из 1С приходят в Windows-1251."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace").encode()
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace").encode()
    return data


def _read_mineru_archive(archive: bytes) -> tuple[str, dict[str, bytes]]:
    """Результат MinerU — ZIP: markdown.md, служебные JSON и каталог images/ с вырезанными картинками.

    Возвращает Markdown и картинки, на которые он ссылается: ключ — путь в том виде, как в Markdown.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as files:
            markdown = files.read("markdown.md").decode("utf-8", errors="replace")
            # В архиве картинки лежат в images/, а в Markdown путь бывает и без каталога: ищем по имени файла
            by_basename = {PurePath(name).name: name for name in files.namelist()}
            images = {}
            for path in dict.fromkeys(IMAGE_REF.findall(markdown)):  # по порядку появления, без повторов
                if name := by_basename.get(PurePath(path).name):
                    images[path] = files.read(name)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise GatewayError(502, f"MinerU вернул результат без markdown.md: {exc}") from exc
    return markdown, images


def _pick_figures(
    markdown: str, images: dict[str, bytes], settings: GatewaySettings
) -> tuple[str, dict[str, bytes]]:
    """Отбирает картинки для Qwen3-VL; ссылки на остальные убирает из Markdown.

    Оркестратор описывает моделью каждую картинку, а это секунды на штуку. Поэтому мелкие
    (логотипы, значки) не отдаются вовсе, а при избытке остаются самые крупные.
    """
    areas: dict[str, int] = {}  # путь -> площадь картинки
    for path, data in images.items():
        try:
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
        except (OSError, ValueError):  # не картинка или файл повреждён
            continue
        if min(width, height) >= settings.mineru_min_figure_side:
            areas[path] = width * height
    kept = list(areas)
    if settings.mineru_max_figures and len(kept) > settings.mineru_max_figures:
        largest = sorted(kept, key=lambda path: areas[path], reverse=True)[: settings.mineru_max_figures]
        kept = [path for path in kept if path in largest]  # порядок — как в документе
    markdown = IMAGE_REF.sub(lambda ref: ref.group(0) if ref.group(1) in kept else "", markdown)
    return markdown, {path: images[path] for path in kept}
