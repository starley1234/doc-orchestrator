"""Вход для Open WebUI (External Document Loader): PUT /process."""

import mimetypes
import secrets
from pathlib import PurePath
from typing import Annotated
from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.api.deps import check_upload, clean_filename
from app.pipelines.markdown_pipe import OcrEngine
from app.schemas.common import ProcessResponse

router = APIRouter(tags=["Open WebUI"])


def require_api_key(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    """Проверяет заголовок Authorization: Bearer <ключ>, если в настройках задан API_KEY."""
    api_key = request.app.state.settings.api_key
    expected = f"Bearer {api_key}".encode()
    if api_key and not secrets.compare_digest((authorization or "").encode(), expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный API-ключ", {"WWW-Authenticate": "Bearer"})


def _filename(x_filename: str, content_type: str | None) -> str:
    """Имя файла из заголовков. Open WebUI передаёт его в percent-кодировке."""
    name = clean_filename(unquote(x_filename))
    if not PurePath(name).suffix and content_type:  # расширения нет — определяем его по Content-Type
        name += mimetypes.guess_extension(content_type.partition(";")[0].strip()) or ""
    return name


@router.put("/process", dependencies=[Depends(require_api_key)])
async def process(
    request: Request,
    x_filename: Annotated[str, Header()],
    content_type: Annotated[str | None, Header()] = None,
    x_ocr_engine: Annotated[OcrEngine, Header()] = "auto",
) -> ProcessResponse:
    """Превращает документ в Markdown. Тело запроса — сырые байты файла.

    Заголовок X-OCR-Engine: paddle отправляет скан в PaddleOCR-VL (печати, сложные таблицы);
    по умолчанию (auto) сканы распознаёт MinerU. Запрос ждёт результата: очередь к GPU — внутри.
    """
    state = request.app.state
    file_bytes = check_upload(await request.body(), state.settings)
    return await state.markdown_pipeline.run(file_bytes, _filename(x_filename, content_type), x_ocr_engine)
