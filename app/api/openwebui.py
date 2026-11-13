"""Вход для Open WebUI (External Document Loader): PUT /process."""

import mimetypes
import secrets
from pathlib import PurePath
from typing import Annotated
from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.api.deps import check_upload, clean_filename
from app.schemas.common import ProcessResponse

router = APIRouter(tags=["Open WebUI"])


def require_token(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    """Проверяет Bearer-токен, если он задан в настройках (OPENWEBUI_API_TOKEN)."""
    token = request.app.state.settings.openwebui_api_token
    expected = f"Bearer {token}".encode()
    if token and not secrets.compare_digest((authorization or "").encode(), expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный токен", {"WWW-Authenticate": "Bearer"})


def _filename(x_filename: str, content_type: str | None) -> str:
    """Имя файла из заголовков. Open WebUI передаёт его в percent-кодировке."""
    name = clean_filename(unquote(x_filename))
    if not PurePath(name).suffix and content_type:  # расширения нет — определяем его по Content-Type
        name += mimetypes.guess_extension(content_type.partition(";")[0].strip()) or ""
    return name


@router.put("/process", dependencies=[Depends(require_token)])
async def process(
    request: Request,
    x_filename: Annotated[str, Header()],
    content_type: Annotated[str | None, Header()] = None,
) -> ProcessResponse:
    """Превращает документ в Markdown. Тело запроса — сырые байты файла.

    Запрос ждёт результата: внутри пайплайна задача сама встаёт в очередь к GPU.
    """
    state = request.app.state
    file_bytes = check_upload(await request.body(), state.settings)
    return await state.markdown_pipeline.run(file_bytes, _filename(x_filename, content_type))
