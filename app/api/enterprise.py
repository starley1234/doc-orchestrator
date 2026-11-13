"""Вход для 1С:Документооборот и Teamcenter: извлечение строгого JSON из документа."""

import asyncio
import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Form, HTTPException, Request, Response, UploadFile, status
from redis.exceptions import RedisError
from starlette.datastructures import State

from app.api.deps import check_upload, clean_filename
from app.core.errors import OrchestratorError
from app.schemas.common import DocType, StructuredResult, TaskStatus

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["1С:ДО и Teamcenter"])


# response_model_exclude_none: незаполненные поля ответа (task_id, error и т. п.) в JSON не выводятся
@router.post("/extract-structured", response_model_exclude_none=True)
async def extract_structured(
    request: Request,
    response: Response,
    file: UploadFile,
    doc_type: Annotated[DocType, Form()],
    has_seals: Annotated[bool, Form()] = False,
    async_mode: Annotated[bool, Form()] = False,
) -> StructuredResult:
    """Извлекает из документа JSON по схеме doc_type и проверяет его валидаторами.

    has_seals=true — на скане есть печати или сложные таблицы: его прочитает PaddleOCR-VL.
    async_mode=false — ответ с результатом; ошибки валидации приходят в validation.warnings с кодом 200.
    async_mode=true — обработка идёт в фоне (код 202), результат отдаёт GET /api/tasks/{task_id}.
    """
    state = request.app.state
    file_bytes = check_upload(await file.read(), state.settings)
    filename = clean_filename(file.filename)
    if not async_mode:
        return await state.structured_pipeline.run(file_bytes, filename, doc_type, has_seals)

    task = StructuredResult(task_id=str(uuid4()), status=TaskStatus.QUEUED, doc_type=doc_type)
    await _save(state, task)
    background = asyncio.create_task(_run_task(state, task, file_bytes, filename, has_seals))
    state.background_tasks.add(background)  # ссылка нужна, чтобы задачу не удалил сборщик мусора
    background.add_done_callback(state.background_tasks.discard)
    response.status_code = status.HTTP_202_ACCEPTED
    return StructuredResult(task_id=task.task_id, status=TaskStatus.QUEUED)


@router.get("/tasks/{task_id}", response_model_exclude_none=True)
async def get_task(request: Request, task_id: str) -> StructuredResult:
    """Состояние фоновой задачи, а после завершения — и её результат."""
    saved = await request.app.state.redis.get(f"task:{task_id}")
    if saved is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Задача не найдена или срок её хранения истёк")
    return StructuredResult.model_validate_json(saved)


async def _save(state: State, task: StructuredResult) -> None:
    """Записывает состояние задачи в Redis; срок хранения отсчитывается заново."""
    await state.redis.set(f"task:{task.task_id}", task.model_dump_json(), ex=state.settings.task_ttl_sec)


async def _run_task(
    state: State, task: StructuredResult, file_bytes: bytes, filename: str, has_seals: bool
) -> None:
    """Фоновая обработка документа: итог (результат или причина сбоя) сохраняется в Redis."""

    def failed(reason: str) -> StructuredResult:
        return task.model_copy(update={"status": TaskStatus.FAILED, "error": reason})

    try:
        async with state.task_slots:  # лишние задачи ждут здесь, а не в очереди к GPU с её таймаутом
            await _save(state, task.model_copy(update={"status": TaskStatus.PROCESSING}))
            result = await state.structured_pipeline.run(file_bytes, filename, task.doc_type, has_seals)
        result = result.model_copy(update={"task_id": task.task_id})
    except asyncio.CancelledError:  # сервис останавливается
        await _save(state, failed("Сервис был остановлен: отправьте документ повторно"))
        raise
    except OrchestratorError as exc:
        result = failed(str(exc))
    except Exception:
        log.exception("Задача %s: непредвиденная ошибка", task.task_id)
        result = failed("Внутренняя ошибка сервиса")
    try:
        await _save(state, result)
    except RedisError:
        log.exception("Задача %s: не удалось сохранить результат", task.task_id)
