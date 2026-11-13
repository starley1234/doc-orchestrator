"""Вход для 1С:Документооборот и Teamcenter: извлечение строгого JSON из документа."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, Request, Response, UploadFile, status

from app.api.deps import check_upload, clean_filename
from app.schemas.common import DocType, StructuredResult, TaskStatus

router = APIRouter(prefix="/api", tags=["1С и Teamcenter"])


# response_model_exclude_none: незаполненные поля ответа (task_id, error и т. п.) в JSON не выводятся
@router.post("/extract-structured", response_model_exclude_none=True)
async def extract_structured(
    request: Request,
    response: Response,
    file: UploadFile,
    doc_type: Annotated[DocType, Form()],
    async_mode: Annotated[bool, Form()] = False,
) -> StructuredResult:
    """Извлекает из документа JSON по схеме doc_type и проверяет его валидаторами.

    async_mode=false — ответ с результатом; ошибки валидации приходят в validation.warnings с кодом 200.
    async_mode=true — задача уходит в очередь (код 202), результат отдаёт GET /api/tasks/{task_id}.
    """
    state = request.app.state
    file_bytes = check_upload(await file.read(), state.settings)
    filename = clean_filename(file.filename)
    if async_mode:
        task_id = await state.tasks.enqueue(file_bytes, filename, doc_type)
        response.status_code = status.HTTP_202_ACCEPTED
        return StructuredResult(task_id=task_id, status=TaskStatus.QUEUED)
    return await state.structured_pipeline.run(file_bytes, filename, doc_type)


@router.get("/tasks/{task_id}", response_model_exclude_none=True)
async def get_task(request: Request, task_id: UUID) -> StructuredResult:
    """Состояние асинхронной задачи, а после завершения — и её результат."""
    task = await request.app.state.tasks.get(str(task_id))
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Задача не найдена или срок её хранения истёк")
    return task
