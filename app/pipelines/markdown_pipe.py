"""Пайплайн А: документ -> Markdown для Open WebUI."""

import re

from app.core.aio import run_cpu
from app.core.classifier import RouteType, classify_file, is_table_document
from app.core.errors import EngineError, InvalidFileError
from app.pipelines.base import Engines
from app.schemas.common import ProcessMetadata, ProcessResponse

DESCRIBE_PROMPT = "Опиши схему текстом или переведи блок-схему в синтаксис Mermaid"


class MarkdownPipeline:
    def __init__(self, engines: Engines) -> None:
        self._engines = engines

    async def run(self, file_bytes: bytes, filename: str) -> ProcessResponse:
        """Классифицирует файл, распознаёт его подходящим движком и проверяет результат в DLP."""
        engines = self._engines
        route = await run_cpu(classify_file, file_bytes, filename, None)
        if route in (RouteType.DIGITAL_OFFICE, RouteType.DIGITAL_PDF):
            markdown, engine_used = (
                await engines.docling.convert(file_bytes, filename),
                "docling",
            )  # CPU, без GPU
        elif is_table_document(filename):
            markdown, engine_used = await engines.paddle_ocr(file_bytes, filename), "paddle"
        else:
            markdown, engine_used = await self._mineru_with_figures(file_bytes, filename)

        checked = await engines.dlp.inspect(markdown)  # post-check: секреты и ПДн маскируются
        metadata = ProcessMetadata(filename=filename, engine_used=engine_used, dlp_masked=checked.masked)
        return ProcessResponse(page_content=checked.text, metadata=metadata)

    async def _mineru_with_figures(self, file_bytes: bytes, filename: str) -> tuple[str, str]:
        """MinerU распознаёт скан, затем Qwen3-VL описывает вырезанные из него картинки."""
        engines = self._engines
        async with engines.gpu.acquire_gpu0("mineru"):
            reply = await engines.mineru.extract(file_bytes, filename)

        markdown = reply.markdown
        for figure in reply.images:  # GPU 0 уже свободен: картинками занимается GPU 1
            try:
                description = await engines.ask_vlm(DESCRIBE_PROMPT, figure.data)
            except InvalidFileError as exc:
                raise EngineError(f"MinerU вернул повреждённую картинку {figure.name}") from exc
            markdown = _insert_description(markdown, figure.name, description)
        return markdown, "mineru+qwen3vl" if reply.images else "mineru"


def _insert_description(markdown: str, image_name: str, description: str) -> str:
    """Заменяет плейсхолдер картинки ![...](image_name) её описанием."""
    placeholder = re.compile(rf"!\[[^\]]*\]\({re.escape(image_name)}\)")
    markdown, replaced = placeholder.subn(lambda _: description, markdown)
    return markdown if replaced else f"{markdown}\n\n{description}"  # плейсхолдера нет — дописываем в конец
