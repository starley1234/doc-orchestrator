"""Пайплайн А: документ -> обогащённый Markdown для Open WebUI."""

import re
from typing import Literal

from app.core.aio import run_cpu
from app.core.errors import EngineError, InvalidFileError
from app.core.image_ops import smart_resize
from app.pipelines.base import Engines
from app.schemas.common import ProcessMetadata, ProcessResponse

OcrEngine = Literal["auto", "mineru", "paddle"]  # значения заголовка X-OCR-Engine

FIGURE_PROMPT = "Переведи схему на изображении в блок кода Mermaid или дай краткое техническое описание."


class MarkdownPipeline:
    def __init__(self, engines: Engines) -> None:
        self._engines = engines

    async def run(self, file_bytes: bytes, filename: str, ocr_engine: OcrEngine = "auto") -> ProcessResponse:
        """Распознаёт файл подходящим движком; схемы из скана Qwen3-VL превращает в Mermaid.

        ocr_engine="paddle" отправляет скан в PaddleOCR-VL (печати, сложные таблицы) вместо MinerU.
        """
        engines = self._engines
        ocr = await engines.ocr(file_bytes, filename, use_paddle=ocr_engine == "paddle")

        markdown, engine_used = ocr.markdown, ocr.engine
        for name, figure in ocr.figures.items():  # GPU 0 уже свободен: картинками занимается GPU 1
            try:
                image = await run_cpu(smart_resize, figure, engines.settings.max_pixels)
            except InvalidFileError as exc:
                raise EngineError(f"MinerU вернул повреждённую картинку {name}") from exc
            description = await engines.ask_vlm(FIGURE_PROMPT, image)
            markdown = _insert_description(markdown, name, description)
        if ocr.figures:
            engine_used += "+" + engines.settings.vlm_model.partition(":")[0]  # например, mineru+qwen3-vl

        metadata = ProcessMetadata(filename=filename, route=ocr.route, engine_used=engine_used)
        return ProcessResponse(page_content=markdown, metadata=metadata)


def _insert_description(markdown: str, image_name: str, description: str) -> str:
    """Заменяет плейсхолдер картинки ![...](image_name) её описанием."""
    placeholder = re.compile(rf"!\[[^\]]*\]\({re.escape(image_name)}\)")
    markdown, replaced = placeholder.subn(lambda _: description, markdown)
    return markdown if replaced else f"{markdown}\n\n{description}"  # плейсхолдера нет — дописываем в конец
