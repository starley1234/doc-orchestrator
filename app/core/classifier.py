"""Классификатор входного файла: выбирает маршрут обработки, то есть нужный движок."""

import re
from enum import StrEnum
from pathlib import PurePath

import pymupdf  # PyMuPDF; прежнее имя модуля — fitz (устарело)

from app.core.errors import UnsupportedFileError
from app.core.image_ops import open_pdf

OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx", ".html"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff"}

MIN_TEXT_CHARS_PER_PAGE = 100  # меньше символов текста в среднем на страницу — это скан
SCAN_IMAGE_COVERAGE = 0.5  # картинка на половину страницы и больше — страница отсканирована

# Начала слов в имени файла, по которым скан считается сложной таблицей или формой
TABLE_KEYWORDS = (
    "упд",
    "upd",
    "счет",
    "счёт",
    "invoice",
    "накладн",
    "табл",
    "table",
    "бланк",
    "анкет",
    "form",
)


class RouteType(StrEnum):
    """Маршрут обработки файла."""

    DIGITAL_OFFICE = "digital_office"  # офисный формат -> Docling (CPU)
    DIGITAL_PDF = "digital_pdf"  # PDF с текстовым слоем -> Docling (CPU)
    SCAN_PDF = "scan_pdf"  # PDF-скан -> MinerU или PaddleOCR-VL (GPU 0)
    SCAN_IMAGE = "scan_image"  # картинка -> MinerU или PaddleOCR-VL (GPU 0)
    DRAWING = "drawing"  # чертёж -> кропы зон + Qwen3-VL (GPU 1)


def classify_file(file_bytes: bytes, filename: str, doc_type_hint: str | None = None) -> RouteType:
    """Определяет маршрут по расширению, подсказке о типе документа и содержимому PDF."""
    extension = PurePath(filename).suffix.lower()
    if extension in OFFICE_EXTENSIONS:
        return RouteType.DIGITAL_OFFICE
    if doc_type_hint == "drawing":
        return RouteType.DRAWING
    if extension in IMAGE_EXTENSIONS:
        return RouteType.SCAN_IMAGE
    if extension == ".pdf":
        return _classify_pdf(file_bytes, doc_type_hint)
    raise UnsupportedFileError(f"Формат файла «{filename}» не поддерживается")


def is_table_document(filename: str) -> bool:
    """Помечен ли документ как сложная таблица или форма: такие сканы распознаёт PaddleOCR-VL."""
    words = re.split(r"[\W_]+", PurePath(filename).stem.lower())
    return any(word.startswith(TABLE_KEYWORDS) for word in words)


def _classify_pdf(file_bytes: bytes, doc_type_hint: str | None) -> RouteType:
    """Цифровой PDF или скан: смотрим на текстовый слой и на страницы-картинки."""
    with open_pdf(file_bytes) as doc:
        text_chars = sum(len(page.get_text().strip()) for page in doc)
        has_text_layer = text_chars / doc.page_count >= MIN_TEXT_CHARS_PER_PAGE
        # В УПД и счетах со сканами текстовому слою не доверяем (это OCR-подложка): распознаём заново
        distrust_text = doc_type_hint in ("upd", "invoice") and _has_scanned_pages(doc)
        return RouteType.DIGITAL_PDF if has_text_layer and not distrust_text else RouteType.SCAN_PDF


def _has_scanned_pages(doc: pymupdf.Document) -> bool:
    """Есть ли в PDF отсканированные страницы: картинка занимает большую часть листа."""
    for page in doc:
        min_area = SCAN_IMAGE_COVERAGE * page.rect.get_area()
        for image in page.get_image_info():
            if (pymupdf.Rect(image["bbox"]) & page.rect).get_area() >= min_area:
                return True
    return False
