"""Классификатор входного файла: выбирает маршрут обработки, то есть нужный движок."""

import codecs
from enum import StrEnum
from pathlib import PurePath

from app.core.errors import UnsupportedFileError
from app.core.image_ops import open_pdf

OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx", ".html", ".md", ".txt"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

MIN_TEXT_CHARS_PER_PAGE = 100  # меньше непробельных символов в среднем на страницу — это скан

# Управляющие байты, которых не бывает в текстовых файлах (табуляция, переводы строк и ESC — бывают)
_BINARY_BYTES = bytes(set(range(32)) - {9, 10, 12, 13, 27})


class RouteType(StrEnum):
    """Маршрут обработки файла."""

    DIGITAL_OFFICE = "DIGITAL_OFFICE"  # офисный файл -> Docling (CPU)
    DIGITAL_PDF = "DIGITAL_PDF"  # PDF с текстовым слоем -> Docling (CPU)
    SCAN_PDF = "SCAN_PDF"  # PDF-скан -> MinerU или PaddleOCR-VL (GPU 0)
    SCAN_IMAGE = "SCAN_IMAGE"  # картинка -> MinerU или PaddleOCR-VL (GPU 0)
    DRAWING = "DRAWING"  # чертёж -> кропы зон + Qwen3-VL (GPU 1)
    TEXT = "TEXT"  # прочий текстовый файл (CSV, JSON, XML, код...) -> отдаётся как есть, без движков


def classify_file(file_bytes: bytes, filename: str, doc_type_hint: str | None = None) -> RouteType:
    """Определяет маршрут по подсказке о типе документа, расширению и содержимому файла."""
    if doc_type_hint == "drawing":
        return RouteType.DRAWING
    extension = PurePath(filename.lower()).suffix
    if extension in OFFICE_EXTENSIONS:
        return RouteType.DIGITAL_OFFICE
    if extension in IMAGE_EXTENSIONS:
        return RouteType.SCAN_IMAGE
    if extension == ".pdf":
        return _classify_pdf(file_bytes)
    if decode_text(file_bytes) is not None:
        return RouteType.TEXT
    raise UnsupportedFileError(f"Формат файла «{filename}» не поддерживается")


def decode_text(file_bytes: bytes) -> str | None:
    """Текст файла или None, если файл двоичный. Кодировки: UTF-8, UTF-16 (с BOM), Windows-1251."""
    if file_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encodings = ["utf-16"]
    elif file_bytes.translate(None, _BINARY_BYTES) != file_bytes:  # есть управляющие байты
        return None
    else:
        encodings = ["utf-8-sig", "cp1251"]  # cp1251 — частая кодировка выгрузок из 1С
    for encoding in encodings:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _classify_pdf(file_bytes: bytes) -> RouteType:
    """Цифровой PDF или скан: считаем непробельные символы текстового слоя."""
    with open_pdf(file_bytes) as doc:
        text_chars = sum(len("".join(page.get_text().split())) for page in doc.pages())
        has_text_layer = text_chars / doc.page_count >= MIN_TEXT_CHARS_PER_PAGE
    return RouteType.DIGITAL_PDF if has_text_layer else RouteType.SCAN_PDF
