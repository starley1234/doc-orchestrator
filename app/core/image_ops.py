"""Препроцессинг изображений: рендер PDF в PNG, ограничение по числу пикселей, кропы зон чертежа."""

import io
import math
from collections.abc import Iterator
from contextlib import contextmanager

import pymupdf  # PyMuPDF; прежнее имя модуля — fitz (устарело)
from PIL import ExifTags, Image, ImageOps

from app.core.errors import InvalidFileError

DEFAULT_MAX_PIXELS = 1_280_000  # больше — Qwen3-VL не поместится в 8 ГБ VRAM
MAX_RENDER_PIXELS = 40_000_000  # потолок растра страницы: лист A0 на 300 dpi занял бы ~420 МБ памяти

# Зоны чертежа ЕСКД в долях листа: (x0, y0, x1, y1)
STAMP_ZONE = (0.55, 0.75, 1.00, 1.00)  # основная надпись — правый нижний угол
TECH_REQ_ZONE = (0.55, 0.35, 1.00, 0.78)  # технические требования — над штампом


def is_pdf(data: bytes) -> bool:
    """PDF узнаём по сигнатуре: по стандарту она находится в первых 1024 байтах файла."""
    return b"%PDF-" in data[:1024]


@contextmanager
def open_pdf(pdf_bytes: bytes) -> Iterator[pymupdf.Document]:
    """Открывает PDF из байтов. Любая ошибка чтения превращается в InvalidFileError."""
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                raise InvalidFileError("PDF защищён паролем")
            if doc.page_count == 0:
                raise InvalidFileError("В PDF нет страниц")
            yield doc
    except (RuntimeError, ValueError, pymupdf.mupdf.FzErrorBase) as exc:  # ошибки PyMuPDF и MuPDF
        raise InvalidFileError(f"Не удалось прочитать PDF: {exc}") from exc


def render_pdf_to_images(pdf_bytes: bytes, dpi: int = 200, max_pages: int | None = None) -> list[bytes]:
    """Рендерит страницы PDF в PNG: первые max_pages страниц или все."""
    with open_pdf(pdf_bytes) as doc:
        images = []
        for page in doc.pages(stop=max_pages):
            zoom = dpi / 72  # размеры PDF заданы в пунктах, 72 пункта на дюйм
            zoom = min(zoom, math.sqrt(MAX_RENDER_PIXELS / page.rect.get_area()))
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            images.append(pixmap.tobytes("png"))
        return images


def first_page_image(file_bytes: bytes, dpi: int) -> bytes:
    """Первая страница документа как картинка: PDF рендерится, изображение отдаётся как есть."""
    if is_pdf(file_bytes):
        return render_pdf_to_images(file_bytes, dpi, max_pages=1)[0]
    return file_bytes


def resize_max_pixels(img_bytes: bytes, max_pixels: int = DEFAULT_MAX_PIXELS) -> bytes:
    """Возвращает PNG не больше max_pixels: крупное изображение пропорционально уменьшается."""
    image = _open_image(img_bytes)
    if image.format == "PNG" and image.width * image.height <= max_pixels:
        return img_bytes  # готовый PNG подходящего размера отдаём без перекодирования
    return _to_png(_shrink(image, max_pixels))


def crop_drawing_zones(img_bytes: bytes, max_pixels: int = DEFAULT_MAX_PIXELS) -> dict[str, bytes]:
    """Вырезает из чертежа ЕСКД штамп и зону техтребований. Оба кропа — PNG не больше max_pixels."""
    image = _open_image(img_bytes)
    zones = {"stamp": STAMP_ZONE, "tech_req": TECH_REQ_ZONE}
    return {name: _to_png(_shrink(_crop(image, zone), max_pixels)) for name, zone in zones.items()}


def _open_image(img_bytes: bytes) -> Image.Image:
    """Открывает изображение. Атрибут .format сохраняется, только если пиксели не менялись."""
    try:
        image = Image.open(io.BytesIO(img_bytes))
        image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidFileError(f"Не удалось прочитать изображение: {exc}") from exc
    if image.getexif().get(ExifTags.Base.Orientation, 1) != 1:
        image = ImageOps.exif_transpose(image)  # фото со смартфона: применяем поворот из EXIF
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")  # палитра, CMYK, 1 бит и т. п. — приводим к обычному RGB
    return image


def _crop(image: Image.Image, zone: tuple[float, float, float, float]) -> Image.Image:
    """Вырезает зону, заданную в долях ширины и высоты."""
    x0, y0, x1, y1 = zone
    width, height = image.size
    return image.crop((round(width * x0), round(height * y0), round(width * x1), round(height * y1)))


def _shrink(image: Image.Image, max_pixels: int) -> Image.Image:
    """Пропорционально уменьшает изображение, если пикселей больше max_pixels."""
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
