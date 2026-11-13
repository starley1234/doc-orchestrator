"""Препроцессинг изображений: рендер PDF в PNG, smart-resize под Qwen3-VL, кропы зон чертежа ЕСКД."""

import io
import math
from collections.abc import Iterator
from contextlib import contextmanager

import pymupdf  # PyMuPDF; прежнее имя модуля — fitz (устарело)
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from app.core.errors import InvalidFileError

DEFAULT_MAX_PIXELS = 1_280_000  # больше — Qwen3-VL не поместится в 8 ГБ VRAM
SIDE_STEP = 32  # визуальный энкодер Qwen3-VL требует сторон, кратных 32 px
MAX_RENDER_PIXELS = 40_000_000  # потолок растра страницы: лист A0 на 300 dpi занял бы ~420 МБ памяти

# Зоны чертежа ЕСКД в долях листа: (x0, y0, x1, y1). Основная надпись 185×55 мм на книжном A4
# занимает почти всю ширину листа, на альбомных A3–A0 — правый нижний угол.
# Технические требования расположены над основной надписью.
PORTRAIT_ZONES = {"stamp": (0.0, 0.72, 1.0, 1.0), "tech_req": (0.0, 0.35, 1.0, 0.74)}
LANDSCAPE_ZONES = {"stamp": (0.5, 0.70, 1.0, 1.0), "tech_req": (0.5, 0.30, 1.0, 0.72)}


def is_pdf(data: bytes) -> bool:
    """PDF узнаём по сигнатуре: по стандарту она находится в первых 1024 байтах файла."""
    return b"%PDF-" in data[:1024]


@contextmanager
def open_pdf(pdf_bytes: bytes) -> Iterator[pymupdf.Document]:
    """Открывает PDF из байтов. Любая ошибка чтения превращается в InvalidFileError (это ValueError)."""
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                raise InvalidFileError("PDF защищён паролем")
            if doc.page_count == 0:
                raise InvalidFileError("Empty PDF")
            yield doc
    except InvalidFileError:
        raise
    except (RuntimeError, ValueError, pymupdf.mupdf.FzErrorBase) as exc:  # ошибки PyMuPDF и MuPDF
        raise InvalidFileError(f"Не удалось прочитать PDF: {exc}") from exc


def render_pdf_to_images(pdf_bytes: bytes, dpi: int = 200, max_pages: int | None = None) -> list[bytes]:
    """Рендерит страницы PDF в PNG: первые max_pages страниц или все."""
    with open_pdf(pdf_bytes) as doc:
        images = []
        for page in doc.pages(stop=max_pages):
            # Размеры страницы заданы в пунктах (72 на дюйм). Огромный лист рендерим с меньшим DPI.
            max_dpi = 72 * math.sqrt(MAX_RENDER_PIXELS / page.rect.get_area())
            images.append(page.get_pixmap(dpi=min(dpi, int(max_dpi))).tobytes("png"))
        return images


def first_page_image(file_bytes: bytes, dpi: int) -> bytes:
    """Первая страница документа как картинка: PDF рендерится, изображение отдаётся как есть."""
    if is_pdf(file_bytes):
        return render_pdf_to_images(file_bytes, dpi, max_pages=1)[0]
    return file_bytes


def smart_resize(img_bytes: bytes, max_pixels: int = DEFAULT_MAX_PIXELS) -> bytes:
    """Готовит картинку для Qwen3-VL: RGB PNG не больше max_pixels со сторонами, кратными 32 px."""
    return _to_png(_fit(_open_image(img_bytes), max_pixels))


def crop_drawing_zones(img_bytes: bytes, max_pixels: int = DEFAULT_MAX_PIXELS) -> dict[str, bytes]:
    """Вырезает из чертежа ЕСКД основную надпись (stamp) и зону техтребований (tech_req).

    Оба кропа проходят smart-resize, то есть готовы к отправке в Qwen3-VL.
    """
    image = _open_image(img_bytes)
    zones = PORTRAIT_ZONES if image.width < image.height else LANDSCAPE_ZONES
    return {name: _to_png(_fit(_crop(image, zone), max_pixels)) for name, zone in zones.items()}


def _open_image(img_bytes: bytes) -> Image.Image:
    """Открывает изображение и приводит его к RGB."""
    try:
        image: Image.Image = Image.open(io.BytesIO(img_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise InvalidFileError("Файл не является ни PDF, ни изображением") from exc
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidFileError(f"Не удалось прочитать изображение: {exc}") from exc
    if image.getexif().get(ExifTags.Base.Orientation, 1) != 1:
        image = ImageOps.exif_transpose(image)  # фото со смартфона: применяем поворот из EXIF
    return image.convert("RGB")


def _crop(image: Image.Image, zone: tuple[float, float, float, float]) -> Image.Image:
    """Вырезает зону, заданную в долях ширины и высоты."""
    x0, y0, x1, y1 = zone
    width, height = image.size
    return image.crop((round(width * x0), round(height * y0), round(width * x1), round(height * y1)))


def _fit(image: Image.Image, max_pixels: int) -> Image.Image:
    """Уменьшает картинку до размеров, допустимых для Qwen3-VL."""
    size = _target_size(image.width, image.height, max_pixels)
    return image if size == image.size else image.resize(size, Image.Resampling.LANCZOS)


def _target_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Размеры картинки для Qwen3-VL: площадь не больше max_pixels, стороны кратны 32 px."""
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    width, height = _snap(width * scale), _snap(height * scale)
    # Сторона не бывает короче 32 px, поэтому очень вытянутая картинка может выйти за лимит:
    # тогда поджимаем длинную сторону (для остальных картинок эти две строки ничего не меняют).
    width = min(width, _snap(max_pixels / height))
    height = min(height, _snap(max_pixels / width))
    return width, height


def _snap(side: float) -> int:
    """Округляет сторону вниз до кратной 32 px (но не меньше 32 px)."""
    return max(SIDE_STEP, int(side // SIDE_STEP) * SIDE_STEP)


def _to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
