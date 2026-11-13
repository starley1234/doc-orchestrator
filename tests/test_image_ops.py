"""Препроцессинг изображений: рендер PDF, ограничение по пикселям, кропы зон чертежа."""

import io
import threading

import pytest
from make_samples import make_pdf
from PIL import ExifTags, Image

from app.core import image_ops
from app.core.aio import run_cpu
from app.core.errors import InvalidFileError
from app.core.image_ops import (
    crop_drawing_zones,
    first_page_image,
    is_pdf,
    render_pdf_to_images,
    resize_max_pixels,
)


def encode(image: Image.Image, fmt: str = "PNG", **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# --- Рендер PDF ---------------------------------------------------------------------------


def test_render_pdf_page_to_png(digital_pdf: bytes):
    (page,) = render_pdf_to_images(digital_pdf, dpi=72)

    assert decode(page).format == "PNG"
    assert decode(page).size == (595, 842)  # A4 в пунктах: при 72 dpi пункт равен пикселю


def test_render_scales_with_dpi(digital_pdf: bytes):
    (page,) = render_pdf_to_images(digital_pdf, dpi=200)
    assert decode(page).size == pytest.approx((595 * 200 / 72, 842 * 200 / 72), abs=1)


def test_render_respects_max_pages():
    pdf = make_pdf(["первая", "вторая", "третья"])

    assert len(render_pdf_to_images(pdf, dpi=36)) == 3
    assert len(render_pdf_to_images(pdf, dpi=36, max_pages=1)) == 1


def test_render_caps_raster_size(digital_pdf: bytes, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(image_ops, "MAX_RENDER_PIXELS", 100_000)

    width, height = decode(render_pdf_to_images(digital_pdf, dpi=300)[0]).size

    assert width * height <= 100_000 * 1.02  # небольшой запас на округление сторон
    assert width / height == pytest.approx(595 / 842, abs=0.01)


def test_first_page_image(digital_pdf: bytes, drawing_png: bytes):
    assert is_pdf(digital_pdf) and not is_pdf(drawing_png)
    assert decode(first_page_image(digital_pdf, dpi=72)).size == (595, 842)  # PDF рендерится
    assert first_page_image(drawing_png, dpi=72) is drawing_png  # картинка отдаётся как есть


# --- Ограничение по пикселям ---------------------------------------------------------------


def test_resize_shrinks_large_image_to_pixel_budget(drawing_png: bytes):
    resized = decode(resize_max_pixels(drawing_png))  # исходник 3000x2000 = 6 Мп

    assert resized.format == "PNG"
    assert 0.98 * 1_280_000 <= resized.width * resized.height <= 1_280_000  # вплотную к лимиту, но не больше
    assert resized.width / resized.height == pytest.approx(3000 / 2000, abs=0.01)  # пропорции сохранены


def test_resize_returns_small_png_as_is():
    small = encode(Image.new("RGB", (640, 480), "white"))
    assert resize_max_pixels(small) is small


def test_resize_honors_custom_limit():
    resized = decode(resize_max_pixels(encode(Image.new("RGB", (400, 200))), max_pixels=20_000))
    assert resized.size == (200, 100)


@pytest.mark.parametrize(("fmt", "mode"), [("JPEG", "RGB"), ("JPEG", "CMYK"), ("TIFF", "1"), ("GIF", "P")])
def test_resize_converts_any_format_to_png(fmt: str, mode: str):
    converted = decode(resize_max_pixels(encode(Image.new(mode, (64, 48)), fmt)))

    assert converted.format == "PNG"
    assert converted.size == (64, 48)
    assert converted.mode in ("RGB", "L")


def test_resize_applies_exif_rotation():
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6  # снимок повёрнут на 90°
    photo = encode(Image.new("RGB", (40, 20)), "JPEG", exif=exif)

    assert decode(resize_max_pixels(photo)).size == (20, 40)


@pytest.mark.parametrize("data", [b"", b"garbage", b"\x89PNG\r\n\x1a\n-truncated-"])
def test_broken_image_is_rejected(data: bytes):
    with pytest.raises(InvalidFileError):
        resize_max_pixels(data)


def test_broken_pdf_is_rejected():
    with pytest.raises(InvalidFileError):
        render_pdf_to_images(b"%PDF-1.7 garbage")


# --- Кропы чертежа -------------------------------------------------------------------------


def test_crop_zones_follow_gost_layout():
    # Картинка-«линейка»: красный канал хранит x, зелёный — y (в процентах от размера листа)
    width, height = 200, 100
    ruler = Image.new("RGB", (width, height))
    ruler.putdata([(x * 100 // width, y * 100 // height, 0) for y in range(height) for x in range(width)])

    zones = crop_drawing_zones(encode(ruler))
    stamp, tech_req = decode(zones["stamp"]), decode(zones["tech_req"])

    # Штамп: x от 55 % до 100 %, y от 75 % до 100 %
    assert stamp.size == (90, 25)
    assert stamp.getpixel((0, 0))[:2] == (55, 75)
    assert stamp.getpixel((89, 24))[:2] == (99, 99)
    # Техтребования: x от 55 % до 100 %, y от 35 % до 78 %
    assert tech_req.size == (90, 43)
    assert tech_req.getpixel((0, 0))[:2] == (55, 35)
    assert tech_req.getpixel((89, 42))[:2] == (99, 77)


def test_crop_zones_of_sample_drawing(drawing_png: bytes):
    zones = crop_drawing_zones(drawing_png)

    assert set(zones) == {"stamp", "tech_req"}
    assert decode(zones["stamp"]).size == (1350, 500)  # 45 % x 25 % от 3000x2000
    assert decode(zones["tech_req"]).size == (1350, 860)
    # В штампе есть линии и текст, то есть вырезан не пустой угол
    assert decode(zones["stamp"]).convert("L").getextrema()[0] < 50


def test_crops_are_resized_to_pixel_budget(drawing_png: bytes):
    for crop in crop_drawing_zones(drawing_png, max_pixels=100_000).values():
        image = decode(crop)
        assert image.format == "PNG"
        assert image.width * image.height <= 100_000


# --- Вынос тяжёлой работы из event loop ------------------------------------------------------


async def test_run_cpu_uses_dedicated_thread():
    thread_name = await run_cpu(lambda: threading.current_thread().name)
    assert thread_name.startswith("doc-cpu")


async def test_run_cpu_propagates_errors():
    with pytest.raises(InvalidFileError):
        await run_cpu(resize_max_pixels, b"garbage")
