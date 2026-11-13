"""Препроцессинг изображений: рендер PDF, smart-resize под Qwen3-VL, кропы зон чертежа."""

import threading

import pytest
from conftest import A1_MM, A4_MM, decode, encode, make_drawing, make_pdf, make_png, stamp_box
from PIL import ExifTags, Image

from app.core import image_ops
from app.core.aio import run_cpu
from app.core.errors import InvalidFileError
from app.core.image_ops import (
    crop_drawing_zones,
    first_page_image,
    is_pdf,
    render_pdf_to_images,
    smart_resize,
)

MAX_PIXELS = 1_280_000


def dark_share(image: Image.Image) -> float:
    """Доля чёрных пикселей: в тестовом чертеже чёрный только прямоугольник основной надписи."""
    histogram = image.convert("L").histogram()
    return sum(histogram[:64]) / (image.width * image.height)


# --- Рендер PDF ---------------------------------------------------------------------------


def test_render_pdf_page_to_png(digital_pdf: bytes):
    (page,) = render_pdf_to_images(digital_pdf, dpi=72)

    assert decode(page).format == "PNG"
    assert decode(page).size == (595, 842)  # A4 в пунктах: при 72 dpi пункт равен пикселю


def test_render_uses_200_dpi_by_default(digital_pdf: bytes):
    (page,) = render_pdf_to_images(digital_pdf)
    assert decode(page).size == pytest.approx((595 * 200 / 72, 842 * 200 / 72), abs=1)


def test_render_scan_page_keeps_the_picture(scan_pdf: bytes):
    (page,) = render_pdf_to_images(scan_pdf, dpi=72)
    assert decode(page).convert("L").getpixel((300, 400)) == 245  # цвет «бумаги» скана


def test_render_respects_max_pages():
    pdf = make_pdf(["первая", "вторая", "третья"])

    assert len(render_pdf_to_images(pdf, dpi=36)) == 3
    assert len(render_pdf_to_images(pdf, dpi=36, max_pages=1)) == 1


def test_render_caps_raster_size(digital_pdf: bytes, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(image_ops, "MAX_RENDER_PIXELS", 100_000)

    width, height = decode(render_pdf_to_images(digital_pdf, dpi=300)[0]).size

    assert 90_000 <= width * height <= 100_000  # вплотную к лимиту, но не больше
    assert width / height == pytest.approx(595 / 842, abs=0.01)


def test_first_page_image(digital_pdf: bytes, drawing_a4: bytes):
    assert is_pdf(digital_pdf) and not is_pdf(drawing_a4)
    assert decode(first_page_image(digital_pdf, dpi=72)).size == (595, 842)  # PDF рендерится
    assert first_page_image(drawing_a4, dpi=72) is drawing_a4  # картинка отдаётся как есть


def test_broken_pdf_is_rejected():
    with pytest.raises(InvalidFileError, match="Не удалось прочитать PDF"):
        render_pdf_to_images(b"%PDF-1.7 garbage")


# --- smart-resize -------------------------------------------------------------------------


@pytest.mark.parametrize(("fixture", "size"), [("drawing_a4", (2480, 3508)), ("drawing_a1", (4000, 2800))])
def test_smart_resize_fits_pixel_budget_with_sides_multiple_of_32(fixture: str, size, request):
    drawing = request.getfixturevalue(fixture)
    assert decode(drawing).size == size

    resized = decode(smart_resize(drawing))

    assert (resized.format, resized.mode) == ("PNG", "RGB")
    assert resized.width % 32 == 0 and resized.height % 32 == 0
    assert 0.93 * MAX_PIXELS <= resized.width * resized.height <= MAX_PIXELS  # близко к лимиту, но не больше
    assert resized.width / resized.height == pytest.approx(size[0] / size[1], rel=0.03)  # пропорции сохранены


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((64, 64), (64, 64)),  # уже подходит
        ((1280, 992), (1280, 992)),  # ровно на границе лимита
        ((100, 50), (96, 32)),  # стороны округляются вниз до кратных 32
        ((1281, 1000), (1280, 992)),
        ((10, 10), (32, 32)),  # но не меньше 32 px
        ((2000, 1000), (1600, 800)),  # 2 Мп -> 1,28 Мп
        ((60000, 16), (40000, 32)),  # очень вытянутая: из-за минимума 32 px поджимается длинная сторона
        ((16, 60000), (32, 40000)),
    ],
)
def test_smart_resize_sizes(size: tuple[int, int], expected: tuple[int, int]):
    assert decode(smart_resize(make_png(*size))).size == expected


def test_smart_resize_honors_custom_limit():
    resized = decode(smart_resize(make_png(400, 200), max_pixels=20_000))

    assert resized.size == (192, 96)
    assert resized.width * resized.height <= 20_000


def test_smart_resize_keeps_the_picture():
    # Левая половина чёрная, правая белая — после уменьшения картина та же
    image = Image.new("L", (3000, 2000), 255)
    image.paste(0, (0, 0, 1500, 2000))

    resized = decode(smart_resize(encode(image)))

    assert dark_share(resized) == pytest.approx(0.5, abs=0.01)
    assert resized.getpixel((10, 10)) == (0, 0, 0)
    assert resized.getpixel((resized.width - 10, 10)) == (255, 255, 255)


def test_smart_resize_uses_smooth_resampling():
    # Чёрно-белые полоски шириной 1 px при уменьшении вдвое сливаются в серый (LANCZOS усредняет соседей)
    stripes = Image.new("L", (256, 256))
    stripes.putdata([255 * (x % 2) for _ in range(256) for x in range(256)])

    resized = decode(smart_resize(encode(stripes), max_pixels=128 * 128))

    assert resized.size == (128, 128)
    assert 100 < resized.getpixel((64, 64))[0] < 155


@pytest.mark.parametrize(
    ("fmt", "mode"),
    [("JPEG", "RGB"), ("JPEG", "CMYK"), ("TIFF", "1"), ("GIF", "P"), ("PNG", "RGBA"), ("BMP", "L")],
)
def test_smart_resize_converts_any_format_to_rgb_png(fmt: str, mode: str):
    converted = decode(smart_resize(encode(Image.new(mode, (64, 96)), fmt)))

    assert (converted.format, converted.mode, converted.size) == ("PNG", "RGB", (64, 96))


def test_smart_resize_applies_exif_rotation():
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6  # снимок повёрнут на 90°
    photo = encode(Image.new("RGB", (128, 64)), "JPEG", exif=exif)

    assert decode(smart_resize(photo)).size == (64, 128)


@pytest.mark.parametrize("data", [b"", b"garbage", b"%PDF-1.7"])
def test_non_image_is_rejected(data: bytes):
    with pytest.raises(InvalidFileError, match="не является ни PDF, ни изображением"):
        smart_resize(data)


def test_truncated_image_is_rejected(drawing_a4: bytes):
    with pytest.raises(InvalidFileError, match="Не удалось прочитать изображение"):
        smart_resize(drawing_a4[: len(drawing_a4) // 2])


# --- Кропы чертежа -------------------------------------------------------------------------

# Зоны из ТЗ в пикселях листа: (x0, y0, x1, y1)
A4_BOXES = {"stamp": (0, 2526, 2480, 3508), "tech_req": (0, 1228, 2480, 2596)}  # y: 72–100 % и 35–74 %
A1_BOXES = {
    "stamp": (2000, 1960, 4000, 2800),
    "tech_req": (2000, 840, 4000, 2016),
}  # x от 50 %, y: 70–100 % и 30–72 %


@pytest.mark.parametrize(
    ("fixture", "boxes"), [("drawing_a4", A4_BOXES), ("drawing_a1", A1_BOXES)], ids=["portrait", "landscape"]
)
def test_crop_zones_follow_sheet_orientation(fixture: str, boxes: dict, request):
    drawing = request.getfixturevalue(fixture)

    zones = crop_drawing_zones(drawing)

    assert set(zones) == {"stamp", "tech_req"}
    for name, box in boxes.items():
        expected = smart_resize(encode(decode(drawing).crop(box)))
        assert decode(zones[name]).size == decode(expected).size
        assert decode(zones[name]).tobytes() == decode(expected).tobytes(), name


@pytest.mark.parametrize(
    ("size", "sheet_mm", "zone"),
    [
        ((2480, 3508), A4_MM, (0, 2526, 2480, 3508)),  # книжный A4: штамп почти во всю ширину листа
        ((4000, 2800), A1_MM, (2000, 1960, 4000, 2800)),  # альбомный A1: штамп в правом нижнем углу
        ((3508, 2480), (420, 297), (1754, 1736, 3508, 2480)),  # альбомный A3
    ],
    ids=["A4", "A1", "A3"],
)
def test_stamp_of_eskd_sheet_is_inside_stamp_crop(size, sheet_mm, zone):
    drawing = make_drawing(*size, sheet_mm)
    left, top, right, bottom = stamp_box(*size, sheet_mm)
    stamp_area = (right - left) * (bottom - top)
    zone_area = (zone[2] - zone[0]) * (zone[3] - zone[1])

    zones = crop_drawing_zones(drawing)

    # Основная надпись попала в кроп целиком: доля чёрного равна доле штампа в зоне
    assert dark_share(decode(zones["stamp"])) == pytest.approx(stamp_area / zone_area, abs=0.01)
    assert dark_share(decode(zones["stamp"])) > 0.1


def test_square_sheet_is_treated_as_landscape():
    zones = crop_drawing_zones(make_png(1000, 1000))
    assert decode(zones["stamp"]).size == (480, 288)  # 50 % x 30 % листа, стороны кратны 32


@pytest.mark.parametrize("fixture", ["drawing_a4", "drawing_a1"])
def test_crops_are_ready_for_vlm(fixture: str, request):
    for crop in crop_drawing_zones(request.getfixturevalue(fixture)).values():
        image = decode(crop)
        assert (image.format, image.mode) == ("PNG", "RGB")
        assert image.width % 32 == 0 and image.height % 32 == 0
        assert image.width * image.height <= MAX_PIXELS


def test_crops_honor_custom_pixel_limit(drawing_a1: bytes):
    for crop in crop_drawing_zones(drawing_a1, max_pixels=100_000).values():
        assert decode(crop).width * decode(crop).height <= 100_000


# --- Вынос тяжёлой работы из event loop ------------------------------------------------------


async def test_run_cpu_uses_dedicated_thread():
    thread_name = await run_cpu(lambda: threading.current_thread().name)
    assert thread_name.startswith("doc-cpu")


async def test_run_cpu_propagates_errors():
    with pytest.raises(InvalidFileError):
        await run_cpu(smart_resize, b"garbage")
