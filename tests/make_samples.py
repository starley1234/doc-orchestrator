"""Генератор тестовых файлов. Запуск: python tests/make_samples.py — файлы появятся в tests/samples/."""

import io
from pathlib import Path

import pymupdf
from PIL import Image

SAMPLES_DIR = Path(__file__).parent / "samples"

# Текст цифрового документа: заметно больше 100 печатных символов на страницу
TEXT = (
    "Договор поставки № ДП-17 от 10.01.2025. ООО «Ромашка» (Поставщик) обязуется передать, "
    "а АО «Вектор» (Покупатель) — принять и оплатить подшипники и комплектующие по спецификации. "
    "Общая сумма договора составляет 34 923,00 руб., включая НДС 20 %. Срок действия — до 31.12.2025."
)
REQUIREMENTS = (
    "1. 42...48 HRC.\n"
    "2. Неуказанные предельные отклонения размеров: H14, h14, ±IT14/2.\n"
    "3. Острые кромки притупить."
)
MM = 3  # масштаб чертежа: пунктов на миллиметр


def _write(
    page: pymupdf.Page, rect: tuple[float, float, float, float], text: str, size: float = 11, **kw
) -> None:
    """Пишет текст в прямоугольник шрифтом с кириллицей."""
    writer = pymupdf.TextWriter(page.rect)
    writer.fill_textbox(pymupdf.Rect(rect), text, font=pymupdf.Font("helv"), fontsize=size)
    writer.write_text(page, **kw)


def _save(doc: pymupdf.Document) -> bytes:
    data = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return data


def make_pdf(pages: list[str], scan: bool = False) -> bytes:
    """PDF формата A4, pages — текст каждой страницы.

    scan=True: страница — картинка во весь лист, а текст невидим (как OCR-слой у скана).
    """
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if scan:
            blank = io.BytesIO()
            Image.new("L", (1240, 1754), 245).save(blank, "PNG")  # пустой лист A4, 150 dpi
            page.insert_image(page.rect, stream=blank.getvalue())
        if text:
            _write(page, (60, 60, 535, 780), text, render_mode=3 if scan else 0)  # 3 — невидимый текст
    return _save(doc)


def make_drawing_png(width: int = 3000, height: int = 2000) -> bytes:
    """Чертёж ЕСКД: рамка, эскиз вала, техтребования и основная надпись в правом нижнем углу."""
    doc = pymupdf.open()
    page = doc.new_page(width=width / 2, height=height / 2)  # рисуем в пунктах, рендерим вдвое крупнее
    right, bottom = page.rect.width - 5 * MM, page.rect.height - 5 * MM
    page.draw_rect((20 * MM, 5 * MM, right, bottom), width=2)  # рамка листа

    # Эскиз: ступенчатый вал с осевой линией
    for x, length, radius in ((200, 150, 40), (350, 300, 70), (650, 120, 50)):
        page.draw_rect((x, 400 - radius, x + length, 400 + radius), width=1.5)
    page.draw_line((170, 400), (800, 400), width=0.5, dashes="[12 4 2 4] 0")

    # Технические требования — над основной надписью
    _write(page, (right - 185 * MM, 420, right - 10, 700), REQUIREMENTS, size=16)

    # Основная надпись 185x55 мм. Колонки: подписи | наименование и материал | масса и масштаб
    left, top = right - 185 * MM, bottom - 55 * MM
    col1, col2 = left + 65 * MM, left + 135 * MM
    row1, row2 = top + 15 * MM, top + 40 * MM  # строки: обозначение | наименование | материал
    page.draw_rect((left, top, right, bottom), width=2)
    for start, end in (((col1, top), (col1, bottom)), ((col2, row1), (col2, bottom))):
        page.draw_line(start, end, width=1.5)
    for y in (row1, row2):
        page.draw_line((col1, y), (right, y), width=1.5)
    fields = (
        ((left + 10, top + 12, col1, bottom), "Разраб.  Иванов\nПров.  Петров\nН. контр.  Сидоров", 12),
        ((col1 + 15, top + 8, right, row1), "АБВГ.123456.001", 24),
        ((col1 + 15, row1 + 20, col2, row2), "Вал приводной", 22),
        ((col1 + 15, row2 + 12, col2, bottom), "Сталь 45 ГОСТ 1050-2013", 13),
        ((col2 + 12, row1 + 20, right, row2), "Масса 12.5\nМасштаб 1:2", 13),
    )
    for rect, text, size in fields:
        _write(page, rect, text, size=size)

    png = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).tobytes("png")
    doc.close()
    return png


SAMPLES = {
    "digital.pdf": lambda: make_pdf([TEXT]),  # PDF с текстовым слоем
    "scan.pdf": lambda: make_pdf([""], scan=True),  # PDF-скан: пустая картинка без текста
    "drawing.png": make_drawing_png,  # большой чертёж 3000x2000
}

if __name__ == "__main__":
    SAMPLES_DIR.mkdir(exist_ok=True)
    for name, make in SAMPLES.items():
        (SAMPLES_DIR / name).write_bytes(make())
        print(f"{name}: {(SAMPLES_DIR / name).stat().st_size} байт")
