"""Создаёт образцы для проверки установленных движков (deploy/smoke_test.py).

    pip install -r requirements-dev.txt && python deploy/samples/make_samples.py

ru-text.pdf — цифровой документ (текстовый слой) для Docling;
ru-scan.png — тот же лист картинкой, без текстового слоя: скан для MinerU, PaddleOCR-VL и Qwen3-VL.
Готовые файлы лежат в репозитории: на сервере ничего генерировать не нужно.
"""

from pathlib import Path

import pymupdf

LINES = [
    (16, "ООО «Ромашка»"),
    (13, "Счёт на оплату № 128 от 14 марта 2025 г."),
    (11, ""),
    (11, "Поставщик: ООО «Ромашка», ИНН 7707083893, КПП 770701001"),
    (11, "Покупатель: АО «Вектор», ИНН 5024156782, КПП 502401001"),
    (11, ""),
    (11, "1. Подшипник 6205-2RS — 10 шт. по 250,00 руб. — 2 500,00 руб."),
    (11, "2. Смазка Литол-24, 0,8 кг — 5 шт. по 320,50 руб. — 1 602,50 руб."),
    (11, ""),
    (12, "Итого к оплате: 4 102,50 руб., в том числе НДС 20 %."),
    (11, ""),
    (11, "Директор ______________ Иванов И. И."),
]
PAGE = pymupdf.Rect(0, 0, 595, 420)  # A5, альбомный лист, в пунктах
SCAN_DPI = 150


def main() -> None:
    folder = Path(__file__).parent
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    writer = pymupdf.TextWriter(page.rect)  # встроенные Base-14 шрифты кириллицу не пишут
    font = pymupdf.Font("helv")
    y = 60.0
    for size, text in LINES:
        if text:
            writer.append((50, y), text, font=font, fontsize=size)
        y += size * 2.1
    writer.write_text(page)
    page.draw_line((50, 108), (545, 108), width=0.8)

    (folder / "ru-text.pdf").write_bytes(doc.tobytes(garbage=3, deflate=True))
    pixmap = page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
    (folder / "ru-scan.png").write_bytes(pixmap.tobytes("png"))
    doc.close()


if __name__ == "__main__":
    main()
