"""Классификатор: выбор маршрута по расширению, подсказке о типе документа и содержимому PDF."""

import pymupdf
import pytest
from make_samples import TEXT, make_pdf

from app.core.classifier import RouteType, classify_file, is_table_document
from app.core.errors import InvalidFileError, UnsupportedFileError

CHARS_100 = "x" * 50 + "\n" + "x" * 49  # страница ровно со 100 символами текста
CHARS_99 = "x" * 49 + "\n" + "x" * 49


@pytest.mark.parametrize("filename", ["report.docx", "PRICE.XLSX", "slides.pptx", "page.html"])
def test_office_formats_are_digital(filename: str):
    assert classify_file(b"", filename, None) is RouteType.DIGITAL_OFFICE


def test_office_format_wins_over_drawing_hint():
    assert classify_file(b"", "drawing.docx", "drawing") is RouteType.DIGITAL_OFFICE


@pytest.mark.parametrize("filename", ["shaft.pdf", "shaft.png", "shaft.tiff", "shaft.dwg"])
def test_drawing_hint_selects_drawing_route(filename: str):
    assert classify_file(b"", filename, "drawing") is RouteType.DRAWING


@pytest.mark.parametrize("filename", ["photo.jpg", "photo.JPEG", "scan.png", "scan.tiff"])
def test_images_are_scans(filename: str):
    assert classify_file(b"", filename, None) is RouteType.SCAN_IMAGE


def test_pdf_with_text_layer_is_digital(digital_pdf: bytes):
    assert classify_file(digital_pdf, "contract.pdf", None) is RouteType.DIGITAL_PDF


def test_pdf_without_text_is_scan(scan_pdf: bytes):
    assert classify_file(scan_pdf, "scan.pdf", None) is RouteType.SCAN_PDF


def test_pdf_with_little_text_is_scan():
    assert classify_file(make_pdf(["Стр. 1 из 1"]), "page.pdf", None) is RouteType.SCAN_PDF


def test_text_threshold_is_100_chars_per_page():
    # Две строки и перевод строки между ними: ровно 100 и 99 символов
    assert classify_file(make_pdf([CHARS_100]), "doc.pdf", None) is RouteType.DIGITAL_PDF
    assert classify_file(make_pdf([CHARS_99]), "doc.pdf", None) is RouteType.SCAN_PDF


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        ([CHARS_100 * 2, ""], RouteType.DIGITAL_PDF),  # (201 + 0) / 2 = 100.5 символа на страницу
        ([CHARS_100, CHARS_99], RouteType.SCAN_PDF),  # 99.5
        ([CHARS_100, "", ""], RouteType.SCAN_PDF),  # 33.3
    ],
)
def test_text_threshold_is_average_over_pages(pages: list[str], expected: RouteType):
    assert classify_file(make_pdf(pages), "doc.pdf", None) is expected


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        (None, RouteType.DIGITAL_PDF),
        ("contract", RouteType.DIGITAL_PDF),
        ("upd", RouteType.SCAN_PDF),  # бухгалтерский документ со сканами распознаём заново
        ("invoice", RouteType.SCAN_PDF),
    ],
)
def test_scan_with_ocr_text_layer(hint: str | None, expected: RouteType):
    sandwich = make_pdf([TEXT], scan=True)  # картинка во весь лист + невидимый текст
    assert classify_file(sandwich, "doc.pdf", hint) is expected


def test_digital_upd_without_scans_stays_digital(digital_pdf: bytes):
    assert classify_file(digital_pdf, "upd.pdf", "upd") is RouteType.DIGITAL_PDF


@pytest.mark.parametrize("filename", ["notes.txt", "archive.zip", "README"])
def test_unknown_format_is_rejected(filename: str):
    with pytest.raises(UnsupportedFileError):
        classify_file(b"data", filename, None)


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.7 truncated"])
def test_broken_pdf_is_rejected(data: bytes):
    with pytest.raises(InvalidFileError):
        classify_file(data, "broken.pdf", None)


def test_password_protected_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page()
    protected = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret")

    with pytest.raises(InvalidFileError, match="паролем"):
        classify_file(protected, "protected.pdf", None)


@pytest.mark.parametrize(
    "filename",
    [
        "УПД_128.pdf",
        "upd-2025.png",
        "Счет на оплату 57.pdf",
        "счёт-фактура.jpg",
        "invoice_57.pdf",
        "Товарная накладная.pdf",
        "scan_table.png",
        "Таблица цен.pdf",
        "анкета.png",
        "application_form.jpg",
        "3f2a9c_упд128.pdf",  # Open WebUI добавляет к имени файла идентификатор
    ],
)
def test_table_documents_are_detected_by_filename(filename: str):
    assert is_table_document(filename)


@pytest.mark.parametrize(
    "filename", ["Договор поставки.pdf", "scan.pdf", "report_2025.png", "platform.pdf", "information.jpg"]
)
def test_other_documents_are_not_tables(filename: str):
    assert not is_table_document(filename)
