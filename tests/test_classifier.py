"""Классификатор: выбор маршрута по подсказке о типе документа, расширению и содержимому файла."""

import pymupdf
import pytest
from conftest import make_pdf

from app.core.classifier import RouteType, classify_file, decode_text
from app.core.errors import InvalidFileError, UnsupportedFileError

CHARS_100 = "x" * 50 + "\n" + "x" * 50  # ровно 100 непробельных символов
CHARS_99 = "x" * 50 + " \n " + "x" * 49  # пробелы и переводы строк не считаются

# PDF без единой страницы
EMPTY_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [] /Count 0 >>
endobj
trailer
<< /Root 1 0 R /Size 3 >>
%%EOF
"""


@pytest.mark.parametrize("filename", ["shaft.pdf", "shaft.png", "shaft.docx", "shaft.dwg", "shaft"])
def test_drawing_hint_wins_over_extension(filename: str):
    assert classify_file(b"", filename, "drawing") is RouteType.DRAWING


@pytest.mark.parametrize(
    "filename", ["report.docx", "PRICE.XLSX", "slides.pptx", "page.html", "README.md", "notes.txt"]
)
def test_office_and_text_formats_go_to_docling(filename: str):
    assert classify_file(b"", filename) is RouteType.DIGITAL_OFFICE


@pytest.mark.parametrize(
    "filename", ["photo.jpg", "photo.JPEG", "scan.png", "scan.tiff", "scan.tif", "scan.bmp", "scan.webp"]
)
def test_images_are_scans(filename: str):
    assert classify_file(b"", filename) is RouteType.SCAN_IMAGE


def test_pdf_with_text_layer_is_digital(digital_pdf: bytes):
    assert classify_file(digital_pdf, "contract.pdf") is RouteType.DIGITAL_PDF


def test_pdf_without_text_is_scan(scan_pdf: bytes):
    assert classify_file(scan_pdf, "scan.pdf") is RouteType.SCAN_PDF


def test_pdf_with_little_text_is_scan():
    assert classify_file(make_pdf(["Стр. 1 из 1"]), "page.pdf") is RouteType.SCAN_PDF


def test_text_threshold_is_100_non_space_chars_per_page():
    assert classify_file(make_pdf([CHARS_100]), "doc.pdf") is RouteType.DIGITAL_PDF
    assert classify_file(make_pdf([CHARS_99]), "doc.pdf") is RouteType.SCAN_PDF


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        ([CHARS_100 * 2, ""], RouteType.DIGITAL_PDF),  # (200 + 0) / 2 = 100 символов на страницу
        ([CHARS_100, CHARS_99], RouteType.SCAN_PDF),  # 99.5
        ([CHARS_100, "", ""], RouteType.SCAN_PDF),  # 33.3
    ],
)
def test_text_threshold_is_average_over_pages(pages: list[str], expected: RouteType):
    assert classify_file(make_pdf(pages), "doc.pdf") is expected


@pytest.mark.parametrize("hint", ["upd", "invoice"])
def test_document_type_does_not_change_pdf_route(digital_pdf: bytes, scan_pdf: bytes, hint: str):
    assert classify_file(digital_pdf, "doc.pdf", hint) is RouteType.DIGITAL_PDF
    assert classify_file(scan_pdf, "doc.pdf", hint) is RouteType.SCAN_PDF


def test_route_values_match_names():
    # Значение маршрута попадает в ответ PUT /process (metadata.route)
    assert [route.value for route in RouteType] == [route.name for route in RouteType]
    assert f"{RouteType.SCAN_PDF}" == "SCAN_PDF"


# --- Ошибки ---------------------------------------------------------------------------------


def test_empty_pdf_is_value_error():
    with pytest.raises(ValueError, match=r"^Empty PDF$"):
        classify_file(EMPTY_PDF, "empty.pdf")


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.7 truncated"])
def test_broken_pdf_is_rejected(data: bytes):
    with pytest.raises(InvalidFileError, match="Не удалось прочитать PDF"):
        classify_file(data, "broken.pdf")


def test_password_protected_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page()
    protected = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret")

    with pytest.raises(InvalidFileError, match=r"^PDF защищён паролем$"):
        classify_file(protected, "protected.pdf")


# --- Текстовые файлы: отдаются как есть -------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "data", "text"),
    [
        ("price.csv", "Товар;Цена\nВал;12500\n".encode(), "Товар;Цена\nВал;12500\n"),
        ("data.json", b'\xef\xbb\xbf{"a": 1}', '{"a": 1}'),  # UTF-8 с BOM
        ("upd.xml", "<Файл ИдФайл='1'/>".encode("cp1251"), "<Файл ИдФайл='1'/>"),  # выгрузка из 1С
        ("notepad.log", "Журнал".encode("utf-16"), "Журнал"),  # «Юникод» Блокнота Windows: UTF-16 с BOM
        ("Makefile", b"all:\n\techo ok\n", "all:\n\techo ok\n"),  # без расширения
        (
            "colors.log",
            b"\x1b[31mERROR\x1b[0m\n",
            "\x1b[31mERROR\x1b[0m\n",
        ),  # ESC-последовательности терминала
    ],
)
def test_other_text_files_are_passed_as_is(filename: str, data: bytes, text: str):
    assert classify_file(data, filename) is RouteType.TEXT
    assert decode_text(data) == text


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("archive.zip", b"PK\x03\x04\x14\x00\x00\x00\x08\x00"),
        ("legacy.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 16),
        ("model.dwg", b"AC1027\x00\x00\x00\x00\x00"),
        ("blob.bin", bytes(range(256))),
        ("ctrl.dat", b"abc\x01def"),  # управляющий символ
        ("broken.csv", b"abc\x98def"),  # не UTF-8, а в Windows-1251 байта 0x98 нет
        ("broken16.txt2", b"\xff\xfe\x00"),  # BOM UTF-16 и оборванный символ
    ],
)
def test_binary_files_are_rejected(filename: str, data: bytes):
    assert decode_text(data) is None
    with pytest.raises(UnsupportedFileError, match=filename):
        classify_file(data, filename)
