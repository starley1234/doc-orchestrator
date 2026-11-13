"""Общее для роутов: проверка загруженного файла."""

from pathlib import PurePath

from app.config import Settings
from app.core.errors import FileTooLargeError, InvalidFileError


def clean_filename(name: str | None) -> str:
    """Имя файла без пути: клиент мог прислать «C:\\docs\\упд.pdf» или «../упд.pdf»."""
    return PurePath((name or "").replace("\\", "/")).name


def check_upload(file_bytes: bytes, settings: Settings) -> bytes:
    """Отсекает пустые и слишком большие файлы до того, как они попадут в очередь к движкам."""
    if not file_bytes:
        raise InvalidFileError("Файл пуст")
    if len(file_bytes) > settings.max_file_size_mb * 1024 * 1024:
        raise FileTooLargeError(f"Файл больше {settings.max_file_size_mb} МБ")
    return file_bytes
