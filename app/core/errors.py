"""Ошибки сервиса. Каждая знает свой HTTP-статус — его отдаёт общий обработчик в app/main.py."""


class OrchestratorError(Exception):
    """Базовая ошибка doc-orchestrator."""

    status_code = 500


class UnsupportedFileError(OrchestratorError):
    """Формат файла не поддерживается."""

    status_code = 415


class FileTooLargeError(OrchestratorError):
    """Файл больше допустимого размера."""

    status_code = 413


class InvalidFileError(OrchestratorError):
    """Файл повреждён, защищён паролем или не читается."""

    status_code = 422


class EngineError(OrchestratorError):
    """Внешний движок (OCR, VLM, DLP) недоступен или вернул ошибку."""

    status_code = 502


class GPUBusyError(OrchestratorError):
    """Задача не дождалась своей очереди к GPU."""

    status_code = 503
