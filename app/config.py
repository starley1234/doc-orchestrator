"""Конфигурация сервиса: значения берутся из переменных окружения или файла .env."""

from functools import lru_cache

from pydantic import PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Все настройки doc-orchestrator. Переменная окружения = имя поля в верхнем регистре."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = "INFO"
    redis_url: str = "redis://localhost:6379/0"

    # --- Внешние движки ---
    # По умолчанию — локальный mock-engines. В проде меняются только эти URL.
    docling_url: str = "http://localhost:8001/docling"
    mineru_url: str = "http://localhost:8001/mineru"
    paddle_url: str = "http://localhost:8001/paddle"
    vlm_gateway_url: str = "http://localhost:8001/gateway"
    dlp_url: str = "http://localhost:8001/dlp"
    vlm_model: str = "qwen3-vl"

    # --- Доступ ---
    # Если задан, PUT /process требует заголовок Authorization: Bearer <токен>
    openwebui_api_token: str | None = None

    # --- Таймауты, секунды ---
    http_timeout_s: PositiveFloat = 120  # один HTTP-запрос к движку
    gpu_lock_timeout_s: PositiveFloat = 300  # TTL лока GPU: страховка на случай падения процесса
    gpu_wait_timeout_s: PositiveFloat = 600  # сколько задача ждёт своей очереди к GPU
    paddle_idle_ttl_s: PositiveFloat = 120  # простой paddle, после которого GPU 0 возвращается mineru
    idle_check_interval_s: PositiveFloat = 30  # период фоновой проверки простоя

    # --- Файлы, изображения и текст для Qwen3-VL ---
    max_file_size_mb: PositiveInt = 50  # файлы крупнее отклоняются (HTTP 413)
    max_pixels: PositiveInt = 1_280_000  # больше пикселей — модель не поместится в 8 ГБ VRAM
    render_dpi: PositiveInt = 200  # рендер страниц PDF
    drawing_dpi: PositiveInt = 300  # рендер чертежей (нужны мелкие надписи штампа)
    vlm_max_text_chars: PositiveInt = 20_000  # предел распознанного текста в промпте

    # --- Асинхронные задачи (async_mode=true) ---
    task_workers: PositiveInt = 2  # по числу GPU: одна задача на OCR, другая на VLM
    task_ttl_s: PositiveInt = 86_400  # сколько хранится результат задачи
    task_poll_interval_s: PositiveFloat = 0.5  # как часто свободный воркер проверяет очередь


@lru_cache
def get_settings() -> Settings:
    """Настройки читаются один раз за время жизни процесса."""
    return Settings()
