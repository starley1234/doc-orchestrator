"""Конфигурация сервиса: значения берутся из переменных окружения или файла .env."""

from functools import lru_cache

from pydantic import Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Все настройки doc-orchestrator. Переменная окружения = имя поля в верхнем регистре."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = "INFO"
    redis_url: str = "redis://redis:6379/0"

    # --- ML-движки ---
    # По умолчанию — заглушка mock-engines из docker-compose.yml.
    # На боевом сервере меняются только эти адреса.
    docling_url: str = "http://mock-engines:8001/docling"
    mineru_url: str = "http://mock-engines:8001/mineru"
    paddle_url: str = "http://mock-engines:8001/paddle"
    ollama_url: str = "http://mock-engines:8001"  # боевой Ollama: http://ollama:11434
    vlm_model: str = "qwen3-vl:4b-instruct-q8_0"  # строго instruct-вариант, не thinking
    vlm_num_ctx: PositiveInt = 8192  # окно контекста модели, токенов

    # --- Доступ ---
    api_key: str = ""  # если задан, PUT /process требует заголовок Authorization: Bearer <ключ>

    # --- Таймауты, секунды ---
    http_timeout_sec: PositiveFloat = 300  # один HTTP-запрос к движку (OCR большого скана идёт минуты)
    gpu_lock_timeout_sec: PositiveFloat = 300  # TTL лока GPU: страховка на случай падения процесса
    gpu_lock_wait_sec: PositiveFloat = 300  # сколько задача ждёт своей очереди к GPU
    paddle_idle_ttl_sec: PositiveFloat = 120  # простой paddle, после которого GPU 0 возвращается mineru
    gpu0_check_interval_sec: PositiveFloat = 15  # период фоновой проверки простоя

    # --- Файлы и изображения ---
    max_file_size_mb: PositiveInt = 50  # файлы крупнее отклоняются (HTTP 413)
    max_pixels: int = Field(1_280_000, ge=32 * 32)  # предел картинки для Qwen3-VL (8 ГБ VRAM)
    overview_dpi: PositiveInt = 150  # рендер первой страницы документа для Qwen3-VL
    drawing_dpi: PositiveInt = 300  # рендер чертежа: нужны мелкие надписи штампа
    vlm_max_text_chars: PositiveInt = 8000  # предел распознанного текста в промпте (чтобы влез в VLM_NUM_CTX)

    # --- Асинхронные задачи (async_mode=true) ---
    task_ttl_sec: PositiveInt = 3600  # сколько хранится состояние и результат задачи
    task_concurrency: PositiveInt = 2  # сколько задач обрабатывается одновременно, остальные ждут


@lru_cache
def get_settings() -> Settings:
    """Настройки читаются один раз за время жизни процесса."""
    return Settings()
