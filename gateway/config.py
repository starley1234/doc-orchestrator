"""Настройки шлюза движков: значения берутся из переменных окружения."""

from pydantic import NonNegativeInt, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """Переменная окружения = имя поля в верхнем регистре.

    Значения по умолчанию — имена контейнеров и порты из «ИНСТРУКЦИЯ_ПО_УСТАНОВКЕ_OCR.md»:
    на сервере, развёрнутом по инструкции, настраивать ничего не нужно.
    """

    model_config = SettingsConfigDict(extra="ignore")

    log_level: str = "INFO"

    # --- Штатные серверы движков ---
    docling_serve_url: str = "http://docling-serve:5001"
    mineru_api_url: str = "http://mineru-api:8000"
    paddle_api_url: str = "http://paddle-gpu0:8000"

    # --- Контейнеры движков GPU 0: load запускает контейнер, unload останавливает ---
    mineru_container: str = "mineru-api"
    paddle_container: str = "paddle-gpu0"
    docker_socket: str = "/var/run/docker.sock"
    container_stop_sec: PositiveInt = 30  # сколько Docker ждёт завершения движка, прежде чем убить процесс

    # --- Время ---
    # Столько оркестратор ждёт ответа движка (у него та же переменная HTTP_TIMEOUT_SEC).
    # Шлюз обязан ответить раньше — см. deadline_sec.
    http_timeout_sec: PositiveFloat = 300
    poll_interval_sec: PositiveFloat = 1  # период опроса задачи движка и готовности контейнера

    # --- Docling ---
    # OCR картинок внутри цифровых документов выключен: в образе docling-serve нет моделей
    # для кириллицы, а сканы оркестратор и так отправляет в MinerU или PaddleOCR-VL.
    docling_do_ocr: bool = False

    # --- Картинки из сканов MinerU: каждую оркестратор отдаёт Qwen3-VL (несколько секунд на штуку) ---
    mineru_max_figures: NonNegativeInt = 12  # больше — остаются только самые крупные; 0 — без ограничения
    mineru_min_figure_side: PositiveInt = 100  # картинки мельче (логотипы, значки) не отдаются, пикселей

    @property
    def deadline_sec(self) -> float:
        """Предел одной операции шлюза: с запасом меньше таймаута оркестратора."""
        return self.http_timeout_sec * 0.9
