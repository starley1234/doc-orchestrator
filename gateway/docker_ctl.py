"""Запуск и остановка контейнеров движков через Docker Engine API (unix-сокет).

У штатных серверов MinerU и PaddleOCR-VL нет команды «выгрузить модель из VRAM»:
видеопамять освобождает только завершение процесса, то есть остановка контейнера.
"""

from typing import Any

import httpx

from gateway.errors import GatewayError


class ContainerControl:
    def __init__(self, docker: httpx.AsyncClient, stop_sec: int) -> None:
        self._docker = docker  # HTTP-клиент, подключённый к сокету Docker
        self._stop_sec = stop_sec

    async def start(self, name: str) -> None:
        """Запускает контейнер. Уже запущенный (HTTP 304) — не ошибка."""
        response = await self._call("POST", f"/containers/{name}/start")
        if response.status_code == 404:
            raise GatewayError(
                502, f"Контейнер {name} не создан: выполните на сервере deploy/install.sh stack"
            )
        self._check(response, 204, 304)

    async def stop(self, name: str) -> None:
        """Останавливает контейнер и дожидается завершения процесса: после ответа VRAM свободна.

        Уже остановленный (304) и не созданный (404) контейнер видеопамять не занимает — это не ошибка.
        """
        response = await self._call("POST", f"/containers/{name}/stop", params={"t": self._stop_sec})
        self._check(response, 204, 304, 404)

    async def status(self, name: str) -> str:
        """Состояние контейнера по версии Docker (running, exited, created...) или missing, если его нет."""
        response = await self._call("GET", f"/containers/{name}/json")
        if response.status_code == 404:
            return "missing"
        self._check(response, 200)
        try:
            return response.json()["State"]["Status"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GatewayError(502, f"Docker вернул неожиданное описание контейнера {name}") from exc

    async def _call(self, method: str, path: str, **request: Any) -> httpx.Response:
        try:
            return await self._docker.request(method, path, **request)
        except httpx.HTTPError as exc:
            raise GatewayError(
                502,
                f"Docker недоступен ({type(exc).__name__}): в контейнер шлюза должен быть проброшен "
                "/var/run/docker.sock",
            ) from exc

    @staticmethod
    def _check(response: httpx.Response, *expected: int) -> None:
        if response.status_code not in expected:
            call = f"{response.request.method} {response.request.url.path}"
            raise GatewayError(502, f"Docker: {call} -> HTTP {response.status_code}: {response.text[:300]}")
