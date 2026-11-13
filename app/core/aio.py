"""Помощники asyncio: тяжёлые вычисления вне event loop и остановка фоновых задач."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

# PyMuPDF не потокобезопасен, поэтому вся тяжёлая работа с документами идёт через один поток
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="doc-cpu")


async def run_cpu(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Выполняет блокирующую функцию в выделенном потоке, не останавливая event loop."""
    return await asyncio.get_running_loop().run_in_executor(_executor, partial(func, *args, **kwargs))


async def stop_task(task: asyncio.Task[Any] | None) -> None:
    """Отменяет фоновую задачу и дожидается её завершения.

    Отмену повторяем, пока задача жива: одного cancel() мало, если CancelledError «проглотит»
    asyncio.wait_for внутри библиотеки (известная гонка в Python 3.11).
    """
    while task and not task.done():
        task.cancel()
        await asyncio.wait([task], timeout=0.1)
