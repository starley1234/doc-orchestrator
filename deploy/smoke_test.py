#!/usr/bin/env python3
"""Проверка установленных движков делом: отправляет образец с русским текстом и читает ответ.

Нужен только Python 3 из системы — без pip и сторонних пакетов. Примеры (на самом сервере):

    python3 smoke_test.py --docling http://localhost:5001 --mineru http://localhost:8002 \\
                          --ollama http://localhost:11434
    python3 smoke_test.py --paddle http://localhost:8003        # пока запущен paddle-gpu0
    python3 smoke_test.py --orchestrator http://localhost:8000 --api-key КЛЮЧ --with-paddle

Проверяется не только «сервер отвечает», но и результат: в распознанном тексте должны
найтись слова образца. Так ловится, например, MinerU на уровне basic: он работает,
но кириллицу не читает. Код возврата — число непройденных проверок.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote, urlsplit

SAMPLES = Path(__file__).resolve().parent / "samples"
KEYWORDS = ["подшипник", "поставщик", "покупатель", "ромашка", "оплат"]  # слова образца (ё заменена на е)
ENOUGH_KEYWORDS = 3
POLL_SEC = 2.0


class CheckFailed(Exception):
    """Проверка не пройдена: текст исключения — объяснение для администратора."""


# --- HTTP на стандартной библиотеке ---------------------------------------------------------


def http(method, url, body=None, headers=None, timeout=60.0):
    """Запрос: возвращает (код ответа, тело). Нет соединения — CheckFailed."""
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:  # 4xx и 5xx — тоже ответ: причину объяснит тело
        return error.code, error.read()
    except (urllib.error.URLError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise CheckFailed(f"нет соединения с {urlsplit(url).netloc}: {reason}") from None


def expect_json(method, url, expected=(200,), **request):
    """Запрос, от которого ждём JSON и один из кодов expected."""
    status, body = http(method, url, **request)
    if status not in expected:
        raise CheckFailed(
            f"{method} {urlsplit(url).path} -> HTTP {status}: {body[:300].decode('utf-8', 'replace')}"
        )
    try:
        return json.loads(body)
    except ValueError:
        raise CheckFailed(f"{method} {urlsplit(url).path}: в ответе не JSON") from None


def multipart(field, filename, content, fields=()):
    """Тело и заголовки multipart/form-data: один файл и текстовые поля."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
    parts.append(head.encode() + b"Content-Type: application/octet-stream\r\n\r\n" + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def same_host(base, link):
    """Ссылка из ответа сервера, но с тем хостом, по которому мы сами к нему обращаемся."""
    target = urlsplit(link)
    return base + target.path + (f"?{target.query}" if target.query else "")


def wait_for(deadline, what, poll):
    """Опрашивает poll(), пока тот не вернёт непустое значение. Не дождались к сроку — CheckFailed."""
    while True:
        result = poll()
        if result:
            return result
        if time.monotonic() > deadline:
            raise CheckFailed(f"{what}: не дождались результата")
        time.sleep(POLL_SEC)


# --- Что считается успешным распознаванием -------------------------------------------------


def require_russian(text, engine_hint=""):
    """Проверяет, что в тексте есть слова образца. Возвращает краткое описание найденного."""
    normalized = text.lower().replace("ё", "е")
    found = [word for word in KEYWORDS if word in normalized]
    if len(found) >= ENOUGH_KEYWORDS:
        return f"русский текст распознан ({len(found)} из {len(KEYWORDS)} слов образца)"
    cyrillic = sum("а" <= char <= "я" for char in normalized)
    if cyrillic == 0:
        raise CheckFailed(
            f"в ответе нет кириллицы ({len(text.strip())} символов без неё). {engine_hint}".strip()
        )
    preview = " ".join(text.split())[:160]
    raise CheckFailed(
        f"текст образца не узнан: найдено слов {len(found)} из {len(KEYWORDS)}. Ответ: «{preview}»"
    )


# --- Проверки движков -----------------------------------------------------------------------


def check_docling(url, timeout):
    expect_json("GET", url + "/health")
    fields = [("to_formats", "md"), ("do_ocr", "false"), ("image_export_mode", "placeholder")]
    body, headers = multipart("files", "ru-text.pdf", (SAMPLES / "ru-text.pdf").read_bytes(), fields)
    reply = expect_json("POST", url + "/v1/convert/file", body=body, headers=headers, timeout=timeout)
    status = reply.get("status")
    markdown = (reply.get("document") or {}).get("md_content")
    if status not in ("success", "partial_success") or not markdown:
        raise CheckFailed(f"документ не преобразован: status={status}, errors={reply.get('errors')}")
    return require_russian(markdown)


def check_mineru(url, timeout):
    deadline = time.monotonic() + timeout
    status, body = http("GET", url + "/v1/health")
    if status != 200:
        hint = "модели не загрузились — смотрите docker logs mineru-api"
        raise CheckFailed(f"/v1/health -> HTTP {status}: {body[:300].decode('utf-8', 'replace')} ({hint})")
    tiers = [tier.get("id") for tier in expect_json("GET", url + "/v1/tiers").get("data", [])]

    scan = (SAMPLES / "ru-scan.png").read_bytes()
    manifest = json.dumps({"files": [{"file_name": "ru-scan.png", "bytes": len(scan)}]}).encode()
    json_headers = {"Content-Type": "application/json"}
    task = expect_json("POST", url + "/v1/tasks", expected=(200, 202), body=manifest, headers=json_headers)
    task_url = url + "/v1/tasks/" + quote(task["task_id"], safe="")
    upload = task["files"][0].get("upload")
    if upload:
        status, body = http(
            "PUT", same_host(url, upload["url"]), body=scan, headers=upload.get("headers") or {}
        )
        if status >= 300:
            raise CheckFailed(f"загрузка файла -> HTTP {status}: {body[:300].decode('utf-8', 'replace')}")

    def finished():
        file = expect_json("GET", task_url)["files"][0]
        if file["status"] in ("failed", "canceled"):
            raise CheckFailed(f"задача завершилась со статусом {file['status']}: {file.get('error')}")
        return file if file["status"] == "completed" else None

    file = wait_for(deadline, "MinerU", finished)
    status, archive = http("GET", same_host(url, file["result_url"]))
    if status != 200:
        raise CheckFailed(f"результат не скачан: HTTP {status}")
    with zipfile.ZipFile(io.BytesIO(archive)) as files:
        markdown = files.read("markdown.md").decode("utf-8", "replace")
    tier = (file.get("parse") or {}).get("tier", "?")
    hint = (
        f"Задачу выполнил уровень {tier}, сервер разрешает: {', '.join(tiers)}. На уровне basic текст читает "
        "PP-OCRv6, в словаре которой нет кириллицы: поставьте MINERU_TIER=standard в /opt/ai-doc-stack/.env "
        "и перезапустите стек (install.sh stack)."
    )
    return require_russian(markdown, hint) + f", уровень {tier}"


def check_paddle(url, timeout):
    deadline = time.monotonic() + timeout
    expect_json("GET", url + "/jobs")
    body, headers = multipart("file", "ru-scan.png", (SAMPLES / "ru-scan.png").read_bytes())
    job_url = url + "/ocr/" + expect_json("POST", url + "/ocr", body=body, headers=headers)["job_id"]

    def finished():
        job = expect_json("GET", job_url)
        if job["status"] in ("failed", "cancelled"):
            raise CheckFailed(f"задача завершилась со статусом {job['status']}: {job.get('error')}")
        return job["status"] == "completed"

    try:
        wait_for(deadline, "PaddleOCR-VL", finished)
        pages = expect_json("GET", job_url + "/result")["pages"]
    finally:
        http("DELETE", job_url)  # сервер хранит задачи бессрочно: убираем за собой
    return require_russian("\n".join(page["markdown"] for page in pages))


def check_ollama(url, model, timeout):
    names = [item.get("name", "") for item in expect_json("GET", url + "/api/tags").get("models", [])]
    if not any(name == model or name.startswith(model + ":") for name in names):
        raise CheckFailed(
            f"модели {model} нет в Ollama (есть: {', '.join(names) or 'ничего'}): выполните install.sh models"
        )
    scan = base64.b64encode((SAMPLES / "ru-scan.png").read_bytes()).decode()
    question = {
        "role": "user",
        "content": "Перепиши весь текст с изображения. Только текст, без пояснений.",
        "images": [scan],
    }
    payload = {"model": model, "messages": [question], "stream": False, "options": {"temperature": 0}}
    chat_headers = {"Content-Type": "application/json"}
    reply = expect_json(
        "POST", url + "/api/chat", body=json.dumps(payload).encode(), headers=chat_headers, timeout=timeout
    )
    summary = require_russian((reply.get("message") or {}).get("content", ""))

    # Модель обязана целиком лежать в видеопамяти: то же, что «100% GPU» в выводе `ollama ps`
    loaded = expect_json("GET", url + "/api/ps").get("models", [])
    current = next(
        (item for item in loaded if item.get("name", "").split(":")[0] == model.split(":")[0]), None
    )
    if current is None:
        raise CheckFailed("модель ответила, но в списке загруженных (/api/ps) её нет")
    size, in_vram = current.get("size", 0), current.get("size_vram", 0)
    if size and in_vram < size:
        share = round(100 * in_vram / size)
        raise CheckFailed(
            f"модель {model} не поместилась в видеопамять: на GPU {share} %, остальное считает процессор. "
            "Уменьшите num_ctx в профиле модели или освободите GPU 1"
        )
    return f"{summary}, модель целиком в видеопамяти ({in_vram / 2**30:.1f} ГБ)"


def check_orchestrator(url, api_key, with_paddle, timeout):
    expect_json("GET", url + "/health")
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def process(sample, engine, extra=None):
        headers = {"X-Filename": sample, "Content-Type": "application/octet-stream", **auth, **(extra or {})}
        content = (SAMPLES / sample).read_bytes()
        reply = expect_json("PUT", url + "/process", body=content, headers=headers, timeout=timeout)
        used = reply["metadata"]["engine_used"]
        if not used.startswith(engine):
            raise CheckFailed(f"{sample}: ждали движок {engine}, сработал {used}")
        require_russian(reply["page_content"], f"Файл {sample}, движок {used}.")
        return used

    engines = [process("ru-text.pdf", "docling"), process("ru-scan.png", "mineru")]

    # Счёт -> JSON: OCR (MinerU), затем Qwen3-VL собирает поля по схеме, затем валидаторы
    fields = [("doc_type", "invoice")]
    body, headers = multipart("file", "ru-scan.png", (SAMPLES / "ru-scan.png").read_bytes(), fields)
    reply = expect_json("POST", url + "/api/extract-structured", body=body, headers=headers, timeout=timeout)
    if reply.get("status") != "completed" or not isinstance(reply.get("data"), dict):
        raise CheckFailed(f"extract-structured: неожиданный ответ {str(reply)[:300]}")
    seller = (reply["data"].get("seller") or {}).get("inn")
    warnings = len((reply.get("validation") or {}).get("warnings") or [])

    if with_paddle:  # последним: смена модели на GPU 0 занимает минуты
        engines.append(process("ru-scan.png", "paddle", {"X-OCR-Engine": "paddle"}))
    summary = f"счёт разобран в JSON (ИНН продавца {seller}, замечаний валидатора: {warnings})"
    return f"движки: {', '.join(engines)}; {summary}"


# --- Запуск ---------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Проверка движков OCR/VLM-стека на образце с русским текстом."
    )
    parser.add_argument("--docling", metavar="URL", help="docling-serve, например http://localhost:5001")
    parser.add_argument("--mineru", metavar="URL", help="MinerU API, например http://localhost:8002")
    parser.add_argument("--paddle", metavar="URL", help="PaddleOCR-VL API, например http://localhost:8003")
    parser.add_argument("--ollama", metavar="URL", help="Ollama, например http://localhost:11434")
    parser.add_argument("--model", default="qwen3-vl-doc", help="модель Ollama (по умолчанию %(default)s)")
    parser.add_argument(
        "--orchestrator", metavar="URL", help="doc-orchestrator, например http://localhost:8000"
    )
    parser.add_argument("--api-key", default="", help="API_KEY оркестратора, если задан")
    parser.add_argument(
        "--with-paddle", action="store_true", help="через оркестратор проверить и PaddleOCR-VL"
    )
    parser.add_argument(
        "--timeout", type=float, default=600, help="предел на одну проверку, секунд (%(default)g)"
    )
    args = parser.parse_args(argv)

    def url(value):
        return value.rstrip("/")

    checks = []
    if args.docling:
        checks.append(("Docling", lambda: check_docling(url(args.docling), args.timeout)))
    if args.mineru:
        checks.append(("MinerU", lambda: check_mineru(url(args.mineru), args.timeout)))
    if args.paddle:
        checks.append(("PaddleOCR-VL", lambda: check_paddle(url(args.paddle), args.timeout)))
    if args.ollama:
        checks.append(
            (f"Ollama ({args.model})", lambda: check_ollama(url(args.ollama), args.model, args.timeout))
        )
    if args.orchestrator:
        checks.append(
            (
                "doc-orchestrator",
                lambda: check_orchestrator(
                    url(args.orchestrator), args.api_key, args.with_paddle, args.timeout
                ),
            )
        )
    if not checks:
        parser.error("не указано, что проверять: задайте хотя бы один адрес")

    failed = 0
    for title, check in checks:
        started = time.monotonic()
        try:
            print(f"[ OK ] {title}: {check()} — {time.monotonic() - started:.1f} с", flush=True)
        except CheckFailed as problem:
            failed += 1
            print(f"[FAIL] {title}: {problem}", flush=True)
        except (KeyError, IndexError, TypeError, ValueError, zipfile.BadZipFile) as problem:
            failed += 1  # сервер ответил не тем, что описано в его API
            print(
                f"[FAIL] {title}: неожиданный ответ сервера ({type(problem).__name__}: {problem})", flush=True
            )
    return failed


if __name__ == "__main__":
    sys.exit(main())
