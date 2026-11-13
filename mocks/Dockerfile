FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Заглушке хватает одного FastAPI (версии — как в requirements.txt оркестратора)
RUN pip install --no-cache-dir fastapi==0.162.5 uvicorn==0.54.0 python-multipart==0.0.33

COPY main.py .

USER nobody

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
