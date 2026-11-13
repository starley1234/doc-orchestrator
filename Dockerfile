FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Зависимости — отдельным слоем: он кэшируется, пока не изменится requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app — сам оркестратор; gateway — шлюз движков для боевого сервера (docker-compose.prod.yml):
# тот же образ, другая команда запуска
COPY app ./app
COPY gateway ./gateway

# Сервису не нужны права root
USER nobody

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
