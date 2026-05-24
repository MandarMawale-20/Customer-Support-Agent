FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY CAG/requirements.txt /tmp/CAG-requirements.txt
COPY RAG/requirements.txt /tmp/RAG-requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/CAG-requirements.txt -r /tmp/RAG-requirements.txt

COPY CAG /app/CAG
COPY RAG /app/RAG
COPY frontend /app/frontend
COPY server.py /app/server.py
COPY requirements.txt /app/requirements.txt
COPY supervisord.conf /app/supervisord.conf

EXPOSE 3000 8000 8001 8080 8081

ENTRYPOINT ["supervisord", "-c", "/app/supervisord.conf"]