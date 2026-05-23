FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# If dont want to rebuild the vector index comment the line below.
RUN python -c "from src.rag.index import build_chroma_index; build_chroma_index('knowledge_base', 'chroma_db')"

CMD ["python", "main.py"]
