FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the embedding and reranker models into the image so containers start without downloading anything.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models')" \
 && python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/models')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
