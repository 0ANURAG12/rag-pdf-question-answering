FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf

WORKDIR /app

COPY requirements.txt ./

# Install CPU-only torch first so sentence-transformers does not pull the
# multi-GB CUDA build from PyPI (the CPU wheel already satisfies the dep).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user; /data is where ChromaDB and the HF cache live.
RUN useradd -m appuser \
    && mkdir -p /data/chroma /data/hf \
    && chown -R appuser:appuser /app /data

ENV CHROMA_DIR=/data/chroma

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
