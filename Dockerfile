# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    TURKCE_RAG_INDEX=/data/index

WORKDIR /app

# CPU'ya özel PyTorch: CUDA tekerleklerine göre imajı ~2 GB küçültür.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[dense,api]"

# Gömme modelini imaja göm: konteyner ilk istekte internete çıkmak zorunda kalmaz.
ARG EMBED_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')"

COPY data/sample ./data/sample

RUN useradd --create-home --uid 1000 rag && mkdir -p /data && chown rag /data
USER rag
VOLUME ["/data"]
EXPOSE 8000

ENTRYPOINT ["turkce-rag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
