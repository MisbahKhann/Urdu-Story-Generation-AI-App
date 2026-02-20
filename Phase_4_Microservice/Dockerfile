# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into a local prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY main.py          ./main.py
COPY bpe_tokenizer.py ./bpe_tokenizer.py
COPY trigram_model.py ./trigram_model.py

# Copy pre-trained model artefacts
# (these must be present in the build context – see README)
COPY bpe_tokenizer_trained.json    ./bpe_tokenizer_trained.json
COPY trigram_model_trained.json    ./trigram_model_trained.json

# ── Runtime config ────────────────────────────────────────────────────────────
ENV BPE_MODEL_PATH=bpe_tokenizer_trained.json
ENV TRIGRAM_MODEL_PATH=trigram_model_trained.json

EXPOSE 8000

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]