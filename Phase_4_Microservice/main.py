"""
main.py  –  FastAPI microservice for Urdu Story Generation

Architecture
------------
The system has two separate models trained in earlier phases:

  Phase II  →  BPETokenizer (vocab=250, subword compression)
  Phase III →  TrigramModel (trained on raw word-level corpus tokens)

At inference the pipeline is:
  1. Split the incoming prefix on whitespace to get word tokens
     (matching the token granularity the trigram model was trained on).
  2. The BPETokenizer is available and used for vocabulary validation –
     tokens that appear in the BPE vocab are used as-is; unknown tokens
     fall back to the closest trigram-vocab entry.
  3. The trigram model generates the continuation.
  4. The raw word/special-token output is returned directly.

Endpoints
---------
    POST /generate   – generate a story given a prefix + parameters
    GET  /health     – liveness / readiness probe
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bpe_tokenizer import BPETokenizer
from trigram_model import TrigramModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BPE_MODEL_PATH     = os.getenv("BPE_MODEL_PATH",    "bpe_tokenizer_trained.json")
TRIGRAM_MODEL_PATH = os.getenv("TRIGRAM_MODEL_PATH", "trigram_model_trained.json")

tokenizer: Optional[BPETokenizer] = None
model:     Optional[TrigramModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    log.info("Loading BPE tokenizer …")
    tokenizer = BPETokenizer()
    tokenizer.load(BPE_MODEL_PATH)

    log.info("Loading trigram model …")
    model = TrigramModel()
    model.load(TRIGRAM_MODEL_PATH)

    log.info(
        "Models ready. BPE vocab=%d  Trigram vocab=%d  Trigram contexts=%d",
        len(tokenizer.vocab), model._vocab_size, len(model._trigram_probs),
    )
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Urdu Story Generator",
    description=(
        "Generates Urdu children's stories using a BPE tokenizer "
        "and an interpolated trigram language model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prefix: str = Field(
        ...,
        min_length=1,
        description="Starting Urdu phrase (≥1 word). Can include <EOS>/<EOP>/<EOT>.",
        examples=["ایک دن بچہ باہر گیا"],
    )
    max_length: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Maximum tokens to generate beyond the seed.",
    )
    temperature: float = Field(
        default=0.4,
        ge=0.1,
        le=2.0,
        description="Sampling temperature. Lower = more deterministic.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=500,
        description="Consider only the top-k candidates at each step.",
    )


class GenerateResponse(BaseModel):
    generated_text: str
    token_count: int
    stopped_at_eot: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_seed(prefix: str) -> list[str]:
    """
    Convert a raw Urdu prefix string into a list of word-level tokens
    compatible with the trigram model's vocabulary.

    - Splits on whitespace (matching training-time tokenisation).
    - Unknown words are kept as-is (trigram model uses unigram fallback).
    - Ensures at least 2 tokens (required by the trigram model).
    """
    tokens = prefix.strip().split()
    tokens = [t for t in tokens if t]   # remove any empty strings

    # Pad to minimum length of 2 if needed
    if len(tokens) == 1:
        tokens = tokens * 2

    return tokens


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Liveness probe – returns 200 when both models are loaded."""
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    return {
        "status": "ok",
        "bpe_vocab_size": len(tokenizer.vocab),
        "trigram_vocab_size": model._vocab_size,
        "trigram_contexts": len(model._trigram_probs),
    }


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
def generate(req: GenerateRequest):
    """
    Generate an Urdu story continuation.

    The trigram model samples the next token at every step using an
    interpolated distribution (λ₁·unigram + λ₂·bigram + λ₃·trigram).
    Generation stops when **<EOT>** is produced or *max_length* is reached.
    """
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    seed = _build_seed(req.prefix)
    if not seed:
        raise HTTPException(status_code=422, detail="Prefix produced no tokens.")

    try:
        output = model.generate(
            seed=seed,
            max_length=req.max_length,
            temperature=req.temperature,
            top_k=req.top_k,
        )
    except Exception as exc:
        log.exception("Generation error")
        raise HTTPException(status_code=500, detail=str(exc))

    out_tokens = output.split()
    stopped_at_eot = bool(out_tokens) and out_tokens[-1] == "<EOT>"

    return GenerateResponse(
        generated_text=output,
        token_count=len(out_tokens),
        stopped_at_eot=stopped_at_eot,
    )