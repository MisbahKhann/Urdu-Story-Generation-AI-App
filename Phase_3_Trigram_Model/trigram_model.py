import collections
import random
import math
from typing import Optional


# ─────────────────────────────────────────────
# Trigram Language Model with Interpolation
# ─────────────────────────────────────────────

class TrigramModel:
    """
    A trigram language model trained with Maximum Likelihood Estimation (MLE)
    and smoothed via linear interpolation of unigram, bigram, and trigram
    probabilities.

    Interpolation formula (Jelinek-Mercer):
        P(w3 | w1, w2) = λ3·P_tri(w3|w1,w2)
                       + λ2·P_bi(w3|w2)
                       + λ1·P_uni(w3)

    where λ1 + λ2 + λ3 = 1.
    """

    EOT_TOKEN = "<EOT>"
    EOS_TOKEN = "<EOS>"
    EOP_TOKEN = "<EOP>"

    def __init__(
        self,
        lambda1: float = 0.1,   # unigram weight
        lambda2: float = 0.3,   # bigram weight
        lambda3: float = 0.6,   # trigram weight
    ):
        assert abs(lambda1 + lambda2 + lambda3 - 1.0) < 1e-6, \
            "Interpolation weights must sum to 1."
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Raw counts
        self._unigram_counts: dict[str, int] = collections.defaultdict(int)
        self._bigram_counts: dict[tuple, int] = collections.defaultdict(int)
        self._trigram_counts: dict[tuple, int] = collections.defaultdict(int)

        # Smoothed probability tables (built after training)
        self._unigram_probs: dict[str, float] = {}
        self._bigram_probs: dict[tuple, dict[str, float]] = {}
        self._trigram_probs: dict[tuple, dict[str, float]] = {}

        self._vocab_size: int = 0
        self._total_tokens: int = 0

    # ── Training ────────────────────────────────

    def train(self, corpus: str) -> None:
        """
        Build unigram, bigram, and trigram count tables from *corpus*,
        then convert counts to MLE probabilities.
        """
        tokens = corpus.split()
        if len(tokens) < 3:
            raise ValueError("Corpus too short to train a trigram model.")

        self._total_tokens = len(tokens)

        # Count
        for tok in tokens:
            self._unigram_counts[tok] += 1

        for i in range(len(tokens) - 1):
            self._bigram_counts[(tokens[i], tokens[i + 1])] += 1

        for i in range(len(tokens) - 2):
            ctx = (tokens[i], tokens[i + 1])
            self._trigram_counts[(ctx, tokens[i + 2])] += 1

        self._vocab_size = len(self._unigram_counts)

        # Convert to MLE probabilities
        # Unigram
        for tok, cnt in self._unigram_counts.items():
            self._unigram_probs[tok] = cnt / self._total_tokens

        # Bigram  P(w2 | w1)
        bigram_context_totals: dict[str, int] = collections.defaultdict(int)
        for (w1, _), cnt in self._bigram_counts.items():
            bigram_context_totals[w1] += cnt

        for (w1, w2), cnt in self._bigram_counts.items():
            if w1 not in self._bigram_probs:
                self._bigram_probs[w1] = {}
            self._bigram_probs[w1][w2] = cnt / bigram_context_totals[w1]

        # Trigram  P(w3 | w1, w2)
        trigram_context_totals: dict[tuple, int] = collections.defaultdict(int)
        for (ctx, _), cnt in self._trigram_counts.items():
            trigram_context_totals[ctx] += cnt

        for (ctx, w3), cnt in self._trigram_counts.items():
            if ctx not in self._trigram_probs:
                self._trigram_probs[ctx] = {}
            self._trigram_probs[ctx][w3] = cnt / trigram_context_totals[ctx]

        print(
            f"[Trigram] Training complete. "
            f"Tokens={self._total_tokens:,}  Vocab={self._vocab_size:,}  "
            f"Trigram contexts={len(self._trigram_probs):,}"
        )

    # ── Interpolated probability ─────────────────

    def _interpolated_prob(self, w1: str, w2: str, w3: str) -> float:
        """
        Return the interpolated probability P(w3 | w1, w2).
        Falls back to uniform for unseen tokens.
        """
        p_uni = self._unigram_probs.get(w3, 1.0 / max(self._vocab_size, 1))
        p_bi  = self._bigram_probs.get(w1, {}).get(w2, p_uni)
        # Note: bigram for "P(w3|w2)" uses w2 as context
        p_bi  = self._bigram_probs.get(w2, {}).get(w3, p_uni)
        p_tri = self._trigram_probs.get((w1, w2), {}).get(w3, 0.0)

        return self.lambda3 * p_tri + self.lambda2 * p_bi + self.lambda1 * p_uni

    # ── Candidate next-token distribution ────────

    def _next_token_dist(self, w1: str, w2: str) -> dict[str, float]:
        """
        Return the full interpolated probability distribution over all
        candidate next tokens given context (w1, w2).
        """
        ctx = (w1, w2)
        # Candidate tokens: trigram candidates ∪ bigram candidates ∪ full vocab
        candidates = set(self._trigram_probs.get(ctx, {}).keys())
        candidates |= set(self._bigram_probs.get(w2, {}).keys())
        candidates |= set(self._unigram_probs.keys())

        dist = {w3: self._interpolated_prob(w1, w2, w3) for w3 in candidates}
        # Normalise
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        return dist

    # ── Text generation ──────────────────────────

    def generate(
        self,
        seed: list[str],
        max_length: int = 300,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        random_seed: Optional[int] = None,
    ) -> str:
        """
        Generate text starting from *seed* tokens.

        Parameters
        ----------
        seed        : list of starting tokens (at least 2)
        max_length  : maximum number of tokens to generate beyond the seed
        temperature : > 1 → more random; < 1 → more deterministic
        top_k       : if set, sample only from the top-k tokens
        random_seed : for reproducibility

        Generation stops when <EOT> is produced or max_length is reached.
        """
        if len(seed) < 2:
            raise ValueError("Seed must contain at least 2 tokens.")
        if not self._unigram_probs:
            raise RuntimeError("Model not trained yet.")

        if random_seed is not None:
            random.seed(random_seed)

        tokens = list(seed)

        for _ in range(max_length):
            w1, w2 = tokens[-2], tokens[-1]
            dist = self._next_token_dist(w1, w2)

            if not dist:
                break

            # Apply temperature
            if temperature != 1.0:
                dist = {k: v ** (1.0 / temperature) for k, v in dist.items()}
                total = sum(dist.values())
                dist = {k: v / total for k, v in dist.items()}

            # Top-k filtering
            if top_k is not None:
                sorted_items = sorted(dist.items(), key=lambda x: -x[1])[:top_k]
                dist = dict(sorted_items)
                total = sum(dist.values())
                dist = {k: v / total for k, v in dist.items()}

            # Weighted random sample
            words = list(dist.keys())
            weights = list(dist.values())
            next_token = random.choices(words, weights=weights, k=1)[0]

            tokens.append(next_token)

            if next_token == self.EOT_TOKEN:
                break

        return " ".join(tokens)

    # ── Perplexity ───────────────────────────────

    def perplexity(self, test_corpus: str) -> float:
        """
        Compute perplexity of *test_corpus* under the interpolated model.
        Lower is better.
        """
        tokens = test_corpus.split()
        if len(tokens) < 3:
            raise ValueError("Test corpus too short.")

        log_prob_sum = 0.0
        count = 0
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            p = self._interpolated_prob(w1, w2, w3)
            log_prob_sum += math.log(max(p, 1e-10))
            count += 1

        return math.exp(-log_prob_sum / count)

    # ── Persistence ──────────────────────────────

    def save(self, path: str) -> None:
        """Serialise model to a JSON file."""
        import json

        # Convert tuple keys to strings for JSON compatibility
        def tri_key(ctx, w3):
            return f"{ctx[0]}\t{ctx[1]}\t{w3}"

        state = {
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "lambda3": self.lambda3,
            "unigram_probs": self._unigram_probs,
            "bigram_probs": {
                w1: probs for w1, probs in self._bigram_probs.items()
            },
            "trigram_probs": {
                f"{ctx[0]}\t{ctx[1]}": probs
                for ctx, probs in self._trigram_probs.items()
            },
            "vocab_size": self._vocab_size,
            "total_tokens": self._total_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        print(f"[Trigram] Saved to {path}")

    def load(self, path: str) -> None:
        """Restore model from a JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.lambda1 = state["lambda1"]
        self.lambda2 = state["lambda2"]
        self.lambda3 = state["lambda3"]
        self._unigram_probs = state["unigram_probs"]
        self._bigram_probs = state["bigram_probs"]
        self._trigram_probs = {
            tuple(k.split("\t")): probs
            for k, probs in state["trigram_probs"].items()
        }
        self._vocab_size = state["vocab_size"]
        self._total_tokens = state["total_tokens"]
        print(f"[Trigram] Loaded from {path}  (vocab={self._vocab_size:,})")