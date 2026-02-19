import re
import json
import collections


# ─────────────────────────────────────────────
# Core BPE helpers
# ─────────────────────────────────────────────

def get_stats(vocab: dict) -> dict:
    """Count adjacent-pair frequencies in the weighted vocabulary."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair: tuple, vocab_in: dict) -> dict:
    """Merge the most frequent pair across every vocabulary entry."""
    vocab_out = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    merged = "".join(pair)
    for word, freq in vocab_in.items():
        vocab_out[pattern.sub(merged, word)] = freq
    return vocab_out


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_corpus(txt_path: str) -> str:
    """Load the preprocessed plain-text corpus (preferred for training)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_data_from_json(json_path: str) -> str:
    """Load raw Urdu stories from the scraped JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                # Support common key names
                text = item.get("full_text") or item.get("story") or item.get("text", "")
                texts.append(text)
    return " ".join(texts).strip()


# ─────────────────────────────────────────────
# BPE Tokenizer class
# ─────────────────────────────────────────────

class BPETokenizer:
    """
    A from-scratch Byte-Pair Encoding tokenizer for Urdu text.

    Attributes
    ----------
    vocab       : set of token strings after training
    merges      : ordered list of (pair -> merged) merge rules
    token2id    : mapping from token string to integer id
    id2token    : reverse mapping
    """

    END_OF_WORD = "</w>"

    def __init__(self):
        self.vocab: set = set()
        self.merges: list[tuple[tuple, str]] = []   # [(pair, merged), ...]
        self.token2id: dict = {}
        self.id2token: dict = {}

    # ── Training ────────────────────────────────

    def train(self, corpus: str, target_vocab_size: int = 250) -> None:
        """
        Train BPE on *corpus* until the vocabulary reaches *target_vocab_size*.
        Special tokens (<EOS>, <EOP>, <EOT>) are preserved as single units.
        """
        # ── 1. Identify and protect special tokens ─────────────────────────
        special_tokens = {"<EOS>", "<EOP>", "<EOT>"}

        # Replace special tokens with placeholders so they survive word-splitting
        placeholder_map = {}
        safe_corpus = corpus
        for i, tok in enumerate(special_tokens):
            placeholder = f"SPECIAL{i}PLACEHOLDER"
            placeholder_map[placeholder] = tok
            safe_corpus = safe_corpus.replace(tok, f" {placeholder} ")

        # ── 2. Build initial character-level vocabulary ─────────────────────
        raw_vocab: dict = collections.defaultdict(int)
        for word in safe_corpus.split():
            if word in placeholder_map:
                raw_vocab[word] += 1          # treat as atomic unit
            else:
                char_seq = " ".join(list(word)) + f" {self.END_OF_WORD}"
                raw_vocab[char_seq] += 1

        # Restore placeholder labels in vocab keys so merges work on real text
        vocab: dict = {}
        for key, freq in raw_vocab.items():
            if key in placeholder_map:
                vocab[placeholder_map[key]] = freq
            else:
                vocab[key] = freq

        # ── 3. Collect initial unique tokens ───────────────────────────────
        unique_tokens: set = set()
        for word in vocab:
            for ch in word.split():
                unique_tokens.add(ch)

        num_merges = max(0, target_vocab_size - len(unique_tokens))
        print(
            f"[BPE] target={target_vocab_size} | "
            f"initial tokens={len(unique_tokens)} | "
            f"planned merges={num_merges}"
        )

        # ── 4. Iterative merging ────────────────────────────────────────────
        self.merges = []
        for i in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break

            # Never merge across special tokens
            pairs = {
                pair: freq
                for pair, freq in pairs.items()
                if pair[0] not in special_tokens and pair[1] not in special_tokens
            }
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            merged = "".join(best)
            vocab = merge_vocab(best, vocab)
            unique_tokens.add(merged)
            self.merges.append((best, merged))

            if (i + 1) % 10 == 0:
                print(f"  merge {i+1:>4}: {''.join(best)!r}  (freq={pairs[best]})")

        self.vocab = unique_tokens | special_tokens
        self._build_index()
        print(f"[BPE] Training complete. Final vocab size: {len(self.vocab)}")

    # ── Index helpers ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        self.token2id = {tok: idx for idx, tok in enumerate(sorted(self.vocab))}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    # ── Encoding / Decoding ──────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned merge rules to a single word (already split into chars)."""
        # Start as character sequence
        symbols = list(word) + [self.END_OF_WORD]

        for (pair, merged) in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == pair[0]
                    and symbols[i + 1] == pair[1]
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str) -> list[int]:
        """
        Convert *text* into a list of token ids.
        Special tokens (<EOS>, <EOP>, <EOT>) are preserved as single tokens.
        Unknown characters are kept as-is (treated as single-char tokens).
        """
        if not self.token2id:
            raise RuntimeError("Tokenizer not trained yet.")

        special_tokens = {"<EOS>", "<EOP>", "<EOT>"}
        # Split preserving special tokens
        parts = re.split(r"(<EOS>|<EOP>|<EOT>)", text)

        ids = []
        for part in parts:
            if not part:
                continue
            if part in special_tokens:
                ids.append(self.token2id.get(part, 0))
            else:
                for word in part.split():
                    tokens = self._tokenize_word(word)
                    for tok in tokens:
                        ids.append(self.token2id.get(tok, 0))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token ids back to a string."""
        if not self.id2token:
            raise RuntimeError("Tokenizer not trained yet.")

        tokens = [self.id2token.get(i, "") for i in ids]
        text = " ".join(tokens)
        # Remove end-of-word markers and clean up spacing
        text = text.replace(f" {self.END_OF_WORD}", "").replace(self.END_OF_WORD, "")
        return text.strip()

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save tokenizer state to a JSON file."""
        state = {
            "vocab": list(self.vocab),
            "merges": [[list(pair), merged] for pair, merged in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[BPE] Saved to {path}")

    def load(self, path: str) -> None:
        """Restore tokenizer state from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.vocab = set(state["vocab"])
        self.merges = [(tuple(pair), merged) for pair, merged in state["merges"]]
        self._build_index()
        print(f"[BPE] Loaded from {path}  (vocab={len(self.vocab)})")