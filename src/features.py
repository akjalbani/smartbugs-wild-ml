"""
Feature extraction from Solidity source code.

We produce three representations, because different models need different
inputs:

  * TF-IDF matrix         -> Random Forest, SVM   (classical bag-of-tokens)
  * padded integer seqs   -> CNN                  (order-aware, learns n-grams)
  * token co-occurrence   -> GNN                  (graph of how tokens relate)

A light Solidity tokenizer is used throughout. It is deliberately simple
(regex-based, not a full Solidity parser) so it never crashes on the malformed
or exotic contracts that appear in a real-world corpus. Robustness beats
precision here: a tokenizer that dies on 3% of contracts would silently bias
the dataset.
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np

from . import config

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z_]\w+|[{}()\[\];,.=+\-*/%<>!&|^~?:]")


def strip_comments(src: str) -> str:
    return _COMMENT_RE.sub(" ", src)


def tokenize(src: str) -> list[str]:
    """Return a list of code tokens, comments removed, lower-cased identifiers."""
    src = strip_comments(src)
    toks = _TOKEN_RE.findall(src)
    return [t.lower() for t in toks]


# --------------------------------------------------------------------------- #
# Loading source
# --------------------------------------------------------------------------- #
def read_contract(address: str) -> str | None:
    """Read one contract's source by address; return None if unavailable."""
    path = config.CONTRACTS_DIR / f"{address}.sol"
    if not path.exists():
        return None
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return None


def load_corpus(addresses: list[str]) -> tuple[list[str], list[str]]:
    """
    Read source for each address. Returns (kept_addresses, raw_sources) for
    those that actually have a readable .sol file, preserving order.
    """
    kept, sources = [], []
    for a in addresses:
        s = read_contract(a)
        if s and s.strip():
            kept.append(a)
            sources.append(s)
    return kept, sources


# --------------------------------------------------------------------------- #
# 1. TF-IDF  (for Random Forest / SVM)
# --------------------------------------------------------------------------- #
def build_tfidf(sources: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        lowercase=False,
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        token_pattern=None,          # silence sklearn warning when tokenizer given
    )
    X = vec.fit_transform(sources)
    return X, vec


# --------------------------------------------------------------------------- #
# 2. Integer sequences  (for CNN)
# --------------------------------------------------------------------------- #
def build_vocab(sources: list[str], vocab_size: int) -> dict[str, int]:
    from collections import Counter
    counter = Counter()
    for s in sources:
        counter.update(tokenize(s))
    # index 0 = padding, 1 = out-of-vocabulary
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counter.most_common(vocab_size - 2):
        vocab[tok] = len(vocab)
    return vocab


def encode_sequences(sources: list[str], vocab: dict[str, int],
                     max_len: int) -> np.ndarray:
    unk = vocab["<unk>"]
    out = np.zeros((len(sources), max_len), dtype=np.int64)
    for i, s in enumerate(sources):
        toks = tokenize(s)[:max_len]
        for j, t in enumerate(toks):
            out[i, j] = vocab.get(t, unk)
    return out


# --------------------------------------------------------------------------- #
# 3. Token co-occurrence graph  (for GNN)
# --------------------------------------------------------------------------- #
def build_cooccurrence_graph(source: str, vocab: dict[str, int],
                             window: int = 3):
    """
    Build a small undirected graph for a single contract:
      * nodes  = unique in-vocab tokens present in the contract
      * edges  = tokens that appear within `window` positions of each other

    Returns (node_ids, edge_index) where edge_index is a 2 x E array. This is a
    lightweight, dependency-free stand-in for a full AST/CFG graph; it is enough
    to demonstrate the GNN pipeline end-to-end on Colab. docs/METHODOLOGY.md
    discusses upgrading to AST/CFG graphs.
    """
    toks = [vocab.get(t, vocab["<unk>"]) for t in tokenize(source)]
    node_list = sorted(set(toks))
    remap = {tok: i for i, tok in enumerate(node_list)}
    edges = set()
    for idx in range(len(toks)):
        for off in range(1, window + 1):
            if idx + off < len(toks):
                a, b = remap[toks[idx]], remap[toks[idx + off]]
                if a != b:
                    edges.add((a, b))
                    edges.add((b, a))
    if not node_list:
        node_list = [vocab["<pad>"]]
    edge_index = (np.array(sorted(edges)).T
                  if edges else np.zeros((2, 0), dtype=np.int64))
    return np.array(node_list, dtype=np.int64), edge_index.astype(np.int64)


if __name__ == "__main__":
    # Tiny self-test that needs no data on disk.
    sample = """
    // a comment
    contract Bank {
        mapping(address => uint) balances;
        function withdraw(uint amount) public {
            require(balances[msg.sender] >= amount);
            msg.sender.call.value(amount)("");
            balances[msg.sender] -= amount;
        }
    }
    """
    print("tokens:", tokenize(sample)[:20])
    vocab = build_vocab([sample], 100)
    print("vocab size:", len(vocab))
    nodes, edges = build_cooccurrence_graph(sample, vocab)
    print("graph nodes:", len(nodes), "edges:", edges.shape[1])
