"""
Turn the raw multi-tool analysis output into a clean multi-label table.

This is the heart of the "weak supervision" idea. SmartBugs-Wild ships with NO
ground-truth labels. What we have instead is results_wild.json: for every
contract, what each of nine analysis tools reported. We convert that into
labels by counting, per category, how many tools agree, and thresholding at
CONSENSUS_K.

IMPORTANT HONESTY NOTE
----------------------
These labels are NOT ground truth. They are the *opinion of a committee of
static/dynamic analysers*, which are themselves imperfect (both false positives
and false negatives are well documented in Durieux et al., ICSE 2020). A model
trained on these labels learns to *predict what the tools would say*, which is a
useful and legitimate task, but it is not the same as "detecting vulnerabilities
with ground-truth certainty." Every results table produced by this pipeline must
be read with that caveat. See docs/METHODOLOGY.md.
"""

from __future__ import annotations
import json
from collections import Counter, defaultdict

import pandas as pd

from . import config


def load_raw_results(path=None) -> dict:
    path = path or config.RESULTS_JSON
    with open(path) as fh:
        return json.load(fh)


def build_label_table(raw: dict | None = None,
                      consensus_k: int | None = None) -> pd.DataFrame:
    """
    Return a DataFrame indexed by contract address with one binary column per
    DASP category (1 = at least `consensus_k` tools flagged that category).

    Also attaches a `n_tools_any` column: how many tools flagged the contract
    for anything at all (useful for filtering out contracts no tool could
    process).
    """
    raw = raw if raw is not None else load_raw_results()
    k = consensus_k if consensus_k is not None else config.CONSENSUS_K

    rows = {}
    for addr, rec in raw.items():
        votes = Counter()                 # category -> number of tools flagging it
        tools_reporting = 0
        for tool, out in rec.get("tools", {}).items():
            cats = out.get("categories", {}) or {}
            hit = [c for c in cats if c in config.DASP_CATEGORIES]
            if cats:
                tools_reporting += 1
            for c in set(hit):            # count each tool at most once per category
                votes[c] += 1
        row = {c: int(votes[c] >= k) for c in config.DASP_CATEGORIES}
        row["n_tools_any"] = tools_reporting
        rows[addr] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "address"
    return df


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable positive-rate summary, one row per category."""
    n = len(df)
    out = []
    for c in config.DASP_CATEGORIES:
        pos = int(df[c].sum())
        out.append({"category": c, "positives": pos,
                    "rate_%": round(100 * pos / n, 2)})
    return pd.DataFrame(out).sort_values("positives", ascending=False)


if __name__ == "__main__":
    raw = load_raw_results()
    df = build_label_table(raw)
    df.to_csv(config.ARTIFACTS_DIR / "labels.csv")
    print(f"Wrote {len(df)} labelled contracts to artifacts/labels.csv")
    print("\nLabel distribution (consensus K =", config.CONSENSUS_K, "):")
    print(label_summary(df).to_string(index=False))
