# Methodology and Limitations

This document explains, in detail, how the pipeline turns an unlabeled corpus
into a supervised learning problem, why that is defensible, and where it falls
short. Read it before drawing conclusions from any result — and cite its caveats
in any paper.

---

## 1. The labeling problem

SmartBugs-Wild is 47,398 contracts scraped from Ethereum via Etherscan. None
carry a vulnerability label. To do supervised learning we need a target `y` for
each contract, and we have to manufacture it.

### What we use as a signal

The companion dataset [smartbugs-results](https://github.com/smartbugs/smartbugs-results)
contains, for every contract, the findings of **nine** analysis tools:

`mythril, slither, oyente, osiris, smartcheck, manticore, maian, securify, honeybadger`

Each tool's raw findings are mapped to the **DASP** taxonomy categories via the
official `vulnerabilities_mapping.csv`. So for each contract we know, per tool,
which DASP categories were raised.

### From tool findings to labels: consensus voting

We define a contract as positive for category *c* when **at least K of the nine
tools** report *c*:

```
label(contract, c) = 1  if  (number of tools reporting c) >= K
                      0  otherwise
```

`K` is set in `config.CONSENSUS_K` (default 2). This is the single most
consequential modeling choice in the project.

- **K = 1** — any single tool is enough. Maximizes positives, maximizes noise.
- **K = 2** — a light consensus filter. Removes a lot of lone-tool false alarms.
  Our default.
- **K ≥ 3** — high precision, but rare classes nearly vanish.

There is no objectively correct K, because there is no ground truth to tune it
against on this corpus. The honest move is to pick one, justify it, and report
how sensitive your conclusions are to it (the tutorial shows how).

---

## 2. Why this is legitimate — and what it is *not*

**Legitimate.** "Predict the consensus of a panel of established analyzers" is a
well-defined supervised task. A model that does it well could, for example,
triage a large contract set far faster than running nine heavyweight tools, or
flag likely-vulnerable contracts for human review. Weak supervision from noisy
labelers is an established paradigm in ML.

**What it is not.** It is **not** ground-truth vulnerability detection. The tools
are wrong sometimes — both directions:

- *False positives:* Durieux et al. (ICSE 2020) found the tools raise many
  warnings that are not real vulnerabilities (notably in the `arithmetic`
  category, which our label distribution shows dominating at 56%).
- *False negatives:* real bugs the tools miss never become positive labels, so
  the model is never taught them.

Therefore a high F1 here means **"agrees well with the tool committee,"** not
"finds real bugs with certainty." Every results table must be read and reported
with that framing. Overstating it would be misleading.

---

## 3. DASP vs. OWASP

SmartBugs labels use the **DASP-10** taxonomy. The newer **OWASP Smart Contract
Top-10** is related but not identical. An approximate mapping:

| DASP (this dataset)   | Closest OWASP SC Top-10 idea         |
|-----------------------|--------------------------------------|
| reentrancy            | Reentrancy                           |
| access_control        | Access Control                       |
| arithmetic            | Integer Over/Underflow               |
| unchecked_low_calls   | Unchecked External Calls             |
| denial_service        | Denial of Service                    |
| front_running         | Front-Running / MEV                  |
| time_manipulation     | Block-values / Timestamp dependence  |
| bad_randomness        | Insecure randomness *(absent here)*  |
| short_addresses       | *(legacy; absent here)*              |

If your paper is framed around OWASP, say clearly that the *labels* are DASP-
derived and give this mapping. Do not silently relabel DASP categories as OWASP
ones — they were produced under the DASP definitions.

### Dropped categories

`bad_randomness` and `short_addresses` have **zero** positives in the tool
outputs for this corpus. We exclude them (see `config.DASP_CATEGORIES`) because a
classifier for a class with no positive examples is meaningless and would
otherwise inflate averages with a trivial "always-negative" perfect score. This
leaves **7** working categories.

---

## 4. Features

Three representations, each feeding the models that suit it:

1. **TF-IDF over code tokens** (Random Forest, SVM). A robust, interpretable
   bag-of-tokens baseline. Comments are stripped; a regex tokenizer keeps it
   crash-proof on malformed contracts.
2. **Padded integer token sequences** (CNN). Preserves local ordering so the
   convolution can learn short code patterns (e.g. a `.call` followed by a state
   update — the reentrancy signature).
3. **Token co-occurrence graph** (GNN). Nodes are tokens, edges join tokens that
   appear within a small window. This is a **deliberately lightweight** stand-in
   for a true program graph.

### Known limitation of the GNN graph

A token co-occurrence graph captures *lexical* proximity, not real program
semantics. A faithful approach would build an **AST** or **control-/data-flow
graph** (e.g. via `solc`/`py-solc-x` or Slither's IR). We avoided that in the
default pipeline because compiling 47k contracts across many Solidity versions
is fragile and slow on Colab — but it is the single most promising upgrade, and
`features.build_cooccurrence_graph` is isolated so you can replace it cleanly.

---

## 5. Models and training protocol

| Model         | Input            | Notes |
|---------------|------------------|-------|
| Random Forest | TF-IDF           | 200 trees, class-weighted, multi-output |
| Linear SVM    | TF-IDF           | calibrated for probabilities, class-weighted |
| CNN           | token sequences  | multi-kernel 1-D conv, BCE with pos-weighting |
| GNN           | co-occurrence    | 2-layer mean-aggregation graph net |
| Ensemble      | (above outputs)  | soft-voting average of probabilities |

Shared protocol, applied identically to every model for a fair comparison:

- One fixed random seed (`config.RANDOM_SEED = 42`).
- The same train/val/test split (70/10/20 by default).
- **Class imbalance** handled by `class_weight="balanced"` (classical) and BCE
  `pos_weight` (neural), so rare classes aren't ignored.
- **Multi-label**: each category is an independent 0/1 decision; one contract
  may have several. Probabilities thresholded at 0.5 (tunable).

---

## 6. Evaluation

- **Per-category precision/recall/F1** — where a model wins or loses.
- **Micro-F1** — pooled over all decisions; reflects the common classes.
- **Macro-F1** — unweighted mean over categories; reflects rare-class ability.
  Report both; the gap between them is informative.
- **Bootstrap 95% CIs** (1,000 resamples) on micro-F1 — quantifies test-set
  uncertainty. Overlapping intervals ⇒ no real difference.
- **McNemar's test** (continuity-corrected) — pairwise significance on the same
  test set. Guards against claiming a win that is within noise.

All of these are implemented from the predictions in `src/evaluate.py`; none are
hard-coded.

---

## 7. Threats to validity (state these in your paper)

1. **Label noise / no ground truth.** The dominant limitation. Scores measure
   agreement with imperfect tools, not true detection.
2. **Class imbalance.** `arithmetic` at 56% can flatter aggregate metrics; lean
   on macro-F1 and per-class numbers.
3. **Dropped classes.** `bad_randomness`, `short_addresses` absent → the study
   covers 7 of 10 DASP categories, not all.
4. **Simplified GNN graph.** Lexical, not semantic; likely understates what a
   proper program-graph GNN could achieve.
5. **Feature leakage risk.** If a tool's *own* signature strings leaked into the
   source text used for features, a model could "cheat." We strip comments to
   reduce this; be alert to it if scores look implausibly high.
6. **Solidity version drift.** The corpus is largely older Solidity; conclusions
   may not transfer to current (0.8+) contracts where arithmetic checks are
   built in.

---

## 8. Recommended honesty checklist before publishing

- [ ] Stated that labels are ≥K-of-9 tool consensus, not ground truth.
- [ ] Reported the label distribution and the class imbalance.
- [ ] Reported macro-F1 as well as micro-F1.
- [ ] Included confidence intervals and McNemar results.
- [ ] Disclosed the dropped categories and the 7-category scope.
- [ ] Disclosed the GNN graph simplification.
- [ ] (Ideally) validated on SmartBugs-Curated ground truth and reported that
      separately.

Meeting this checklist is what makes the work reproducible *and* trustworthy —
the reason to open-source it in the first place.

---

## References

- T. Durieux, J. F. Ferreira, R. Abreu, P. Cruz. *Empirical Review of Automated
  Analysis Tools on 47,587 Ethereum Smart Contracts.* ICSE 2020.
- SmartBugs framework and datasets: <https://github.com/smartbugs>
- DASP Top 10 taxonomy: <https://dasp.co>
- OWASP Smart Contract Top 10: <https://owasp.org>
