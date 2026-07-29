"""
Evaluation and statistics.

Everything here operates on probability matrices of shape (N, C) and binary
label matrices of the same shape. We report:

  * per-category precision / recall / F1
  * micro- and macro-averaged F1
  * bootstrap 95% confidence intervals for micro-F1
  * McNemar's test for pairwise model comparison

No number in this module is invented; each is computed from predictions you
generate. If you did not run a model, you get no row for it.
"""

from __future__ import annotations
import numpy as np

from . import config


def binarize(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob >= threshold).astype(int)


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def per_category_report(Y_true, Y_pred) -> "list[dict]":
    from sklearn.metrics import precision_recall_fscore_support
    rows = []
    for i, cat in enumerate(config.DASP_CATEGORIES):
        p, r, f, _ = precision_recall_fscore_support(
            Y_true[:, i], Y_pred[:, i], average="binary", zero_division=0)
        rows.append({"category": cat,
                     "precision": round(float(p), 4),
                     "recall": round(float(r), 4),
                     "f1": round(float(f), 4),
                     "support": int(Y_true[:, i].sum())})
    return rows


def averaged_f1(Y_true, Y_pred) -> "dict":
    from sklearn.metrics import f1_score
    return {
        "micro_f1": round(float(f1_score(Y_true, Y_pred, average="micro",
                                         zero_division=0)), 4),
        "macro_f1": round(float(f1_score(Y_true, Y_pred, average="macro",
                                         zero_division=0)), 4),
    }


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval for micro-F1
# --------------------------------------------------------------------------- #
def bootstrap_micro_f1_ci(Y_true, Y_pred, n_boot=1000, alpha=0.05, seed=None):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
    n = len(Y_true)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        scores[b] = f1_score(Y_true[idx], Y_pred[idx],
                             average="micro", zero_division=0)
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return round(float(scores.mean()), 4), (round(float(lo), 4), round(float(hi), 4))


# --------------------------------------------------------------------------- #
# McNemar's test for two models
# --------------------------------------------------------------------------- #
def mcnemar_test(Y_true, pred_a, pred_b):
    """
    Compare two models' correctness on the SAME test set.

    We flatten the multi-label predictions to per-(contract, category)
    decisions, then build the 2x2 contingency of where A and B disagree:

              B correct   B wrong
    A correct     -          b
    A wrong       c          -

    The statistic uses only the discordant cells b and c. Returns
    (statistic, p_value, b, c). Small p => the two models differ significantly.
    """
    from scipy.stats import chi2

    yt = Y_true.ravel()
    a_correct = (pred_a.ravel() == yt)
    b_correct = (pred_b.ravel() == yt)
    b = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    c = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    if b + c == 0:
        return 0.0, 1.0, b, c
    # continuity-corrected McNemar
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(chi2.sf(stat, df=1))
    return round(float(stat), 4), round(p, 6), b, c


if __name__ == "__main__":
    # Self-test on synthetic data so the file runs with no dataset present.
    rng = np.random.default_rng(0)
    N, C = 500, len(config.DASP_CATEGORIES)
    Y = (rng.random((N, C)) < 0.2).astype(int)
    good = np.where(rng.random((N, C)) < 0.8, Y, 1 - Y)   # 80% accurate
    bad = np.where(rng.random((N, C)) < 0.6, Y, 1 - Y)    # 60% accurate
    print("averaged F1 (good):", averaged_f1(Y, good))
    mean, ci = bootstrap_micro_f1_ci(Y, good, n_boot=300)
    print("micro-F1 bootstrap:", mean, "95% CI", ci)
    print("McNemar good-vs-bad:", mcnemar_test(Y, good, bad))
