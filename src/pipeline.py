"""
End-to-end pipeline: labels -> sample -> features -> train 5 models -> evaluate.

Run from the repo root with:

    python -m src.pipeline

or import `run()` from a notebook. Every stage prints what it is doing so a
first-time user can see progress and spot where something went wrong.

The defaults in config.py keep the first run small (a few thousand contracts,
CPU-friendly). Raise MAX_CONTRACTS once you trust the setup.
"""

from __future__ import annotations
import json
import numpy as np

from . import config, labeling, features, models, evaluate


def _split_indices(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * config.TEST_SIZE)
    n_val = int(n * config.VAL_SIZE)
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test


def _pos_weights(Y):
    """pos_weight for BCE = (#neg / #pos) per class, clipped for stability."""
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return np.clip(w, 1.0, 50.0)


def run(max_contracts=None, run_neural=True, save=True):
    max_contracts = max_contracts or config.MAX_CONTRACTS

    # 1. Labels -------------------------------------------------------------
    print("[1/6] Building weak labels from tool consensus ...")
    label_df = labeling.build_label_table()
    # keep only contracts at least one tool could analyse
    label_df = label_df[label_df["n_tools_any"] > 0]
    print(f"      {len(label_df)} contracts have at least one tool result")

    # 2. Attach source, sample ---------------------------------------------
    print("[2/6] Loading contract source and sampling ...")
    addresses = list(label_df.index)
    kept, sources = features.load_corpus(addresses)
    if not kept:
        raise RuntimeError(
            "No contract source found. Did you run data_download.download() "
            "with download_contracts=True? Expected .sol files in "
            f"{config.CONTRACTS_DIR}")
    if max_contracts and len(kept) > max_contracts:
        rng = np.random.default_rng(config.RANDOM_SEED)
        sel = rng.choice(len(kept), max_contracts, replace=False)
        kept = [kept[i] for i in sel]
        sources = [sources[i] for i in sel]
    Y = label_df.loc[kept, config.DASP_CATEGORIES].values.astype(int)
    print(f"      using {len(kept)} contracts with readable source")

    # 3. Split --------------------------------------------------------------
    train_i, val_i, test_i = _split_indices(len(kept), config.RANDOM_SEED)
    src_train = [sources[i] for i in train_i]
    src_test = [sources[i] for i in test_i]
    Y_train, Y_test = Y[train_i], Y[test_i]

    # 4. Features -----------------------------------------------------------
    print("[3/6] Extracting features ...")
    X_tr, tfidf = features.build_tfidf(src_train)
    X_te = tfidf.transform(src_test)

    vocab = features.build_vocab(src_train, config.VOCAB_SIZE)
    seq_tr = features.encode_sequences(src_train, vocab, config.SEQ_MAX_LEN)
    seq_te = features.encode_sequences(src_test, vocab, config.SEQ_MAX_LEN)

    # 5. Train --------------------------------------------------------------
    print("[4/6] Training models ...")
    probs = {}

    print("  - Random Forest")
    rf = models.train_random_forest(X_tr, Y_train)
    probs["RandomForest"] = models.predict_proba_sklearn(rf, X_te)

    print("  - Linear SVM")
    svm = models.train_svm(X_tr, Y_train)
    probs["SVM"] = models.predict_proba_sklearn(svm, X_te)

    if run_neural:
        pw = _pos_weights(Y_train)

        print("  - CNN")
        cnn = models.TextCNN(len(vocab), Y.shape[1])
        cnn.fit(seq_tr, Y_train, class_pos_weight=pw)
        probs["CNN"] = cnn.predict_proba(seq_te)

        print("  - GNN")
        g_tr = [features.build_cooccurrence_graph(s, vocab) for s in src_train]
        g_te = [features.build_cooccurrence_graph(s, vocab) for s in src_test]
        gnn = models.SimpleGNN(len(vocab), Y.shape[1])
        gnn.fit(g_tr, Y_train, class_pos_weight=pw)
        probs["GNN"] = gnn.predict_proba(g_te)

    print("  - Ensemble (soft vote)")
    probs["Ensemble"] = models.ensemble_proba(list(probs.values()))

    # 6. Evaluate -----------------------------------------------------------
    print("[5/6] Evaluating ...")
    results = {}
    preds = {}
    for name, p in probs.items():
        yp = evaluate.binarize(p)
        preds[name] = yp
        avg = evaluate.averaged_f1(Y_test, yp)
        mean, ci = evaluate.bootstrap_micro_f1_ci(Y_test, yp, n_boot=500)
        results[name] = {**avg, "micro_f1_ci95": ci,
                         "per_category": evaluate.per_category_report(Y_test, yp)}
        print(f"      {name:13s} micro-F1={avg['micro_f1']:.4f} "
              f"macro-F1={avg['macro_f1']:.4f} CI95={ci}")

    # McNemar: each model vs the ensemble
    print("[6/6] McNemar significance vs Ensemble ...")
    mcnemar = {}
    for name in probs:
        if name == "Ensemble":
            continue
        stat, p, b, c = evaluate.mcnemar_test(Y_test, preds["Ensemble"], preds[name])
        mcnemar[name] = {"statistic": stat, "p_value": p, "b": b, "c": c}
        print(f"      Ensemble vs {name:13s} p={p}")

    out = {"n_train": len(src_train), "n_test": len(src_test),
           "consensus_k": config.CONSENSUS_K,
           "results": results, "mcnemar_vs_ensemble": mcnemar}

    if save:
        path = config.ARTIFACTS_DIR / "metrics.json"
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print("Saved metrics to", path)
    return out


if __name__ == "__main__":
    run()
