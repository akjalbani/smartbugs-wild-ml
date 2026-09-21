"""Staged, memory-safe execution of src/pipeline.run() on ALL contracts.
Uses the repository's own functions and config unchanged; only saves
intermediate results to disk between stages. Equivalence notes:
 - Split, features, models, threshold, metrics: identical calls to src.*.
 - Random Forest: MultiOutputClassifier fits one independent clone of the
   same RandomForestClassifier (same random_state) per category; here each
   clone is fitted/predicted one at a time and then freed (same result, less RAM).
"""
import sys, json, time, pickle, numpy as np, scipy.sparse as sp
from src import config, labeling, features, models, evaluate, pipeline
S = "stage/"; stage = sys.argv[1]; t0 = time.time()
def log(k, v):
    try: d = json.load(open(S+"timing.json"))
    except Exception: d = {}
    d[k] = v; json.dump(d, open(S+"timing.json","w"), indent=1)

if stage == "prep":
    df = labeling.build_label_table(); df = df[df["n_tools_any"] > 0]
    kept, sources = features.load_corpus(list(df.index))
    Y = df.loc[kept, config.DASP_CATEGORIES].values.astype(int)
    tr, va, te = pipeline._split_indices(len(kept), config.RANDOM_SEED)
    pickle.dump({"kept":kept,"tr":tr,"va":va,"te":te,"Y":Y}, open(S+"split.pkl","wb"))
    src_tr = [sources[i] for i in tr]; src_te = [sources[i] for i in te]
    t=time.time(); Xtr, tf = features.build_tfidf(src_tr); Xte = tf.transform(src_te); log("tfidf_s", time.time()-t)
    sp.save_npz(S+"Xtr.npz", Xtr); sp.save_npz(S+"Xte.npz", Xte)
    t=time.time(); vocab = features.build_vocab(src_tr, config.VOCAB_SIZE)
    np.save(S+"seq_tr.npy", features.encode_sequences(src_tr, vocab, config.SEQ_MAX_LEN))
    np.save(S+"seq_te.npy", features.encode_sequences(src_te, vocab, config.SEQ_MAX_LEN)); log("seq_s", time.time()-t)
    json.dump(vocab, open(S+"vocab.json","w"))
    t=time.time()
    for nm, srcs in (("tr",src_tr),("te",src_te)):
        g = [features.build_cooccurrence_graph(s, vocab) for s in srcs]
        g = [(n.astype(np.int32), e.astype(np.int32)) for n,e in g]
        pickle.dump(g, open(S+f"g_{nm}.pkl","wb"), protocol=4); del g
    log("graph_s", time.time()-t)
    print("prep done", len(kept), len(tr), len(va), len(te), Xtr.shape)

d = pickle.load(open(S+"split.pkl","rb")) if stage != "prep" else None
if stage == "rf":
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    Xtr = sp.load_npz(S+"Xtr.npz"); Xte = sp.load_npz(S+"Xte.npz"); Ytr = d["Y"][d["tr"]]
    base = models.RandomForestClassifier if hasattr(models,"RandomForestClassifier") else RandomForestClassifier
    est = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=config.RANDOM_SEED)
    P = []; ttr = tin = 0
    for k in range(Ytr.shape[1]):
        m = clone(est); t=time.time(); m.fit(Xtr, Ytr[:,k]); ttr += time.time()-t
        t=time.time(); P.append(m.predict_proba(Xte)[:,1]); tin += time.time()-t
        del m; print("rf cat", k, flush=True)
    np.save(S+"p_RandomForest.npy", np.column_stack(P)); log("train_RandomForest", ttr); log("infer_RandomForest", tin)
if stage == "svm":
    Xtr = sp.load_npz(S+"Xtr.npz"); Xte = sp.load_npz(S+"Xte.npz")
    t=time.time(); m = models.train_svm(Xtr, d["Y"][d["tr"]]); log("train_SVM", time.time()-t)
    t=time.time(); np.save(S+"p_SVM.npy", models.predict_proba_sklearn(m, Xte)); log("infer_SVM", time.time()-t)
if stage in ("cnn","gnn"):
    Ytr = d["Y"][d["tr"]]; pw = pipeline._pos_weights(Ytr); V = len(json.load(open(S+"vocab.json")))
    if stage == "cnn":
        m = models.TextCNN(V, Ytr.shape[1]); t=time.time(); m.fit(np.load(S+"seq_tr.npy"), Ytr, class_pos_weight=pw); log("train_CNN", time.time()-t)
        t=time.time(); np.save(S+"p_CNN.npy", m.predict_proba(np.load(S+"seq_te.npy"))); log("infer_CNN", time.time()-t)
    else:
        gtr = pickle.load(open(S+"g_tr.pkl","rb"))
        m = models.SimpleGNN(V, Ytr.shape[1]); t=time.time(); m.fit(gtr, Ytr, class_pos_weight=pw); log("train_GNN", time.time()-t)
        del gtr; gte = pickle.load(open(S+"g_te.pkl","rb"))
        t=time.time(); np.save(S+"p_GNN.npy", m.predict_proba(gte)); log("infer_GNN", time.time()-t)
if stage == "eval":
    Yte = d["Y"][d["te"]]; names = ["RandomForest","SVM","CNN","GNN"]
    probs = {n: np.load(S+f"p_{n}.npy") for n in names}
    probs["Ensemble"] = models.ensemble_proba([probs[n] for n in names])
    res, preds = {}, {}
    for n,p in probs.items():
        yp = evaluate.binarize(p); preds[n] = yp; avg = evaluate.averaged_f1(Yte, yp)
        mean, ci = evaluate.bootstrap_micro_f1_ci(Yte, yp, n_boot=500)
        res[n] = {**avg, "micro_f1_ci95": ci, "per_category": evaluate.per_category_report(Yte, yp)}
    mc = {}
    for a in probs:
        for b in probs:
            if a < b:
                st,p,x,y = evaluate.mcnemar_test(Yte, preds[a], preds[b]); mc[f"{a}|{b}"] = {"statistic":st,"p_value":p,"b":x,"c":y}
    out = {"n_total":len(d["kept"]),"n_train":len(d["tr"]),"n_val":len(d["va"]),"n_test":len(d["te"]),
           "consensus_k":config.CONSENSUS_K,"results":res,"mcnemar_pairwise":mc,
           "timing_s":json.load(open(S+"timing.json"))}
    json.dump(out, open("artifacts/metrics_full.json","w"), indent=2)
    np.savez("artifacts/preds_full.npz", Y_test=Yte, **{n:probs[n] for n in probs})
    for n,r in res.items(): print(n, r["micro_f1"], r["macro_f1"], r["micro_f1_ci95"])
print("stage", stage, "done in", round(time.time()-t0,1), "s", flush=True)
