"""
The five models compared in the study.

  1. Random Forest      (sklearn, TF-IDF input)          - strong classical baseline
  2. Linear SVM         (sklearn, TF-IDF input)          - classical baseline
  3. CNN                (PyTorch, token-sequence input)  - learns local patterns
  4. GNN                (PyTorch, co-occurrence graph)   - relational view
  5. Ensemble           (soft-voting over 1-4)           - combines the above

All classifiers are MULTI-LABEL: a contract may carry several vulnerability
categories at once, so every model outputs one probability per category and we
threshold each independently.

Design choice: the neural models are written in plain PyTorch (no
torch-geometric requirement for the GNN — we implement a minimal graph-conv by
hand) so the whole thing installs and runs on a stock Colab runtime. If you
have torch-geometric, docs/METHODOLOGY.md shows how to swap in a real GCNConv.
"""

from __future__ import annotations
import numpy as np

from . import config

# --------------------------------------------------------------------------- #
# Classical models
# --------------------------------------------------------------------------- #
def train_random_forest(X_train, Y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    base = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1,
        class_weight="balanced", random_state=config.RANDOM_SEED,
    )
    clf = MultiOutputClassifier(base)
    clf.fit(X_train, Y_train)
    return clf


def train_svm(X_train, Y_train):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.multioutput import MultiOutputClassifier

    # LinearSVC has no predict_proba; wrap it so the ensemble can average
    # calibrated probabilities rather than hard 0/1 decisions.
    base = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=config.RANDOM_SEED),
        cv=3,
    )
    clf = MultiOutputClassifier(base)
    clf.fit(X_train, Y_train)
    return clf


def predict_proba_sklearn(clf, X) -> np.ndarray:
    """MultiOutputClassifier.predict_proba returns a list; stack to (N, C)."""
    probs = clf.predict_proba(X)
    # each element is (N, 2); take P(class = 1)
    return np.column_stack([p[:, 1] for p in probs])


# --------------------------------------------------------------------------- #
# Neural models (PyTorch)
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


class TextCNN:
    """1-D convolutional network over token-embedding sequences."""

    def __init__(self, vocab_size, n_classes):
        torch = _torch()
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, V, C, E=config.EMBED_DIM):
                super().__init__()
                self.emb = nn.Embedding(V, E, padding_idx=0)
                self.convs = nn.ModuleList(
                    [nn.Conv1d(E, 100, k, padding=k // 2) for k in (3, 4, 5)]
                )
                self.drop = nn.Dropout(0.5)
                self.fc = nn.Linear(300, C)

            def forward(self, x):
                e = self.emb(x).transpose(1, 2)                 # (B, E, L)
                pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
                h = self.drop(torch.cat(pooled, dim=1))         # (B, 300)
                return self.fc(h)                               # logits (B, C)

        self.torch = torch
        self.net = _Net(vocab_size, n_classes)

    def fit(self, X_seq, Y, class_pos_weight=None):
        torch = self.torch
        import torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=config.LEARNING_RATE)
        pw = (torch.tensor(class_pos_weight, dtype=torch.float32, device=dev)
              if class_pos_weight is not None else None)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        Xt = torch.tensor(X_seq, dtype=torch.long)
        Yt = torch.tensor(Y, dtype=torch.float32)
        n = len(Xt)
        for epoch in range(config.EPOCHS):
            self.net.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, config.BATCH_SIZE):
                idx = perm[i:i + config.BATCH_SIZE]
                xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            print(f"    CNN epoch {epoch+1}/{config.EPOCHS} loss={total/n:.4f}")
        return self

    def predict_proba(self, X_seq) -> np.ndarray:
        torch = self.torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        Xt = torch.tensor(X_seq, dtype=torch.long)
        with torch.no_grad():
            for i in range(0, len(Xt), config.BATCH_SIZE):
                xb = Xt[i:i + config.BATCH_SIZE].to(dev)
                out.append(torch.sigmoid(self.net(xb)).cpu().numpy())
        return np.vstack(out)


class SimpleGNN:
    """
    Minimal graph classifier. Each contract is a graph of tokens; we embed
    nodes, do two rounds of mean neighbour-aggregation (a hand-rolled graph
    convolution), mean-pool to a graph vector, and classify. No external graph
    library required.
    """

    def __init__(self, vocab_size, n_classes):
        torch = _torch()
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, V, C, E=config.EMBED_DIM):
                super().__init__()
                self.emb = nn.Embedding(V, E, padding_idx=0)
                self.lin1 = nn.Linear(E, E)
                self.lin2 = nn.Linear(E, E)
                self.drop = nn.Dropout(0.5)
                self.fc = nn.Linear(E, C)

            def _aggregate(self, h, edge_index, n_nodes):
                # mean of neighbours; falls back to self when isolated
                if edge_index.shape[1] == 0:
                    return h
                src, dst = edge_index[0], edge_index[1]
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, h[src])
                deg = torch.zeros(n_nodes, device=h.device)
                deg.index_add_(0, dst, torch.ones_like(src, dtype=torch.float))
                deg = deg.clamp(min=1).unsqueeze(1)
                return agg / deg

            def forward(self, nodes, edge_index):
                h = self.emb(nodes)
                h = torch.relu(self.lin1(self._aggregate(h, edge_index, len(nodes))))
                h = torch.relu(self.lin2(self._aggregate(h, edge_index, len(nodes))))
                g = self.drop(h.mean(dim=0, keepdim=True))       # graph vector
                return self.fc(g)                                # (1, C)

        self.torch = torch
        self.net = _Net(vocab_size, n_classes)

    def fit(self, graphs, Y, class_pos_weight=None):
        torch = self.torch
        import torch.nn as nn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=config.LEARNING_RATE)
        pw = (torch.tensor(class_pos_weight, dtype=torch.float32, device=dev)
              if class_pos_weight is not None else None)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        Yt = torch.tensor(Y, dtype=torch.float32, device=dev)

        for epoch in range(config.EPOCHS):
            self.net.train()
            perm = np.random.permutation(len(graphs))
            total = 0.0
            for j in perm:
                nodes, edge_index = graphs[j]
                nodes = torch.tensor(nodes, dtype=torch.long, device=dev)
                ei = torch.tensor(edge_index, dtype=torch.long, device=dev)
                opt.zero_grad()
                logit = self.net(nodes, ei)
                loss = loss_fn(logit, Yt[j:j+1])
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"    GNN epoch {epoch+1}/{config.EPOCHS} loss={total/len(graphs):.4f}")
        return self

    def predict_proba(self, graphs) -> np.ndarray:
        torch = self.torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for nodes, edge_index in graphs:
                nodes = torch.tensor(nodes, dtype=torch.long, device=dev)
                ei = torch.tensor(edge_index, dtype=torch.long, device=dev)
                out.append(torch.sigmoid(self.net(nodes, ei)).cpu().numpy())
        return np.vstack(out)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
def ensemble_proba(prob_list: list[np.ndarray],
                   weights: list[float] | None = None) -> np.ndarray:
    """Soft-voting: (optionally weighted) average of per-model probabilities."""
    probs = np.stack(prob_list, axis=0)               # (M, N, C)
    if weights is None:
        return probs.mean(axis=0)
    w = np.array(weights).reshape(-1, 1, 1)
    return (probs * w).sum(axis=0) / w.sum()
