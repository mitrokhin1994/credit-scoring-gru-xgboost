"""GRU over the client's event sequence -> one feature for the scorecard (stacking).

* multi-task: main head NPL30d90 + auxiliary FPD30/FPD15/FPD10 heads (weight 0.3) regularise the encoder;
* out-of-fold by client: the score of a train/valid application comes from a model that never saw
  this client, OOT applications get the mean logit of all fold models;
* inference without torch: ``GruNumpyScorer`` reproduces the network in numpy for deployment.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from scipy.special import erf
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .sequences import SequenceStore

SEGMENT_INDEX = {"L2": 0, "L3": 1, "L4+": 2}


@dataclass
class GRUConfig:
    max_len: int = 128
    d: int = 64
    hidden: int = 64
    dropout: float = 0.2
    lr: float = 2e-3
    wd: float = 1e-4
    batch: int = 512
    epochs: int = 10
    patience: int = 2
    aux_weight: float = 0.3
    seed: int = 42
    threads: int = 8


class SeqModel(nn.Module):
    """token -> Linear+GELU -> GRU -> [h_last, segment embedding, has_history] -> MLP -> task logits."""

    def __init__(self, n_feat: int, n_tasks: int, cfg: GRUConfig, n_segments: int = 3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat + 1, cfg.d), nn.GELU())   # +1: token age relative to the application
        self.rnn = nn.GRU(cfg.d, cfg.hidden, batch_first=True)
        self.seg = nn.Embedding(n_segments, 4)
        self.head = nn.Sequential(nn.Linear(cfg.hidden + 4 + 1, 64), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(64, n_tasks))

    def forward(self, xs, lengths, seg):
        has = (lengths > 0).float()
        packed = pack_padded_sequence(self.inp(xs), lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[-1] * has[:, None]
        return self.head(torch.cat([h, self.seg(seg), has[:, None]], -1)), h


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def make_batch(ix: np.ndarray, store: SequenceStore, max_len: int):
    """Last ``max_len`` tokens before every application, left-aligned, plus token age (signed log days / 3)."""
    s = np.maximum(store.seq_start[ix], store.seq_end[ix] - max_len)
    L = store.seq_end[ix] - s
    T = max(int(L.max()), 1)
    pos = s[:, None] + np.arange(T)[None, :]
    m = np.arange(T)[None, :] < L[:, None]
    pos = np.where(m, pos, 0)
    d = (store.app_t[ix][:, None] - store.t[pos]) / 86400.0
    age = np.sign(d) * np.log1p(np.abs(d)) / 3.0
    xb = np.concatenate([store.x[pos] * m[..., None], (age * m)[..., None]], -1).astype(np.float32)
    return torch.from_numpy(xb), torch.from_numpy(L)


def _batches(ix, lengths, bs, rng, shuffle):
    """Length-bucketed batches: less padding, ~2x faster on CPU."""
    if not shuffle:
        ix = ix[np.argsort(lengths[ix], kind="stable")]
        return [ix[i:i + bs] for i in range(0, len(ix), bs)]
    ix = rng.permutation(ix)
    chunks = [ix[i:i + bs * 50] for i in range(0, len(ix), bs * 50)]
    ix = np.concatenate([c[np.argsort(lengths[c], kind="stable")] for c in chunks])
    out = [ix[i:i + bs] for i in range(0, len(ix), bs)]
    return [out[i] for i in rng.permutation(len(out))]


@torch.no_grad()
def predict(model: SeqModel, store: SequenceStore, seg: np.ndarray, ix: np.ndarray, cfg: GRUConfig,
            return_embedding: bool = False):
    model.eval()
    lengths = np.minimum(store.lengths, cfg.max_len)
    logit = np.zeros(len(ix), np.float32)
    emb = np.zeros((len(ix), cfg.hidden), np.float32) if return_embedding else None
    pos = pd.Series(np.arange(len(ix)), index=ix)
    for b in _batches(np.asarray(ix), lengths, 4096, None, False):
        xb, L = make_batch(b, store, cfg.max_len)
        out, h = model(xb, L, torch.from_numpy(seg[b]))
        k = pos.loc[b].to_numpy()
        logit[k] = out[:, 0].numpy()
        if return_embedding:
            emb[k] = h.numpy()
    return (logit, emb) if return_embedding else logit


def train_gru(store: SequenceStore, seg: np.ndarray, Y: np.ndarray, tr_ix: np.ndarray, va_ix: np.ndarray,
              cfg: GRUConfig, log=print) -> tuple[SeqModel, list[dict]]:
    """Train one model with early stopping by Gini of the main target on ``va_ix``."""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    lengths = np.minimum(store.lengths, cfg.max_len)
    model = SeqModel(store.x.shape[1], Y.shape[1], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = cfg.epochs * math.ceil(len(tr_ix) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    w = torch.tensor([1.0] + [cfg.aux_weight] * (Y.shape[1] - 1))
    Yt = torch.tensor(Y, dtype=torch.float32)
    segt = torch.tensor(seg, dtype=torch.int64)   # copy: arrays from pandas are read-only under copy-on-write
    best, bad, best_state, history = -np.inf, 0, None, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in _batches(tr_ix, lengths, cfg.batch, rng, True):
            xb, L = make_batch(b, store, cfg.max_len)
            out, _ = model(xb, L, segt[b])
            y = Yt[b]
            mask = ~torch.isnan(y)
            loss = (bce(out, torch.nan_to_num(y)) * mask * w).sum() / (mask * w).sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        g = gini(Y[va_ix, 0], predict(model, store, seg, va_ix, cfg))
        history.append({"epoch": ep + 1, "train_loss": tot / len(tr_ix), "valid_gini": g, "sec": time.time() - t0})
        log(f"  epoch {ep + 1}: loss={tot / len(tr_ix):.4f} valid_gini={g:.4f} ({time.time() - t0:.0f}s)")
        if g > best:
            best, bad, best_state = g, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, history


def client_folds(clients: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row; all applications of a client share the fold."""
    uniq = np.unique(clients)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = pd.Series(perm % n_folds, index=uniq)
    return fold_of.reindex(clients).to_numpy()


@dataclass
class OOFResult:
    score: np.ndarray           # OOF logit for pool rows, mean of fold logits for the rest
    fold: np.ndarray            # fold id (-1 outside the pool)
    models: list
    histories: list
    config: dict


def fit_oof(store: SequenceStore, seg: np.ndarray, Y: np.ndarray, clients: np.ndarray, pool: np.ndarray,
            cfg: GRUConfig, n_folds: int = 4, inner_valid: float = 0.15, log=print) -> OOFResult:
    """K-fold by client on the pool (train+valid). Early stopping uses an inner split of the training clients."""
    fold = np.full(len(clients), -1)
    fold[pool] = client_folds(clients[pool], n_folds, cfg.seed)
    all_ix = np.arange(len(clients))
    score, outside, models, histories = np.zeros(len(clients), np.float32), [], [], []
    for f in range(n_folds):
        log(f"fold {f + 1}/{n_folds}")
        train_rows = np.flatnonzero(pool & (fold != f))
        inner = client_folds(clients[train_rows], round(1 / inner_valid), cfg.seed + 1 + f) == 0
        model, hist = train_gru(store, seg, Y, train_rows[~inner], train_rows[inner], cfg, log=log)
        logit = predict(model, store, seg, all_ix, cfg)
        held = fold == f
        score[held] = logit[held]
        outside.append(logit[~pool])
        models.append(model)
        histories.append(hist)
    score[~pool] = np.mean(outside, axis=0)
    return OOFResult(score=score, fold=fold, models=models, histories=histories, config=asdict(cfg))


# ---------------------------------------------------------------- deployment
def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GruNumpyScorer:
    """The fold models re-implemented in numpy (float64): no torch dependency in production."""

    def __init__(self, models: list[SeqModel], max_len: int):
        self.folds = [{k: v.detach().double().numpy() for k, v in m.state_dict().items()} for m in models]
        self.max_len = max_len

    @staticmethod
    def _fold_logit(w, x: np.ndarray, seg: int) -> float:
        H = w["rnn.weight_hh_l0"].shape[1]
        h, has = np.zeros(H), 0.0
        if len(x):
            z = _gelu(x @ w["inp.0.weight"].T + w["inp.0.bias"])
            gi = z @ w["rnn.weight_ih_l0"].T + w["rnn.bias_ih_l0"]
            for t in range(len(x)):          # torch.nn.GRU equations, gates r, z, n
                gh = w["rnn.weight_hh_l0"] @ h + w["rnn.bias_hh_l0"]
                r = _sigmoid(gi[t, :H] + gh[:H])
                u = _sigmoid(gi[t, H:2 * H] + gh[H:2 * H])
                n = np.tanh(gi[t, 2 * H:] + r * gh[2 * H:])
                h = (1 - u) * n + u * h
            has = 1.0
        feat = np.concatenate([h * has, w["seg.weight"][seg], [has]])
        hid = _gelu(feat @ w["head.0.weight"].T + w["head.0.bias"])
        return float((hid @ w["head.3.weight"].T + w["head.3.bias"])[0])

    def logit(self, x: np.ndarray, seg: int) -> float:
        """x: (T, n_fields + 1) standardized token fields + token age, chronological."""
        x = np.asarray(x, dtype=np.float64)[-self.max_len:]
        return float(np.mean([self._fold_logit(w, x, seg) for w in self.folds]))
