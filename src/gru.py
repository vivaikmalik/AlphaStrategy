"""
src/gru.py - Step 4: GRU autoencoder on 11 month-to-month changes of ranked features (trained <= 2018-11, then frozen).
"""
import copy
import numpy as np
import polars as pl
import torch
import torch.nn as nn


class _GruAE(nn.Module):
    """1-layer GRU encoder (hidden 64) -> linear to d; GRU decoder reconstructs the sequence from z."""
    def __init__(self, n_feat, d, hidden, seq):
        super().__init__()
        self.seq = seq
        self.enc, self.to_z = nn.GRU(n_feat, hidden, batch_first=True), nn.Linear(hidden, d)
        self.dec, self.out = nn.GRU(d, hidden, batch_first=True), nn.Linear(hidden, n_feat)

    def encode(self, x):
        return self.to_z(self.enc(x)[1][0])

    def forward(self, x):
        z = self.encode(x)
        return self.out(self.dec(z[:, None, :].expand(-1, self.seq, -1))[0]), z


def _gru_windows(df, factors, seq):
    """Dense (permno, month, feat) changes D and validity V; row indices (p, m) and valid-change counts per row."""
    mi = (df["eom"].dt.year() * 12 + df["eom"].dt.month() - 1).to_numpy().astype(np.int64)
    m = mi - (mi.min() - seq)  # pad so every window exists
    p_ids, p = np.unique(df["permno"].to_numpy(), return_inverse=True)
    X = np.full((len(p_ids), m.max() + 1, len(factors)), np.nan, np.float32)
    X[p, m] = df.select(factors).to_numpy().astype(np.float32)
    pres = ~np.isnan(X[:, :, 0])
    V = np.zeros(pres.shape, bool)
    V[:, 1:] = pres[:, 1:] & pres[:, :-1]  # change valid only between calendar-consecutive present months
    D = np.zeros_like(X)
    D[:, 1:] = np.where(V[:, 1:, None], X[:, 1:] - X[:, :-1], 0.0)
    nvalid = V[p[:, None], m[:, None] + np.arange(-seq + 1, 1)].sum(1)
    return D, V, p, m, nvalid


def _gru_batch(D, V, p, m, seq, dev):
    ix = m[:, None] + np.arange(-seq + 1, 1)
    return (torch.from_numpy(D[p[:, None], ix]).to(dev),
            torch.from_numpy(V[p[:, None], ix]).to(dev).float().unsqueeze(-1))


def _gru_loss(model, x, mask):
    xh, _ = model(x)
    return (mask * (xh - x) ** 2).sum() / (mask.sum() * x.shape[-1]).clamp(min=1)  # masked MSE


def gru_embeddings(df, factors, d, cfg):
    """Train on eom <= 2018-11, early-stop on 2019-2020 loss, pick weight decay, embed all (permno, eom)."""
    seq, dev, bs = cfg["gru_seq_len"], cfg["device"], cfg["gru_batch"]
    df = df.sort(["permno", "eom"])
    D, V, p, m, nvalid = _gru_windows(df, factors, seq)
    ok = nvalid >= cfg["gru_min_valid"]
    eom = df["eom"]
    tr_end = pl.Series([cfg["gru_train_end_eom"]]).str.to_date()[0]
    v0, v1 = [pl.Series([s]).str.to_date()[0] for s in cfg["gru_val_eom"]]
    tr = np.where(ok & (eom <= tr_end).to_numpy())[0]
    va = np.where(ok & ((eom >= v0) & (eom <= v1)).to_numpy())[0]
    if len(va) == 0:
        va = tr
    wd_losses, best = {}, None
    for wd in cfg["gru_wd_grid"]:
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        model = _GruAE(len(factors), d, cfg["gru_hidden"], seq).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["gru_lr"], weight_decay=wd)
        best_l, best_sd, bad = np.inf, None, 0
        for _ in range(cfg["gru_max_epochs"]):
            model.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                x, mk = _gru_batch(D, V, p[b], m[b], seq, dev)
                opt.zero_grad()
                _gru_loss(model, x, mk).backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gru_clip"])
                opt.step()
            model.eval()
            with torch.no_grad():  # val loss: mean masked MSE over batches, weighted by batch size
                tot = sum(_gru_loss(model, *_gru_batch(D, V, p[va[i:i + bs]], m[va[i:i + bs]], seq, dev)).item()
                          * len(va[i:i + bs]) for i in range(0, len(va), bs)) / len(va)
            if tot < best_l - 1e-9:
                best_l, best_sd, bad = tot, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg["gru_patience"]:
                    break
        wd_losses[wd] = best_l
        if best is None or best_l < best[0]:
            best = (best_l, wd, best_sd)
    model = _GruAE(len(factors), d, cfg["gru_hidden"], seq).to(dev)
    model.load_state_dict(best[2])
    model.eval()
    emb = np.full((len(df), d), np.nan, np.float32)
    idx = np.where(ok)[0]
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            emb[b] = model.encode(_gru_batch(D, V, p[b], m[b], seq, dev)[0]).cpu().numpy()
    out = df.select("permno", "eom").with_columns([pl.Series(f"gru_{j + 1}", emb[:, j]) for j in range(d)])
    return out, {"d": d, "best_wd": best[1], "val_loss": best[0], "wd_losses": wd_losses}
