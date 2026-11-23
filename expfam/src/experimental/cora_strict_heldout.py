"""Cora strict held-out X-family comparison (CORA-SH1) -- library.

Design: ``reports/experiment_design/additional_experiment_design_20261123.md``
(commit 59eac14), protocol draft section 9.  NOT a frozen protocol.

What this module does (and nothing else):

* parses Cora exactly like the historical runners (``parse_cora`` there);
* draws a dyad split over ALL unordered pairs of the full graph
  (train / test, no other category inside the graph);
* chooses the node set WITHOUT Y labels (primary: ``full`` = all 2708 nodes;
  alternative ``random_per_class``).  ``train_degree`` (per-class top by train
  degree) is kept only as a documented non-primary diagnostic: it uses train
  labels and was found to make train and test pairs non-exchangeable inside
  the subset (train edge density ~0.0064 vs test ~0.0037 at n=700);
* restricts the masks to the node set.  Pair categories of the full graph:
  ``node-set train`` (in the likelihood), ``node-set test`` (scored only),
  ``outside node set`` (in neither; never treated as an implicit 0; empty for
  ``full``);
* fixes the 50-word vocabulary from the full-Cora column sums of X (Y-free);
* runs one MCEM fit per (arm, K, split, seed) through the unchanged
  ``em_runner.run_em_experimental`` with ``numerics_mode='consistent'`` and
  ``failure_policy='fail_fast'`` (no retry, no replacement, no seed change);
* computes the plug-in Z estimators (joint mode at theta-hat, last, mean5)
  and held-out Bernoulli scores on the test pairs.

Lineage E (``DualExpFamLSMConsistent``, scalar family, experimental
prototype).  The only runtime substitution is ``CoraConsistentFast``: the
Y-side precision term ``Z.T @ np.diag(v) @ Z`` is evaluated as
``(Z * v[:, None]).T @ Z`` (same sum, no n x n temporary).  Its equivalence
is tested; it changes no formula.  Historical code is not modified.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
_ROOT = _SRC.parent.parent
for _p in (_HERE, _SRC, _ROOT / "reproduction" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import em_runner                                                  # noqa: E402
from model_dual_expfam_consistent import DualExpFamLSMConsistent  # noqa: E402
from objective_consistent_numerics import (                       # noqa: E402
    bernoulli_log_likelihood, bernoulli_mean,
)

DATA_DIR = _ROOT / "expfam" / "data" / "cora"
RUNNER_VERSION = "cora-sh1-runner-v2-newtonF"

# ── protocol values.  v2 (2026-11-24): frozen for the main run by the
#    researcher's instruction "Newton法の採用と本実験" (BX F update = exact
#    per-column Newton; GXc as K=6 secondary; 105 fits).  The smoke run used
#    v1 (Adam F update, model_kind "fast") and stays reproducible with it. ──
CONFIG: dict[str, Any] = {
    "test_fraction": 0.2,
    "selection": "full",        # primary candidate (Y-free, no selection)
    "per_class": 100,           # only for random_per_class / train_degree
    "selection_seed_base": 733_000,
    "d_vocab": 50,
    "K_values": [6, 3],          # primary first (draft): K=6, sensitivity K=3
    "L": 5,
    "num_iter": 8,
    "newton_max_iter": 10,       # fixed inside em_runner (not a parameter)
    "newton_alpha": 0.5,         # em_runner, retry 0 (fail_fast => no retry)
    "family_y": "bernoulli",
    "numerics_mode": "consistent",
    "failure_policy": "fail_fast",
    "mode_grad_tol": 1e-8,       # same as E1 / #168 / #128
    "mode_max_iter": 200,
    # primary predictive (draft): MC average of p over the final E-step samples.
    # The joint mode is opt-in only: on full Cora K=3 it did not reach
    # stationarity in 200 Newton iterations (2459 s, smoke split, 1-iteration
    # state) and K=6 needs a 2.1 GB nK x nK Hessian per iteration.
    "primary_estimator": "mc_predictive",
    "n_splits": 5,
    "n_seeds": 3,
    "split_seed_base": 731_000,
    "init_seed_base": 7_310_000,
    "mc_seed_base": 7_320_000,
    "smoke_split_index": 90,     # reserved; never one of the main splits 0..4
    # v2 additions
    "model_kind": "fast_newton",  # BX: exact per-column Newton F; GX: closed form
    "f_newton_max_iter": 100,
    "f_newton_tol_dec": 1e-12,    # stop when max_j g'H^{-1}g/2 < tol (per column)
    "secondary_cells": [["GXc", 6]],   # GXc only at K=6 (pre-registered secondary)
}

ARMS: dict[str, dict[str, Any]] = {
    # primary contrast
    "BX": {"family_x": "bernoulli", "fix_x": False, "x_variant": "raw",
           "role": "primary"},
    "GX": {"family_x": "gaussian", "fix_x": False, "x_variant": "raw",
           "role": "primary"},
    # secondary comparator: F fixed at 0, so the X term is identically zero.
    # family_x is irrelevant for the fit; 'bernoulli' keeps X in its support.
    "YO": {"family_x": "bernoulli", "fix_x": True, "x_variant": "raw",
           "role": "secondary"},
    # optional pre-defined operational sensitivity (needs a human decision)
    "GXc": {"family_x": "gaussian", "fix_x": False, "x_variant": "centered",
            "role": "secondary_sensitivity_optional"},
}
PRIMARY_MANIFEST_ARMS = ("BX", "GX", "YO")


# ═════════════════════════════════════════════════════════════════════════════
# hashing / provenance helpers
# ═════════════════════════════════════════════════════════════════════════════

def array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════════════
# data
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CoraData:
    node_ids: np.ndarray
    X_full: np.ndarray          # (2708, 1433) float 0/1
    labels: np.ndarray          # (2708,) int, label_set order of first appearance
    label_set: tuple
    edges: np.ndarray           # (E, 2) int, i < j, lexicographically sorted


def load_cora(data_dir: Path = DATA_DIR) -> CoraData:
    """Same parsing as ``run_fixed_real_cora_heldout_link_prediction.parse_cora``."""
    node_ids, features, labels, label_set = [], [], [], []
    with open(Path(data_dir) / "cora.content", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            node_ids.append(parts[0])
            features.append([int(x) for x in parts[1:-1]])
            lbl = parts[-1]
            if lbl not in label_set:
                label_set.append(lbl)
            labels.append(lbl)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edges = set()
    with open(Path(data_dir) / "cora.cites", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u in id_to_idx and v in id_to_idx:
                i, j = id_to_idx[u], id_to_idx[v]
                if i != j:
                    edges.add((min(i, j), max(i, j)))
    return CoraData(
        node_ids=np.array(node_ids),
        X_full=np.array(features, dtype=np.float64),
        labels=np.array([label_set.index(l) for l in labels], dtype=int),
        label_set=tuple(label_set),
        edges=np.array(sorted(edges), dtype=np.int64),
    )


def pair_index(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Row-major index of the unordered pair (i, j), i < j, among n(n-1)/2."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if np.any(i >= j):
        raise ValueError("pair_index requires i < j")
    return i * n - (i * (i + 1)) // 2 + (j - i - 1)


def make_dyad_split(n_nodes: int, split_seed: int, test_fraction: float) -> np.ndarray:
    """Boolean ``is_test`` over all n(n-1)/2 unordered pairs (exact count).

    Labels are not consulted: edges and non-edges are split uniformly together.
    """
    n_pairs = n_nodes * (n_nodes - 1) // 2
    n_test = int(round(test_fraction * n_pairs))
    rng = np.random.default_rng(split_seed)
    is_test = np.zeros(n_pairs, dtype=bool)
    is_test[rng.choice(n_pairs, size=n_test, replace=False)] = True
    return is_test


def train_degree(edges: np.ndarray, is_test: np.ndarray, n_nodes: int) -> np.ndarray:
    """Degree counted from TRAIN edges only (test edges are excluded)."""
    pidx = pair_index(edges[:, 0], edges[:, 1], n_nodes)
    keep = ~is_test[pidx]
    e = edges[keep]
    return (np.bincount(e[:, 0], minlength=n_nodes)
            + np.bincount(e[:, 1], minlength=n_nodes))


def select_nodes(labels: np.ndarray, degree: np.ndarray, per_class: int) -> np.ndarray:
    """Per class (label_set order) the top-``per_class`` nodes by ``degree``;
    ties broken by ascending node index.  Returns full-graph node indices."""
    selected = []
    for c in range(int(labels.max()) + 1):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) < per_class:
            raise ValueError(f"class {c} has only {len(idx_c)} nodes")
        order = np.lexsort((idx_c, -degree[idx_c]))   # last key is primary
        selected.extend(idx_c[order[:per_class]].tolist())
    return np.array(selected, dtype=np.int64)


def select_vocabulary(X_full: np.ndarray, d_vocab: int) -> np.ndarray:
    """Top-``d_vocab`` columns by full-Cora column sum (Y-free); ties by index."""
    colsum = X_full.sum(axis=0)
    order = np.lexsort((np.arange(X_full.shape[1]), -colsum))
    return np.sort(order[:d_vocab])


@dataclass(frozen=True)
class SplitData:
    split_index: int
    split_seed: int
    selected: np.ndarray        # full-graph node indices (class-concatenated)
    vocab: np.ndarray
    X_raw: np.ndarray           # (n, d) 0/1
    X_centered: np.ndarray      # (n, d) column-centred (operational arm only)
    Y_true: np.ndarray          # (n, n) symmetric 0/1, zero diagonal
    Y_fit: np.ndarray           # Y_true with every test entry set to 0 (canary 0)
    train_mask: np.ndarray      # (n, n) bool
    test_mask: np.ndarray       # (n, n) bool
    summary: dict


SELECTIONS = ("full", "random_per_class", "train_degree")


def choose_nodes(data: CoraData, selection: str, is_test: np.ndarray,
                 split_index: int, cfg: dict[str, Any]) -> np.ndarray:
    """Node set for one split.

    full             : every node (no selection; primary candidate).
    random_per_class : ``per_class`` random nodes per class (Y-free).
    train_degree     : per-class top by TRAIN degree.  NOT for the primary
                       design: it uses train Y labels and makes train and test
                       pairs non-exchangeable inside the subset (train edges
                       are over-represented because they drove the selection).
    """
    n_full = len(data.node_ids)
    if selection == "full":
        return np.arange(n_full, dtype=np.int64)
    if selection == "random_per_class":
        rng = np.random.default_rng(cfg["selection_seed_base"] + split_index)
        return np.concatenate([
            np.sort(rng.choice(np.where(data.labels == c)[0], cfg["per_class"],
                               replace=False))
            for c in range(len(data.label_set))]).astype(np.int64)
    if selection == "train_degree":
        return select_nodes(data.labels, train_degree(data.edges, is_test, n_full),
                            cfg["per_class"])
    raise ValueError(f"unknown selection {selection!r}")


def build_split(data: CoraData, split_index: int, cfg: dict[str, Any] = CONFIG,
                *, split_seed: int | None = None,
                selection: str | None = None) -> SplitData:
    n_full = len(data.node_ids)
    seed = cfg["split_seed_base"] + split_index if split_seed is None else split_seed
    selection = str(cfg["selection"] if selection is None else selection)
    is_test = make_dyad_split(n_full, seed, cfg["test_fraction"])
    deg_tr = train_degree(data.edges, is_test, n_full)
    sel = choose_nodes(data, selection, is_test, split_index, cfg)
    vocab = select_vocabulary(data.X_full, cfg["d_vocab"])
    n = len(sel)

    pos = -np.ones(n_full, dtype=np.int64)
    pos[sel] = np.arange(n)
    Y = np.zeros((n, n))
    pa, pb = pos[data.edges[:, 0]], pos[data.edges[:, 1]]
    inside = (pa >= 0) & (pb >= 0)
    Y[pa[inside], pb[inside]] = 1.0
    Y[pb[inside], pa[inside]] = 1.0

    a, b = np.triu_indices(n, k=1)
    u, v = sel[a], sel[b]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    test_upper = is_test[pair_index(lo, hi, n_full)]
    test_mask = np.zeros((n, n), dtype=bool)
    test_mask[a[test_upper], b[test_upper]] = True
    test_mask |= test_mask.T
    train_mask = ~test_mask
    np.fill_diagonal(train_mask, False)
    validate_masks(train_mask, test_mask)

    Y_fit = np.where(test_mask, 0.0, Y)
    X_raw = data.X_full[np.ix_(sel, vocab)].copy()
    X_ctr = X_raw - X_raw.mean(axis=0, keepdims=True)

    up_tr, up_te = np.triu(train_mask, 1), np.triu(test_mask, 1)
    deg_sub_train = (Y * train_mask).sum(axis=1)
    summary = {
        "split_index": split_index, "split_seed": seed, "selection": selection,
        "n": n, "d": len(vocab),
        "n_pairs": n * (n - 1) // 2,
        "n_train_pairs": int(up_tr.sum()), "n_test_pairs": int(up_te.sum()),
        "n_train_pos": int(Y[up_tr].sum()), "n_test_pos": int(Y[up_te].sum()),
        "train_density": float(Y[up_tr].mean()),
        "n_isolated_in_train_subgraph": int((deg_sub_train == 0).sum()),
        "mean_train_degree_full_graph_selected": float(deg_tr[sel].mean()),
        "train_pair_density": float(Y[up_tr].mean()),
        "test_pair_density": float(Y[up_te].mean()),
        "x_col_mean_min": float(X_raw.mean(0).min()),
        "x_col_mean_median": float(np.median(X_raw.mean(0))),
        "x_col_mean_max": float(X_raw.mean(0).max()),
        "node_hash": array_hash(sel), "vocab_hash": array_hash(vocab),
        "train_mask_hash": array_hash(train_mask),
        "test_mask_hash": array_hash(test_mask),
        "y_fit_hash": array_hash(Y_fit), "x_raw_hash": array_hash(X_raw),
    }
    return SplitData(split_index, seed, sel, vocab, X_raw, X_ctr, Y, Y_fit,
                     train_mask, test_mask, summary)


def validate_masks(train_mask: np.ndarray, test_mask: np.ndarray) -> None:
    for name, m in (("train", train_mask), ("test", test_mask)):
        if m.dtype != np.bool_:
            raise ValueError(f"{name} mask must be bool")
        if not np.array_equal(m, m.T):
            raise ValueError(f"{name} mask must be symmetric")
        if np.any(np.diag(m)):
            raise ValueError(f"{name} mask diagonal must be False")
    if np.any(train_mask & test_mask):
        raise ValueError("train and test masks overlap")
    off = ~np.eye(train_mask.shape[0], dtype=bool)
    if not np.array_equal(train_mask | test_mask, off):
        raise ValueError("train | test must cover every off-diagonal subset pair")


# ═════════════════════════════════════════════════════════════════════════════
# model: counting + mathematically identical fast precision
# ═════════════════════════════════════════════════════════════════════════════

class _CountingRNG:
    """Delegates to the generator; counts successful multivariate_normal draws.

    RNG consumption is unchanged (every call goes to the same generator)."""

    def __init__(self, rng):
        self._rng = rng
        self.mvn_success = 0

    def multivariate_normal(self, *args, **kwargs):
        out = self._rng.multivariate_normal(*args, **kwargs)
        self.mvn_success += 1
        return out

    def __getattr__(self, name):
        return getattr(self._rng, name)


class CoraConsistentCounting(DualExpFamLSMConsistent):
    """``DualExpFamLSMConsistent`` + count of E-step sampling fallbacks.

    ``reproduction/src/model.py`` keeps the Newton point silently when the
    Laplace draw fails; here the number of such fallbacks is counted."""

    def calc_eta_newton(self, X, Y, rng=None, **kwargs):
        if rng is None:
            raise ValueError("an explicit rng is required (no hidden seeding)")
        proxy = _CountingRNG(rng)
        Z = super().calc_eta_newton(X, Y, rng=proxy, **kwargs)
        stats = self.__dict__.setdefault(
            "sampling_stats", {"estep_sweeps": 0, "draws": 0, "fallbacks": 0})
        stats["estep_sweeps"] += 1
        stats["draws"] += proxy.mvn_success
        stats["fallbacks"] += self.n - proxy.mvn_success
        return Z


class CoraConsistentFast(CoraConsistentCounting):
    """Same precision matrix as ``DualExpFamLSMMasked._calc_precision_matrix``;
    only ``Z.T @ np.diag(var_y) @ Z`` -> ``(Z * var_y[:, None]).T @ Z``."""

    def _calc_precision_matrix(self, Z, F, sigma, var_z, w0, w, i):
        z_i = Z[i, :]
        k = Z.shape[1]
        term1 = (1.0 / var_z) * np.eye(k)
        eta_x_i = F @ z_i
        if self.family_x == "gaussian":
            sigma_inv_diag = 1.0 / np.maximum(np.diag(sigma), 1e-8)
            term2 = F.T @ (F * sigma_inv_diag[:, None])
        else:
            var_x_i = self._variance_function_x(eta_x_i)
            term2 = F.T @ (F * var_x_i[:, None])
        eta_y = w0 + w * (Z @ z_i)
        var_y = self._variance_function(eta_y) * self._mask_f[i, :]
        term3 = (w ** 2) * ((Z * var_y[:, None]).T @ Z)
        return term1 + term2 + term3


# ── Bernoulli-X F M-step: exact maximiser of Q_X (v2) ─────────────────────
#
# Q_X(F) = (1/L) sum_l sum_i sum_j [x_ij eta - log(1+e^eta)],  eta = f_j' z_i^(l)
# (canonical Bernoulli, no bias, phi=1).  Separable over columns j and concave
# in each f_j:  grad_j = (1/L) sum (x_ij - s(eta)) z,
#               H_j    = (1/L) sum s(eta)(1-s(eta)) z z'.
# Damped Newton with Armijo backtracking per column; a column whose line
# search fails keeps its current value, so Q_X never decreases.  Same
# objective that DualExpFamLSM._calc_F_adam only ascends for 50 Adam steps.

class FNewtonError(FloatingPointError):
    """Non-finite value, singular curvature or separation in the F M-step."""


def bernoulli_qx_cols(X, Zs, F):
    L = Zs.shape[2]
    q = np.zeros(F.shape[0])
    for l in range(L):
        eta = Zs[:, :, l] @ F.T
        q += (X * eta - np.logaddexp(0.0, eta)).sum(axis=0)
    return q / L


def bernoulli_qx_grad_hess(X, Zs, F):
    L = Zs.shape[2]
    d, k = F.shape
    g = np.zeros((d, k))
    H = np.zeros((d, k, k))
    for l in range(L):
        Z = Zs[:, :, l]
        p = bernoulli_mean(Z @ F.T)
        g += (X - p).T @ Z
        H += np.einsum("ik,id,im->dkm", Z, p * (1.0 - p), Z, optimize=True)
    return g / L, H / L


def bernoulli_f_newton(X, Zs, F0, max_iter=100, tol_dec=1e-12):
    """Exact per-column maximiser of Q_X.  Returns (F, info)."""
    n = X.shape[0]
    colsum = X.sum(axis=0)
    if np.any((colsum == 0) | (colsum == n)):
        raise FNewtonError("complete separation: an X column is constant")
    if not (np.all(np.isfinite(Zs)) and np.all(np.isfinite(F0))):
        raise FNewtonError("non-finite input to the F M-step")
    F = np.array(F0, dtype=float, copy=True)
    q = bernoulli_qx_cols(X, Zs, F)
    q_start = float(q.sum())
    status, it, ls_fail = "max_iter", 0, 0
    for it in range(1, max_iter + 1):
        g, H = bernoulli_qx_grad_hess(X, Zs, F)
        try:
            delta = np.linalg.solve(H, g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError as exc:
            raise FNewtonError(f"singular Q_X Hessian: {exc}") from exc
        dec = np.einsum("dk,dk->d", g, delta)
        if not np.all(np.isfinite(dec)):
            raise FNewtonError("non-finite Newton decrement")
        if np.max(dec) / 2.0 < tol_dec:
            status = "converged"
            break
        t = np.ones(F.shape[0])
        accepted = np.zeros(F.shape[0], bool)
        qc = q
        for _ in range(40):
            qc = bernoulli_qx_cols(X, Zs, F + t[:, None] * delta)
            accepted = qc >= q + 1e-4 * t * dec
            if accepted.all():
                break
            t = np.where(accepted, t, t / 2.0)
        ls_fail += int((~accepted).sum())
        F = F + np.where(accepted[:, None], t[:, None] * delta, 0.0)
        q = np.where(accepted, qc, q)
        if not accepted.any():
            status = "line_search_stalled"
            break
    if not np.all(np.isfinite(F)):
        raise FNewtonError("non-finite F after Newton")
    g, H = bernoulli_qx_grad_hess(X, Zs, F)
    eta_max = max(float(np.max(np.abs(Zs[:, :, l] @ F.T))) for l in range(Zs.shape[2]))
    return F, {"q_before": q_start, "q_after": float(q.sum()),
               "iters": it, "status": status, "line_search_failures": ls_fail,
               "grad_max": float(np.max(np.abs(g))),
               "min_eig_H": float(np.min(np.linalg.eigvalsh(H))),
               "F_absmax": float(np.max(np.abs(F))), "eta_absmax": eta_max}


class CoraConsistentFastNewton(CoraConsistentFast):
    """``CoraConsistentFast`` whose Bernoulli-X F update is the exact
    per-column Newton maximiser of Q_X instead of Adam-50.  Gaussian X keeps the
    inherited closed form (``calc_F`` sends only non-Gaussian X to
    ``_calc_F_adam``); likelihood, bias, dispersion and the Y side unchanged."""

    def _calc_F_adam(self, X, Z_samples, **kwargs):
        if self.family_x != "bernoulli":
            raise NotImplementedError("Newton F update is for Bernoulli X only")
        F, info = bernoulli_f_newton(X, Z_samples, self.params["F"],
                                     max_iter=CONFIG["f_newton_max_iter"],
                                     tol_dec=CONFIG["f_newton_tol_dec"])
        log = self.__dict__.setdefault("f_newton_log", [])
        info["mstep"] = len(log) + 1
        log.append(info)
        return F


MODEL_CLASSES = {"fast": CoraConsistentFast, "original": CoraConsistentCounting,
                 "fast_newton": CoraConsistentFastNewton}


@contextlib.contextmanager
def use_model_class(cls):
    """Swap the class of the model ``em_runner.build_model`` returns.

    The instance is built by the unchanged builder (same constructor, same
    state); only methods of the listed subclass are substituted."""
    original = em_runner.build_model

    def patched(*args, **kwargs):
        model = original(*args, **kwargs)
        if type(model) is not DualExpFamLSMConsistent:
            raise TypeError(f"unexpected model class {type(model).__name__}")
        model.__class__ = cls
        model.sampling_stats = {"estep_sweeps": 0, "draws": 0, "fallbacks": 0}
        return model

    em_runner.build_model = patched
    try:
        yield
    finally:
        em_runner.build_model = original


# ═════════════════════════════════════════════════════════════════════════════
# seeds and manifest
# ═════════════════════════════════════════════════════════════════════════════

def seeds_for(split_index: int, K: int, seed_index: int,
              cfg: dict[str, Any] = CONFIG) -> tuple[int, int]:
    """(init_seed, mc_seed); shared by every arm of the same (split, K, seed)."""
    offset = 10_000 * split_index + 100 * K + seed_index
    return cfg["init_seed_base"] + offset, cfg["mc_seed_base"] + offset


def build_manifest(cfg: dict[str, Any] = CONFIG,
                   arms=PRIMARY_MANIFEST_ARMS,
                   secondary_cells=None) -> list[dict[str, Any]]:
    """Primary arms at every K plus the (arm, K) secondary cells (v2: GXc K=6)."""
    if secondary_cells is None:
        secondary_cells = cfg.get("secondary_cells", [])
    cells = []
    for s in range(cfg["n_splits"]):
        for K in cfg["K_values"]:
            for r in range(cfg["n_seeds"]):
                init_seed, mc_seed = seeds_for(s, K, r, cfg)
                extra = [a for a, kk in secondary_cells if kk == K]
                for arm in list(arms) + extra:
                    cells.append({"cell_id": f"s{s}-K{K}-r{r}-{arm}",
                                  "split_index": s, "K": K, "seed_index": r,
                                  "arm": arm, "init_seed": init_seed,
                                  "mc_seed": mc_seed})
    return cells


# ═════════════════════════════════════════════════════════════════════════════
# scoring
# ═════════════════════════════════════════════════════════════════════════════

def heldout_pair_arrays(split: SplitData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(np.triu(split.test_mask, 1))
    return rows, cols, split.Y_true[rows, cols]


def heldout_scores(Z: np.ndarray, w0: float, w: float, split: SplitData) -> dict:
    """Bernoulli scores on all test pairs (natural ratio, no negative sampling)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows, cols, y = heldout_pair_arrays(split)
    eta = float(w0) + float(w) * np.einsum("ij,ij->i", Z[rows], Z[cols])
    ll = bernoulli_log_likelihood(y, eta)
    p = bernoulli_mean(eta)
    pos = y == 1.0
    out = {
        "log_score": float(ll.mean()),
        "log_score_pos": float(ll[pos].mean()) if pos.any() else float("nan"),
        "log_score_neg": float(ll[~pos].mean()),
        "ap": float(average_precision_score(y, p)) if pos.any() else float("nan"),
        "auc": float(roc_auc_score(y, p)) if 0 < pos.sum() < len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_pred": float(p.mean()), "test_pos_rate": float(y.mean()),
        "n_test_pairs": int(len(y)), "n_test_pos": int(pos.sum()),
    }
    return out


def heldout_scores_mc_predictive(Z_samples: np.ndarray, w0: float, w: float,
                                 split: SplitData) -> dict:
    """Scores of the MC predictive  p_ij = (1/L) sum_l sigmoid(w0 + w z_i^l . z_j^l)
    over the L samples of the final E-step (no point estimate of Z).

    log p and log(1-p) are computed with logsumexp for stability."""
    from scipy.special import logsumexp
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows, cols, y = heldout_pair_arrays(split)
    L = Z_samples.shape[2]
    eta = np.stack([float(w0) + float(w) * np.einsum(
        "ij,ij->i", Z_samples[rows, :, l], Z_samples[cols, :, l]) for l in range(L)])
    if not np.all(np.isfinite(eta)):
        raise FloatingPointError("non-finite eta in MC predictive")
    log_p = logsumexp(-np.logaddexp(0.0, -eta), axis=0) - math.log(L)
    log_q = logsumexp(-np.logaddexp(0.0, eta), axis=0) - math.log(L)
    ll = y * log_p + (1.0 - y) * log_q
    p = np.exp(log_p)
    pos = y == 1.0
    return {
        "log_score": float(ll.mean()),
        "log_score_pos": float(ll[pos].mean()) if pos.any() else float("nan"),
        "log_score_neg": float(ll[~pos].mean()),
        "ap": float(average_precision_score(y, p)) if pos.any() else float("nan"),
        "auc": float(roc_auc_score(y, p)) if 0 < pos.sum() < len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_pred": float(p.mean()), "test_pos_rate": float(y.mean()),
        "n_test_pairs": int(len(y)), "n_test_pos": int(pos.sum()),
    }


def reliability_bins(Z, w0, w, split: SplitData, n_bins: int = 10) -> list[dict]:
    rows, cols, y = heldout_pair_arrays(split)
    eta = float(w0) + float(w) * np.einsum("ij,ij->i", Z[rows], Z[cols])
    p = bernoulli_mean(eta)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()),
             "mean_pred": float(p[idx == b].mean()) if (idx == b).any() else float("nan"),
             "obs_rate": float(y[idx == b].mean()) if (idx == b).any() else float("nan")}
            for b in range(n_bins)]


def constant_reference_scores(split: SplitData) -> dict:
    """REF: train density as a constant probability (no fit)."""
    up = np.triu(split.train_mask, 1)
    rho = float(split.Y_true[up].mean())
    rows, cols, y = heldout_pair_arrays(split)
    eta = np.full(len(y), math.log(rho / (1.0 - rho)))
    ll = bernoulli_log_likelihood(y, eta)
    return {"log_score": float(ll.mean()), "train_density": rho}


# ═════════════════════════════════════════════════════════════════════════════
# one fit
# ═════════════════════════════════════════════════════════════════════════════

def f_gradient_diagnostic(model, X, Z_samples, F, sigma) -> float:
    """max |dQ_X/dF| at the final F and final E-step samples (descriptive).

    Closed-form Gaussian F maximises Q_X exactly (value ~ 0); a non-zero value
    for Bernoulli X measures how far the fixed-budget Adam M-step stopped."""
    n, k, L = Z_samples.shape
    grad = np.zeros_like(F)
    inv = None
    if model.family_x == "gaussian":
        inv = 1.0 / np.maximum(np.diag(sigma), 1e-8)
    for l in range(L):
        Z_l = Z_samples[:, :, l]
        resid = X - model._mean_function_x(Z_l @ F.T)
        if model.family_x == "gaussian":
            resid = resid * inv[None, :]
        grad += resid.T @ Z_l
    return float(np.max(np.abs(grad / L)))


def run_cell(split: SplitData, arm: str, K: int, init_seed: int, mc_seed: int,
             cfg: dict[str, Any] = CONFIG, *, model_kind: str | None = None,
             num_iter: int | None = None, Y_fit_override: np.ndarray | None = None,
             mode_second_start: bool = False, compute_mode: bool = False,
             ) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    """Fit one cell.  Returns (technical_row, score_rows, states).

    No retry, replacement or seed change is possible: fail_fast is passed and
    any exception is recorded as the cell status."""
    import laplace_k_criterion as lk
    spec = ARMS[arm]
    X = split.X_raw if spec["x_variant"] == "raw" else split.X_centered
    Y_fit = split.Y_fit if Y_fit_override is None else Y_fit_override
    n_iter = cfg["num_iter"] if num_iter is None else num_iter
    model_kind = cfg["model_kind"] if model_kind is None else model_kind
    tech: dict[str, Any] = {
        "arm": arm, "family_x": spec["family_x"], "fix_x": spec["fix_x"],
        "x_variant": spec["x_variant"], "K": K, "split_index": split.split_index,
        "init_seed": init_seed, "mc_seed": mc_seed, "L": cfg["L"],
        "num_iter": n_iter, "model_kind": model_kind,
        "numerics_mode": cfg["numerics_mode"],
        "failure_policy": cfg["failure_policy"],
        "train_mask_hash": split.summary["train_mask_hash"],
        "test_mask_hash": split.summary["test_mask_hash"],
        "y_fit_hash": array_hash(Y_fit), "x_hash": array_hash(X),
    }
    states: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with use_model_class(MODEL_CLASSES[model_kind]):
                res = em_runner.run_em_experimental(
                    X, Y_fit, spec["family_x"], cfg["family_y"], k=K,
                    L=cfg["L"], num_iter=n_iter, seed=init_seed,
                    train_mask=split.train_mask, fix_x=spec["fix_x"],
                    compute_strict_Q=True, mstep_q_diagnostic=True,
                    numerics_mode=cfg["numerics_mode"],
                    failure_policy=cfg["failure_policy"],
                    init_seed=init_seed, mc_seed=mc_seed)
            tech["fit_status"] = "OK"
        except Exception as exc:                       # recorded, never retried
            res = None
            tech["fit_status"] = f"FAILED:{type(exc).__name__}"
            tech["fit_error"] = str(exc)[:300]
    tech["em_seconds"] = round(time.perf_counter() - t0, 3)
    tech["n_warnings_em"] = len(caught)
    tech["warning_kinds_em"] = ";".join(sorted({w.category.__name__ for w in caught}))
    if res is None:
        return tech, [], states

    model = res["model"]
    hist = res["mstep_q_history"]
    tech.update({
        "retry_count": res["retry_count"], "replacement_count": res["replacement_count"],
        "seed_rescue_count": res["seed_rescue_count"],
        "nan_count": res["nan_count"], "q_bic_failed": res["q_bic_failed"],
        "Q_strict": res["Q_strict"], "w0": res["w0"], "w": res["w"],
        "mstep_q_decreases": int(sum(h["decreased"] for h in hist)),
        "mstep_q_min_diff": float(min(h["q_diff"] for h in hist)) if hist else float("nan"),
        "estep_sweeps": model.sampling_stats["estep_sweeps"],
        "sampling_draws": model.sampling_stats["draws"],
        "sampling_fallbacks": model.sampling_stats["fallbacks"],
        "F_abs_max": float(np.max(np.abs(res["F"]))),
        "F_grad_max": (f_gradient_diagnostic(model, X, res["Z_samples"], res["F"],
                                             res["sigma"])
                       if not spec["fix_x"] else 0.0),
        "sigma_x_min": float(np.min(np.diag(res["sigma"]))),
        "Z_mean_norm": float(np.linalg.norm(res["Z_samples"][:, :, -1].mean(axis=0))),
    })
    flog = list(getattr(model, "f_newton_log", []))
    tech["f_update"] = ("none" if spec["fix_x"] else
                        "closed_form" if spec["family_x"] == "gaussian" else
                        "newton" if flog else "adam50")
    if flog:
        tech.update({
            "fnewton_msteps": len(flog),
            "fnewton_all_converged": all(r["status"] == "converged" for r in flog),
            "fnewton_max_iters": max(r["iters"] for r in flog),
            "fnewton_qx_decreases": sum(r["q_after"] < r["q_before"] for r in flog),
            "fnewton_final_grad_max": flog[-1]["grad_max"],
            "fnewton_min_eig_H": min(r["min_eig_H"] for r in flog),
            "fnewton_eta_absmax": max(r["eta_absmax"] for r in flog),
            "fnewton_line_search_failures": sum(r["line_search_failures"] for r in flog),
        })
    tech["_f_newton_log"] = flog
    tech["_mstep_q_history"] = list(hist)
    states.update(Z_samples=res["Z_samples"], F=res["F"], sigma=res["sigma"],
                  w0=np.array(res["w0"]), w=np.array(res["w"]))

    estimators = {"last": res["Z_samples"][:, :, -1],
                  "mean5": res["Z_samples"].mean(axis=2)}
    if compute_mode:
        starts = [("mode", estimators["last"])]
        if mode_second_start:
            starts.append(("mode_from_mean5", estimators["mean5"]))
        for name, Z0 in starts:
            t1 = time.perf_counter()
            try:
                lap = lk.laplace_log_observed(model, X, Y_fit, Z0,
                                              grad_tol=cfg["mode_grad_tol"],
                                              max_iter=cfg["mode_max_iter"])
                tech[f"{name}_status"] = lap.status
                tech[f"{name}_grad_inf"] = lap.grad_inf
                tech[f"{name}_min_eig_H"] = lap.min_eigenvalue_H
                tech[f"{name}_iterations"] = lap.iterations
                Zh = np.asarray(lap.Z_hat, float)
                estimators[name] = Zh
                states[name] = Zh
            except Exception as exc:
                tech[f"{name}_status"] = f"ERROR:{type(exc).__name__}"
            tech[f"{name}_seconds"] = round(time.perf_counter() - t1, 3)
        if "mode" in estimators and "mode_from_mean5" in estimators:
            tech["mode_two_start_max_diff"] = float(np.max(np.abs(
                estimators["mode"] @ estimators["mode"].T
                - estimators["mode_from_mean5"] @ estimators["mode_from_mean5"].T)))

    score_rows = []
    for name, Z in estimators.items():
        row = {"arm": arm, "K": K, "split_index": split.split_index,
               "init_seed": init_seed, "estimator": name,
               "estimator_status": tech.get(f"{name}_status", "OK")}
        try:
            row.update(heldout_scores(Z, res["w0"], res["w"], split))
        except Exception as exc:
            row["score_error"] = f"{type(exc).__name__}: {exc}"[:200]
        score_rows.append(row)
    row = {"arm": arm, "K": K, "split_index": split.split_index,
           "init_seed": init_seed, "estimator": "mc_predictive",
           "estimator_status": "OK"}
    try:
        row.update(heldout_scores_mc_predictive(res["Z_samples"], res["w0"], res["w"],
                                                split))
    except Exception as exc:
        row["estimator_status"] = "ERROR"
        row["score_error"] = f"{type(exc).__name__}: {exc}"[:200]
    score_rows.append(row)
    tech["total_seconds"] = round(time.perf_counter() - t0, 3)
    return tech, score_rows, states


# ═════════════════════════════════════════════════════════════════════════════
# provenance
# ═════════════════════════════════════════════════════════════════════════════

def environment_info() -> dict[str, Any]:
    import platform
    import subprocess
    import scipy
    import sklearn

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unavailable"

    req = _ROOT / "requirements.txt"
    return {
        "runner_version": RUNNER_VERSION,
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "requirements_sha256": file_sha256(req) if req.exists() else None,
        "cora_content_sha256": file_sha256(DATA_DIR / "cora.content"),
        "cora_cites_sha256": file_sha256(DATA_DIR / "cora.cites"),
        "config": CONFIG, "config_hash": config_hash(CONFIG),
    }
