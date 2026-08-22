"""
Issue #33 — User-disjoint real-data validation on MovieLens 100K with train-only
movie selection and train-only derived attributes.

The pre-registered design is frozen in GitHub Issue #33 (repository
H-T-0726/expfam-lsm-attributed-relational-data), design base
main = e4be01afd1e911ee0d6bed491166258a07af1f0f.  This script implements that
design; the Issue body remains the single source of truth for it.

Implementation lineage (KI-002):
    expfam/src/experimental/em_runner.py
    expfam/src/experimental/model_dual_expfam_consistent.py
        DualExpFamLSMConsistent / DualExpFamLSMPerColumnConsistent
    which descend from model_dual_expfam_fixed.py (the "extra 1/2 なし" series).
    The per-column X model is a PROTOTYPE (CLAUDE.md 1 / 3); no manuscript
    adoption.  Every fit uses numerics_mode="consistent".

No file under expfam/src/** or reproduction/src/** is modified by this script.
Everything is built from the tracked expfam/data/ml-100k.zip; the untracked
expfam/data/movielens_pilot/*.npy artifacts are never read.

Phases:
    --phase gate    PHASE 2  structural provenance gate, 0 model fits
    --phase smoke   PHASE 3  gate(1 split) + 1 x 2 x 6 = 12 fits
    --phase full    PHASE 4  gate(30 splits) + 30 x 2 x 6 = 360 fits
    --phase docval  deterministic document validator, 0 fits

Only --phase full writes the tracked deliverable CSVs and figures.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import re
import subprocess
import sys
import time
import types
import typing
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "expfam" / "src"))
sys.path.insert(0, str(_ROOT / "expfam" / "src" / "experimental"))

from em_runner import run_em_experimental, predict_mu_y        # noqa: E402
from eval_utils import heldout_count_metrics                   # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Frozen constants (Issue #33)
# ─────────────────────────────────────────────────────────────────────

ZIP_PATH = _ROOT / "expfam" / "data" / "ml-100k.zip"
OUT_DIR = _ROOT / "expfam" / "results" / "real_data" / "movielens_userdisjoint"
FIG_DIR = _ROOT / "figures" / "real_data" / "movielens_userdisjoint"
RUN_TAG = "20260822"
STEM = "movielens_userdisjoint"

SPLIT_SEED_BASE = 130000
MODEL_SEED_BASE = 131000
N_SPLITS = 30
N_MODEL_SEEDS = 2

N_USERS_TOTAL = 943
N_TRAIN_USERS = 471
N_TEST_USERS = 471
N_UNUSED_USERS = 1

K = 3
L = 5
NUM_ITER = 8
FAMILY_Y = "poisson"
NUMERICS_MODE = "consistent"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]
TARGET_GENRES = [
    "Drama", "Comedy", "Action", "Thriller", "Romance",
    "Adventure", "Crime", "Sci-Fi", "Horror", "Mystery",
]
PER_GENRE = 10
N_MOVIES = len(TARGET_GENRES) * PER_GENRE          # 100
N_PAIRS = N_MOVIES * (N_MOVIES - 1) // 2           # 4950

# Fixed historical rate thresholds, NOT retuned.  Provenance: the 2026-06
# full-data pilot design, expfam/src/prepare_movielens_data.py:58-59
# (MIN_RATINGS=30 / MAX_RATINGS=200 and its 20-300 / 10-500 fallbacks),
# re-expressed as per-user rates by dividing by the 943 corpus users.
RATE_WINDOWS = [(30 / 943, 200 / 943), (20 / 943, 300 / 943), (10 / 943, 500 / 943)]

CONDITION_ORDER = [
    "y_only",
    "genre_only",
    "genre_year",
    "genre_logcount_train",
    "mixed_train_log",
    "mixed_train_raw_poisson",
]

FLOAT_TOL = 1e-12
IDENTITY_TOL = 1e-10

REFERENCE_FLOAT_BLOCKS = [
    "mean_rating_train_raw",
    "mean_rating_train_z",
    "year_z",
    "log_count_train_z",
]

EXPOSURE_DIAGNOSTICS = [
    "n_train_events", "n_test_events", "event_ratio", "log_event_ratio",
    "Y_train_mean", "Y_test_mean", "Y_train_zero_pairs", "Y_test_zero_pairs",
    "per_movie_train_events_min", "per_movie_train_events_max",
    "per_movie_test_events_min", "per_movie_test_events_max",
    "unused_uid",
]


class ProvenanceError(Exception):
    """Structural provenance guard violation."""


class StopCondition(Exception):
    """Issue #33 ABSOLUTE STOP CONDITION."""


# ─────────────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────────────

def normalized_event_hash(df: pd.DataFrame) -> str:
    """sha256 of rating events, independent of row and column order.

    Sorts by (uid, mid, ts, rating) and hashes the int64 bytes of all four
    columns plus the row count.
    """
    arr = np.ascontiguousarray(
        df.loc[:, ["uid", "mid", "ts", "rating"]].to_numpy(dtype=np.int64))
    order = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
    arr = np.ascontiguousarray(arr[order])
    h = hashlib.sha256()
    h.update(np.int64(arr.shape[0]).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def item_metadata_hash(df: pd.DataFrame) -> str:
    """sha256 of rating-independent item metadata (mid, title, 19 genre flags)."""
    sub = df.sort_values("mid")
    h = hashlib.sha256()
    h.update(np.int64(len(sub)).tobytes())
    h.update(np.ascontiguousarray(sub["mid"].to_numpy(dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(
        sub.loc[:, GENRES].to_numpy(dtype=np.int64)).tobytes())
    h.update(b"\x00".join(t.encode("utf-8") for t in sub["title"].tolist()))
    return h.hexdigest()


def array_hash(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def int_array_hash(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.int64))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# L1 — structural guards with use-time revalidation
# ─────────────────────────────────────────────────────────────────────

_FORBIDDEN_META_COLUMNS = {
    "uid", "rating", "ts", "ratings_count", "mean_rating", "count",
    "rate", "n_ratings", "rating_sum", "mean_rating_train", "count_train",
    "user", "item", "timestamp", "rpm", "popularity",
}

_EVENT_COLUMNS = ["uid", "mid", "rating", "ts"]


@dataclass(frozen=True)
class EventView:
    """The ONLY rating-event input type."""
    df: pd.DataFrame
    tag: str
    allowed_uids: frozenset
    sha256: str

    def __post_init__(self):
        if self.tag not in ("corpus", "train", "test"):
            raise ProvenanceError("EventView: illegal tag " + repr(self.tag))
        if list(self.df.columns) != _EVENT_COLUMNS:
            raise ProvenanceError(
                "EventView: unexpected columns " + repr(list(self.df.columns)))
        if normalized_event_hash(self.df) != self.sha256:
            raise ProvenanceError("EventView: hash mismatch at construction")
        actual = frozenset(int(u) for u in self.df["uid"].unique())
        if not actual <= self.allowed_uids:
            raise ProvenanceError("EventView: uid outside allowed_uids")


@dataclass(frozen=True)
class ItemMetadataView:
    """The ONLY rating-independent metadata input type."""
    df: pd.DataFrame
    sha256: str

    def __post_init__(self):
        cols = list(self.df.columns)
        expected = ["mid", "title"] + GENRES
        if cols != expected:
            raise ProvenanceError(
                "ItemMetadataView: column set " + repr(cols)
                + " != " + repr(expected))
        lowered = {str(c).lower() for c in cols}
        bad = lowered & _FORBIDDEN_META_COLUMNS
        if bad:
            raise ProvenanceError(
                "ItemMetadataView: rating-derived column(s) present: "
                + repr(sorted(bad)))
        if self.df["mid"].duplicated().any():
            raise ProvenanceError("ItemMetadataView: duplicate mid")
        g = self.df.loc[:, GENRES].to_numpy()
        if not np.all(np.isin(g, (0, 1))):
            raise ProvenanceError("ItemMetadataView: genre values outside {0,1}")
        if item_metadata_hash(self.df) != self.sha256:
            raise ProvenanceError("ItemMetadataView: hash mismatch at construction")


def _require_train(ev) -> str:
    """Use-time revalidation.  Returns the CURRENT recomputed event hash."""
    if not isinstance(ev, EventView):
        raise ProvenanceError("_require_train: not an EventView: " + repr(type(ev)))
    current_hash = normalized_event_hash(ev.df)
    if current_hash != ev.sha256:
        raise ProvenanceError("EventView mutated after construction")
    actual_uids = frozenset(int(u) for u in ev.df["uid"].unique())
    if not actual_uids <= ev.allowed_uids:
        raise ProvenanceError("_require_train: uid outside allowed_uids")
    if actual_uids != ev.allowed_uids:
        raise ProvenanceError("_require_train: allowed_uids not fully realised")
    if len(actual_uids) != N_TRAIN_USERS:
        raise ProvenanceError(
            "_require_train: " + str(len(actual_uids)) + " uids != "
            + str(N_TRAIN_USERS))
    if ev.tag != "train":
        raise ProvenanceError("_require_train: tag " + repr(ev.tag) + " != 'train'")
    return current_hash


def _require_meta(meta) -> str:
    if not isinstance(meta, ItemMetadataView):
        raise ProvenanceError(
            "_require_meta: not an ItemMetadataView: " + repr(type(meta)))
    current = item_metadata_hash(meta.df)
    if current != meta.sha256:
        raise ProvenanceError("ItemMetadataView mutated after construction")
    return current


# ─────────────────────────────────────────────────────────────────────
# Train-only construction (the two signature-linted functions)
# ─────────────────────────────────────────────────────────────────────

def select_movies(train_events: EventView, item_meta: ItemMetadataView) -> np.ndarray:
    """Train-only genre-stratified movie selection (10 target genres x 10 movies)."""
    _require_train(train_events)
    _require_meta(item_meta)

    counts = train_events.df.groupby("mid")["rating"].count()
    rate = {int(m): int(c) / N_TRAIN_USERS for m, c in counts.items()}

    meta = item_meta.df.sort_values("mid")
    selected_set = set()
    per_genre_counts = {}
    for genre in TARGET_GENRES:
        genre_mids = [int(m) for m in meta.loc[meta[genre] == 1, "mid"].tolist()]
        chosen = []
        for lo, hi in RATE_WINDOWS:
            cands = [m for m in genre_mids
                     if m not in selected_set and m in rate
                     and lo <= rate[m] <= hi]
            # stable sort by descending train rate; ties break by ascending mid
            chosen = sorted(cands, key=lambda m: rate[m], reverse=True)[:PER_GENRE]
            if len(chosen) == PER_GENRE:
                break
        per_genre_counts[genre] = len(chosen)
        selected_set.update(chosen)
    if any(v != PER_GENRE for v in per_genre_counts.values()):
        raise StopCondition(
            "select_movies: per-genre counts " + repr(per_genre_counts)
            + " != " + str(PER_GENRE) + " each")
    movie_ids = np.array(sorted(selected_set), dtype=np.int64)
    if movie_ids.size != N_MOVIES:
        raise StopCondition(
            "select_movies: " + str(movie_ids.size) + " movies != " + str(N_MOVIES))
    return movie_ids


def build_train_attributes(train_events: EventView, item_meta: ItemMetadataView,
                           movie_ids) -> dict:
    """Train-only derived attributes on the already-fixed movie_ids."""
    _require_train(train_events)
    _require_meta(item_meta)

    mids = [int(m) for m in movie_ids]
    grp = train_events.df.groupby("mid")["rating"]
    cnt = grp.count()
    tot = grp.sum()

    count_train = np.array([int(cnt.get(m, 0)) for m in mids], dtype=np.int64)
    sum_train = np.array([int(tot.get(m, 0)) for m in mids], dtype=np.int64)
    if np.any(count_train <= 0):
        raise StopCondition("build_train_attributes: zero train count on a "
                            "selected movie (mean_rating_train undefined)")
    mean_rating_train = (sum_train.astype(np.float64)
                         / count_train.astype(np.float64))

    meta = item_meta.df.set_index("mid")
    genre19 = meta.loc[mids, GENRES].to_numpy(dtype=np.float64)

    years = []
    for m in mids:
        mo = re.search(r"\((\d{4})\)\s*$", str(meta.loc[m, "title"]))
        if mo is None:
            raise StopCondition(
                "build_train_attributes: release year does not parse for mid="
                + str(m))
        years.append(int(mo.group(1)))
    year = np.array(years, dtype=np.int64)

    def zscore(v):
        v = np.asarray(v, dtype=np.float64)
        return (v - v.mean()) / v.std()

    return dict(
        movie_ids=np.array(mids, dtype=np.int64),
        genre19=genre19,
        count_train_int=count_train,
        sum_train_int=sum_train,
        year_int=year,
        count_train_raw=count_train.astype(np.float64),
        mean_rating_train_raw=mean_rating_train,
        mean_rating_train_z=zscore(mean_rating_train),
        year_z=zscore(year),
        log_count_train_z=zscore(np.log1p(count_train.astype(np.float64))),
    )


# ─────────────────────────────────────────────────────────────────────
# Split construction (the ONLY emitter of tag="train"/"test")
# ─────────────────────────────────────────────────────────────────────

def make_split(corpus: EventView, s: int) -> dict:
    if corpus.tag != "corpus":
        raise ProvenanceError("make_split: expects the corpus EventView")
    if normalized_event_hash(corpus.df) != corpus.sha256:
        raise ProvenanceError("EventView mutated after construction")

    all_uids = np.sort(np.unique(corpus.df["uid"].to_numpy(dtype=np.int64)))
    if all_uids.size != N_USERS_TOTAL:
        raise StopCondition(
            "make_split: " + str(all_uids.size) + " uids != " + str(N_USERS_TOTAL))
    rng = np.random.default_rng(SPLIT_SEED_BASE + s)
    perm = rng.permutation(all_uids)
    train_uids = perm[:N_TRAIN_USERS]
    test_uids = perm[N_TRAIN_USERS:N_TRAIN_USERS + N_TEST_USERS]
    unused_uids = perm[N_TRAIN_USERS + N_TEST_USERS:]

    tr = set(int(u) for u in train_uids)
    te = set(int(u) for u in test_uids)
    un = set(int(u) for u in unused_uids)
    if (len(tr) != N_TRAIN_USERS or len(te) != N_TEST_USERS
            or len(un) != N_UNUSED_USERS):
        raise StopCondition("make_split: split sizes wrong")
    if tr & te or tr & un or te & un:
        raise StopCondition("make_split: user sets not disjoint")
    if tr | te | un != set(int(u) for u in all_uids):
        raise StopCondition("make_split: union != all uids")

    df = corpus.df
    df_tr = df.loc[df["uid"].isin(tr)].reset_index(drop=True)
    df_te = df.loc[df["uid"].isin(te)].reset_index(drop=True)
    e_train = EventView(df_tr, "train", frozenset(tr), normalized_event_hash(df_tr))
    e_test = EventView(df_te, "test", frozenset(te), normalized_event_hash(df_te))
    return dict(e_train=e_train, e_test=e_test,
                train_uids=np.sort(train_uids), test_uids=np.sort(test_uids),
                unused_uid=int(unused_uids[0]))


def build_Y(ev: EventView, movie_ids) -> np.ndarray:
    """Y[i,j] = number of users in `ev` who rated both movie i and movie j."""
    if not isinstance(ev, EventView):
        raise ProvenanceError("build_Y: not an EventView")
    if normalized_event_hash(ev.df) != ev.sha256:
        raise ProvenanceError("EventView mutated after construction")
    if ev.tag not in ("train", "test"):
        raise ProvenanceError("build_Y: illegal tag " + repr(ev.tag))
    mids = [int(m) for m in movie_ids]
    idx = {m: i for i, m in enumerate(mids)}
    sub = ev.df.loc[ev.df["mid"].isin(idx)]
    if len(sub) == 0:
        raise StopCondition("build_Y: no events on the selected movies")
    _, uinv = np.unique(sub["uid"].to_numpy(dtype=np.int64), return_inverse=True)
    cols = np.array([idx[int(m)] for m in sub["mid"].to_numpy()], dtype=np.int64)
    B = np.zeros((int(uinv.max()) + 1, len(mids)), dtype=np.float64)
    B[uinv, cols] = 1.0
    Y = B.T @ B
    np.fill_diagonal(Y, 0.0)
    return Y


# ─────────────────────────────────────────────────────────────────────
# L2 — resolved-annotation signature lint
# ─────────────────────────────────────────────────────────────────────

_BANNED_PARAM_NAMES = {
    "ratings", "ratings_df", "corpus", "all_events", "full_events",
    "test_events", "E_test", "E_all", "uid_all", "u_data",
}


def _walk_code_names(code) -> set:
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            names |= _walk_code_names(c)
    return names


def signature_lint(fn) -> dict:
    """L2.  Raises StopCondition on any violation."""
    import inspect
    hints = typing.get_type_hints(fn)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    n_event = 0
    n_meta = 0
    for p in params:
        ann = hints.get(p.name, inspect.Parameter.empty)
        if ann is EventView:
            n_event += 1
        elif ann is ItemMetadataView:
            n_meta += 1
        elif ann is pd.DataFrame:
            raise StopCondition(
                "L2: " + fn.__name__ + " has raw pd.DataFrame parameter "
                + p.name)
        if p.name in _BANNED_PARAM_NAMES:
            raise StopCondition(
                "L2: " + fn.__name__ + " has banned parameter name " + p.name)
    if n_event != 1:
        raise StopCondition(
            "L2: " + fn.__name__ + " has " + str(n_event)
            + " EventView-annotated parameters (expected 1)")
    if n_meta != 1:
        raise StopCondition(
            "L2: " + fn.__name__ + " has " + str(n_meta)
            + " ItemMetadataView-annotated parameters (expected 1)")

    # closure cells
    for cell in (fn.__closure__ or ()):
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, (pd.DataFrame, EventView)):
            raise StopCondition(
                "L2: " + fn.__name__ + " closes over "
                + type(val).__name__)
    # module globals reachable from the function's code (recursively)
    g = fn.__globals__
    for name in sorted(_walk_code_names(fn.__code__)):
        if name in g and isinstance(g[name], (pd.DataFrame, EventView)):
            raise StopCondition(
                "L2: " + fn.__name__ + " reads module global " + name
                + " bound to " + type(g[name]).__name__)
    return dict(fn=fn.__name__, n_event_params=n_event, n_meta_params=n_meta,
                n_params=len(params), status="pass")


# ─────────────────────────────────────────────────────────────────────
# L3 — falsification negative controls
# ─────────────────────────────────────────────────────────────────────

def _expect_provenance_error(label, fn, *args, contains=None):
    try:
        fn(*args)
    except ProvenanceError as exc:
        if contains is not None and contains not in str(exc):
            raise StopCondition(
                "L3: " + label + " raised ProvenanceError without "
                + repr(contains) + ": " + str(exc))
        return str(exc)
    except Exception as exc:                                  # noqa: BLE001
        raise StopCondition(
            "L3: " + label + " raised " + type(exc).__name__
            + " instead of ProvenanceError: " + str(exc))
    raise StopCondition("L3: " + label + " did not raise — the checker is vacuous")


def falsification_controls(corpus: EventView, e_train: EventView,
                           e_test: EventView, meta: ItemMetadataView,
                           movie_ids) -> list:
    """L3.  Every guard must fire; otherwise STOP."""
    records = []
    for viewname, view in (("corpus", corpus), ("test", e_test)):
        for fname, fn in (("select_movies", select_movies),
                          ("build_train_attributes", build_train_attributes)):
            label = fname + "(" + viewname + ")"
            if fname == "select_movies":
                msg = _expect_provenance_error(label, fn, view, meta)
            else:
                msg = _expect_provenance_error(
                    label, lambda v, m: fn(v, m, movie_ids), view, meta)
            records.append(dict(control=label, fired=True, message=msg[:160]))

    # mutated deep copy
    mutated_df = e_train.df.copy(deep=True)
    mutated = EventView(mutated_df, "train", e_train.allowed_uids, e_train.sha256)
    old = int(mutated.df.at[0, "rating"])
    mutated.df.at[0, "rating"] = 1 if old != 1 else 2
    for fname, fn in (("select_movies", select_movies),
                      ("build_train_attributes", build_train_attributes)):
        label = fname + "(mutated train)"
        if fname == "select_movies":
            msg = _expect_provenance_error(label, fn, mutated, meta,
                                           contains="mutated after construction")
        else:
            msg = _expect_provenance_error(
                label, lambda v, m: fn(v, m, movie_ids), mutated, meta,
                contains="mutated after construction")
        records.append(dict(control=label, fired=True, message=msg[:160]))

    # ItemMetadataView carrying a rating-derived column
    bad_df = meta.df.copy(deep=True)
    bad_df["ratings_count"] = 0
    label = "ItemMetadataView(ratings_count)"
    msg = _expect_provenance_error(
        label, lambda d: ItemMetadataView(d, meta.sha256), bad_df)
    records.append(dict(control=label, fired=True, message=msg[:160]))
    return records


# ─────────────────────────────────────────────────────────────────────
# L4 — independent reference cross-check (raw numpy, no pandas groupby)
# ─────────────────────────────────────────────────────────────────────

def reference_recompute(uid_arr, mid_arr, rating_arr, train_uids,
                        item_mid, item_genre, item_titles) -> dict:
    """Separately written reference over raw numpy arrays."""
    n_mid = int(item_mid.max()) + 1
    train_sorted = np.sort(np.asarray(train_uids, dtype=np.int64))
    in_train = np.isin(uid_arr, train_sorted)
    tr_mid = mid_arr[in_train]
    tr_rating = rating_arr[in_train]

    ref_count_all = np.bincount(tr_mid, minlength=n_mid).astype(np.int64)
    ref_sum_all = np.bincount(tr_mid, weights=tr_rating.astype(np.float64),
                              minlength=n_mid)
    ref_sum_all = np.rint(ref_sum_all).astype(np.int64)

    rate_all = ref_count_all.astype(np.float64) / float(N_TRAIN_USERS)

    genre_pos = {g: i for i, g in enumerate(GENRES)}
    order = np.argsort(item_mid, kind="stable")
    mids_sorted = item_mid[order]
    genre_sorted = item_genre[order]

    selected = []
    selected_mask = np.zeros(n_mid, dtype=bool)
    for genre in TARGET_GENRES:
        gcol = genre_sorted[:, genre_pos[genre]]
        gmids = mids_sorted[gcol == 1]
        chosen = np.array([], dtype=np.int64)
        for lo, hi in RATE_WINDOWS:
            r = rate_all[gmids]
            ok = (~selected_mask[gmids]) & (ref_count_all[gmids] > 0) \
                & (r >= lo) & (r <= hi)
            cand = gmids[ok]
            if cand.size == 0:
                continue
            idx = np.argsort(-rate_all[cand], kind="stable")[:PER_GENRE]
            chosen = cand[idx]
            if chosen.size == PER_GENRE:
                break
        selected_mask[chosen] = True
        selected.extend(int(m) for m in chosen)
    ref_movie_ids = np.array(sorted(selected), dtype=np.int64)

    ref_count = ref_count_all[ref_movie_ids]
    ref_sum = ref_sum_all[ref_movie_ids]

    pos_of_mid = {int(m): i for i, m in enumerate(item_mid)}
    ref_genre = np.array(
        [item_genre[pos_of_mid[int(m)]] for m in ref_movie_ids], dtype=np.float64)
    ref_year = []
    for m in ref_movie_ids:
        t = str(item_titles[pos_of_mid[int(m)]])
        mo = re.search(r"\((\d{4})\)\s*$", t)
        if mo is None:
            raise StopCondition("reference: year does not parse for mid=" + str(m))
        ref_year.append(int(mo.group(1)))
    ref_year = np.array(ref_year, dtype=np.int64)

    def z(v):
        v = np.asarray(v, dtype=np.float64)
        return (v - v.mean()) / v.std()

    ref_mean = ref_sum.astype(np.float64) / ref_count.astype(np.float64)
    return dict(
        movie_ids=ref_movie_ids,
        count_train_int=ref_count,
        sum_train_int=ref_sum,
        year_int=ref_year,
        genre19=ref_genre,
        count_train_raw=ref_count.astype(np.float64),
        mean_rating_train_raw=ref_mean,
        mean_rating_train_z=z(ref_mean),
        year_z=z(ref_year),
        log_count_train_z=z(np.log1p(ref_count.astype(np.float64))),
    )


def cross_check_reference(attrs: dict, ref: dict, split: int) -> dict:
    """Pre-registered EXACT / FLOAT split.  Raises StopCondition on failure."""
    # EXACT
    exact_pairs = [
        ("movie_ids", attrs["movie_ids"], ref["movie_ids"]),
        ("genre19", attrs["genre19"], ref["genre19"]),
        ("year_int", attrs["year_int"], ref["year_int"]),
        ("count_train_int", attrs["count_train_int"], ref["count_train_int"]),
        ("sum_train_int", attrs["sum_train_int"], ref["sum_train_int"]),
        ("count_train_raw", attrs["count_train_raw"], ref["count_train_raw"]),
    ]
    for name, a, b in exact_pairs:
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise StopCondition(
                "L4 EXACT mismatch on " + name + " (split " + str(split) + ")")
    if not np.array_equal(attrs["count_train_raw"].astype(np.int64),
                          attrs["count_train_int"]):
        raise StopCondition(
            "L4: count_train_raw != per-movie train rating count (split "
            + str(split) + ")")

    # FLOAT
    errs = {}
    for name in REFERENCE_FLOAT_BLOCKS:
        a = np.asarray(attrs[name], dtype=np.float64)
        b = np.asarray(ref[name], dtype=np.float64)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise StopCondition(
                "L4: non-finite value in " + name + " (split " + str(split) + ")")
        if not np.allclose(a, b, rtol=0.0, atol=FLOAT_TOL, equal_nan=False):
            raise StopCondition(
                "L4 FLOAT mismatch on " + name + " (split " + str(split) + ")")
        errs["reference_float_max_abs_error_" + name] = float(
            np.max(np.abs(a - b)))
    for v in errs.values():
        if v > FLOAT_TOL:
            raise StopCondition(
                "L4: reference_float_max_abs_error " + str(v) + " > "
                + str(FLOAT_TOL) + " (split " + str(split) + ")")
    return errs


# ─────────────────────────────────────────────────────────────────────
# Data loading (corpus stays in local scope; nothing bound at module level)
# ─────────────────────────────────────────────────────────────────────

def load_corpus():
    if not ZIP_PATH.exists():
        raise StopCondition("data zip not found: " + str(ZIP_PATH))
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("ml-100k/u.data") as f:
            raw = f.read().decode("latin-1")
        rows = []
        for line in raw.strip().split("\n"):
            p = line.strip().split("\t")
            if len(p) >= 4:
                rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3])))
        events = pd.DataFrame(rows, columns=_EVENT_COLUMNS)

        with zf.open("ml-100k/u.item") as f:
            raw_i = f.read().decode("latin-1")
        irows = []
        for line in raw_i.strip().split("\n"):
            p = line.split("|")
            if len(p) < 24:
                continue
            irows.append([int(p[0]), p[1]] + [int(p[i + 5]) for i in range(19)])
        items = pd.DataFrame(irows, columns=["mid", "title"] + GENRES)

    corpus_uids = frozenset(int(u) for u in events["uid"].unique())
    corpus = EventView(events, "corpus", corpus_uids,
                       normalized_event_hash(events))
    meta = ItemMetadataView(items, item_metadata_hash(items))
    return corpus, meta


# ─────────────────────────────────────────────────────────────────────
# Conditions
# ─────────────────────────────────────────────────────────────────────

def build_conditions(attrs: dict) -> dict:
    genre = attrs["genre19"]
    mean_z = attrs["mean_rating_train_z"].reshape(-1, 1)
    year_z = attrs["year_z"].reshape(-1, 1)
    logc_z = attrs["log_count_train_z"].reshape(-1, 1)
    craw = attrs["count_train_raw"].reshape(-1, 1)

    conds = {}
    conds["y_only"] = dict(
        X=genre, kw=dict(family_x="gaussian", fix_x=True), n_cols_used=0,
        family_note="X unused (fix_x=True, F fixed at 0); genre19 array passed")
    conds["genre_only"] = dict(
        X=genre, kw=dict(family_x="bernoulli"), n_cols_used=19,
        family_note="bernoulli*19")
    conds["genre_year"] = dict(
        X=np.hstack([genre, year_z]),
        kw=dict(family_x=None, family_x_list=["bernoulli"] * 19 + ["gaussian"]),
        n_cols_used=20, family_note="bernoulli*19 + gaussian(year_z)")
    conds["genre_logcount_train"] = dict(
        X=np.hstack([genre, logc_z]),
        kw=dict(family_x=None, family_x_list=["bernoulli"] * 19 + ["gaussian"]),
        n_cols_used=20,
        family_note="bernoulli*19 + gaussian(log_count_train_z)")
    conds["mixed_train_log"] = dict(
        X=np.hstack([genre, mean_z, year_z, logc_z]),
        kw=dict(family_x=None,
                family_x_list=["bernoulli"] * 19 + ["gaussian"] * 3),
        n_cols_used=22,
        family_note="bernoulli*19 + gaussian(mean_rating_train_z, year_z, "
                    "log_count_train_z)")
    conds["mixed_train_raw_poisson"] = dict(
        X=np.hstack([genre, mean_z, year_z, craw]),
        kw=dict(family_x=None,
                family_x_list=["bernoulli"] * 19 + ["gaussian"] * 2 + ["poisson"]),
        n_cols_used=22,
        family_note="bernoulli*19 + gaussian(mean_rating_train_z, year_z) + "
                    "poisson(count_train_raw)")
    if list(conds.keys()) != CONDITION_ORDER:
        raise StopCondition("condition order drifted from the pre-registration")
    return conds


# ─────────────────────────────────────────────────────────────────────
# Structural gate (PHASE 2)
# ─────────────────────────────────────────────────────────────────────

def prepare_split(corpus, meta, s, corpus_arrays, prov_rows, verbose=True):
    """Structural construction + L1/L4 provenance rows for one split."""
    sp = make_split(corpus, s)
    e_train, e_test = sp["e_train"], sp["e_test"]

    train_hash_used = _require_train(e_train)
    if train_hash_used == corpus.sha256:
        raise StopCondition("split " + str(s) + ": E_train hash == corpus hash")
    if train_hash_used == e_test.sha256:
        raise StopCondition("split " + str(s) + ": E_train hash == E_test hash")

    movie_ids = select_movies(e_train, meta)
    attrs = build_train_attributes(e_train, meta, movie_ids)

    ref = reference_recompute(
        corpus_arrays["uid"], corpus_arrays["mid"], corpus_arrays["rating"],
        sp["train_uids"], corpus_arrays["item_mid"],
        corpus_arrays["item_genre"], corpus_arrays["item_title"],
        )
    float_errs = cross_check_reference(attrs, ref, s)

    Y_train = build_Y(e_train, movie_ids)
    Y_test = build_Y(e_test, movie_ids)
    for name, Yv in (("Y_train", Y_train), ("Y_test", Y_test)):
        if Yv.shape != (N_MOVIES, N_MOVIES):
            raise StopCondition("split " + str(s) + ": " + name + " shape wrong")
        if not np.allclose(Yv, Yv.T):
            raise StopCondition("split " + str(s) + ": " + name + " not symmetric")
        if np.any(Yv < 0) or np.any(Yv != np.rint(Yv)):
            raise StopCondition("split " + str(s) + ": " + name + " not counts")

    iu = np.triu_indices(N_MOVIES, 1)
    dfc = corpus.df
    n_train_events = int(dfc["uid"].isin(set(int(u) for u in sp["train_uids"])).sum())
    n_test_events = int(dfc["uid"].isin(set(int(u) for u in sp["test_uids"])).sum())
    mid_set = set(int(m) for m in movie_ids)
    tr_sel = e_train.df.loc[e_train.df["mid"].isin(mid_set)]
    te_sel = e_test.df.loc[e_test.df["mid"].isin(mid_set)]
    per_tr = tr_sel.groupby("mid")["rating"].count().reindex(
        [int(m) for m in movie_ids], fill_value=0).to_numpy()
    per_te = te_sel.groupby("mid")["rating"].count().reindex(
        [int(m) for m in movie_ids], fill_value=0).to_numpy()

    diag = {
        "n_train_events": n_train_events,
        "n_test_events": n_test_events,
        # Sign convention is fixed by the pre-registered numbers in Issue #33, which record
        # split s=2 (45076 train / 54848 test events) as log_event_ratio = +0.1962, i.e.
        # log(test / train).  Record only; never used to filter, drop or rebalance a split.
        "event_ratio": float(n_test_events) / float(n_train_events),
        "log_event_ratio": float(np.log(n_test_events / n_train_events)),
        "Y_train_mean": float(Y_train[iu].mean()),
        "Y_test_mean": float(Y_test[iu].mean()),
        "Y_train_zero_pairs": int((Y_train[iu] == 0).sum()),
        "Y_test_zero_pairs": int((Y_test[iu] == 0).sum()),
        "per_movie_train_events_min": int(per_tr.min()),
        "per_movie_train_events_max": int(per_tr.max()),
        "per_movie_test_events_min": int(per_te.min()),
        "per_movie_test_events_max": int(per_te.max()),
        "unused_uid": sp["unused_uid"],
    }

    conds = build_conditions(attrs)
    x_prov = {c: array_hash(conds[c]["X"]) for c in CONDITION_ORDER}
    movie_set_hash = int_array_hash(movie_ids)
    y_train_hash = array_hash(Y_train)
    y_test_hash = array_hash(Y_test)
    if y_train_hash == y_test_hash:
        raise StopCondition("split " + str(s) + ": Y_train_hash == Y_test_hash")

    base = dict(
        stage="structural", split=s, condition="", model_seed="",
        tag="train", n_uids=N_TRAIN_USERS,
        events_sha256=train_hash_used, stored_sha256=e_train.sha256,
        corpus_sha256=corpus.sha256, test_events_sha256=e_test.sha256,
        item_metadata_sha256=meta.sha256,
        train_uid_hash=int_array_hash(sp["train_uids"]),
        test_uid_hash=int_array_hash(sp["test_uids"]),
        movie_set_hash=movie_set_hash,
        y_train_hash=y_train_hash, y_test_hash=y_test_hash,
        x_input_hash="", expected_x_provenance_hash="",
        numerics_mode="", n_movies=int(movie_ids.size),
    )
    base.update(diag)
    base.update(float_errs)
    prov_rows.append(base)
    for c in CONDITION_ORDER:
        row = dict(base)
        row["condition"] = c
        row["x_input_hash"] = x_prov[c]
        row["expected_x_provenance_hash"] = x_prov[c]
        row["numerics_mode"] = NUMERICS_MODE
        prov_rows.append(row)

    if verbose:
        print("  split %2d: movies=%d  train_ev=%d test_ev=%d  logratio=%+.4f  "
              "Ytr_mean=%.2f Yte_mean=%.2f  te_min=%d"
              % (s, movie_ids.size, n_train_events, n_test_events,
                 diag["log_event_ratio"], diag["Y_train_mean"],
                 diag["Y_test_mean"], diag["per_movie_test_events_min"]))

    return dict(split=s, movie_ids=movie_ids, attrs=attrs, conds=conds,
                Y_train=Y_train, Y_test=Y_test, diag=diag, x_prov=x_prov,
                movie_set_hash=movie_set_hash,
                y_train_hash=y_train_hash, y_test_hash=y_test_hash,
                train_uid_hash=base["train_uid_hash"],
                test_uid_hash=base["test_uid_hash"])


def structural_gate(splits, verbose=True):
    corpus, meta = load_corpus()
    corpus_arrays = dict(
        uid=corpus.df["uid"].to_numpy(dtype=np.int64),
        mid=corpus.df["mid"].to_numpy(dtype=np.int64),
        rating=corpus.df["rating"].to_numpy(dtype=np.int64),
        item_mid=meta.df["mid"].to_numpy(dtype=np.int64),
        item_genre=meta.df.loc[:, GENRES].to_numpy(dtype=np.int64),
        item_title=meta.df["title"].tolist(),
    )

    print("=== L2 resolved-annotation signature lint ===")
    lint = [signature_lint(select_movies), signature_lint(build_train_attributes)]
    for r in lint:
        print("  " + r["fn"] + ": " + r["status"])

    print("=== L3 falsification negative controls ===")
    sp0 = make_split(corpus, splits[0])
    mids0 = select_movies(sp0["e_train"], meta)
    l3 = falsification_controls(corpus, sp0["e_train"], sp0["e_test"], meta, mids0)
    print("  " + str(len(l3)) + " guards fired (all required)")

    print("=== L1/L4 per-split structural construction ===")
    prov_rows = []
    prepared = []
    for s in splits:
        prepared.append(prepare_split(corpus, meta, s, corpus_arrays,
                                      prov_rows, verbose=verbose))

    hashes = {p["split"]: (p["train_uid_hash"], p["test_uid_hash"])
              for p in prepared}
    tr_hashes = [v[0] for v in hashes.values()]
    if len(set(tr_hashes)) != len(tr_hashes):
        raise StopCondition("duplicate train_uid_hash across splits")
    te_hashes = [v[1] for v in hashes.values()]
    if len(set(te_hashes)) != len(te_hashes):
        raise StopCondition("duplicate test_uid_hash across splits")

    return dict(prepared=prepared, prov_rows=prov_rows, lint=lint, l3=l3,
                corpus_sha256=corpus.sha256, meta_sha256=meta.sha256)


# ─────────────────────────────────────────────────────────────────────
# Fits (PHASE 3 / PHASE 4)
# ─────────────────────────────────────────────────────────────────────

def run_one_fit(prep, cname, model_seed, prov_rows, stage):
    cond = prep["conds"][cname]
    X = np.ascontiguousarray(np.asarray(cond["X"], dtype=np.float64))
    Y = np.ascontiguousarray(np.asarray(prep["Y_train"], dtype=np.float64))

    x_input_hash = array_hash(X)
    y_input_hash = array_hash(Y)
    expected = prep["x_prov"][cname]
    if x_input_hash != expected:
        raise StopCondition(
            "fit ledger: x_input_hash != expected_x_provenance_hash ("
            + cname + ", split " + str(prep["split"]) + ")")
    if y_input_hash != prep["y_train_hash"]:
        raise StopCondition(
            "fit ledger: y_input_hash != y_train_hash (" + cname + ", split "
            + str(prep["split"]) + ")")
    if y_input_hash == prep["y_test_hash"]:
        raise StopCondition(
            "fit ledger: y_input_hash == y_test_hash (" + cname + ", split "
            + str(prep["split"]) + ")")

    prov_rows.append(dict(
        stage=stage, split=prep["split"], condition=cname,
        model_seed=model_seed, tag="fit", n_uids=N_TRAIN_USERS,
        events_sha256="", stored_sha256="", corpus_sha256="",
        test_events_sha256="", item_metadata_sha256="",
        train_uid_hash=prep["train_uid_hash"],
        test_uid_hash=prep["test_uid_hash"],
        movie_set_hash=prep["movie_set_hash"],
        y_train_hash=prep["y_train_hash"], y_test_hash=prep["y_test_hash"],
        x_input_hash=x_input_hash,
        expected_x_provenance_hash=expected,
        numerics_mode=NUMERICS_MODE, n_movies=N_MOVIES,
        ledger_valid=True,
    ))

    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(buf):
            res = run_em_experimental(
                X, Y, family_y=FAMILY_Y, k=K, L=L, num_iter=NUM_ITER,
                seed=model_seed, train_mask=None,
                validate_support=True, verbose=True,
                numerics_mode=NUMERICS_MODE, **cond["kw"])
    n_warnings = len(wlist)
    internal_retry = buf.getvalue().count("[NaN iter=")

    mu_y = predict_mu_y(res)
    full_mask = np.ones((N_MOVIES, N_MOVIES), dtype=bool)
    np.fill_diagonal(full_mask, False)
    m_te = heldout_count_metrics(prep["Y_test"], mu_y, full_mask, FAMILY_Y)
    m_tr = heldout_count_metrics(prep["Y_train"], mu_y, full_mask, FAMILY_Y)

    row = dict(
        split=prep["split"], model_seed=model_seed, condition=cname,
        n_cols_used=cond["n_cols_used"], family_note=cond["family_note"],
        k=K, L=L, num_iter=NUM_ITER, family_y=FAMILY_Y,
        numerics_mode=res.get("numerics_mode", NUMERICS_MODE),
        test_y_ll=m_te.get("mean_ll", float("nan")),
        test_y_rmse=m_te.get("rmse", float("nan")),
        test_spearman=m_te.get("spearman", float("nan")),
        test_pearson=m_te.get("pearson", float("nan")),
        test_n_pairs=m_te.get("n_pairs", -1),
        train_y_ll=m_tr.get("mean_ll", float("nan")),
        train_y_rmse=m_tr.get("rmse", float("nan")),
        train_spearman=m_tr.get("spearman", float("nan")),
        w0=float(res["w0"]), w=float(res["w"]),
        nan_occurred=bool(res["nan_occurred"]),
        nan_count=int(res.get("nan_count", 0)),
        internal_retry=int(internal_retry),
        q_bic_failed=bool(res.get("q_bic_failed", False)),
        n_warnings=int(n_warnings),
        validate_support_passed=True,
        runtime_s=float(res["runtime_s"]),
        movie_set_hash=prep["movie_set_hash"],
        x_input_hash=x_input_hash,
        y_train_hash=prep["y_train_hash"],
        y_test_hash=prep["y_test_hash"],
    )
    for key in ("test_y_ll", "test_y_rmse", "test_spearman", "test_pearson",
                "train_y_ll", "train_y_rmse", "train_spearman", "w0", "w"):
        if not np.isfinite(row[key]):
            raise StopCondition(
                "non-finite metric " + key + " (" + cname + ", split "
                + str(prep["split"]) + ", seed " + str(model_seed) + ")")
    if row["test_n_pairs"] != N_PAIRS:
        raise StopCondition("test evaluated on " + str(row["test_n_pairs"])
                            + " pairs != " + str(N_PAIRS))
    if internal_retry >= 1:
        raise StopCondition("internal retry detected (" + cname + ", split "
                            + str(prep["split"]) + ")")
    if row["nan_occurred"] or row["nan_count"] > 0:
        raise StopCondition("NaN occurred (" + cname + ", split "
                            + str(prep["split"]) + ")")
    if row["q_bic_failed"]:
        raise StopCondition("q_bic_failed (" + cname + ", split "
                            + str(prep["split"]) + ")")
    if n_warnings > 0:
        raise StopCondition(
            "warning raised (" + cname + ", split " + str(prep["split"])
            + "): " + str([str(w.message)[:120] for w in wlist]))
    if row["numerics_mode"] != NUMERICS_MODE:
        raise StopCondition("numerics_mode drift: " + str(row["numerics_mode"]))
    return row


def run_fits(prepared, prov_rows, stage, verbose=True):
    rows = []
    for prep in prepared:
        for m in range(N_MODEL_SEEDS):
            model_seed = MODEL_SEED_BASE + prep["split"] * 10 + m
            for cname in CONDITION_ORDER:
                r = run_one_fit(prep, cname, model_seed, prov_rows, stage)
                rows.append(r)
                if verbose:
                    print("  s=%2d m=%d %-24s te_ll=%9.4f te_rmse=%7.2f "
                          "tr_ll=%9.4f (%.1fs)"
                          % (prep["split"], m, cname, r["test_y_ll"],
                             r["test_y_rmse"], r["train_y_ll"], r["runtime_s"]),
                          flush=True)
    return rows


# ─────────────────────────────────────────────────────────────────────
# Aggregation / contrasts
# ─────────────────────────────────────────────────────────────────────

CONTRASTS = [
    ("delta_primary", "mixed_train_log", "genre_only", "primary"),
    ("A_component", "mixed_train_log", "genre_logcount_train", "decomposition"),
    ("B_component", "genre_logcount_train", "genre_only", "decomposition"),
    ("P_positive_control", "genre_only", "y_only", "positive_control"),
    ("sec_mixed_vs_yonly", "mixed_train_log", "y_only", "secondary"),
    ("sec_genreyear_vs_genreonly", "genre_year", "genre_only", "secondary"),
    ("sec_rawpoisson_vs_mixedlog", "mixed_train_raw_poisson", "mixed_train_log",
     "secondary"),
]

SECONDARY_METRICS = ["test_y_rmse", "test_spearman", "train_y_ll"]


def summarize_values(values):
    v = np.asarray(values, dtype=np.float64)
    mean = float(v.mean())
    abs_mean = abs(mean)
    p10, p90 = np.percentile(v, [10, 90], method="linear")
    ratio = (float(v.std(ddof=1) / abs_mean) if abs_mean >= 1e-12 else "undefined")
    return dict(
        n=int(v.size), mean=mean, median=float(np.median(v)),
        min=float(v.min()), max=float(v.max()),
        count_positive=int((v > 0).sum()),
        std=float(v.std(ddof=1)), abs_mean=abs_mean,
        spread_to_abs_mean_ratio=ratio,
        p10=float(p10), p90=float(p90),
    )


def build_aggregates(df):
    """Seed-averaging first, then per-split contrasts."""
    cell = (df.groupby(["condition", "split"])
              .agg(test_y_ll=("test_y_ll", "mean"),
                   test_y_rmse=("test_y_rmse", "mean"),
                   test_spearman=("test_spearman", "mean"),
                   test_pearson=("test_pearson", "mean"),
                   train_y_ll=("train_y_ll", "mean"),
                   train_y_rmse=("train_y_rmse", "mean"),
                   n_seeds=("test_y_ll", "count"))
              .reset_index())
    if not (cell["n_seeds"] == N_MODEL_SEEDS).all():
        raise StopCondition("seed averaging: not every cell has "
                            + str(N_MODEL_SEEDS) + " seeds")

    pivot = cell.pivot(index="split", columns="condition", values="test_y_ll")
    splits = list(pivot.index)

    paired_rows = []
    for name, a, b, role in CONTRASTS:
        vals = (pivot[a] - pivot[b]).to_numpy()
        for s, v in zip(splits, vals):
            paired_rows.append(dict(contrast=name, role=role, metric="test_y_ll",
                                    cond_a=a, cond_b=b, split=int(s),
                                    value=float(v)))
    for metric in SECONDARY_METRICS:
        pv = cell.pivot(index="split", columns="condition", values=metric)
        vals = (pv["mixed_train_log"] - pv["genre_only"]).to_numpy()
        for s, v in zip(splits, vals):
            paired_rows.append(dict(
                contrast="sec_mixed_vs_genreonly_" + metric, role="secondary",
                metric=metric, cond_a="mixed_train_log", cond_b="genre_only",
                split=int(s), value=float(v)))
    paired = pd.DataFrame(paired_rows)

    delta = (pivot["mixed_train_log"] - pivot["genre_only"]).to_numpy()
    a_comp = (pivot["mixed_train_log"] - pivot["genre_logcount_train"]).to_numpy()
    b_comp = (pivot["genre_logcount_train"] - pivot["genre_only"]).to_numpy()
    identity_err = float(np.max(np.abs(delta - (a_comp + b_comp))))
    if not identity_err < IDENTITY_TOL:
        raise StopCondition(
            "decomposition identity violated: max |Delta - (A+B)| = "
            + str(identity_err))

    agg_rows = []
    for cname in CONDITION_ORDER:
        sub = cell.loc[cell["condition"] == cname]
        agg_rows.append(dict(
            kind="condition", name=cname, metric="test_y_ll",
            **summarize_values(sub["test_y_ll"].to_numpy())))
        for metric in ["test_y_rmse", "test_spearman", "train_y_ll"]:
            agg_rows.append(dict(
                kind="condition", name=cname, metric=metric,
                **summarize_values(sub[metric].to_numpy())))
    for name in paired["contrast"].unique():
        sub = paired.loc[paired["contrast"] == name]
        agg_rows.append(dict(
            kind="contrast", name=name, metric=str(sub["metric"].iloc[0]),
            **summarize_values(sub["value"].to_numpy())))
    agg = pd.DataFrame(agg_rows)
    return cell, paired, agg, dict(
        identity_err=identity_err, delta=delta, A=a_comp, B=b_comp,
        splits=splits, pivot=pivot)


def rank_correlation_record_only(log_event_ratio, delta):
    rx = scipy.stats.rankdata(np.asarray(log_event_ratio, dtype=np.float64))
    ry = scipy.stats.rankdata(np.asarray(delta, dtype=np.float64))
    return float(np.corrcoef(rx, ry)[0, 1])


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def make_figures(cell, paired, extra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    splits = extra["splits"]
    delta = extra["delta"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(splits, delta, color=["#3b7dd8" if d > 0 else "#d1495b" for d in delta])
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axhline(float(np.mean(delta)), color="#2e7d32", lw=1.2, ls="--",
               label="mean = %+.4f" % float(np.mean(delta)))
    ax.set_xlabel("split s (user-disjoint, seed 130000+s)")
    ax.set_ylabel("Delta = LL_test(mixed_train_log) - LL_test(genre_only)")
    ax.set_title("Primary contrast across repeated user-disjoint splits "
                 "(not independent replicates)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / (STEM + "_" + RUN_TAG + "_delta_stability.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    means, stds = [], []
    for c in CONDITION_ORDER:
        v = cell.loc[cell["condition"] == c, "test_y_ll"].to_numpy()
        means.append(v.mean())
        stds.append(v.std(ddof=1))
    ax.bar(range(len(CONDITION_ORDER)), means, yerr=stds, capsize=3,
           color="#4c72b0")
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels(CONDITION_ORDER, rotation=20, ha="right")
    ax.set_ylabel("mean LL_test per pair (Poisson score)")
    ax.set_title("Held-out test-user LL by condition "
                 "(error bars = across-split spread, not standard errors)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / (STEM + "_" + RUN_TAG + "_condition_ll.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.28
    xs = np.arange(len(splits))
    ax.bar(xs - w, extra["A"], w, label="A = mixed_train_log - genre_logcount_train")
    ax.bar(xs, extra["B"], w, label="B = genre_logcount_train - genre_only")
    ax.bar(xs + w, delta, w, label="Delta = A + B")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xs[::2])
    ax.set_xticklabels([str(s) for s in splits[::2]])
    ax.set_xlabel("split s")
    ax.set_ylabel("difference in LL_test per pair")
    ax.set_title("Mandatory decomposition of the primary contrast")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / (STEM + "_" + RUN_TAG + "_decomposition.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Document validator
# ─────────────────────────────────────────────────────────────────────

REQUIRED_DISCLAIMER_PHRASES = [
    "No significance test, confidence interval, or power analysis was performed",
    "the 30 splits reuse the same 943 users and are not independent replicates",
    "it is not an untouched external confirmation",
    "The primary real-data contrast evaluates a heterogeneous-X Bernoulli/Gaussian configuration",
    "does not validate simultaneous Bernoulli/Gaussian/Poisson X integration on real data",
    "The Poisson per-pair log-likelihood is used as a score, not as evidence of correct specification",
]

FORBIDDEN_CLAIM_PHRASES = [
    "first genuinely leakage-free real-data validation",
    "fully independent confirmatory MovieLens test",
    "untouched external confirmation",
    "leakage-free validation",
    "all three X families are validated on MovieLens",
    "Bernoulli/Gaussian/Poisson joint integration improves real-data performance",
    "the three-family per-column method was validated on real data",
    "the Poisson model fits",
    "we predicted unseen co-rating counts on MovieLens",
]

_BEGIN_MARK = "<!-- PROHIBITED-CLAIM-LIST:BEGIN -->"
_END_MARK = "<!-- PROHIBITED-CLAIM-LIST:END -->"


def _normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t)
    return t.lower()


def validate_document(text: str, label: str) -> dict:
    """Deterministic phrase-level validator (STEP 1-6)."""
    # STEP 1 normalize
    scan = _normalize(text)

    # STEP 2 remove only the explicit prohibited-list region
    begin, end = _normalize(_BEGIN_MARK), _normalize(_END_MARK)
    regions = 0
    while True:
        i = scan.find(begin)
        if i < 0:
            break
        j = scan.find(end, i)
        if j < 0:
            raise StopCondition(
                label + ": unterminated PROHIBITED-CLAIM-LIST region")
        scan = scan[:i] + " " + scan[j + len(end):]
        regions += 1

    # STEP 3 verify required disclaimers first
    missing = [p for p in REQUIRED_DISCLAIMER_PHRASES
               if _normalize(p) not in scan]
    found = len(REQUIRED_DISCLAIMER_PHRASES) - len(missing)
    if missing:
        raise StopCondition(
            label + ": required disclaimer(s) absent ("
            + str(found) + "/6): " + repr(missing))

    # STEP 4 remove only the approved disclaimer occurrences
    removed = 0
    for p in REQUIRED_DISCLAIMER_PHRASES:
        np_ = _normalize(p)
        cnt = scan.count(np_)
        removed += cnt
        scan = scan.replace(np_, " ")

    # STEP 5 forbidden phrase scan
    hits = []
    for p in FORBIDDEN_CLAIM_PHRASES:
        c = scan.count(_normalize(p))
        if c:
            hits.append((p, c))
    final_violations = sum(c for _, c in hits)

    out = dict(
        document=label,
        prohibited_list_regions_removed=regions,
        required_disclaimers_expected=len(REQUIRED_DISCLAIMER_PHRASES),
        required_disclaimers_found=found,
        approved_disclaimer_occurrences_removed=removed,
        forbidden_raw_hits_after_whitelist=len(hits),
        final_violations=final_violations,
        hits=repr(hits),
    )
    if final_violations > 0:
        raise StopCondition(label + ": forbidden phrase(s) remain: " + repr(hits))
    if found != 6:
        raise StopCondition(label + ": required_disclaimers_found != 6")
    return out


# ─────────────────────────────────────────────────────────────────────
# Runinfo
# ─────────────────────────────────────────────────────────────────────

def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=_ROOT).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


def git_branch():
    try:
        return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True,
                              cwd=_ROOT).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["gate", "smoke", "full", "docval"])
    ap.add_argument("--targets", nargs="*", default=[],
                    help="files to validate in --phase docval")
    args = ap.parse_args()

    t0 = time.perf_counter()

    if args.phase == "docval":
        rows = []
        for t in args.targets:
            p = Path(t)
            rows.append(validate_document(p.read_text(encoding="utf-8"), p.name))
        for r in rows:
            print(r)
        print("DOCVAL OK")
        return 0

    splits = [0] if args.phase == "smoke" else list(range(N_SPLITS))
    print("=== Issue #33 user-disjoint MovieLens validation | phase=%s ==="
          % args.phase)
    print("lineage: em_runner + model_dual_expfam_consistent "
          "(PerColumn prototype), numerics_mode=%s, K=%d, L=%d, num_iter=%d"
          % (NUMERICS_MODE, K, L, NUM_ITER))

    gate = structural_gate(splits, verbose=True)
    prepared = gate["prepared"]
    prov_rows = gate["prov_rows"]

    diag_df = pd.DataFrame([dict(split=p["split"], **p["diag"]) for p in prepared])
    print("\n=== split diagnostics (record only) ===")
    print("  log_event_ratio: mean %+.4f sd %.4f |max| %.4f"
          % (diag_df["log_event_ratio"].mean(), diag_df["log_event_ratio"].std(ddof=1),
             diag_df["log_event_ratio"].abs().max()))
    print("  per_movie_test_events_min over splits: %d"
          % diag_df["per_movie_test_events_min"].min())
    print("  Y_test zero pairs: %d..%d ; Y_train zero pairs: %d..%d"
          % (diag_df["Y_test_zero_pairs"].min(), diag_df["Y_test_zero_pairs"].max(),
             diag_df["Y_train_zero_pairs"].min(), diag_df["Y_train_zero_pairs"].max()))

    if args.phase == "gate":
        print("\nGATE OK (%d splits, 0 fits, %.1fs)"
              % (len(splits), time.perf_counter() - t0))
        return 0

    stage = "fit_time"
    print("\n=== fits (%d) ===" % (len(prepared) * N_MODEL_SEEDS
                                   * len(CONDITION_ORDER)))
    rows = run_fits(prepared, prov_rows, stage, verbose=True)
    df = pd.DataFrame(rows)

    expected_fits = len(prepared) * N_MODEL_SEEDS * len(CONDITION_ORDER)
    if len(df) != expected_fits:
        raise StopCondition("fit count " + str(len(df)) + " != "
                            + str(expected_fits))
    if df.duplicated(subset=["split", "model_seed", "condition"]).any():
        raise StopCondition("duplicate (split, model_seed, condition)")
    if int(df["nan_occurred"].sum()) != 0:
        raise StopCondition("nan_occurred total != 0")
    if int(df["internal_retry"].sum()) != 0:
        raise StopCondition("internal retry total != 0")
    if int(df["q_bic_failed"].sum()) != 0:
        raise StopCondition("q_bic_failed total != 0")
    if int(df["n_warnings"].sum()) != 0:
        raise StopCondition("warnings total != 0")
    if not (df["numerics_mode"] == NUMERICS_MODE).all():
        raise StopCondition("not every fit ran in consistent mode")

    prov = pd.DataFrame(prov_rows)
    n_fit_rows = int((prov["stage"] == "fit_time").sum())
    if n_fit_rows != expected_fits:
        raise StopCondition("fit-ledger rows " + str(n_fit_rows) + " != "
                            + str(expected_fits))
    print("\nfit ledger: %d/%d rows valid" % (n_fit_rows, expected_fits))

    cell, paired, agg, extra = build_aggregates(df)
    rho = rank_correlation_record_only(
        diag_df.sort_values("split")["log_event_ratio"].to_numpy(), extra["delta"])

    print("\n=== contrasts (test_y_ll, seed-averaged) ===")
    for name, a, b, role in CONTRASTS:
        v = paired.loc[paired["contrast"] == name, "value"].to_numpy()
        st = summarize_values(v)
        print("  %-28s mean %+.5f median %+.5f  %d/%d positive  std %.5f  "
              "[p10 %+.5f, p90 %+.5f]"
              % (name, st["mean"], st["median"], st["count_positive"], st["n"],
                 st["std"], st["p10"], st["p90"]))
    print("  decomposition identity max |Delta-(A+B)| = %.3e"
          % extra["identity_err"])
    print("  record-only rank corr(log_event_ratio, Delta) = %+.4f "
          "[log(n_test_events / n_train_events); no p-value]" % rho)

    if args.phase == "smoke":
        print("\nSMOKE OK (%d fits, %.1f min)"
              % (len(df), (time.perf_counter() - t0) / 60))
        return 0

    # ── PHASE 4 outputs ────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / (STEM + "_" + RUN_TAG + "_")
    df.to_csv(str(base) + "summary.csv", index=False)
    agg.to_csv(str(base) + "agg.csv", index=False)
    paired.to_csv(str(base) + "paired.csv", index=False)
    diag_df.to_csv(str(base) + "splitdiag.csv", index=False)
    prov.to_csv(str(base) + "provenance.csv", index=False)

    runinfo = pd.DataFrame([dict(
        script="tools/research_audit/run_movielens_userdisjoint_validation.py",
        issue=33, datetime=datetime.now().isoformat(timespec="seconds"),
        git_head=git_head(), branch=git_branch(),
        design_base="e4be01afd1e911ee0d6bed491166258a07af1f0f",
        lineage="expfam/src/experimental/em_runner.py + "
                "model_dual_expfam_consistent.py "
                "(DualExpFamLSMConsistent / DualExpFamLSMPerColumnConsistent; "
                "extra-1/2-なし fixed 系列の派生, per-column は prototype)",
        n_splits=N_SPLITS, n_model_seeds=N_MODEL_SEEDS,
        n_conditions=len(CONDITION_ORDER), n_fits=len(df),
        split_seed_base=SPLIT_SEED_BASE, model_seed_base=MODEL_SEED_BASE,
        k=K, L=L, num_iter=NUM_ITER, family_y=FAMILY_Y,
        numerics_mode=NUMERICS_MODE,
        n_movies=N_MOVIES, n_pairs=N_PAIRS,
        n_train_users=N_TRAIN_USERS, n_test_users=N_TEST_USERS,
        n_unused_users=N_UNUSED_USERS,
        corpus_sha256=gate["corpus_sha256"],
        item_metadata_sha256=gate["meta_sha256"],
        data_zip_sha256=file_sha256(ZIP_PATH),
        decomposition_identity_max_err=extra["identity_err"],
        rank_corr_logeventratio_delta=rho,
        rank_corr_convention="log_event_ratio = log(n_test_events / "
                             "n_train_events), per the pre-registered Issue #33 values; "
                             "record only; no p-value",
        percentile_method="numpy linear",
        python_version=sys.version.split()[0],
        platform=__import__("platform").platform(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        pandas_version=pd.__version__,
        requirements_sha256=file_sha256(_ROOT / "requirements.txt"),
        requirements_dev_sha256=file_sha256(_ROOT / "requirements-dev.txt"),
        total_runtime_s=round(time.perf_counter() - t0, 1),
    )])
    runinfo.to_csv(str(base) + "runinfo.csv", index=False)

    make_figures(cell, paired, extra)
    print("\nFULL OK (%d fits, %.1f min)"
          % (len(df), (time.perf_counter() - t0) / 60))
    print("wrote: " + str(OUT_DIR))
    print("wrote: " + str(FIG_DIR))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StopCondition as exc:
        print("\n!!! ABSOLUTE STOP CONDITION !!!")
        print(str(exc))
        sys.exit(3)
    except ProvenanceError as exc:
        print("\n!!! PROVENANCE ERROR (STOP) !!!")
        print(str(exc))
        sys.exit(4)
