"""Leakage-safe held-out K-selection harness for Issue #41.

The module keeps EM behind explicit, validated boundaries.  Importing it and
running ``--validate-only`` cannot import or call the EM runner.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPFAM_SRC = ROOT / "expfam" / "src"
EXPERIMENTAL = ROOT / "expfam" / "src" / "experimental"
if str(EXPERIMENTAL) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTAL))

from eval_utils import make_pair_split  # noqa: E402
from objective_consistent_numerics import bernoulli_log_likelihood  # noqa: E402


FAMILY_X = "poisson"
FAMILY_Y = "bernoulli"
K_TRUE = 3
N_NODES = 75
N_FEATURES = 15
TEST_RATIO = 0.20
L_SAMPLES = 5
NUM_ITER = 8
NUMERICS_MODE = "consistent"
K_CANDIDATES = tuple(range(1, 8))
SMOKE_K_CANDIDATES = (2, 3, 4)
START_LABELS = (1, 2)
TIE_TOLERANCE = np.float64(1e-12)
CANARY_ATOL = np.float64(1e-12)
CANARY_RTOL = np.float64(1e-10)

DATA_SEED_BASE = 41000
SPLIT_SEED_BASE = 42000
MODEL_SEED_BASE = 43000


class HarnessStop(RuntimeError):
    """A blocking preflight/leakage condition; callers must stop globally."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessStop(message)


def stable_array_hash(*arrays: np.ndarray) -> str:
    """SHA-256 over shape, dtype, and contiguous bytes.

    This provenance hash is runtime/representation sensitive.  It is not a
    cross-endian canonical hash of logical array values.
    """

    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        shape = json.dumps(array.shape, separators=(",", ":")).encode("ascii")
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(shape).to_bytes(8, "big"))
        digest.update(shape)
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def stable_config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of a canonical JSON configuration."""

    payload = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitDiagnostics:
    n_nodes: int
    train_pairs: int
    test_pairs: int
    min_train_degree: int
    min_test_degree: int


def validate_pair_masks(
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    expected_test_pairs: int,
) -> SplitDiagnostics:
    """Validate a dyad split using PAIR-MASK TOPOLOGY ONLY.

    The API intentionally has no Y argument, so outcomes cannot affect guards.
    """

    train = np.asarray(train_mask)
    test = np.asarray(test_mask)
    _require(train.ndim == 2 and train.shape[0] == train.shape[1], "train mask must be square")
    n_nodes = train.shape[0]
    _require(n_nodes > 0, "empty mask is not allowed")
    _require(test.shape == train.shape, "train/test mask shapes differ")
    _require(train.dtype == np.bool_, "train mask dtype must be bool")
    _require(test.dtype == np.bool_, "test mask dtype must be bool")
    _require(np.array_equal(train, train.T), "train mask must be symmetric")
    _require(np.array_equal(test, test.T), "test mask must be symmetric")
    _require(not np.any(np.diag(train)), "train mask diagonal must be false")
    _require(not np.any(np.diag(test)), "test mask diagonal must be false")
    _require(not np.any(train & test), "train/test masks overlap")

    off_diagonal = ~np.eye(n_nodes, dtype=bool)
    _require(np.array_equal(train | test, off_diagonal), "off-diagonal union is incomplete")

    upper = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), 1)
    train_pairs = int(np.count_nonzero(train & upper))
    test_pairs = int(np.count_nonzero(test & upper))
    total_pairs = n_nodes * (n_nodes - 1) // 2
    _require(train_pairs + test_pairs == total_pairs, "upper-pair total is incorrect")
    _require(test_pairs == int(expected_test_pairs), "actual test-pair count differs from expected")

    visited = np.zeros(n_nodes, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(train[node] & ~visited):
            visited[neighbor] = True
            stack.append(int(neighbor))
    _require(bool(np.all(visited)), "train-mask graph is disconnected")

    train_degree = np.sum(train, axis=1, dtype=np.int64)
    test_degree = np.sum(test, axis=1, dtype=np.int64)
    _require(int(train_degree.min()) >= 2, "minimum train-mask degree is below 2")
    _require(int(test_degree.min()) >= 1, "minimum test-mask degree is below 1")
    return SplitDiagnostics(
        n_nodes=n_nodes,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        min_train_degree=int(train_degree.min()),
        min_test_degree=int(test_degree.min()),
    )


def heldout_bernoulli_mean_log_score(
    y_test_pairs: np.ndarray,
    eta_test_pairs: np.ndarray,
) -> float:
    """Family-correct held-out plug-in mean log score from raw eta."""

    y = np.asarray(y_test_pairs, dtype=np.float64)
    eta = np.asarray(eta_test_pairs, dtype=np.float64)
    _require(y.ndim == 1 and eta.ndim == 1, "score inputs must be one-dimensional")
    _require(y.shape == eta.shape and y.size > 0, "score inputs must be nonempty and aligned")
    _require(bool(np.all(np.isfinite(y))), "held-out targets must be finite")
    _require(bool(np.all(np.isfinite(eta))), "held-out eta must be finite")
    _require(bool(np.all((y == 0.0) | (y == 1.0))), "Bernoulli targets must be 0 or 1")
    score_each = np.asarray(bernoulli_log_likelihood(y, eta), dtype=np.float64)
    _require(bool(np.all(np.isfinite(score_each))), "Bernoulli score is nonfinite")
    return float(np.mean(score_each, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class EtaPairs:
    n_nodes: int
    rows: np.ndarray
    cols: np.ndarray
    eta: np.ndarray
    test_mask_hash: str


def heldout_raw_eta_pairs(
    Z: np.ndarray,
    w0: float,
    w: float,
    test_mask: np.ndarray,
) -> EtaPairs:
    """Compute raw w0 + w z_i^T z_j and extract upper test dyads."""

    latent = np.asarray(Z, dtype=np.float64)
    mask = np.asarray(test_mask)
    _require(latent.ndim == 2 and latent.shape[0] > 0, "Z must be a nonempty matrix")
    _require(bool(np.all(np.isfinite(latent))), "Z must be finite")
    _require(np.isfinite(w0) and np.isfinite(w), "w0 and w must be finite")
    _require(mask.dtype == np.bool_, "test mask dtype must be bool")
    _require(mask.shape == (latent.shape[0], latent.shape[0]), "test mask shape does not match Z")
    _require(np.array_equal(mask, mask.T), "test mask must be symmetric")
    _require(not np.any(np.diag(mask)), "test mask diagonal must be false")
    eta_matrix = np.float64(w0) + np.float64(w) * (latent @ latent.T)
    rows, cols = np.where(np.triu(mask, 1))
    _require(rows.size > 0, "test mask contains no upper-triangle pairs")
    values = eta_matrix[rows, cols]
    _require(bool(np.all(np.isfinite(values))), "raw held-out eta is nonfinite")
    return EtaPairs(
        latent.shape[0],
        _readonly_copy(rows, np.int64),
        _readonly_copy(cols, np.int64),
        _readonly_copy(values, np.float64),
        stable_array_hash(mask),
    )


def _validate_pair_coordinates(
    n_nodes: int,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    label: str,
) -> None:
    _require(n_nodes > 0, f"{label} n_nodes must be positive")
    _require(rows.dtype == np.int64 and cols.dtype == np.int64, f"{label} indices must be int64")
    _require(rows.ndim == cols.ndim == values.ndim == 1, f"{label} fields must be one-dimensional")
    _require(rows.size == cols.size == values.size and rows.size > 0, f"{label} fields are not aligned")
    _require(bool(np.all((0 <= rows) & (rows < n_nodes))), f"{label} rows are out of range")
    _require(bool(np.all((0 <= cols) & (cols < n_nodes))), f"{label} cols are out of range")
    _require(bool(np.all(rows < cols)), f"{label} must contain upper-triangle pairs only")
    pairs = list(zip(rows.tolist(), cols.tolist(), strict=True))
    _require(len(pairs) == len(set(pairs)), f"{label} contains duplicate pairs")


def score_heldout_bernoulli(target: ScoreOnlyTarget, eta_pairs: EtaPairs) -> float:
    """Score raw eta only after exact held-out pair/provenance alignment."""

    _require(type(target) is ScoreOnlyTarget, "scorer requires ScoreOnlyTarget")
    _require(type(eta_pairs) is EtaPairs, "scorer requires EtaPairs")
    _validate_pair_coordinates(target.n_nodes, target.rows, target.cols, target.values, "target")
    _validate_pair_coordinates(eta_pairs.n_nodes, eta_pairs.rows, eta_pairs.cols, eta_pairs.eta, "eta")
    _require(target.n_nodes == eta_pairs.n_nodes, "target/eta n_nodes mismatch")
    _require(np.array_equal(target.rows, eta_pairs.rows), "target/eta rows mismatch")
    _require(np.array_equal(target.cols, eta_pairs.cols), "target/eta cols mismatch")
    _require(target.test_mask_hash == eta_pairs.test_mask_hash, "test mask hash mismatch")
    return heldout_bernoulli_mean_log_score(target.values, eta_pairs.eta)


def _readonly_copy(value: np.ndarray, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class TrainingYValues:
    n_nodes: int
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    train_mask_hash: str
    provenance_version: str
    _authority: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ScoreOnlyTarget:
    n_nodes: int
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    test_mask_hash: str


@dataclass(frozen=True, slots=True)
class FitPayload:
    Y_fit: np.ndarray
    train_mask: np.ndarray
    payload_hash: str
    train_mask_hash: str
    expected_shape: tuple[int, int]
    expected_dtype: str
    provenance_version: str
    canary_value: int
    _authority: Any = field(repr=False, compare=False)


_TRAINING_Y_AUTHORITY = object()
_FIT_PAYLOAD_AUTHORITY = object()
TRAINING_Y_PROVENANCE = "training-y-upper-pairs-v1"
FIT_PAYLOAD_PROVENANCE = "masked-fit-matrix-v1"


def _extract_bernoulli_pairs(
    Y: np.ndarray, mask: np.ndarray, target_type: type[TrainingYValues] | type[ScoreOnlyTarget]
) -> TrainingYValues | ScoreOnlyTarget:
    matrix = np.asarray(Y, dtype=np.float64)
    pair_mask = np.asarray(mask)
    _require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "Y must be square")
    _require(np.array_equal(matrix, matrix.T), "Y must be symmetric")
    _require(pair_mask.dtype == np.bool_ and pair_mask.shape == matrix.shape, "pair mask is invalid")
    _require(np.array_equal(pair_mask, pair_mask.T), "pair mask must be symmetric")
    _require(not np.any(np.diag(pair_mask)), "pair mask diagonal must be false")
    rows, cols = np.where(np.triu(pair_mask, 1))
    values = matrix[rows, cols]
    _require(bool(np.all(np.isfinite(values))), "selected Bernoulli values must be finite")
    _require(bool(np.all((values == 0.0) | (values == 1.0))), "selected values are outside Bernoulli support")
    arguments: list[Any] = [
        matrix.shape[0],
        _readonly_copy(rows, np.int64),
        _readonly_copy(cols, np.int64),
        _readonly_copy(values, np.float64),
    ]
    if target_type is ScoreOnlyTarget:
        arguments.append(stable_array_hash(pair_mask))
    else:
        arguments.extend(
            [
                stable_array_hash(pair_mask),
                TRAINING_Y_PROVENANCE,
                _TRAINING_Y_AUTHORITY,
            ]
        )
    return target_type(*arguments)


def make_training_y_values(Y: np.ndarray, train_mask: np.ndarray) -> TrainingYValues:
    """Copy only training dyads into a fit-side typed view."""

    result = _extract_bernoulli_pairs(Y, train_mask, TrainingYValues)
    assert isinstance(result, TrainingYValues)
    return result


def make_score_only_target(Y: np.ndarray, test_mask: np.ndarray) -> ScoreOnlyTarget:
    """Copy only held-out dyads into a scoring-only typed view."""

    result = _extract_bernoulli_pairs(Y, test_mask, ScoreOnlyTarget)
    assert isinstance(result, ScoreOnlyTarget)
    return result


def build_fit_payload(
    training_values: TrainingYValues,
    train_mask: np.ndarray,
    masked_canary_value: int,
) -> FitPayload:
    """Build fit Y without accepting a held-out target object."""

    _require(type(training_values) is TrainingYValues, "fit builder requires TrainingYValues")
    _require(training_values._authority is _TRAINING_Y_AUTHORITY, "training Y provenance is unauthorized")
    _require(training_values.provenance_version == TRAINING_Y_PROVENANCE, "training Y provenance version changed")
    _require(masked_canary_value in (0, 1), "Bernoulli canary must be 0 or 1")
    train = np.asarray(train_mask)
    n_nodes = training_values.n_nodes
    _require(train.dtype == np.bool_ and train.shape == (n_nodes, n_nodes), "train mask is invalid")
    train_mask_hash = stable_array_hash(train)
    _require(training_values.train_mask_hash == train_mask_hash, "training Y train-mask provenance mismatch")
    _require(np.array_equal(train, train.T) and not np.any(np.diag(train)), "train mask topology is invalid")
    expected_rows, expected_cols = np.where(np.triu(train, 1))
    _require(
        np.array_equal(training_values.rows, expected_rows)
        and np.array_equal(training_values.cols, expected_cols),
        "training pair indices do not match train mask",
    )
    _require(bool(np.all(np.isfinite(training_values.values))), "training values must be finite")
    _require(
        bool(np.all((training_values.values == 0.0) | (training_values.values == 1.0))),
        "training values are outside Bernoulli support",
    )
    payload = np.full((n_nodes, n_nodes), float(masked_canary_value), dtype=np.float64)
    np.fill_diagonal(payload, 0.0)
    payload[expected_rows, expected_cols] = training_values.values
    payload[expected_cols, expected_rows] = training_values.values
    payload = _readonly_copy(payload, np.float64)
    train_copy = _readonly_copy(train, np.bool_)
    return FitPayload(
        Y_fit=payload,
        train_mask=train_copy,
        payload_hash=stable_array_hash(payload),
        train_mask_hash=stable_array_hash(train_copy),
        expected_shape=(n_nodes, n_nodes),
        expected_dtype=np.dtype(np.float64).str,
        provenance_version=FIT_PAYLOAD_PROVENANCE,
        canary_value=masked_canary_value,
        _authority=_FIT_PAYLOAD_AUTHORITY,
    )


def build_two_canary_payloads(
    training_values: TrainingYValues,
    train_mask: np.ndarray,
) -> tuple[FitPayload, FitPayload]:
    """Return finite Bernoulli-support-valid A=0 and B=1 masked payloads."""

    return (
        build_fit_payload(training_values, train_mask, 0),
        build_fit_payload(training_values, train_mask, 1),
    )


@dataclass(frozen=True, slots=True)
class FitCallEvidence:
    fit_y_object_id: int
    fit_y_hash: str
    train_mask_hash: str


@dataclass(frozen=True, slots=True)
class FrozenFitConfig:
    """The complete allowlist of values that may cross the fit boundary."""

    family_x: str
    family_y: str
    k_est: int
    L: int
    num_iter: int
    seed: int
    numerics_mode: str
    compute_strict_Q: bool = True
    verbose: bool = False
    validate_support: bool = True
    allow_support_mismatch: bool = False
    mstep_q_diagnostic: bool = True
    compute_clip_diagnostic: bool = False


def _reject_forbidden_fit_objects(
    root: Any,
    targets: Sequence[ScoreOnlyTarget],
) -> None:
    """Defense-in-depth for test adapters; sealed provenance is primary."""

    target_arrays = tuple(target.values for target in targets)
    target_ids = frozenset(id(target) for target in targets)
    visited: set[int] = set()

    def visit(value: Any) -> None:
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        _require(value_id not in target_ids, "ScoreOnlyTarget reached fit boundary")
        _require(not isinstance(value, ScoreOnlyTarget), "ScoreOnlyTarget reached fit boundary")
        if isinstance(value, np.ndarray):
            for target_array in target_arrays:
                _require(value is not target_array, "target.values reached fit boundary")
                try:
                    shares_target_memory = bool(np.shares_memory(value, target_array))
                except (TypeError, ValueError):
                    shares_target_memory = False
                _require(not shares_target_memory, "target.values alias reached fit boundary")
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic, Path, type)):
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for item_field in fields(value):
                visit(getattr(value, item_field.name))
            return
        if isinstance(value, functools.partial):
            visit(value.func)
            visit(value.args)
            visit(value.keywords)
        function_defaults = getattr(value, "__defaults__", None)
        if function_defaults is not None:
            visit(function_defaults)
        function_kwdefaults = getattr(value, "__kwdefaults__", None)
        if function_kwdefaults is not None:
            visit(function_kwdefaults)
        closure = getattr(value, "__closure__", None)
        if closure is not None:
            for cell in closure:
                try:
                    visit(cell.cell_contents)
                except ValueError:
                    continue
        bound_owner = getattr(value, "__self__", None)
        if bound_owner is not None and bound_owner is not value:
            visit(bound_owner)
        try:
            attributes = vars(value)
        except TypeError:
            return
        for item in attributes.values():
            visit(item)

    visit(root)


_FIT_INVOCATION_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class _AuthorizedFitInvocation:
    X: np.ndarray
    Y: np.ndarray
    train_mask: np.ndarray
    config: FrozenFitConfig
    _authority: Any = field(repr=False, compare=False)

    def keyword_arguments(self) -> dict[str, Any]:
        _require(self._authority is _FIT_INVOCATION_AUTHORITY, "fit invocation provenance is unauthorized")
        return {
            "X": self.X,
            "Y": self.Y,
            "train_mask": self.train_mask,
            "family_x": self.config.family_x,
            "family_y": self.config.family_y,
            "k": self.config.k_est,
            "L": self.config.L,
            "num_iter": self.config.num_iter,
            "seed": self.config.seed,
            "numerics_mode": self.config.numerics_mode,
            "compute_strict_Q": self.config.compute_strict_Q,
            "verbose": self.config.verbose,
            "validate_support": self.config.validate_support,
            "allow_support_mismatch": self.config.allow_support_mismatch,
            "mstep_q_diagnostic": self.config.mstep_q_diagnostic,
            "compute_clip_diagnostic": self.config.compute_clip_diagnostic,
        }


class AuthorizedEMFitAdapter:
    """Sealed production adapter; it has no injectable callable or runner."""

    __slots__ = ()

    def fit(self, invocation: _AuthorizedFitInvocation) -> CanaryFitResult:
        _require(type(invocation) is _AuthorizedFitInvocation, "production fit invocation is unauthorized")
        arguments = invocation.keyword_arguments()
        import em_runner  # noqa: PLC0415

        return run_em_with_initialization_capture(em_runner, **arguments)


_TEST_ADAPTER_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class _TestAuthorizedFitAdapter:
    callback: Callable[..., Any] = field(repr=False, compare=False)
    _authority: Any = field(repr=False, compare=False)

    def fit(self, invocation: _AuthorizedFitInvocation) -> Any:
        _require(type(invocation) is _AuthorizedFitInvocation, "test fit invocation is unauthorized")
        return self.callback(**invocation.keyword_arguments())


def _make_test_fit_adapter(
    callback: Callable[..., Any],
    *,
    score_targets: Sequence[ScoreOnlyTarget],
) -> _TestAuthorizedFitAdapter:
    """Static-test-only factory; unavailable through the production CLI."""

    _require(callable(callback), "test adapter callback must be callable")
    _reject_forbidden_fit_objects(callback, score_targets)
    return _TestAuthorizedFitAdapter(callback, _TEST_ADAPTER_AUTHORITY)


_BOUNDARY_CONSTRUCTION_AUTHORITY = object()


class FitCallBoundary:
    """Own every production fit input; per-fit callers supply only canary 0/1."""

    __slots__ = (
        "_X",
        "_training_values",
        "_train_mask",
        "_test_mask",
        "_config",
        "_adapter",
        "_test_only",
        "x_hash",
        "training_y_hash",
        "train_mask_hash",
        "test_mask_hash",
        "last_evidence",
    )

    def __init__(
        self,
        prepared: PreparedTrainingData,
        config: FrozenFitConfig,
        adapter: AuthorizedEMFitAdapter | _TestAuthorizedFitAdapter,
        *,
        test_only: bool,
        authority: Any,
    ) -> None:
        _require(authority is _BOUNDARY_CONSTRUCTION_AUTHORITY, "boundary construction is unauthorized")
        self._X = _readonly_copy(prepared.X, np.float64)
        training = prepared.training_values
        self._training_values = TrainingYValues(
            training.n_nodes,
            _readonly_copy(training.rows, np.int64),
            _readonly_copy(training.cols, np.int64),
            _readonly_copy(training.values, np.float64),
            training.train_mask_hash,
            training.provenance_version,
            _TRAINING_Y_AUTHORITY,
        )
        self._train_mask = _readonly_copy(prepared.train_mask, np.bool_)
        self._test_mask = _readonly_copy(prepared.test_mask, np.bool_)
        self._config = config
        self._adapter = adapter
        self._test_only = test_only
        self.x_hash = stable_array_hash(self._X)
        self.training_y_hash = stable_array_hash(
            self._training_values.rows,
            self._training_values.cols,
            self._training_values.values,
        )
        self.train_mask_hash = stable_array_hash(self._train_mask)
        self.test_mask_hash = stable_array_hash(self._test_mask)
        self.last_evidence: FitCallEvidence | None = None

    @classmethod
    def from_preflight(
        cls,
        prepared: PreparedTrainingData,
        preflight: CanaryPreflight,
        config: FrozenFitConfig,
        adapter: AuthorizedEMFitAdapter,
    ) -> FitCallBoundary:
        cls._validate_prepared(prepared, preflight, config)
        _require(type(adapter) is AuthorizedEMFitAdapter, "production fit adapter is unauthorized")
        return cls(
            prepared,
            config,
            adapter,
            test_only=False,
            authority=_BOUNDARY_CONSTRUCTION_AUTHORITY,
        )

    @classmethod
    def _from_preflight_test_only(
        cls,
        prepared: PreparedTrainingData,
        preflight: CanaryPreflight,
        config: FrozenFitConfig,
        adapter: _TestAuthorizedFitAdapter,
    ) -> FitCallBoundary:
        cls._validate_prepared(prepared, preflight, config)
        _require(type(adapter) is _TestAuthorizedFitAdapter, "test fit adapter is unauthorized")
        _require(adapter._authority is _TEST_ADAPTER_AUTHORITY, "test fit adapter provenance is unauthorized")
        return cls(
            prepared,
            config,
            adapter,
            test_only=True,
            authority=_BOUNDARY_CONSTRUCTION_AUTHORITY,
        )

    @staticmethod
    def _validate_prepared(
        prepared: PreparedTrainingData,
        preflight: CanaryPreflight,
        config: FrozenFitConfig,
    ) -> None:
        _require(type(prepared) is PreparedTrainingData, "prepared training data type is invalid")
        _require(prepared._authority is _PREPARED_DATA_AUTHORITY, "prepared training data is unauthorized")
        _require(prepared.provenance_version == PREPARED_DATA_PROVENANCE, "prepared data version changed")
        _validate_canary_preflight(preflight, prepared.train_mask, prepared.test_mask)
        _require(type(config) is FrozenFitConfig, "fit config type is invalid")
        _require(stable_array_hash(prepared.X) == prepared.x_hash, "prepared X hash changed")
        training = prepared.training_values
        _require(type(training) is TrainingYValues, "prepared training Y type changed")
        _require(training._authority is _TRAINING_Y_AUTHORITY, "prepared training Y authority changed")
        _require(
            stable_array_hash(training.rows, training.cols, training.values)
            == prepared.training_y_hash,
            "prepared training Y hash changed",
        )
        _require(stable_array_hash(prepared.train_mask) == prepared.train_mask_hash, "prepared train mask changed")
        _require(stable_array_hash(prepared.test_mask) == prepared.test_mask_hash, "prepared test mask changed")
        expected_fit_provenance = stable_config_hash(
            {
                "provenance_version": prepared.provenance_version,
                "x_hash": prepared.x_hash,
                "training_y_hash": prepared.training_y_hash,
                "train_mask_hash": prepared.train_mask_hash,
                "test_mask_hash": prepared.test_mask_hash,
            }
        )
        _require(
            expected_fit_provenance == prepared.fit_provenance_hash,
            "prepared fit provenance hash changed",
        )

    def call(self, canary_value: int) -> Any:
        """Build boundary-owned Y and run; there is deliberately no X/Y argument."""

        payload = build_fit_payload(self._training_values, self._train_mask, canary_value)
        _require(type(payload) is FitPayload, "boundary fit payload type changed")
        _require(payload._authority is _FIT_PAYLOAD_AUTHORITY, "boundary fit payload is unauthorized")
        _require(payload.train_mask_hash == self.train_mask_hash, "boundary train mask binding changed")
        actual_hash = stable_array_hash(payload.Y_fit)
        _require(actual_hash == payload.payload_hash, "boundary fit payload hash changed")
        self.last_evidence = FitCallEvidence(
            id(payload.Y_fit), actual_hash, payload.train_mask_hash
        )
        invocation = _AuthorizedFitInvocation(
            X=self._X,
            Y=payload.Y_fit,
            train_mask=self._train_mask,
            config=self._config,
            _authority=_FIT_INVOCATION_AUTHORITY,
        )
        if self._test_only:
            _require(type(self._adapter) is _TestAuthorizedFitAdapter, "test adapter binding changed")
        else:
            _require(type(self._adapter) is AuthorizedEMFitAdapter, "production adapter binding changed")
        return self._adapter.fit(invocation)


@dataclass(frozen=True, slots=True)
class InitializationSnapshot:
    Z: np.ndarray
    F: np.ndarray
    w0: float
    w: float
    sigma_y: float | None


def snapshot_initialization(model: Any) -> InitializationSnapshot:
    """Copy informed initialization immediately before the first E-step."""

    params = model.params
    sigma_y = getattr(model, "sigma_y", None)
    return InitializationSnapshot(
        Z=_readonly_copy(params["Z"], np.float64),
        F=_readonly_copy(params["F"], np.float64),
        w0=float(params["w0"]),
        w=float(params["w"]),
        sigma_y=None if sigma_y is None else float(sigma_y),
    )


@dataclass(frozen=True, slots=True)
class CanaryFitResult:
    initialization: InitializationSnapshot
    Z: np.ndarray
    F: np.ndarray
    w0: float
    w: float
    sigma_y: float | None
    Q_strict: float
    train_objective_diagnostics: Any
    internal_retry: int
    q_failure: bool
    warnings: tuple[str, ...]
    nan_occurred: bool


def run_em_with_initialization_capture(runner_module: Any, **fit_arguments: Any) -> CanaryFitResult:
    """Call an injected runner while observing informed init exactly once.

    The runner's local ``build_model`` binding is temporarily wrapped.  Each
    returned model's existing ``calc_eta_newton`` method is observed on its
    first call, after informed Y/X initialization and immediately before the
    first E-step calculation.  The wrapper does not call initialization, draw
    random numbers, or alter arguments.  The real runner is never imported by
    this module; the caller must explicitly inject it.
    """

    original_build_model = runner_module.build_model
    build_count = 0
    snapshots: list[InitializationSnapshot] = []

    def build_model_spy(*args: Any, **kwargs: Any) -> Any:
        nonlocal build_count
        model = original_build_model(*args, **kwargs)
        build_count += 1
        original_calc_eta_newton = model.calc_eta_newton
        observed = False

        def calc_eta_newton_spy(*calc_args: Any, **calc_kwargs: Any) -> Any:
            nonlocal observed
            if not observed:
                if not snapshots:
                    snapshots.append(snapshot_initialization(model))
                observed = True
            return original_calc_eta_newton(*calc_args, **calc_kwargs)

        model.calc_eta_newton = calc_eta_newton_spy
        return model

    runner_module.build_model = build_model_spy
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = runner_module.run_em_experimental(**fit_arguments)
    finally:
        runner_module.build_model = original_build_model

    _require(build_count > 0, "runner did not build a model")
    _require(len(snapshots) == 1, "initialization snapshot was not captured exactly once")
    final_snapshot = snapshots[0]
    return CanaryFitResult(
        initialization=final_snapshot,
        Z=_readonly_copy(raw["Z_est"], np.float64),
        F=_readonly_copy(raw["F"], np.float64),
        w0=float(raw["w0"]),
        w=float(raw["w"]),
        sigma_y=None if raw.get("sigma_y_est") is None else float(raw["sigma_y_est"]),
        Q_strict=float(raw["Q_strict"]),
        train_objective_diagnostics=tuple(raw.get("mstep_q_history", ())),
        internal_retry=build_count - 1,
        q_failure=bool(raw.get("q_bic_failed", False)),
        warnings=tuple(f"{item.category.__name__}: {item.message}" for item in caught),
        nan_occurred=bool(raw.get("nan_occurred", False)),
    )


@dataclass(frozen=True, slots=True)
class ManifestRow:
    replicate: int
    data_seed: int
    split_seed: int
    k: int
    start: int
    model_seed: int


@dataclass(frozen=True, slots=True)
class SplitPlan:
    replicate: int
    split_seed: int
    expected_test_pairs: int
    train_mask: np.ndarray
    test_mask: np.ndarray
    diagnostics: SplitDiagnostics


_PREFLIGHT_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class CanaryPreflight:
    replicate: int
    train_mask_hash: str
    test_mask_hash: str
    _authority: Any = field(repr=False, compare=False)


_PREPARED_DATA_AUTHORITY = object()
PREPARED_DATA_PROVENANCE = "heldout-training-only-data-v2"


class PreparedTrainingData:
    """Immutable fit-only data; held-out outcome values are structurally absent."""

    __slots__ = (
        "_X",
        "_training_values",
        "_train_mask",
        "_test_mask",
        "x_hash",
        "training_y_hash",
        "fit_provenance_hash",
        "train_mask_hash",
        "test_mask_hash",
        "x_shape",
        "x_dtype",
        "provenance_version",
        "_authority",
    )

    def __init__(
        self,
        *,
        X: np.ndarray,
        training_values: TrainingYValues,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        authority: Any,
    ) -> None:
        _require(authority is _PREPARED_DATA_AUTHORITY, "prepared data construction is unauthorized")
        object.__setattr__(self, "_X", _readonly_copy(X, np.float64))
        object.__setattr__(
            self,
            "_training_values",
            TrainingYValues(
                training_values.n_nodes,
                _readonly_copy(training_values.rows, np.int64),
                _readonly_copy(training_values.cols, np.int64),
                _readonly_copy(training_values.values, np.float64),
                training_values.train_mask_hash,
                training_values.provenance_version,
                _TRAINING_Y_AUTHORITY,
            ),
        )
        object.__setattr__(self, "_train_mask", _readonly_copy(train_mask, np.bool_))
        object.__setattr__(self, "_test_mask", _readonly_copy(test_mask, np.bool_))
        object.__setattr__(self, "x_hash", stable_array_hash(self._X))
        object.__setattr__(
            self,
            "training_y_hash",
            stable_array_hash(
                self._training_values.rows,
                self._training_values.cols,
                self._training_values.values,
            ),
        )
        object.__setattr__(self, "train_mask_hash", stable_array_hash(self._train_mask))
        object.__setattr__(self, "test_mask_hash", stable_array_hash(self._test_mask))
        object.__setattr__(
            self,
            "fit_provenance_hash",
            stable_config_hash(
                {
                    "provenance_version": PREPARED_DATA_PROVENANCE,
                    "x_hash": self.x_hash,
                    "training_y_hash": self.training_y_hash,
                    "train_mask_hash": self.train_mask_hash,
                    "test_mask_hash": self.test_mask_hash,
                }
            ),
        )
        object.__setattr__(self, "x_shape", self._X.shape)
        object.__setattr__(self, "x_dtype", self._X.dtype.str)
        object.__setattr__(self, "provenance_version", PREPARED_DATA_PROVENANCE)
        object.__setattr__(self, "_authority", authority)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise AttributeError("PreparedTrainingData is immutable")

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def training_values(self) -> TrainingYValues:
        return self._training_values

    @property
    def train_mask(self) -> np.ndarray:
        return self._train_mask

    @property
    def test_mask(self) -> np.ndarray:
        return self._test_mask


def prepare_training_data(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    preflight: CanaryPreflight,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> PreparedTrainingData:
    """Freeze generated data once, before ScoreOnlyTarget exists."""

    _validate_canary_preflight(preflight, train_mask, test_mask)
    _require(type(X) is np.ndarray and X.dtype == np.float64, "raw X must be float64 ndarray")
    _require(type(Y) is np.ndarray and Y.dtype == np.float64, "raw Y must be float64 ndarray")
    _require(X.ndim == 2 and X.shape[0] == Y.shape[0], "raw X/Y shapes are incompatible")
    _require(bool(np.all(np.isfinite(X))), "raw X must be finite")
    training = make_training_y_values(Y, train_mask)
    return PreparedTrainingData(
        X=X,
        training_values=training,
        train_mask=train_mask,
        test_mask=test_mask,
        authority=_PREPARED_DATA_AUTHORITY,
    )


def authorize_canary_preflight(split_plan: SplitPlan) -> CanaryPreflight:
    """Issue a non-forgeable-in-normal-use token for an already valid split."""

    _require(type(split_plan) is SplitPlan, "canary requires a validated SplitPlan")
    diagnostics = validate_pair_masks(
        split_plan.train_mask,
        split_plan.test_mask,
        split_plan.expected_test_pairs,
    )
    _require(diagnostics == split_plan.diagnostics, "split diagnostics changed after preflight")
    return CanaryPreflight(
        replicate=split_plan.replicate,
        train_mask_hash=stable_array_hash(split_plan.train_mask),
        test_mask_hash=stable_array_hash(split_plan.test_mask),
        _authority=_PREFLIGHT_AUTHORITY,
    )


def expected_model_seed(replicate: int, k: int, start: int) -> int:
    return MODEL_SEED_BASE + replicate * 1000 + k * 10 + start


def build_manifest(
    replicates: Sequence[int],
    k_candidates: Sequence[int],
    starts: Sequence[int] = START_LABELS,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for replicate in replicates:
        data_seed = DATA_SEED_BASE + int(replicate)
        split_seed = SPLIT_SEED_BASE + int(replicate)
        for k in k_candidates:
            for start in starts:
                rows.append(
                    ManifestRow(
                        replicate=int(replicate),
                        data_seed=data_seed,
                        split_seed=split_seed,
                        k=int(k),
                        start=int(start),
                        model_seed=expected_model_seed(int(replicate), int(k), int(start)),
                    )
                )
    return rows


def validate_manifest(
    rows: Sequence[ManifestRow],
    replicates: Sequence[int],
    k_candidates: Sequence[int],
    starts: Sequence[int] = START_LABELS,
) -> None:
    expected_keys = {
        (int(replicate), int(k), int(start))
        for replicate in replicates
        for k in k_candidates
        for start in starts
    }
    actual_keys = [(row.replicate, row.k, row.start) for row in rows]
    _require(len(actual_keys) == len(set(actual_keys)), "duplicate manifest key")
    actual_key_set = set(actual_keys)
    _require(actual_key_set == expected_keys, "manifest expected key set differs from actual key set")
    model_seeds = [row.model_seed for row in rows]
    _require(len(model_seeds) == len(set(model_seeds)), "duplicate model seed")

    for row in rows:
        _require(row.data_seed == DATA_SEED_BASE + row.replicate, "K-specific or invalid data seed")
        _require(row.split_seed == SPLIT_SEED_BASE + row.replicate, "K-specific or invalid split seed")
        _require(
            row.model_seed == expected_model_seed(row.replicate, row.k, row.start),
            "model seed violates the fixed (replicate, K, start) convention",
        )


def preflight_all_splits(
    replicates: Sequence[int],
    *,
    n_nodes: int = N_NODES,
    test_ratio: float = TEST_RATIO,
) -> list[SplitPlan]:
    """Generate every planned split, then require every topology guard to pass.

    There is no redraw, replacement, repair, ratio change, or replicate-drop
    branch.  A single invalid planned split raises ``HarnessStop`` globally.
    """

    _require(len(replicates) == len(set(replicates)), "duplicate dataset replicate")
    expected_test_pairs = _expected_test_pairs(n_nodes, test_ratio)
    generated: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for replicate in replicates:
        split_seed = SPLIT_SEED_BASE + int(replicate)
        train_mask, test_mask = make_pair_split(n_nodes, test_ratio, split_seed)
        generated.append((int(replicate), split_seed, train_mask, test_mask))

    plans: list[SplitPlan] = []
    for replicate, split_seed, train_mask, test_mask in generated:
        diagnostics = validate_pair_masks(train_mask, test_mask, expected_test_pairs)
        plans.append(
            SplitPlan(
                replicate=replicate,
                split_seed=split_seed,
                expected_test_pairs=expected_test_pairs,
                train_mask=_readonly_copy(train_mask, np.bool_),
                test_mask=_readonly_copy(test_mask, np.bool_),
                diagnostics=diagnostics,
            )
        )
    return plans


@dataclass(frozen=True, slots=True)
class ComparabilityRow:
    manifest: ManifestRow
    x_hash: str
    training_y_hash: str
    preprocessing_hash: str
    train_mask_hash: str
    test_mask_hash: str
    fit_provenance_hash: str
    target_topology_hash: str
    score_config_hash: str


def _validate_cross_k_internal_consistency(
    rows: Sequence[ComparabilityRow],
    expected_manifest: Sequence[ManifestRow],
) -> None:
    """Require manifest completeness, seed rules, and row-wise consistency."""

    _require(bool(rows), "comparability manifest is empty")
    actual_keys = [
        (row.manifest.replicate, row.manifest.k, row.manifest.start) for row in rows
    ]
    _require(len(actual_keys) == len(set(actual_keys)), "duplicate actual comparability key")
    expected_keys = [
        (row.replicate, row.k, row.start) for row in expected_manifest
    ]
    _require(len(expected_keys) == len(set(expected_keys)), "duplicate expected manifest key")
    _require(set(actual_keys) == set(expected_keys), "actual key set differs from expected frozen manifest")

    expected_by_key = {
        (row.replicate, row.k, row.start): row for row in expected_manifest
    }
    expected_model_seeds = [row.model_seed for row in expected_manifest]
    _require(len(expected_model_seeds) == len(set(expected_model_seeds)), "duplicate expected model seed")
    for row in expected_manifest:
        _require(row.data_seed == DATA_SEED_BASE + row.replicate, "expected manifest has wrong data seed")
        _require(row.split_seed == SPLIT_SEED_BASE + row.replicate, "expected manifest has wrong split seed")
        _require(
            row.model_seed == expected_model_seed(row.replicate, row.k, row.start),
            "expected manifest has wrong model seed",
        )
    actual_model_seeds = [row.manifest.model_seed for row in rows]
    _require(len(actual_model_seeds) == len(set(actual_model_seeds)), "duplicate actual model seed")
    for row in rows:
        key = (row.manifest.replicate, row.manifest.k, row.manifest.start)
        expected = expected_by_key[key]
        _require(row.manifest.data_seed == expected.data_seed, "actual manifest has wrong data seed")
        _require(row.manifest.split_seed == expected.split_seed, "actual manifest has wrong split seed")
        _require(row.manifest.model_seed == expected.model_seed, "actual manifest has wrong model seed")

    invariant_fields = (
        "x_hash",
        "training_y_hash",
        "preprocessing_hash",
        "train_mask_hash",
        "test_mask_hash",
        "fit_provenance_hash",
        "target_topology_hash",
        "score_config_hash",
    )
    by_replicate: dict[int, list[ComparabilityRow]] = {}
    for row in rows:
        by_replicate.setdefault(row.manifest.replicate, []).append(row)
    for replicate_rows in by_replicate.values():
        for field in invariant_fields:
            _require(
                len({getattr(row, field) for row in replicate_rows}) == 1,
                f"cross-K comparability mismatch: {field}",
            )
        _require(
            len({row.manifest.data_seed for row in replicate_rows}) == 1,
            "cross-K comparability mismatch: data seed",
        )
        _require(
            len({row.manifest.split_seed for row in replicate_rows}) == 1,
            "cross-K comparability mismatch: split seed",
        )


def validate_cross_k_comparability(
    rows: Sequence[ComparabilityRow],
    expected_rows: Sequence[ComparabilityRow],
) -> None:
    """Bind complete actual metadata to complete frozen expected metadata."""

    _require(bool(expected_rows), "expected comparability manifest is empty")
    expected_manifest = [row.manifest for row in expected_rows]
    _validate_cross_k_internal_consistency(rows, expected_manifest)
    _validate_cross_k_internal_consistency(expected_rows, expected_manifest)
    _require(len(rows) == len(expected_rows), "comparability row count changed")
    for index, (actual, expected) in enumerate(
        zip(rows, expected_rows, strict=True)
    ):
        for field_name in (field.name for field in fields(ComparabilityRow)):
            _require(
                getattr(actual, field_name) == getattr(expected, field_name),
                f"comparability row {index} differs from frozen expected: {field_name}",
            )


@dataclass(frozen=True, slots=True)
class StartScore:
    k: int
    start: int
    score: np.float64


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_k: int
    mean_scores: Mapping[int, np.float64]
    tie_candidates: tuple[int, ...]


def select_k_from_two_starts(
    rows: Sequence[StartScore],
    k_candidates: Sequence[int],
    starts: Sequence[int] = START_LABELS,
) -> SelectionResult:
    expected_keys = {(int(k), int(start)) for k in k_candidates for start in starts}
    actual_keys = [(int(row.k), int(row.start)) for row in rows]
    _require(len(actual_keys) == len(set(actual_keys)), "duplicate start score key")
    _require(set(actual_keys) == expected_keys, "each K must have exactly two fixed starts")
    _require(len(tuple(starts)) == 2 and len(set(starts)) == 2, "selector requires exactly two starts")
    _require(all(np.isfinite(np.float64(row.score)) for row in rows), "start score is nonfinite")

    mean_scores: dict[int, np.float64] = {}
    for k in k_candidates:
        values = np.asarray([row.score for row in rows if row.k == k], dtype=np.float64)
        _require(values.size == 2, "each K must have exactly two scores")
        mean_scores[int(k)] = np.mean(values, dtype=np.float64)
    best = max(mean_scores.values())
    ties = tuple(sorted(k for k, score in mean_scores.items() if best - score <= TIE_TOLERANCE))
    _require(bool(ties), "selector produced no tie candidates")
    return SelectionResult(min(ties), mean_scores, ties)


def require_no_blocking_failures(failures: Mapping[str, bool]) -> None:
    """Fail globally; no drop, retry, replacement, or seed-rescue path exists."""

    active = sorted(name for name, failed in failures.items() if bool(failed))
    _require(not active, "PILOT GLOBAL STOP: " + ", ".join(active))


@dataclass(frozen=True, slots=True)
class FrozenScoreConfig:
    family_y: str = "bernoulli"
    score_name: str = "heldout_bernoulli_mean_log_score"
    formula_version: str = "raw_eta_logaddexp_v1"
    raw_eta: bool = True
    aggregation: str = "mean_upper_test_pairs"
    test_ratio: float = 0.20
    tie_tolerance: float = 1e-12
    tie_rule: str = "smallest_k"
    upper_triangle_only: bool = True
    probability_clipping: str = "none"
    predict_mu_y: bool = False
    starts_per_k: int = 2


SCORE_CONFIG_FIELDS = tuple(item.name for item in fields(FrozenScoreConfig))


def frozen_score_config() -> FrozenScoreConfig:
    return FrozenScoreConfig()


def score_config_hash(config: FrozenScoreConfig | Mapping[str, Any]) -> str:
    """Hash a complete score contract using canonical JSON."""

    if type(config) is FrozenScoreConfig:
        values = asdict(config)
    else:
        values = dict(config)
    _require(set(values) == set(SCORE_CONFIG_FIELDS), "score config fields are incomplete or unexpected")
    return stable_config_hash(values)


@dataclass(frozen=True, slots=True)
class CanaryInvarianceReport:
    config_hash: str
    fit_payload_a_hash: str
    fit_payload_b_hash: str
    initialization_equal: bool
    final_outputs_equal: bool
    internal_retry: int


def _require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, np.ndarray):
        _require(bool(np.all(np.isfinite(value))), f"{label} is nonfinite")
        return
    if isinstance(value, (float, np.floating)):
        _require(bool(np.isfinite(value)), f"{label} is nonfinite")


def _require_canary_equal(left: Any, right: Any, label: str) -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        _require(left_array.shape == right_array.shape, f"BLOCKING LEAKAGE FAILURE: {label} shape differs")
        _require(
            bool(np.allclose(left_array, right_array, atol=CANARY_ATOL, rtol=CANARY_RTOL, equal_nan=False)),
            f"BLOCKING LEAKAGE FAILURE: {label} differs",
        )
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        _require(set(left) == set(right), f"BLOCKING LEAKAGE FAILURE: {label} keys differ")
        for key in left:
            _require_canary_equal(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        _require(len(left) == len(right), f"BLOCKING LEAKAGE FAILURE: {label} length differs")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _require_canary_equal(left_item, right_item, f"{label}[{index}]")
        return
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        _require(
            bool(np.isclose(left, right, atol=CANARY_ATOL, rtol=CANARY_RTOL, equal_nan=False)),
            f"BLOCKING LEAKAGE FAILURE: {label} differs",
        )
        return
    _require(left == right, f"BLOCKING LEAKAGE FAILURE: {label} differs")


def _validate_canary_preflight(
    preflight: CanaryPreflight,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    _require(type(preflight) is CanaryPreflight, "canary requires validated preflight")
    _require(preflight._authority is _PREFLIGHT_AUTHORITY, "canary preflight is invalid or unvalidated")
    _require(preflight.train_mask_hash == stable_array_hash(train_mask), "canary train mask bypasses preflight")
    _require(preflight.test_mask_hash == stable_array_hash(test_mask), "canary test mask bypasses preflight")


@dataclass(frozen=True, slots=True)
class _CanaryFitPhaseResult:
    result_a: CanaryFitResult
    result_b: CanaryFitResult
    payload_a_hash: str
    payload_b_hash: str


def _run_two_canary_fit_phase(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    config: FrozenFitConfig,
    adapter: AuthorizedEMFitAdapter | _TestAuthorizedFitAdapter,
    test_only: bool,
) -> _CanaryFitPhaseResult:
    """Run A/B from training-only state; raw Y is not part of this API graph."""

    _validate_canary_preflight(preflight, prepared.train_mask, prepared.test_mask)
    _require(config.family_y == FAMILY_Y, "canary family_y freeze changed")
    _require(config.k_est > 0 and config.seed >= 0, "canary K/seed is invalid")
    if test_only:
        _require(type(adapter) is _TestAuthorizedFitAdapter, "test canary requires test adapter")
        boundary = FitCallBoundary._from_preflight_test_only(
            prepared, preflight, config, adapter
        )
    else:
        _require(type(adapter) is AuthorizedEMFitAdapter, "production canary requires production adapter")
        boundary = FitCallBoundary.from_preflight(
            prepared, preflight, config, adapter
        )
    result_a = boundary.call(0)
    payload_a_hash = boundary.last_evidence.fit_y_hash if boundary.last_evidence else ""
    result_b = boundary.call(1)
    payload_b_hash = boundary.last_evidence.fit_y_hash if boundary.last_evidence else ""
    _require(type(result_a) is CanaryFitResult and type(result_b) is CanaryFitResult, "fit callable returned invalid canary result")
    return _CanaryFitPhaseResult(
        result_a, result_b, payload_a_hash, payload_b_hash
    )


def _run_two_canary_falsification(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    score_Y: np.ndarray,
    config: FrozenFitConfig,
    adapter: AuthorizedEMFitAdapter | _TestAuthorizedFitAdapter,
    test_only: bool,
) -> CanaryInvarianceReport:
    """Run A=0/B=1 complete-fit falsification after all preflight gates.

    Numeric tolerances are frozen module constants before any results exist:
    arrays/scalars use ``CANARY_ATOL``/``CANARY_RTOL``; hashes, config, seeds,
    retry/failure/warning states use exact equality.  Tolerances must never be
    relaxed after inspecting canary output.
    """

    fit_phase = _run_two_canary_fit_phase(
        preflight=preflight,
        prepared=prepared,
        config=config,
        adapter=adapter,
        test_only=test_only,
    )
    result_a = fit_phase.result_a
    result_b = fit_phase.result_b
    payload_a_hash = fit_phase.payload_a_hash
    payload_b_hash = fit_phase.payload_b_hash

    for label, result in (("A", result_a), ("B", result_b)):
        _require(result.internal_retry == 0, f"canary {label} internal_retry > 0")
        _require(not result.q_failure, f"canary {label} Q failure")
        _require(not result.nan_occurred, f"canary {label} NaN/nonfinite state")
        _require(not result.warnings, f"canary {label} emitted warnings")
        _require_finite_tree(result.initialization.Z, f"canary {label} init Z")
        _require_finite_tree(result.initialization.F, f"canary {label} init F")
        _require_finite_tree(result.initialization.w0, f"canary {label} init w0")
        _require_finite_tree(result.initialization.w, f"canary {label} init w")
        _require_finite_tree(result.initialization.sigma_y, f"canary {label} init sigma_y")
        _require_finite_tree(result.Z, f"canary {label} Z")
        _require_finite_tree(result.F, f"canary {label} F")
        _require_finite_tree(result.w0, f"canary {label} w0")
        _require_finite_tree(result.w, f"canary {label} w")
        _require_finite_tree(result.sigma_y, f"canary {label} sigma_y")
        _require_finite_tree(result.Q_strict, f"canary {label} Q_strict")
        _require_finite_tree(result.train_objective_diagnostics, f"canary {label} diagnostics")

    _require_canary_equal(result_a.initialization.Z, result_b.initialization.Z, "initialization.Z")
    _require_canary_equal(result_a.initialization.F, result_b.initialization.F, "initialization.F")
    _require_canary_equal(result_a.initialization.w0, result_b.initialization.w0, "initialization.w0")
    _require_canary_equal(result_a.initialization.w, result_b.initialization.w, "initialization.w")
    _require_canary_equal(result_a.initialization.sigma_y, result_b.initialization.sigma_y, "initialization.sigma_y")
    _require_canary_equal(result_a.Z, result_b.Z, "final.Z")
    _require_canary_equal(result_a.F, result_b.F, "final.F")
    _require_canary_equal(result_a.w0, result_b.w0, "final.w0")
    _require_canary_equal(result_a.w, result_b.w, "final.w")
    _require_canary_equal(result_a.sigma_y, result_b.sigma_y, "final.sigma_y")
    _require_canary_equal(result_a.Q_strict, result_b.Q_strict, "Q_strict")
    _require_canary_equal(
        result_a.train_objective_diagnostics,
        result_b.train_objective_diagnostics,
        "train_objective_diagnostics",
    )
    _require(result_a.internal_retry == result_b.internal_retry, "BLOCKING LEAKAGE FAILURE: retry state differs")
    _require(result_a.q_failure == result_b.q_failure, "BLOCKING LEAKAGE FAILURE: Q failure state differs")
    _require(result_a.warnings == result_b.warnings, "BLOCKING LEAKAGE FAILURE: warning state differs")
    _require(result_a.nan_occurred == result_b.nan_occurred, "BLOCKING LEAKAGE FAILURE: NaN state differs")
    # Raw score Y is outside the fit phase API.  Materialize the score-only
    # target only after both fits and every clean/invariance gate pass.
    target = make_score_only_target(score_Y, prepared.test_mask)
    _require(target.test_mask_hash == prepared.test_mask_hash, "score target mask hash mismatch")
    return CanaryInvarianceReport(
        config_hash=stable_config_hash(asdict(config)),
        fit_payload_a_hash=payload_a_hash,
        fit_payload_b_hash=payload_b_hash,
        initialization_equal=True,
        final_outputs_equal=True,
        internal_retry=0,
    )


def run_two_canary_falsification(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    score_Y: np.ndarray,
    config: FrozenFitConfig,
    adapter: AuthorizedEMFitAdapter,
) -> CanaryInvarianceReport:
    """Production canary entry point: only the sealed EM adapter is accepted."""

    _require(type(adapter) is AuthorizedEMFitAdapter, "production canary requires production adapter")
    return _run_two_canary_falsification(
        preflight=preflight,
        prepared=prepared,
        score_Y=score_Y,
        config=config,
        adapter=adapter,
        test_only=False,
    )


def _run_two_canary_falsification_test_only(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    score_Y: np.ndarray,
    config: FrozenFitConfig,
    adapter: _TestAuthorizedFitAdapter,
) -> CanaryInvarianceReport:
    """Pure static-test entry point; production CLI has no reference to it."""

    return _run_two_canary_falsification(
        preflight=preflight,
        prepared=prepared,
        score_Y=score_Y,
        config=config,
        adapter=adapter,
        test_only=True,
    )


@dataclass(frozen=True, slots=True)
class SmokeFitResult:
    k: int
    start: int
    data_seed: int
    split_seed: int
    model_seed: int
    fit_status: str
    heldout_mean_log_score: float
    Q_strict: float
    internal_retry: int
    warnings: tuple[str, ...]
    warning_count: int
    q_failure: bool
    nan_occurred: bool
    finite_state: bool
    x_hash: str
    training_y_hash: str
    train_mask_hash: str
    test_mask_hash: str
    fit_provenance_hash: str
    target_topology_hash: str
    score_target_hash: str
    score_config_hash: str
    fit_config_hash: str


@dataclass(frozen=True, slots=True)
class SmokeKSummary:
    k: int
    start_1_score: float
    start_2_score: float
    mean_score: float


@dataclass(frozen=True, slots=True)
class SmokeSelectionReport:
    rows: tuple[SmokeFitResult, ...]
    summaries: tuple[SmokeKSummary, ...]
    selected_k: int
    tie_candidates: tuple[int, ...]
    em_fits_executed: int
    score_config_hash: str
    target_topology_hash: str
    score_target_hash: str


@dataclass(frozen=True, slots=True)
class StoredSmokeFit:
    """Readonly fit-only snapshot retained until the deferred score phase."""

    k: int
    start: int
    data_seed: int
    split_seed: int
    model_seed: int
    Z: np.ndarray
    w0: float
    w: float
    Q_strict: float
    train_objective_diagnostics: Any
    internal_retry: int
    warnings: tuple[str, ...]
    q_failure: bool
    nan_occurred: bool
    x_hash: str
    training_y_hash: str
    train_mask_hash: str
    test_mask_hash: str
    fit_provenance_hash: str
    target_topology_hash: str
    score_config_hash: str
    fit_config_hash: str


def _freeze_smoke_audit_value(value: Any) -> Any:
    """Deep-copy a diagnostic tree into readonly/immutable containers."""

    if isinstance(value, np.ndarray):
        return _readonly_copy(value)
    if isinstance(value, Mapping):
        return tuple(
            (copy.deepcopy(key), _freeze_smoke_audit_value(item))
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_smoke_audit_value(item) for item in value)
    return copy.deepcopy(value)


def _store_smoke_fit(
    manifest_row: ManifestRow,
    result: CanaryFitResult,
    config: FrozenFitConfig,
    prepared: PreparedTrainingData,
    frozen_score_hash: str,
) -> StoredSmokeFit:
    """Detach the exact relation-side scoring state from a completed fit."""

    return StoredSmokeFit(
        k=manifest_row.k,
        start=manifest_row.start,
        data_seed=manifest_row.data_seed,
        split_seed=manifest_row.split_seed,
        model_seed=manifest_row.model_seed,
        Z=_readonly_copy(result.Z, np.float64),
        w0=float(result.w0),
        w=float(result.w),
        Q_strict=float(result.Q_strict),
        train_objective_diagnostics=_freeze_smoke_audit_value(
            result.train_objective_diagnostics
        ),
        internal_retry=result.internal_retry,
        warnings=tuple(result.warnings),
        q_failure=result.q_failure,
        nan_occurred=result.nan_occurred,
        x_hash=prepared.x_hash,
        training_y_hash=prepared.training_y_hash,
        train_mask_hash=prepared.train_mask_hash,
        test_mask_hash=prepared.test_mask_hash,
        fit_provenance_hash=prepared.fit_provenance_hash,
        target_topology_hash=_smoke_target_topology_hash(prepared),
        score_config_hash=frozen_score_hash,
        fit_config_hash=stable_config_hash(asdict(config)),
    )


def _smoke_target_topology_hash(prepared: PreparedTrainingData) -> str:
    """Outcome-blind score-scope provenance; not a target outcome hash."""

    return stable_config_hash(
        {
            "test_mask_hash": prepared.test_mask_hash,
            "pair_selection": "upper_test_pairs",
            "target_materialization": "after_all_six_fits",
        }
    )


def build_expected_smoke_comparability(
    prepared: PreparedTrainingData,
    manifest: Sequence[ManifestRow],
) -> list[ComparabilityRow]:
    """Build exact expected fit/score-topology metadata from frozen state."""

    _require(type(prepared) is PreparedTrainingData, "smoke requires PreparedTrainingData")
    common = {
        "x_hash": prepared.x_hash,
        "training_y_hash": prepared.training_y_hash,
        "preprocessing_hash": stable_config_hash(
            {"preprocessing": "identity", "uses_y_test": False}
        ),
        "train_mask_hash": prepared.train_mask_hash,
        "test_mask_hash": prepared.test_mask_hash,
        "fit_provenance_hash": prepared.fit_provenance_hash,
        "target_topology_hash": _smoke_target_topology_hash(prepared),
        "score_config_hash": score_config_hash(frozen_score_config()),
    }
    return [ComparabilityRow(manifest=row, **common) for row in manifest]


def build_smoke_comparability(
    prepared: PreparedTrainingData,
    manifest: Sequence[ManifestRow],
) -> list[ComparabilityRow]:
    """Compatibility wrapper for the expected frozen smoke metadata builder."""

    return build_expected_smoke_comparability(prepared, manifest)


def validate_smoke_comparability(
    rows: Sequence[ComparabilityRow],
    prepared: PreparedTrainingData,
    expected_manifest: Sequence[ManifestRow],
) -> None:
    """Bind every smoke row to actual prepared state and frozen configuration."""

    expected_rows = build_expected_smoke_comparability(prepared, expected_manifest)
    validate_cross_k_comparability(rows, expected_rows)


def _smoke_fit_config(row: ManifestRow) -> FrozenFitConfig:
    _require(row.replicate == 1, "smoke replicate changed")
    _require(row.k in SMOKE_K_CANDIDATES, "smoke K is unexpected")
    _require(row.start in START_LABELS, "smoke start is unexpected")
    _require(row.data_seed == DATA_SEED_BASE + 1, "smoke data seed changed")
    _require(row.split_seed == SPLIT_SEED_BASE + 1, "smoke split seed changed")
    _require(
        row.model_seed == expected_model_seed(row.replicate, row.k, row.start),
        "smoke model seed changed",
    )
    return FrozenFitConfig(
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
        k_est=row.k,
        L=L_SAMPLES,
        num_iter=NUM_ITER,
        seed=row.model_seed,
        numerics_mode=NUMERICS_MODE,
    )


def _require_clean_smoke_fit(result: CanaryFitResult, label: str) -> None:
    _require(type(result) is CanaryFitResult, f"{label} returned invalid fit result")
    _require(result.internal_retry == 0, f"{label} internal_retry > 0")
    _require(not result.warnings, f"{label} emitted warnings")
    _require(not result.q_failure, f"{label} Q failure")
    _require(not result.nan_occurred, f"{label} NaN/nonfinite state")
    _require_finite_tree(result.initialization.Z, f"{label} init Z")
    _require_finite_tree(result.initialization.F, f"{label} init F")
    _require_finite_tree(result.initialization.w0, f"{label} init w0")
    _require_finite_tree(result.initialization.w, f"{label} init w")
    _require_finite_tree(result.initialization.sigma_y, f"{label} init sigma_y")
    _require_finite_tree(result.Z, f"{label} final Z")
    _require_finite_tree(result.F, f"{label} final F")
    _require_finite_tree(result.w0, f"{label} final w0")
    _require_finite_tree(result.w, f"{label} final w")
    _require_finite_tree(result.sigma_y, f"{label} final sigma_y")
    _require_finite_tree(result.Q_strict, f"{label} Q_strict")
    _require_finite_tree(result.train_objective_diagnostics, f"{label} diagnostics")


@dataclass(frozen=True, slots=True)
class _SmokeFitPhaseResult:
    stored_fits: tuple[StoredSmokeFit, ...]
    fit_count: int
    score_config_hash: str


def _run_smoke_fit_phase(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    manifest: Sequence[ManifestRow],
    comparability: Sequence[ComparabilityRow],
    adapter: AuthorizedEMFitAdapter | _TestAuthorizedFitAdapter,
    test_only: bool,
) -> _SmokeFitPhaseResult:
    """Fit the frozen six rows without accepting raw Y or score targets."""

    expected_manifest = build_manifest((1,), SMOKE_K_CANDIDATES, START_LABELS)
    validate_manifest(manifest, (1,), SMOKE_K_CANDIDATES, START_LABELS)
    _require(tuple(manifest) == tuple(expected_manifest), "smoke manifest order/content changed")
    validate_smoke_comparability(comparability, prepared, expected_manifest)
    _require(
        tuple(row.manifest for row in comparability) == tuple(expected_manifest),
        "smoke comparability manifest order changed",
    )
    _validate_canary_preflight(preflight, prepared.train_mask, prepared.test_mask)
    _require(len(expected_manifest) == 6, "smoke manifest must contain exactly six rows")

    stored_fits: list[StoredSmokeFit] = []
    frozen_score_hash = score_config_hash(frozen_score_config())
    fit_count = 0
    for manifest_row in expected_manifest:
        config = _smoke_fit_config(manifest_row)
        if test_only:
            _require(type(adapter) is _TestAuthorizedFitAdapter, "test smoke requires test adapter")
            boundary = FitCallBoundary._from_preflight_test_only(
                prepared, preflight, config, adapter
            )
        else:
            _require(type(adapter) is AuthorizedEMFitAdapter, "production smoke requires production adapter")
            boundary = FitCallBoundary.from_preflight(
                prepared, preflight, config, adapter
            )
        fit_count += 1
        result = boundary.call(0)
        label = f"smoke K={manifest_row.k} start={manifest_row.start}"
        _require_clean_smoke_fit(result, label)
        stored_fits.append(
            _store_smoke_fit(
                manifest_row, result, config, prepared, frozen_score_hash
            )
        )

    expected_order = tuple(
        (row.k, row.start, row.model_seed) for row in expected_manifest
    )
    stored_order = tuple((row.k, row.start, row.model_seed) for row in stored_fits)
    _require(fit_count == 6, "smoke did not execute exactly six fits")
    _require(len(stored_fits) == 6, "smoke did not store exactly six clean fits")
    _require(stored_order == expected_order, "stored smoke fit order changed")
    return _SmokeFitPhaseResult(tuple(stored_fits), fit_count, frozen_score_hash)


def _run_smoke_selection(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    score_Y: np.ndarray,
    manifest: Sequence[ManifestRow],
    comparability: Sequence[ComparabilityRow],
    adapter: AuthorizedEMFitAdapter | _TestAuthorizedFitAdapter,
    test_only: bool,
) -> SmokeSelectionReport:
    """Execute exactly the frozen six-row smoke or stop on the first failure."""

    fit_phase = _run_smoke_fit_phase(
        preflight=preflight,
        prepared=prepared,
        manifest=manifest,
        comparability=comparability,
        adapter=adapter,
        test_only=test_only,
    )
    stored_fits = fit_phase.stored_fits
    fit_count = fit_phase.fit_count
    frozen_score_hash = fit_phase.score_config_hash

    # Phase B: score only.  The outcome-bearing target is created exactly once,
    # after all six clean fits and the hard fit-count/order gates have passed.
    target = make_score_only_target(score_Y, prepared.test_mask)
    score_target_hash = stable_array_hash(target.rows, target.cols, target.values)
    rows: list[SmokeFitResult] = []
    for stored in stored_fits:
        eta_pairs = heldout_raw_eta_pairs(
            stored.Z,
            stored.w0,
            stored.w,
            prepared.test_mask,
        )
        score = score_heldout_bernoulli(target, eta_pairs)
        _require(np.isfinite(score), "smoke held-out score is nonfinite")
        rows.append(
            SmokeFitResult(
                k=stored.k,
                start=stored.start,
                data_seed=stored.data_seed,
                split_seed=stored.split_seed,
                model_seed=stored.model_seed,
                fit_status="clean",
                heldout_mean_log_score=float(score),
                Q_strict=stored.Q_strict,
                internal_retry=stored.internal_retry,
                warnings=stored.warnings,
                warning_count=len(stored.warnings),
                q_failure=stored.q_failure,
                nan_occurred=stored.nan_occurred,
                finite_state=True,
                x_hash=stored.x_hash,
                training_y_hash=stored.training_y_hash,
                train_mask_hash=stored.train_mask_hash,
                test_mask_hash=stored.test_mask_hash,
                fit_provenance_hash=stored.fit_provenance_hash,
                target_topology_hash=stored.target_topology_hash,
                score_target_hash=score_target_hash,
                score_config_hash=stored.score_config_hash,
                fit_config_hash=stored.fit_config_hash,
            )
        )

    _require(len(rows) == 6, "smoke did not score exactly six stored fits")
    _require(bool(score_target_hash), "smoke score target was not created")

    start_scores = [
        StartScore(row.k, row.start, np.float64(row.heldout_mean_log_score))
        for row in rows
    ]
    selection = select_k_from_two_starts(
        start_scores, SMOKE_K_CANDIDATES, START_LABELS
    )
    summaries: list[SmokeKSummary] = []
    for k in SMOKE_K_CANDIDATES:
        by_start = {
            row.start: row.heldout_mean_log_score for row in rows if row.k == k
        }
        _require(set(by_start) == set(START_LABELS), "smoke aggregation start set changed")
        expected_mean = np.mean(
            np.asarray([by_start[1], by_start[2]], dtype=np.float64),
            dtype=np.float64,
        )
        _require(
            np.float64(selection.mean_scores[k]) == expected_mean,
            "smoke aggregation is not the unweighted two-start mean",
        )
        summaries.append(
            SmokeKSummary(k, by_start[1], by_start[2], float(expected_mean))
        )

    return SmokeSelectionReport(
        rows=tuple(rows),
        summaries=tuple(summaries),
        selected_k=selection.selected_k,
        tie_candidates=selection.tie_candidates,
        em_fits_executed=fit_count,
        score_config_hash=frozen_score_hash,
        target_topology_hash=_smoke_target_topology_hash(prepared),
        score_target_hash=score_target_hash,
    )


def run_smoke_selection(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    score_Y: np.ndarray,
    manifest: Sequence[ManifestRow],
    comparability: Sequence[ComparabilityRow],
    adapter: AuthorizedEMFitAdapter,
) -> SmokeSelectionReport:
    """Production smoke entry point; only the sealed EM adapter is accepted."""

    _require(type(adapter) is AuthorizedEMFitAdapter, "production smoke requires production adapter")
    return _run_smoke_selection(
        preflight=preflight,
        prepared=prepared,
        score_Y=score_Y,
        manifest=manifest,
        comparability=comparability,
        adapter=adapter,
        test_only=False,
    )


def _run_smoke_selection_test_only(
    *,
    preflight: CanaryPreflight,
    prepared: PreparedTrainingData,
    score_Y: np.ndarray,
    manifest: Sequence[ManifestRow],
    comparability: Sequence[ComparabilityRow],
    adapter: _TestAuthorizedFitAdapter,
) -> SmokeSelectionReport:
    """Pure fake-fit smoke entry point; production CLI cannot select it."""

    return _run_smoke_selection(
        preflight=preflight,
        prepared=prepared,
        score_Y=score_Y,
        manifest=manifest,
        comparability=comparability,
        adapter=adapter,
        test_only=True,
    )


def frozen_config() -> dict[str, Any]:
    return {
        "family_x": FAMILY_X,
        "family_y": FAMILY_Y,
        "k_true": K_TRUE,
        "n": N_NODES,
        "d": N_FEATURES,
        "test_ratio": TEST_RATIO,
        "L": L_SAMPLES,
        "num_iter": NUM_ITER,
        "numerics_mode": NUMERICS_MODE,
        "future_k_candidates": list(K_CANDIDATES),
        "starts_per_k": len(START_LABELS),
        "primary_score": "held-out Bernoulli plug-in Y mean log score",
    }


def _expected_test_pairs(n_nodes: int, test_ratio: float) -> int:
    return max(1, int(round((n_nodes * (n_nodes - 1) // 2) * test_ratio)))


def run_validate_only() -> dict[str, Any]:
    """Run deterministic pure/static checks; this path cannot reach EM."""

    config = frozen_config()
    _require(config["family_x"] == "poisson" and config["family_y"] == "bernoulli", "family freeze changed")
    _require(tuple(config["future_k_candidates"]) == K_CANDIDATES, "K candidate freeze changed")
    _require(config["starts_per_k"] == 2, "two-start freeze changed")

    replicates = (1,)
    full_manifest = build_manifest(replicates, K_CANDIDATES)
    smoke_manifest = build_manifest(replicates, SMOKE_K_CANDIDATES)
    validate_manifest(full_manifest, replicates, K_CANDIDATES)
    validate_manifest(smoke_manifest, replicates, SMOKE_K_CANDIDATES)

    split_plans = preflight_all_splits(replicates)
    split_plan = split_plans[0]
    expected_test_pairs = split_plan.expected_test_pairs
    train_mask = split_plan.train_mask
    test_mask = split_plan.test_mask
    split = split_plan.diagnostics

    # Static sentinels exercise typing, hashes, canaries, and comparability only.
    # They are never supplied to a model and are not persisted.
    X_sentinel = np.zeros((N_NODES, N_FEATURES), dtype=np.float64)
    Y_sentinel = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    preflight = authorize_canary_preflight(split_plan)
    prepared = prepare_training_data(
        X_sentinel,
        Y_sentinel,
        preflight=preflight,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    training = prepared.training_values
    canary_a, canary_b = build_two_canary_payloads(training, train_mask)
    _require(np.array_equal(canary_a.Y_fit[train_mask], canary_b.Y_fit[train_mask]), "canary train values differ")
    _require(np.any(canary_a.Y_fit[test_mask] != canary_b.Y_fit[test_mask]), "canary test payloads do not differ")

    preprocessing_hash = stable_config_hash({"preprocessing": "identity", "uses_y_test": False})
    frozen_score_hash = score_config_hash(frozen_score_config())
    common = {
        "x_hash": prepared.x_hash,
        "training_y_hash": prepared.training_y_hash,
        "preprocessing_hash": preprocessing_hash,
        "train_mask_hash": stable_array_hash(train_mask),
        "test_mask_hash": stable_array_hash(test_mask),
        "fit_provenance_hash": prepared.fit_provenance_hash,
        "target_topology_hash": _smoke_target_topology_hash(prepared),
        "score_config_hash": frozen_score_hash,
    }
    comparability = [ComparabilityRow(manifest=row, **common) for row in smoke_manifest]
    validate_smoke_comparability(comparability, prepared, smoke_manifest)

    require_no_blocking_failures({})
    return {
        "mode": "validate-only",
        "em_fits_executed": 0,
        "config_hash": stable_config_hash(config),
        "full_manifest_keys": len(full_manifest),
        "smoke_manifest_keys": len(smoke_manifest),
        "expected_test_pairs": expected_test_pairs,
        "split": split,
        "fit_payload_a_hash": canary_a.payload_hash,
        "fit_payload_b_hash": canary_b.payload_hash,
        "score_config_hash": frozen_score_hash,
    }


def run_canary_cli() -> CanaryInvarianceReport:
    """Run the deliberately gated complete-run canary; never called by validation."""

    replicate = 1
    k_est = K_TRUE
    start = START_LABELS[0]
    expected_manifest = build_manifest((replicate,), (k_est,), (start,))
    validate_manifest(expected_manifest, (replicate,), (k_est,), (start,))

    # All topology guards complete before any callable capable of fitting is
    # imported or invoked.
    split_plans = preflight_all_splits((replicate,))
    split_plan = split_plans[0]
    preflight = authorize_canary_preflight(split_plan)

    if str(EXPFAM_SRC) not in sys.path:
        sys.path.insert(0, str(EXPFAM_SRC))
    from data_generator_expfam import generate_dual_data  # noqa: PLC0415

    data = generate_dual_data(
        n=N_NODES,
        d=N_FEATURES,
        k=K_TRUE,
        seed=DATA_SEED_BASE + replicate,
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
    )
    X = _readonly_copy(data["X"], np.float64)
    Y = _readonly_copy(data["Y"], np.float64)
    prepared = prepare_training_data(
        X,
        Y,
        preflight=preflight,
        train_mask=split_plan.train_mask,
        test_mask=split_plan.test_mask,
    )
    config = FrozenFitConfig(
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
        k_est=k_est,
        L=L_SAMPLES,
        num_iter=NUM_ITER,
        seed=expected_model_seed(replicate, k_est, start),
        numerics_mode=NUMERICS_MODE,
    )
    return run_two_canary_falsification(
        preflight=preflight,
        prepared=prepared,
        score_Y=Y,
        config=config,
        adapter=AuthorizedEMFitAdapter(),
    )


def run_smoke_cli() -> SmokeSelectionReport:
    """Run the explicitly gated K={2,3,4}, starts={1,2} smoke."""

    replicate = 1
    manifest = build_manifest((replicate,), SMOKE_K_CANDIDATES, START_LABELS)
    validate_manifest(manifest, (replicate,), SMOKE_K_CANDIDATES, START_LABELS)

    # Complete every manifest/split/provenance guard before importing EM code.
    split_plans = preflight_all_splits((replicate,))
    split_plan = split_plans[0]
    preflight = authorize_canary_preflight(split_plan)

    if str(EXPFAM_SRC) not in sys.path:
        sys.path.insert(0, str(EXPFAM_SRC))
    from data_generator_expfam import generate_dual_data  # noqa: PLC0415

    data = generate_dual_data(
        n=N_NODES,
        d=N_FEATURES,
        k=K_TRUE,
        seed=DATA_SEED_BASE + replicate,
        family_x=FAMILY_X,
        family_y=FAMILY_Y,
    )
    X = _readonly_copy(data["X"], np.float64)
    score_Y = _readonly_copy(data["Y"], np.float64)
    prepared = prepare_training_data(
        X,
        score_Y,
        preflight=preflight,
        train_mask=split_plan.train_mask,
        test_mask=split_plan.test_mask,
    )
    comparability = build_smoke_comparability(prepared, manifest)
    validate_smoke_comparability(comparability, prepared, manifest)
    return run_smoke_selection(
        preflight=preflight,
        prepared=prepared,
        score_Y=score_Y,
        manifest=manifest,
        comparability=comparability,
        adapter=AuthorizedEMFitAdapter(),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true", help="pure/static validation; performs zero fits")
    modes.add_argument("--canary", action="store_true", help="run gated two-payload canary")
    modes.add_argument("--smoke", action="store_true", help="run gated K={2,3,4}, two-start smoke")
    modes.add_argument("--full", action="store_true", help="out of scope for Issue #41")
    parser.add_argument(
        "--allow-em",
        action="store_true",
        help="second explicit authorization required with --canary or --smoke",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        result = run_validate_only()
        result["split"] = asdict(result["split"])
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.canary:
        _require(args.allow_em, "--canary requires the additional --allow-em authorization")
        report = run_canary_cli()
        print(json.dumps(asdict(report), sort_keys=True, allow_nan=False))
        return 0
    if args.smoke:
        _require(args.allow_em, "--smoke requires the additional --allow-em authorization")
        report = run_smoke_cli()
        print(json.dumps(asdict(report), sort_keys=True, allow_nan=False))
        return 0
    raise HarnessStop("full pilot is not implemented or authorized")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessStop as error:
        print(f"BLOCKING: {error}", file=sys.stderr)
        raise SystemExit(2) from error
