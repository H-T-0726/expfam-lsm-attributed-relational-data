from __future__ import annotations

import csv
import inspect
import functools
import json
import sys
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_heldout_k_selection_pilot as harness


def valid_masks() -> tuple[np.ndarray, np.ndarray]:
    n = 6
    test = np.zeros((n, n), dtype=bool)
    for i, j in ((0, 1), (2, 3), (4, 5)):
        test[i, j] = test[j, i] = True
    train = ~test & ~np.eye(n, dtype=bool)
    return train, test


def test_bernoulli_scorer_matches_independent_reference() -> None:
    y = np.array([0, 1, 1, 0], dtype=float)
    eta = np.array([-2.5, -0.2, 0.7, 3.0], dtype=float)
    expected = np.mean(y * eta - np.logaddexp(0.0, eta))
    assert harness.heldout_bernoulli_mean_log_score(y, eta) == pytest.approx(expected)


def test_bernoulli_scorer_is_stable_at_extreme_eta() -> None:
    eta = np.array([-100.0, -20.0, 0.0, 20.0, 100.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    expected = np.mean(y * eta - np.logaddexp(0.0, eta))
    actual = harness.heldout_bernoulli_mean_log_score(y, eta)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected)
    source = inspect.getsource(harness.heldout_bernoulli_mean_log_score)
    assert "predict_mu_y" not in source
    assert "clip" not in source


def test_raw_eta_uses_upper_test_pairs_without_probability_transform() -> None:
    Z = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 3.0]])
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 2] = mask[2, 0] = True
    pairs = harness.heldout_raw_eta_pairs(Z, 0.5, 2.0, mask)
    assert np.array_equal(pairs.rows, [0])
    assert np.array_equal(pairs.cols, [2])
    assert pairs.eta[0] == pytest.approx(0.5 + 2.0 * np.dot(Z[0], Z[2]))
    assert "predict_mu_y" not in inspect.getsource(harness.heldout_raw_eta_pairs)


def test_topology_validator_valid_and_has_no_y_api() -> None:
    train, test = valid_masks()
    result = harness.validate_pair_masks(train, test, 3)
    assert result.test_pairs == 3
    assert result.min_train_degree == 4
    assert list(inspect.signature(harness.validate_pair_masks).parameters) == [
        "train_mask",
        "test_mask",
        "expected_test_pairs",
    ]


def test_topology_validator_rejects_disconnected_train_graph() -> None:
    train = np.zeros((6, 6), dtype=bool)
    for group in ((0, 1, 2), (3, 4, 5)):
        for i in group:
            for j in group:
                if i != j:
                    train[i, j] = True
    test = ~train & ~np.eye(6, dtype=bool)
    with pytest.raises(harness.HarnessStop, match="disconnected"):
        harness.validate_pair_masks(train, test, 9)


def test_topology_validator_rejects_min_train_degree() -> None:
    train = np.zeros((6, 6), dtype=bool)
    for j in range(1, 6):
        train[0, j] = train[j, 0] = True
    test = ~train & ~np.eye(6, dtype=bool)
    with pytest.raises(harness.HarnessStop, match="train-mask degree"):
        harness.validate_pair_masks(train, test, 10)


def test_topology_validator_rejects_min_test_degree() -> None:
    test = np.zeros((6, 6), dtype=bool)
    test[0, 1] = test[1, 0] = True
    train = ~test & ~np.eye(6, dtype=bool)
    with pytest.raises(harness.HarnessStop, match="test-mask degree"):
        harness.validate_pair_masks(train, test, 1)


def test_topology_validator_rejects_overlap() -> None:
    train, test = valid_masks()
    train = train.copy()
    train[0, 1] = train[1, 0] = True
    with pytest.raises(harness.HarnessStop, match="overlap"):
        harness.validate_pair_masks(train, test, 3)


def test_topology_validator_rejects_missing_union() -> None:
    train, test = valid_masks()
    train = train.copy()
    train[0, 2] = train[2, 0] = False
    with pytest.raises(harness.HarnessStop, match="union"):
        harness.validate_pair_masks(train, test, 3)


def test_topology_validator_rejects_asymmetry() -> None:
    train, test = valid_masks()
    train = train.copy()
    train[0, 2] = False
    with pytest.raises(harness.HarnessStop, match="symmetric"):
        harness.validate_pair_masks(train, test, 3)


def test_topology_validator_rejects_true_diagonal() -> None:
    train, test = valid_masks()
    train = train.copy()
    train[0, 0] = True
    with pytest.raises(harness.HarnessStop, match="diagonal"):
        harness.validate_pair_masks(train, test, 3)


def test_topology_validator_rejects_wrong_test_count() -> None:
    train, test = valid_masks()
    with pytest.raises(harness.HarnessStop, match="test-pair count"):
        harness.validate_pair_masks(train, test, 4)


@pytest.mark.parametrize(
    ("train_transform", "test_transform", "message"),
    [
        (lambda value: value[:, :-1], lambda value: value, "square"),
        (lambda value: value.astype(np.int8), lambda value: value, "dtype"),
        (lambda value: value, lambda value: value.astype(np.int8), "dtype"),
    ],
)
def test_topology_validator_rejects_wrong_shape_or_dtype(train_transform, test_transform, message: str) -> None:
    train, test = valid_masks()
    with pytest.raises(harness.HarnessStop, match=message):
        harness.validate_pair_masks(train_transform(train), test_transform(test), 3)


def test_topology_validator_rejects_empty_mask_as_harness_stop() -> None:
    empty = np.zeros((0, 0), dtype=bool)
    with pytest.raises(harness.HarnessStop, match="empty mask"):
        harness.validate_pair_masks(empty, empty, 0)


def test_hash_is_stable_and_sensitive_to_shape_dtype_and_bytes() -> None:
    value = np.arange(6, dtype=np.int64).reshape(2, 3)
    digest = harness.stable_array_hash(value)
    assert len(digest) == 64
    assert digest == harness.stable_array_hash(np.asfortranarray(value))
    assert digest != harness.stable_array_hash(value.reshape(3, 2))
    assert digest != harness.stable_array_hash(value.astype(np.float64))
    changed = value.copy()
    changed[0, 0] = 99
    assert digest != harness.stable_array_hash(changed)


def test_bool_mask_single_flip_changes_hash() -> None:
    train, _ = valid_masks()
    changed = train.copy()
    changed[0, 1] = not changed[0, 1]
    assert harness.stable_array_hash(train) != harness.stable_array_hash(changed)


def test_pair_aligned_scorer_accepts_only_exact_target_coordinates() -> None:
    train, test = valid_masks()
    Y = np.zeros((6, 6), dtype=float)
    Y[0, 1] = Y[1, 0] = 1.0
    target = harness.make_score_only_target(Y, test)
    eta = harness.EtaPairs(
        target.n_nodes,
        target.rows.copy(),
        target.cols.copy(),
        np.zeros(target.values.size),
        target.test_mask_hash,
    )
    assert harness.score_heldout_bernoulli(target, eta) == pytest.approx(-np.log(2.0))

    reversed_eta = replace(eta, rows=eta.rows[::-1], cols=eta.cols[::-1], eta=eta.eta[::-1])
    with pytest.raises(harness.HarnessStop, match="rows mismatch"):
        harness.score_heldout_bernoulli(target, reversed_eta)
    train_mixed = replace(eta, rows=np.array([0, 2, 4]), cols=np.array([2, 3, 5]))
    with pytest.raises(harness.HarnessStop, match="cols mismatch"):
        harness.score_heldout_bernoulli(target, train_mixed)
    with pytest.raises(harness.HarnessStop, match="cols mismatch"):
        changed_cols = eta.cols.copy()
        changed_cols[0] += 1
        harness.score_heldout_bernoulli(target, replace(eta, cols=changed_cols))
    with pytest.raises(harness.HarnessStop, match="mask hash"):
        harness.score_heldout_bernoulli(target, replace(eta, test_mask_hash="wrong"))
    duplicate = replace(
        eta,
        rows=np.array([0, 0, 4], dtype=np.int64),
        cols=np.array([1, 1, 5], dtype=np.int64),
    )
    with pytest.raises(harness.HarnessStop, match="duplicate"):
        harness.score_heldout_bernoulli(target, duplicate)
    lower = replace(
        eta,
        rows=eta.cols.copy(),
        cols=eta.rows.copy(),
    )
    with pytest.raises(harness.HarnessStop, match="upper-triangle"):
        harness.score_heldout_bernoulli(target, lower)


def test_target_and_fit_paths_are_typed_copied_and_separate() -> None:
    train, test = valid_masks()
    Y = np.zeros((6, 6), dtype=float)
    Y[test] = 1.0
    training = harness.make_training_y_values(Y, train)
    target = harness.make_score_only_target(Y, test)
    payload = harness.build_fit_payload(training, train, 0)
    assert type(training) is harness.TrainingYValues
    assert type(target) is harness.ScoreOnlyTarget
    assert payload.Y_fit is not Y
    assert payload.Y_fit is not target.values
    assert not np.shares_memory(payload.Y_fit, Y)
    assert not np.shares_memory(payload.Y_fit, target.values)
    assert "ScoreOnlyTarget" not in str(inspect.signature(harness.build_fit_payload))
    with pytest.raises(harness.HarnessStop, match="TrainingYValues"):
        harness.build_fit_payload(target, train, 0)  # type: ignore[arg-type]


def boundary_parts() -> tuple[object, ...]:
    train, test = valid_masks()
    diagnostics = harness.validate_pair_masks(train, test, 3)
    plan = harness.SplitPlan(1, harness.SPLIT_SEED_BASE + 1, 3, train, test, diagnostics)
    preflight = harness.authorize_canary_preflight(plan)
    Y = np.zeros((6, 6), dtype=float)
    X = np.zeros((6, 2), dtype=np.float64)
    prepared = harness.prepare_training_data(
        X, Y, preflight=preflight, train_mask=train, test_mask=test
    )
    target = harness.make_score_only_target(Y, test)
    config = harness.FrozenFitConfig("poisson", "bernoulli", 3, 5, 8, 43031, "consistent")
    return train, test, Y, preflight, prepared, target, config


def test_prepared_training_data_contains_training_pairs_but_no_full_source_y() -> None:
    train, test, Y, _, prepared, _, _ = boundary_parts()
    assert not hasattr(prepared, "source_Y")
    assert "_source_Y" not in prepared.__slots__
    assert "source_y_hash" not in prepared.__slots__
    training = prepared.training_values
    expected_rows, expected_cols = np.where(np.triu(train, 1))
    assert np.array_equal(training.rows, expected_rows)
    assert np.array_equal(training.cols, expected_cols)
    assert np.array_equal(training.values, Y[expected_rows, expected_cols])
    assert not np.shares_memory(training.values, Y)
    assert training.values.size == np.count_nonzero(np.triu(train, 1))
    assert training.values.size + np.count_nonzero(np.triu(test, 1)) == 15


def test_y_test_counterfactual_changes_only_score_side_provenance() -> None:
    train, test = valid_masks()
    diagnostics = harness.validate_pair_masks(train, test, 3)
    plan = harness.SplitPlan(1, harness.SPLIT_SEED_BASE + 1, 3, train, test, diagnostics)
    preflight = harness.authorize_canary_preflight(plan)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    Y_a = np.zeros((6, 6), dtype=np.float64)
    Y_b = Y_a.copy()
    Y_b[test] = 1.0

    prepared_a = harness.prepare_training_data(
        X, Y_a, preflight=preflight, train_mask=train, test_mask=test
    )
    prepared_b = harness.prepare_training_data(
        X, Y_b, preflight=preflight, train_mask=train, test_mask=test
    )
    assert np.array_equal(prepared_a.X, prepared_b.X)
    assert np.array_equal(prepared_a.training_values.rows, prepared_b.training_values.rows)
    assert np.array_equal(prepared_a.training_values.cols, prepared_b.training_values.cols)
    assert np.array_equal(prepared_a.training_values.values, prepared_b.training_values.values)
    assert prepared_a.training_y_hash == prepared_b.training_y_hash
    assert prepared_a.fit_provenance_hash == prepared_b.fit_provenance_hash
    assert prepared_a.train_mask_hash == prepared_b.train_mask_hash
    assert prepared_a.test_mask_hash == prepared_b.test_mask_hash
    manifest = harness.build_manifest(
        (1,), harness.SMOKE_K_CANDIDATES, harness.START_LABELS
    )
    comparability_a = harness.build_smoke_comparability(prepared_a, manifest)
    comparability_b = harness.build_smoke_comparability(prepared_b, manifest)
    assert comparability_a == comparability_b
    assert {
        row.target_topology_hash for row in comparability_a
    } == {row.target_topology_hash for row in comparability_b}

    calls_a: list[dict[str, object]] = []
    calls_b: list[dict[str, object]] = []
    config = harness.FrozenFitConfig("poisson", "bernoulli", 3, 5, 8, 43031, "consistent")
    boundary_a = harness.FitCallBoundary._from_preflight_test_only(
        prepared_a,
        preflight,
        config,
        harness._make_test_fit_adapter(
            lambda **kwargs: calls_a.append(kwargs) or "ok", score_targets=()
        ),
    )
    boundary_b = harness.FitCallBoundary._from_preflight_test_only(
        prepared_b,
        preflight,
        config,
        harness._make_test_fit_adapter(
            lambda **kwargs: calls_b.append(kwargs) or "ok", score_targets=()
        ),
    )
    assert boundary_a.call(0) == boundary_b.call(0) == "ok"
    for field in ("X", "Y", "train_mask"):
        assert np.array_equal(calls_a[0][field], calls_b[0][field])

    target_a = harness.make_score_only_target(Y_a, test)
    target_b = harness.make_score_only_target(Y_b, test)
    score_target_hash_a = harness.stable_array_hash(
        target_a.rows, target_a.cols, target_a.values
    )
    score_target_hash_b = harness.stable_array_hash(
        target_b.rows, target_b.cols, target_b.values
    )
    assert score_target_hash_a != score_target_hash_b


def test_boundary_graph_and_inputs_have_no_raw_y_reference_or_heldout_outcomes() -> None:
    train, test = valid_masks()
    diagnostics = harness.validate_pair_masks(train, test, 3)
    plan = harness.SplitPlan(1, harness.SPLIT_SEED_BASE + 1, 3, train, test, diagnostics)
    preflight = harness.authorize_canary_preflight(plan)
    X = np.zeros((6, 2), dtype=np.float64)
    raw_Y = np.zeros((6, 6), dtype=np.float64)
    raw_Y[test] = 1.0
    prepared = harness.prepare_training_data(
        X, raw_Y, preflight=preflight, train_mask=train, test_mask=test
    )
    calls: list[dict[str, object]] = []
    adapter = harness._make_test_fit_adapter(
        lambda **kwargs: calls.append(kwargs) or "ok", score_targets=()
    )
    config = harness.FrozenFitConfig("poisson", "bernoulli", 3, 5, 8, 43031, "consistent")
    boundary = harness.FitCallBoundary._from_preflight_test_only(
        prepared, preflight, config, adapter
    )

    for name in boundary.__slots__:
        value = getattr(boundary, name)
        if isinstance(value, np.ndarray):
            assert value is not raw_Y
            assert not np.shares_memory(value, raw_Y)
    training = boundary._training_values
    assert training.values is not raw_Y
    assert not np.shares_memory(training.values, raw_Y)
    assert boundary.call(0) == "ok"
    assert calls[0]["Y"] is not raw_Y
    assert not np.shares_memory(calls[0]["Y"], raw_Y)
    assert np.all(np.asarray(calls[0]["Y"])[test] == 0.0)


def test_raw_y_test_mutation_after_preparation_cannot_change_boundary_input() -> None:
    train, test = valid_masks()
    diagnostics = harness.validate_pair_masks(train, test, 3)
    plan = harness.SplitPlan(1, harness.SPLIT_SEED_BASE + 1, 3, train, test, diagnostics)
    preflight = harness.authorize_canary_preflight(plan)
    X = np.zeros((6, 2), dtype=np.float64)
    raw_Y = np.zeros((6, 6), dtype=np.float64)
    prepared = harness.prepare_training_data(
        X, raw_Y, preflight=preflight, train_mask=train, test_mask=test
    )
    before_hash = prepared.fit_provenance_hash
    before_values = prepared.training_values.values.copy()
    raw_Y[test] = 1.0

    calls: list[np.ndarray] = []
    adapter = harness._make_test_fit_adapter(
        lambda **kwargs: calls.append(np.array(kwargs["Y"], copy=True)) or "ok",
        score_targets=(),
    )
    config = harness.FrozenFitConfig("poisson", "bernoulli", 3, 5, 8, 43031, "consistent")
    boundary = harness.FitCallBoundary._from_preflight_test_only(
        prepared, preflight, config, adapter
    )
    assert boundary.call(0) == "ok"
    assert prepared.fit_provenance_hash == before_hash
    assert np.array_equal(prepared.training_values.values, before_values)
    assert np.all(calls[0][test] == 0.0)


def test_fit_boundary_test_adapter_receives_only_authorized_payloads() -> None:
    train, _, Y, preflight, prepared, target, config = boundary_parts()
    calls: list[dict[str, object]] = []

    def spy(**kwargs: object) -> str:
        calls.append(kwargs)
        return "sentinel"

    adapter = harness._make_test_fit_adapter(spy, score_targets=(target,))
    boundary = harness.FitCallBoundary._from_preflight_test_only(
        prepared, preflight, config, adapter
    )
    result = boundary.call(1)
    assert result == "sentinel"
    assert calls[0]["X"] is not prepared.X
    assert not np.shares_memory(calls[0]["X"], prepared.X)
    assert all(value is not target.values and value is not Y for value in calls[0].values())
    assert boundary.last_evidence is not None
    assert np.array_equal(calls[0]["train_mask"], train)


def test_production_adapter_and_authorized_x_are_structurally_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, preflight, prepared, _, config = boundary_parts()
    adapter = harness.AuthorizedEMFitAdapter()
    assert type(adapter) is harness.AuthorizedEMFitAdapter
    assert list(inspect.signature(harness.AuthorizedEMFitAdapter).parameters) == []
    with pytest.raises(harness.HarnessStop, match="fit invocation is unauthorized"):
        adapter.fit(prepared.X)  # type: ignore[arg-type]
    monkeypatch.setattr(
        harness.AuthorizedEMFitAdapter,
        "fit",
        lambda self, invocation: (self, invocation.keyword_arguments()),
    )
    boundary = harness.FitCallBoundary.from_preflight(
        prepared, preflight, config, adapter
    )
    used_adapter, passed = boundary.call(0)
    assert used_adapter is adapter
    assert passed["X"] is not prepared.X
    assert not np.shares_memory(passed["X"], prepared.X)


@pytest.mark.parametrize(
    ("label", "forbidden_builder"),
    [
        ("direct", lambda target: target),
        ("values", lambda target: target.values),
        ("view", lambda target: target.values[:]),
        ("copy", lambda target: target.values.copy()),
        ("dict", lambda target: {"x": target}),
        ("list", lambda target: [target]),
        ("tuple", lambda target: (target.values[:],)),
        ("dataclass", lambda target: TargetBox(target)),
        ("custom-attr", lambda target: SimpleNamespace(attr=target)),
        ("shares-memory", lambda target: target.values.view()),
        ("astype-float32", lambda target: target.values.astype(np.float32)),
        ("astype-int8", lambda target: target.values.astype(np.int8)),
        ("reshape", lambda target: target.values.reshape((-1, 1))),
        ("flatten", lambda target: target.values.flatten()),
        ("transpose-copy", lambda target: target.values.T.copy()),
    ],
)
def test_fit_boundary_rejects_every_raw_or_transformed_x(
    label: str, forbidden_builder
) -> None:
    del label
    _, _, _, preflight, prepared, target, config = boundary_parts()
    adapter = harness._make_test_fit_adapter(lambda **_: None, score_targets=(target,))
    boundary = harness.FitCallBoundary._from_preflight_test_only(
        prepared, preflight, config, adapter
    )
    assert list(inspect.signature(boundary.call).parameters) == ["canary_value"]
    transformed = forbidden_builder(target)
    with pytest.raises(TypeError):
        boundary.call(0, X=transformed)  # type: ignore[call-arg]


@dataclass
class TargetBox:
    value: object


def test_fit_boundary_has_no_arbitrary_kwargs_forwarding() -> None:
    signature = inspect.signature(harness.FitCallBoundary.call)
    assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def test_production_boundary_rejects_lambda_bound_method_and_test_adapter() -> None:
    _, _, _, preflight, prepared, target, config = boundary_parts()

    class Owner:
        def fit(self, **_: object) -> None:
            return None

    test_adapter = harness._make_test_fit_adapter(lambda **_: None, score_targets=(target,))
    for arbitrary in (lambda **_: None, Owner().fit, test_adapter):
        with pytest.raises(harness.HarnessStop, match="production fit adapter is unauthorized"):
            harness.FitCallBoundary.from_preflight(
                prepared,
                preflight,
                config,
                arbitrary,  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("leak_kind", ["closure", "bound-owner", "partial-target", "partial-values", "default", "kwdefault"])
def test_test_adapter_factory_rejects_hidden_target_callable_state(leak_kind: str) -> None:
    _, _, _, _, _, target, _ = boundary_parts()

    def base(*_: object, **__: object) -> None:
        return None

    if leak_kind == "closure":
        def callback(**_: object) -> object:
            return target
    elif leak_kind == "bound-owner":
        class Owner:
            def __init__(self) -> None:
                self.target = target

            def callback(self, **_: object) -> None:
                return None
        callback = Owner().callback
    elif leak_kind == "partial-target":
        callback = functools.partial(base, target)
    elif leak_kind == "partial-values":
        callback = functools.partial(base, target.values)
    elif leak_kind == "default":
        def callback(value: object = target, **_: object) -> object:
            return value
    else:
        def callback(*, value: object = target, **_: object) -> object:
            return value

    with pytest.raises(harness.HarnessStop, match="target|Target"):
        harness._make_test_fit_adapter(callback, score_targets=(target,))


def test_forged_fit_payload_and_training_y_authority_are_rejected() -> None:
    train, _, _, preflight, prepared, target, config = boundary_parts()
    payload = harness.build_fit_payload(prepared.training_values, train, 0)
    forged = replace(payload, Y_fit=target.values.copy(), payload_hash=harness.stable_array_hash(target.values))
    assert forged._authority is payload._authority
    assert "fit_payload" not in inspect.signature(harness.FitCallBoundary.call).parameters
    adapter = harness._make_test_fit_adapter(lambda **_: None, score_targets=(target,))
    boundary = harness.FitCallBoundary._from_preflight_test_only(
        prepared, preflight, config, adapter
    )
    with pytest.raises(TypeError):
        boundary.call(0, fit_payload=forged)  # type: ignore[call-arg]
    training = harness.make_training_y_values(np.zeros((6, 6)), train)
    with pytest.raises(harness.HarnessStop, match="training Y provenance"):
        harness.build_fit_payload(replace(training, _authority=None), train, 0)


def test_public_x_reauthorization_api_is_absent_and_prepared_is_not_replaceable() -> None:
    assert not hasattr(harness, "authorize_x_payload")
    assert not hasattr(harness, "AuthorizedXPayload")
    _, _, _, _, prepared, _, _ = boundary_parts()
    with pytest.raises(TypeError):
        replace(prepared, _X=np.ones((6, 2)))  # type: ignore[call-overload]


def test_boundary_owned_x_and_training_y_ignore_later_source_and_target_mutation() -> None:
    train, test = valid_masks()
    diagnostics = harness.validate_pair_masks(train, test, 3)
    plan = harness.SplitPlan(1, harness.SPLIT_SEED_BASE + 1, 3, train, test, diagnostics)
    preflight = harness.authorize_canary_preflight(plan)
    X = np.zeros((6, 2), dtype=np.float64)
    Y = np.zeros((6, 6), dtype=np.float64)
    prepared = harness.prepare_training_data(
        X, Y, preflight=preflight, train_mask=train, test_mask=test
    )
    calls: list[dict[str, object]] = []
    adapter = harness._make_test_fit_adapter(
        lambda **kwargs: calls.append(kwargs) or "ok", score_targets=()
    )
    config = harness.FrozenFitConfig("poisson", "bernoulli", 3, 5, 8, 43031, "consistent")
    boundary = harness.FitCallBoundary._from_preflight_test_only(
        prepared, preflight, config, adapter
    )
    target = harness.make_score_only_target(Y, test)
    X[:] = 99.0
    Y[:] = 1.0
    target.values.flags.writeable = True
    target.values[:] = 1.0
    assert boundary.call(0) == "ok"
    assert np.all(calls[0]["X"] == 0.0)
    assert np.all(calls[0]["Y"][train] == 0.0)
    assert np.all(calls[0]["Y"][test] == 0.0)


def test_initialization_snapshot_copies_all_required_values() -> None:
    class FakeModel:
        params = {
            "Z": np.ones((3, 2)),
            "F": np.full((4, 2), 2.0),
            "w0": -0.25,
            "w": 0.75,
        }
        sigma_y = 1.5

    snapshot = harness.snapshot_initialization(FakeModel())
    assert np.array_equal(snapshot.Z, FakeModel.params["Z"])
    assert np.array_equal(snapshot.F, FakeModel.params["F"])
    assert snapshot.w0 == -0.25
    assert snapshot.w == 0.75
    assert snapshot.sigma_y == 1.5
    assert not np.shares_memory(snapshot.Z, FakeModel.params["Z"])
    assert not np.shares_memory(snapshot.F, FakeModel.params["F"])


def test_canary_payloads_keep_train_equal_and_change_only_test() -> None:
    train, test = valid_masks()
    Y = np.zeros((6, 6), dtype=float)
    Y[train] = np.arange(np.count_nonzero(train)) % 2
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    training = harness.make_training_y_values(Y, train)
    target = harness.make_score_only_target(Y, test)
    a, b = harness.build_two_canary_payloads(training, train)
    assert np.array_equal(a.Y_fit[train], b.Y_fit[train])
    assert np.all(a.Y_fit[test] == 0.0)
    assert np.all(b.Y_fit[test] == 1.0)
    assert np.array_equal(a.Y_fit, a.Y_fit.T)
    assert np.array_equal(b.Y_fit, b.Y_fit.T)
    assert np.all(np.diag(a.Y_fit) == 0.0)
    assert np.isfinite(a.Y_fit).all() and np.isfinite(b.Y_fit).all()
    assert set(np.unique(a.Y_fit)) <= {0.0, 1.0}
    assert set(np.unique(b.Y_fit)) <= {0.0, 1.0}
    assert a.Y_fit is not target.values and b.Y_fit is not target.values


def test_manifest_has_exact_complete_key_set_and_fair_seeds() -> None:
    rows = harness.build_manifest((1, 2), (1, 2, 3))
    harness.validate_manifest(rows, (1, 2), (1, 2, 3))
    assert len(rows) == 12
    for replicate in (1, 2):
        selected = [row for row in rows if row.replicate == replicate]
        assert len({row.data_seed for row in selected}) == 1
        assert len({row.split_seed for row in selected}) == 1
    assert len({row.model_seed for row in rows}) == len(rows)


def test_all_splits_are_preflighted_without_redraw_or_drop_api() -> None:
    plans = harness.preflight_all_splits((1, 2))
    assert [plan.replicate for plan in plans] == [1, 2]
    assert all(plan.expected_test_pairs == 555 for plan in plans)
    assert all(plan.diagnostics.test_pairs == 555 for plan in plans)
    signature = inspect.signature(harness.preflight_all_splits)
    assert "Y" not in signature.parameters
    source = inspect.getsource(harness.preflight_all_splits)
    assert "redraw" not in source.lower().replace("no redraw", "")


def test_manifest_rejects_missing_key() -> None:
    rows = harness.build_manifest((1,), (1, 2))
    with pytest.raises(harness.HarnessStop, match="expected key set"):
        harness.validate_manifest(rows[:-1], (1,), (1, 2))


def test_manifest_rejects_duplicate_key() -> None:
    rows = harness.build_manifest((1,), (1, 2))
    with pytest.raises(harness.HarnessStop, match="duplicate manifest key"):
        harness.validate_manifest(rows + [rows[0]], (1,), (1, 2))


def test_manifest_rejects_k_specific_seed_and_model_seed_rescue() -> None:
    rows = harness.build_manifest((1,), (1, 2))
    bad_data = [replace(rows[0], data_seed=999), *rows[1:]]
    with pytest.raises(harness.HarnessStop, match="data seed"):
        harness.validate_manifest(bad_data, (1,), (1, 2))
    bad_model = [replace(rows[0], model_seed=999), *rows[1:]]
    with pytest.raises(harness.HarnessStop, match="model seed"):
        harness.validate_manifest(bad_model, (1,), (1, 2))


def comparability_rows() -> list[harness.ComparabilityRow]:
    manifest = harness.build_manifest((1,), (1, 2))
    return [
        harness.ComparabilityRow(
            manifest=row,
            x_hash="x",
            training_y_hash="training-y",
            preprocessing_hash="prep",
            train_mask_hash="train",
            test_mask_hash="test",
            fit_provenance_hash="fit",
            target_topology_hash="topology",
            score_config_hash="score",
        )
        for row in manifest
    ]


def test_cross_k_comparability_accepts_identical_provenance() -> None:
    rows = comparability_rows()
    harness.validate_cross_k_comparability(rows, list(rows))


def test_cross_k_comparability_rejects_hash_mismatch() -> None:
    rows = comparability_rows()
    expected = list(rows)
    rows[2] = replace(rows[2], target_topology_hash="different")
    with pytest.raises(harness.HarnessStop, match="target_topology_hash"):
        harness.validate_cross_k_comparability(rows, expected)


def test_cross_k_comparability_rejects_incomplete_duplicate_and_unexpected_keys() -> None:
    complete = comparability_rows()
    expected = list(complete)
    harness.validate_cross_k_comparability(complete, expected)
    with pytest.raises(harness.HarnessStop, match="key set"):
        harness.validate_cross_k_comparability(complete[:-1], expected)
    with pytest.raises(harness.HarnessStop, match="duplicate actual"):
        harness.validate_cross_k_comparability(complete + [complete[0]], expected)
    with pytest.raises(harness.HarnessStop, match="key set"):
        harness.validate_cross_k_comparability(
            [replace(complete[0], manifest=replace(complete[0].manifest, k=9)), *complete[1:]],
            expected,
        )
    with pytest.raises(harness.HarnessStop, match="key set"):
        harness.validate_cross_k_comparability(
            [replace(complete[0], manifest=replace(complete[0].manifest, start=9)), *complete[1:]],
            expected,
        )


@pytest.mark.parametrize(
    ("seed_field", "message"),
    [("model_seed", "model seed"), ("data_seed", "data seed"), ("split_seed", "split seed")],
)
def test_cross_k_comparability_rejects_wrong_actual_seed(seed_field: str, message: str) -> None:
    rows = comparability_rows()
    expected = list(rows)
    rows[0] = replace(rows[0], manifest=replace(rows[0].manifest, **{seed_field: 999}))
    with pytest.raises(harness.HarnessStop, match=message):
        harness.validate_cross_k_comparability(rows, expected)


def scores(values: dict[int, tuple[float, ...]]) -> list[harness.StartScore]:
    return [
        harness.StartScore(k, start, np.float64(value))
        for k, pair in values.items()
        for start, value in enumerate(pair, 1)
    ]


def test_two_start_selector_averages_then_selects() -> None:
    result = harness.select_k_from_two_starts(
        scores({1: (0.0, 2.0), 2: (1.5, 1.5), 3: (0.0, 1.0)}),
        (1, 2, 3),
    )
    assert result.mean_scores[1] == pytest.approx(1.0)
    assert result.selected_k == 2


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.0, 1), (0.5e-12, 1), (1.0e-12, 1), (1.1e-12, 2)],
)
def test_tie_rule_uses_float64_tolerance_and_smallest_k(delta: float, expected: int) -> None:
    result = harness.select_k_from_two_starts(
        scores({1: (1.0 - delta, 1.0 - delta), 2: (1.0, 1.0)}),
        (1, 2),
    )
    assert result.selected_k == expected


def test_two_start_selector_rejects_missing_or_extra_or_duplicate_start() -> None:
    complete = scores({1: (0.0, 1.0), 2: (0.0, 1.0)})
    with pytest.raises(harness.HarnessStop, match="two fixed starts"):
        harness.select_k_from_two_starts(complete[:-1], (1, 2))
    extra = complete + [harness.StartScore(1, 3, np.float64(2.0))]
    with pytest.raises(harness.HarnessStop, match="two fixed starts"):
        harness.select_k_from_two_starts(extra, (1, 2))
    duplicate = complete + [complete[0]]
    with pytest.raises(harness.HarnessStop, match="duplicate"):
        harness.select_k_from_two_starts(duplicate, (1, 2))


def test_two_start_selector_rejects_nan() -> None:
    rows = scores({1: (0.0, 1.0), 2: (0.0, 1.0)})
    rows[0] = replace(rows[0], score=np.float64(np.nan))
    with pytest.raises(harness.HarnessStop, match="nonfinite"):
        harness.select_k_from_two_starts(rows, (1, 2))


@pytest.mark.parametrize("nonfinite", [np.inf, -np.inf])
def test_two_start_selector_rejects_infinity(nonfinite: float) -> None:
    rows = scores({1: (0.0, 1.0), 2: (0.0, 1.0)})
    rows[0] = replace(rows[0], score=np.float64(nonfinite))
    with pytest.raises(harness.HarnessStop, match="nonfinite"):
        harness.select_k_from_two_starts(rows, (1, 2))


def test_score_config_hash_is_complete_order_independent_and_sensitive() -> None:
    base = harness.frozen_score_config()
    values = harness.asdict(base)
    reversed_values = dict(reversed(tuple(values.items())))
    digest = harness.score_config_hash(values)
    assert digest == harness.score_config_hash(reversed_values)
    for field_name, original_value in values.items():
        if isinstance(original_value, bool):
            changed_value = not original_value
        elif isinstance(original_value, str):
            changed_value = original_value + "-changed"
        elif isinstance(original_value, int):
            changed_value = original_value + 1
        else:
            changed_value = float(original_value) + 0.01
        changed = dict(values)
        changed[field_name] = changed_value
        assert harness.score_config_hash(changed) != digest, field_name
    incomplete = dict(values)
    incomplete.pop("formula_version")
    with pytest.raises(harness.HarnessStop, match="incomplete"):
        harness.score_config_hash(incomplete)


def test_blocking_policy_raises_instead_of_dropping_or_retrying() -> None:
    with pytest.raises(harness.HarnessStop, match="PILOT GLOBAL STOP"):
        harness.require_no_blocking_failures({"failed_start_seed_rescue": True})


def canary_inputs() -> dict[str, object]:
    train, test = valid_masks()
    diagnostics = harness.validate_pair_masks(train, test, 3)
    plan = harness.SplitPlan(1, harness.SPLIT_SEED_BASE + 1, 3, train, test, diagnostics)
    Y = np.zeros((6, 6), dtype=float)
    X = np.zeros((6, 2), dtype=np.float64)
    preflight = harness.authorize_canary_preflight(plan)
    prepared = harness.prepare_training_data(
        X, Y, preflight=preflight, train_mask=train, test_mask=test
    )
    return {
        "preflight": preflight,
        "prepared": prepared,
        "score_Y": Y,
        "config": harness.FrozenFitConfig(
            "poisson", "bernoulli", 3, 5, 8, 43031, "consistent"
        ),
    }


def run_static_canary(
    callback,
    inputs: dict[str, object] | None = None,
) -> harness.CanaryInvarianceReport:
    selected = canary_inputs() if inputs is None else inputs
    adapter = harness._make_test_fit_adapter(
        callback, score_targets=()
    )
    return harness._run_two_canary_falsification_test_only(
        **selected, adapter=adapter
    )


def fake_canary_result(**changes: object) -> harness.CanaryFitResult:
    initialization = harness.InitializationSnapshot(
        Z=np.ones((6, 3)), F=np.ones((2, 3)), w0=-0.5, w=0.5, sigma_y=1.0
    )
    values: dict[str, object] = {
        "initialization": initialization,
        "Z": np.full((6, 3), 2.0),
        "F": np.full((2, 3), 3.0),
        "w0": -0.25,
        "w": 0.75,
        "sigma_y": 1.0,
        "Q_strict": -123.0,
        "train_objective_diagnostics": ({"q_before": -130.0, "q_after": -123.0},),
        "internal_retry": 0,
        "q_failure": False,
        "warnings": (),
        "nan_occurred": False,
    }
    values.update(changes)
    return harness.CanaryFitResult(**values)  # type: ignore[arg-type]


def run_static_smoke(
    callback,
    *,
    inputs: dict[str, object] | None = None,
    manifest: list[harness.ManifestRow] | None = None,
    comparability: list[harness.ComparabilityRow] | None = None,
) -> harness.SmokeSelectionReport:
    selected_inputs = canary_inputs() if inputs is None else inputs
    prepared = selected_inputs["prepared"]
    score_Y = selected_inputs["score_Y"]
    assert isinstance(prepared, harness.PreparedTrainingData)
    assert isinstance(score_Y, np.ndarray)
    selected_manifest = (
        harness.build_manifest((1,), harness.SMOKE_K_CANDIDATES, harness.START_LABELS)
        if manifest is None
        else manifest
    )
    selected_comparability = (
        harness.build_smoke_comparability(prepared, selected_manifest)
        if comparability is None
        else comparability
    )
    adapter = harness._make_test_fit_adapter(callback, score_targets=())
    return harness._run_smoke_selection_test_only(
        preflight=selected_inputs["preflight"],
        prepared=prepared,
        score_Y=score_Y,
        manifest=selected_manifest,
        comparability=selected_comparability,
        adapter=adapter,
    )


def test_static_smoke_runs_exact_frozen_six_rows_and_selects() -> None:
    calls: list[tuple[int, int]] = []

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        seed = int(kwargs["seed"])
        k = int(kwargs["k"])
        calls.append((k, seed))
        return fake_canary_result(w0=-float(k), w=0.0)

    report = run_static_smoke(fake_fit)
    assert calls == [
        (2, 44021),
        (2, 44022),
        (3, 44031),
        (3, 44032),
        (4, 44041),
        (4, 44042),
    ]
    assert report.em_fits_executed == 6
    assert [(row.k, row.start, row.model_seed) for row in report.rows] == [
        (2, 1, 44021),
        (2, 2, 44022),
        (3, 1, 44031),
        (3, 2, 44032),
        (4, 1, 44041),
        (4, 2, 44042),
    ]
    assert len(report.rows) == 6
    assert len(report.summaries) == 3
    assert all(
        row.fit_status == "clean"
        and row.data_seed == 41001
        and row.split_seed == 42001
        and row.internal_retry == 0
        and row.warning_count == 0
        for row in report.rows
    )
    assert len({row.x_hash for row in report.rows}) == 1
    assert len({row.training_y_hash for row in report.rows}) == 1
    assert len({row.train_mask_hash for row in report.rows}) == 1
    assert len({row.test_mask_hash for row in report.rows}) == 1
    assert len({row.target_topology_hash for row in report.rows}) == 1
    assert len({row.score_target_hash for row in report.rows}) == 1
    assert len({row.fit_provenance_hash for row in report.rows}) == 1
    assert len({row.score_config_hash for row in report.rows}) == 1
    assert report.selected_k == 4


def test_all_six_smoke_boundaries_receive_training_only_y() -> None:
    inputs = canary_inputs()
    prepared = inputs["prepared"]
    score_Y = inputs["score_Y"]
    assert isinstance(prepared, harness.PreparedTrainingData)
    assert isinstance(score_Y, np.ndarray)
    # This is the same caller-owned raw Y used during preparation.  Mutating
    # only held-out outcomes afterward must affect scoring, never fit inputs.
    score_Y[prepared.test_mask] = 1.0
    seen = 0

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        nonlocal seen
        seen += 1
        fit_Y = np.asarray(kwargs["Y"])
        assert fit_Y is not score_Y
        assert not np.shares_memory(fit_Y, score_Y)
        assert np.all(fit_Y[prepared.test_mask] == 0.0)
        return fake_canary_result()

    run_static_smoke(fake_fit, inputs=inputs)
    assert seen == 6


@pytest.mark.parametrize(
    ("failure_call", "changes", "message"),
    [
        (1, {"q_failure": True}, "Q failure"),
        (3, {"q_failure": True}, "Q failure"),
        (6, {"q_failure": True}, "Q failure"),
        (2, {"warnings": ("warning",)}, "emitted warnings"),
        (2, {"internal_retry": 1}, "internal_retry"),
        (2, {"nan_occurred": True}, "NaN/nonfinite"),
        (2, {"w": float("nan")}, "nonfinite"),
        (2, {"Q_strict": float("nan")}, "Q_strict is nonfinite"),
        (
            2,
            {"train_objective_diagnostics": ({"q": float("nan")},)},
            "diagnostics.*nonfinite",
        ),
    ],
)
def test_static_smoke_stops_on_first_unclean_fit(
    failure_call: int,
    changes: dict[str, object],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    target_creations = 0
    score_calls = 0
    original_target = harness.make_score_only_target
    original_score = harness.score_heldout_bernoulli

    def target_spy(*args: object, **kwargs: object) -> harness.ScoreOnlyTarget:
        nonlocal target_creations
        target_creations += 1
        return original_target(*args, **kwargs)

    def score_spy(
        target: harness.ScoreOnlyTarget, eta_pairs: harness.EtaPairs
    ) -> float:
        nonlocal score_calls
        score_calls += 1
        return original_score(target, eta_pairs)

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result(**(changes if calls == failure_call else {}))

    monkeypatch.setattr(harness, "make_score_only_target", target_spy)
    monkeypatch.setattr(harness, "score_heldout_bernoulli", score_spy)
    with pytest.raises(harness.HarnessStop, match=message):
        run_static_smoke(fake_fit)
    assert calls == failure_call
    assert target_creations == 0
    assert score_calls == 0


def test_static_smoke_rejects_nonfinite_score_without_extra_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    selector_calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result()

    def selector_spy(*_: object, **__: object) -> object:
        nonlocal selector_calls
        selector_calls += 1
        raise AssertionError("selector must not see partial scored rows")

    monkeypatch.setattr(harness, "score_heldout_bernoulli", lambda *_: float("inf"))
    monkeypatch.setattr(harness, "select_k_from_two_starts", selector_spy)
    with pytest.raises(harness.HarnessStop, match="score is nonfinite"):
        run_static_smoke(fake_fit)
    assert calls == 6
    assert selector_calls == 0


def test_static_smoke_pair_mismatch_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = harness.heldout_raw_eta_pairs

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result()

    def mismatched(*args: object, **kwargs: object) -> harness.EtaPairs:
        pairs = original(*args, **kwargs)
        return replace(
            pairs,
            rows=pairs.rows[::-1],
            cols=pairs.cols[::-1],
            eta=pairs.eta[::-1],
        )

    monkeypatch.setattr(harness, "heldout_raw_eta_pairs", mismatched)
    with pytest.raises(harness.HarnessStop, match="rows mismatch"):
        run_static_smoke(fake_fit)
    assert calls == 6


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "one-start",
        "third-start",
        "extra-k-1",
        "extra-k-5",
        "start-0",
        "start-3",
        "wrong-model-seed",
        "wrong-data-seed",
        "wrong-split-seed",
    ],
)
def test_static_smoke_invalid_manifest_stops_before_fit(case: str) -> None:
    calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result()

    complete = harness.build_manifest((1,), harness.SMOKE_K_CANDIDATES, harness.START_LABELS)
    if case == "missing":
        manifest = complete[:-1]
    elif case == "duplicate":
        manifest = complete + [complete[0]]
    elif case == "one-start":
        manifest = harness.build_manifest((1,), harness.SMOKE_K_CANDIDATES, (1,))
    elif case == "third-start":
        manifest = harness.build_manifest((1,), harness.SMOKE_K_CANDIDATES, (1, 2, 3))
    elif case == "extra-k-1":
        manifest = harness.build_manifest((1,), (1, 2, 3, 4), harness.START_LABELS)
    elif case == "extra-k-5":
        manifest = harness.build_manifest((1,), (2, 3, 4, 5), harness.START_LABELS)
    elif case == "start-0":
        manifest = harness.build_manifest((1,), harness.SMOKE_K_CANDIDATES, (0, 1, 2))
    elif case == "start-3":
        manifest = harness.build_manifest((1,), harness.SMOKE_K_CANDIDATES, (1, 2, 3))
    elif case == "wrong-model-seed":
        manifest = [replace(complete[0], model_seed=999), *complete[1:]]
    elif case == "wrong-data-seed":
        manifest = [replace(complete[0], data_seed=999), *complete[1:]]
    else:
        manifest = [replace(complete[0], split_seed=999), *complete[1:]]
    with pytest.raises(harness.HarnessStop):
        run_static_smoke(fake_fit, manifest=manifest)
    assert calls == 0


@pytest.mark.parametrize(
    "field_name",
    [
        "x_hash",
        "training_y_hash",
        "train_mask_hash",
        "test_mask_hash",
        "fit_provenance_hash",
        "score_config_hash",
        "target_topology_hash",
    ],
)
@pytest.mark.parametrize("corruption", ["single", "uniform"])
def test_static_smoke_wrong_comparability_hash_stops_before_fit(
    field_name: str, corruption: str
) -> None:
    calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result()

    inputs = canary_inputs()
    prepared = inputs["prepared"]
    assert isinstance(prepared, harness.PreparedTrainingData)
    manifest = harness.build_manifest((1,), harness.SMOKE_K_CANDIDATES, harness.START_LABELS)
    comparability = harness.build_smoke_comparability(prepared, manifest)
    if corruption == "single":
        comparability[2] = replace(comparability[2], **{field_name: "wrong"})
    else:
        comparability = [
            replace(row, **{field_name: "uniform-wrong"}) for row in comparability
        ]
    with pytest.raises(harness.HarnessStop, match=field_name):
        run_static_smoke(fake_fit, manifest=manifest, comparability=comparability)
    assert calls == 0


def test_static_smoke_target_created_once_after_all_fits_and_reused_only_for_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    target_ids: list[int] = []
    original_target = harness.make_score_only_target
    original_score = harness.score_heldout_bernoulli

    def target_spy(*args: object, **kwargs: object) -> harness.ScoreOnlyTarget:
        events.append("target")
        return original_target(*args, **kwargs)

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        assert "target" not in events
        events.append("fit")
        return fake_canary_result()

    def score_spy(
        target: harness.ScoreOnlyTarget, eta_pairs: harness.EtaPairs
    ) -> float:
        events.append("score")
        target_ids.append(id(target))
        return original_score(target, eta_pairs)

    monkeypatch.setattr(harness, "make_score_only_target", target_spy)
    monkeypatch.setattr(harness, "score_heldout_bernoulli", score_spy)
    run_static_smoke(fake_fit)
    assert events == [
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
        "target",
        "score",
        "score",
        "score",
        "score",
        "score",
        "score",
    ]
    assert len(target_ids) == 6
    assert len(set(target_ids)) == 1


def test_static_smoke_stored_state_is_detached_from_reused_fit_array_and_target_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backing = np.zeros((6, 3), dtype=np.float64)
    intended = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    observed: list[float] = []
    calls = 0
    original_target = harness.make_score_only_target
    original_eta = harness.heldout_raw_eta_pairs

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        backing.fill(intended[calls])
        calls += 1
        return fake_canary_result(Z=backing, w0=0.0, w=0.1)

    def target_spy(*args: object, **kwargs: object) -> harness.ScoreOnlyTarget:
        target = original_target(*args, **kwargs)
        target.values.flags.writeable = True
        target.values[:] = 1.0 - target.values
        target.values.flags.writeable = False
        # Target materialization occurs only after all fits.  Mutating the fake
        # adapter's reusable buffer now cannot alter any stored fit snapshot.
        backing.fill(999.0)
        return target

    def eta_spy(
        Z: np.ndarray, w0: float, w: float, test_mask: np.ndarray
    ) -> harness.EtaPairs:
        assert not Z.flags.writeable
        observed.append(float(Z[0, 0]))
        return original_eta(Z, w0, w, test_mask)

    monkeypatch.setattr(harness, "make_score_only_target", target_spy)
    monkeypatch.setattr(harness, "heldout_raw_eta_pairs", eta_spy)
    run_static_smoke(fake_fit)
    assert calls == 6
    assert observed == intended


def test_static_canary_identical_complete_outputs_pass() -> None:
    calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result()

    report = run_static_canary(fake_fit)
    assert calls == 2
    assert report.initialization_equal and report.final_outputs_equal


def test_static_canary_fit_provenance_and_training_inputs_are_identical() -> None:
    inputs = canary_inputs()
    prepared = inputs["prepared"]
    assert isinstance(prepared, harness.PreparedTrainingData)
    fit_hash_before = prepared.fit_provenance_hash
    calls: list[dict[str, object]] = []

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        calls.append(kwargs)
        return fake_canary_result()

    run_static_canary(fake_fit, inputs)
    assert len(calls) == 2
    assert prepared.fit_provenance_hash == fit_hash_before
    assert np.array_equal(calls[0]["X"], calls[1]["X"])
    assert np.array_equal(calls[0]["train_mask"], calls[1]["train_mask"])
    train_mask = np.asarray(calls[0]["train_mask"])
    Y_a = np.asarray(calls[0]["Y"])
    Y_b = np.asarray(calls[1]["Y"])
    assert np.array_equal(Y_a[train_mask], Y_b[train_mask])
    assert set(np.unique(Y_a[~train_mask & ~np.eye(6, dtype=bool)])) == {0.0}
    assert set(np.unique(Y_b[~train_mask & ~np.eye(6, dtype=bool)])) == {1.0}


def test_static_canary_seals_boundary_and_finishes_fits_before_target_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_factory = harness.FitCallBoundary._from_preflight_test_only
    original_target = harness.make_score_only_target

    def sealing_spy(*args: object, **kwargs: object) -> harness.FitCallBoundary:
        events.append("boundary-sealed")
        return original_factory(*args, **kwargs)

    def target_spy(*args: object, **kwargs: object) -> harness.ScoreOnlyTarget:
        events.append("target-created")
        return original_target(*args, **kwargs)

    monkeypatch.setattr(
        harness.FitCallBoundary,
        "_from_preflight_test_only",
        staticmethod(sealing_spy),
    )
    monkeypatch.setattr(harness, "make_score_only_target", target_spy)

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        events.append("fit")
        return fake_canary_result()

    run_static_canary(fake_fit)
    assert events == ["boundary-sealed", "fit", "fit", "target-created"]


@pytest.mark.parametrize(
    ("second_change", "message"),
    [
        (
            {
                "initialization": harness.InitializationSnapshot(
                    np.full((6, 3), 1.1), np.ones((2, 3)), -0.5, 0.5, 1.0
                )
            },
            "initialization.Z",
        ),
        ({"Q_strict": -122.0}, "Q_strict"),
        ({"warnings": ("RuntimeWarning: changed",)}, "canary B emitted warnings"),
        ({"w": float("nan")}, "nonfinite"),
    ],
)
def test_static_canary_rejects_output_init_q_retry_warning_and_nan(
    second_change: dict[str, object], message: str
) -> None:
    calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result(**(second_change if calls == 2 else {}))

    with pytest.raises(harness.HarnessStop, match=message):
        run_static_canary(fake_fit)


def test_static_canary_rejects_one_changed_final_element() -> None:
    calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return fake_canary_result()
        changed = np.full((6, 3), 2.0)
        changed[0, 0] += 1e-3
        return fake_canary_result(Z=changed)

    with pytest.raises(harness.HarnessStop, match="final.Z"):
        run_static_canary(fake_fit)


@pytest.mark.parametrize(
    ("first_change", "message"),
    [({"internal_retry": 1}, "canary A internal_retry"), ({"q_failure": True}, "canary A Q failure")],
)
def test_static_canary_rejects_retry_or_q_failure_in_a(
    first_change: dict[str, object], message: str
) -> None:
    calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result(**(first_change if calls == 1 else {}))

    with pytest.raises(harness.HarnessStop, match=message):
        run_static_canary(fake_fit)


@pytest.mark.parametrize(
    ("warnings_a", "warnings_b", "message"),
    [
        (("same",), (), "canary A emitted warnings"),
        ((), ("same",), "canary B emitted warnings"),
        (("same",), ("same",), "canary A emitted warnings"),
        (("warning-a",), ("warning-b",), "canary A emitted warnings"),
    ],
)
def test_static_canary_blocks_every_nonempty_warning_state(
    warnings_a: tuple[str, ...], warnings_b: tuple[str, ...], message: str
) -> None:
    calls = 0

    def fake_fit(**_: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        return fake_canary_result(warnings=warnings_a if calls == 1 else warnings_b)

    with pytest.raises(harness.HarnessStop, match=message):
        run_static_canary(fake_fit)


def test_static_canary_rejects_target_bearing_fit_callable() -> None:
    inputs = canary_inputs()
    score_Y = inputs["score_Y"]
    prepared = inputs["prepared"]
    assert isinstance(score_Y, np.ndarray)
    assert isinstance(prepared, harness.PreparedTrainingData)
    target = harness.make_score_only_target(score_Y, prepared.test_mask)

    class TargetBearingFit:
        def __init__(self, target: object) -> None:
            self.target = target

        def __call__(self, **_: object) -> harness.CanaryFitResult:
            return fake_canary_result()

    with pytest.raises(harness.HarnessStop, match="ScoreOnlyTarget"):
        harness._make_test_fit_adapter(
            TargetBearingFit(target), score_targets=(target,)
        )


def test_static_canary_rejects_unvalidated_preflight() -> None:
    inputs = canary_inputs()
    valid = inputs["preflight"]
    assert isinstance(valid, harness.CanaryPreflight)
    inputs["preflight"] = harness.CanaryPreflight(
        valid.replicate, valid.train_mask_hash, valid.test_mask_hash, None
    )
    with pytest.raises(harness.HarnessStop, match="invalid or unvalidated"):
        run_static_canary(lambda **_: fake_canary_result(), inputs)


def test_initialization_capture_spy_observes_informed_init_once_without_extra_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    init_calls = 0
    rng_calls = 0
    snapshot_calls = 0
    original_snapshot = harness.snapshot_initialization

    def snapshot_spy(model: object) -> harness.InitializationSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(model)

    monkeypatch.setattr(harness, "snapshot_initialization", snapshot_spy)

    class FakeModel:
        def __init__(self) -> None:
            self.params = {
                "Z": np.zeros((2, 1)),
                "F": np.full((1, 1), 5.0),
                "w0": 0.0,
                "w": 0.0,
            }
            self.sigma_y = 1.0

        def initialize_params(self) -> None:
            nonlocal init_calls
            init_calls += 1
            events.append("initialize")

        def calc_eta_newton(self) -> np.ndarray:
            nonlocal rng_calls
            rng_calls += 1
            events.append("first-e-step")
            return self.params["Z"]

    class FakeRunner:
        def build_model(self) -> FakeModel:
            return FakeModel()

        def run_em_experimental(self, **_: object) -> dict[str, object]:
            model = self.build_model()
            model.initialize_params()
            model.params["w0"] = -0.75
            model.params["w"] = 0.5
            model.params["F"] *= 0.2
            events.append("informed-init-complete")
            model.calc_eta_newton()
            model.calc_eta_newton()
            return {
                "Z_est": model.params["Z"],
                "F": model.params["F"],
                "w0": model.params["w0"],
                "w": model.params["w"],
                "sigma_y_est": model.sigma_y,
                "Q_strict": -1.0,
                "mstep_q_history": (),
                "q_bic_failed": False,
                "nan_occurred": False,
            }

    runner = FakeRunner()
    result = harness.run_em_with_initialization_capture(runner)
    assert events == ["initialize", "informed-init-complete", "first-e-step", "first-e-step"]
    assert init_calls == 1
    assert rng_calls == 2
    assert snapshot_calls == 1
    assert result.initialization.w0 == -0.75
    assert result.initialization.w == 0.5
    assert result.initialization.F[0, 0] == 1.0
    assert result.internal_retry == 0


def test_validate_only_never_calls_production_adapter_canary_or_smoke(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def exploding_canary() -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("canary/fit path must not run")

    def exploding_adapter() -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("production adapter must not be instantiated")

    monkeypatch.setattr(harness, "run_canary_cli", exploding_canary)
    monkeypatch.setattr(harness, "run_smoke_cli", exploding_canary)
    monkeypatch.setattr(harness, "AuthorizedEMFitAdapter", exploding_adapter)
    assert harness.main(["--validate-only"]) == 0
    assert calls == 0
    assert '"em_fits_executed": 0' in capsys.readouterr().out


def test_production_cli_has_no_test_adapter_or_callable_injection_api() -> None:
    signature = inspect.signature(harness.run_canary_cli)
    assert not signature.parameters
    source = inspect.getsource(harness.run_canary_cli)
    assert "_TestAuthorizedFitAdapter" not in source
    assert "_run_two_canary_falsification_test_only" not in source
    assert "authorize_x_payload" not in source
    assert "make_score_only_target" not in source
    assert source.index("prepare_training_data") < source.index("run_two_canary_falsification")
    smoke_source = inspect.getsource(harness.run_smoke_cli)
    assert "_TestAuthorizedFitAdapter" not in smoke_source
    assert "_run_smoke_selection_test_only" not in smoke_source
    assert "make_score_only_target" not in smoke_source
    assert smoke_source.index("prepare_training_data") < smoke_source.index("run_smoke_selection")
    assert list(inspect.signature(harness.main).parameters) == ["argv"]


def test_fit_phase_apis_cannot_receive_raw_y_or_score_target() -> None:
    for fit_phase in (
        harness._run_two_canary_fit_phase,
        harness._run_smoke_fit_phase,
    ):
        parameters = inspect.signature(fit_phase).parameters
        assert "score_Y" not in parameters
        assert "raw_Y" not in parameters
        assert "target" not in parameters
        source = inspect.getsource(fit_phase)
        assert "make_score_only_target" not in source


def test_canary_cli_requires_two_explicit_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(harness.HarnessStop, match="--allow-em"):
        harness.main(["--canary"])
    calls = 0

    def static_stub() -> harness.CanaryInvarianceReport:
        nonlocal calls
        calls += 1
        return harness.CanaryInvarianceReport("config", "a", "b", True, True, 0)

    monkeypatch.setattr(harness, "run_canary_cli", static_stub)
    assert harness.main(["--canary", "--allow-em"]) == 0
    assert calls == 1


def test_smoke_cli_requires_allow_em_and_static_stub_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def static_stub() -> harness.SmokeSelectionReport:
        nonlocal calls
        calls += 1
        return harness.SmokeSelectionReport(
            (), (), 2, (2,), 6, "score", "topology", "score-target"
        )

    monkeypatch.setattr(harness, "run_smoke_cli", static_stub)
    with pytest.raises(harness.HarnessStop, match="--allow-em"):
        harness.main(["--smoke"])
    assert calls == 0
    assert harness.main(["--smoke", "--allow-em"]) == 0
    assert calls == 1


def test_full_is_fail_closed_without_every_authorization_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 7e triple gate: --full alone and --full --allow-em perform 0 fits."""

    calls = 0

    def exploding(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("fit path must not run")

    monkeypatch.setattr(harness, "run_canary_cli", exploding)
    monkeypatch.setattr(harness, "run_smoke_cli", exploding)
    monkeypatch.setattr(harness, "run_full_pilot_cli", exploding)
    with pytest.raises(harness.HarnessStop, match="--allow-em"):
        harness.main(["--full"])
    assert calls == 0
    with pytest.raises(harness.HarnessStop, match="--confirm-full-pilot"):
        harness.main(["--full", "--allow-em"])
    assert calls == 0
    with pytest.raises(harness.HarnessStop, match="--allow-em"):
        harness.main(["--full", "--confirm-full-pilot"])
    assert calls == 0


def test_default_cli_does_not_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def exploding() -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("fit path must not run")

    monkeypatch.setattr(harness, "run_canary_cli", exploding)
    monkeypatch.setattr(harness, "run_smoke_cli", exploding)
    with pytest.raises(SystemExit):
        harness.main([])
    assert calls == 0


# ---------------------------------------------------------------------------
# Phase 7e full held-out K-selection pilot (Issue #43)
# ---------------------------------------------------------------------------


def full_replicate_inputs(
    *,
    replicates=harness.FULL_REPLICATES,
    zero_targets: bool = False,
) -> list[harness.FullReplicateInputs]:
    """Three tiny fake replicates that satisfy every frozen provenance gate."""

    manifest = harness.build_full_manifest()
    inputs: list[harness.FullReplicateInputs] = []
    for replicate in replicates:
        train, test = valid_masks()
        diagnostics = harness.validate_pair_masks(train, test, 3)
        plan = harness.SplitPlan(
            replicate, harness.SPLIT_SEED_BASE + replicate, 3, train, test, diagnostics
        )
        preflight = harness.authorize_canary_preflight(plan)
        # Distinct per-replicate data so cross-replicate provenance differs.
        Y = np.zeros((6, 6), dtype=float)
        if not zero_targets:
            upper_rows, upper_cols = np.where(np.triu(test, 1))
            for index, (row_index, col_index) in enumerate(
                zip(upper_rows.tolist(), upper_cols.tolist(), strict=True)
            ):
                value = float((index + replicate) % 3 == 0)
                Y[row_index, col_index] = value
                Y[col_index, row_index] = value
        X = np.full((6, 2), float(replicate), dtype=np.float64)
        prepared = harness.prepare_training_data(
            X, Y, preflight=preflight, train_mask=train, test_mask=test
        )
        subset = tuple(row for row in manifest if row.replicate == replicate)
        comparability = tuple(harness.build_full_comparability(prepared, subset))
        inputs.append(
            harness.FullReplicateInputs(
                replicate=replicate,
                preflight=preflight,
                prepared=prepared,
                score_Y=Y,
                manifest=subset,
                comparability=comparability,
            )
        )
    return inputs


def run_static_full_pilot(
    callback,
    inputs: list[harness.FullReplicateInputs] | None = None,
) -> harness.FullPilotReport:
    selected = full_replicate_inputs() if inputs is None else inputs
    adapter = harness._make_test_fit_adapter(callback, score_targets=())
    return harness._run_full_pilot_test_only(
        replicate_inputs=selected, adapter=adapter
    )


def full_fake_fit_factory(calls: list[tuple[int, int]]):
    """Deterministic fake fit whose score peaks at a K that is not K_TRUE."""

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        k = int(kwargs["k"])
        seed = int(kwargs["seed"])
        calls.append((k, seed))
        return fake_canary_result(w0=-float(k), w=0.0, Z=np.zeros((6, k)))

    return fake_fit


# --- 1-4: frozen manifest completeness -------------------------------------


def test_full_manifest_is_exactly_42_rows_in_frozen_order() -> None:
    manifest = harness.build_full_manifest()
    assert len(manifest) == 42
    assert harness.EXPECTED_FULL_FITS == 42
    expected = [
        (replicate, k, start)
        for replicate in (1, 2, 3)
        for k in (1, 2, 3, 4, 5, 6, 7)
        for start in (1, 2)
    ]
    assert [(row.replicate, row.k, row.start) for row in manifest] == expected
    harness.validate_full_manifest(manifest)


def test_full_manifest_replicate_set_is_exactly_one_two_three() -> None:
    manifest = harness.build_full_manifest()
    assert sorted({row.replicate for row in manifest}) == [1, 2, 3]
    assert harness.FULL_REPLICATES == (1, 2, 3)
    truncated = [row for row in manifest if row.replicate != 3]
    with pytest.raises(harness.HarnessStop):
        harness.validate_full_manifest(truncated)
    extended = [
        *manifest,
        *harness.build_manifest((4,), harness.FULL_K_CANDIDATES, harness.START_LABELS),
    ]
    with pytest.raises(harness.HarnessStop):
        harness.validate_full_manifest(extended)


def test_full_manifest_k_set_is_exactly_one_to_seven() -> None:
    manifest = harness.build_full_manifest()
    assert sorted({row.k for row in manifest}) == [1, 2, 3, 4, 5, 6, 7]
    assert harness.FULL_K_CANDIDATES == (1, 2, 3, 4, 5, 6, 7)
    without_k1 = [row for row in manifest if row.k != 1]
    with pytest.raises(harness.HarnessStop):
        harness.validate_full_manifest(without_k1)


def test_full_manifest_starts_are_exactly_one_and_two() -> None:
    manifest = harness.build_full_manifest()
    assert sorted({row.start for row in manifest}) == [1, 2]
    assert harness.START_LABELS == (1, 2)
    for row in manifest:
        assert row.model_seed == 43000 + row.replicate * 1000 + row.k * 10 + row.start
    assert manifest[0].model_seed == 44011
    assert manifest[1].model_seed == 44012
    assert manifest[13].model_seed == 44072
    assert manifest[14].model_seed == 45011
    assert manifest[-1].model_seed == 46072


def test_full_manifest_seed_convention_is_independent_of_harness_code() -> None:
    """Recompute every seed from the written convention, not from the harness."""

    manifest = harness.build_full_manifest()
    expected: list[tuple[int, int, int, int, int, int]] = []
    for replicate in (1, 2, 3):
        for k in (1, 2, 3, 4, 5, 6, 7):
            for start in (1, 2):
                expected.append(
                    (
                        replicate,
                        k,
                        start,
                        41000 + replicate,
                        42000 + replicate,
                        43000 + replicate * 1000 + k * 10 + start,
                    )
                )
    assert [
        (row.replicate, row.k, row.start, row.data_seed, row.split_seed, row.model_seed)
        for row in manifest
    ] == expected


# --- 5: all three splits preflighted before the first fit ------------------


def test_all_three_splits_are_validated_before_the_first_fake_fit() -> None:
    calls: list[tuple[int, int]] = []
    report = run_static_full_pilot(full_fake_fit_factory(calls))
    events = list(report.events)
    assert events[:3] == [
        ("preflight_validated", 1),
        ("preflight_validated", 2),
        ("preflight_validated", 3),
    ]
    assert events[3][0] == "fit"
    assert all(event[0] != "fit" for event in events[:3])


def test_full_pilot_requires_all_three_replicates_before_any_fit() -> None:
    calls: list[tuple[int, int]] = []
    partial = full_replicate_inputs(replicates=(1, 2))
    with pytest.raises(harness.HarnessStop, match="exactly three dataset replicates"):
        run_static_full_pilot(full_fake_fit_factory(calls), partial)
    assert calls == []


def test_full_pilot_rejects_out_of_order_replicates_before_any_fit() -> None:
    calls: list[tuple[int, int]] = []
    reordered = list(reversed(full_replicate_inputs()))
    with pytest.raises(harness.HarnessStop):
        run_static_full_pilot(full_fake_fit_factory(calls), reordered)
    assert calls == []


# --- 6-9: corrupted manifest rows / seeds stop before any fit --------------


@pytest.mark.parametrize(
    "corruption",
    ["missing_row", "duplicate_row", "extra_row", "reordered_rows"],
)
def test_full_pilot_rejects_wrong_missing_or_duplicate_row(corruption: str) -> None:
    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    rows = list(inputs[1].manifest)
    if corruption == "missing_row":
        rows = rows[:-1]
    elif corruption == "duplicate_row":
        rows[-1] = rows[0]
    elif corruption == "extra_row":
        rows = [*rows, rows[0]]
    else:
        rows = [rows[1], rows[0], *rows[2:]]
    inputs[1] = replace(inputs[1], manifest=tuple(rows))
    with pytest.raises(harness.HarnessStop):
        run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    assert calls == []


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("data_seed", 41999),
        ("split_seed", 42999),
        ("model_seed", 99999),
    ],
)
def test_full_pilot_rejects_wrong_seed_before_any_fit(
    field_name: str, bad_value: int
) -> None:
    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    rows = list(inputs[0].manifest)
    rows[5] = replace(rows[5], **{field_name: bad_value})
    inputs[0] = replace(inputs[0], manifest=tuple(rows))
    with pytest.raises(harness.HarnessStop):
        run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    assert calls == []


def test_full_pilot_rejects_uniformly_wrong_model_seed_before_any_fit() -> None:
    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    rows = [
        replace(row, model_seed=row.model_seed + 100000) for row in inputs[2].manifest
    ]
    inputs[2] = replace(inputs[2], manifest=tuple(rows))
    with pytest.raises(harness.HarnessStop):
        run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    assert calls == []


# --- 10-12: provenance / preprocessing hash corruption ---------------------


def test_full_pilot_rejects_uniform_wrong_fit_provenance_before_any_fit() -> None:
    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    corrupted = tuple(
        replace(row, fit_provenance_hash="0" * 64) for row in inputs[0].comparability
    )
    inputs[0] = replace(inputs[0], comparability=corrupted)
    with pytest.raises(harness.HarnessStop):
        run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    assert calls == []


def test_full_pilot_rejects_single_preprocessing_hash_corruption() -> None:
    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    rows = list(inputs[1].comparability)
    rows[7] = replace(rows[7], preprocessing_hash="f" * 64)
    inputs[1] = replace(inputs[1], comparability=tuple(rows))
    with pytest.raises(harness.HarnessStop, match="preprocessing_hash"):
        run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    assert calls == []


def test_full_pilot_rejects_uniform_preprocessing_hash_corruption() -> None:
    """Phase 7d LOW-02: uniqueness checks alone would miss this corruption."""

    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    corrupted = tuple(
        replace(row, preprocessing_hash="f" * 64) for row in inputs[1].comparability
    )
    assert len({row.preprocessing_hash for row in corrupted}) == 1
    inputs[1] = replace(inputs[1], comparability=corrupted)
    with pytest.raises(harness.HarnessStop, match="preprocessing_hash"):
        run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    assert calls == []


@pytest.mark.parametrize(
    "field_name",
    [
        "x_hash",
        "training_y_hash",
        "train_mask_hash",
        "test_mask_hash",
        "target_topology_hash",
        "score_config_hash",
    ],
)
def test_full_pilot_rejects_uniform_hash_corruption_for_every_field(
    field_name: str,
) -> None:
    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    corrupted = tuple(
        replace(row, **{field_name: "a" * 64}) for row in inputs[2].comparability
    )
    inputs[2] = replace(inputs[2], comparability=corrupted)
    with pytest.raises(harness.HarnessStop):
        run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    assert calls == []


# --- 13: held-out counterfactual -------------------------------------------


def test_full_pilot_y_test_counterfactual_changes_only_score_side() -> None:
    calls_a: list[tuple[int, int]] = []
    inputs_a = full_replicate_inputs()
    report_a = run_static_full_pilot(full_fake_fit_factory(calls_a), inputs_a)

    calls_b: list[tuple[int, int]] = []
    inputs_b = full_replicate_inputs()
    for item in inputs_b:
        # Mutate held-out outcomes only; fit-side state was frozen already.
        mask = item.prepared.test_mask
        item.score_Y[mask] = 1.0 - item.score_Y[mask]
    report_b = run_static_full_pilot(full_fake_fit_factory(calls_b), inputs_b)

    assert calls_a == calls_b
    for row_a, row_b in zip(report_a.rows, report_b.rows, strict=True):
        assert row_a.x_hash == row_b.x_hash
        assert row_a.training_y_hash == row_b.training_y_hash
        assert row_a.train_mask_hash == row_b.train_mask_hash
        assert row_a.test_mask_hash == row_b.test_mask_hash
        assert row_a.fit_provenance_hash == row_b.fit_provenance_hash
        assert row_a.preprocessing_hash == row_b.preprocessing_hash
        assert row_a.target_topology_hash == row_b.target_topology_hash
        assert row_a.score_config_hash == row_b.score_config_hash
        assert row_a.fit_config_hash == row_b.fit_config_hash
        assert row_a.score_target_hash != row_b.score_target_hash
        assert row_a.heldout_mean_log_score != row_b.heldout_mean_log_score


def test_full_pilot_every_boundary_receives_training_only_y() -> None:
    inputs = full_replicate_inputs()
    for item in inputs:
        item.score_Y[item.prepared.test_mask] = 1.0
    test_masks = {item.replicate: item.prepared.test_mask for item in inputs}
    assert all(np.any(item.score_Y[item.prepared.test_mask] == 1.0) for item in inputs)
    raw_targets = [item.score_Y for item in inputs]
    seen = 0

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        nonlocal seen
        seen += 1
        fit_Y = np.asarray(kwargs["Y"])
        for raw in raw_targets:
            assert fit_Y is not raw
            assert not np.shares_memory(fit_Y, raw)
        assert any(
            np.all(fit_Y[mask] == 0.0) for mask in test_masks.values()
        )
        return fake_canary_result(Z=np.zeros((6, int(kwargs["k"]))))

    run_static_full_pilot(fake_fit, inputs)
    assert seen == 42


# --- 14-16: failure timing and target creation timing ----------------------


@pytest.mark.parametrize(
    ("failure_call", "expected_targets"),
    [(1, 0), (14, 0), (15, 1), (28, 1), (29, 2), (42, 2)],
)
def test_full_pilot_global_stop_never_targets_an_incomplete_replicate(
    failure_call: int, expected_targets: int
) -> None:
    calls = 0
    targets: list[object] = []
    original = harness.make_score_only_target

    def counting_target(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        targets.append(result)
        return result

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        changes: dict[str, object] = {"Z": np.zeros((6, int(kwargs["k"])))}
        if calls == failure_call:
            changes["q_failure"] = True
        return fake_canary_result(**changes)

    saved = harness.make_score_only_target
    harness.make_score_only_target = counting_target  # type: ignore[assignment]
    try:
        with pytest.raises(harness.HarnessStop, match="Q failure"):
            run_static_full_pilot(fake_fit)
    finally:
        harness.make_score_only_target = saved  # type: ignore[assignment]
    assert calls == failure_call
    assert len(targets) == expected_targets


def test_full_pilot_creates_one_target_per_replicate_after_its_fourteen_fits() -> None:
    calls: list[tuple[int, int]] = []
    report = run_static_full_pilot(full_fake_fit_factory(calls))
    events = list(report.events)
    fit_events = [event for event in events if event[0] == "fit"]
    target_events = [event for event in events if event[0] == "target_create"]
    score_events = [event for event in events if event[0] == "score"]
    assert len(fit_events) == 42
    assert len(target_events) == 3
    assert len(score_events) == 42
    assert report.targets_created == 3
    for index, replicate in enumerate(harness.FULL_REPLICATES):
        offset = 3 + index * 29
        assert [event[1] for event in events[offset : offset + 14]] == [replicate] * 14
        assert all(event[0] == "fit" for event in events[offset : offset + 14])
        assert events[offset + 14] == ("target_create", replicate)
        assert all(
            event[0] == "score" for event in events[offset + 15 : offset + 29]
        )


def test_full_pilot_produces_exactly_three_targets_and_42_score_rows() -> None:
    calls: list[tuple[int, int]] = []
    report = run_static_full_pilot(full_fake_fit_factory(calls))
    assert report.em_fits_executed == 42
    assert report.targets_created == 3
    assert report.score_rows == 42
    assert len(report.rows) == 42
    assert len(calls) == 42
    assert calls == [
        (row.k, row.model_seed) for row in harness.build_full_manifest()
    ]
    assert len({row.score_target_hash for row in report.rows}) == 3
    for replicate in harness.FULL_REPLICATES:
        replicate_rows = [row for row in report.rows if row.replicate == replicate]
        assert len(replicate_rows) == 14
        assert len({row.x_hash for row in replicate_rows}) == 1
        assert len({row.training_y_hash for row in replicate_rows}) == 1
        assert len({row.train_mask_hash for row in replicate_rows}) == 1
        assert len({row.test_mask_hash for row in replicate_rows}) == 1
        assert len({row.fit_provenance_hash for row in replicate_rows}) == 1
        assert len({row.preprocessing_hash for row in replicate_rows}) == 1
        assert len({row.target_topology_hash for row in replicate_rows}) == 1
        assert len({row.score_target_hash for row in replicate_rows}) == 1
        assert len({row.score_config_hash for row in replicate_rows}) == 1
        assert len({row.fit_config_hash for row in replicate_rows}) == 14


# --- 17-19: aggregation and selector ---------------------------------------


@pytest.mark.parametrize(
    ("failure_call", "changes", "message"),
    [
        (1, {"internal_retry": 1}, "internal_retry"),
        (14, {"warnings": ("warning",)}, "emitted warnings"),
        (15, {"nan_occurred": True}, "NaN/nonfinite"),
        (28, {"Q_strict": float("nan")}, "Q_strict is nonfinite"),
        (29, {"w": float("nan")}, "nonfinite"),
        (42, {"q_failure": True}, "Q failure"),
    ],
)
def test_full_pilot_never_aggregates_only_successful_fits(
    failure_call: int, changes: dict[str, object], message: str
) -> None:
    calls = 0

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        nonlocal calls
        calls += 1
        values: dict[str, object] = {"Z": np.zeros((6, int(kwargs["k"])))}
        if calls == failure_call:
            values.update(changes)
        return fake_canary_result(**values)

    with pytest.raises(harness.HarnessStop, match=message):
        run_static_full_pilot(fake_fit)
    assert calls == failure_call


def test_full_pilot_selector_runs_independently_inside_every_replicate() -> None:
    # score decreases with K inside replicate 1, peaks at K=4 in replicate 2,
    # and peaks at K=7 in replicate 3.
    peaks = {1: 1, 2: 4, 3: 7}
    order = harness.build_full_manifest()
    index = 0

    # All held-out targets are 0, so the Bernoulli log score is strictly
    # decreasing in eta; a larger penalty therefore yields a worse score.
    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        nonlocal index
        row = order[index]
        index += 1
        assert int(kwargs["k"]) == row.k
        assert int(kwargs["seed"]) == row.model_seed
        penalty = abs(row.k - peaks[row.replicate])
        return fake_canary_result(
            w0=-5.0 + 0.1 * penalty, w=0.0, Z=np.zeros((6, row.k))
        )

    report = run_static_full_pilot(
        fake_fit, full_replicate_inputs(zero_targets=True)
    )
    assert [item.selected_k for item in report.replicate_selections] == [1, 4, 7]
    assert report.true_k == 3
    assert report.true_k_selected_count == 0
    assert report.descriptive_recovery_rate == 0.0
    assert report.selected_k_counts == ((1, 1), (4, 1), (7, 1))
    for selection in report.replicate_selections:
        assert len(selection.summaries) == 7
        for summary in selection.summaries:
            assert summary.mean_score == pytest.approx(
                (summary.start_1_score + summary.start_2_score) / 2, abs=0.0, rel=0.0
            )
        assert selection.best_mean_score >= selection.second_best_mean_score
        assert selection.margin == pytest.approx(
            selection.best_mean_score - selection.second_best_mean_score
        )
    assert len(report.k_aggregates) == 7
    for aggregate in report.k_aggregates:
        values = [
            summary.mean_score
            for selection in report.replicate_selections
            for summary in selection.summaries
            if summary.k == aggregate.k
        ]
        assert len(values) == 3
        assert aggregate.mean_across_replicates == pytest.approx(float(np.mean(values)))
        assert aggregate.std_across_replicates == pytest.approx(
            float(np.std(values, ddof=1))
        )
        assert aggregate.min_across_replicates == pytest.approx(min(values))
        assert aggregate.max_across_replicates == pytest.approx(max(values))


def test_full_pilot_recovery_rate_is_descriptive_over_three_replicates() -> None:
    order = harness.build_full_manifest()
    index = 0
    peaks = {1: 3, 2: 3, 3: 6}

    def fake_fit(**kwargs: object) -> harness.CanaryFitResult:
        nonlocal index
        row = order[index]
        index += 1
        penalty = abs(row.k - peaks[row.replicate])
        return fake_canary_result(
            w0=-5.0 + 0.1 * penalty, w=0.0, Z=np.zeros((6, row.k))
        )

    report = run_static_full_pilot(
        fake_fit, full_replicate_inputs(zero_targets=True)
    )
    assert [item.selected_k for item in report.replicate_selections] == [3, 3, 6]
    assert report.n_replicates == 3
    assert report.true_k_selected_count == 2
    assert report.descriptive_recovery_rate == pytest.approx(2 / 3)


# --- 20: CLI surface --------------------------------------------------------


def test_full_pilot_cli_exposes_no_k_start_seed_or_tolerance_option() -> None:
    parser = harness._build_parser()
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--validate-only",
        "--canary",
        "--smoke",
        "--full",
        "--allow-em",
        "--confirm-full-pilot",
    }
    for bad in (
        ["--full", "--allow-em", "--confirm-full-pilot", "--k", "5"],
        ["--full", "--allow-em", "--confirm-full-pilot", "--seed", "1"],
        ["--full", "--allow-em", "--confirm-full-pilot", "--replicates", "5"],
        ["--full", "--allow-em", "--confirm-full-pilot", "--tie-tolerance", "1e-3"],
    ):
        with pytest.raises(SystemExit):
            harness._build_parser().parse_args(bad)


def test_full_pilot_cli_requires_three_gates_and_calls_runner_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def static_stub(command: str) -> dict[str, object]:
        calls.append(command)
        return {"mode": "full", "em_fits_executed": 42}

    monkeypatch.setattr(harness, "run_full_pilot_cli", static_stub)
    with pytest.raises(harness.HarnessStop, match="--allow-em"):
        harness.main(["--full"])
    with pytest.raises(harness.HarnessStop, match="--confirm-full-pilot"):
        harness.main(["--full", "--allow-em"])
    assert calls == []
    assert harness.main(["--full", "--allow-em", "--confirm-full-pilot"]) == 0
    assert len(calls) == 1
    assert "--full --allow-em --confirm-full-pilot" in calls[0]


def test_validate_only_covers_the_frozen_42_row_manifest_without_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exploding(*args: object, **kwargs: object) -> None:
        raise AssertionError("fit path must not run")

    monkeypatch.setattr(harness.AuthorizedEMFitAdapter, "fit", exploding)
    monkeypatch.setattr(harness, "run_full_pilot_cli", exploding)
    result = harness.run_validate_only()
    assert result["em_fits_executed"] == 0
    assert result["full_pilot_manifest_keys"] == 42
    assert result["full_pilot_expected_fits"] == 42
    assert result["full_pilot_replicates"] == [1, 2, 3]
    assert result["full_pilot_k_candidates"] == [1, 2, 3, 4, 5, 6, 7]
    assert result["full_pilot_starts"] == [1, 2]
    assert result["full_pilot_split_preflight_pass"] == 3


def test_full_pilot_production_entry_rejects_the_test_adapter() -> None:
    inputs = full_replicate_inputs()
    adapter = harness._make_test_fit_adapter(
        lambda **kwargs: fake_canary_result(), score_targets=()
    )
    with pytest.raises(harness.HarnessStop, match="production full pilot requires"):
        harness.run_full_pilot(replicate_inputs=inputs, adapter=adapter)  # type: ignore[arg-type]


def test_full_pilot_output_artifacts_are_frozen_and_never_overwritten(
    tmp_path: Path,
) -> None:
    assert harness.FULL_PILOT_OUTPUT_DIR.name == "heldout_full_pilot_20260824"
    assert set(harness.FULL_PILOT_ARTIFACT_NAMES) == {
        "manifest.csv",
        "fit_results.csv",
        "replicate_selection.csv",
        "aggregate_summary.csv",
        "score_by_k.csv",
        "runinfo.json",
        "runinfo.md",
    }
    harness._require_no_existing_full_artifacts(tmp_path)
    (tmp_path / "fit_results.csv").write_text("x", encoding="utf-8")
    with pytest.raises(harness.HarnessStop, match="refusing to overwrite"):
        harness._require_no_existing_full_artifacts(tmp_path)


def test_full_pilot_rejects_unexpected_generated_artifact(tmp_path: Path) -> None:
    (tmp_path / "runinfo.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stdout.log").write_text("", encoding="utf-8")
    assert harness._require_only_expected_artifacts(tmp_path) == [
        "runinfo.json",
        "stdout.log",
    ]
    (tmp_path / "sneaky.csv").write_text("", encoding="utf-8")
    with pytest.raises(harness.HarnessStop, match="unexpected generated artifact"):
        harness._require_only_expected_artifacts(tmp_path)


def test_full_pilot_runinfo_contains_every_required_provenance_field() -> None:
    calls: list[tuple[int, int]] = []
    inputs = full_replicate_inputs()
    report = run_static_full_pilot(full_fake_fit_factory(calls), inputs)
    manifest = harness.build_full_manifest()
    runinfo = harness.build_full_pilot_runinfo(
        command="python tools/research_audit/run_heldout_k_selection_pilot.py --full",
        branch="experiment/full-heldout-k-selection-pilot",
        run_code_sha="0" * 40,
        base_main_sha="1" * 40,
        started_utc="2026-08-24T00:00:00+00:00",
        started_local="2026-08-24T09:00:00+09:00",
        finished_utc="2026-08-24T01:00:00+00:00",
        manifest=manifest,
        replicate_inputs=inputs,
        report=report,
        git_status_before="",
        git_status_after="",
        failure_state="none",
        artifacts=["manifest.csv"],
    )
    required = {
        "issue",
        "branch",
        "run_code_sha",
        "base_main_sha",
        "timestamp_utc_start",
        "timestamp_local_start",
        "command",
        "python_version",
        "numpy_version",
        "platform",
        "config",
        "candidate_k",
        "starts",
        "replicates",
        "manifest",
        "data_seeds",
        "split_seeds",
        "model_seeds",
        "per_replicate_provenance",
        "git_status_before",
        "git_status_after_scientific_execution",
        "stdout_log",
        "expected_fit_count",
        "actual_fit_count",
        "failure_state",
        "generated_artifacts",
    }
    assert required <= set(runinfo)
    assert runinfo["issue"] == 43
    assert len(runinfo["manifest"]) == 42
    assert len(runinfo["model_seeds"]) == 42
    assert runinfo["data_seeds"] == [41001, 41002, 41003]
    assert runinfo["split_seeds"] == [42001, 42002, 42003]
    assert runinfo["expected_fit_count"] == 42
    assert runinfo["actual_fit_count"] == 42
    assert len(runinfo["per_replicate_provenance"]) == 3
    for entry in runinfo["per_replicate_provenance"]:
        for key in (
            "x_hash",
            "training_y_hash",
            "train_mask_hash",
            "test_mask_hash",
            "fit_provenance_hash",
            "target_topology_hash",
            "preprocessing_hash",
            "score_config_hash",
            "score_target_hash",
        ):
            assert entry[key]
    # JSON must be serialisable without NaN for the immutable record.
    json.dumps(runinfo, allow_nan=False)
    assert harness._render_runinfo_markdown(runinfo).startswith("# Phase 7e")


def test_full_pilot_result_csvs_are_generated_from_machine_readable_results(
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, int]] = []
    report = run_static_full_pilot(full_fake_fit_factory(calls))
    harness.write_full_pilot_manifest_csv(tmp_path, harness.build_full_manifest())
    harness.write_full_pilot_result_csvs(tmp_path, report)
    manifest_rows = list(
        csv.DictReader((tmp_path / "manifest.csv").read_text(encoding="utf-8").splitlines())
    )
    assert len(manifest_rows) == 42
    fit_rows = list(
        csv.DictReader((tmp_path / "fit_results.csv").read_text(encoding="utf-8").splitlines())
    )
    assert len(fit_rows) == 42
    assert all(row["fit_status"] == "clean" for row in fit_rows)
    assert all(row["retry"] == "0" for row in fit_rows)
    selection_rows = list(
        csv.DictReader(
            (tmp_path / "replicate_selection.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    assert len(selection_rows) == 21
    aggregate_rows = list(
        csv.DictReader(
            (tmp_path / "aggregate_summary.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    assert sum(1 for row in aggregate_rows if row["section"] == "k_wise") == 7
    assert {
        row["key"] for row in aggregate_rows if row["section"] == "pilot"
    } == {
        "n_replicates",
        "true_k",
        "selected_k_counts",
        "true_k_selected_count",
        "descriptive_recovery_rate",
    }
    score_rows = list(
        csv.DictReader((tmp_path / "score_by_k.csv").read_text(encoding="utf-8").splitlines())
    )
    assert len(score_rows) == 7
    # every stored float must round-trip exactly
    for row, source in zip(fit_rows, report.rows, strict=True):
        assert float(row["heldout_mean_log_score"]) == source.heldout_mean_log_score
