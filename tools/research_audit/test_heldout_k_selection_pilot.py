from __future__ import annotations

import inspect
import functools
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


def test_full_remains_fail_closed_without_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def exploding() -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("fit path must not run")

    monkeypatch.setattr(harness, "run_canary_cli", exploding)
    monkeypatch.setattr(harness, "run_smoke_cli", exploding)
    with pytest.raises(harness.HarnessStop, match="full pilot"):
        harness.main(["--full", "--allow-em"])
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
