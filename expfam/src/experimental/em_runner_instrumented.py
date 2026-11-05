"""Instrumented MCEM loop for the convergence-sensitivity experiment.

``em_runner.run_em_experimental`` is left BYTE-UNCHANGED: it is the code path
that produced every archived result, and rewriting it would silently redefine
what those results mean.  This module owns its own copy of the same loop so
that it can

  * evaluate the selection criteria at several iteration counts inside ONE run
    (checkpoints), instead of paying for a separate run per budget,
  * record per-iteration budget-stability diagnostics, and
  * make a NaN restart VISIBLE (``em_runner`` resets ``nan_count`` per retry
    and breaks on the first clean attempt, so an internal restart is
    indistinguishable from a run that never had one -- finding F-08).

WHAT THE EQUIVALENCE CLAIM IS
-----------------------------
This is a **clean-path mirror for the frozen Poisson-X / Bernoulli-Y
configuration**, not a line-for-line reimplementation of ``em_runner`` in
general.  Precisely:

  * On a run in which no NaN latent sample ever occurs, with
    ``family_x='poisson'``, ``family_y='bernoulli'`` and
    ``numerics_mode='consistent'``, the sequence of operations, the RNG
    consumption and the end-of-run Q/BIC block match ``em_runner``.  That is
    measured, not asserted (probe I1), and re-proved on the real cells by the
    replication gate and by the all-fit checkpoint-8 comparison.
  * The NaN / retry / exception paths are NOT the same.  ``em_runner``
    restarts the whole run up to twice with ``seed + attempt*1000``; production
    here raises :class:`RetryDetected` on the first NaN instead.  A restart
    re-seeds, so its iterations are no longer the ones the archived baseline
    took and that fit's pairing is void.  Consequently the scientific evidence
    from this experiment covers the clean path only, and a NaN is an attempt
    failure rather than a result.
  * Families other than the frozen configuration are out of scope of the
    equivalence claim entirely.

Nothing here decides anything.  There is no adaptive stopping: every fit runs
the full budget, so the iteration count cannot depend on K, on the data, or on
a held-out score.  The diagnostics are DESCRIPTIVE budget-stability measures,
not a convergence proof -- ``L`` latent draws are redrawn every iteration, so
none of them decays to zero at a fixed point.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from em_runner import build_model                              # noqa: E402
from eval_utils import calc_Q_dual_strict_exp, calc_bic_exp    # noqa: E402

RUNNER_VERSION = "em-instrumented-v3"

#: Denominators use ``1 + |x|`` rather than ``|x| + eps``.  A pure relative
#: change explodes when the parameter passes near zero, which for ``w`` and
#: ``w0`` is an ordinary event, not a signal.  Mixed absolute-relative scaling
#: keeps the diagnostic bounded there while staying scale-aware for large
#: values.  Fixed here, before any result exists.
_MIXED_OFFSET = 1.0

#: Every per-iteration diagnostic.  Parameter-space movement (``f_raw``) is
#: kept for comparison but is NOT sufficient on its own; ``gram`` is rotation
#: invariant, and the two ``eta`` measures are what the likelihood actually
#: sees.
DIAGNOSTIC_KEYS = ("gram_rel_change", "f_raw_change", "w_change", "w0_change",
                   "eta_x_rel_change", "eta_y_rel_change")


class RetryDetected(RuntimeError):
    """A NaN restart was requested while ``on_nan='fail'``.

    Production treats an internal retry as a fit failure rather than as
    something to absorb: the restart re-seeds with ``seed + 1000``, so the
    run's first iterations are no longer the ones the archived baseline took,
    and the paired comparison for that fit is void.
    """

    def __init__(self, iteration: int, attempt: int):
        super().__init__(
            f"NaN latent sample at iteration {iteration} of attempt {attempt}; "
            f"a restart would break pairing with the archived baseline")
        self.iteration = iteration
        self.attempt = attempt


def _frobenius(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)))


def _read_only(a: np.ndarray) -> np.ndarray:
    """A copy the callback cannot write through.

    The callback receives state the loop is still using.  Handing it a
    writeable alias would let a scoring bug silently perturb the estimation
    path, which is the one thing the checkpoint design must not permit.
    """

    out = np.array(a, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _predictors(z: np.ndarray, f: np.ndarray, w0: float, w: float,
                dyad_rows: np.ndarray, dyad_cols: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """The two natural parameters the likelihood is actually a function of.

    ``F`` and ``Z`` are identified only up to an orthogonal transform, and even
    ``F F^T`` says nothing about where the latent positions went.  ``eta_X`` and
    ``eta_Y`` are invariant to that rotation by construction and are what both
    the objective and the held-out score consume, so movement in them is the
    movement that matters.  ``eta_Y`` is evaluated on a FIXED dyad index set so
    that the diagnostic is comparable across iterations and across budgets.
    """

    eta_x = z @ f.T
    inner = np.einsum("ij,ij->i", z[dyad_rows], z[dyad_cols])
    eta_y = w0 + w * inner
    return eta_x, eta_y


def _step_diagnostics(f_now, f_prev, w_now, w_prev, w0_now, w0_prev,
                      eta_x_now, eta_x_prev, eta_y_now, eta_y_prev
                      ) -> dict[str, float]:
    """Budget-stability diagnostics for one iteration.

    Read ``eta_x_rel_change`` and ``eta_y_rel_change`` first: they measure the
    predictors.  ``gram_rel_change`` is next, being invariant to the rotation
    that ``f_raw_change`` mistakes for movement.  ``f_raw_change`` is recorded
    only so the gap between it and the invariant measures is visible; it must
    NOT be used alone to call a fit stable.
    """

    gram_now = f_now @ f_now.T
    gram_prev = f_prev @ f_prev.T
    return {
        "gram_rel_change": (_frobenius(gram_now - gram_prev)
                            / (_MIXED_OFFSET + _frobenius(gram_prev))),
        "f_raw_change": (_frobenius(f_now - f_prev)
                         / (_MIXED_OFFSET + _frobenius(f_prev))),
        "w_change": abs(w_now - w_prev) / (_MIXED_OFFSET + abs(w_prev)),
        "w0_change": abs(w0_now - w0_prev) / (_MIXED_OFFSET + abs(w0_prev)),
        "eta_x_rel_change": (_frobenius(eta_x_now - eta_x_prev)
                             / (_MIXED_OFFSET + _frobenius(eta_x_prev))),
        "eta_y_rel_change": (_frobenius(eta_y_now - eta_y_prev)
                             / (_MIXED_OFFSET + _frobenius(eta_y_prev))),
    }


def _aggregate(diags: Sequence[dict[str, float]], window: int) -> dict[str, float]:
    recent = list(diags)[-window:]
    if not recent:
        return {f"max_{k}_last{window}": float("nan") for k in DIAGNOSTIC_KEYS}
    return {f"max_{k}_last{window}": max(float(d[k]) for d in recent)
            for k in DIAGNOSTIC_KEYS}


def run_em_instrumented(
    X, Y,
    *,
    family_x: str,
    family_y: str,
    k: int,
    L: int,
    max_iter: int,
    checkpoints: Sequence[int],
    seed: int,
    train_mask,
    numerics_mode: str = "consistent",
    stability_window: int = 4,
    on_nan: str = "fail",
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    on_iteration: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run one MCEM fit, evaluating Q and BIC at each requested checkpoint.

    ``on_checkpoint`` is called with the live model state at each checkpoint,
    BEFORE the next E-step overwrites it.  The sweep computes the held-out
    score and S3 inside this callback so every criterion sees exactly the state
    an equivalent short run would have ended on.  ``Z_est``, ``F``, ``sigma``
    and the predictor arrays handed to it are read-only copies, so a callback
    cannot write through into the estimation path.

    ``on_nan='fail'`` (production) raises :class:`RetryDetected` instead of
    restarting.  ``'legacy_retry'`` reproduces ``em_runner``'s restart
    behaviour and exists only so tests can exercise the path the archived
    baseline used.
    """

    if on_nan not in ("fail", "legacy_retry"):
        raise ValueError(f"on_nan must be 'fail' or 'legacy_retry', got {on_nan!r}")
    checkpoints = tuple(int(c) for c in checkpoints)
    if list(checkpoints) != sorted(set(checkpoints)):
        raise ValueError("checkpoints must be sorted and unique")
    if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > int(max_iter):
        raise ValueError("checkpoints must lie in 1..max_iter")

    n, d = X.shape
    max_retries = 2 if on_nan == "legacy_retry" else 0
    t0 = time.perf_counter()

    retry_events: list[dict[str, Any]] = []
    nan_detected = False

    # Fixed dyad index set for the eta_Y diagnostic: the TRAIN dyads, in a
    # deterministic order.  Train only -- no test dyad enters any recorded
    # diagnostic, so the diagnostics cannot carry held-out information even in
    # principle.
    train_arr = np.asarray(train_mask, dtype=bool)
    dyad_rows, dyad_cols = np.where(np.triu(train_arr, 1))

    for retry in range(max_retries + 1):
        # ---- mirror of em_runner: schedule, generator, informed init -------
        newton_alpha = 0.5 / (2 ** retry)
        attempt_seed = seed + retry * 1000
        rng = np.random.default_rng(attempt_seed)
        model = build_model(n, d, k, L, family_x, family_y,
                            nb_r=None, train_mask=train_mask,
                            family_x_list=None, numerics_mode=numerics_mode)
        model.initialize_params(true_params=None, seed=attempt_seed)

        obs_upper = np.triu(model.train_mask, k=1)
        y_obs = Y[obs_upper]
        if family_y == "bernoulli":
            density = float(np.clip(y_obs.mean(), 1e-6, 1 - 1e-6))
            model.params["w0"] = np.log(density / (1 - density))
            model.params["w"] = 0.5
        elif family_y in ("poisson", "nb"):
            mean_cnt = float(y_obs[y_obs > 0].mean()) if np.any(y_obs > 0) else 1.0
            model.params["w0"] = np.log(mean_cnt + 1e-10)
            model.params["w"] = 0.1 / (2 ** retry)
        else:  # gaussian
            model.params["w0"] = float(y_obs.mean())
            model.params["w"] = 0.5
            model.sigma_y = float(max(y_obs.std(), 0.01))

        if family_x in ("bernoulli", "poisson"):
            model.params["F"] *= 0.2

        Z = model.params["Z"].copy()
        F = model.params["F"].copy()
        sigma = model.params["sigma"].copy()
        w0 = float(model.params["w0"])
        w = float(model.params["w"])
        var_z = float(model.params["var_z"])

        Z_prev = Z.copy()
        nan_count = 0
        step_diagnostics: list[dict[str, float]] = []
        checkpoint_records: list[dict[str, Any]] = []
        attempt_dirty = False
        eta_x_prev, eta_y_prev = _predictors(Z, F, w0, w, dyad_rows, dyad_cols)

        for iteration in range(1, int(max_iter) + 1):
            # Announced BEFORE any work, so a test can prove which iterations
            # were entered -- in particular that a checkpoint callback which
            # raises stops the loop before the next one begins.
            if on_iteration is not None:
                on_iteration(iteration)
            f_before, w_before, w0_before = F.copy(), w, w0

            # ---- E-step (identical draw order to em_runner) ---------------
            Z_samples = np.zeros((n, k, L))
            for sample in range(L):
                model.params.update(dict(Z=Z.copy(), F=F, sigma=sigma,
                                         w0=w0, w=w))
                Z_new = model.calc_eta_newton(
                    X, Y, rng=rng, max_iter=10, alpha=newton_alpha)
                Z_samples[:, :, sample] = Z_new
                Z = Z_new.copy()

            if np.any(np.isnan(Z_samples)) or np.any(np.isinf(Z_samples)):
                nan_detected = True
                retry_events.append({
                    "attempt": retry, "iteration": iteration,
                    "kind": "nan_latent_sample", "attempt_seed": attempt_seed,
                })
                if on_nan == "fail":
                    raise RetryDetected(iteration=iteration, attempt=retry)
                nan_count += 1
                attempt_dirty = True
                Z_samples = np.stack([Z_prev] * L, axis=2)
                Z = Z_prev.copy()

            Z_samples = model.scale_Z(Z_samples)
            Z_prev = Z.copy()
            Z = Z_samples[:, :, -1].copy()

            # ---- M-step ---------------------------------------------------
            F = model.calc_F(X, Z_samples)
            sigma = model.calc_sigma(X, Z_samples, F)
            w0 = float(model.calc_w0(Y, Z_samples, w0, w, max_iter=50))
            w = float(model.calc_w(Y, Z_samples, w0, w, max_iter=50))
            if family_y == "gaussian":
                model.calc_sigma_y(Y, Z_samples, w0, w)

            eta_x_now, eta_y_now = _predictors(Z, F, w0, w, dyad_rows, dyad_cols)
            step_diagnostics.append(_step_diagnostics(
                F, f_before, w, w_before, w0, w0_before,
                eta_x_now, eta_x_prev, eta_y_now, eta_y_prev))
            eta_x_prev, eta_y_prev = eta_x_now, eta_y_now

            # ---- checkpoint ------------------------------------------------
            if iteration in checkpoints:
                # em_runner performs this same update before its final Q/BIC
                # block; the next E-step overwrites it, so the estimation path
                # is unchanged (measured, not assumed: probes N2 and I3).
                model.params.update({"F": F, "sigma": sigma, "w0": w0, "w": w})
                q_strict = float(calc_Q_dual_strict_exp(
                    X, Y, Z_samples, F, sigma, var_z, w0, w, model))
                label = getattr(model, "family_y_label", model.family)
                n_gauss_cols = (len(model.columns_of("gaussian"))
                                if model.family_x == "mixed" else None)
                bic, npar = calc_bic_exp(q_strict, k, n, d,
                                         model.family_x, label,
                                         n_gaussian_x_cols=n_gauss_cols)
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "Z_est": _read_only(Z_samples[:, :, -1]),
                    # The FULL Monte-Carlo draw the Q above was computed from.
                    # Read-only, and archived, because without it no auditor
                    # can recompute Q at this checkpoint (finding H2-01).
                    "Z_samples": _read_only(Z_samples),
                    "F": _read_only(F), "sigma": _read_only(sigma),
                    "var_z": var_z, "w0": w0, "w": w, "model": model,
                    "q_strict": q_strict, "bic": float(bic),
                    "num_params": int(npar),
                    "cumulative_runtime_s": round(time.perf_counter() - t0, 3),
                    "nan_iterations_so_far": nan_count,
                    "attempt_index": retry,
                }
                record.update(step_diagnostics[-1])
                record.update(_aggregate(step_diagnostics, stability_window))
                checkpoint_records.append(record)
                if on_checkpoint is not None:
                    on_checkpoint(record)

        if not attempt_dirty:
            break

    return {
        "runner_version": RUNNER_VERSION,
        "checkpoint_records": checkpoint_records,
        "iteration_diagnostics": step_diagnostics,
        "step_diagnostics": step_diagnostics,
        "actual_iterations": int(max_iter),
        "stopping_reason": "fixed_budget",
        "retry_events": retry_events,
        "retry_count": len(retry_events),
        "retry_seed": seed + len(retry_events) * 1000 if retry_events else seed,
        "nan_detected": bool(nan_detected),
        "nan_count_total_across_attempts": len(retry_events),
        "attempts_used": retry + 1,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "numerics_mode": numerics_mode,
        "on_nan": on_nan,
        "dyad_count": int(dyad_rows.size),
    }
