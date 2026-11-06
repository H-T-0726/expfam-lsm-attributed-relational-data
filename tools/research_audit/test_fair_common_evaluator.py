"""Tests for the fair-comparison common evaluator and equivalence checks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_common_evaluator as ev  # noqa: E402
import fair_model_equivalence as fe  # noqa: E402


def test_all_equivalence_checks_behave_as_documented():
    df = fe.checks()
    expected_fail = df.note.str.contains("expected to FAIL")
    assert df.loc[~expected_fail, "pass"].all(), df.loc[~expected_fail & ~df["pass"], "check"].tolist()
    assert (~df.loc[expected_fail, "pass"]).all()


def test_q_terms_closure_and_prior_identity_under_unit_mean_square():
    X, Y, Z, th = fe.instance("gaussian", n=12)
    Zs = np.stack([Z, Z * 1.1, Z * 0.9], axis=2)
    Zs = Zs / np.sqrt(np.mean(Zs ** 2))           # scale_Z convention
    q = ev.q_terms(X, Y, Zs, th, "gaussian")
    n, k, _ = Zs.shape
    assert abs(-2 * q["Q_Z"] - n * k * (1 + np.log(2 * np.pi))) < 1e-9
    c = ev.criteria_abc(X, Y, Zs, Zs[:, :, -1], th, "gaussian")
    assert abs((c["A_CQ"] - c["B_CQ_minus_PZ"]) - c["P_Z"]) < 1e-9


def test_mask_excludes_pairs():
    X, Y, Z, th = fe.instance("gaussian", n=8)
    mask = np.ones((8, 8), bool)
    mask[0, 1] = mask[1, 0] = False
    full = ev.loglik_y(Y, Z, th["w0"], th["w"])
    eta = th["w0"] + th["w"] * Z[0] @ Z[1]
    one = Y[0, 1] * eta - np.logaddexp(0, eta)
    assert abs(full - one - ev.loglik_y(Y, Z, th["w0"], th["w"], mask)) < 1e-10
