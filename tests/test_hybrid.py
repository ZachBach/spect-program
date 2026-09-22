"""Tests for the hybrid network scaffold.

The first test is the one that makes the rest worth running: the torch loss is
checked against the numpy partial log-likelihood, which is itself checked
against values worked out by hand. A loss that trains smoothly and optimizes
the wrong objective looks exactly like one that works.

torch is an optional dependency, so the whole module skips without it. Nothing
else in the codebase imports it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spect.eval.survival import SurvivalData, concordance_index  # noqa: E402
from spect.models.cox import partial_log_likelihood  # noqa: E402
from spect.models.hybrid import (  # noqa: E402
    HybridSurvivalNet,
    cox_partial_likelihood_loss,
    normalize_volumes,
    predict_risk,
    train,
)


def tensors(time, event, risk):
    return (
        torch.tensor(risk, dtype=torch.float32),
        torch.tensor(time, dtype=torch.float32),
        torch.tensor(event, dtype=torch.bool),
    )


# --- the objective ----------------------------------------------------------


def test_the_torch_loss_matches_the_numpy_partial_likelihood():
    """Cross-validated against an implementation tested by hand.

    The numpy version returns the summed log-likelihood; this returns the
    negative mean over events, so the two differ by exactly that factor.
    """
    rng = np.random.default_rng(0)
    n = 80
    time = rng.uniform(1.0, 10.0, n)
    event = rng.random(n) < 0.6
    risk = rng.normal(size=n)

    data = SurvivalData(time=time, event=event)
    expected = -partial_log_likelihood(np.array([1.0]), risk[:, None], data) / event.sum()

    loss = cox_partial_likelihood_loss(*tensors(time, event, risk))
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_the_loss_matches_on_a_hand_computed_case():
    """Three subjects, all events at times 1, 2, 3, risk 0, 1, 2. The summed
    log-likelihood is -log(1 + e + e^2) - log(1 + e); averaged over three
    events and negated, that is about 1.24029.
    """
    loss = cox_partial_likelihood_loss(*tensors([1.0, 2.0, 3.0], [1, 1, 1], [0.0, 1.0, 2.0]))
    assert float(loss) == pytest.approx(3.720868 / 3, abs=1e-5)


def test_tied_event_times_share_a_risk_set():
    """Same Breslow convention as the numpy implementation. At zero risk the
    summed likelihood is -2 log(3), so the averaged negative is 2 log(3) / 3."""
    loss = cox_partial_likelihood_loss(*tensors([1.0, 1.0, 2.0], [1, 1, 1], [0.0, 0.0, 0.0]))
    assert float(loss) == pytest.approx(2 * np.log(3) / 3, abs=1e-6)


def test_shifting_risk_does_not_change_the_loss():
    args = ([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 1], [0.2, -0.5, 1.0, 0.3])
    base = cox_partial_likelihood_loss(*tensors(*args))
    shifted = cox_partial_likelihood_loss(
        *tensors(args[0], args[1], [v + 50.0 for v in args[2]])
    )
    assert float(shifted) == pytest.approx(float(base), abs=1e-4)


def test_a_batch_with_no_events_is_refused():
    with pytest.raises(ValueError, match="no events"):
        cox_partial_likelihood_loss(*tensors([1.0, 2.0], [0, 0], [0.5, 0.2]))


def test_the_loss_is_differentiable():
    risk = torch.zeros(4, requires_grad=True)
    loss = cox_partial_likelihood_loss(
        risk, torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([True, True, False, True])
    )
    loss.backward()
    assert risk.grad is not None and torch.any(risk.grad != 0)


# --- the network ------------------------------------------------------------


def toy(n=60, size=8, seed=0):
    """Volumes whose mean intensity carries the risk, plus a tabular echo.

    Trivially learnable on purpose. This checks the training path runs and
    optimizes in the right direction, not that the architecture is any good.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    volumes = rng.normal(0.0, 0.1, size=(n, size, size, size)) + signal[:, None, None, None]
    tabular = signal[:, None] + rng.normal(0.0, 0.1, size=(n, 1))
    time = rng.exponential(1.0 / np.exp(signal))
    event = rng.random(n) < 0.75
    return volumes, tabular, time, event


def test_the_forward_pass_gives_one_risk_per_subject():
    volumes, tabular, _, _ = toy(n=12)
    model = HybridSurvivalNet(n_tabular=1)
    risk = predict_risk(model, volumes, tabular)
    assert risk.shape == (12,)
    assert np.all(np.isfinite(risk))


def test_the_wrong_number_of_tabular_features_is_refused():
    volumes, _, _, _ = toy(n=8)
    model = HybridSurvivalNet(n_tabular=3)
    with pytest.raises(ValueError, match="3 tabular feature"):
        predict_risk(model, volumes, np.zeros((8, 1)))


def test_training_reduces_the_loss():
    volumes, tabular, time, event = toy(seed=1)
    model = HybridSurvivalNet(n_tabular=1, channels=4, embedding=8, hidden=8)
    result = train(model, volumes, tabular, time, event, epochs=40, seed=1)
    assert result.epochs == 40
    assert result.improved


def test_the_trained_model_orders_risk_the_right_way():
    """A sign error anywhere would put this below 0.5, which is the failure
    mode worth catching early since it looks like a merely weak model."""
    volumes, tabular, time, event = toy(n=120, seed=2)
    model = HybridSurvivalNet(n_tabular=1, channels=4, embedding=8, hidden=8)
    train(model, volumes, tabular, time, event, epochs=80, seed=2)

    data = SurvivalData(time=time, event=event)
    assert concordance_index(data, predict_risk(model, volumes, tabular)).index > 0.6


def test_training_is_deterministic_when_the_model_is_also_seeded():
    """Seeding the training loop alone is not enough, because weights are
    initialized at construction, before `train` runs. Found by this test
    failing: two runs with the same training seed diverged entirely."""
    volumes, tabular, time, event = toy(n=40, seed=3)

    def run():
        model = HybridSurvivalNet(n_tabular=1, channels=4, embedding=8, hidden=8, seed=7)
        train(model, volumes, tabular, time, event, epochs=15, seed=7)
        return predict_risk(model, volumes, tabular)

    assert np.allclose(run(), run())


def test_an_unseeded_model_starts_from_different_weights():
    """The failure the test above guards, pinned so nobody reinstates it by
    making the seed optional in a way that silently does nothing."""
    volumes, tabular, _, _ = toy(n=10, seed=8)
    first = predict_risk(HybridSurvivalNet(n_tabular=1, seed=None), volumes, tabular)
    second = predict_risk(HybridSurvivalNet(n_tabular=1, seed=None), volumes, tabular)
    assert not np.allclose(first, second)


def test_the_same_construction_seed_gives_the_same_initial_weights():
    volumes, tabular, _, _ = toy(n=10, seed=9)
    first = predict_risk(HybridSurvivalNet(n_tabular=1, seed=11), volumes, tabular)
    second = predict_risk(HybridSurvivalNet(n_tabular=1, seed=11), volumes, tabular)
    assert np.allclose(first, second)


def test_a_validation_set_is_tracked_separately():
    volumes, tabular, time, event = toy(n=80, seed=4)
    model = HybridSurvivalNet(n_tabular=1, channels=4, embedding=8, hidden=8)
    result = train(
        model, volumes[:50], tabular[:50], time[:50], event[:50],
        epochs=20, seed=4,
        validation=(volumes[50:], tabular[50:], time[50:], event[50:]),
    )
    assert len(result.validation_losses) == 20
    assert 0 <= result.best_epoch < 20


def test_prediction_puts_the_model_in_eval_mode():
    """Dropout and batch norm otherwise make a subject's prediction depend on
    whoever else is in the batch, which is not a property a prediction should
    have."""
    volumes, tabular, _, _ = toy(n=30, seed=5)
    model = HybridSurvivalNet(n_tabular=1, dropout=0.5)
    model.train()
    first = predict_risk(model, volumes, tabular)
    second = predict_risk(model, volumes, tabular)
    assert np.allclose(first, second)
    assert not model.training


def test_mismatched_row_counts_are_refused():
    volumes, tabular, time, event = toy(n=20, seed=6)
    model = HybridSurvivalNet(n_tabular=1)
    with pytest.raises(ValueError, match="volume"):
        train(model, volumes[:10], tabular, time, event, epochs=1)


# --- normalization ----------------------------------------------------------


def test_normalizing_removes_a_per_subject_scale_factor():
    """Injected dose and camera sensitivity scale a whole volume for reasons
    unrelated to disease. Both arms divide by the reference region, so a
    difference between them is not a difference in preprocessing."""
    rng = np.random.default_rng(7)
    base = rng.uniform(1.0, 2.0, size=(5, 4, 4, 4))
    scales = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    scaled = base * scales[:, None, None, None]

    normalized = normalize_volumes(scaled, reference_level=scales)
    assert np.allclose(normalized, base)


def test_a_non_positive_reference_level_is_refused():
    with pytest.raises(ValueError, match="positive"):
        normalize_volumes(np.ones((2, 3, 3, 3)), reference_level=np.array([1.0, 0.0]))


def test_a_reference_level_per_volume_is_required():
    with pytest.raises(ValueError, match="reference level"):
        normalize_volumes(np.ones((3, 2, 2, 2)), reference_level=np.array([1.0]))


def test_volumes_of_the_wrong_rank_are_refused():
    model = HybridSurvivalNet(n_tabular=1)
    with pytest.raises(ValueError, match=r"\(N, D, H, W\)"):
        predict_risk(model, np.ones((4, 4)), np.zeros((4, 1)))
