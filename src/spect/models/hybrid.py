"""A 3D CNN encoder fused with tabular features, trained on a Cox objective.

The scaffold for the hybrid arm. Deliberately small: it exists so the training
path can be exercised end to end before real data arrives, not to be a good
model.

**Read D5 before believing any number this produces.** On the current phantom,
region-mean features are close to sufficient statistics for the outcome unless
`severity_from_gradient` is set above zero. Trained on a cohort without that
term, this network is learning to reconstruct a binding ratio and its score is
a smoke test, not evidence about the hybrid hypothesis. With the term set, the
information a network could add is present and a posterior/anterior split
already recovers much of it, which is the bar a network has to clear to have
earned its complexity.

**The loss is the Cox partial likelihood**, the same objective
`models/cox.py` maximizes. That is deliberate: a network trained on a
different objective and then compared on concordance is being compared on
something neither model optimized, and differences in the comparison metric
then partly reflect the mismatch. A test checks this implementation against
the numpy one, which is itself tested against values worked out by hand.

**Training is full batch.** A mini-batch Cox loss can only build risk sets
from the subjects in the batch, so it silently optimizes a different quantity
that approaches the real one only as the batch approaches the cohort. At the
cohort sizes this program will see, full batch is exact and cheap. If batching
ever becomes necessary, that approximation must be stated wherever the result
is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "the hybrid model needs torch. Install the CPU build with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


def cox_partial_likelihood_loss(
    risk: Tensor, time: Tensor, event: Tensor
) -> Tensor:
    """Negative Breslow partial log-likelihood, averaged over events.

    Same convention as `models/cox.py`: a pair is comparable when the earlier
    time is an observed event, tied event times share the full risk set, and
    the linear predictor is centred before exponentiating because the shift
    cancels in the ratio.

    Averaged rather than summed so the scale of the loss does not depend on
    how many events a cohort happens to contain, which would otherwise make a
    learning rate cohort-specific.
    """
    if risk.ndim != 1:
        risk = risk.reshape(-1)
    if not torch.any(event):
        raise ValueError(
            "no events in this batch: a partial likelihood is built entirely "
            "from event times, so there is nothing to optimize"
        )

    order = torch.argsort(time, stable=True)
    risk_sorted = risk[order] - risk.mean()
    event_sorted = event[order]

    # Reverse cumulative sum gives every risk-set total in one pass, because
    # sorting ascending makes each risk set a suffix of the array.
    weights = torch.exp(risk_sorted)
    reverse_cumsum = torch.flip(torch.cumsum(torch.flip(weights, [0]), 0), [0])

    # Tied times take the risk set of the earliest row in their group.
    times_sorted = time[order]
    first_of_tie = torch.searchsorted(times_sorted, times_sorted, right=False)
    risk_totals = reverse_cumsum[first_of_tie]

    contributions = risk_sorted[event_sorted] - torch.log(risk_totals[event_sorted])
    return -contributions.mean()


class VolumeEncoder(nn.Module):
    """A small 3D convolutional encoder.

    Three blocks and a global average pool, which is enough to represent a
    spatial gradient inside a structure and little enough to train on a CPU.
    Global pooling rather than a flatten so the encoder does not depend on the
    input grid size, letting the phantom resolution change without rewriting
    the head.
    """

    def __init__(self, channels: int = 8, embedding: int = 32) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(channels * 2, embedding, kernel_size=3, padding=1),
            nn.BatchNorm3d(embedding),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.embedding = embedding

    def forward(self, volume: Tensor) -> Tensor:
        return self.blocks(volume).flatten(1)


class HybridSurvivalNet(nn.Module):
    """Volume encoder fused with tabular features, emitting a risk score.

    The output is a log hazard ratio on the same scale as `CoxModel.risk`,
    with no activation on the final layer. Higher means earlier expected
    event, which is what `concordance_index` expects; squashing it would
    change the loss surface without changing the ordering, and adding a sign
    flip somewhere downstream is how a model ends up scoring 1 - C.
    """

    def __init__(
        self,
        n_tabular: int,
        channels: int = 8,
        embedding: int = 32,
        hidden: int = 32,
        dropout: float = 0.2,
        seed: int | None = None,
    ) -> None:
        """`seed` fixes the weight initialization.

        It belongs here rather than only in `train`, because initialization
        happens at construction. Seeding only inside the training loop leaves
        every run starting from different weights while appearing to be
        seeded, which is reproducibility that is claimed and not delivered.

        Note that this sets the global torch RNG, as most initialization does.
        Pass None to leave the caller's RNG state alone.
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.encoder = VolumeEncoder(channels=channels, embedding=embedding)
        self.head = nn.Sequential(
            nn.Linear(embedding + n_tabular, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.n_tabular = n_tabular

    def forward(self, volume: Tensor, tabular: Tensor) -> Tensor:
        if tabular.shape[1] != self.n_tabular:
            raise ValueError(
                f"model expects {self.n_tabular} tabular feature(s), "
                f"got {tabular.shape[1]}"
            )
        fused = torch.cat([self.encoder(volume), tabular], dim=1)
        return self.head(fused).reshape(-1)


@dataclass
class TrainingResult:
    """What happened during a fit, so it can be reported rather than assumed."""

    losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    epochs: int = 0
    best_epoch: int = 0

    @property
    def improved(self) -> bool:
        return bool(self.losses) and self.losses[-1] < self.losses[0]


def train(
    model: HybridSurvivalNet,
    volumes: np.ndarray,
    tabular: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    epochs: int = 60,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> TrainingResult:
    """Full-batch training on the Cox partial likelihood.

    Deterministic in `seed`. Volumes arrive as (N, D, H, W) and a channel axis
    is added here, so callers do not have to remember which position it goes
    in.
    """
    torch.manual_seed(seed)

    volume_tensor = _as_volume_tensor(volumes)
    tabular_tensor = torch.as_tensor(tabular, dtype=torch.float32)
    time_tensor = torch.as_tensor(np.asarray(time), dtype=torch.float32)
    event_tensor = torch.as_tensor(np.asarray(event), dtype=torch.bool)

    if len(volume_tensor) != len(tabular_tensor):
        raise ValueError(
            f"{len(volume_tensor)} volume(s) and {len(tabular_tensor)} tabular row(s)"
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    result = TrainingResult()
    best = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = cox_partial_likelihood_loss(
            model(volume_tensor, tabular_tensor), time_tensor, event_tensor
        )
        loss.backward()
        optimizer.step()
        result.losses.append(float(loss.detach()))

        if validation is not None:
            model.eval()
            with torch.no_grad():
                v_vol, v_tab, v_time, v_event = validation
                v_loss = cox_partial_likelihood_loss(
                    model(_as_volume_tensor(v_vol), torch.as_tensor(v_tab, dtype=torch.float32)),
                    torch.as_tensor(np.asarray(v_time), dtype=torch.float32),
                    torch.as_tensor(np.asarray(v_event), dtype=torch.bool),
                )
            result.validation_losses.append(float(v_loss))
            if float(v_loss) < best:
                best = float(v_loss)
                result.best_epoch = epoch

        result.epochs = epoch + 1

    return result


def predict_risk(
    model: HybridSurvivalNet, volumes: np.ndarray, tabular: np.ndarray
) -> np.ndarray:
    """Risk scores, on the same scale as `CoxModel.risk`.

    Puts the model in eval mode, which matters here: the head carries dropout
    and the encoder carries batch norm, and leaving them in training mode
    makes predictions depend on the rest of the batch.
    """
    model.eval()
    with torch.no_grad():
        risk = model(
            _as_volume_tensor(volumes),
            torch.as_tensor(tabular, dtype=torch.float32),
        )
    return risk.numpy()


def _as_volume_tensor(volumes: np.ndarray) -> Tensor:
    array = np.asarray(volumes, dtype=np.float32)
    if array.ndim == 4:
        array = array[:, None]  # add the channel axis
    elif array.ndim != 5:
        raise ValueError(
            f"expected volumes shaped (N, D, H, W) or (N, C, D, H, W), got {array.shape}"
        )
    return torch.as_tensor(array)


def normalize_volumes(volumes: np.ndarray, reference_level: np.ndarray) -> np.ndarray:
    """Divide each volume by its own reference level.

    Without this the network can read injected dose and camera sensitivity
    straight off the raw counts, which vary between subjects for reasons that
    have nothing to do with disease. Dividing by the occipital reference is
    what the binding ratio does too, so both arms see the same normalization
    and a difference between them is not just a difference in preprocessing.
    """
    reference = np.asarray(reference_level, dtype=np.float64).reshape(-1)
    if reference.size != len(volumes):
        raise ValueError(
            f"{reference.size} reference level(s) for {len(volumes)} volume(s)"
        )
    if np.any(reference <= 0):
        raise ValueError("reference levels must be positive")
    return np.asarray(volumes, dtype=np.float64) / reference[:, None, None, None]
