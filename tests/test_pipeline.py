"""End-to-end smoke tests on synthetic patients (no real data required)."""
import random

import numpy as np
import pytest
import torch

from labforecast import baselines, data, evaluate, train
from labforecast.model import DiagnosisModel, ImputationModel

N_DIAG = 162


def synthetic_records(n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    records = []
    for i in range(n):
        T = int(torch.randint(5, 60, (1,), generator=g))
        values = torch.randn(T, 40, generator=g)
        values[torch.rand(T, 40, generator=g) < 0.6] = float("nan")
        values[:, 0] = torch.randn(1, generator=g)
        days = torch.cumsum(torch.randint(1, 15, (T,), generator=g), 0) - 1
        E = int(torch.randint(1, 8, (1,), generator=g))
        records.append({"values": values, "days": days, "sex": i % 2,
                        "event_codes": torch.randint(0, N_DIAG, (E,), generator=g),
                        "event_days": torch.randint(-50, int(days[-1]) + 800, (E,), generator=g)})
    return records


def test_codes_and_groups():
    codes = data.diagnosis_codes()
    assert len(codes) == N_DIAG and codes[157] == "DEATH"
    groups = data.group_indices(data.MAIN_GROUPS)
    assert groups["All-cause mortality"] == [157] and len(groups["Myelodysplastic syndrome"]) == 3


def test_windows_labels_are_at_risk_only():
    rng = random.Random(0)
    for r in synthetic_records():
        for w in data.make_windows(r, N_DIAG, mask_together=data.mask_together_indices(), rng=rng):
            assert not (w["label"] & ~w["at_risk"]).any()
            s, e = w["rows"]
            assert 0 <= s < e <= len(r["days"])
            assert int(r["days"][e - 1] - r["days"][s]) <= 29


def test_models_train_and_evaluate():
    records = synthetic_records()
    imputer = ImputationModel(hidden_size=32)
    train.impute_records(records, imputer, device="cpu")
    batch = data.collate(records, [data.make_windows(r, N_DIAG, rng=random.Random(1)) for r in records])
    assert not torch.isnan(batch["imputed"]).any()
    model = DiagnosisModel(hidden_size=32)
    logits = model(batch["imputed"], ~torch.isnan(batch["values"]), batch["sex"], batch["attention"], batch["position_ids"])
    loss = train.focal_loss(logits, batch["labels"], batch["at_risk"], torch.ones(N_DIAG))
    loss.backward()
    assert torch.isfinite(loss)
    stacked = evaluate.predict_logits(batch, imputer, [model.eval(), model.eval()])
    assert stacked.shape == (2, len(batch["labels"]), N_DIAG)


def test_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = y * 0.3 + rng.random(500)
    m = evaluate.endpoint_metrics(y, p)
    assert 0.5 < m["auroc"] <= 1 and m["enrichment"] == pytest.approx(m["pr_auc"] / y.mean())
    probs = rng.random((300, N_DIAG)); labels = (rng.random((300, N_DIAG)) < 0.1).astype(float)
    out = evaluate.bootstrap(probs, labels, np.ones_like(labels), n_resamples=20)
    assert set(out) == set(data.MAIN_GROUPS)


def test_trajectory_features():
    records = synthetic_records(5)
    batch = data.collate(records, [data.make_windows(r, N_DIAG, rng=random.Random(2)) for r in records])
    X = baselines.trajectory_features(batch, n_jobs=1)
    assert X.shape == (len(batch["labels"]), 40 * baselines.N_STATS + 3)
    assert len(baselines.FAMILIES["xgboost_all"]) == X.shape[1]
