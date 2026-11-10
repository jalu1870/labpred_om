"""Inference and evaluation.

Metrics follow the paper:
  * scores are sigmoid(mean of fold logits);
  * per-diagnosis metrics use the windows at risk for that diagnosis;
  * a grouped endpoint scores each window by the maximum over its member diagnoses, is positive if
    any member is positive and is evaluated on windows at risk for at least one member;
  * PR-AUC is the trapezoidal area under the precision-recall curve; enrichment is PR-AUC divided by
    prevalence;
  * uncertainty: 1,000 bootstrap resamples of windows (or of patients, clustered) with medians and
    2.5/97.5 % percentiles.
"""
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from .data import MAIN_GROUPS, group_indices


@torch.no_grad()
def predict_logits(batch, imputation_model, diagnosis_models, device="cpu"):
    """Logits [n_folds, n_windows, n_diagnoses] for a collated batch (see `data.collate`)."""
    values = batch["values"].to(device)
    att, pos, sex = batch["attention"].to(device), batch["position_ids"].to(device), batch["sex"].to(device)
    imputed = imputation_model.to(device).impute(values, sex, att, pos)
    observed = ~torch.isnan(values)
    return torch.stack([m.to(device)(imputed, observed, sex, att, pos, bidirectional=False).float().cpu()
                        for m in diagnosis_models])


def probabilities(logits_stack):
    return torch.sigmoid(torch.as_tensor(logits_stack).float().mean(0)).numpy()


def pr_auc(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    order = np.argsort(recall)
    return auc(recall[order], precision[order])


def grouped(probs, labels, at_risk, indices):
    """(labels, scores, at-risk) of a grouped endpoint."""
    return labels[:, indices].max(1), probs[:, indices].max(1), at_risk[:, indices].max(1) == 1


def endpoint_metrics(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    prevalence = y.mean()
    ap = pr_auc(y, p)
    return {"auroc": roc_auc_score(y, p), "pr_auc": ap, "prevalence": prevalence, "enrichment": ap / prevalence}


def per_diagnosis_auroc(probs, labels, at_risk, min_events=10):
    """AUROC of every diagnosis with at least `min_events` positive windows: {index: auroc}."""
    out = {}
    for c in range(labels.shape[1]):
        m = at_risk[:, c] == 1
        y = labels[m, c]
        if min_events <= y.sum() < len(y):
            out[c] = roc_auc_score(y, probs[m, c])
    return out


def bootstrap(probs, labels, at_risk, groups=None, n_resamples=1000, seed=5, patient_ids=None):
    """Bootstrap the grouped endpoints. Resample r uses numpy.random.default_rng(seed + r).

    With `patient_ids`, patients are resampled with replacement (cluster bootstrap) instead of windows.
    Returns {group: {metric: (median, lower, upper)}}; enrichment is median PR-AUC / median prevalence.
    """
    groups = groups or group_indices(MAIN_GROUPS)
    arrays = {g: grouped(probs, labels, at_risk, idx) for g, idx in groups.items()}
    if patient_ids is not None:
        _, inverse = np.unique(patient_ids, return_inverse=True)
        clusters = [np.where(inverse == k)[0] for k in range(inverse.max() + 1)]
    draws = {g: {"auroc": [], "pr_auc": [], "prevalence": []} for g in groups}
    n = len(labels)
    for r in range(n_resamples):
        rng = np.random.default_rng(seed + r)
        if patient_ids is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([clusters[k] for k in rng.integers(0, len(clusters), len(clusters))])
        for g, (y, p, m) in arrays.items():
            res = endpoint_metrics(y[idx][m[idx]], p[idx][m[idx]])
            if res is not None:
                for k in draws[g]:
                    draws[g][k].append(res[k])
    out = {}
    for g, d in draws.items():
        if not d["auroc"]:  # no resample with both classes
            out[g] = {k: (np.nan,) * 3 for k in ("auroc", "pr_auc", "prevalence", "enrichment")}
            continue
        summary = {k: (np.median(v), np.quantile(v, 0.025), np.quantile(v, 0.975)) for k, v in d.items()}
        prev = summary["prevalence"][0]
        summary["enrichment"] = tuple(x / prev for x in summary["pr_auc"])
        out[g] = summary
    return out


def summary_table(probs, labels, at_risk, patient_ids=None, n_resamples=1000):
    """Per grouped endpoint: AUROC [95% CI], PR-AUC, enrichment [95% CI] (window bootstrap) and,
    if `patient_ids` is given, AUROC [95% CI] from the patient-clustered bootstrap."""
    windows = bootstrap(probs, labels, at_risk, n_resamples=n_resamples)
    patients = bootstrap(probs, labels, at_risk, n_resamples=n_resamples, patient_ids=patient_ids) if patient_ids is not None else None
    rows = []
    for g, s in windows.items():
        row = {"endpoint": g, "auroc": s["auroc"], "pr_auc": s["pr_auc"][0], "enrichment": s["enrichment"]}
        if patients is not None:
            row["auroc_patient_bootstrap"] = patients[g]["auroc"]
        rows.append(row)
    return rows
