"""Training: imputation model, imputation of patient timelines, and 5-fold diagnosis models."""
import copy
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup

from .data import SELECTION_GROUPS, collate, make_windows, mask_together_indices
from .model import DiagnosisModel, ImputationModel

log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _timelines(records):
    """Pad full patient timelines into one batch (no CLS row)."""
    values = pad_sequence([torch.as_tensor(r["values"]).float() for r in records], batch_first=True, padding_value=float("nan"))
    days = pad_sequence([torch.as_tensor(r["days"]).long() for r in records], batch_first=True)
    lengths = torch.tensor([len(r["days"]) for r in records])
    attention = (torch.arange(values.shape[1])[None] < lengths[:, None]).float()
    values[attention == 0] = 0.0
    return values, days, torch.tensor([int(r["sex"]) for r in records]), attention


# --- imputation model ------------------------------------------------------------------------------

def mask_for_reconstruction(values, observed, mask_prob, timepoint_prob):
    """Hide observed entries: of the selected ones 80 % are set missing, 10 % replaced by Gaussian noise
    (per-channel batch mean/std) and 10 % kept; additionally whole days are hidden with `timepoint_prob`."""
    B, T, C = values.shape
    selected = (torch.rand_like(values) < mask_prob) & observed
    kind = torch.rand_like(values)
    masked = values.clone()
    masked[(kind < 0.8) & selected] = float("nan")
    obs_vals = values.reshape(-1, C).masked_fill(~observed.reshape(-1, C), float("nan"))
    mean = torch.nanmean(obs_vals, 0)
    var = (obs_vals - mean).nan_to_num(0.0).pow(2).sum(0) / (~torch.isnan(obs_vals)).sum(0).clamp(min=1)
    std = torch.where(var > 0, var.sqrt(), torch.ones_like(var))
    noise = torch.randn_like(values) * std + mean
    replace = (kind >= 0.8) & (kind < 0.9) & selected
    masked[replace] = noise[replace]
    masked[(torch.rand(B, T, 1, device=values.device) < timepoint_prob) & observed] = float("nan")
    return masked


def train_imputation_model(train_records, val_records, cfg, device="cuda"):
    """Masked reconstruction on full timelines; loss = MSE over all observed entries."""
    set_seed(cfg["seed"])
    model = ImputationModel(**cfg["imputation_model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = get_cosine_schedule_with_warmup(opt, cfg["warmup_epochs"], cfg["max_epochs"])
    scaler = torch.amp.GradScaler()
    val_batches = []
    for i in range(0, len(val_records), cfg["batch_size"]):
        v, d, s, a = _timelines(val_records[i:i + cfg["batch_size"]])
        obs = ~torch.isnan(v) & (a[..., None] == 1)
        val_batches.append((mask_for_reconstruction(v, obs, cfg["val_mask_prob"], cfg["val_timepoint_prob"]), v, obs, d, s, a))

    def step(masked, v, obs, d, s, a):
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            recon = model(masked.to(device), s.to(device), a.to(device), d.to(device))
            return F.mse_loss(recon[obs.to(device)].float(), v.to(device)[obs.to(device)])

    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(cfg["max_epochs"]):
        model.train()
        order = list(range(len(train_records)))
        random.shuffle(order)
        for i in range(0, len(order), cfg["batch_size"]):
            v, d, s, a = _timelines([train_records[j] for j in order[i:i + cfg["batch_size"]]])
            obs = ~torch.isnan(v) & (a[..., None] == 1)
            loss = step(mask_for_reconstruction(v, obs, cfg["mask_prob"], cfg["timepoint_prob"]), v, obs, d, s, a)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([step(*b).item() for b in val_batches]))
        if val < best:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        log.info("imputation epoch %d val_loss %.4f (best %.4f @ %d)", epoch, val, best, best_epoch)
        if epoch - best_epoch > cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def impute_records(records, imputation_model, device="cuda"):
    """Add `imputed` to every record by imputing the full timeline of each patient."""
    model = imputation_model.to(device).eval()
    for r in records:
        v, d, s, a = _timelines([r])
        r["imputed"] = model.impute(v.to(device), s.to(device), a.to(device), d.to(device))[0].cpu()
    return records


# --- diagnosis model -------------------------------------------------------------------------------

def focal_loss(logits, labels, at_risk, pos_weight, gamma=2.0):
    """Focal binary cross-entropy with per-diagnosis positive weights, averaged over at-risk diagnoses."""
    p = torch.sigmoid(logits)
    p_t = p * labels + (1 - p) * (1 - labels)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    loss = (1 - p_t) ** gamma * bce * at_risk
    return (loss.sum(1) / at_risk.sum(1).clamp(min=1)).mean()


def positive_weights(records, n_diagnoses, max_weight=50.0, **window_kwargs):
    """(negatives / positives) per diagnosis over one pass of training windows, capped at `max_weight`."""
    pos, n = torch.zeros(n_diagnoses), 0
    for r in records:
        for w in make_windows(r, n_diagnoses, **window_kwargs):
            pos += w["label"].float()
            n += 1
    return ((n - pos) / (pos + 1e-8)).clamp(max=max_weight)


@torch.no_grad()
def selection_score(model, batch, groups, device):
    """Mean AUROC over the selection groups; a group's score is the maximum probability over its members
    at risk in that window."""
    logits = model(batch["imputed"].to(device), ~torch.isnan(batch["values"].to(device)), batch["sex"].to(device),
                   batch["attention"].to(device), batch["position_ids"].to(device))
    p, y, m = torch.sigmoid(logits.float()).cpu().numpy(), batch["labels"].numpy(), batch["at_risk"].numpy()
    aucs = []
    for idx in groups.values():
        keep = m[:, idx].max(1) == 1
        if not idx or keep.sum() == 0:
            continue
        yy = (y[:, idx] * m[:, idx]).max(1)[keep]
        pp = np.where(m[:, idx] == 1, p[:, idx], -np.inf).max(1)[keep]
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc_score(yy, pp))
    return float(np.mean(aucs))


def train_diagnosis_model(train_records, val_records, cfg, device="cuda"):
    n_diag = cfg["diagnosis_model"]["n_diagnoses"]
    win = dict(cfg["windows"], mask_together=mask_together_indices())
    rng = random.Random(cfg["seed"])
    val_batch = collate(val_records, [make_windows(r, n_diag, rng=rng, **dict(win, horizon=cfg["val_horizon"]))
                                      for r in val_records])
    pos_weight = positive_weights(train_records, n_diag, rng=rng, **win).to(device)

    model = DiagnosisModel(**cfg["diagnosis_model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = get_cosine_schedule_with_warmup(opt, cfg["warmup_epochs"], cfg["max_epochs"])
    best, best_epoch, best_state = -1.0, -1, None
    for epoch in range(cfg["max_epochs"]):
        model.train()
        order = list(range(len(train_records)))
        rng.shuffle(order)
        for i in range(0, len(order), cfg["batch_size"]):
            recs = [train_records[j] for j in order[i:i + cfg["batch_size"]]]
            batch = collate(recs, [make_windows(r, n_diag, rng=rng, **win) for r in recs])
            if batch is None:
                continue
            with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                logits = model(batch["imputed"].to(device), ~torch.isnan(batch["values"].to(device)), batch["sex"].to(device),
                               batch["attention"].to(device), batch["position_ids"].to(device))
                loss = focal_loss(logits.float(), batch["labels"].to(device), batch["at_risk"].to(device), pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        score = selection_score(model, val_batch, SELECTION_GROUPS, device)
        if score > best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        log.info("diagnosis epoch %d val group AUROC %.4f (best %.4f @ %d)", epoch, score, best, best_epoch)
        if epoch - best_epoch > cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model.eval()


def train_folds(pool_records, cfg, device="cuda"):
    """5-fold cross-fitting over the development pool; returns one diagnosis model per fold."""
    set_seed(cfg["seed"])
    models = []
    for fold, (tr, va) in enumerate(KFold(cfg["n_folds"], shuffle=True, random_state=cfg["seed"]).split(pool_records)):
        log.info("fold %d: %d training / %d validation patients", fold, len(tr), len(va))
        models.append(train_diagnosis_model([pool_records[i] for i in tr], [pool_records[i] for i in va], cfg, device))
    return models
