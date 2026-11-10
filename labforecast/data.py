"""Patient records, prediction windows, labels and endpoint groups.

A patient record is a dict:
    values       float32 [T, 40]  z-scored channels per measurement day (channel 0 = age), NaN = missing
    days         int64   [T]      day index of each measurement day (0 = first laboratory day)
    sex          int              1 = male, 0 = female
    event_codes  int64   [E]      diagnosis indices (see `diagnosis_codes()`) of recorded events
    event_days   int64   [E]      day index of each event (may be negative: recorded before first lab)
    imputed      float32 [T, 40]  optional, added by `labforecast.train.impute_records`

Prediction windows are cut from a record at training/evaluation time (`make_windows`): a window
covers 10-30 calendar days of laboratory data and is labelled with the diagnoses newly recorded
within the following 720 days. Diagnoses recorded before the window ends are excluded from the
loss/evaluation for that window ("at risk" mask).
"""
import json
import random
from importlib import resources

import torch
from torch.nn.utils.rnn import pad_sequence

CHANNELS = [
    "age", "albumin", "alkaline phosphatase", "anion gap", "aPTT", "basophils", "total bilirubin", "CRP",
    "total protein", "eosinophils", "erythrocytes", "gamma-GT", "eGFR (CKD-EPI)", "eGFR (MDRD)", "glucose",
    "AST", "ALT", "haematocrit", "haemoglobin", "uric acid", "urea nitrogen (BUN)", "INR", "potassium",
    "calcium", "creatinine", "leukocytes", "lymphocytes", "MCH", "MCHC", "MCV", "monocytes", "MPV", "sodium",
    "neutrophils", "plateletcrit", "PDW", "prothrombin time (Quick)", "RDW-CV", "platelets", "TSH",
]

# Codes within a group are excluded together once any member has been recorded.
MASK_TOGETHER = [["D46.7", "D46.2", "D46.9"], ["M89.59", "M89.55", "M89.58", "M89.50"]]

# The eight grouped endpoints reported in the paper (members matched on the ICD-10 code prefix).
MAIN_GROUPS = {
    "All-cause mortality": {"DEATH"},
    "Myelodysplastic syndrome": {"D46"},
    "Metastatic disease": {"C77", "C78", "C79"},
    "Cardiovascular disease": {"I10", "I11", "I12", "I13", "I15", "I20", "I21", "I25", "I42", "I48", "I50", "I80", "I82"},
    "Type 2 diabetes": {"E11"},
    "Kidney disease": {"N17", "N18"},
    "Fungal infections": {f"B{i}" for i in range(35, 50)},
    "Bacterial infections": {f"A{i:02d}" for i in range(80)} | {"B95", "B96", "B99"},
}

# Model output indices monitored on the validation set for checkpoint selection of the published model.
SELECTION_GROUPS = {
    "cardiovascular": [0, 4, 9, 27, 28, 32, 38, 39, 46, 52, 54, 56, 58, 59, 69, 71, 72, 78, 79, 83, 85, 86, 88, 93,
                       98, 100, 101, 103, 104, 106, 109, 110, 117, 121, 123, 133, 141, 143, 155, 160],
    "metastatic": [7, 29, 37, 57, 76, 77, 81, 84, 94, 102, 115, 119, 127, 130, 138, 139, 140, 147, 148, 149, 151, 154,
                   156, 159, 161],
    "bacterial": [6, 8, 15, 18, 25, 42, 47, 55, 66, 67, 74, 75, 92, 96, 112, 113, 129, 146, 153],
    "death": [157],
    "renal": [11, 40, 53, 61, 64, 68, 80, 97, 120, 124, 126, 132, 135, 137, 145, 152],
    "t2dm": [14, 122, 150],
    "osteolysis": [89, 95, 131, 158],
    "mds": [35, 51, 118],
}

def diagnosis_codes() -> list:
    """The 162 ICD-10-GM target codes, ordered by model output index."""
    mapping = json.loads(resources.files("labforecast").joinpath("resources/diagnosis_codes.json").read_text())
    return [code for code, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def group_indices(groups: dict, codes=None) -> dict:
    codes = codes or diagnosis_codes()
    return {name: [i for i, c in enumerate(codes) if c.split(".")[0] in prefixes] for name, prefixes in groups.items()}


def mask_together_indices(codes=None) -> list:
    codes = codes or diagnosis_codes()
    return [[codes.index(c) for c in group] for group in MASK_TOGETHER]


def split_patients(n_patients: int, test_fraction: float = 0.1, seed: int = 55):
    """Patient-level split into a development pool (training + validation folds) and a test set."""
    n_test = int(test_fraction * n_patients)
    train, val, test = torch.utils.data.random_split(
        range(n_patients), [n_patients - 2 * n_test, n_test, n_test], generator=torch.Generator().manual_seed(seed))
    return train.indices + val.indices, test.indices


def make_windows(record, n_diagnoses, horizon=720, min_followup=100, window_days=(10, 30),
                 mask_together=(), max_windows=10, rare_indices=(), rare_max_windows=20, rng=random):
    """Cut a patient record into consecutive prediction windows.

    Window lengths are drawn uniformly from `window_days` (calendar days). After each window,
    half of its measurement days are skipped before the next one starts. Windowing stops once
    neither a later diagnosis nor a later laboratory day lies at least `min_followup` days after the
    window end. Patients with more than `max_windows` windows are subsampled (`rare_max_windows`
    if any window is positive for one of `rare_indices`).
    """
    days = torch.as_tensor(record["days"]).long().reshape(-1)
    codes = torch.as_tensor(record["event_codes"]).long()
    event_days = torch.as_tensor(record["event_days"]).long()
    if len(codes) == 0:
        return []
    events = torch.zeros(len(codes), n_diagnoses, dtype=torch.bool)
    events[torch.arange(len(codes)), codes] = True
    T, last_event, windows, start = len(days), int(event_days.max()), [], 0
    while start < T:
        length = rng.randint(*window_days)
        end = start
        while end < T and days[end] - days[start] < length:
            end += 1
        window_end = int(days[end - 1])
        future_start = window_end + 1
        if last_event - future_start + 1 < min_followup and int(days[-1]) - future_start < min_followup:
            break
        prior = events[event_days < future_start].any(0)
        for group in mask_together:
            if prior[group].any():
                prior[group] = True
        in_horizon = (event_days >= future_start) & (event_days <= window_end + horizon)
        at_risk = ~prior
        windows.append({"rows": (start, end), "label": events[in_horizon].any(0) & at_risk, "at_risk": at_risk})
        start = end + (end - start) // 2
    cap = rare_max_windows if any(w["label"][list(rare_indices)].any() for w in windows) else max_windows
    if len(windows) > cap:
        windows = rng.sample(windows, cap)
    return windows


def collate(records, windows_per_record):
    """Stack windows into a padded batch with a leading all-zero CLS row per window."""
    vals, imps, pos, sex, labels, at_risk = [], [], [], [], [], []
    for rec, windows in zip(records, windows_per_record):
        v = torch.as_tensor(rec["values"]).float()
        imp = torch.as_tensor(rec["imputed"]).float() if "imputed" in rec else None
        d = torch.as_tensor(rec["days"]).long().reshape(-1)
        for w in windows:
            s, e = w["rows"]
            vals.append(v[s:e])
            imps.append(imp[s:e] if imp is not None else v[s:e])
            pos.append(d[s:e] - d[s])
            sex.append(int(rec["sex"]))
            labels.append(w["label"])
            at_risk.append(w["at_risk"])
    if not vals:
        return None
    B, C = len(vals), vals[0].shape[1]
    lengths = torch.tensor([len(v) for v in vals])
    cls = torch.zeros(B, 1, C)
    values = torch.cat([cls, pad_sequence(vals, batch_first=True)], 1)
    imputed = torch.cat([cls, pad_sequence(imps, batch_first=True)], 1)
    attention = torch.cat([torch.ones(B, 1), (torch.arange(values.shape[1] - 1)[None] < lengths[:, None]).float()], 1)
    position_ids = torch.cat([torch.zeros(B, 1, dtype=torch.long), pad_sequence(pos, batch_first=True)], 1)
    return {"values": values, "imputed": imputed, "attention": attention, "position_ids": position_ids,
            "sex": torch.tensor(sex), "labels": torch.stack(labels).float(), "at_risk": torch.stack(at_risk).float()}


def load_records(path):
    return torch.load(path, weights_only=False)
