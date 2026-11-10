"""Build patient records (see `labforecast.data`) from long-format laboratory and diagnosis tables.

Inputs (pandas DataFrames):
    labs        patient_id, sex ("M"/"F"), age_years, timestamp, analyte, value
                `analyte` must be one of `data.CHANNELS[1:]`; values in the units listed in the README.
    diagnoses   patient_id, icd_code (ICD-10-GM, e.g. "I48.0"), date
    deaths      patient_id, date   (optional; recorded as the diagnosis "DEATH")

Steps: laboratory values are averaged per patient, analyte and calendar day; every day with at least one
measurement becomes one row, with the patient's age on that day as channel 0. Days are counted from the
patient's first laboratory day. Patients with fewer than 25 % observed entries are removed, and all
channels are z-scored. Diagnoses are mapped to the 162 target codes and timed relative to the first
laboratory day; patients without any target diagnosis are removed.
"""
import numpy as np
import pandas as pd
import torch

from .data import CHANNELS, diagnosis_codes


def lab_sequences(labs: pd.DataFrame, min_observed_fraction: float = 0.25):
    """Per patient: (values [T, 40] in original units, day index [T], sex, first laboratory date)."""
    labs = labs[labs["analyte"].isin(CHANNELS[1:])].copy()
    labs["date"] = pd.to_datetime(labs["timestamp"]).dt.floor("D")
    daily = labs.groupby(["patient_id", "date", "analyte"])["value"].mean().unstack("analyte")
    daily = daily.reindex(columns=CHANNELS[1:])
    age = labs.sort_values("timestamp").groupby(["patient_id", "date"])["age_years"].first()
    sex = labs.groupby("patient_id")["sex"].first()
    out = {}
    for pid, df in daily.groupby(level="patient_id"):
        dates = df.index.get_level_values("date")
        values = np.column_stack([age.loc[pid].reindex(dates).to_numpy(), df.to_numpy()]).astype(np.float32)
        if np.mean(~np.isnan(values)) < min_observed_fraction:
            continue
        days = np.asarray((dates - dates[0]).days, dtype=np.int64)
        out[pid] = (values, days, 1 if sex.loc[pid] == "M" else 0, dates[0])
    return out


def normalisation_stats(sequences):
    """Per-channel mean and standard deviation over all measurement days."""
    rows = pd.DataFrame(np.concatenate([v for v, *_ in sequences.values()]))
    return rows.mean().to_numpy(np.float32), rows.std().to_numpy(np.float32)


def build_records(labs, diagnoses, deaths=None, stats=None, min_observed_fraction=0.25):
    """Returns (records, (mean, std)). Pass `stats` from the development cohort to normalise an external
    cohort with the same statistics."""
    sequences = lab_sequences(labs, min_observed_fraction)
    mean, std = stats if stats is not None else normalisation_stats(sequences)
    index = {code: i for i, code in enumerate(diagnosis_codes())}
    events = diagnoses[["patient_id", "icd_code", "date"]]
    if deaths is not None:
        events = pd.concat([events, deaths.assign(icd_code="DEATH")[["patient_id", "icd_code", "date"]]])
    events = events[events["icd_code"].isin(index)].copy()
    events["date"] = pd.to_datetime(events["date"]).dt.floor("D")
    by_patient = dict(tuple(events.groupby("patient_id")))
    records = []
    for pid in sorted(sequences):
        values, days, sex, first_day = sequences[pid]
        ev = by_patient.get(pid, events.iloc[:0])
        pairs = np.unique(np.column_stack([ev["icd_code"].map(index).to_numpy(np.int64),
                                           (ev["date"] - first_day).dt.days.to_numpy(np.int64)]), axis=0)
        if len(pairs) == 0:
            continue
        records.append({"patient_id": pid, "values": torch.from_numpy((values - mean) / std),
                        "days": torch.from_numpy(days), "sex": sex,
                        "event_codes": torch.from_numpy(pairs[:, 0]), "event_days": torch.from_numpy(pairs[:, 1])})
    return records, (mean, std)
