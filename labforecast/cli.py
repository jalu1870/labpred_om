"""Command-line interface: python -m labforecast <command> --help"""
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import baselines, data, evaluate, structure, train
from .model import load_diagnosis_model, load_imputation_model

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _split(records, cfg):
    pool, test = data.split_patients(len(records), cfg["split"]["test_fraction"], cfg["split"]["seed"])
    return [records[i] for i in pool], [records[i] for i in test]


def _test_windows(records, cfg, seed=0):
    codes = data.diagnosis_codes()
    w = cfg["diagnosis"]["windows"]
    t = cfg["test_windows"]
    rng = random.Random(seed)
    windows = [data.make_windows(r, len(codes), horizon=w["horizon"], min_followup=w["min_followup"],
                                 window_days=tuple(w["window_days"]), mask_together=data.mask_together_indices(),
                                 max_windows=t["max_windows"], rare_indices=[codes.index(c) for c in t["rare_codes"]],
                                 rare_max_windows=t["rare_max_windows"], rng=rng) for r in records]
    batch = data.collate(records, windows)
    batch["patient_ids"] = np.repeat(np.arange(len(records)), [len(x) for x in windows])
    return batch


def cmd_structure(a, cfg):
    stats = torch.load(a.stats) if a.stats else None
    records, stats = structure.build_records(pd.read_csv(a.labs), pd.read_csv(a.diagnoses),
                                             pd.read_csv(a.deaths) if a.deaths else None, stats=stats)
    torch.save(records, a.out)
    torch.save(stats, Path(a.out).with_suffix(".stats.pt"))
    logging.info("%d patients written to %s", len(records), a.out)


def cmd_train_imputer(a, cfg):
    pool, _ = _split(data.load_records(a.records), cfg)
    n_val = len(pool) // 10
    model = train.train_imputation_model(pool[n_val:], pool[:n_val], cfg["imputation"], a.device)
    torch.save(model.state_dict(), a.out)


def cmd_impute(a, cfg):
    records = data.load_records(a.records)
    torch.save(train.impute_records(records, load_imputation_model(a.imputer), a.device), a.out)


def cmd_train(a, cfg):
    pool, _ = _split(data.load_records(a.records), cfg)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, model in enumerate(train.train_folds(pool, cfg["diagnosis"], a.device)):
        torch.save(model.state_dict(), out / f"diagnosis_fold{k}.pt")


def cmd_evaluate(a, cfg):
    _, test = _split(data.load_records(a.records), cfg)
    batch = _test_windows(test, cfg)
    models = [load_diagnosis_model(p, **cfg["diagnosis"]["diagnosis_model"]) for p in sorted(Path(a.models).glob("*.pt*"))]
    logits = evaluate.predict_logits(batch, load_imputation_model(a.imputer), models, a.device)
    probs, labels, at_risk = evaluate.probabilities(logits), batch["labels"].numpy(), batch["at_risk"].numpy()
    rows = evaluate.summary_table(probs, labels, at_risk, patient_ids=batch["patient_ids"],
                                  n_resamples=cfg["evaluation"]["n_resamples"])
    per = evaluate.per_diagnosis_auroc(probs, labels, at_risk, cfg["evaluation"]["min_events"])
    ci = lambda t: f"{t[0]:.3f} [{t[1]:.3f}-{t[2]:.3f}]"
    table = pd.DataFrame([{"endpoint": r["endpoint"], "AUROC (windows)": ci(r["auroc"]),
                           "AUROC (patients)": ci(r["auroc_patient_bootstrap"]), "PR-AUC": f"{r['pr_auc']:.3f}",
                           "enrichment": f"{r['enrichment'][0]:.1f} [{r['enrichment'][1]:.1f}-{r['enrichment'][2]:.1f}]"}
                          for r in rows])
    print(table.to_string(index=False))
    print(f"\nmean AUROC over {len(per)} diagnoses with >= {cfg['evaluation']['min_events']} events: {np.mean(list(per.values())):.3f}")
    if a.out:
        torch.save({"logits": logits, "labels": labels, "at_risk": at_risk, "patient_ids": batch["patient_ids"]}, a.out)


def cmd_baselines(a, cfg):
    pool, test = _split(data.load_records(a.records), cfg)
    rng = random.Random(0)
    w = dict(cfg["diagnosis"]["windows"], mask_together=data.mask_together_indices())
    n_diag = cfg["diagnosis"]["diagnosis_model"]["n_diagnoses"]
    windows = [data.make_windows(r, n_diag, rng=rng, **w) for r in pool]
    dev = data.collate(pool, windows)
    dev_patients = np.repeat(np.arange(len(pool)), [len(x) for x in windows])
    tst = _test_windows(test, cfg)
    X, Xt = baselines.trajectory_features(dev), baselines.trajectory_features(tst)
    y, m = dev["labels"].numpy(), dev["at_risk"].numpy()
    results = {}
    for name, cols in baselines.FAMILIES.items():
        p = (baselines.fit_xgboost_bagged(X[:, cols], y, m, dev_patients, Xt[:, cols]) if name.startswith("xgboost")
             else baselines.fit_logistic(X[:, cols], y, m, Xt[:, cols]))
        per = evaluate.per_diagnosis_auroc(p, tst["labels"].numpy(), tst["at_risk"].numpy(), cfg["evaluation"]["min_events"])
        results[name] = float(np.mean(list(per.values())))
        print(f"{name:15s} mean AUROC over {len(per)} diagnoses: {results[name]:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser(prog="labforecast")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("structure", help="build patient records from long-format tables")
    s.add_argument("--labs", required=True); s.add_argument("--diagnoses", required=True)
    s.add_argument("--deaths"); s.add_argument("--stats", help="normalisation statistics of the development cohort")
    s.add_argument("--out", required=True)
    s = sub.add_parser("train-imputer"); s.add_argument("--records", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("impute"); s.add_argument("--records", required=True); s.add_argument("--imputer", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("train"); s.add_argument("--records", required=True); s.add_argument("--out-dir", required=True)
    s = sub.add_parser("evaluate"); s.add_argument("--records", required=True); s.add_argument("--imputer", required=True)
    s.add_argument("--models", required=True, help="directory with one diagnosis checkpoint per fold"); s.add_argument("--out")
    s = sub.add_parser("baselines"); s.add_argument("--records", required=True); s.add_argument("--out")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(open(a.config))
    {"structure": cmd_structure, "train-imputer": cmd_train_imputer, "impute": cmd_impute, "train": cmd_train,
     "evaluate": cmd_evaluate, "baselines": cmd_baselines}[a.command](a, cfg)


if __name__ == "__main__":
    main()
