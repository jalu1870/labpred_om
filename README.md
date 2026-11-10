# LabTrajectory

Code for **"Routine laboratory trajectories predict organ-level complications in cancer"**
(Lübberstedt et al., *npj Digital Medicine*, 2026, https://doi.org/10.1038/s41746-026-03405-8).
Project page: https://jalu1870.github.io/lab-forecast/

A transformer reads 10 to 30 days of routine laboratory results (39 analytes plus age and sex) and
predicts which of 162 treatment-associated complications (ICD-10-GM codes, including death) will be
newly recorded within the next 720 days.

```
labforecast/
  structure.py   long-format lab and diagnosis tables -> patient records
  data.py        patient split, prediction windows, labels, endpoint groups
  model.py       imputation model and diagnosis model (Qwen3 backbone)
  train.py       imputation-model training, timeline imputation, 5-fold diagnosis training
  evaluate.py    inference, AUROC / PR-AUC / enrichment, window- and patient-level bootstrap
  baselines.py   engineered-trajectory XGBoost and logistic-regression baselines
  cli.py         command-line interface
configs/default.yaml   all hyperparameters of the published model
```

## Installation

```bash
pip install -e .            # Python >= 3.10, PyTorch >= 2.1
pytest tests                # synthetic-data smoke tests (pip install -e ".[test]")
```

## Data

The clinical data cannot be shared (EU GDPR). To run the pipeline on your own data, provide three CSV files:

| file | columns |
|---|---|
| labs | `patient_id, sex` (`M`/`F`)`, age_years, timestamp, analyte, value` |
| diagnoses | `patient_id, icd_code, date` (ICD-10-GM, e.g. `I48.0`) |
| deaths (optional) | `patient_id, date` |

`analyte` must be one of the names in `labforecast.data.CHANNELS`. Units:

| analyte | unit | analyte | unit |
|---|---|---|---|
| albumin | g/dl | INR | – |
| alkaline phosphatase | U/l | potassium | mmol/l |
| anion gap | mmol/l | calcium | mmol/l |
| aPTT | s | creatinine | mg/dl |
| basophils | % | leukocytes | G/l |
| total bilirubin | mg/dl | lymphocytes | % |
| CRP | mg/dl | MCH | pg |
| total protein | g/dl | MCHC | g/dl |
| eosinophils | % | MCV | fl |
| erythrocytes | T/l | monocytes | % |
| gamma-GT | U/l | MPV | fl |
| eGFR (CKD-EPI) | ml/min | sodium | mmol/l |
| eGFR (MDRD) | ml/min | neutrophils | % |
| glucose | mg/dl | plateletcrit | % |
| AST | U/l | PDW | fl |
| ALT | U/l | prothrombin time (Quick) | % |
| haematocrit | % | RDW-CV | % |
| haemoglobin | g/dl | platelets | G/l |
| uric acid | mg/dl | TSH | µIU/ml |
| urea nitrogen (BUN) | mg/dl | | |

The 162 target codes are listed in `labforecast/resources/diagnosis_codes.json`.

## Usage

```bash
labforecast structure     --labs labs.csv --diagnoses diagnoses.csv --deaths deaths.csv --out records.pt
labforecast train-imputer --records records.pt --out imputer.pt
labforecast impute        --records records.pt --imputer imputer.pt --out records_imputed.pt
labforecast train         --records records_imputed.pt --out-dir models/
labforecast evaluate      --records records.pt --imputer imputer.pt --models models/
labforecast baselines     --records records.pt
```

- **Split.** Patients are split 90/10 into a development pool and a held-out test set (`split.seed`). The diagnosis model is trained by 5-fold cross-fitting on the pool, and test predictions average the logits of the five fold models.
- **External cohorts.** Use `structure --stats` with the statistics file written for the development cohort, so the external cohort is normalised the same way.
- **Evaluation output.** `evaluate` reports, for the eight grouped endpoints of the paper:
  - AUROC with 95% confidence intervals from 1,000 window-level and patient-clustered bootstrap resamples;
  - PR-AUC;
  - enrichment (PR-AUC relative to prevalence);
  - the mean AUROC over all diagnoses with at least ten events.

## Model weights

The trained weights (imputation model and five fold models) will be provided as a release asset of
this repository. Load them with `labforecast.model.load_imputation_model` and `load_diagnosis_model`.

## Citation

```bibtex
@article{luebberstedt2026labtrajectories,
  title   = {Routine laboratory trajectories predict organ-level complications in cancer},
  author  = {L{\"u}bberstedt, Jannik and Braitsch, Krischan and Lammert, Jacqueline and Winter, Christof
             and Gabriel, Florian and Lemke, Tristan and Zirn, Christopher and Graf, Markus
             and Puttkammer, Friedrich and H{\"a}ntze, Hartmut and Moll, Johannes and Narayanan, Anirudh
             and Zhukov, Andrei and Drexel, Fabian and Ben Chaaben, Zeineb and Ziegelmayer, Sebastian
             and Kim, Su Hwan and H{\"o}gner, Marion and Kirschke, Jan and Bassermann, Florian
             and Makowski, Marcus and Wachinger, Christian and Adams, Lisa and Bressem, Keno},
  journal = {npj Digital Medicine},
  year    = {2026},
  doi     = {10.1038/s41746-026-03405-8}
}
```

## License

MIT
