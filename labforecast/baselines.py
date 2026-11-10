"""Non-sequential baselines on engineered trajectory features.

Each window is summarised per channel (40 channels incl. age) by 11 statistics of its observed values:
last, minimum, maximum, mean, standard deviation, least-squares slope over days, last-minus-previous,
exponentially weighted mean (weights exp((day - last day) / 7)), number of observations, days since
the last observation, and last-minus-first; plus sex, the number of measurement days and the window
span. One binary model per diagnosis is fitted on the windows at risk for that diagnosis.
"""
import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_STATS = 11
XGB_PARAMS = dict(n_estimators=10000, max_depth=6, learning_rate=0.1, subsample=0.8,
                  early_stopping_rounds=100, tree_method="hist", verbosity=0)


def _window_features(values, days):
    """values [T, C] (NaN = missing), days [T] -> [C * 11] summary statistics."""
    C = values.shape[1]
    f = np.full(C * N_STATS, np.nan, dtype=np.float32)
    window_end = days[-1]
    for c in range(C):
        obs = ~np.isnan(values[:, c])
        v, d = values[obs, c], days[obs]
        if len(v) == 0:
            continue
        o = c * N_STATS
        f[o:o + 5] = v[-1], v.min(), v.max(), v.mean(), v.std() if len(v) > 1 else 0.0
        if len(v) > 1 and d.max() > d.min():
            w = np.exp((d - d.max()) / 7.0)
            f[o + 5:o + 8] = np.polyfit(d, v, 1)[0], v[-1] - v[-2], (w * v).sum() / w.sum()
        else:
            f[o + 5:o + 8] = 0.0, 0.0, v[-1]
        f[o + 8:o + 11] = len(v), window_end - d[-1], v[-1] - v[0] if len(v) > 1 else 0.0
    return f


def trajectory_features(batch, n_jobs=8):
    """Features [n_windows, 40 * 11 + 3] from a collated batch (raw values, CLS row skipped)."""
    values = batch["values"].numpy()
    days = batch["position_ids"].numpy().astype(float)
    n_tok = batch["attention"].numpy().sum(1).astype(int)
    stats = Parallel(n_jobs=n_jobs)(delayed(_window_features)(values[i, 1:n_tok[i]], days[i, 1:n_tok[i]])
                                    for i in range(len(values)))
    span = np.array([days[i, n_tok[i] - 1] - days[i, 1] if n_tok[i] > 1 else 0.0 for i in range(len(values))])
    extra = np.stack([batch["sex"].numpy(), n_tok - 1, span], axis=1).astype(np.float32)
    return np.concatenate([np.stack(stats), extra], axis=1)


def feature_columns(stats=range(N_STATS), window_shape=True, n_channels=40):
    """Column indices of a feature family: the given per-channel statistics, sex, and (optionally) the
    number of measurement days and window span."""
    sex = n_channels * N_STATS
    cols = [c * N_STATS + k for c in range(n_channels) for k in stats] + [sex]
    return sorted(cols + ([sex + 1, sex + 2] if window_shape else []))


# Feature families of the published baselines.
FAMILIES = {
    "xgboost_all": feature_columns(),                                       # all 11 statistics
    "xgboost_last": feature_columns(stats=[0], window_shape=False),         # last value per channel
    "logistic_last": feature_columns(stats=[0], window_shape=False),
    "logistic_all": feature_columns(stats=range(9)),                        # first 9 statistics
}


def fit_xgboost_bagged(X, labels, at_risk, patient_ids, X_test, n_folds=5, seed=5, n_jobs=4):
    """Per diagnosis, 5 patient-disjoint folds: train on 4, early-stop on the held-out fold, and average
    the fold models' test probabilities. Returns [n_test, n_diagnoses] (0 where no model could be fitted)."""
    patients = np.unique(patient_ids)
    perm = np.random.default_rng(seed).permutation(len(patients))
    fold_of = {p: k for k in range(n_folds) for p in patients[perm[k::n_folds]]}
    fold = np.array([fold_of[p] for p in patient_ids])

    def fit(c, k):
        tr = (fold != k) & (at_risk[:, c] == 1)
        va = (fold == k) & (at_risk[:, c] == 1)
        if labels[tr, c].sum() < 2 or len(np.unique(labels[va, c])) < 2:
            return None
        model = xgb.XGBClassifier(n_jobs=8, **XGB_PARAMS)
        model.fit(X[tr], labels[tr, c], eval_set=[(X[va], labels[va, c])], verbose=False)
        return model.predict_proba(X_test)[:, 1].astype(np.float32)

    n_diag = labels.shape[1]
    total, n_models = np.zeros((len(X_test), n_diag)), np.zeros(n_diag)
    for k in range(n_folds):
        for c, p in enumerate(Parallel(n_jobs=n_jobs, backend="threading")(delayed(fit)(c, k) for c in range(n_diag))):
            if p is not None:
                total[:, c] += p
                n_models[c] += 1
    return (total / np.maximum(n_models, 1)).astype(np.float32)


def fit_logistic(X, labels, at_risk, X_test, n_jobs=8):
    """Per diagnosis: median imputation, standardisation, class-balanced logistic regression."""
    def fit(c):
        m = at_risk[:, c] == 1
        if labels[m, c].sum() < 2:
            return np.zeros(len(X_test), dtype=np.float32)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=3000, class_weight="balanced"))
        model.fit(X[m], labels[m, c])
        return model.predict_proba(X_test)[:, 1].astype(np.float32)

    return np.stack(Parallel(n_jobs=n_jobs)(delayed(fit)(c) for c in range(labels.shape[1])), axis=1)
