"""
credit_scorecard.py
===================
An end-to-end probability-of-default (PD) scorecard built the way retail credit
models are built in banks: weight-of-evidence (WoE) binning, information-value
(IV) feature selection, a logistic regression on WoE features, validation of
both discrimination and calibration, and points scaling into a scorecard.

It also includes a fairness audit: what excluding protected characteristics
costs in predictive power, whether the remaining features act as proxies, and
how exclusion redistributes predicted risk between groups — compared across
two countries.

Data
----
- German Credit (Statlog), 1,000 approved loans, 30% bad.  Fetched from OpenML.
- Taiwan credit-card default (Yeh & Lien, 2009), 30,000 borrowers.  OpenML id 42477.
  Used only for the cross-country fairness comparison.

Run
---
    pip install -r requirements.txt
    python credit_scorecard.py

Outputs are written to ./outputs/.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
N_BINS = 5                 # quantile bins for numeric features
WOE_EPS = 0.5              # smoothing so empty bins don't give log(0)
IV_THRESHOLD = 0.02        # below this a feature is treated as uninformative

# Points scaling: a score of BASE_SCORE corresponds to good:bad odds of
# BASE_ODDS, and every PDO points doubles the odds.
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50

# Protected characteristics excluded from the model (UK Equality Act 2010).
# `personal_status` encodes sex alongside marital status.
PROTECTED = ["personal_status", "foreign_worker"]

OUT_DIR = "outputs"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_german() -> pd.DataFrame:
    df = fetch_openml("credit-g", version=1, as_frame=True, parser="auto").frame
    df["default"] = (df["class"] == "bad").astype(int)   # bad = 1 = default
    return df.drop(columns=["class"])


# --------------------------------------------------------------------------
# Weight of evidence
# --------------------------------------------------------------------------
def make_bins(train: pd.DataFrame, feature: str, numeric: list[str]):
    """Quantile bin edges for numeric features, fitted on training data only."""
    if feature not in numeric:
        return None                     # categoricals keep their own levels
    _, edges = pd.qcut(train[feature], q=N_BINS, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf   # so out-of-range test values still bin
    return edges


def apply_bins(series: pd.Series, edges) -> pd.Series:
    if edges is None:
        return series.astype(str)
    return pd.cut(series, bins=edges).astype(str)


def woe_table(train: pd.DataFrame, feature: str, edges) -> pd.DataFrame:
    """
    WoE = ln(%goods in bin / %bads in bin).  Positive = safer than average.
    IV  = sum((%goods - %bads) * WoE) — the feature's overall predictive power.
    """
    b = apply_bins(train[feature], edges)
    t = pd.DataFrame({"bin": b, "default": train["default"]})
    g = t.groupby("bin", observed=True)["default"].agg(n="count", bads="sum")
    g["goods"] = g["n"] - g["bads"]
    g["pct_good"] = (g["goods"] + WOE_EPS) / (g["goods"].sum() + WOE_EPS)
    g["pct_bad"] = (g["bads"] + WOE_EPS) / (g["bads"].sum() + WOE_EPS)
    g["bad_rate"] = g["bads"] / g["n"]
    g["woe"] = np.log(g["pct_good"] / g["pct_bad"])
    g["iv"] = (g["pct_good"] - g["pct_bad"]) * g["woe"]
    return g


def iv_strength(v: float) -> str:
    return ("useless" if v < 0.02 else "weak" if v < 0.1 else
            "medium" if v < 0.3 else "strong" if v < 0.5 else "suspicious")


def to_woe(data, features, bins, woe_maps) -> pd.DataFrame:
    """Replace raw values with training-set WoE. Unseen bins get 0 (neutral)."""
    out = pd.DataFrame(index=data.index)
    for f in features:
        out[f] = apply_bins(data[f], bins[f]).map(woe_maps[f]).fillna(0.0)
    return out


# --------------------------------------------------------------------------
# Model build
# --------------------------------------------------------------------------
def build_scorecard(df: pd.DataFrame) -> dict:
    df = df.drop(columns=PROTECTED)
    features = [c for c in df.columns if c != "default"]
    numeric = df[features].select_dtypes(include="number").columns.tolist()

    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=df["default"])
    print(f"Train {len(train)} rows ({train['default'].mean():.1%} bad) | "
          f"Test {len(test)} rows ({test['default'].mean():.1%} bad)")

    # --- WoE / IV on training data only ---
    bins, tables, iv = {}, {}, {}
    for f in features:
        bins[f] = make_bins(train, f, numeric)
        tables[f] = woe_table(train, f, bins[f])
        iv[f] = tables[f]["iv"].sum()
    iv = pd.Series(iv).sort_values(ascending=False)
    print("\nInformation value:")
    print(pd.DataFrame({"IV": iv.round(3), "strength": iv.map(iv_strength)}).to_string())

    selected = iv[iv >= IV_THRESHOLD].index.tolist()
    woe_maps = {f: tables[f]["woe"].to_dict() for f in selected}

    # --- Fit, then drop any feature with a wrong-signed coefficient ---
    # With WoE coding, positive WoE = safer, so every coefficient predicting
    # default should be NEGATIVE. A positive one means the model is using the
    # feature against its own evidence (typically multicollinearity) — which
    # is indefensible in a regulated scorecard, so it is removed and refit.
    y_train, y_test = train["default"], test["default"]
    while True:
        X_train = to_woe(train, selected, bins, woe_maps)
        lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        coefs = pd.Series(lr.coef_[0], index=selected)
        wrong = coefs[coefs > 0]
        if wrong.empty:
            break
        drop = wrong.idxmax()
        print(f"\nDropping '{drop}': positive coefficient {wrong[drop]:+.3f} "
              f"(IV {iv[drop]:.3f}) — sign flip")
        selected.remove(drop)

    X_test = to_woe(test, selected, bins, woe_maps)
    return dict(train=train, test=test, y_train=y_train, y_test=y_test,
                X_train=X_train, X_test=X_test, lr=lr, selected=selected,
                bins=bins, woe_maps=woe_maps, tables=tables, iv=iv)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(m: dict) -> dict:
    lr, X_train, X_test = m["lr"], m["X_train"], m["X_test"]
    y_train, y_test = m["y_train"], m["y_test"]
    p_train = lr.predict_proba(X_train)[:, 1]
    p_test = lr.predict_proba(X_test)[:, 1]

    # Discrimination: does the model rank borrowers correctly?
    res = {}
    for name, y, p in [("train", y_train, p_train), ("test", y_test, p_test)]:
        auc = roc_auc_score(y, p)
        ks = ks_2samp(p[y == 1], p[y == 0]).statistic
        res[name] = dict(auc=auc, gini=2 * auc - 1, ks=ks)

    print("\nDiscrimination:")
    for name in ("train", "test"):
        r = res[name]
        print(f"  {name:5s}  AUC {r['auc']:.3f}  Gini {r['gini']:.3f}  KS {r['ks']:.3f}")

    # Calibration: are the PD values themselves right?
    cal = pd.DataFrame({"pd": p_test, "default": y_test.values})
    cal["decile"] = pd.qcut(cal["pd"], 10, labels=False)
    cal_tab = cal.groupby("decile").agg(n=("default", "size"),
                                        mean_pd=("pd", "mean"),
                                        observed=("default", "mean"))
    print("\nCalibration by decile of predicted PD (test):")
    print(cal_tab.round(3).to_string())
    print(f"  overall: predicted {p_test.mean():.3f}  observed {y_test.mean():.3f}")

    m.update(p_test=p_test, cal_tab=cal_tab, metrics=res)
    return m


# --------------------------------------------------------------------------
# Points scaling
# --------------------------------------------------------------------------
def scale_points(m: dict) -> dict:
    lr, selected = m["lr"], m["selected"]
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    b0 = lr.intercept_[0]
    coefs = dict(zip(selected, lr.coef_[0]))
    n = len(selected)

    # ln(good:bad odds) = -(b0 + sum b_j * WoE_j), with the intercept and
    # offset shared equally across features so every bin gets a point value.
    def pts(f, w):
        return -(coefs[f] * w + b0 / n) * factor + offset / n

    rows = [{"feature": f, "bin": b, "woe": round(w, 3), "points": round(pts(f, w))}
            for f in selected for b, w in m["woe_maps"][f].items()]
    card = pd.DataFrame(rows)

    scores = sum(pts(f, m["X_test"][f]) for f in selected)
    implied_pd = 1 / (1 + np.exp((scores - offset) / factor))
    err = float(np.abs(implied_pd - m["p_test"]).max())
    print(f"\nPoints scaling (PDO {PDO}, {BASE_SCORE} = {BASE_ODDS}:1 odds)")
    print(f"  score range {int(scores.min())}–{int(scores.max())}; "
          f"max |implied PD - model PD| = {err:.1e}")

    bands = pd.qcut(scores, 5)
    band_tab = (pd.DataFrame({"default": m["y_test"].values, "band": bands})
                .groupby("band", observed=True)
                .agg(n=("default", "size"), default_rate=("default", "mean")))
    print(band_tab.round(3).to_string())

    m.update(scorecard=card, scores=scores, band_tab=band_tab)
    return m


# --------------------------------------------------------------------------
# Fairness audit
# --------------------------------------------------------------------------
def _pipe(cols, data, categorical=None):
    if categorical is None:
        categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in categorical if c in cols]
    numeric = [c for c in cols if c not in categorical]
    ct = make_column_transformer((StandardScaler(), numeric),
                                 (OneHotEncoder(handle_unknown="ignore"), categorical))
    return make_pipeline(ct, LogisticRegression(max_iter=3000))


def fairness_audit(df, sex, with_cols, without_cols, n_repeats, categorical=None):
    """
    1. Predictive cost of excluding protected characteristics.
    2. Proxy strength: can the remaining features predict sex?
    3. Redistribution: predicted vs actual default rate by sex, from the model
       trained WITHOUT sex. The protected attribute is held out of the model
       but used to audit it — the standard practice.
    """
    y = df["default"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)
    auc = {}
    for name, cols in [("with", with_cols), ("without", without_cols)]:
        auc[name] = cross_val_score(_pipe(cols, df, categorical), df[cols], y,
                                    cv=cv, scoring="roc_auc").mean()
    proxy = cross_val_score(_pipe(without_cols, df, categorical), df[without_cols],
                            (sex == "female").astype(int), cv=cv, scoring="roc_auc").mean()
    pd_hat = cross_val_predict(_pipe(without_cols, df, categorical), df[without_cols], y,
                               cv=5, method="predict_proba")[:, 1]
    groups = pd.DataFrame({"sex": sex, "actual": y, "predicted": pd_hat}) \
               .groupby("sex")[["actual", "predicted"]].mean()
    return auc, proxy, groups


def run_fairness(german: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70 + "\nFAIRNESS AUDIT\n" + "=" * 70)
    results = {}

    sex = german["personal_status"].astype(str).str.startswith("female") \
                .map({True: "female", False: "male"})
    with_cols = [c for c in german.columns if c != "default"]
    without = [c for c in with_cols if c not in PROTECTED]
    results["Germany"] = fairness_audit(german, sex, with_cols, without, n_repeats=20)

    try:
        tw = fetch_openml(data_id=42477, as_frame=True, parser="auto").frame.apply(pd.to_numeric)
        tw["default"] = tw["y"].astype(int)
        tw_sex = tw["x2"].map({1: "male", 2: "female"})       # x2 = sex, x4 = marital
        feats = [f"x{i}" for i in range(1, 24)]
        results["Taiwan"] = fairness_audit(tw, tw_sex, feats,
                                           [c for c in feats if c not in ("x2", "x4")],
                                           n_repeats=4, categorical=["x2", "x3", "x4"])
    except Exception as e:                                     # network/OpenML issues
        print(f"  (Taiwan comparison skipped: {e})")

    rows = []
    for country, (auc, proxy, g) in results.items():
        rows.append({
            "country": country,
            "AUC with protected": auc["with"],
            "AUC without": auc["without"],
            "AUC cost": auc["with"] - auc["without"],
            "proxy AUC (predict sex)": proxy,
            "actual default F": g.loc["female", "actual"],
            "actual default M": g.loc["male", "actual"],
            "predicted F": g.loc["female", "predicted"],
            "predicted M": g.loc["male", "predicted"],
        })
    table = pd.DataFrame(rows).set_index("country")
    print(table.round(3).T.to_string())
    return table


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot(m: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    c = m["cal_tab"]
    ax = axes[0]
    ax.plot([0, 0.8], [0, 0.8], ls="--", c="grey", lw=1, label="perfect calibration")
    ax.plot(c["mean_pd"], c["observed"], "o-", c="#1F3A5F", label="model (test deciles)")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate",
           title="Calibration")
    ax.legend(frameon=False)

    ax = axes[1]
    s, y = m["scores"], m["y_test"].values
    ax.hist(s[y == 0], bins=20, alpha=0.6, label="good", color="#1F3A5F")
    ax.hist(s[y == 1], bins=20, alpha=0.6, label="bad", color="#C0504D")
    g = m["metrics"]["test"]
    ax.set(xlabel="Score", ylabel="Borrowers",
           title=f"Score distribution (Gini {g['gini']:.2f}, KS {g['ks']:.2f})")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "validation.png"), dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    german = load_german()

    print("=" * 70 + "\nSCORECARD BUILD — German Credit\n" + "=" * 70)
    m = build_scorecard(german)
    m = validate(m)
    m = scale_points(m)
    plot(m)

    m["scorecard"].to_csv(os.path.join(OUT_DIR, "scorecard.csv"), index=False)
    m["iv"].rename("iv").to_csv(os.path.join(OUT_DIR, "information_value.csv"))
    run_fairness(german).to_csv(os.path.join(OUT_DIR, "fairness_audit.csv"))
    print(f"\nOutputs written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
