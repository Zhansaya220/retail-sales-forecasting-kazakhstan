"""Reproduce the national retail pilot. Run validation, then evaluate frozen settings."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import numpy as np
import pandas as pd
import xlrd
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PROTOCOL = json.loads((ROOT / "protocol.json").read_text())
SCALE = 1_000_000.0
WARNINGS = []


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def load_data():
    raw = ROOT / "data/bns_retail_2015_2026.xls"
    checksum = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert checksum == PROTOCOL["raw_sha256"], "Use the archived workbook; the live file may be revised."
    book = xlrd.open_workbook(str(raw))
    sheet = book.sheet_by_name("Indicator")
    rows, converted, empty = [], [], []
    for r in range(sheet.nrows):
        year = sheet.cell_value(r, 0)
        if not isinstance(year, (int, float)) or not 2015 <= year <= 2026:
            continue
        for month in range(1, 13):
            value = sheet.cell_value(r, month)
            date = pd.Timestamp(int(year), month, 1)
            if value == "":
                empty.append(str(date.date()))
                continue
            if isinstance(value, str):
                converted.append({"date": str(date.date()), "original": value})
                value = float(value.replace("\xa0", "").replace(" ", "").replace(",", "."))
            rows.append({"date": date, "turnover_million_kzt": float(value)})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    assert not df.date.duplicated().any()
    assert np.isfinite(df.turnover_million_kzt).all()
    assert (df.turnover_million_kzt > 0).all()
    primary = df.loc[df.date <= "2025-12-01"].copy()
    assert len(primary) == 132
    assert list(primary.date) == list(pd.date_range("2015-01-01", "2025-12-01", freq="MS"))
    df.to_csv(ROOT / "data/clean_monthly.csv", index=False, date_format="%Y-%m-%d")
    audit = {
        "source_url": PROTOCOL["source_url"], "raw_sha256": checksum,
        "raw_grid_cells": 144, "numeric_cells": len(rows), "unavailable_cells": empty,
        "converted_text_values": converted, "primary_records": len(primary),
        "duplicates_removed": 0, "missing_primary": 0, "outliers_removed": 0,
        "primary_excluded_2026_observations": len(df) - len(primary),
        "lag_warmup_months": 12, "train_raw_months": 96,
        "train_supervised_rows": 84, "validation_months": 24, "test_months": 12,
        "source_unit_conflict": "Metadata says percentage; Indicator explicitly says mln.tenge.",
        "source_vintage": "Revised workbook retrieved 24 October 2026, not archived real-time releases.",
        "target_description": primary.turnover_million_kzt.describe().to_dict(),
        "train_description": primary.loc[primary.date <= "2022-12-01", "turnover_million_kzt"].describe().to_dict(),
        "validation_description": primary.loc[primary.date.between("2023-01-01", "2024-12-01"), "turnover_million_kzt"].describe().to_dict(),
        "test_description": primary.loc[primary.date >= "2025-01-01", "turnover_million_kzt"].describe().to_dict(),
    }
    dump("data_audit.json", audit)
    return df


def features(y, dates, full):
    y = pd.Series(y, dtype=float)
    f = pd.DataFrame({f"lag{k}": y.shift(k) for k in PROTOCOL["lag_features"]})
    if full:
        f["month"] = pd.DatetimeIndex(dates).month
        for k in [3, 12]:
            f[f"mean{k}"] = y.shift(1).rolling(k).mean()
    return f


def forecast(y, dates, index, family, params, full=False):
    past = np.asarray(y[:index]) / SCALE
    if family == "Seasonal naive":
        return float(y[index - 12])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if family == "ETS":
            model = ExponentialSmoothing(past, trend="add", seasonal_periods=12,
                seasonal=params["seasonal"], damped_trend=params["damped_trend"],
                initialization_method="estimated").fit(optimized=True)
            pred = float(np.asarray(model.forecast(1))[0])
        elif family == "ARIMA":
            model = ARIMA(past, order=tuple(params["order"]), trend="t").fit()
            pred = float(np.asarray(model.forecast(1))[0])
        else:
            # The current target is replaced by NaN before constructing its features.
            available_y = np.r_[past, np.nan]
            x = features(available_y, dates[:index+1], full)
            train_x = x.iloc[12:index]
            train_y = available_y[12:index]
            assert len(train_x) == index - 12 and not train_x.isna().any().any()
            assert not x.iloc[[-1]].isna().any().any()
            model = LGBMRegressor(objective="regression", random_state=42, n_jobs=2,
                verbosity=-1, deterministic=True, force_col_wise=True, **params)
            model.fit(train_x, train_y)
            pred = float(model.predict(x.iloc[[-1]])[0])
        for w in caught:
            if "feature names" not in str(w.message):
                WARNINGS.append({"family": family, "date": str(dates[index]),
                                 "type": type(w.message).__name__, "message": str(w.message)})
    assert np.isfinite(pred)
    return pred * SCALE


def run_origins(df, dates, family, params, full=False):
    y = df.turnover_million_kzt.to_numpy()
    return np.array([forecast(y, df.date, i, family, params, full)
                     for i in np.flatnonzero(df.date.isin(dates))])


def metrics(actual, pred, rmsse_denominator):
    e = pred - actual
    return {"MAE": float(np.mean(abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))),
            "WAPE": float(100 * abs(e).sum() / abs(actual).sum()),
            "RMSSE": float(np.sqrt(np.mean(e**2) / rmsse_denominator)),
            "Bias": float(np.mean(e))}


def validation(df):
    started = time.perf_counter()
    val_dates = pd.date_range(*PROTOCOL["validation"], freq="MS")
    truth = df.loc[df.date.isin(val_dates), "turnover_million_kzt"].to_numpy()
    train = df.loc[df.date <= PROTOCOL["train"][1], "turnover_million_kzt"].to_numpy()
    denominator = float(np.mean((train[12:] - train[:-12])**2))
    chosen, records = {}, []
    grids = {
      "Seasonal naive": [{}],
      "ETS": PROTOCOL["ets_grid"],
      "ARIMA": [{"order": x} for x in PROTOCOL["arima_orders"]],
    }
    keys = list(PROTOCOL["lgbm_grid"])
    lgbm_grid = [dict(zip(keys, v)) for v in itertools.product(*(PROTOCOL["lgbm_grid"][k] for k in keys))]
    grids["LightGBM lag"] = lgbm_grid
    grids["LightGBM full"] = lgbm_grid
    for family, grid in grids.items():
        best_score, best_params = np.inf, None
        for j, params in enumerate(grid):
            preds = run_origins(df, val_dates, family, params, full=family.endswith("full"))
            score = metrics(truth, preds, denominator)
            records.append({"model": family, "candidate": j + 1,
                            "parameters": json.dumps(params, sort_keys=True), **score})
            if score["WAPE"] < best_score:
                best_score, best_params = score["WAPE"], dict(params)
        chosen[family] = {"parameters": best_params, "validation_WAPE": best_score}
        print(f"SELECTED {family}: WAPE {best_score:.4f}, {best_params}", flush=True)
    baseline = min(["Seasonal naive", "ETS", "ARIMA"], key=lambda x: chosen[x]["validation_WAPE"])
    lock = {"protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
            "models": chosen, "selected_baseline": baseline,
            "rmsse_denominator": denominator,
            "validation_seconds": time.perf_counter() - started}
    dump("frozen_settings.json", lock)
    pd.DataFrame(records).to_csv(OUT / "validation_search.csv", index=False)
    dump("validation_warnings.json", WARNINGS)


def bootstrap_indices(n, block, repetitions, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(repetitions, int(np.ceil(n/block))))
    return ((starts[..., None] + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]


def wape_samples(actual, pred, indices):
    return 100 * np.abs(pred - actual)[indices].sum(axis=1) / np.abs(actual)[indices].sum(axis=1)


def paired_test(actual, candidate, comparator):
    d = abs(candidate - actual) - abs(comparator - actual)
    blocks = d.reshape(4, 3).mean(axis=1)
    null = np.asarray(list(itertools.product([-1, 1], repeat=4))) @ blocks / 4
    observed = float(d.mean())
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-8))
    return {"mean_absolute_loss_difference": observed, "p_two_sided": p,
            "quarter_block_differences": blocks.tolist()}


def evaluate(df):
    started = time.perf_counter()
    lock_path = OUT / "frozen_settings.json"
    lock = json.loads(lock_path.read_text())
    assert lock["protocol_sha256"] == hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    dump("evaluation_lock.json", {"frozen_settings_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                                 "test_opened_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    all_preds = []
    timing = {}
    for period in ["validation", "test", "robustness"]:
        dates = pd.date_range(*PROTOCOL[period], freq="MS")
        truth = df.loc[df.date.isin(dates), "turnover_million_kzt"].to_numpy()
        assert len(truth) == len(dates)
        for family, selected in lock["models"].items():
            t0 = time.perf_counter()
            pred = run_origins(df, dates, family, selected["parameters"], family.endswith("full"))
            timing[f"{period} {family}"] = time.perf_counter()-t0
            for date, actual, estimated in zip(dates, truth, pred):
                all_preds.append({"period": period, "date": date.strftime("%Y-%m-%d"),
                                  "model": family, "actual": float(actual), "prediction": float(estimated)})
    pred_df = pd.DataFrame(all_preds)
    pred_df.to_csv(OUT / "predictions.csv", index=False)
    summary = []
    ci_idx = bootstrap_indices(12, 3, 10000, 2026)
    for (period, family), group in pred_df.groupby(["period", "model"], sort=False):
        actual, pred = group.actual.to_numpy(), group.prediction.to_numpy()
        row = {"period": period, "model": family, **metrics(actual, pred, lock["rmsse_denominator"])}
        if period == "test":
            ci = np.quantile(wape_samples(actual, pred, ci_idx), [.025, .975])
            row.update({"WAPE_CI_low": float(ci[0]), "WAPE_CI_high": float(ci[1])})
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "metrics.csv", index=False)
    pivot = pred_df.loc[pred_df.period == "test"].pivot(index="date", columns="model", values="prediction")
    actual = pred_df.loc[(pred_df.period == "test") & (pred_df.model == "LightGBM full"), "actual"].to_numpy()
    comparisons = []
    for comparator in [lock["selected_baseline"], "LightGBM lag"]:
        a, b = pivot["LightGBM full"].to_numpy(), pivot[comparator].to_numpy()
        rec = {"candidate": "LightGBM full", "comparator": comparator,
               **paired_test(actual, a, b),
               "monthly_wins": int(np.sum(abs(a-actual) < abs(b-actual))),
               "relative_WAPE_reduction_percent": float(100 * (1 - abs(a-actual).sum()/abs(b-actual).sum()))}
        for length in [2,3,4]:
            idx = bootstrap_indices(12, length, 10000, 2026)
            diffs = wape_samples(actual,a,idx)-wape_samples(actual,b,idx)
            lo,hi=np.quantile(diffs,[.025,.975])
            rec[f"difference_CI_block{length}"] = [float(lo),float(hi)]
        comparisons.append(rec)
    dump("paired_comparisons.json", comparisons)
    full = pred_df.loc[(pred_df.period == "test") & (pred_df.model == "LightGBM full")].copy()
    full["signed_error"] = full.prediction-full.actual
    full["absolute_error"] = abs(full.signed_error)
    full["absolute_percentage_error"] = 100*full.absolute_error/full.actual
    full.sort_values("absolute_error",ascending=False).head(4).to_csv(OUT/"worst_errors.csv",index=False)
    def sysctl(key):
        try: return subprocess.check_output(["sysctl","-n",key],text=True).strip()
        except (OSError,subprocess.SubprocessError): return "unavailable"
    packages = ["numpy","pandas","scipy","scikit-learn","statsmodels","lightgbm","matplotlib","xlrd"]
    env = {"python":sys.version.split()[0],"platform":platform.platform(),
           "cpu":sysctl("machdep.cpu.brand_string"),"physical_cpu_cores":sysctl("hw.physicalcpu"),
           "ram_bytes":sysctl("hw.memsize"),"model_threads":2,"model_seed":42,"bootstrap_seed":2026,
           "packages":{p:importlib.metadata.version(p) for p in packages},
           "validation_seconds":lock["validation_seconds"],
           "evaluation_seconds":time.perf_counter()-started,
           "model_fit_times_seconds":timing,
           "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump("environment.json",env)
    dump("evaluation_warnings.json",WARNINGS)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=["validation","evaluate"])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    data=load_data()
    if args.stage == "validation": validation(data)
    else: evaluate(data)
