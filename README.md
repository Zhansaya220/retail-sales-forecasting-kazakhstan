# Retail sales forecasting in Kazakhstan

Reproducible preliminary forecasts of national monthly retail turnover using official Kazakhstan Bureau of National Statistics data. The experiment compares seasonal naive, ETS, ARIMA, lag-only LightGBM and full-feature LightGBM.

## Recorded results

All models use the same twelve expanding one-month forecast origins in 2025. Specifications are selected using only the 2023–2024 validation period and frozen before the test.

| Model | Test WAPE (%) | Test MAE (billion KZT) | Test RMSE (billion KZT) |
|---|---:|---:|---:|
| Seasonal naive | 14.95 | 328.88 | 350.42 |
| ETS | 2.98 | 65.45 | 79.87 |
| ARIMA | 15.97 | 351.25 | 559.45 |
| LightGBM lag | 9.60 | 211.04 | 300.43 |
| LightGBM full | 11.03 | 242.53 | 312.80 |

The practical targets for both research questions are not attained on the primary test. The additional January–August 2026 check is reported separately. Neither comparison supports a significance claim under the supplementary exact block sign test. Negative findings and the largest errors are retained.

## Data

Source index: https://stat.gov.kz/en/industries/economy/local-market/dynamic-tables/

Workbook: https://stat.gov.kz/api/iblock/element/38031/file/en/

The archived workbook was retrieved on 24 October 2026. SHA-256:

```text
868c064be290fd7a36cd0a4594feb3984966c5bc088d67c2c856e6667133f823
```

The Indicator sheet gives national nominal monthly turnover in million KZT. Its unit conflicts with the percentage label in Metadata; the analysis uses the Indicator sheet's explicit unit. One text-formatted numerical entry is normalized during reshaping. No primary target is deleted or imputed.

The primary series contains 132 complete months, January 2015–December 2025. Initial training uses 2015–2022, validation 2023–2024 and testing 2025. The first twelve months initialize lag features, leaving 84 supervised examples in the initial 96-month training window. Eight available 2026 months form a separate temporal check. Four remaining 2026 cells are empty.

## Reproduce

Use Python 3.12. Create a virtual environment and install the pinned packages:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pilot.py --stage validation
python run_pilot.py --stage evaluate
```

On Windows, activate the environment with `.venv\Scripts\activate`. On macOS, LightGBM requires an available OpenMP runtime.

The first stage evaluates the predefined grids on validation origins and writes `results/frozen_settings.json`. The second checks the protocol hash, records the settings hash and generates the 2025 test forecasts and separate 2026 check. At each origin, only earlier target values are available. There is no retuning on test outcomes. Model seed 42, bootstrap seed 2026 and two LightGBM threads are fixed.

The workbook is archived so a new download is unnecessary. The live source may be revised; the script intentionally rejects a workbook whose checksum does not match the archived snapshot.

To regenerate the vector figures and LaTeX tables after running the experiment:

```sh
python build_assets.py
```

In a standalone clone, these assets are written to `article_assets/`. The script also supports the accompanying Overleaf project layout. All reported monetary errors in the article are converted to billion KZT; calculations retain unrounded forecasts.

## Included evidence

`data/` contains the unchanged source XLS and the normalized monthly CSV. `results/` contains the data audit, every validation candidate, frozen model settings, the evaluation lock, all dated predictions, metrics, paired comparisons, worst errors, warning logs and the measured software/hardware environment. `protocol.json` fixes the grids, partitions, practical criteria and supplementary uncertainty procedure.

The recorded analysis ran with Python 3.12.14, an Apple M2 Pro, 16 GiB RAM and two LightGBM threads. Validation took approximately 30.77 seconds and evaluation approximately 7.92 seconds, excluding environment installation, downloads, plotting and document compilation. Run times vary by machine.

## Scope and uncertainty

This is a retrospective pilot on one revised national monthly series. It does not establish store–SKU performance, hierarchical coherence, inventory savings, probabilistic calibration or Big Data scalability. Neural models and external regressors are not tested.

The supplementary paired circular block bootstrap uses 10,000 resamples, a primary block length of three months, and sensitivity lengths of two and four months. The exact two-sided block sign test enumerates sixteen sign assignments of four quarterly mean absolute-loss differences. Its minimum possible p-value is 0.125, so it cannot establish significance at 0.05. The inferential procedures are exploratory and were not externally preregistered. The source vintage is later than the forecast periods, so this is not a real-time release-vintage evaluation.
