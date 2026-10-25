"""Create vector figures and LaTeX tables from the saved pilot predictions."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

if (ROOT.parent / "main.tex").is_file():
    PAPER = ROOT.parent
elif (ROOT.parent / "user_text_project" / "main.tex").is_file():
    PAPER = ROOT.parent / "user_text_project"
else:
    PAPER = ROOT / "article_assets"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
TABLES = PAPER / "pilot_tables"
TABLES.mkdir(parents=True, exist_ok=True)
pred = pd.read_csv(ROOT / "results/predictions.csv", parse_dates=["date"])
metrics = pd.read_csv(ROOT / "results/metrics.csv")
data = pd.read_csv(ROOT / "data/clean_monthly.csv", parse_dates=["date"])
audit = json.loads((ROOT / "results/data_audit.json").read_text())
lock = json.loads((ROOT / "results/frozen_settings.json").read_text())
paired = json.loads((ROOT / "results/paired_comparisons.json").read_text())
env = json.loads((ROOT / "results/environment.json").read_text())
models = ["Seasonal naive", "ETS", "ARIMA", "LightGBM lag", "LightGBM full"]
colors = {"Seasonal naive":"#777777", "ETS":"#16677C", "ARIMA":"#B07925",
          "LightGBM lag":"#8B4D80", "LightGBM full":"#1A704B"}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
                     "axes.spines.top":False, "axes.spines.right":False,
                     "pdf.fonttype":42, "svg.fonttype":"none"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def texfile(name, text):
    (TABLES/name).write_text(text + "\n")


def number(x, places=2):
    return f"{x:,.{places}f}".replace(",",r"\,")


fig,ax=plt.subplots(figsize=(6.1,2.6))
d=data.loc[data.date<"2026-01-01"]
ax.plot(d.date,d.turnover_million_kzt/1000,color="#203E52",lw=1.25)
ax.axvspan(pd.Timestamp("2023-01-01"),pd.Timestamp("2024-12-31"),color="#DDECF0",label="Validation")
ax.axvspan(pd.Timestamp("2025-01-01"),pd.Timestamp("2025-12-31"),color="#E4EEDC",label="Test")
ax.set(xlabel="Month",ylabel="Nominal turnover (billion KZT)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="y",alpha=.2)
ax.legend(frameon=False,loc="upper left",ncol=2)
fig.tight_layout()
save(fig,"pilot_series")

test=pred.loc[pred.period=="test"]
fig,ax=plt.subplots(figsize=(6.1,2.8))
actual=test.loc[test.model=="ETS"]
ax.plot(actual.date,actual.actual/1000,color="black",lw=2,marker="o",ms=3,label="Observed")
for model in models:
    g=test.loc[test.model==model]
    ax.plot(g.date,g.prediction/1000,label=model,color=colors[model],lw=1.2,
            ls="--" if model in ["Seasonal naive","LightGBM lag"] else "-")
ax.set(xlabel="Forecast month in 2025",ylabel="Nominal turnover (billion KZT)")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.grid(axis="y",alpha=.2)
ax.legend(frameon=False,ncol=3,fontsize=7.5,loc="upper left")
fig.tight_layout()
save(fig,"pilot_forecasts")

matrix=np.array([((g:=test.loc[test.model==m]).prediction.to_numpy()-g.actual.to_numpy())/g.actual.to_numpy()*100 for m in models])
limit=float(np.ceil(np.abs(matrix).max()/5)*5)
fig,ax=plt.subplots(figsize=(6.1,2.5))
im=ax.imshow(matrix,cmap="RdBu_r",aspect="auto",vmin=-limit,vmax=limit)
ax.set_xticks(range(12),actual.date.dt.strftime("%b"))
ax.set_yticks(range(5),models)
ax.set_xlabel("Forecast month in 2025")
for i in range(5):
    for j in range(12):
        ax.text(j,i,f"{matrix[i,j]:.0f}",ha="center",va="center",fontsize=7,
                color="white" if abs(matrix[i,j])>limit*.55 else "black")
cbar=fig.colorbar(im,ax=ax,pad=.025)
cbar.set_label("Signed error (%)")
fig.tight_layout()
save(fig,"pilot_errors")

dataset=r"""\begin{table}[htbp]
\caption{Final national monthly dataset, cleaning audit and chronological partitions. Statistical models use all available past months. LightGBM loses the initial 12 months only when constructing supervised training examples. The target summary statistics are for the primary 132-month sample.}
\label{tab:pilot-data}
\centering\small
\begin{tabularx}{\textwidth}{@{}lXrr@{}}\toprule
Partition & Period & Raw months & ML rows\\\midrule
Initial training & Jan 2015--Dec 2022 & 96 & 84\\
Validation origins & Jan 2023--Dec 2024 & 24 & 24\\
Primary test origins & Jan--Dec 2025 & 12 & 12\\
Primary sample & Jan 2015--Dec 2025 & 132 & 120\\
Extra temporal check & Jan--Aug 2026 & 8 & 8\\\bottomrule
\end{tabularx}
\par\vspace{0.4em}
\begin{tabularx}{\textwidth}{@{}p{0.23\textwidth}X@{}}\toprule
Source and target & BNS Indicator sheet; national nominal turnover in million KZT.\\
Records and scope & 132 primary observations after cleaning. Eight 2026 observations are reserved for the temporal check; four 2026 cells are empty.\\
Cleaning & One text-to-number conversion; no missing primary targets, duplicates, imputations or outlier removals. No primary record was removed.\\
Target distribution & Mean SAMPLEMEAN, median SAMPLEMEDIAN, SD SAMPLESD; range SAMPLEMIN--SAMPLEMAX billion KZT.\\\bottomrule
\end{tabularx}
\end{table}"""
for token,key in [("SAMPLEMEAN","mean"),("SAMPLEMEDIAN","50%"),("SAMPLESD","std"),("SAMPLEMIN","min"),("SAMPLEMAX","max")]:
    dataset=dataset.replace(token,number(audit["target_description"][key]/1000))
texfile("data.tex",dataset)
rows=[]
for m in models:
    r=metrics.loc[(metrics.period=="test")&(metrics.model==m)].iloc[0]
    rows.append(m+" & "+" & ".join([number(r.MAE/1000),number(r.RMSE/1000),number(r.WAPE),number(r.RMSSE,3),number(r.Bias/1000)])+r"\\")
texfile("main.tex",r"""\begin{table}[htbp]
\caption{Preliminary one-month-ahead results on the same 12 test months in 2025. MAE, RMSE and signed mean bias are in billion KZT; WAPE is a percentage and RMSSE is unitless. Positive bias means overprediction. RMSSE uses the seasonal scale from the initial training sample only. Lower error values are favourable, but no row is labelled statistically superior.}
\label{tab:pilot-main}
\centering\small
\begin{tabular}{@{}lrrrrr@{}}\toprule
Model & MAE & RMSE & WAPE & RMSSE & Bias\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}
\end{table}""")
rows=[]
for m in models:
    r=metrics.loc[(metrics.period=="test")&(metrics.model==m)].iloc[0]
    rows.append(m+f" & {number(r.WAPE)} & [{number(r.WAPE_CI_low)}, {number(r.WAPE_CI_high)}]"+r"\\")
part_a="\n".join(rows)
rows=[]
for r in paired:
    lo,hi=r["difference_CI_block3"]
    rows.append("Full vs "+r["comparator"].replace("LightGBM ","")+f" & [{number(lo)}, {number(hi)}] & {r['p_two_sided']:.3f} & {r['monthly_wins']}/12"+r"\\")
texfile("uncertainty.tex",r"""\begin{table}[htbp]
\caption{Uncertainty estimates for 2025. Panel A gives 95\% circular block-bootstrap percentile intervals for WAPE using three-month blocks, 10,000 resamples and seed 2026. Panel B compares full-feature LightGBM with the validation-selected ETS baseline and lag-only LightGBM. Differences are candidate minus comparator WAPE in percentage points. The two-sided exact sign test flips four quarterly mean loss differences and has a minimum possible p-value of 0.125.}
\label{tab:pilot-uncertainty}
\centering\small
\begin{tabular}{@{}lrr@{}}\toprule
\multicolumn{3}{l}{Panel A: WAPE variability}\\
Model & WAPE (\%) & 95\% bootstrap interval\\\midrule
"""+part_a+r"""
\bottomrule\end{tabular}
\par\vspace{0.4em}
\begin{tabular}{@{}llrr@{}}\toprule
\multicolumn{4}{l}{Panel B: paired comparisons}\\
Comparison & 95\% interval for difference & $p$ & Wins\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}
\end{table}""")
rows=[]
for m in models:
    vals=[metrics.loc[(metrics.period==period)&(metrics.model==m),"WAPE"].iloc[0] for period in ["validation","test","robustness"]]
    rows.append(m+" & "+" & ".join(number(v) for v in vals)+r"\\")
texfile("ablation.tex",r"""\begin{table}[htbp]
\caption{Feature ablation and temporal robustness using frozen model settings. Entries are WAPE (\%). Validation covers 24 expanding origins in 2023--2024, the primary test covers 12 origins in 2025, and the additional temporal check covers eight origins in January--August 2026. Lag-only LightGBM uses lags 1, 3 and 12; the full version adds month and trailing means over 3 and 12 months. Validation values were used for selection and are not independent test results.}
\label{tab:pilot-ablation}
\centering\small
\begin{tabular}{@{}lrrr@{}}\toprule
Model & Validation & 2025 test & 2026 check\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}
\end{table}""")
worst=pd.read_csv(ROOT/"results/worst_errors.csv",parse_dates=["date"])
rows=[]
for _,r in worst.iterrows():
    vals=[number(r.actual/1000),number(r.prediction/1000),number(r.signed_error/1000),number(r.absolute_percentage_error)]
    rows.append(r.date.strftime("%b %Y")+" & "+" & ".join(vals)+r"\\")
texfile("worst.tex",r"""\begin{table}[htbp]
\caption{Four largest absolute errors for full-feature LightGBM in the 2025 test, ranked by absolute error. Observed values, forecasts and signed errors are in billion KZT; APE is the month-specific absolute percentage error. These months were included in every reported metric.}
\label{tab:pilot-worst}
\centering\small
\begin{tabular}{@{}lrrrr@{}}\toprule
Month & Observed & Forecast & Error & APE (\%)\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}
\end{table}""")

macros=[]
for period,tag in [("validation","Val"),("test","Test"),("robustness","Check")]:
    for m,short in zip(models,["Naive","ETS","ARIMA","Lag","Full"]):
        r=metrics.loc[(metrics.period==period)&(metrics.model==m)].iloc[0]
        for metric in ["MAE","RMSE","WAPE","RMSSE","Bias"]:
            value=r[metric]/1000 if metric in ["MAE","RMSE","Bias"] else r[metric]
            macros.append("\\newcommand{\\"+tag+short+metric+"}{"+number(value,3 if metric=="RMSSE" else 2)+"}")
for i,r in enumerate(paired):
    tag=["Baseline","Ablation"][i]
    macros.append("\\newcommand{\\"+tag+"Wins}{"+str(r["monthly_wins"])+"}")
    macros.append("\\newcommand{\\"+tag+"Reduction}{"+number(r["relative_WAPE_reduction_percent"])+"}")
    macros.append("\\newcommand{\\"+tag+"PValue}{"+number(r["p_two_sided"],3)+"}")
for label,key in [("SampleMean","mean"),("SampleMin","min"),("SampleMax","max"),("SampleMedian","50%"),("SampleSD","std")]:
    macros.append("\\newcommand{\\"+label+"}{"+number(audit["target_description"][key]/1000)+"}")
macros.append(r"\newcommand{\PilotRunMinutes}{"+number((env["validation_seconds"]+env["evaluation_seconds"])/60)+"}")
texfile("values.tex","\n".join(macros))
print("Generated three vector figures, five numbered tables, and numeric macros.")
