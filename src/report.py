"""
src/report.py - Steps 13-14: performance pack (performance.json + charts) and submission files (holdings.csv, returns.csv).
Chart style ported from the AlphaBERT repo (src/evaluate.py): 10x5 figures, default tab colours, curves anchored at 0 before
the first month, zero lines in black, hurdle as red dashed line, titles end with "gross of trading costs, <period>".
"""
import json
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics import summary_stats, oos_r2


def _rep_savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rep_line(series, title, path, hline=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, s in series.items():
        ax.plot(s.index, s.values, label=label)
    if hline is not None:
        ax.axhline(hline, color="black", lw=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend()
    _rep_savefig(fig, path)


def _rep_labels(keys, panel, filings) -> pl.DataFrame:
    """keys[permno, eom] -> + ticker, company_name from the panel row; gaps filled from the 8-K row nearest in filing_date."""
    k = keys.join(panel.select("permno", "eom", "ticker", "company_name"), on=["permno", "eom"], how="left").sort("eom")
    for col in ["ticker", "company_name"]:
        if filings is None or col not in filings.columns:
            continue
        fl = filings.select("permno", "filing_date", pl.col(col).alias(col + "_8k")).drop_nulls().sort("filing_date")
        if fl.height:
            k = (k.join_asof(fl, left_on="eom", right_on="filing_date", by="permno", strategy="nearest")
                 .with_columns(pl.col(col).fill_null(pl.col(col + "_8k"))).drop("filing_date", col + "_8k"))
    return k


def _rep_contrib(weights, panel, filings) -> pl.DataFrame:
    """Total gross contribution (sum of w*ret) of each stock, labelled "TICKER, Company Name", sorted descending."""
    c = (weights.join(panel.select("permno", "eom", "ret_exc_lead1m"), on=["permno", "eom"], how="left")
         .with_columns(c=pl.col("weight") * pl.col("ret_exc_lead1m").fill_null(0.0))
         .group_by("permno").agg(contrib=pl.col("c").sum(), eom=pl.col("eom").max()))
    c = _rep_labels(c, panel, filings).with_columns(
        label=pl.col("ticker").fill_null("?") + ", " + pl.col("company_name").fill_null("?").str.to_titlecase())
    return c.sort("contrib", descending=True).select("permno", "label", "contrib")


def _rep_composition(weights, panel) -> dict:
    """Largest weight, top-10 share of gross, short-book market cap / dollar volume / small-cap share (monthly means)."""
    w = weights.to_pandas().assign(a=lambda x: x["weight"].abs())
    top10 = w.groupby("eom")["a"].apply(lambda s: s.nlargest(10).sum() / s.sum())
    me = "me_raw" if "me_raw" in panel.columns else "market_equity"
    dv = "dolvol_raw" if "dolvol_raw" in panel.columns else "dolvol_126d"
    s = (weights.filter(pl.col("weight") < 0).join(
        panel.select("permno", "eom", me, dv, "size_grp"), on=["permno", "eom"], how="left").to_pandas())
    s["a"] = -s["weight"]
    s["small"] = s["size_grp"].isin(["micro", "nano", "small"]) * s["a"]
    g = s.groupby("eom")
    wavg = lambda col: float(g.apply(lambda x: np.average(x[col].fillna(0), weights=x["a"])).mean())
    return {"largest_weight": float(w["a"].max()), "top10_share_of_gross": float(top10.mean()),
            "short_avg_market_cap": wavg(me), "short_median_market_cap": float(s[me].median()),
            "short_avg_dollar_volume": wavg(dv), "short_median_dollar_volume": float(s[dv].median()),
            "short_small_cap_share": float((g["small"].sum() / g["a"].sum()).mean()),
            "small_cap_groups": ["micro", "nano", "small"]}


def performance_pack(rets, weights, panel, filings, preds, cfg) -> dict:
    """Write output/performance.json and the charts in output/figures/. Returns the stats dict."""
    out = cfg["output_dir"]
    figs = out / "figures"
    stats = summary_stats(rets, cfg)
    p = preds if "ret_exc_lead1m" in preds.columns else preds.join(
        panel.select("permno", "eom", "ret_exc_lead1m"), on=["permno", "eom"], how="left")
    stats["oos_r2"] = oos_r2(p)
    stats["book"] = _rep_composition(weights, panel)
    stats["n_missing_returns"] = int(weights.join(panel.select("permno", "eom", "ret_exc_lead1m"), on=["permno", "eom"],
                                                  how="left")["ret_exc_lead1m"].is_null().sum())
    stats["notes"] = ["S&P 500 series is the total return index from SP500.csv", "returns are gross of trading costs"]
    d = rets.to_pandas().set_index("Date")
    title = lambda t: f"{t}, gross of trading costs, {d.index[0]:%b %Y} to {d.index[-1]:%b %Y}"
    cum = lambda s: (1 + s).cumprod() - 1
    dd = lambda s: (1 + s).cumprod() / (1 + s).cumprod().cummax() - 1
    anchor = lambda s: pd.concat([pd.Series([0.0], index=[d.index[0] - pd.offsets.MonthBegin(1)]), s])
    _rep_line({"Strategy": anchor(cum(d["total"])), "Benchmark": anchor(cum(d["benchmark"])), "S&P 500": anchor(cum(d["sp500"]))},
              title("Cumulative return"), figs / "cumulative_returns.png", ylabel="Cumulative return")
    _rep_line({"Strategy": anchor(dd(d["total"])), "S&P 500": anchor(dd(d["sp500"]))},
              title("Drawdown"), figs / "underwater.png", ylabel="Drawdown")
    _rep_line({"Active return (ann.)": d["active"].rolling(12).mean() * 12},
              title("Rolling 12-month annualized active return"), figs / "rolling_active_return.png", hline=0)
    _rep_line({"IR": np.sqrt(12) * d["active"].rolling(12).mean() / d["active"].rolling(12).std(ddof=1)},
              title("Rolling 12-month information ratio"), figs / "rolling_ir.png", hline=0)
    x = d["sp500"] - d["rf"]
    _rep_line({"Beta": d["excess"].rolling(12).cov(x) / x.rolling(12).var()},
              title("Rolling 12-month beta vs S&P 500"), figs / "rolling_beta.png", hline=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(d["total"], bins=20, color="tab:blue", alpha=0.85)
    ax.axvline(d["benchmark"].mean(), color="red", linestyle="--", label="Avg monthly hurdle")
    ax.set_title(title("Distribution of monthly returns"))
    ax.legend()
    _rep_savefig(fig, figs / "return_histogram.png")
    c = _rep_contrib(weights, panel, filings).to_pandas()
    both = pd.concat([c.head(10), c.tail(10)])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(both["label"], both["contrib"], color=["tab:green" if v >= 0 else "tab:red" for v in both["contrib"]])
    ax.invert_yaxis()
    ax.set_title(title("Top/bottom return contributors (excess-return contribution)"))
    _rep_savefig(fig, figs / "contributors.png")
    stats["top_contributors"] = c.head(10).to_dict("records")
    stats["bottom_contributors"] = c.tail(10).to_dict("records")
    (out / "performance.json").write_text(json.dumps(stats, indent=2, default=float))
    return stats


def write_submission(weights, panel, filings, rets, cfg) -> dict:
    """output/holdings.csv (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT in % of NAV) and output/returns.csv."""
    out = cfg["output_dir"]
    h = _rep_labels(weights.select("permno", "eom", "weight"), panel, filings)
    unl = h.filter(pl.col("ticker").is_null() | pl.col("company_name").is_null())["permno"].unique().to_list()
    h.sort("eom", "permno").select(
        Date=pl.col("eom").dt.offset_by("1mo").dt.month_start().dt.strftime("%Y-%m-%d"), PERMNO=pl.col("permno"),
        TICKER=pl.col("ticker").fill_null(""), **{"COMPANY NAME": pl.col("company_name").fill_null("")},
        WEIGHT=(pl.col("weight") * 100).round(6)).write_csv(out / "holdings.csv")
    rets.with_columns(Date=pl.col("Date").dt.strftime("%Y-%m-%d")).select(
        "Date", "total", "excess", "active", "long_leg", "short_leg", "benchmark", "sp500").write_csv(out / "returns.csv")
    if unl:
        print(f"WARNING: {len(unl)} stocks stay unlabeled after 8-K fill: {unl[:20]}")
    return {"unlabeled_permnos": unl}
