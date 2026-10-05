"""
src/metrics.py - Step 13 numbers: monthly book returns, summary statistics, OOS R2 and rank-IC summary.
Convention: weights at formation month-end t are realised with ret_exc_lead1m (excess return of month t+1).
"""
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm


def load_market(cfg) -> pl.DataFrame:
    """TB3MS (annualised %) and S&P 500 monthly total return keyed on month-end eom."""
    f = cfg["data_files"]
    tb = pl.read_csv(f["tb3ms"], try_parse_dates=True).select(pl.col("eom").cast(pl.Date), "tb3ms")
    sp = pl.read_csv(f["sp500"], try_parse_dates=True).select(pl.col("eom").cast(pl.Date), "sp500_ret")
    return tb.join(sp, on="eom", how="full", coalesce=True).sort("eom")


def book_returns(weights, panel, market, cfg):
    """One row per formation month: returns, benchmark, legs, exposures, counts, turnover. Returns (rets, n_missing)."""
    w = weights.select("permno", "eom", "weight").join(
        panel.select("permno", "eom", "ret_exc_lead1m"), on=["permno", "eom"], how="left")
    n_missing = int(w["ret_exc_lead1m"].is_null().sum())
    w = w.with_columns(r=pl.col("ret_exc_lead1m").fill_null(0.0)).with_columns(c=pl.col("weight") * pl.col("r"))
    g = w.group_by("eom").agg(
        long_leg=pl.col("c").filter(pl.col("weight") > 0).sum(),
        short_leg=pl.col("c").filter(pl.col("weight") < 0).sum(),
        gross=pl.col("weight").abs().sum(), net=pl.col("weight").sum(),
        n_long=(pl.col("weight") > 0).sum(), n_short=(pl.col("weight") < 0).sum()).sort("eom")
    # turnover = sum |w_t - w_{t-1}| over the union of names (absent = 0); first month starts from cash
    pw = w.to_pandas().pivot_table(index="eom", columns="permno", values="weight", aggfunc="sum").sort_index().fillna(0.0)
    to = pw.diff().abs().sum(axis=1)
    if len(to):
        to.iloc[0] = pw.iloc[0].abs().sum()
    g = g.join(pl.DataFrame({"eom": pl.Series(to.index).cast(pl.Date), "turnover": to.values}), on="eom")
    mk = market.with_columns(eom=pl.col("eom").dt.offset_by("-1mo").dt.month_end()).select(  # key on formation month
        "eom", rf=pl.col("tb3ms") / 1200, sp500=pl.col("sp500_ret"))
    prem = cfg["premium_annual"] / 12
    rets = (g.join(mk, on="eom", how="left")
            .with_columns(Date=pl.col("eom").dt.offset_by("1mo").dt.month_start(),
                          excess=pl.col("long_leg") + pl.col("short_leg"))
            .with_columns(total=pl.col("rf") + pl.col("excess"), benchmark=pl.col("rf") + prem,
                          active=pl.col("excess") - prem)
            .select("eom", "Date", "total", "excess", "active", "long_leg", "short_leg", "benchmark", "sp500",
                    "gross", "net", "n_long", "n_short", "turnover", "rf"))
    return rets, n_missing


def _ir(a):
    sd = np.std(a, ddof=1)
    return float(np.sqrt(12) * np.mean(a) / sd) if sd > 0 else float("nan")


def _leg(r):
    """Arithmetic/geometric stats for one monthly return series."""
    n = len(r)
    cum = float(np.prod(1 + r) - 1)
    return {"mean_monthly": float(np.mean(r)), "ann_arith": float(np.mean(r) * 12),
            "cagr": float((1 + cum) ** (12 / n) - 1), "cumulative": cum}


def _mm(x):
    return {"avg": float(np.mean(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def _maxdd(r):
    c = np.cumprod(1 + np.asarray(r))
    return float((c / np.maximum.accumulate(np.r_[1.0, c])[1:] - 1).min())


def summary_stats(rets, cfg) -> dict:
    """Headline statistics of the monthly book (see contract)."""
    d = rets.to_pandas()
    rf = d["rf"].values if "rf" in d else (d["benchmark"] - cfg["premium_annual"] / 12).values
    tot, exc, act, sp = d["total"].values, d["excess"].values, d["active"].values, d["sp500"].values
    sd_tot = np.std(tot - rf, ddof=1)
    out = {"n_months": len(d), "ir": _ir(act), "sharpe": float(np.sqrt(12) * np.mean(tot - rf) / sd_tot),
           "strategy": _leg(tot), "benchmark": _leg(d["benchmark"].values), "sp500": _leg(sp),
           "long_leg": _leg(d["long_leg"].values), "short_leg": _leg(d["short_leg"].values),
           "max_drawdown": _maxdd(tot), "sp500_max_drawdown": _maxdd(sp),
           "best_month": float(tot.max()), "worst_month": float(tot.min()),
           "hit_rate": float(np.mean((exc > 0) & (act > 0))),
           "gross": _mm(d["gross"]), "net": _mm(d["net"]), "turnover": _mm(d["turnover"]),
           "n_long": _mm(d["n_long"]), "n_short": _mm(d["n_short"]),
           "corr_sp500": float(np.corrcoef(tot, sp)[0, 1])}
    ols = sm.OLS(exc, sm.add_constant(sp - rf)).fit(cov_type="HAC", cov_kwds={"maxlags": cfg["nw_lags"]})
    out.update(alpha_monthly=float(ols.params[0]), alpha_annual=float(ols.params[0] * 12), alpha_t=float(ols.tvalues[0]),
               beta=float(ols.params[1]), beta_t=float(ols.tvalues[1]))
    yr = d["Date"].dt.year
    cal = lambda x: pd.Series(x).groupby(yr.values).apply(lambda s: float(np.prod(1 + s) - 1))
    out["calendar_years"] = {str(y): {"strategy": v, "benchmark": float(cal(d["benchmark"].values)[y]),
                                      "sp500": float(cal(sp)[y])} for y, v in cal(tot).items()}
    return out


def oos_r2(pred) -> float:
    """Zero-benchmark OOS R2 = 1 - sum (r - r_hat)^2 / sum r^2 on rows with non-null target."""
    p = pred.drop_nulls(["ret_exc_lead1m", "ret_hat"])
    r, h = p["ret_exc_lead1m"].to_numpy(), p["ret_hat"].to_numpy()
    return float(1 - np.sum((r - h) ** 2) / np.sum(r ** 2))


def ic_summary(pred, cfg) -> dict:
    """Mean monthly Spearman rank IC of score vs target, ICIR = mean/std * sqrt(12)."""
    p = pred.drop_nulls(["ret_exc_lead1m", "score"]).to_pandas()
    ic = p.groupby("eom").apply(lambda g: g["score"].corr(g["ret_exc_lead1m"], method="spearman")).dropna()
    sd = ic.std(ddof=1)
    return {"mean_ic": float(ic.mean()), "icir": float(ic.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan"),
            "n_months": int(len(ic))}
