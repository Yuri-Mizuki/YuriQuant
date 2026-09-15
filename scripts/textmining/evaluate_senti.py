"""
AI 41 研报情感因子族评估
========================

对 six 因子（senti / senti_adj / report_num / report_score / senti_res /
senti_adj_res）统一评估，对齐 AI 41 测试框架：

1. 覆盖度（月均覆盖股票数）
2. 月度 RankIC：均值 / ICIR / 正占比 / Newey-West t（项目口径 stats.robust_stats）
3. 分层回测 5 层：月度调仓等权，基准 **全A等权**（AI 41 分层基准为等权基准；
   本机缓存无中证500 指数日线，以全A等权替代，如实声明）
4. 六因子汇总对比表 + 多空（L1-L5）细节

注意：旧框架 evaluate_sue_txt.load_next_ret 硬编码 daily_hs300.parquet
（zz1000 因子会错配收益），本脚本自带按 daily_all_a 的收益加载器。

用法：
    python -m scripts.textmining.evaluate_senti --pool zz1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.textmining._paths import Out  # noqa: E402
from stats.robust_stats import nw_tstat  # noqa: E402

OUT_DIR = Out("senti")
log = setup_logging("senti_eval")

FACTORS = ["senti", "senti_adj", "report_num", "report_score",
           "senti_res", "senti_adj_res"]


def load_next_ret(begin: str = "20190101") -> tuple[pd.DataFrame, pd.Series]:
    """个股下月收益宽表（date×code）+ 全A等权月收益（来自 daily_all_a）。"""
    from config import Config
    p = Path(str(Config.cache()["root"]).replace("//", "/")) / "daily_all_a.parquet"
    daily = pd.read_parquet(p)
    daily = daily[daily.index.get_level_values("date") >= pd.Timestamp(begin)]
    close = daily["close"].unstack("code")
    monthly_last = close.groupby(close.index.to_period("M")).last()
    ret = monthly_last.pct_change().shift(-1)  # T 月末 → T+1 月末
    bench = ret.mean(axis=1).rename("bench")   # 全A等权
    return ret, bench


def _to_month(factor: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    """因子月末交易日 → 月 Period 连接键（收益宽表 index 本身是 Period）。"""
    f = factor.copy()
    f["month"] = pd.to_datetime(f["date"]).dt.to_period("M")
    long = ret.stack().rename("ret").reset_index()
    long.columns = ["month", "code", "ret"]  # 按位置命名（索引名不定）
    return f[["month", "code", "factor"]].merge(long, on=["month", "code"],
                                                how="inner")


def rank_ic_stats(factor: pd.DataFrame, ret: pd.DataFrame) -> dict:
    m = _to_month(factor, ret)
    ics = m.groupby("month").apply(
        lambda g: g["factor"].corr(g["ret"], method="spearman"),
        include_groups=False).dropna()
    t, _, _ = nw_tstat(ics.values)
    return {"ic_mean": ics.mean(), "icir": ics.mean() / ics.std() if ics.std() else np.nan,
            "ic_positive": (ics > 0).mean(), "ic_nw_t": t, "n_months": len(ics)}


def stratified(factor: pd.DataFrame, ret: pd.DataFrame, bench: pd.Series,
               n_layers: int = 5, min_stocks: int = 50) -> dict:
    m = _to_month(factor, ret)
    cov = m.groupby("month")["code"].nunique()
    m = m[m["month"].isin(cov[cov >= min_stocks].index)]
    if m.empty:
        return {}
    m["layer"] = m.groupby("month")["factor"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_layers, labels=False) + 1)
    layer_ret = m.groupby(["month", "layer"])["ret"].mean().unstack("layer")
    layer_ret = layer_ret.join(bench, how="left").dropna(subset=["bench"])

    def annual(s: pd.Series) -> float:
        s = s.dropna()
        return (1 + s).prod() ** (12 / len(s)) - 1 if len(s) >= 12 else np.nan

    out = {"n_months": int(layer_ret["bench"].notna().sum()),
           "bench_annual": annual(layer_ret["bench"])}
    for c in layer_ret.columns:
        if c == "bench":
            continue
        out[f"L{c}_annual"] = annual(layer_ret[c])
    out["L1_excess"] = out["L1_annual"] - out["bench_annual"]
    out["L1_L5"] = out["L1_annual"] - out[f"L{n_layers}_annual"]
    return out


def _load_neutral_covs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """行业（申万一级）与对数市值月度面板（Period 索引），来自全A滚动实验 _base。"""
    base = ROOT / "reports" / "alla_rolling" / "_base"
    ind = pd.read_parquet(base / "industry.parquet")
    cap = pd.read_parquet(base / "market_cap.parquet")

    def _monthly(pnl: pd.DataFrame) -> pd.DataFrame:
        me = pnl.index.to_series().groupby(pnl.index.to_period("M")).last()
        p = pnl[pnl.index.isin(set(me))].copy()
        p.index = pd.to_datetime(p.index).to_period("M")
        return p[~p.index.duplicated(keep="last")]

    return _monthly(ind), np.log(_monthly(cap))


def _winsor_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad and mad > 0:
        s = s.clip(med - 5 * mad, med + 5 * mad)
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * np.nan


def neutralized_ic_stats(factor: pd.DataFrame, ret: pd.DataFrame,
                         ind_m: pd.DataFrame, cap_m: pd.DataFrame) -> dict:
    """行业+市值中性化后的月度 RankIC（因子健壮性体检）。

    每月截面：因子 winsor+标准化 → 对 [行业哑变量, ln_mktcap] 回归取残差。
    """
    f = factor.copy()
    f["month"] = pd.to_datetime(f["date"]).dt.to_period("M")
    long = ret.stack().rename("ret").reset_index()
    long.columns = ["month", "code", "ret"]
    m = f[["month", "code", "factor"]].merge(long, on=["month", "code"],
                                             how="inner")
    ics = []
    for mo, g in m.groupby("month"):
        if mo not in ind_m.index or mo not in cap_m.index:
            continue
        g = g.copy()
        g["ind"] = ind_m.loc[mo].reindex(g["code"]).values
        g["lncap"] = cap_m.loc[mo].reindex(g["code"]).values
        g["z"] = _winsor_z(g["factor"])
        X = [pd.Series(1.0, index=g.index), g["lncap"]]
        X.extend(pd.get_dummies(g["ind"], drop_first=True).astype(float)
                 .reindex(g.index).fillna(0).pipe(lambda d: [d[c] for c in d.columns]))
        X = pd.concat(X, axis=1)
        ok = g["z"].notna() & X.notna().all(axis=1)
        if ok.sum() < 30:
            continue
        beta, *_ = np.linalg.lstsq(X[ok].values, g.loc[ok, "z"].values, rcond=None)
        res = pd.Series(np.nan, index=g.index)
        res[ok] = g.loc[ok, "z"].values - X[ok].values @ beta
        gg = pd.DataFrame({"month": mo, "code": g["code"], "factor": res})
        gg = gg.merge(long, on=["month", "code"], how="inner")
        if len(gg) >= 30:
            ics.append(gg["factor"].corr(gg["ret"], method="spearman"))
    ics = pd.Series(ics).dropna()
    t, _, _ = nw_tstat(ics.values)
    return {"neu_ic_mean": ics.mean(),
            "neu_ic_nw_t": t,
            "neu_ic_positive": (ics > 0).mean(), "neu_n_months": len(ics)}


def run(pool: str = "zz1000", begin: str = "20190101"):
    ret, bench = load_next_ret(begin)
    ind_m, cap_m = _load_neutral_covs()
    summary, details = [], {}
    for name in FACTORS:
        p = OUT_DIR / f"senti_factor_{name}_{pool}.parquet"
        if not p.exists():
            log.warning("%s 不存在，跳过", p)
            continue
        f = pd.read_parquet(p).reset_index()
        f["date"] = pd.to_datetime(f["date"])
        cov = f.groupby("date")["code"].size()
        ic = rank_ic_stats(f, ret)
        nic = neutralized_ic_stats(f, ret, ind_m, cap_m)
        st = stratified(f, ret, bench)
        summary.append({"factor": name,
                        "cov_mean": cov.mean(), "cov_median": cov.median(),
                        **ic, **nic, **st})
        details[name] = st
    sm = pd.DataFrame(summary).round(4)
    print("\n== AI 41 研报情感因子族评估（池 %s，基准 全A等权）==" % pool)
    cols = ["factor", "cov_mean", "ic_mean", "ic_nw_t", "ic_positive",
            "neu_ic_mean", "neu_ic_nw_t", "L1_excess", "L1_L5", "n_months"]
    print(sm[cols].to_string(index=False))

    lines = [f"# AI 41 因子族评估（pool={pool}，基准=全A等权，月度调仓）", "",
             "## 主表（含行业+市值中性化 IC 体检）", "",
             sm[cols].to_string(index=False), "",
             "全表：", sm.to_string(index=False), ""]
    for name, st in details.items():
        lines.append(f"## {name}")
        lines.append({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in st.items()}.__str__())
        lines.append("")
    out = OUT_DIR / f"senti_eval_{pool}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("评估已存: %s", out)
    return sm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000", "all_a"])
    ap.add_argument("--begin", default="20190101")
    args = ap.parse_args()
    run(args.pool, args.begin)
