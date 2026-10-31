"""研报复现专用个股因子构建（只在 `reports/report_replica/` 落盘，不进 920 因子库）。

动机：研报 §2.2/§2.4 的「超预期」「估值相关」两类因子是**成分股加权**，需要个股层
因子再做行业内自由流通市值加权。因子库里现成的只有 `sue_q`（单季扣非）、
`ep_ttm`、`roe_ttm`，缺研报点名的其余变体，所以在这里补建。

## 建的因子（研报原表 → 本实现）

超预期（§2.2.1，全部「带/不带漂移项」两种）：
  · `sue_roa_ttm_d`  标准化预期外归母 ROA(TTM)-带漂移项
  · `sue_roe_ttm`    标准化预期外归母 ROE(TTM)
  · `sue_np_sq_d`    标准化预期外单季度归母净利润-带漂移项
  · `sue_op_sq`      标准化预期外单季度营业利润

估值（§2.4，只建不依赖一致预期的）：
  · `cfp_inv_chg240`      市现率倒数的 240 日变动
  · `bp_inv_roe_rank`     市净率倒数 + ROE 截面排序之和（研报用 ROE-FY3，此处 ROE-TTM）
  · `peg_hist`            PEG（研报用分析师 FY2 增速，此处用 TTM 净利同比替代）

## SUE 公式（研报 p10 原式 + 学术标准式）

研报给的是
    E(X_q) = X_{q−4} + μ_q ,  μ_q = Σ_{n=1..4}(X_{q−4} − X_{q−n−4}) / 4
    SUE_q  = (X_q − E(X_q)) / σ
其中 μ 的写法（以 q−4 为终点、往前 1~4 期）与 Bernard-Thomas 的 drifting SUE
（μ = 过去 4 个**年度差**的均值）不一致——研报这条公式疑为笔误：按字面，
n=1..4 取的是 q−5..q−8，与「漂移」的经济含义（盈利季节性的趋势）对不上。

本实现取**学术标准式**：
    d_q = X_q − X_{q−4}
    μ_q = mean(d_{q−1}, …, d_{q−4})
    SUE_q = (d_q − μ_q) / σ ,  σ = std(d，过去 8 个 report_period，min 4)
σ 用 8 期而非研报的 4 期，是为了与库内 `sue_q` 口径一致；4 期只够算出一个极不稳的
标准差。不带漂移项的变体（`sue_roe_ttm` / `sue_op_sq`）令 μ=0。

## 用法
    python -m scripts.builders.build_replica_factors            # 全量并落盘
    python -m scripts.builders.build_replica_factors --only sue # 只建 SUE
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "report_replica" / "panels"

SUE_MIN_OBS = 4        # σ 最少样本（report_period 期数）
SUE_WIN = 8            # σ 回望期数


# --------------------------------------------------------------------- 工具
def _pit32(df, cal_idx, codes, field) -> pd.DataFrame:
    from scripts.builders.build_alla_constructed_factors import pit
    return pit(df, cal_idx, codes, field).astype(np.float32)


def _sue(df: pd.DataFrame, field: str, drift: bool) -> pd.DataFrame:
    """按 report_period 序列算 d = X_q − X_{q−4}，再带漂移标准化。"""
    from scripts.builders.build_alla_constructed_factors import year_offset

    df = df.sort_values(["code", "report_period"]).copy()
    if field not in df.columns or df[field].isna().all():
        df["_sue"] = np.nan
        return df
    df = year_offset(df, field, "delta")          # → _off_<field> = d_q
    d = "_off_" + field
    g = df.groupby("code")[d]
    if drift:
        mu = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    else:
        mu = 0.0
    sd = g.transform(lambda s: s.rolling(SUE_WIN, min_periods=SUE_MIN_OBS).std())
    df["_sue"] = (df[d] - mu) / sd.replace(0.0, np.nan)
    return df


# --------------------------------------------------------------------- 主建
def build(only: str | None = None) -> dict[str, pd.DataFrame]:
    from scripts.builders.build_alla_fundamental_factors import (
        add_single_quarter,
        add_ttm_yoy,
        load_panels,
    )

    t0 = time.time()
    close_adj, close_raw, inc, bal, cfo, equ, div = load_panels()
    codes, cal_idx = close_adj.columns, close_adj.index
    close = close_raw.astype(float).reindex(columns=codes)
    print(f"数据装载 {time.time() - t0:.1f}s · 日历 {len(cal_idx)} 日 · {len(codes)} 股")
    out: dict[str, pd.DataFrame] = {}

    # ---- income：TTM 与单季 ----
    inc = add_ttm_yoy(inc, "NET_PRO_EXCL_MIN_INT_INC", "NPE_TTM", None)   # 归母
    inc = add_ttm_yoy(inc, "OPERA_PROFIT", "OP_TTM", None)
    inc = add_single_quarter(inc, "NET_PRO_EXCL_MIN_INT_INC", "NPE_SQ")
    inc = add_single_quarter(inc, "OPERA_PROFIT", "OP_SQ")

    # 总资产/净资产按 report_period 并到 income 上，好让 ROA/ROE 在报告期维度算
    bs = bal[["code", "report_period", "TOTAL_ASSETS",
              "TOT_SHARE_EQUITY_EXCL_MIN_INT", "TOT_SHARE"]].copy()
    bs = bs.sort_values(["code", "report_period"]).drop_duplicates(
        subset=["code", "report_period"], keep="last")
    inc = inc.merge(bs, on=["code", "report_period"], how="left", suffixes=("", "_bs"))
    inc["ROA_TTM"] = inc["NPE_TTM"] / inc["TOTAL_ASSETS"].replace(0.0, np.nan)
    inc["ROE_TTM"] = inc["NPE_TTM"] / inc["TOT_SHARE_EQUITY_EXCL_MIN_INT"].replace(0.0, np.nan)

    if only in (None, "sue"):
        for name, field, drift in (
            ("sue_roa_ttm_d", "ROA_TTM", True),
            ("sue_roe_ttm", "ROE_TTM", False),
            ("sue_np_sq_d", "NPE_SQ", True),
            ("sue_op_sq", "OP_SQ", False),
        ):
            t = time.time()
            d = _sue(inc, field, drift)
            out[name] = _pit32(d, cal_idx, codes, "_sue")
            print(f"  {name:16s} 中位有效 {out[name].notna().sum(axis=1).median():.0f} 只 "
                  f"({time.time() - t:.1f}s)")

    if only in (None, "val"):
        # 总市值（总股本 × 未复权收盘）
        cap = _pit32(bal, cal_idx, codes, "TOT_SHARE").astype(float) * close
        # ① 市现率倒数 = 经营现金流 TTM / 总市值，取 240 日变动
        cfo2 = add_ttm_yoy(cfo, "NET_CASH_FLOWS_OPERA_ACT", "OCF_TTM", None)
        ocf = _pit32(cfo2, cal_idx, codes, "OCF_TTM").astype(float)
        cfp_inv = (ocf / cap.replace(0.0, np.nan)).astype(np.float32)
        out["cfp_inv_chg240"] = (cfp_inv - cfp_inv.shift(240)).astype(np.float32)
        # ② 市净率倒数 + ROE 排序之和（截面 pct rank，行业聚合前先在全市场排）
        bp_inv = ((_pit32(bal, cal_idx, codes, "TOT_SHARE_EQUITY_EXCL_MIN_INT")
                   / cap.replace(0.0, np.nan)).astype(np.float32))
        roe = _pit32(inc, cal_idx, codes, "ROE_TTM")
        out["bp_inv_roe_rank"] = (bp_inv.rank(axis=1, pct=True)
                                  + roe.rank(axis=1, pct=True)).astype(np.float32)
        # ③ PEG：PE_TTM / 净利 TTM 同比（%）。研报用分析师 FY2 增速，此处替代
        inc2 = add_ttm_yoy(inc, "NET_PRO_EXCL_MIN_INT_INC", "NPE_TTM", "NPE_YOY")
        yoy = _pit32(inc2, cal_idx, codes, "NPE_YOY").astype(float)
        pe = cap / _pit32(inc, cal_idx, codes, "NPE_TTM").astype(float).replace(0.0, np.nan)
        out["peg_hist"] = (pe / (yoy * 100).replace(0.0, np.nan)).astype(np.float32)
        for k in ("cfp_inv_chg240", "bp_inv_roe_rank", "peg_hist"):
            print(f"  {k:16s} 中位有效 {out[k].notna().sum(axis=1).median():.0f} 只")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=[None, "sue", "val"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = build(args.only)
    for k, v in out.items():
        p = OUT / f"{k}.parquet"
        v.to_parquet(p)
        print(f"→ {p}  {v.shape}")


if __name__ == "__main__":
    main()
