"""EV 分母价值因子 + 研发加回盈利因子（华泰 AI95 两项数据改造，P0 增量）。

背景（2026-11-12）：读 `20250811-华泰证券-以空间换时间_多目标基本面选股因子挖掘框架`，
它 71 个输入指标里有两类改造是我们现成池缺的、且数据都在本地：

1. **企业价值 EV 作价值因子分母**。现有 `bp`/`ep_ttm`/`sp_ttm` 全用总市值，
   而分子（归母净资产 / 净利润 / 营收）同时服务于股东与债权人 ⇒ 分子分母错配。
   高杠杆公司被系统性低估价值。
2. **研发加回盈利**。研发费用被一次性费用化，压制当期利润；加回后得到
   "假设研发资本化"的盈利能力，对高研发公司是另一套排序。

只产出 4 个面板，作为**候选增量**，不直接并库——先过 `ev_rd_validate.py` 的
三段式 + 增量检验，通过才入 registry。

口径（与 `build_alla_fundamental_factors` 严格同源）：
- 市值 ``cap = TOT_SHARE × close_raw``（未复权收盘 × 期末总股本，元）
- ``EV = cap + 总负债 + 少数股东权益 − 现金``
  其中 ``总负债 = TOTAL_LIAB_SHARE_EQUITY − TOT_SHARE_EQUITY_INCL_MIN_INT``
  （报表无 TOTAL_LIAB 字段，用"负债和股东权益合计 − 股东权益合计(含少数)"还原）
  无优先股字段（`OTH_EQUITY_TOOLS` 是其他权益工具，含永续债，不并入）
- 现金：优先 ``END_BAL_CASH_CASH_EQU``（覆盖 91.5%），缺失回落 ``CASH_END_BAL``（50.0%）。
  ⚠️ 是**现金及现金等价物**，非"货币资金"（后者不含短期理财）；方向上等价。
- ``bp_ev = 股东权益合计(含少数) / EV`` —— 与 EV 同为"全部资本提供者"口径
- ``ep_ev = 净利润(含少数) TTM / EV``
- ``sp_ev = 营业收入 TTM / EV``
- ``ep_rd = (净利润(含少数) TTM + 研发费用年化 × (1−税率)) / cap``
  研发是税前费用 ⇒ 加回净利润须乘 (1−税率)；税率 = INCOME_TAX/TOTAL_PROFIT，
  clip(0, 0.6)，缺省 0.25（与 `roic_ttm` 同一处理）。

研发口径的选择（实测驱动）：
`RD_EXP_TTM` 只有 21.4% 覆盖（2018 前为 0，2020+ 也仅 46%）——因为 A 股季报
多数不披露研发，四期连续才能算 TTM。改用**年化**：
``RD_ann = RD_EXP × 4 / 报告期季度序号``，覆盖率跟着 `RD_EXP` 走（2020+ 87~92%）。
代价是季节性偏差（研发多在下半年），见报告 §边界。RD 缺失**保持 NaN**，
不填 0——2018 年前不是"没有研发"而是"未单列"。

用法:
    python -m scripts.builders.build_alla_ev_rd_factors
    python -m scripts.builders.build_alla_ev_rd_factors --diag
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

from config import Config  # noqa: E402
from data.financials import build_pit_panel  # noqa: E402
from scripts.builders.common import load_close_adj  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.common.fundamental_common import add_ttm_yoy  # noqa: E402

log = setup_logging("build_ev_rd")

OUT = ROOT / "reports" / "factor_ev_rd" / "panels"
KEEP_FROM = "2018-01-01"

#: 比值因子的物理裁剪界（与 build_alla_fundamental_factors.CLIP_RATIO 一致）
CLIP_RATIO = 1e4


def _pit(df: pd.DataFrame, field: str, cal: pd.DatetimeIndex,
         codes: pd.Index) -> pd.DataFrame:
    if field not in df.columns:
        return pd.DataFrame(np.nan, index=cal, columns=codes)
    return (build_pit_panel(df, cal, field)
            .reindex(index=cal, columns=codes).astype(np.float64))


def _annualize_rd(inc: pd.DataFrame) -> pd.DataFrame:
    """研发费用年化：RD_EXP 是年初至今累计值 ⇒ ×4/季度序号。

    只在当期 RD_EXP 非空时成立；不做跨期填充（缺了就 NaN）。
    """
    if "RD_EXP" not in inc.columns:
        return inc
    q = inc["report_period"].dt.quarter.astype(float)
    inc["RD_ANN"] = inc["RD_EXP"] * 4.0 / q
    return inc


def build(diag: bool = False) -> dict[str, pd.DataFrame]:
    cache = Path(str(Config.cache()["root"]))
    inc = pd.read_parquet(cache / "income.parquet")
    bal = pd.read_parquet(cache / "balance_sheet.parquet")
    cf = pd.read_parquet(cache / "cash_flow.parquet")
    close_adj, close_raw = load_close_adj(with_raw=True)

    # ---- 长表加 TTM（与生产线同一 helper，保证口径一致）----
    inc = add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", None)
    inc = add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", None)
    inc = _annualize_rd(inc)

    # ---- 现金：两列取先可用者 ----
    cf = cf.copy()
    for c in ("END_BAL_CASH_CASH_EQU", "CASH_END_BAL"):
        if c in cf.columns:
            cf[c] = pd.to_numeric(cf[c], errors="coerce")
    if {"END_BAL_CASH_CASH_EQU", "CASH_END_BAL"} <= set(cf.columns):
        cf["CASH_USE"] = cf["END_BAL_CASH_CASH_EQU"].fillna(cf["CASH_END_BAL"])
    elif "END_BAL_CASH_CASH_EQU" in cf.columns:
        cf["CASH_USE"] = cf["END_BAL_CASH_CASH_EQU"]
    else:
        cf["CASH_USE"] = np.nan
        log.warning("现金流表无现金余额字段 ⇒ EV 不退现金")

    cal = close_raw.index
    codes = close_raw.columns

    pit = {}
    for f in ("TOTAL_LIAB_SHARE_EQUITY", "TOT_SHARE_EQUITY_INCL_MIN_INT",
              "TOT_SHARE"):
        pit[f] = _pit(bal, f, cal, codes)
    for f in ("NET_PRO_TTM", "OPERA_REV_TTM", "RD_ANN",
              "INCOME_TAX", "TOTAL_PROFIT"):
        pit[f] = _pit(inc, f, cal, codes)
    pit["CASH"] = _pit(cf, "CASH_USE", cal, codes)

    cap = (pit["TOT_SHARE"] * close_raw.astype(float)).astype(np.float64)
    equity_all = pit["TOT_SHARE_EQUITY_INCL_MIN_INT"]
    liab_total = pit["TOTAL_LIAB_SHARE_EQUITY"] - equity_all
    ev = cap + liab_total.fillna(0.0) - pit["CASH"].fillna(0.0)

    tax = (pit["INCOME_TAX"] / pit["TOTAL_PROFIT"].replace(0.0, np.nan)
           ).clip(0.0, 0.6).fillna(0.25)

    def _ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
        r = num / den.replace(0.0, np.nan)
        return r.where(np.isfinite(r)).clip(-CLIP_RATIO, CLIP_RATIO)

    panels = {
        "bp_ev": _ratio(equity_all, ev),
        "ep_ev": _ratio(pit["NET_PRO_TTM"], ev),
        "sp_ev": _ratio(pit["OPERA_REV_TTM"], ev),
        "ep_rd": _ratio(pit["NET_PRO_TTM"] + pit["RD_ANN"] * (1.0 - tax), cap),
        # 诊断用（不入候选）：研发强度本身。用来判断 ep_rd 的增量是否只等价于它。
        "rd_to_cap": _ratio(pit["RD_ANN"], cap),
    }

    if diag:
        _diagnose(panels, ev, cap, liab_total, equity_all, pit)
    return panels


def _diagnose(panels, ev, cap, liab_total, equity_all, pit) -> None:
    """诊断：覆盖率、EV 合理性、与母因子的截面相关。只打印，不改结果。"""
    print("\n=== 覆盖率（按年，日频面板均值）===")
    cov = pd.DataFrame({k: v.notna().mean(axis=1) for k, v in panels.items()})
    print(cov.groupby(cov.index.year).mean().loc[2018:].round(3).to_string())

    print("\n=== EV / 市值 分布（衡量'错配修正'力度）===")
    lev = (ev / cap.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    lev = lev.where(lev > 0)
    print(lev.stack().describe(percentiles=[.01, .1, .5, .9, .99]).round(3).to_string())
    print(f"EV<cap 占比（现金 > 负债，即净现金公司）: {float((lev < 1).mean().mean()):.3f}")

    print("\n=== 金融/地产剔除诊断：liab/equity > 5 的样本占比 ===")
    g = (liab_total / equity_all.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    print(f"每日截面占比中位: {float((g > 5).mean(axis=1).median()):.3f}")

    print("\n=== 与母因子的截面秩相关（同一日、同一批股票，中位）===")
    ds = Path(str(Config.get()["factor_library"]["root"])) / "all_a_2018_2026" / "panels"
    pairs = {"bp_ev": "bp", "ep_ev": "ep_ttm", "sp_ev": "sp_ttm", "ep_rd": "ep_ttm"}
    for new, old in pairs.items():
        p = ds / f"{old}.parquet"
        if not p.exists():
            print(f"  {new:6s} vs {old:8s}: 母因子面板缺失")
            continue
        old_p = pd.read_parquet(p).reindex(index=panels[new].index,
                                           columns=panels[new].columns)
        rhos = []
        for d in panels[new].index[::10]:          # 抽稀：每 10 日一个截面
            a = panels[new].loc[d]
            b = old_p.loc[d]
            m = a.notna() & b.notna()
            if m.sum() >= 100:
                rhos.append(a[m].corr(b[m], method="spearman"))
        s = pd.Series(rhos)
        print(f"  {new:6s} vs {old:8s}: ρ 中位 {s.median():+.3f} "
              f"(p10 {s.quantile(.1):+.3f} / p90 {s.quantile(.9):+.3f}, n={len(s)})")

    print("\n=== 研发年化口径校验（RD_ANN / 年化营收 中位，应落在 0.03~0.06）===")
    r = (pit["RD_ANN"] / pit["OPERA_REV_TTM"].replace(0.0, np.nan))
    r = r.where(np.isfinite(r) & (r > 0) & (r < 1))
    print(f"全部 {float(r.stack().median()):.4f} | 2021+ "
          f"{float(r.loc['2021':].stack().median()):.4f}")


def save(panels: dict[str, pd.DataFrame]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in panels.items():
        d = v.loc[v.index >= pd.Timestamp(KEEP_FROM)]
        # 全 NaN 列（退市/未上市）直接砍，避免中性化退化成假面板
        d = d.loc[:, d.notna().any(axis=0)]
        path = OUT / f"{k}.parquet"
        d.astype(np.float32).to_parquet(path)
        print(f"  saved {path.name:12s} {d.shape} "
              f"{d.index.min().date()}~{d.index.max().date()} "
              f"覆盖 {float(d.notna().mean().mean()):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="打印覆盖率/分布/母因子相关")
    args = ap.parse_args()

    panels = build(diag=args.diag)
    save(panels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
