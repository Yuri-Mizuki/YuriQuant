"""
国君研报复现：因子回测评估与研报基准对比
==========================================

对 ``mine_factors --gp-gtja`` 挖出的因子做统一口径回测（与适应度同口径）：

- 组合：Top/Bottom 10% 等权、日频调仓、次日 VWAP 执行链收益、双边千三费用；
- 指标：多空年化收益 / 最大回撤 / 年化夏普（费后）、多头超额年化及其夏普、
  RankIC 均值、IC 胜率、年化双边换手；
- 样本内 20220101–20221231，样本外为研报同款窗口 20230103–20230227；
- 因子方向按样本内多空收益符号归一，样本外不重定向（防泄漏）。

对比基准（研报《遗传规划解构与投资思考》2023-08-23）：
    前5因子样本内平均：多空年化 22.90%、回撤 -5.14%、夏普 2.33、
    多头超额年化 25.59%、RankIC 4.91%、IC 胜率 57.10%、双边换手 42.56；
    最优因子：37.00% / -3.67% / 3.42 / 换手 38.04；
    样本外（2023 两个月）基准组前因子组合夏普 3.83。

用法:
    python scripts/gtja_repro_eval.py --result reports/gtja_repro/baseline.csv
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("gtja_repro_eval")

TOP_FRAC = 0.1
FEE_RT = 0.003          # 双边千三
IS_RANGE = (20220101, 20221231)
OOS_RANGE = (20230103, 20230227)
# 研报基准（样本内 top5 平均 / 最优因子；样本外为基准组 top 组合夏普）
REPORT_IS_TOP5 = {"ls_ann": 0.2290, "ls_dd": -0.0514, "ls_sharpe": 2.33,
                  "long_excess_ann": 0.2559, "rankic": 0.0491,
                  "ic_win": 0.5710, "turnover": 42.56}
REPORT_IS_BEST = {"ls_ann": 0.3700, "ls_dd": -0.0367, "ls_sharpe": 3.42, "turnover": 38.04}
REPORT_OOS_SHARPE = 3.83

SIX = ["open", "high", "low", "close", "volume", "vwap"]


def build_eval_env(begin: int, end: int, bwd: pd.DataFrame | None = None):
    """面板 + VWAP 执行链收益 + 可交易性掩码（与适应度同口径，离线缓存）。

    ``bwd`` 复权因子只取一次并复用，保证样本内/外口径一致（SDK 登录抖动时
    不会出现一段复权一段未复权的混合链）。
    """
    from config import Config
    from data.cache_helpers import build_panel, build_tradable_mask
    from scripts.mine_factors import _build_vwap_exec_returns

    cfg = Config.get()
    cfg["universe"]["default"] = "all_a"
    panel, _ = build_panel(cfg, begin, end, offline=True)
    # 执行链收益需要 amount/volume/close，先于终端裁剪构建
    rets = _build_vwap_exec_returns(panel, bwd=bwd)
    if "vwap" not in panel and "amount" in panel:
        panel = dict(panel)
        panel["vwap"] = panel["amount"] / panel["volume"]
    bwd_al = (bwd.reindex(index=panel["close"].index, columns=panel["close"].columns)
              .ffill() if bwd is not None and len(bwd) else None)
    mask = build_tradable_mask(panel["close"], bwd=bwd_al)
    panel = {k: panel[k] for k in SIX if k in panel}
    return panel, rets, mask


def load_backward_once(codes=None):
    """取一次后复权因子：优先本地 parquet（本次已全量拉取），避免 SDK 登录抖动。"""
    from pathlib import Path
    import pandas as pd
    p = Path("e:/data/parquet/backward_factor.parquet")
    if p.exists():
        try:
            bwd = pd.read_parquet(p)
            return bwd.reindex(columns=codes) if codes else bwd
        except Exception as exc:
            log.warning("本地复权因子读取失败: %s", exc)
    from data.cache import DataCache
    from data.cache_helpers import load_backward_factor
    from data.datasource import create_datasource
    return load_backward_factor(DataCache(create_datasource()), codes)


def ls_metrics(fp: pd.DataFrame, rets: pd.DataFrame,
               tradable: pd.DataFrame | None = None) -> dict:
    """多空组合费后指标 + RankIC/IC胜率/多头超额/双边换手。方向假定已归一。

    与适应度同口径：剔除因子覆盖 < 截面最大覆盖 50% 的退化日（防止几乎
    全 NaN 的树用个别股票的极端多空制造假夏普），并报告平均覆盖率。
    """
    from factor.genetic_mining import _ls_net_stats
    from research.factor_analysis import calc_ic_series

    valid0 = fp.notna() & rets.notna()
    if len(valid0) == 0 or valid0.notna().sum().sum() == 0:
        return {k: float("nan") for k in
                ("ls_ann", "ls_dd", "ls_sharpe", "long_excess_ann", "long_excess_sharpe",
                 "rankic", "ic_win", "turnover", "n_days", "coverage")}
    n_col = valid0.sum(axis=1)
    # 与适应度同口径：覆盖 < 面板全宽 50% 的退化日剔除
    keep = n_col >= max(10, int(np.ceil(0.5 * fp.shape[1])))
    coverage = float(n_col[keep].mean() / max(fp.shape[1], 1)) if keep.any() else 0.0
    fp = fp[keep]
    rets = rets[keep]
    if tradable is not None:
        tr = tradable.reindex(index=fp.index, columns=fp.columns).fillna(True).astype(bool)
        fp = fp.where(tr)
        rets = rets.where(tr)

    st = _ls_net_stats(fp, rets, top_frac=TOP_FRAC, fee_rt=FEE_RT, tradable=tradable)
    valid = fp.notna() & rets.notna()
    ic = calc_ic_series(fp, rets, method="spearman").dropna()
    # 多头超额（Top 组 - 全池等权），费前（与研报多头超额口径一致）
    fr = fp.where(valid).rank(axis=1, pct=True)
    long_m = (fr >= 1.0 - TOP_FRAC) & valid
    rv = rets.where(valid).fillna(0.0)
    nL = long_m.sum(axis=1).clip(lower=1)
    base = rv.sum(axis=1) / valid.sum(axis=1).clip(lower=1)
    ex = (rv * long_m).sum(axis=1) / nL - base
    ex = ex.dropna()
    ex_sharpe = float(ex.mean() / ex.std() * np.sqrt(252)) if len(ex) > 2 and ex.std() > 0 else float("nan")
    # 年化双边换手：每日两腿换手比例之和 × 252
    churn = (~(long_m.shift(fill_value=False)) & long_m).sum(axis=1) / nL
    fr_s = fp.where(valid).rank(axis=1, pct=True)
    short_m = (fr_s <= TOP_FRAC) & valid
    nS = short_m.sum(axis=1).clip(lower=1)
    churn = churn + (~short_m.shift(fill_value=False) & short_m).sum(axis=1) / nS
    turn_ann = float(churn.iloc[1:].mean() * 252)
    return {
        "ls_ann": st["ann_ret"], "ls_dd": -st["max_dd"], "ls_sharpe": st["sharpe"],
        "long_excess_ann": float(ex.mean() * 252) if len(ex) else float("nan"),
        "long_excess_sharpe": ex_sharpe,
        "rankic": float(ic.mean()) if len(ic) else float("nan"),
        "ic_win": float((ic > 0).mean()) if len(ic) else float("nan"),
        "turnover": turn_ann, "n_days": st["n"], "coverage": coverage,
    }


def eval_factors(formulas: list[str], is_env, oos_env) -> pd.DataFrame:
    """逐因子两段评估；方向按样本内多空收益符号归一（样本外不重定向）。"""
    from factor.formula import formula_builder

    is_panel, is_rets, is_mask = is_env
    oos_panel, oos_rets, oos_mask = oos_env
    feats = list(is_panel.keys())
    rows = []
    for f in formulas:
        try:
            build = formula_builder(f, features=feats)
            fp_is = build(is_panel)
            st = _ls_net_stats_sign(fp_is, is_rets)
            if st is None or not np.isfinite(st):
                log.warning("样本内无有效多空收益，跳过: %s", f)
                continue
            sign = 1.0 if st >= 0 else -1.0
            m_is = ls_metrics(fp_is * sign, is_rets, tradable=is_mask)
            fp_oos = build(oos_panel) * sign          # 沿用样本内方向
            fp_oos = fp_oos.reindex(
                index=[d for d in fp_oos.index
                       if OOS_RANGE[0] <= int(d.strftime("%Y%m%d")) <= OOS_RANGE[1]])
            if fp_oos.empty or fp_oos.notna().sum().sum() == 0:
                log.warning("样本外无有效覆盖，跳过: %s", f)
                continue
            r_oos = oos_rets.loc[fp_oos.index]
            m_oos = ls_metrics(fp_oos, r_oos, tradable=oos_mask)
            row = {"formula": f, "sign": int(sign),
                   **{f"is_{k}": v for k, v in m_is.items()},
                   **{f"oos_{k}": v for k, v in m_oos.items()}}
            rows.append(row)
        except Exception as exc:
            log.warning("评估失败 %s: %s", f, exc)
    return pd.DataFrame(rows)


def _ls_net_stats_sign(fp, rets) -> float | None:
    from factor.genetic_mining import _ls_net_stats
    st = _ls_net_stats(fp, rets, top_frac=TOP_FRAC, fee_rt=FEE_RT)
    if st["n"] < 20 or not np.isfinite(st["ann_ret"]):
        return None
    return st["ann_ret"]


def main():
    ap = argparse.ArgumentParser(description="国君研报复现：因子回测评估")
    ap.add_argument("--result", default="reports/gtja_repro/baseline.csv")
    ap.add_argument("--top", type=int, default=10, help="评估前 N 个因子（默认10）")
    ap.add_argument("--composite", type=int, default=10,
                    help="等权复合因子使用前 N 个因子（研报实践：10 个等权）")
    ap.add_argument("--out", default="reports/gtja_repro/eval.csv")
    args = ap.parse_args()

    res = pd.read_csv(args.result)
    formulas = res["formula"].head(args.top).tolist()
    log.info("载入 %d 个候选（共 %d），评估前 %d", len(res), len(res), len(formulas))

    log.info("构建样本内环境 %s ...", IS_RANGE)
    bwd = load_backward_once()
    is_env = build_eval_env(*IS_RANGE, bwd=bwd)
    log.info("构建样本外环境（2022 全年 + 2023Q1 作深窗口预热）...")
    oos_env = build_eval_env(20220101, 20230331, bwd=bwd)

    df = eval_factors(formulas, is_env, oos_env)

    # 等权复合因子（前 composite 个，zscore 后等权合成）
    if len(df) >= 2 and args.composite >= 2:
        log.info("构建前 %d 因子等权复合 ...", min(args.composite, len(df)))
        from factor.formula import formula_builder
        is_panel, is_rets, is_mask_c = is_env
        oos_panel, oos_rets, oos_mask_c = oos_env
        feats = list(is_panel.keys())
        comps_is, comps_oos = [], []
        for f in df["formula"].head(args.composite):
            try:
                build = formula_builder(f, features=feats)
                fp_i = build(is_panel)
                sign = float(df.loc[df["formula"] == f, "sign"].iloc[0])
                comps_is.append(fp_i * sign)
                comps_oos.append(build(oos_panel) * sign)
            except Exception:
                continue

        def _zs(d):
            m = d.mean(axis=1)
            s = d.std(axis=1)
            return d.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)

        comp_is = pd.concat([_zs(c) for c in comps_is]).groupby(level=0).mean()
        comp_oos = pd.concat([_zs(c) for c in comps_oos]).groupby(level=0).mean()
        comp_oos = comp_oos.reindex(
            index=[d for d in comp_oos.index
                   if OOS_RANGE[0] <= int(d.strftime("%Y%m%d")) <= OOS_RANGE[1]])
        if comp_oos.empty or comp_oos.notna().sum().sum() == 0:
            log.warning("等权复合样本外无有效覆盖，跳过复合评估")
        else:
            m_is = ls_metrics(comp_is, is_rets, tradable=is_mask_c)
            m_oos = ls_metrics(comp_oos, oos_rets.loc[comp_oos.index], tradable=oos_mask_c)
            df.loc[len(df)] = {"formula": f"EQW_TOP{args.composite}", "sign": 1,
                               **{f"is_{k}": v for k, v in m_is.items()},
                               **{f"oos_{k}": v for k, v in m_oos.items()}}

    df.to_csv(args.out, index=False)
    log.info("已保存 %s", args.out)

    # ---- 与研报基准对比 ----
    show = ["formula", "sign", "is_ls_ann", "is_ls_dd", "is_ls_sharpe",
            "is_long_excess_ann", "is_rankic", "is_ic_win", "is_turnover",
            "is_coverage", "oos_ls_sharpe", "oos_rankic", "oos_coverage"]
    with pd.option_context("display.width", 240, "display.float_format",
                           lambda v: f"{v:.4f}"):
        print("\n===== 逐因子（前 %d）=====" % len(df))
        print(df[show].to_string(index=False))

        top5 = df.head(5)
        print("\n===== 与研报基准对比 =====")
        comp = pd.DataFrame({
            "研报top5平均": [REPORT_IS_TOP5["ls_ann"], REPORT_IS_TOP5["ls_dd"],
                          REPORT_IS_TOP5["ls_sharpe"], REPORT_IS_TOP5["long_excess_ann"],
                          REPORT_IS_TOP5["rankic"], REPORT_IS_TOP5["ic_win"],
                          REPORT_IS_TOP5["turnover"]],
            "本次top5平均": [top5["is_ls_ann"].mean(), top5["is_ls_dd"].mean(),
                          top5["is_ls_sharpe"].mean(), top5["is_long_excess_ann"].mean(),
                          top5["is_rankic"].mean(), top5["is_ic_win"].mean(),
                          top5["is_turnover"].mean()],
            "研报最优因子": [REPORT_IS_BEST["ls_ann"], REPORT_IS_BEST["ls_dd"],
                          REPORT_IS_BEST["ls_sharpe"], np.nan,
                          np.nan, np.nan, REPORT_IS_BEST["turnover"]],
            "本次最优因子": [df["is_ls_ann"].max(), df.loc[df["is_ls_ann"].idxmax(), "is_ls_dd"],
                          df["is_ls_sharpe"].max(), df["is_long_excess_ann"].max(),
                          df["is_rankic"].max(), df["is_ic_win"].max(),
                          df["is_turnover"].min()],
        }, index=["多空年化收益", "最大回撤", "多空夏普(费后)", "多头超额年化",
                  "RankIC均值", "IC胜率", "双边换手(年化)"])
        print(comp.to_string())
        compo = df[df["formula"].str.startswith("EQW")]
        if len(compo):
            print(f"\n样本外（2023 两个月窗口）：等权复合 Sharpe = "
                  f"{compo['oos_ls_sharpe'].iloc[0]:.2f}（研报基准组合 3.83）；"
                  f"因子级 OOS Sharpe 中位数 = {df['oos_ls_sharpe'].median():.2f}")


if __name__ == "__main__":
    main()
