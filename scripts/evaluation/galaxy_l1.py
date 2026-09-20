"""银河 0608 复现 L1：三标签 GBDT + 验证集 RankIC 质量门槛（stage2 Phase 2 前置）。

研读依据：``reports/docs/research_notes/银河0608_研读_时序截面三层预测与QP口径.md``；
分层定稿见 RESEARCH_TODO（L1 = Phase 2 信号设计本身）。

做什么：经典 12 因子 + 银河特征族 ~36 个 → 三标签（alpha_22 / sharpe_22 /
mdd_66）→ 按研报口径滚动训练（3 年窗、前 80% 训练后 20% 验证、年更）→
报告**验证集**日均 RankIC（研报的质量门槛参照：>0.05 可用；参照数字
Sharpe 0.0679 / MaxDD 0.1792）与**测试年**日均 RankIC（防"门槛自证"）。

判定逻辑（写进产出 report.md）：
- mdd_66 的验证集 RankIC 达到研报量级（≥0.10）→「风险标签可测」成立，
  stage2 的 z 信号升级为 MaxDD 预测；
- sharpe_22 ≥0.05 → 收益侧标签可用；
- alpha_22 显著低于 sharpe → 印证「收益难测」与回退机制必要性。

复现边界：日线 mult=1（研报小时线）；基准=成分等权代理（项目无指数权重
面板）；GBDT 替代三层网络（L2 后置，触发条件见 TODO）。**L1 结果不与
研报数字直接对标排序，只对标量级与相对结构**（风险>Sharpe>Alpha）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("galaxy_l1")

#: 研报参照（HS300、小时线、7 个滚动模型均值）
REPORT_REF = {"sharpe": 0.0679, "mdd": 0.1792, "gate": 0.05}


def build_dataset(begin: str, end: str) -> pd.DataFrame:
    """特征宽表 + 三标签的行式样本（仅成分内、标签非缺失行）。"""
    from factor.classic import (
        build_galaxy_labels,
        compute_classic_features,
        compute_galaxy_features,
    )
    from scripts.factors.run_portfolio_ppo import load_panels

    px, mask = load_panels(begin, end)
    codes = sorted(set(mask.columns) & set(px["close"].columns))
    mask = mask.reindex(columns=codes).fillna(False)

    # 等权基准代理：成分内逐日收益均值 → 累积成"指数收盘"
    member = mask.reindex(px["close"].index).fillna(False)
    rets = px["close"][codes].pct_change(fill_method=None)
    mkt_ret = rets.where(member).mean(axis=1).fillna(0.0)
    idx_close = (1.0 + mkt_ret).cumprod()

    feats = compute_classic_features(px)
    feats.update(compute_galaxy_features(px, idx_close))
    labels = build_galaxy_labels(px, idx_close)

    df = pd.concat(
        {**{f"f_{k}": v.stack() for k, v in feats.items()},
         **{k: v.stack() for k, v in labels.items()}},
        axis=1,
    )
    df = df[member.stack().reindex(df.index).fillna(False)]
    keep = df[[c for c in labels]].notna().all(axis=1)
    df = df[keep]
    df.index.names = ["date", "code"]
    log.info("样本 %d 行 × %d 特征（%s ~ %s）",
             len(df), len([c for c in df.columns if c.startswith("f_")]),
             df.index.get_level_values(0).min().date(),
             df.index.get_level_values(0).max().date())
    return df


def daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    """逐日截面 Spearman RankIC（每日 ≥15 只有效才计入）。"""
    df = pd.concat({"p": pred, "y": label}, axis=1).dropna()
    def _ic(g):
        return g["p"].corr(g["y"], method="spearman") if len(g) >= 15 else np.nan
    return df.groupby(level=0).apply(_ic).dropna()


def run_one_year(df: pd.DataFrame, year: int, seed: int) -> dict:
    """研报口径单年模型：3 年窗、前 80% 训练后 20% 验证；测试=第 4 年。"""
    import lightgbm as lgb
    from stats.robust_stats import nw_tstat

    dates = df.index.get_level_values(0)
    d0, d1 = dates.min(), dates.max()
    train_end = pd.Timestamp(f"{year}-12-31")
    valid_start = train_end - pd.offsets.BYearEnd(0) - pd.Timedelta(days=0)
    # 验证集 = 训练窗最后 20%（研报：约等于第 3 年）
    win = df[(dates >= pd.Timestamp(f"{year - 3}-01-01")) & (dates <= train_end)]
    if win.empty:
        return {}
    udates = win.index.get_level_values(0).unique().sort_values()
    n_train = int(len(udates) * 0.8)
    tr_dates, va_dates = udates[:n_train], udates[n_train:]
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    label_cols = [c for c in df.columns if c.startswith("label_")]

    tr = win[win.index.get_level_values(0).isin(tr_dates)]
    va = win[win.index.get_level_values(0).isin(va_dates)]
    te = df[(dates > train_end) & (dates <= pd.Timestamp(f"{year + 1}-12-31"))]
    if tr.empty or va.empty or te.empty:
        return {}

    out: dict = {}
    for lab in label_cols:
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            verbose=-1)
        model.fit(tr[feature_cols], tr[lab],
                  eval_set=[(va[feature_cols], va[lab])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        for split, frame in (("valid", va), ("test", te)):
            pred = pd.Series(model.predict(frame[feature_cols]),
                             index=frame.index)
            ic = daily_rank_ic(pred, frame[lab])
            t_nw, _se, _lag = nw_tstat(ic.to_numpy(dtype=float))
            out[f"{lab}|{split}"] = {
                "rank_ic": float(ic.mean()), "t_nw": float(t_nw),
                "n_days": int(len(ic))}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--begin", default="2020-10-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--years", default="2024,2025,2026")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "galaxy_l1"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.begin, args.end)
    rows: dict[tuple, dict] = {}
    for year in [int(y) for y in args.years.split(",")]:
        log.info("=== 模型年 %d（训练 %d-%d，80/20 划分）===", year, year - 3, year)
        for key, m in run_one_year(df, year, args.seed).items():
            lab, split = key.split("|")
            rows[(year, lab, split)] = m

    res = pd.DataFrame(rows).T
    res.index.names = ["model_year", "label", "split"]
    res.to_csv(out_dir / "summary.csv")

    lines = ["# 银河 0608 复现 L1：三标签 GBDT 的 RankIC", "",
             "| 模型年 | 标签 | split | RankIC | NW-t | 天数 |", "|---|---|---|---|---|---|"]
    for (y, lab, split), m in rows.items():
        lines.append(f"| {y} | {lab} | {split} | {m['rank_ic']:.4f} "
                     f"| {m['t_nw']:.1f} | {m['n_days']} |")

    # 汇总判定（验证集口径 vs 研报参照）
    lines += ["", "## 判定（验证集均值 vs 研报参照）", ""]
    verdict = {}
    for lab in sorted({k[1] for k in rows}):
        v = [m["rank_ic"] for (y, l, s), m in rows.items() if l == lab and s == "valid"]
        if v:
            mean = float(np.mean(v))
            ref = REPORT_REF.get("sharpe" if "sharpe" in lab else
                                 "mdd" if "mdd" in lab else "alpha", None)
            verdict[lab] = mean
            lines.append(f"- `{lab}`：验证集均值 **{mean:.4f}**"
                         + (f"（研报参照 {ref}，门槛 {REPORT_REF['gate']}）" if ref else ""))
    lines += ["", f"- 风险标签可测判定：mdd 验证集均值 ≥0.10 → 成立；"
              f"当前 = {verdict.get('label_mdd_66', float('nan')):.4f}",
              "- 复现边界：日线 mult=1、等权基准代理、GBDT 替代三层网络（见 TODO）。"]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("已写出 %s", out_dir)
    print("\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
