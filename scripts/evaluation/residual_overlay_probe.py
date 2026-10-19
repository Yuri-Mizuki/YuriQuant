"""A2 残差负向因子 → 组合层剔除/低配 overlay 探针（2026-10-19）。

## 定位

`TODO.md` §六 A2 下一步的第二臂。残差 IC 首轮（`residual_ic_probe.py`，
`reports/frontier_audit_2026-10-19/`）发现非量价残差信息集中在**负向**
（融资/两融/SUE/资产增长等 risk/quality 因子，`both` t≤−3 且 9 年方向稳定），
读数是「主模型多头里该回避的股票」。本脚本在**组合层**检验这个读数：
把这些因子做成信号层惩罚（低配/变相剔除），看生产口径净 Sharpe 是否改善。

## 机制（信号层叠加，不动组合形态）

    score' = rank_pct(pred) − w · mean_k rank_pct(因子_k)

- `pred` = 882 ortho `ens_h1h5__h1`（本机工作基线）；组合形态与基线**完全相同**
  （`BufferedTopFracLongOnly(0.15, 0.40)`，月频，生产 profile = 固定10bp +
  T+1 open + 1e6）——唯一差异是信号面板，走同一条引擎路径。
- 负向因子方向：`both` IC 为负 ⇒ 因子值越高越危险 ⇒ penalty = 因子**原值**
  rank（高值 → 高惩罚）。不做残差化（组合层要的是可用信息，不是干净归因）。
- 因子缺失 → 该因子不计入均值；全缺 → penalty 取中性 0.5（不减分不加分）。
- w 语义：rank 空间最大位移。w=0.20 已接近「硬剔除」（把满惩罚股票压出
  Top15% 截止线），w=0.05~0.10 是温和低配。

## 臂设计（可证伪）

| 臂 | 因子集 | w | 回答 |
|---|---|---|---|
| base | — | 0 | 基线（=登记表口径复核） |
| neg9_w005/010/020 | 9 个负向（t≤−3 & 同号≥7/9 & cov≥0.3） | 0.05/0.10/0.20 | 剂量反应 |
| neg6_w010 | 6 强因子（t≤−5） | 0.10 | 因子集敏感性 |
| placebo_w010 | neg9 时间循环平移 250 交易日 | 0.10 | 安慰剂：ΔSharpe 应≈0 |

**安慰剂**：因子面板沿时间轴循环平移 250 日（保留截面结构与自相关、破坏与
当期收益/主信号的对齐）。若 placebo 的 ΔSharpe 与真实臂同量级 ⇒ 改善来自
机制偏差（如覆盖率/小市值暴露）而非信息。

## 已知边界（结论须带上）

1. **因子选择是全期 in-sample 的**（残差 IC 用 2018–2026 全期算），组合回测
   同期 ⇒ overlay 改善有选择层前视风险。缓解：① 方向大多与 literature 一致
   （asset growth / 两融拥挤 / A股 Piotroski 负号是项目已知）；② placebo 臂；
   ③ 逐年超额拆分。920 定版口径复验挂好机器。
2. 882 口径主信号；920 定版 pred 在好机器。

## 用法

    python -m scripts.evaluation.residual_overlay_probe
    python -m scripts.evaluation.residual_overlay_probe --profile fair
    python -m scripts.evaluation.residual_overlay_probe --weights 0.05,0.10,0.20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("residual_overlay_probe")

# 残差 IC 首轮筛出的负向因子（reports/frontier_audit_2026-10-19/
# residual_ic_h1_nonpv_s3.csv，both_nw_t≤−3 & 同号≥7/9 & cov≥0.3，2026-10-19）。
NEG_FACTORS: dict[str, list[str]] = {
    "neg9": [
        "financing_chg_1d", "sue_q", "margin_bal_chg_5d", "asset_growth_yoy",
        "margin_delta_ttm", "securities_chg_1d", "inv_rev_gap",
        "margin_bal_chg_20d", "piotroski_f",
    ],
    "neg6": [
        "financing_chg_1d", "sue_q", "margin_bal_chg_5d", "asset_growth_yoy",
        "margin_delta_ttm", "securities_chg_1d",
    ],
}


def load_panel(fp: Path, idx: pd.DatetimeIndex,
               cols: pd.Index) -> pd.DataFrame | None:
    """读一张因子面板对齐到 (idx, cols)。与 residual_ic_probe._load_panel 同逻辑
    （私有名不跨入口 import，layering 守卫）。"""
    raw = pd.read_parquet(fp)
    if not isinstance(raw.index, pd.DatetimeIndex):
        try:
            raw.index = pd.to_datetime(raw.index)
        except Exception:
            return None
    if raw.columns.has_duplicates:
        raw = raw.loc[:, ~raw.columns.duplicated()]
    raw = raw.select_dtypes(include=[np.number])
    if raw.shape[1] == 0:
        return None
    x = raw.reindex(index=idx, columns=cols)
    return x.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def penalty_mean(names: list[str], pdir: Path, idx: pd.DatetimeIndex,
                 cols: pd.Index, shift: int = 0) -> pd.DataFrame:
    """等权平均因子截面百分位 rank。shift>0 = 时间循环平移（安慰剂）。

    某股票部分因子缺失时用可得因子均值；全缺为 NaN（由调用方 fillna(0.5)）。
    """
    acc: pd.DataFrame | None = None
    cnt: pd.DataFrame | None = None
    for n in names:
        x = load_panel(pdir / f"{n}.parquet", idx, cols)
        if x is None:
            log.warning("因子面板缺失/不可用：%s", n)
            continue
        if shift:
            x = pd.DataFrame(np.roll(x.to_numpy(), shift, axis=0),
                             index=x.index, columns=x.columns)
        r = x.rank(axis=1, pct=True)
        if acc is None:
            acc = r.astype(np.float32).where(r.notna(), np.nan)
            cnt = r.notna().astype(np.float32)
        else:
            acc = acc.add(r.where(r.notna()), fill_value=0.0)
            cnt = cnt.add(r.notna().astype(np.float32), fill_value=0.0)
        del x
    assert acc is not None and cnt is not None
    return (acc / cnt.replace(0, np.nan)).astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="production",
                    help="caliber 口径组合（production / fair）")
    ap.add_argument("--signal", default="ens_h1h5")
    ap.add_argument("--window", default="prod")
    ap.add_argument("--panels-dir", default="panels_neu")
    ap.add_argument("--weights", default="0.05,0.10,0.20")
    ap.add_argument("--placebo-shift", type=int, default=250)
    ap.add_argument("--out", default="reports/frontier_audit_2026-10-19/overlay")
    args = ap.parse_args(argv)

    from scripts.common.alla_common import ds_root
    from scripts.evaluation.caliber import (
        cost_profile,
        evaluate_strategy,
        get_profile,
        load_context,
    )
    from strategy.examples import BufferedTopFracLongOnly

    t0 = time.time()
    prof = get_profile(args.profile)
    ctx = load_context(signal=args.signal, window=args.window,
                       execution=prof.execution)
    cost = cost_profile(prof.cost)
    log.info("口径 %s（cost=%s / exec=%s / capital=%.0e）| 信号 %s | %s ~ %s（%d 日）",
             prof.name, prof.cost, prof.execution, prof.capital, args.signal,
             ctx.begin, ctx.pred.index[-1].date(), ctx.n_days)

    pdir = ds_root() / args.panels_dir
    idx, cols = ctx.pred.index, ctx.pred.columns

    score = ctx.pred.rank(axis=1, pct=True).astype(np.float32)
    pens = {k: penalty_mean(v, pdir, idx, cols).fillna(0.5)
            for k, v in NEG_FACTORS.items()}
    for k, p in pens.items():
        log.info("penalty[%s] 覆盖 %.3f（fill 0.5 占 %.3f）", k,
                 float(p.ne(0.5).to_numpy(dtype=bool).mean()),
                 float((p == 0.5).to_numpy(dtype=float).mean()))
    pens["placebo"] = penalty_mean(NEG_FACTORS["neg9"], pdir, idx, cols,
                                   shift=args.placebo_shift).fillna(0.5)

    arms: list[tuple[str, pd.DataFrame | None]] = [("base", None)]
    ws = [float(x) for x in args.weights.split(",")]
    for w in ws:
        for k in ("neg9", "neg6"):
            arms.append((f"{k}_w{int(round(w*100)):03d}",
                         (score - w * pens[k]).astype(np.float32)))
    w_pl = ws[1] if len(ws) > 1 else ws[0]
    arms.append((f"placebo_w{int(round(w_pl*100)):03d}",
                 (score - w_pl * pens["placebo"]).astype(np.float32)))

    rows = []
    for name, sig in arms:
        t1 = time.time()
        m = evaluate_strategy(BufferedTopFracLongOnly(0.15, 0.40), ctx, cost,
                              prof.capital, signal=sig, yearly=True)
        rows.append({"arm": name, **m, "sec": round(time.time() - t1, 1)})
        log.info("%-16s ann=%.2f%% sharpe=%.3f mdd=%.3f turn=%.1f%% "
                 "cost=%.0fbp n_act=%.0f | %.0fs",
                 name, 100*m["ann"], m["sharpe"], m["max_dd"],
                 100*m["turnover"], m["cost_bp"], m["n_active"], rows[-1]["sec"])

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    base = df[df.arm == "base"].iloc[0]
    ycols = [c for c in df.columns if c.startswith("yearly_ret_")]
    for c in ("ann", "sharpe", "sharpe_arith", "max_dd", *ycols):
        df[f"d_{c.replace('yearly_ret_', 'yr')}"] = df[c] - base[c]
    df.to_csv(dest / "summary.csv", index=False, encoding="utf-8-sig")

    meta = {
        "profile": prof.name, "signal": args.signal, "window": args.window,
        "panels_dir": args.panels_dir, "weights": ws,
        "neg_factors": NEG_FACTORS, "placebo_shift": args.placebo_shift,
        "strategy": "BufferedTopFracLongOnly(0.15, 0.40) rebalance=M",
        "oos": f"{ctx.begin} ~ {ctx.pred.index[-1].date()} ({ctx.n_days} days)",
        "caveat": ("因子集为全期 in-sample 残差 IC 所选（选择层前视风险，见脚本 "
                   "docstring「已知边界」）；主信号 882 ortho，920 复验挂好机器。"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (dest / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    log.info("完成 %.0fs -> %s", time.time() - t0, dest / "summary.csv")
    show = ["arm", "ann", "sharpe", "max_dd", "turnover", "cost_bp",
            "n_active", "d_sharpe"]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
