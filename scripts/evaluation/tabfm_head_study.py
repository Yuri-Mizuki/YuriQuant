"""批次 8 组合层传导研究：IC 高的模型为何组合层不占优（头部诊断）。

输入 = tabfm_rolling --save-preds 落盘的逐臂 OOS 预测面板
（reports/tabfm_rolling/<ds>/pred_<arm>.parquet）。全部为事后算术（无重训）：

1. **头部浓缩度**：逐日 top-decile 名单的前瞻收益相对池均值的超额（年化口径）
   ——直接回答"全截面 IC 的优势是否集中在头部"；
2. **头部质量 vs 全截面 IC**：全截面 IC（RankIC）与 top-decile 内部 IC
   （头部内排序是否还有区分度）并排；
3. **头部稳定性**：top-decile 成员的 1 日留存率（换手成本侧写）；
4. **跨臂头部重叠**：臂间 top-decile Jaccard（头部信息是否同源）；
5. **头部合成**：三臂（gbdt/tabpfn/tabicl）秩平均后的头部浓缩度 vs 单臂。

用法:
    python -m scripts.evaluation.tabfm_head_study \
        --pred-dir reports/tabfm_rolling/hs300_2022_2025_tb2025-01
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


def _ns(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).as_unit("ns")


def _load_close_fwd() -> pd.DataFrame:
    """h=1 前瞻收益（close 口径，与 tabfm eval 一致）。"""
    from config import Config
    from data.cache_helpers import build_panel
    panel, _ = build_panel(Config.get(), int(Config.discipline()["begin"]),
                           20261231, offline=True)
    close = panel["close"]
    close.index = _ns(close.index)
    return close.shift(-1) / close - 1.0


def _head_stats(pred: pd.DataFrame, fwd: pd.DataFrame, frac: float = 0.10) -> dict:
    """逐日头部统计 → 汇总。"""
    common = pred.index.intersection(fwd.index)
    p, f = pred.loc[common], fwd.loc[common]
    n_head = max(1, int(p.shape[1] * frac))
    head_spreads, head_ics, full_ics, stay1 = [], [], [], []
    prev_head = None
    for d in common:
        s = p.loc[d]
        r = f.loc[d]
        ok = s.notna() & r.notna()
        if ok.sum() < 50:
            prev_head = None
            continue
        sv, rv = s[ok], r[ok]
        k = min(n_head, int(ok.sum() * frac)) or 1
        top = sv.nlargest(k).index
        head_spreads.append(float(rv.loc[top].mean() - rv.mean()))
        head_ics.append(float(sv.loc[top].corr(rv.loc[top], method="spearman")))
        full_ics.append(float(sv.corr(rv, method="spearman")))
        if prev_head is not None and len(prev_head):
            stay1.append(len(top.intersection(prev_head)) / len(top))
        prev_head = top
    ann = 244.0
    return {
        "head_spread_ann": float(np.nanmean(head_spreads) * ann),
        "head_ic": float(np.nanmean(head_ics)),
        "full_ic": float(np.nanmean(full_ics)),
        "head_stay1": float(np.nanmean(stay1)) if stay1 else float("nan"),
        "n_days": len(head_spreads),
    }


def _head_overlap(a: pd.DataFrame, b: pd.DataFrame, frac: float = 0.10) -> float:
    common = a.index.intersection(b.index)
    js = []
    for d in common:
        sa, sb = a.loc[d], b.loc[d]
        k = max(1, int(sa.notna().sum() * frac))
        ta = set(sa.dropna().nlargest(k).index)
        tb = set(sb.dropna().nlargest(k).index)
        if ta and tb:
            js.append(len(ta & tb) / len(ta | tb))
    return float(np.nanmean(js)) if js else float("nan")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/tabfm_rolling/hs300_2022_2025_tb2025-01")
    ap.add_argument("--frac", type=float, default=0.10)
    args = ap.parse_args(argv)

    pred_dir = Path(args.pred_dir)
    out = pred_dir / "head_study"
    out.mkdir(parents=True, exist_ok=True)
    fwd = _load_close_fwd()
    preds = {}
    for f in sorted(pred_dir.glob("pred_*.parquet")):
        arm = f.stem[5:]
        df = pd.read_parquet(f)
        df.index = _ns(df.index)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        preds[arm] = df
    if len(preds) < 2:
        raise SystemExit(f"pred 面板不足: {list(preds)}")

    rows = []
    for arm, p in preds.items():
        st = _head_stats(p, fwd, args.frac)
        st["arm"] = arm
        rows.append(st)
        print(f"[{arm}] 头部超额(年化)={st['head_spread_ann'] * 100:+.2f}pp "
              f"头部IC={st['head_ic']:.4f} 全截面IC={st['full_ic']:.4f} "
              f"头部1日留存={st['head_stay1']:.2f}", flush=True)

    # 跨臂头部重叠
    arms = list(preds)
    ov = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            j = _head_overlap(preds[a], preds[b], args.frac)
            ov[f"{a}|{b}"] = round(j, 3)
            print(f"头部Jaccard {a}|{b}: {j:.3f}", flush=True)

    # 头部合成：全部臂秩平均后再测头部
    rank_mean = 0.0
    for p in preds.values():
        rank_mean = rank_mean + p.rank(axis=1, pct=True)
    ens = rank_mean / len(preds)
    st = _head_stats(ens, fwd, args.frac)
    st["arm"] = "ENSEMBLE_rankavg"
    rows.append(st)
    print(f"[ENSEMBLE] 头部超额(年化)={st['head_spread_ann'] * 100:+.2f}pp "
          f"头部IC={st['head_ic']:.4f} 全截面IC={st['full_ic']:.4f}", flush=True)

    pd.DataFrame(rows).set_index("arm").to_csv(out / "head_stats.csv",
                                               encoding="utf-8-sig")
    pd.Series(ov).to_csv(out / "head_overlap.csv", encoding="utf-8-sig")
    print(f"已写出 {out}", flush=True)


if __name__ == "__main__":
    main()
