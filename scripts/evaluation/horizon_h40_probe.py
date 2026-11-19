"""h=40 新周期腿的对照探针（2026-11-15）。

问的是：把主模型 horizon 从 20 拉到 40，能不能带来**新信息**。
结论（见 ``reports/horizon_h40/README.md``）：不能，h40 是 h20 的平滑副本。

三层证据，本脚本一次跑完：

1. **标签可比性**：h40 对自身 40 日标签 IC vs 对**同一 20 日标签** IC
   ——若优势在换尺子后消失，说明涨幅来自标签平滑而非选股能力。
2. **信号层**（统计功效最高）：91 个月度截面 rank IC + 配对 NW t + 十分位曲线。
   信号层若无差，组合层的差就是选股抖动。
3. **组合层**：T+1 开盘、buffer30、K ∈ {15,25,45} 的月均超额与配对 t。

用法::

    python -m scripts.evaluation.horizon_h40_probe
    python -m scripts.evaluation.horizon_h40_probe --start 2019-01-02 --end 2026-09-07
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

#: 主臂（h1/5/10/20）与 h40 补充臂的 pred 目录
BASE_ARM = "reports/alla_rolling_ortho920fundind/pred"
H40_ARM = "reports/alla_rolling_ortho920fundind_h40/pred"
BASE_DIR = "reports/alla_rolling/_base"

#: 合成信号的候选臂：key = 展示名，value = 按 [h1,h5,h10,h20,h40] 的权重
ARMS: dict[str, list[float]] = {
    "mid13 2:1:1:2": [2, 1, 1, 2, 0],
    "h10:h20 1:1": [0, 0, 1, 1, 0],
    "h20 单腿": [0, 0, 0, 1, 0],
    "h40 单腿": [0, 0, 0, 0, 1],
    "h20:h40 1:1": [0, 0, 0, 1, 1],
    "h10:h20:h40": [0, 0, 1, 1, 1],
}
HORIZONS = [1, 5, 10, 20, 40]


def _nw(x: np.ndarray, lag: int = 3) -> tuple[float, float]:
    """Newey-West 均值与 t（Bartlett 核）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    s = (e * e).sum() / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).sum() / n
    return float(x.mean()), float(x.mean() / np.sqrt(s / n))


def load_panels(start: str, end: str):
    """读五条腿的 OOS 预测，取共同有效掩码后转截面 rank。"""
    close = pd.read_parquet(f"{BASE_DIR}/close_adj.parquet")
    mask = pd.read_parquet(f"{BASE_DIR}/tradable_mask.parquet")
    cols = close.columns
    idx = pd.read_parquet(f"{BASE_ARM}/gbdt__h1.parquet").index
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]

    ranked: dict[int, pd.DataFrame] = {}
    valid = None
    for h in HORIZONS:
        path = f"{BASE_ARM if h < 40 else H40_ARM}/gbdt__h{h}.parquet"
        p = pd.read_parquet(path).reindex(index=idx, columns=cols)
        v = p.notna() & mask.reindex(index=idx, columns=cols).fillna(False)
        ranked[h] = p.where(v).rank(axis=1, pct=True)
        valid = v if valid is None else (valid & v)
    ranked = {h: s.where(valid) for h, s in ranked.items()}
    return ranked, valid, idx, close


def monthly_pairs(idx: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """月末最后交易日为信号日的相邻期配对（位置索引）。"""
    s = pd.Series(idx, index=idx)
    rebal = s.groupby([idx.year, idx.month]).last().values
    pos = {d: i for i, d in enumerate(idx)}
    ri = [pos[d] for d in rebal]
    return [(a, b) for a, b in zip(ri[:-1], ri[1:]) if b + 1 < len(idx)]


def hold_returns(px: pd.DataFrame, idx: pd.DatetimeIndex, pairs, sig: pd.DataFrame,
                 k: int, buf: int) -> np.ndarray:
    """BufferedTopK 的逐期超额（%，T+1 开盘进 → 下期 T+1 开盘出）。"""
    ex, hold = [], None
    for a, b in pairs:
        r = (px.loc[idx[b + 1]] / px.loc[idx[a + 1]] - 1).replace([np.inf, -np.inf], np.nan)
        s0 = sig.loc[idx[a]]
        ok = s0.notna() & r.notna()
        s2 = s0.where(ok)
        rk = s2.rank(ascending=False)
        keep = [c for c in (hold or []) if rk.get(c, 1e9) <= buf]
        add = [c for c in s2.nlargest(k + buf).index if c not in keep][: k - len(keep)]
        pk = keep + add
        ex.append((r[pk].mean() - r.where(ok).mean()) * 100)
        hold = pk
    return np.array(ex)


def main() -> None:
    ap = argparse.ArgumentParser(description="h=40 新周期腿对照探针")
    ap.add_argument("--start", default="2019-01-02")
    ap.add_argument("--end", default="2026-09-07")
    args = ap.parse_args()

    ranked, valid, idx, close = load_panels(args.start, args.end)
    op = pd.read_parquet(f"{BASE_DIR}/open_adj.parquet")
    pairs = monthly_pairs(idx)
    print(f"有效日 {int(valid.any(axis=1).sum())}  日均在册 {int(valid.sum(axis=1).mean())}  "
          f"调仓期 {len(pairs)}")

    # ---- ① 标签可比性 ------------------------------------------------
    print("\n=== ① 标签可比性（IC 随尺子变化）===")
    for lbl in (20, 40):
        ret = close.pct_change(lbl).shift(-lbl).reindex(index=idx, columns=close.columns)
        line = []
        for h in (20, 40):
            p = ranked[h]
            ok = p.notna() & ret.notna()
            ic = p.where(ok).rank(axis=1).corrwith(
                ret.where(ok).rank(axis=1), axis=1).dropna()
            line.append(f"pred h{h} IC {ic.mean():.4f} (IR {ic.mean() / ic.std():.2f})")
        print(f"  {lbl:2d} 日标签: " + "   ".join(line))

    print("\n  逐日预测截面 rank 相关：")
    for a, b in [(20, 40), (20, 10), (10, 40), (1, 40), (1, 20)]:
        m = ranked[a].notna() & ranked[b].notna()
        c = ranked[a].where(m).rank(axis=1).corrwith(
            ranked[b].where(m).rank(axis=1), axis=1).dropna()
        print(f"    h{a:2d} vs h{b:2d}  mean {c.mean():.4f}  p5 {c.quantile(.05):.4f}")

    # ---- 信号合成 ----------------------------------------------------
    def comp(w: list[float]) -> pd.DataFrame:
        w = np.asarray(w, float)
        num = None
        for h, wt in zip(HORIZONS, w):
            if not wt:
                continue
            num = ranked[h] * wt if num is None else num + ranked[h] * wt
        return num / w.sum()

    sigs = {nm: comp(w) for nm, w in ARMS.items()}

    # ---- ② 信号层：月度截面 IC ---------------------------------------
    print("\n=== ② 信号层：月度截面 rank IC（配对 vs mid13）===")
    ics: dict[str, pd.Series] = {}
    for nm, sg in sigs.items():
        v = []
        for a, b in pairs:
            r = (op.loc[idx[b + 1]] / op.loc[idx[a + 1]] - 1).replace([np.inf, -np.inf], np.nan)
            s0 = sg.loc[idx[a]]
            ok = s0.notna() & r.notna()
            v.append(s0.where(ok).corr(r.where(ok), method="spearman"))
        ics[nm] = pd.Series(v, index=[idx[a] for a, b in pairs]).dropna()
        m, t = _nw(ics[nm])
        print(f"  {nm:14s} IC {m:.4f}  t {t:+.2f}")
    base = ics["mid13 2:1:1:2"]
    for nm in ARMS:
        if nm.startswith("mid13"):
            continue
        m, t = _nw((ics[nm] - base).dropna())
        print(f"    Δ {nm:14s} {m:+.4f}  t {t:+.2f}")

    print("\n=== ② 十分位（月均超额 %，D1 最好）===")
    for nm in ("mid13 2:1:1:2", "h20 单腿", "h40 单腿"):
        sg = sigs[nm]
        dec = {i: [] for i in range(10)}
        for a, b in pairs:
            r = (op.loc[idx[b + 1]] / op.loc[idx[a + 1]] - 1).replace([np.inf, -np.inf], np.nan)
            s0 = sg.loc[idx[a]]
            ok = s0.notna() & r.notna()
            s2, rr = s0.where(ok), r.where(ok)
            q = pd.qcut(s2.rank(ascending=False), 10, labels=False)
            for i in range(10):
                m_ = q == i
                dec[i].append((rr[m_].mean() - rr.mean()) * 100)
        print(f"  {nm:14s} " + " ".join(f"D{i+1}{np.mean(dec[i]):+5.2f}" for i in range(10)))

    # ---- ③ 组合层 ----------------------------------------------------
    print("\n=== ③ 组合层：T+1 开盘 + buffer30，月均超额 %（NW t）===")
    exs: dict[tuple[str, int], np.ndarray] = {}
    for nm, sg in sigs.items():
        cells = []
        for k in (15, 25, 45):
            e = hold_returns(op, idx, pairs, sg, k, 30)
            exs[(nm, k)] = e
            cells.append(f"K{k} {e.mean():+.3f}(t{_nw(e)[1]:+.2f})")
        print(f"  {nm:14s} " + "  ".join(cells))
    print("\n  配对 vs mid13：")
    for nm in ARMS:
        if nm.startswith("mid13"):
            continue
        d = exs[(nm, 15)] - exs[("mid13 2:1:1:2", 15)]
        print(f"    {nm:14s} Δ {d.mean():+.3f}pp  t {_nw(d)[1]:+.2f}")


if __name__ == "__main__":
    main()
