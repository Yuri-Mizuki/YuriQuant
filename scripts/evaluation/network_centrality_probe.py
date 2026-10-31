"""跨股票网络中心度：P1 探针（算子空间造不出的那一类，2026-10-27）。

## 为什么这是真缺口

`factor/operators.py` 的 80 个可调用名里，`ts_*` 是逐股票时序、`cs_*` 是同一天
截面逐元素 —— **没有任何「跨股票配对」原语**：没有 `corr(i,j)`、没有邻接矩阵、
没有特征分解。所以「网络中心度」在现有算子空间里**任何组合都造不出来**，
这是归纳偏置的缺口，不是数据的缺口（查重报告 §5）。

## 口径

- **图**：申万一级行业内部。每年 1 月按**上一年末市值**取前 `--top-n`（默认 40）
  只建完全图 ⇒ 年频换成分，用年末市值决定次年成分，无前视。
- **边**：过去 60 个交易日的日收益相关（窗口内精确标准化，不是长期波动率近似）。
- **中心度**：`mean_j corr(i, j)`，j ≠ i。高 = 与同行高度同步 = 特质信息少。
- **覆盖**：约 31 行业 × 40 只 ≈ 1200 只 / 5800 ⇒ ~20%。**这是设计约束不是缺陷**：
  全 5801×5801 相关矩阵做不了，行业前 N 是业界标准的最小可行口径。

## 已知局限（必须在报告里写明）

1. 只有行业内边，**没有跨行业边** —— 「产业链/供应链网络」这条完全没覆盖。
2. 年频换成分会让成分变更日附近出现跳变（因子值不连续）。
3. 完全图 + 等权，没有做「显著性阈值 / 稀疏化」——真实网络通常只保留显著边。
4. 与 Barra 的行业因子、`cov_vol` 可能共线（都是「同类股票一起动」），
   所以**必须看中性化后**的结果。

## 用法

    python -m scripts.evaluation.network_centrality_probe build --neutralize
    python -m scripts.evaluation.residual_ic_probe \
        --ds-root reports/network_probe_2026-10-27/all_a_2018_2026 \
        --panels-dir panels_neu --out reports/network_probe_2026-10-27 --out-tag neu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("network_centrality_probe")

OUT_ROOT = ROOT / "reports" / "network_probe_2026-10-27"
DATASET = "all_a_2018_2026"
BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"


def _rolling_cent(R: np.ndarray, win: int) -> np.ndarray:
    """滚动相关矩阵的行均值（剔除自身）。

    Args:
        R: (T, m) 日收益矩阵，NaN = 停牌/未上市。
        win: 窗口长度。

    Returns:
        (T, m) 中心度，前 ``win-1`` 行为 NaN。

    实现要点：相关必须**逐窗口标准化**（每个窗口用自己的均值/方差），
    不能对每只股票各跑一次 ``rolling``——那样每个 t 用的窗口不同，
    拼出来的不是同一个相关矩阵（常见错误，会把因子变成慢变的噪声）。
    """
    T, m = R.shape
    out = np.full((T, m), np.nan, dtype=np.float64)
    if T < win or m < 3:
        return out
    idx = np.arange(win)[None, :] + np.arange(T - win + 1)[:, None]
    W = R[idx]                                    # (T-win+1, win, m)
    with warnings.catch_warnings():
        # 全 NaN 列（当年新上市 / 整窗停牌）会触发 "Mean of empty slice"，
        # 结果确实是 NaN，属预期，不必刷屏。
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(W, axis=1, keepdims=True)
        # 🚨 ddof 必须是 0：相关系数 = Σ z_i z_j / n，若 z 用 ddof=1 的样本标准差
        # 标准化，结果会整体缩 (n-1)/n 倍（对角线 0.983 而不是 1）。
        # 2026-10-27 实测：ddof=1 时对角 0.983333，与 pandas corr 差 1.7e-2。
        sd = np.nanstd(W, axis=1, ddof=0, keepdims=True)
    Z = (W - mu) / np.where(sd > 0, sd, np.nan)
    ok = np.isfinite(Z)                           # (T-win+1, win, m)
    # 缺测按「贡献 0」处理，但对角线用**逐对同时有效天数**做分母，
    # 否则停牌会系统性低估相关（除以 win 时把缺失当成 0 收益）。
    Z0 = np.where(ok, Z, 0.0)
    num = np.einsum("twi,twj->tij", Z0, Z0)
    cnt = np.einsum("twi,twj->tij", ok.astype(np.float64), ok.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = num / cnt
    # 逐对有效数不足窗口 60% ⇒ 该边无效；单只票整体无效时它的所有边都无效
    bad = cnt < win * 0.6
    C[bad] = np.nan
    d = np.arange(m)
    C[:, d, d] = np.nan                           # 剔除自身
    with np.errstate(invalid="ignore"):
        cent = np.nanmean(C, axis=2)              # (T-win+1, m)
    out[win - 1:] = cent
    return out


def build_centrality(base: dict, top_n: int, win: int) -> pd.DataFrame:
    """按「年 × 行业」分块算中心度，填回全 A 面板。"""
    close = base["close"]
    mktcap = base["mktcap"]
    ind = pd.read_parquet(BASE_DIR / "industry.parquet").reindex(
        index=close.index, columns=close.columns)
    ret = close.pct_change(fill_method=None)

    panel = pd.DataFrame(np.nan, index=close.index, columns=close.columns,
                         dtype=np.float32)
    dates = close.index
    years = sorted({d.year for d in dates})
    log.info("年份 %s | 行业面板覆盖 %.3f", years,
             float(ind.notna().to_numpy(dtype=float).mean()))

    n_cell = 0
    for y in years:
        # 上年末市值决定本年成分（12-31 之前最后一个有值的交易日）
        prev_cap = mktcap.loc[:f"{y - 1}-12-31"]
        if prev_cap.empty:
            log.info("%d 年跳过（无上年末市值）", y)
            continue
        cap0 = prev_cap.ffill().iloc[-1]
        prev_ind = ind.loc[:f"{y - 1}-12-31"]
        ind0 = prev_ind.ffill().iloc[-1] if not prev_ind.empty else None
        if ind0 is None:
            continue
        # 本年 + 前置 win 个交易日（让 1 月就有值）
        i0 = int(dates.searchsorted(pd.Timestamp(f"{y}-01-01")))
        i1 = int(dates.searchsorted(pd.Timestamp(f"{y + 1}-01-01")))
        lo = max(0, i0 - win)
        sl = dates[lo:i1]
        pos = np.arange(len(sl))
        # 本年内的行，且 cent 已过暖机（cent 第 k 行 = sl 第 k+win-1 行）
        take = pos[(pos >= win - 1) & (pos >= i0 - lo)]
        if take.size == 0:
            continue

        grp = ind0.dropna()
        for _name, codes in grp.groupby(grp).groups.items():
            codes = [c for c in codes if pd.notna(cap0.get(c))]
            if len(codes) < 3:
                continue
            codes = sorted(codes, key=lambda c: -float(cap0[c]))[:top_n]
            sub = ret.loc[sl, codes].to_numpy(dtype=np.float64)
            cent = _rolling_cent(sub, win)
            block = cent[take - (win - 1)]
            panel.loc[sl[take], codes] = block.astype(np.float32)
            n_cell += int(np.isfinite(block).sum())
        log.info("%d 年完成（累计有效格 %d）", y, n_cell)

    cov = float(panel.notna().to_numpy(dtype=float).mean())
    log.info("net_cent_%d 覆盖 %.4f | 截面中位 %.4f", win, cov,
             float(np.nanmedian(panel.to_numpy())))
    return panel


def neutralized(pnl: pd.DataFrame, base: dict) -> pd.DataFrame:
    from factor.preprocessing import neutralize_batch

    ind = pd.read_parquet(BASE_DIR / "industry.parquet")
    cov = {k: pd.read_parquet(BASE_DIR / f"cov_{k}.parquet")
           for k in ("mom", "vol", "turn")}
    r = neutralize_batch(pnl, market_cap_panel=base["mktcap"],
                         industry_panel=ind, extra_covariates=cov)
    return r.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-n", type=int, default=40, help="行业内取市值前 N 只")
    ap.add_argument("--win", type=int, default=60, help="相关窗口（交易日）")
    ap.add_argument("--neutralize", action="store_true")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)

    from research.factor_library import FactorLibrary
    from scripts.common.alla_common import load_base

    out_root = Path(args.out_root)
    t0 = time.time()
    base = load_base()
    base["mktcap"] = base["market_cap"].astype(np.float64)

    name = f"net_cent_{args.win}"
    pnl = build_centrality(base, args.top_n, args.win)
    dest = out_root / DATASET
    (dest / "panels").mkdir(parents=True, exist_ok=True)
    pnl.to_parquet(dest / "panels" / f"{name}.parquet")

    if args.neutralize:
        neu = neutralized(pnl, base)
        (dest / "panels_neu").mkdir(parents=True, exist_ok=True)
        neu.to_parquet(dest / "panels_neu" / f"{name}.parquet")
        log.info("中性化覆盖 %.4f", float(neu.notna().to_numpy(dtype=float).mean()))

    lib = FactorLibrary(root=out_root, dataset=DATASET)
    st = lib.upsert_rows(
        [{"name": name, "set": "behav_net",
          "label": f"行业网络中心度 top{args.top_n} win{args.win}（P1 探针）"}],
        source=f"behav:{name}:2026-10-27", family="行为金融", kind="raw",
        maturity="experimental", frequency="日频", set_name="behav_net")
    log.info("registry: %s", st)

    (out_root / "manifest.json").write_text(json.dumps({
        "name": name, "top_n": args.top_n, "win": args.win,
        "coverage": float(pnl.notna().to_numpy(dtype=float).mean()),
        "elapsed_s": round(time.time() - t0, 1),
        "caveat": "仅行业内边（申万一级）；年频换成分；完全图等权未稀疏化；"
                  "与 Barra 行业/波动暴露可能共线，必须看中性化后结果。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成 %.0fs -> %s", time.time() - t0, dest / "panels" / f"{name}.parquet")


if __name__ == "__main__":
    main()
