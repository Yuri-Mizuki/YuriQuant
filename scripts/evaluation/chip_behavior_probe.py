"""筹码 / 处置效应 / 凸显理论：现有算子造廉价近似，先证残差有没有空间（2026-10-27）。

## 定位

`QuantsPlaybook` 查重（`reports/docs/consistency_checks/QuantsPlaybook_59篇研报_
与我们因子池查重.md` §5）指出「族 4 行为金融（分布型）」是 950 因子池的**结构缺口** ——
不是缺数据（日频量价本来就有），是 `factor/operators.py` 的算子空间里**没有
状态/分布型原语**：29 个 `ts_*` 全是无状态的滚动统计，造不出「按价格分档累积
持仓量」这种对象。

本脚本**不建因子族**，只做 P0 探针：用现有算子造 3 个廉价近似，交给
`residual_ic_probe.one_factor` 判「主信号之外还有没有增量」。理由见查重报告 §7 ——
920 入池因子里 DPP 选到第 50 席还在抢量价，`ic_headroom` 最优线性合成上界
0.0785、候选扩到 150 反掉到 0.0598（多样性过头稀释）。**不先证残差有空间就
开工建族，是 10-10「补数据不自动兑现」同一个坑。**

## 三个近似（全部日频量价，零新数据）

| 家族 | 研报原意 | 廉价近似 | 预期方向 |
|---|---|---|---|
| 处置效应 | 浮盈 → 落袋卖压 | `close_adj / cost_ref - 1` | 负 |
| 筹码分布 | 按价格分档累积换手衰减的持仓量 | `(avg_cost - close) / std_cost`（换手加权矩） | 正 |
| 凸显理论 | 极端收益被过度加权 | `ts_sum((cs_rank(dev)-0.5) * ret1, N) / N` | 负 |

三个都**不是**研报原算法：
- 筹码分布的「分档」被退化成矩：原意是价格 × 持仓量的直方图，这里只取重心与
  离散度。分档信息（双峰、长尾）全丢。
- STR 的 Kahneman 权重函数被退化成「截面凸显度去均值」，只保留「异常波动日
  获得额外权重」这一条。
- 处置效应的「持仓成本」用 VWAP 滚动均值代理，真实成本取决于每个投资者的
  建仓时点，无法观测。

细节口径：
- 价格全部用 `_base` 的**复权**面板（`close_adj` / `vwap_adj`）——`vwap_adj`
  是 `(amount/volume) × 后复权因子`，与 `close_adj` 同源，不许混用未复权价
  （未复权面板造假跳空，~4pp/年）。
- 换手率 = `volume / shares`，`shares = market_cap / close_raw`（同 `_base`
  口径反推，float32 往返误差 ~1e-7）；换手率本身不受复权影响。
- 滚动和一律做**有效日归一**（`ts_sum(x) / ts_sum(notna)`），停牌不会让整窗
  变 NaN。无缺失时与 `ts_mean` 逐位一致。

## 用法

    # 造面板（约 2~3 分钟）
    python -m scripts.evaluation.chip_behavior_probe build

    # 造 + 中性化（三因子 + 五风格，约 5~8 分钟）
    python -m scripts.evaluation.chip_behavior_probe build --neutralize

    # 然后交给探针（两组口径各跑一次）
    python -m scripts.evaluation.residual_ic_probe \
        --ds-root reports/chip_behavior_probe_2026-10-27 --panels-dir panels
    python -m scripts.evaluation.residual_ic_probe \
        --ds-root reports/chip_behavior_probe_2026-10-27 --panels-dir panels_neu \
        --out-tag neu
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

log = setup_logging("chip_behavior_probe")

#: 探针产物根（`reports/` 根只允许 5 项，故落子目录）
OUT_ROOT = ROOT / "reports" / "chip_behavior_probe_2026-10-27"
DATASET = "all_a_2018_2026"
BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"

#: 三个族的窗口网格。扫窗口是为了避免「窗口选错就封线」的误杀，但**同一段 OOS
#: 上选窗口本身就是数据窥探**，报告里必须标注；正式立项须样本外复验。
GRID: dict[str, tuple[int, ...]] = {"de": (20, 60, 120),
                                    "chip": (60, 120, 250),
                                    "str": (20, 60)}


# ---------------------------------------------------------------- 基础面板

def load_base() -> dict[str, pd.DataFrame]:
    """复权价 / VWAP / 换手率 / 市值 / 掩码（全部来自 `_base`，不重跑 prep）。"""
    from scripts.common.alla_common import load_base as _lb

    base = _lb()
    close = base["close"].astype(np.float64)
    vwap = pd.read_parquet(BASE_DIR / "vwap_adj.parquet").astype(np.float64)
    close_raw = pd.read_parquet(BASE_DIR / "close_raw.parquet").astype(np.float64)
    mktcap = pd.read_parquet(BASE_DIR / "market_cap.parquet").astype(np.float64)

    from config import Config

    # 股本 = 市值 / 未复权收盘（与 rolling_grid_alla.stage_prep 同一恒等式）
    shares = mktcap / close_raw.where(close_raw > 0)
    # 成交量：一次 pivot（1100 万行 → 2500 日 × 5800 股）
    daily = pd.read_parquet(Path(str(Config.cache()["root"]))
                            / "daily_all_a.parquet", columns=["volume"])
    daily.index = daily.index.set_levels(
        daily.index.levels[0].normalize(), level=0)
    vol = daily["volume"].unstack("code").reindex(
        index=close.index, columns=close.columns)

    turn = (vol / shares).replace([np.inf, -np.inf], np.nan)
    log.info("基础面板: %d 日 × %d 股 | vwap 覆盖 %.3f | 换手覆盖 %.3f",
             len(close), close.shape[1],
             float(vwap.notna().to_numpy(dtype=float).mean()),
             float(turn.notna().to_numpy(dtype=float).mean()))
    return {"close": close, "vwap": vwap, "turn": turn,
            "mask": base["mask"].astype(bool), "mktcap": mktcap}


def _wmean(x: pd.DataFrame, w: pd.DataFrame, n: int) -> pd.DataFrame:
    """换手/有效性加权的 N 日滚动均值（有效日归一，停牌不阻断整窗）。

    与 ``operators.ts_mean`` 在无缺失时逐位一致；缺失时按 ``w`` 归一，
    不会因单日 NaN 把整窗打成 NaN。
    """
    from factor.operators import safe_div, ts_sum

    w = w.fillna(0.0).clip(lower=0.0)
    num = ts_sum((x * w).fillna(0.0), n)
    den = ts_sum(w, n)
    return safe_div(num, den).where(den > 0)


# ---------------------------------------------------------------- 三个族

def f_de(base: dict, n: int) -> pd.DataFrame:
    """处置效应：现价相对 N 日成交量加权成本的高度。

    正 = 浮盈（处置效应下卖压大 → 预期负收益）。用 VWAP 而非收盘价做成本，
    是因为收盘价是单点价格，VWAP 才接近当日成交者的平均建仓成本。
    """
    from factor.operators import safe_div

    close, vwap = base["close"], base["vwap"]
    cost = _wmean(vwap, vwap.notna().astype(float), n)
    return safe_div(close, cost) - 1.0


def f_chip(base: dict, n: int) -> pd.DataFrame:
    """筹码一阶矩：换手加权成本重心相对现价的**标准化**偏离。

    分子是「成本重心 - 现价」（正 = 浮亏），分母是筹码分布的离散度
    （换手加权标准差）⇒ 因子 = 偏离了几个筹码标准差。用换手加权而非等权，
    是因为筹码分布的历史成交量按换手衰减，近期活跃的价格档权重更高。
    """
    from factor.operators import safe_div

    close, vwap, turn = base["close"], base["vwap"], base["turn"]
    w = turn.where(vwap.notna())
    avg = _wmean(vwap, w, n)
    var = _wmean((vwap - avg) ** 2, w, n)
    disp = np.sqrt(var.clip(lower=0))
    return safe_div(avg - close, disp)


def f_str(base: dict, n: int) -> pd.DataFrame:
    """凸显理论：极端收益日的收益被「过度加权」后的 N 日**日均**。

    `dev = |ret1 - ts_mean(ret1, 20)|` 是当日收益相对自身近期常态的偏离，
    `cs_rank(dev) - 0.5` 把它转成截面凸显权重（去均值 ⇒ 剔除「均匀加权」的
    动量部分，只留凸显溢价）。乘以 ret1 再取 N 日均值。

    🚨 这里**不能**套 :func:`_wmean`：凸显权重含负值、逐日 ``sum(w) ≈ 0``，
    按权重归一（除以 Σw）会爆掉。只除以「有效日数」。
    """
    from factor.operators import cs_rank, safe_div, ts_sum

    close = base["close"]
    ret1 = close.pct_change(fill_method=None)
    rbar = _wmean(ret1, ret1.notna().astype(float), 20)
    dev = (ret1 - rbar).abs()
    w = cs_rank(dev) - 0.5
    num = ts_sum((w * ret1).where(ret1.notna()).fillna(0.0), n)
    cnt = ts_sum(ret1.notna().astype(float), n)
    return safe_div(num, cnt).where(cnt > 0)


BUILDERS = {"de": f_de, "chip": f_chip, "str": f_str}


def build_all(base: dict) -> dict[str, pd.DataFrame]:
    """按 GRID 造全部近似因子面板。"""
    out: dict[str, pd.DataFrame] = {}
    for fam, windows in GRID.items():
        for n in windows:
            name = f"{fam}_{n}"
            t0 = time.time()
            pnl = BUILDERS[fam](base, n)
            pnl = (pnl.replace([np.inf, -np.inf], np.nan)
                   .astype(np.float32))
            cov = float(pnl.notna().to_numpy(dtype=float).mean())
            out[name] = pnl
            log.info("%-14s 覆盖 %.3f | 截面中位 %.4f | %.0fs",
                     name, cov, float(np.nanmedian(pnl.to_numpy())),
                     time.time() - t0)
    return out


# ---------------------------------------------------------------- 落盘与登记

def neutralized(panels: dict[str, pd.DataFrame],
                base: dict) -> dict[str, pd.DataFrame]:
    """五风格中性化（行业 + log 市值 + mom/vol/turn），与生产 `panels_neu` 同口径。"""
    from factor.preprocessing import neutralize_batch

    ind = pd.read_parquet(BASE_DIR / "industry.parquet")
    cov = {k: pd.read_parquet(BASE_DIR / f"cov_{k}.parquet")
           for k in ("mom", "vol", "turn")}
    out: dict[str, pd.DataFrame] = {}
    for name, pnl in panels.items():
        t0 = time.time()
        r = neutralize_batch(pnl, market_cap_panel=base["mktcap"],
                             industry_panel=ind, extra_covariates=cov)
        out[name] = r.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        log.info("%-14s 中性化 -> 覆盖 %.3f (%.0fs)", name,
                 float(out[name].notna().to_numpy(dtype=float).mean()),
                 time.time() - t0)
    return out


def _write(dest: Path, panels: dict[str, pd.DataFrame]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, pnl in panels.items():
        pnl.to_parquet(dest / f"{name}.parquet")


def _register(names: list[str], out_root: Path) -> None:
    """写最小 registry（探针按 name 过滤，需要 set 列做统计分组）。"""
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(root=out_root, dataset=DATASET)
    by_set: dict[str, list[str]] = {}
    for nm in names:
        fam = nm.rsplit("_", 1)[0]
        by_set.setdefault({"de": "behav_de", "chip": "behav_chip",
                           "str": "behav_str"}[fam], []).append(nm)
    for set_name, group in sorted(by_set.items()):
        rows = [{"name": nm, "set": set_name,
                 "label": f"廉价近似 {nm}（P0 探针，未入生产池）"}
                for nm in group]
        st = lib.upsert_rows(rows, source=f"behav:{set_name}:2026-10-27",
                             family="行为金融", kind="raw",
                             maturity="experimental", frequency="日频",
                             set_name=set_name)
        log.info("registry %s: %s", set_name, st)


# ---------------------------------------------------------------- 诊断

def _xcorr(a: pd.DataFrame, b: pd.DataFrame, mask: pd.DataFrame) -> float:
    """两面板逐日截面秩相关的均值（掩码内）。"""
    v = a.notna() & b.notna() & mask
    if int(v.to_numpy(dtype=bool).sum()) < 1000:
        return float("nan")
    ra = a.where(v).rank(axis=1)
    rb = b.where(v).rank(axis=1)
    return float(ra.corrwith(rb, axis=1).mean())


def diagnose(panels: dict[str, pd.DataFrame], base: dict) -> dict:
    """因子间互相关 + 与风格协变量的相关（判「是不是老东西换皮」）。"""
    mask = base["mask"]
    names = list(panels)
    style = {k: pd.read_parquet(BASE_DIR / f"cov_{k}.parquet")
             for k in ("mom", "vol", "turn", "size")}
    style["size"] = np.log(style["size"].where(style["size"] > 0))
    pair: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair[f"{a}~{b}"] = round(_xcorr(panels[a], panels[b], mask), 4)
    vs_style: dict[str, dict] = {}
    for nm, p in panels.items():
        vs_style[nm] = {k: round(_xcorr(p, s, mask), 4)
                        for k, s in style.items()}
        log.info("%-14s vs 风格: %s", nm,
                 ", ".join(f"{k}={v:+.3f}" for k, v in vs_style[nm].items()))
    worst = sorted(pair.items(), key=lambda kv: -abs(kv[1]))[:6]
    log.info("最相关的因子对: %s",
             ", ".join(f"{k}={v:+.3f}" for k, v in worst))
    return {"pairs": pair, "vs_style": vs_style}


# ---------------------------------------------------------------- 入口

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["build", "diag"], help="build=造面板；diag=只诊断")
    ap.add_argument("--neutralize", action="store_true", help="同时产出 panels_neu")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    t0 = time.time()
    base = load_base()

    if args.action == "diag":
        panels = {p.stem: pd.read_parquet(p)
                  for p in sorted((out_root / DATASET / "panels").glob("*.parquet"))}
    else:
        panels = build_all(base)
        _write(out_root / DATASET / "panels", panels)
        _register(list(panels), out_root)
        log.info("raw 面板落盘 -> %s", out_root / DATASET / "panels")
        if args.neutralize:
            neu = neutralized(panels, base)
            _write(out_root / DATASET / "panels_neu", neu)
            log.info("中性化面板落盘 -> %s", out_root / DATASET / "panels_neu")

    diag = diagnose(panels, base)
    (out_root / "diag.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成 %.0fs | 诊断 -> %s", time.time() - t0, out_root / "diag.json")


if __name__ == "__main__":
    main()
