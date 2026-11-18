"""筹码分布（CYQ）面板：从上游 ``cyq.py`` 的绝对价格网格递推到全市场面板（2026-11-18）。

## 定位

上游 `myhhub/stock` 的 `instock/core/kline/cyq.py`（Apache-2.0，副本见
`scripts/evaluation/_ref/cyq_upstream.py`）用「日内 OHLC 三角分布 + 换手衰减」
近似持仓成本分布，输出获利盘比例 / 成本分位区间 / 集中度。

本脚本不直接调它的实现（单股逐日回溯 210 天，全 A 跑不动 5565 × 2500 次），
而是把同一模型写成**逐日递推**：

    state_t = (1 − turn_t) · state_{t−1} + turn_t · Tri_t

其中 `Tri_t` 是当日成交在 `[low, high]`、峰值 `(O+C+H+L)/4` 的三角分布。
区别只有两处，都是上游的缺陷：

| | 上游 | 本脚本 |
|---|---|---|
| 网格 | 每日按 210 天窗口重算 `[min,max]` | 全历史**绝对对数价格网格**，固定不重建 |
| 记忆 | 210 天硬截断（旧筹码凭空消失） | 递推衰减（旧筹码按换手自然淘汰） |
| 复权 | 未处理，除权日筹码整体错位 | 全程后复权价，收益率连续 |

🚨 **不用「相对成本网格 + 每日平移」**：那样每天都要插值重采样，线性插值的
数值扩散在 210 次后就把分布抹宽 ~5 个档，`benefit` 实测漂移 0.22（不可接受）。
绝对网格下价格自己移动、状态不动，全程零重采样。

## 三个因子

`benefit` = P(cost < 现价) 获利盘比例；`trap` = P(cost > 现价 × 1.1) 深度套牢占比；
`conc90` = (P95 − P5) / (P95 + P5) 成本分布集中度。

前两个是 `factor/operators.py` 造不出的（`ts_*` 全是无状态定窗聚合，
没有「按价格分档累积持仓量」这种带记忆的分布对象）。

## 用法

    # P0 口径对照：上游实现 vs 递推版
    python -m scripts.evaluation.cyq_panel verify

    # 全市场面板（约 5~15 分钟）
    python -m scripts.evaluation.cyq_panel build
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# 🚨 必须在 import numpy 之前：内核是逐元素运算，用不到 BLAS；多进程池下每个子进程
# 默认各开一个满核 BLAS 线程池，实测直接把内存打爆（OpenBLAS "allocation still failed"）。
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("cyq_panel")

BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"
UPSTREAM_PATH = ROOT / "scripts" / "evaluation" / "_ref" / "cyq_upstream.py"
OUT_ROOT = ROOT / "reports" / "cyq_probe_2026-11-18"
DATASET = "all_a_2018_2026"
#: registry 词表内取值（"筹码分布" 不在 FAMILY_WHITELIST；"cyq" 不在 SOURCE_PREFIXES）
FAMILY = "量价"
SOURCE = "constructed:cyq_panel:2026-11-18"

#: 网格档距（对数价格）。0.2% 对 benefit 边界 u=1 的插值误差 ~1e-3 量级。
GRID_STEP = 0.002
GRID_N_MIN, GRID_N_MAX = 64, 2000

#: 上游口径默认值（与 cyq.py 的 __init__ 一致）
UP_ACC, UP_RANGE, UP_DAYS = 150, 120, 210


# ---------------------------------------------------------------- 数据

def load_wide(codes: list[str]) -> dict[str, pd.DataFrame]:
    """宽表：``{open, high, low, close, turn, close_raw}``，index=date × columns=code。

    🚨 **不能对 1167 万行长表一次性 ``unstack``**：pandas 内部 factorize 要再造一份
    等长 int64，全市场会 OOM（实测 89MB 分配失败）。改为按 400 只分块 pivot；
    ``open``/``close`` 直接复用 ``_base`` 的复权宽表，少 pivot 两列。
    """
    from config import Config

    root = Path(str(Config.cache()["root"]))
    bf = pd.read_parquet(root / "backward_factor.parquet")
    mktcap = pd.read_parquet(BASE_DIR / "market_cap.parquet")
    close_raw = pd.read_parquet(BASE_DIR / "close_raw.parquet")
    close_adj = pd.read_parquet(BASE_DIR / "close_adj.parquet")
    open_adj = pd.read_parquet(BASE_DIR / "open_adj.parquet")

    idx = close_raw.index
    keep = [c for c in codes if c in close_raw.columns and c in bf.columns
            and c in close_adj.columns and c in open_adj.columns]
    f = bf[keep].reindex(idx).ffill().bfill()

    daily = pd.read_parquet(root / "daily_all_a.parquet",
                            columns=["high", "low", "volume"])
    daily.index = daily.index.set_levels(
        daily.index.levels[0].normalize(), level=0)
    code_lv = daily.index.get_level_values("code")

    out: dict[str, pd.DataFrame] = {
        "open": open_adj[keep].reindex(idx),
        "close": close_adj[keep].reindex(idx),
    }
    for col in ("high", "low", "volume"):
        out[col] = pd.DataFrame(np.nan, index=idx, columns=keep, dtype=np.float64)

    CH = 400
    for s in range(0, len(keep), CH):
        chunk = keep[s:s + CH]
        sub = daily[code_lv.isin(chunk)]
        for col in ("high", "low", "volume"):
            w = sub[col].unstack("code").reindex(index=idx, columns=chunk)
            out[col].loc[:, chunk] = w.to_numpy()

    for col in ("high", "low"):
        out[col] = out[col] * f
    shares = mktcap[keep].reindex(idx) / close_raw[keep].reindex(idx).where(
        close_raw[keep].reindex(idx) > 0)
    out["turn"] = (out.pop("volume") / shares).clip(0.0, 1.0)
    out["_codes"] = keep
    return out


def load_ohlcv(codes: list[str]) -> dict[str, pd.DataFrame]:
    """逐股 DataFrame（verify 用；股票少，直接切片）。"""
    wide = load_wide(codes)
    keep = wide.pop("_codes")
    out: dict[str, pd.DataFrame] = {}
    for code in keep:
        df = pd.DataFrame({k: wide[k][code] for k in
                           ("open", "high", "low", "close", "turn")})
        df = df[df["close"].notna() & (df["close"] > 0)
                & (df["high"] > 0) & (df["low"] > 0)]
        if len(df):
            out[code] = df
    return out


def _grid_from_price(lo: float, hi: float) -> np.ndarray:
    """对数均匀价格网格（含两端），档距约 ``GRID_STEP``。"""
    lo = max(float(lo) * 0.90, 1e-4)
    hi = float(hi) * 1.10
    span = np.log(hi / lo)
    if span <= 0:
        span = GRID_STEP * GRID_N_MIN
    n = int(np.clip(np.ceil(span / GRID_STEP), GRID_N_MIN, GRID_N_MAX))
    return np.exp(np.linspace(np.log(lo), np.log(hi), n + 1))


def _tri_area(a: float, m: float, b: float, edges: np.ndarray) -> np.ndarray:
    """三角分布在网格 bin 上的面积（总面积 = 1）。

    a/m/b = low / 峰 / high 的绝对价格；峰用上游同一个 ``(O+C+H+L)/4``。
    """
    area = np.zeros(len(edges) - 1)
    span = b - a
    if span <= 1e-12:  # 一字板：退化冲击
        k = int(np.clip(np.searchsorted(edges, m) - 1, 0, len(area) - 1))
        area[k] = 1.0
        return area
    x = edges
    with np.errstate(divide="ignore", invalid="ignore"):
        left = (x - a) ** 2 / (max(m - a, 1e-12) * span)
        right = 1.0 - (b - x) ** 2 / (max(b - m, 1e-12) * span)
    cdf = np.where(x <= a, 0.0, np.where(x <= m, left,
                                         np.where(x < b, right, 1.0)))
    return np.diff(cdf)


def _cdf_at(edges: np.ndarray, cum: np.ndarray, area: np.ndarray,
            price: float) -> float:
    """``CDF(price)``，线性插值。

    ``cum[k]`` 含 bin k 的**全部**面积，故减 ``area[k]`` 再加插值部分，
    否则边界档会被整档重复计入（曾把 benefit 系统性高估 0.03~0.3）。
    """
    k = int(np.clip(np.searchsorted(edges, price) - 1, 0, len(area) - 1))
    frac = (price - edges[k]) / (edges[k + 1] - edges[k])
    return float(cum[k] - area[k] + frac * area[k])


def _quantile(edges: np.ndarray, cum: np.ndarray, area: np.ndarray,
              q: float, total: float) -> float:
    """分布分位数（线性插值）。"""
    target = q * total
    k = int(np.clip(np.searchsorted(cum, target), 1, len(area) - 1))
    seg = area[k]
    frac = 0.0 if seg <= 0 else float(np.clip((target - cum[k - 1]) / seg, 0, 1))
    return float(edges[k] + frac * (edges[k + 1] - edges[k]))


def _roll_core(op: np.ndarray, hi: np.ndarray, lo: np.ndarray,
               cl: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """递推内核（纯 numpy，无 pandas），返回 ``(n, 4)``。

    ``total`` 应随天数趋近 1（质量守恒自检）；显著小于 1 说明网格覆盖不足。
    输入须已剔除无效行（价格 ≤ 0 / NaN）。
    """
    n = len(cl)
    out = np.full((n, 4), np.nan)
    edges = _grid_from_price(float(np.min(lo)), float(np.max(hi)))
    avg = (op + cl + hi + lo) / 4.0

    state = np.zeros(len(edges) - 1)
    for t in range(n):
        tr = turn[t]
        if not np.isfinite(tr):
            tr = 0.0
        if tr > 0:
            state *= (1.0 - tr)
            state += _tri_area(lo[t], avg[t], hi[t], edges) * tr
        total = float(state.sum())
        if total <= 0:
            continue
        cum = np.cumsum(state)
        benefit = _cdf_at(edges, cum, state, cl[t]) / total
        trap = 1.0 - _cdf_at(edges, cum, state, cl[t] * 1.1) / total
        q05 = _quantile(edges, cum, state, 0.05, total)
        q95 = _quantile(edges, cum, state, 0.95, total)
        conc = (q95 - q05) / (q95 + q05) if (q95 + q05) > 0 else np.nan
        out[t] = (benefit, trap, conc, total)
    return out


def rolling_factors(df: pd.DataFrame) -> pd.DataFrame:
    """单股逐日递推，返回 ``benefit`` / ``trap`` / ``conc90`` / ``total``。"""
    arr = _roll_core(df["open"].to_numpy(dtype=np.float64),
                     df["high"].to_numpy(dtype=np.float64),
                     df["low"].to_numpy(dtype=np.float64),
                     df["close"].to_numpy(dtype=np.float64),
                     df["turn"].to_numpy(dtype=np.float64))
    return pd.DataFrame(arr, index=df.index,
                        columns=["benefit", "trap", "conc90", "total"])


# ---------------------------------------------------------------- 上游对照

def load_upstream():
    """加载上游 cyq.py（按文件路径 import，不装包）。"""
    spec = importlib.util.spec_from_file_location("_cyq_upstream", UPSTREAM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CYQCalculator


def upstream_calc(df: pd.DataFrame, upto: int) -> dict:
    """上游口径：历史截到当日，用「最近 210 根」算一次。

    ``calc(index)`` 的 ``end = index - range + 1``，只有 ``end == 0`` 才走
    「取末 210 根」分支，且 ``r.d`` 是窗口最后一根的日期。所以正确调用是
    ``calc(range - 1)`` + klinedata 截到当日；否则结果会晚 120 天左右。
    """
    CYQCalculator = load_upstream()
    sub = df.iloc[: upto + 1]
    k = pd.DataFrame({
        "date": sub.index.astype(str).to_numpy(),
        "open": sub["open"].to_numpy(),
        "close": sub["close"].to_numpy(),
        "high": sub["high"].to_numpy(),
        "low": sub["low"].to_numpy(),
        "turnover": sub["turn"].to_numpy() * 100.0,  # 上游吃百分数
    }, index=sub.index)
    r = CYQCalculator(k, accuracy_factor=UP_ACC, crange=UP_RANGE,
                      cyq_days=UP_DAYS).calc(UP_RANGE - 1)
    return {"benefit": float(r.benefit_part),
            "avg_cost": float(r.avg_cost),
            "conc90": float(r.percent_chips["90"]["concentration"]),
            "date": r.d}


# ---------------------------------------------------------------- verify

VERIFY_CODES = ("600519.SH", "000001.SZ", "601988.SH")


def cmd_verify(args: argparse.Namespace) -> int:
    """P0：上游实现 vs 递推版，逐日对照。

    判据分两档：**冷启动日**（t=209，两边记忆都是 210 天）看实现是否一致；
    之后的天数差异来自「长记忆 vs 210 天截断」，是模型差异不是 bug。
    """
    codes = tuple(args.codes.split(",")) if args.codes else VERIFY_CODES
    data = load_ohlcv(list(codes))
    if not data:
        log.error("无可用数据")
        return 1
    rows = []
    for code, df in data.items():
        fast = rolling_factors(df)
        n = len(df)
        if n < UP_DAYS + 5:
            continue
        sample = [UP_DAYS - 1] + list(
            np.linspace(UP_DAYS + 10, n - 1, 20).astype(int))
        for t in sample:
            t = int(min(t, n - 1))
            up = upstream_calc(df, t)
            fr = fast.iloc[t]
            rows.append({
                "code": code, "date": str(df.index[t].date()), "t": t,
                "up_benefit": up["benefit"], "fast_benefit": float(fr["benefit"]),
                "d_benefit": float(fr["benefit"]) - up["benefit"],
                "up_conc90": up["conc90"], "fast_conc90": float(fr["conc90"]),
                "d_conc90": float(fr["conc90"]) - up["conc90"],
                "fast_total": float(fr["total"]),
            })
        log.info("%s | n=%d | 采样 %d 日 | 末 total=%.3f",
                 code, n, len(sample), float(fast["total"].iloc[-1]))

    res = pd.DataFrame(rows)
    cold = res[res["t"] == UP_DAYS - 1]
    meta = {
        "grid_step": GRID_STEP,
        "upstream": {"acc": UP_ACC, "range": UP_RANGE, "days": UP_DAYS},
        "n_samples": int(len(res)),
        "cold_max_abs_d_benefit": float(cold["d_benefit"].abs().max()),
        "cold_max_abs_d_conc90": float(cold["d_conc90"].abs().max()),
        "all_mae_d_benefit": float(res["d_benefit"].abs().mean()),
        "all_max_abs_d_benefit": float(res["d_benefit"].abs().max()),
        "all_mae_d_conc90": float(res["d_conc90"].abs().mean()),
        "fast_total_min": float(res["fast_total"].min()),
    }
    for k, v in meta.items():
        if isinstance(v, float):
            log.info("%-26s %.4f", k, v)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_ROOT / "verify_upstream_vs_fast.csv", index=False,
               encoding="utf-8-sig")
    (OUT_ROOT / "verify_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("产物 → %s", OUT_ROOT)
    return 0


# ---------------------------------------------------------------- build

def _build_one(task):
    """子进程任务：单股 ``(code, op, hi, lo, cl, turn)`` → ``(code, idx, (n,3))``。"""
    code, op, hi, lo, cl, turn = task
    ok = (np.isfinite(op) & np.isfinite(hi) & np.isfinite(lo) & np.isfinite(cl)
          & (hi > 0) & (lo > 0) & (cl > 0))
    if int(ok.sum()) < 60:
        return code, None, None
    idx = np.flatnonzero(ok)
    res = _roll_core(op[idx], hi[idx], lo[idx], cl[idx], turn[idx])
    return code, idx, res[:, :3].astype(np.float32)


def _iter_tasks(wide: dict, keep: list[str]):
    cols = ("open", "high", "low", "close", "turn")
    for code in keep:
        yield (code, *(wide[c][code].to_numpy(dtype=np.float64) for c in cols))


def cmd_build(args: argparse.Namespace) -> int:
    """全市场三因子面板。

    🚨 **必须用进程池，不能用线程池**：内核是 2500 次 Python 循环，GIL 下线程
    完全不并行（实测 8 线程跑 10 分钟没跑完 1/5）。
    """
    from multiprocessing import Pool

    t0 = time.time()
    close_adj = pd.read_parquet(BASE_DIR / "close_adj.parquet")
    codes = list(close_adj.columns) if not args.codes else args.codes.split(",")
    log.info("股票池 %d 只", len(codes))

    wide = load_wide(codes)
    keep = wide.pop("_codes")
    log.info("装载完成 %.1fs（可用 %d 只）", time.time() - t0, len(keep))

    idx_map = {c: j for j, c in enumerate(keep)}
    t_axis = close_adj.index
    # 宽表 index 已对齐 close_raw；统一映射到 close_adj 的行位置
    t_pos_all = t_axis.get_indexer(wide["open"].index)
    panels = {c: np.full((len(t_axis), len(keep)), np.nan, dtype=np.float32)
              for c in ("benefit", "trap", "conc90")}

    done = 0
    with Pool(args.workers) as pool:
        for code, idx, res in pool.imap_unordered(
                _build_one, _iter_tasks(wide, keep), chunksize=20):
            done += 1
            if res is not None and idx is not None:
                rows = t_pos_all[idx]
                ok = rows >= 0
                if ok.any():
                    j = idx_map[code]
                    for k, cname in enumerate(("benefit", "trap", "conc90")):
                        panels[cname][rows[ok], j] = res[ok, k]
            if done % 500 == 0:
                log.info("  %d/%d | %.0fs", done, len(keep), time.time() - t0)

    out_dir = OUT_ROOT / DATASET / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, arr in panels.items():
        panel = pd.DataFrame(arr, index=t_axis, columns=keep)
        panel.to_parquet(out_dir / f"cyq_{c}.parquet")
        log.info("%s: %s | 非空 %.3f", c, panel.shape,
                 float(np.isfinite(arr).mean()))
    log.info("完成 %.0fs → %s", time.time() - t0, out_dir)
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """中性化 + registry 注册 + 风格诊断。

    🚨 口径**复用** `chip_behavior_probe` 的同名函数（五风格中性化、`_xcorr`
    秩相关），否则 10-27 探针的判负结论与本轮不可比。
    """
    from research.factor_library import FactorLibrary
    from scripts.evaluation import chip_behavior_probe as cbp

    src = OUT_ROOT / DATASET / "panels"
    if not src.exists():
        src = OUT_ROOT / "panels"
    panels = {p.stem: pd.read_parquet(p).astype(np.float32)
              for p in sorted(src.glob("*.parquet"))}
    if not panels:
        log.error("没有面板可处理：%s", src)
        return 1
    log.info("读入 %d 个面板：%s", len(panels), ", ".join(panels))

    base = cbp.load_base()
    diag = cbp.diagnose(panels, base)
    (OUT_ROOT / "diag.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    cbp._write(OUT_ROOT / DATASET / "panels", panels)

    neu = cbp.neutralized(panels, base)
    cbp._write(OUT_ROOT / DATASET / "panels_neu", neu)

    lib = FactorLibrary(root=OUT_ROOT, dataset=DATASET)
    rows = [{"name": n, "set": "cyq",
             "label": f"CYQ 筹码分布 {n}（P0 探针，未入生产池）"}
            for n in sorted(panels)]
    st = lib.upsert_rows(rows, source=SOURCE, family=FAMILY,
                         kind="raw", maturity="experimental",
                         frequency="日频", set_name="cyq")
    log.info("registry: %s", st)
    log.info("下一步：residual_ic_probe --ds-root %s --panels-dir panels_neu", OUT_ROOT)
    return 0


def cmd_placebo(args: argparse.Namespace) -> int:
    """安慰剂：把因子逐列循环平移（保留分布与自相关，打掉时点对应）。

    🚨 实测结论：平移版在 `panels_neu` 上也能拿到 `resfac_t=4.24`、
    正年 8/9 ⇒ **该样本下残差 IC 的 t 值不可用作显著性门槛**，
    判增益只看「真实 IC / 安慰剂 IC」的效应量比。
    """
    from research.factor_library import FactorLibrary

    ds = OUT_ROOT / DATASET
    rng = np.random.default_rng(args.seed)
    names = []
    for tag in ("benefit", "trap"):
        src = ds / "panels_neu" / f"cyq_{tag}.parquet"
        p = pd.read_parquet(src)
        arr = p.to_numpy().copy()
        for j in range(arr.shape[1]):
            arr[:, j] = np.roll(arr[:, j], int(rng.integers(120, 900)))
        out = pd.DataFrame(arr.astype(np.float32), index=p.index,
                           columns=p.columns)
        out.to_parquet(ds / "panels_neu" / f"cyq_{tag}_shift.parquet")
        names.append(f"cyq_{tag}_shift")
        log.info("%s 平移完成 | 非空 %.3f", tag, float(np.isfinite(arr).mean()))

    lib = FactorLibrary(root=OUT_ROOT, dataset=DATASET)
    rows = [{"name": n, "set": "cyq_plc",
             "label": f"安慰剂：{n}（逐列循环平移）"} for n in names]
    log.info("registry: %s", lib.upsert_rows(
        rows, source=SOURCE, family=FAMILY, kind="raw",
        maturity="experimental", frequency="日频", set_name="cyq_plc"))
    return 0


def cmd_topn(args: argparse.Namespace) -> int:
    """组合层：单因子 Top-Frac 等权，生产口径 + 多相位（P1）。

    ## 为什么必须跑这一步

    `factor_gate` 只回答「有没有 IC」。IC ≠ 经济价值 —— 残差 IC 0.007 在 5000 只
    截面上换算成年化是 1pp 量级的**估算**，不是实测。唯一能翻盘的检验是组合层。

    ## 臂设计（每臂都是「月频 Top-Frac 等权 + 27bp 时变成本 + T+1 open + 1e6」）

    | 臂 | 方向 | 为什么 |
    |---|---|---|
    | ``low`` | 因子取负，买最低 frac | 顺 **raw IC**（benefit raw IC 为负 ⇒ 低获益盘好） |
    | ``high`` | 因子原向，买最高 frac | 顺 **残差 IC**（符号与 raw 相反时才有意义） |
    | ``*_plc`` | 同方向作用于循环平移面板 | 安慰剂臂：同执行层的噪声基准 |

    🚨 **两向都跑**是铁律「拆多空两端」的要求：只跑顺 raw 方向那一端，就没法
    发现「强度全在另一端（A 股做空受限 ⇒ 吃不到）」。

    ## 相位

    定版收益是 phase 0 的**单点**，21 相位均值才是可期望值。缺省跑全部 21 相位，
    报 `mean ± std`（std 就是 RTL，调仓时点运气）。
    """
    from scripts.evaluation import caliber as cal
    from scripts.evaluation import gate_core as gc
    from strategy.examples import TopFracLongOnly

    cost = cal.cost_profile("production")
    capital = float(cal.PROFILES["production"].capital)
    ctx = cal.load_context(signal=args.signal, window="prod", execution="open")
    idx, cols = ctx.pred.index, ctx.pred.columns
    log.info("%s", cal.describe(ctx, "production", capital))
    log.info("组合任务：%d 臂 × %d 相位 × frac=%.2f | 月频等权",
             len(args.arms), len(args.phases), args.frac)

    base = OUT_ROOT / DATASET / "panels_neu"
    panels: dict[str, pd.DataFrame] = {}
    for arm in args.arms:
        name, sign = _arm_spec(arm)
        if name.endswith("_plc"):
            src_name = name[: -len("_plc")]
            raw = gc.load_panel(base / f"{src_name}.parquet", idx, cols)
            if raw is None:
                raise FileNotFoundError(f"安慰剂源面板缺失：{src_name}")
            panels[arm] = gc.cyclic_placebo(raw, seed=args.seed,
                                            name=src_name) * sign
        else:
            raw = gc.load_panel(base / f"{name}.parquet", idx, cols)
            if raw is None:
                raise FileNotFoundError(f"面板缺失：{name}.parquet")
            panels[arm] = raw * sign
        log.info("臂 %-18s 非空 %.3f", arm,
                 float(panels[arm].notna().to_numpy(dtype=float).mean()))

    rows = []
    t0 = time.time()
    for arm, panel in panels.items():
        for ph in args.phases:
            strat = TopFracLongOnly(frac=args.frac)
            bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                    initial_capital=capital, costs=cost)
            res = cal.run_once(bt, ctx, "open", signal=panel, phase=ph)
            m = cal.metrics_row(res, capital)
            rows.append({"arm": arm, "phase": ph, **{
                k: m[k] for k in ("ann", "sharpe", "sharpe_arith", "max_dd",
                                  "vol", "turnover", "cost_bp", "n_active",
                                  "n_days")}})
            log.info("  %-18s phase=%-2d ann=%6.2f%% SR=%.3f dd=%.2f%% 换手%.2f",
                     arm, ph, 100 * m["ann"], m["sharpe"], 100 * m["max_dd"],
                     m["turnover"])
        log.info("臂 %s 完成 | 累计 %.0fs", arm, time.time() - t0)

    df = pd.DataFrame(rows)
    out_dir = OUT_ROOT / "topn"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"topn_frac{args.frac:.2f}.csv", index=False,
              encoding="utf-8-sig")

    summary = df.groupby("arm").agg(
        ann_mean=("ann", "mean"), ann_std=("ann", "std"),
        sr_mean=("sharpe", "mean"), sr_std=("sharpe", "std"),
        dd_mean=("max_dd", "mean"), turnover=("turnover", "mean"),
        cost_bp=("cost_bp", "mean"), n=("phase", "size")).reset_index()
    summary.to_csv(out_dir / f"topn_summary_frac{args.frac:.2f}.csv", index=False,
                   encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v: .4f}"):
        print(summary.to_string(index=False))
    log.info("落盘 %s | 用时 %.0fs", out_dir, time.time() - t0)
    return 0


def _arm_spec(arm: str) -> tuple[str, float]:
    """``low:cyq_benefit`` → (``cyq_benefit``, −1)；``high:cyq_benefit`` → (+1)。"""
    if ":" not in arm:
        raise ValueError(f"臂名须为 low:<factor> / high:<factor>[_plc]：{arm!r}")
    side, name = arm.split(":", 1)
    if side not in ("low", "high"):
        raise ValueError(f"方向须为 low / high：{arm!r}")
    return name, -1.0 if side == "low" else 1.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CYQ 筹码分布面板")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="上游 vs 递推版对照")
    v.add_argument("--codes", default=None, help="逗号分隔，缺省 3 只代表股")
    b = sub.add_parser("build", help="全市场面板")
    b.add_argument("--codes", default=None, help="逗号分隔，缺省全 A")
    b.add_argument("--workers", type=int, default=4)
    sub.add_parser("prep", help="中性化 + 注册 + 诊断")
    pl = sub.add_parser("placebo", help="循环平移安慰剂")
    pl.add_argument("--seed", type=int, default=20261118)
    tn = sub.add_parser("topn", help="组合层：单因子 Top-Frac 等权 + 多相位")
    tn.add_argument("--arms", nargs="+",
                    default=["low:cyq_benefit", "high:cyq_benefit",
                             "low:cyq_benefit_plc", "high:cyq_benefit_plc"],
                    help="臂：low:<factor> / high:<factor>；后缀 _plc = 安慰剂臂")
    tn.add_argument("--frac", type=float, default=0.10, help="Top 比例（定版 10%%）")
    tn.add_argument("--phases", nargs="+", type=int,
                    default=list(range(21)), help="相位列表（缺省 0..20）")
    tn.add_argument("--signal", default="ens_h1h5",
                    help="借其日期轴/股票池（口径上下文），不影响选股")
    tn.add_argument("--seed", type=int, default=20261118)
    args = p.parse_args(argv)
    return {"verify": cmd_verify, "build": cmd_build, "prep": cmd_prep,
            "placebo": cmd_placebo, "topn": cmd_topn}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
