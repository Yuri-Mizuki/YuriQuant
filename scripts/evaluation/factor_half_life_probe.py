"""因子信息半衰期探针：有没有一批因子在「周频尺度」上显著更强？

背景与要回答的问题
------------------
2026-11-11，由莉酱指出周频检验的方法论缺陷：`solo_weekly` 扫的 11 种 horizon 组合
（含 `gbdt_h5`）**只换了训练标签，特征池还是同一批 882 个中长期量价+基本面因子**。
标签匹配 ≠ 特征匹配 —— 特征的信息半衰期没变，模型学不到「一周内会变的东西」。
所以「周频不行」这个结论目前只证明了「换标签不行」，没证明「周频专用因子不存在」。

本探针做**前置判断**（不做新因子工程）：用现有中性化因子面板，逐个算同一因子在
两个持有期尺度上的截面 IC，然后问：

    是否存在足够大的一批因子，使 IC(h=5) 显著优于 IC(h=20)？

- 若**存在** ⇒ 周频专用模型值得投入（有原料，值得做特征筛选/新模型）；
- 若**不存在** ⇒ 周频方向从因子层就已封死，不必再投入分钟级数据。

口径（六件套）
--------------
- 窗口：`WINDOW`（默认与生产一致 2019-01 起）；因子面板自带 2016 起，用前置做暖机。
- 因子：`E:/data/factor_library/all_a_2018_2026/panels_neu/`（**已中性化**，与生产
  `preproc=ortho` 同源）。
- 收益：`close_adj` 的 h 日**前瞻**收益 `pct_change(h).shift(-h)`，h ∈ {1,5,20}。
  🚨 这是**重叠**口径（相邻日样本强相关）⇒ t 值必须用 Newey-West 修正，不能用 naive。
- IC：Spearman 截面相关（`stats.ic.calc_ic_series`，全库唯一实现）。
- 判据：**配对差** `IC_h5 − IC_h20` 的 NW-t，而不是两列均值直接比（同一因子两期限
  高度正相关，独立比均值会把共同冲击当差异）。

🚨 关于重叠样本的诚实边界
------------------------
日频采样 + h 日前瞻 ⇒ 相邻 h 个样本共享 h−1 天收益，重叠率 (h−1)/h。
h=20 时重叠 95%，有效独立样本 ≈ T/20。故：
  - NW-t 的 lags 取 `h` 量级（h=20 → lags=20）；
  - **额外做非重叠采样对照**（每 h 天取一个样本），两套 t 值都报。
只报重叠口径的 t = 没做。

用法::

    python -m scripts.evaluation.factor_half_life_probe            # 全量（约 10~20 分钟）
    python -m scripts.evaluation.factor_half_life_probe --quick    # 80 个因子抽样
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "factor_half_life"

#: 因子库数据集名（路径走 ``FactorLibrary`` → ``Config.paths()``，不写盘符字面量，
#: 见 ``tests/test_no_hardcoded_drive_paths.py``）。
LIB_DATASET = "all_a_2018_2026"
LIB_HORIZONS = (1, 5, 10, 20)


def _library():
    """按数据集名打开因子库（根路径由 config 解析，跨机器可移植）。"""
    from research.factor_library import FactorLibrary
    return FactorLibrary(dataset=LIB_DATASET)


PANELS = _library().root / "panels_neu"

# 收益/掩码走 caliber 的 BASE_DIR（= reports/alla_rolling/_base），不另写路径字面量：
# 那里是 882 信号与回测共用的同源数据态，与生产口径一致。
import scripts.evaluation.caliber as cal  # noqa: E402

BASE = cal.BASE_DIR

WINDOW_START = "2019-01-02"
WINDOW_END = "2026-09-07"
HORIZONS = (1, 5, 20)


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------
def nw_t(x, lags: int) -> tuple[float, float]:
    """均值与 Newey-West 修正 t 值（Bartlett 核）。

    Args:
        x: 序列（可含 NaN，内部剔除）。
        lags: 滞后阶数。重叠前瞻口径应取 >= h，否则 t 被严重高估。

    Returns:
        (mean, t)。样本不足时返回 (nan, nan)。
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 10:
        return float("nan"), float("nan")
    m = a.mean()
    e = a - m
    s = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * (e[lag:] @ e[:-lag]) / n
    if s <= 0:
        return float(m), float("nan")
    return float(m), float(m / np.sqrt(s / n))


def regress_out_market(ic_df: pd.DataFrame) -> pd.DataFrame:
    """剔除横截面共同冲击（全因子当日 IC 均值），再做配对检验。

    理由：所有因子在同一天的 IC 共享大盘/风格冲击 ⇒ 配对差的时序仍含共同项。
    去均值后剩下「该因子相对其他因子的超额分辨力」，配对更干净。
    只减均值不改排序，不影响「谁更强」的判断，但会显著降低噪声。
    """
    return ic_df.sub(ic_df.mean(axis=1), axis=0)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_returns(h: int) -> pd.DataFrame:
    """前瞻 h 日收益面板（重叠口径）。"""
    close = pd.read_parquet(BASE / "close_adj.parquet")
    fwd = close.pct_change(h, fill_method=None).shift(-h)
    return fwd.replace([np.inf, -np.inf], np.nan)


def load_mask() -> pd.DataFrame | None:
    """可交易掩码（涨跌停/停牌/ST）。缺失时返回 None（退化为不掩码并告警）。"""
    p = BASE / "tradable_mask.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def factor_files(quick: bool) -> list[Path]:
    """因子面板文件列表。quick 模式按名字排序等距抽样 80 个。"""
    fs = sorted(PANELS.glob("*.parquet"))
    if quick:
        step = max(1, len(fs) // 80)
        fs = fs[::step][:80]
    return fs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _init_shared(idx, cols) -> None:
    """worker 初始化：只收小对象（索引/列名）。

    🚨 **不进 initargs 的东西**：``fwd`` 三个面板各 59MB、``mask`` 15MB。Windows
    只有 spawn（无 fork），``initargs`` 要走 pickle ⇒ 8 个 worker 复制 ~750MB，
    启动和序列化成本超过计算本身（实测 80 因子 15 分钟仍未完）。
    改为 worker **首次调用时自己从磁盘加载**（见 ``_ensure_data``）：8 个进程并行
    各读 2s，总成本 2s 而非 750MB 的序列化。
    """
    _CACHE["idx"] = idx
    _CACHE["cols"] = cols


def _ensure_data() -> None:
    """懒加载收益/掩码面板到模块级缓存（每个 worker 进程只做一次）。"""
    if "fwd" in _CACHE:
        return
    _CACHE["fwd"] = {h: load_returns(h) for h in HORIZONS}
    _CACHE["mask"] = load_mask()


def _worker(f: str) -> list[dict]:
    """单个因子的三期限 IC + 配对差（供进程池调用）。

    🚨 每个 worker 只用**本因子**的 valid 掩码算收益 rank，不跨因子共享：
    实测不同因子的 NaN 模式差异达 6%（`accruals` 0.7151 vs `alpha101_001` 0.7620），
    共享会让 rank 基准漂移、IC 口径不再等价于 `calc_ic_series` 默认路径。
    代价是收益 rank 无法复用 —— 用并行度换，不牺牲口径。
    """
    from stats.ic import calc_ic_series

    _ensure_data()
    idx, cols = _CACHE["idx"], _CACHE["cols"]

    # 收益/掩码的 reindex 只做一次（对齐同一批 idx×cols），不随因子重复。
    # 实测这是单因子 10.5s 里可省掉的一部分：mask 有 3 个期限 × 每因子一次。
    if "fwd_r" not in _CACHE:
        _CACHE["fwd_r"] = {h: _CACHE["fwd"][h].reindex(index=idx, columns=cols)
                           for h in HORIZONS}
        m = _CACHE["mask"]
        _CACHE["mask_r"] = (None if m is None else
                            m.reindex(index=idx, columns=cols).fillna(True).astype(bool))
    fwd, mask = _CACHE["fwd_r"], _CACHE["mask_r"]

    try:
        fac = pd.read_parquet(f).reindex(index=idx, columns=cols)
    except Exception:                                  # noqa: BLE001
        return []
    if fac.notna().sum().sum() < 1e4:
        return []

    rec: dict[str, float | str] = {"factor": str(f).replace("\\", "/").rsplit("/", 1)[-1]
                                   .replace(".parquet", "")}
    ics: dict[int, pd.Series] = {}
    for h in HORIZONS:
        try:
            ic = calc_ic_series(fac, fwd[h], method="spearman", tradable=mask,
                                min_samples=30)
        except Exception:                              # noqa: BLE001
            return []
        ic = ic.dropna()
        if len(ic) < 100:
            return []
        ics[h] = ic
        rec[f"ic_h{h}"] = float(ic.mean())
        rec[f"ic_std_h{h}"] = float(ic.std())
        rec[f"ic_n_h{h}"] = int(len(ic))
        rec[f"ic_pos_h{h}"] = float((ic > 0).mean())

    # ---- 配对差 ----
    # 🚨 **口径三次修正的最终形态：定向差 `sign(ic_h20) × (ic_h5 − ic_h20)`**。
    #
    # 踩过两个坑，都不能用：
    #
    # ❌ v1 带符号差 `ic_h5 − ic_h20`：池内 **260/944（27.5%）的 IC 为负**，
    #    对负 IC 因子「|IC| 在 h5 更大」表现为 `ic_h5` **更负** ⇒ 符号记反。
    #
    # ❌ v2 逐日绝对值 `mean(|ic_h5|) − mean(|ic_h20|)`：这是**方差假象**。
    #    逐日取绝对值后噪声不再抵消 ⇒ `E|IC_t|` 是折半正态，随日度波动 **机械性增大**。
    #    实测 79.2% 的因子 `std_h5 > std_h20`，而 `corr(Δ_abs, std_h5−std_h20) = +0.883`
    #    ⇒ 该口径 88% 只是「h5 的 IC 日波动更大」的线性映射，与分辨力无关。
    #    它把结论从 36% 正占比翻到 66%，纯属假象。
    #
    # ✅ v3 定向差：先用 `sign(ic_h20)` 把因子方向**对齐成正**，再比均值大小。
    #    等价于「若该因子按其历史方向使用，5 日持有比 20 日持有准多少」。
    #    实测正占比 35.2%、中位 −0.00196，与独立的「|mean| 之差」口径（36.0%）吻合。
    #
    # `sign(ic_h20)` 用**整段均值**的符号（不是逐日符号）：逐日符号会被噪声翻转，
    # 等价于又一次绝对值化。因子方向在整个窗口内视为固定，这是标准做法（同 IC 符号约定）。
    for h, base_h in ((5, 20), (1, 20), (1, 5)):
        d_signed = (ics[h] - ics[base_h]).dropna()
        # 方向对齐：以基准期的均值符号为准
        sgn = 1.0 if float(ics[base_h].mean()) >= 0 else -1.0
        d = (sgn * (ics[h] - ics[base_h])).dropna()
        if len(d) < 100:
            rec[f"d_h{h}_h{base_h}"] = float("nan")
            rec[f"d_signed_h{h}_h{base_h}"] = float("nan")
            rec[f"d_absdiff_h{h}_h{base_h}"] = float("nan")
            rec[f"dir_h{h}_h{base_h}"] = float("nan")
            rec[f"t_h{h}_h{base_h}"] = float("nan")
            rec[f"t_nonoverlap_h{h}_h{base_h}"] = float("nan")
            continue
        mean_d, t_d = nw_t(d, lags=max(h, base_h))
        rec[f"d_h{h}_h{base_h}"] = mean_d          # ← 主判据：定向差
        rec[f"t_h{h}_h{base_h}"] = t_d
        rec[f"dir_h{h}_h{base_h}"] = sgn
        # 两个诊断口径，不进判据（保留是为了今后复现「假象是怎么来的」）
        rec[f"d_signed_h{h}_h{base_h}"] = float(d_signed.mean())
        rec[f"d_absdiff_h{h}_h{base_h}"] = float(
            (ics[h].abs() - ics[base_h].abs()).dropna().mean())
        step = max(h, base_h)
        _, t_ni = nw_t(d.iloc[::step], lags=2)
        rec[f"t_nonoverlap_h{h}_h{base_h}"] = t_ni
    return [rec]


def run(quick: bool = False, workers: int | None = None, resume: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_files = factor_files(quick)
    n_workers = workers or max(1, (os.cpu_count() or 4) - 1)

    # ---- 断点续跑：已完成因子按名字跳过（部分写盘，见下）----
    done_names: set[str] = set()
    part = OUT / "_partial.jsonl"
    if not resume and part.exists():
        # --fresh：必须**清空**旧部分结果。否则 append 模式会把上一轮（可能是
        # --quick 的 80 个抽样）与全量结果混在同一文件里，drop_duplicates 也救不了
        # 「quick 抽到的因子」与「全量重算的同一因子」的先后覆盖顺序。
        part.unlink()
    if resume and part.exists():
        with open(part, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_names.add(json.loads(line)["factor"])
                    except Exception:                  # noqa: BLE001
                        pass
    files = [f for f in all_files if f.stem not in done_names]
    print(f"因子面板 {len(all_files)} 个（已完成 {len(done_names)}，本次 {len(files)}）| "
          f"窗口 {WINDOW_START}..{WINDOW_END} | horizons {HORIZONS} | workers {n_workers}",
          flush=True)

    ref = pd.read_parquet(all_files[0])
    idx = ref.index[(ref.index >= pd.Timestamp(WINDOW_START))
                    & (ref.index <= pd.Timestamp(WINDOW_END))]
    cols = ref.columns
    del ref

    # ⚠️ Windows 只有 spawn：只把 idx/cols 进 initargs，收益/掩码由 worker 自加载
    # （见 _init_shared 的注释）。这与「15 分钟跑不完 80 因子」的失败方案的区别就在此。
    rows: list[dict] = []
    n_total = len(files)
    if n_total:
        with open(part, "a", encoding="utf-8") as sink:
            def _emit(out: list[dict], n: int) -> None:
                rows.extend(out)
                for r in out:
                    sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                if out:
                    sink.flush()                       # 立刻落盘，崩溃可续
                if n % 25 == 0 or n == n_total:
                    rate = n / max(1e-9, time.time() - t0)
                    eta = (n_total - n) / rate if rate > 0 else float("nan")
                    print(f"  {n}/{n_total}  {time.time() - t0:.0f}s  "
                          f"ETA {eta / 60:.0f}min", flush=True)

            if n_workers > 1:
                import multiprocessing as mp
                ctx = mp.get_context("spawn")
                with ctx.Pool(n_workers, initializer=_init_shared,
                              initargs=(idx, cols)) as pool:
                    for n, out in enumerate(pool.imap_unordered(
                            _worker, [str(f) for f in files], chunksize=4), 1):
                        _emit(out, n)
            else:
                _init_shared(idx, cols)
                for n, f in enumerate(files, 1):
                    _emit(_worker(str(f)), n)

    # ---- 汇总（含续跑前已完成的部分）----
    recs = []
    if part.exists():
        with open(part, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:                  # noqa: BLE001
                        pass
    df = pd.DataFrame(recs).drop_duplicates("factor", keep="last")
    if df.empty:
        print("⚠️ 无有效因子记录，检查面板路径/窗口", flush=True)
        return

    df = df.sort_values("ic_h20", ascending=False)
    df.to_csv(OUT / "factor_ic_by_horizon.csv", index=False, encoding="utf-8-sig")

    # ---- 汇总：全因子 IC 均值的共同冲击（大盘/风格在三个尺度上的可预测性）----
    summary: dict[str, float | int] = {}
    for h in HORIZONS:
        s = df[f"ic_h{h}"].dropna()
        summary[f"ic_h{h}_mean"] = float(s.mean())
        summary[f"ic_h{h}_median"] = float(s.median())
        summary[f"ic_h{h}_pos_share"] = float((s > 0).mean())

    # ---- 关键判据 1：IC(h=5) 是否整体强于 IC(h=20)？----
    d5_20 = df["d_h5_h20"].dropna()
    summary["d5_20_mean"] = float(d5_20.mean())
    summary["d5_20_median"] = float(d5_20.median())
    summary["d5_20_pos_share"] = float((d5_20 > 0).mean())
    summary["d5_20_n"] = int(len(d5_20))
    summary["d5_20_nw_t"] = nw_t(d5_20, lags=20)[1]

    # ---- 关键判据 2：有多少因子「周频显著优于月频」？----
    tcol = "t_h5_h20"
    tncol = "t_nonoverlap_h5_h20"
    for thr in (1.96, 2.58):
        summary[f"n_h5_gt_h20_t{thr}"] = int((df[tcol] > thr).sum())
        summary[f"n_h5_gt_h20_nonoverlap_t{thr}"] = int((df[tncol] > thr).sum())
    summary["n_both_t1.96"] = int(((df[tcol] > 1.96) & (df[tncol] > 1.96)).sum())
    summary["n_factor"] = int(len(df))

    # ---- 关键判据 3：反向（月频更强）的数量，判断是不是单向结构 ----
    summary["n_h20_gt_h5_t1.96"] = int((df[tcol] < -1.96).sum())

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 落盘：按 t 值排序的头部因子（供人工看名字族）----
    top = df.nlargest(50, tcol)[["factor", "ic_h1", "ic_h5", "ic_h20",
                                 "d_h5_h20", tcol, tncol]]
    top.to_csv(OUT / "top_h5_over_h20.csv", index=False, encoding="utf-8-sig")
    bottom = df.nsmallest(50, tcol)[["factor", "ic_h1", "ic_h5", "ic_h20",
                                     "d_h5_h20", tcol, tncol]]
    bottom.to_csv(OUT / "bottom_h5_over_h20.csv", index=False, encoding="utf-8-sig")

    print("\n=== 汇总 ===", flush=True)
    for k, v in summary.items():
        print(f"  {k:34s} {v}", flush=True)
    print(f"\n产物 → {OUT}  ({time.time() - t0:.0f}s)", flush=True)


def run_from_library() -> None:
    """**直接读因子库现成的 `ic_h{1,5,10,20}.parquet`**（秒级，不重算）。

    🚨 为什么该先走这条
    -------------------
    2026-11-11 由莉酱问「不是应该直接看一下因子库里因子们的周频 IC 就知道了吗」——
    **问得对，我确实绕了路**：因子库里本来就有 `ic_h1/5/10/20.parquet`
    （2511 日 × 936 因子），是 builder 每次入库时顺手融合好的，我却从
    `panels_neu` 逐因子重算，白花了 37 分钟。

    两条路的口径差异（已实测）
    -------------------------
    - **库版**：`scripts.builders.common.merge_ic` 用 `load_close_adj()`（从
      `daily_all_a.parquet` 现构建）+ **`IC_CODE_STRIDE=3` 抽稀到 1/3 股票**。
    - **本探针**：`caliber.BASE_DIR/close_adj.parquet`（882 生产信号同源）+ 全票。
    两者复权价最大差 **3.5%** ⇒ 同因子 IC 有漂移（`ic_h20` 逐因子相关 0.983）。
    抽样复现：`is_st` 用库版重算得 **+0.00896**，库值 **+0.00896**，相关 **1.0000**
    ⇒ 库口径确实是「库版 close_adj + stride 3」，差异来自数据版本而非算法。

    结论方向两条路**一致**（见 `reports/factor_half_life/README.md` §二.8）：
    库版正占比 0.193、本探针 0.352，都远离 0.5，且强因子档位分别是 0.000 / 0.048
    ⇒ 「无原料」不依赖口径选择。

    产物与 `run()` 同构（同样的 CSV/summary），便于 `factor_half_life_verdict` 直接消费。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lib = _library()
    ics = {}
    for h in LIB_HORIZONS:
        m = lib.ic_matrix(h)
        if m is None:
            raise SystemExit(
                f"缺 {lib.root}/ic_h{h}.parquet —— 因子库未建该期限 IC 或数据集名变了")
        ics[h] = m
    w = slice(WINDOW_START, WINDOW_END)
    n_days = len(ics[5].loc[w])
    print(f"库窗口 {WINDOW_START}..{WINDOW_END}（{n_days} 日）| "
          f"因子 {ics[5].shape[1]} | horizons {LIB_HORIZONS}", flush=True)

    m = {h: ics[h].loc[w].mean() for h in LIB_HORIZONS}
    sd = {h: ics[h].loc[w].std() for h in LIB_HORIZONS}
    cnt = {h: ics[h].loc[w].notna().sum() for h in LIB_HORIZONS}

    # 主判据期限固定为 5 vs 20（与重算版一致），h=10 只作辅助
    base_h, cmp_h = 20, 5
    names = [c for c in m[cmp_h].index
             if c in m[base_h].index and np.isfinite(m[cmp_h][c]) and np.isfinite(m[base_h][c])]
    recs = []
    for n in names:
        r: dict[str, float | str] = {"factor": n}
        for h in LIB_HORIZONS:
            r[f"ic_h{h}"] = float(m[h][n])
            r[f"ic_std_h{h}"] = float(sd[h][n])
            r[f"ic_n_h{h}"] = int(cnt[h][n])
        a_cmp, a_base = float(m[cmp_h][n]), float(m[base_h][n])
        sgn = 1.0 if a_base >= 0 else -1.0
        r[f"d_h{cmp_h}_h{base_h}"] = sgn * (a_cmp - a_base)
        r[f"d_signed_h{cmp_h}_h{base_h}"] = a_cmp - a_base
        r[f"d_absdiff_h{cmp_h}_h{base_h}"] = abs(a_cmp) - abs(a_base)
        r[f"dir_h{cmp_h}_h{base_h}"] = sgn
        recs.append(r)
    df = pd.DataFrame(recs).sort_values(f"ic_h{base_h}", ascending=False)
    # 🚨 不覆盖重算版的 `factor_ic_by_horizon.csv` —— 两者口径不同（见 docstring），
    # 覆盖会让 `factor_half_life_verdict` 在不知情的情况下换数据源。各存一份。
    df.to_csv(OUT / "factor_ic_from_library.csv", index=False, encoding="utf-8-sig")

    summary: dict[str, float | int] = {"source": "library_ic_parquet",
                                       "window": f"{WINDOW_START}..{WINDOW_END}",
                                       "n_days": int(n_days)}
    for h in LIB_HORIZONS:
        s = df[f"ic_h{h}"].dropna()
        summary[f"ic_h{h}_mean"] = float(s.mean())
        summary[f"ic_h{h}_median"] = float(s.median())
    d = df[f"d_h{cmp_h}_h{base_h}"].dropna()
    summary[f"d{cmp_h}_{base_h}_mean"] = float(d.mean())
    summary[f"d{cmp_h}_{base_h}_median"] = float(d.median())
    summary[f"d{cmp_h}_{base_h}_pos_share"] = float((d > 0).mean())
    summary[f"d{cmp_h}_{base_h}_n"] = int(len(d))
    summary["n_factor"] = int(len(df))
    with open(OUT / "summary_from_library.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    df.nlargest(50, f"d_h{cmp_h}_h{base_h}")[
        ["factor"] + [f"ic_h{h}" for h in LIB_HORIZONS] + [f"d_h{cmp_h}_h{base_h}"]
    ].to_csv(OUT / "top_h5_over_h20_from_library.csv", index=False, encoding="utf-8-sig")
    df.nsmallest(50, f"d_h{cmp_h}_h{base_h}")[
        ["factor"] + [f"ic_h{h}" for h in LIB_HORIZONS] + [f"d_h{cmp_h}_h{base_h}"]
    ].to_csv(OUT / "bottom_h5_over_h20_from_library.csv", index=False, encoding="utf-8-sig")

    print("\n=== 汇总（库版 IC，秒级）===", flush=True)
    for k, v in summary.items():
        print(f"  {k:26s} {v}", flush=True)
    print("\n⚠️ 库版无逐日序列的 t 值（因未重采样），`factor_half_life_verdict` 的 "
          "G3/G4 依赖 t 列 ⇒ 只走 G0/G1/G2 三关。要全四关请跑重算版。", flush=True)
    print(f"产物 → {OUT}  ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="因子信息半衰期探针")
    ap.add_argument("--quick", action="store_true", help="80 个因子抽样冒烟")
    ap.add_argument("--workers", type=int, default=None, help="并行进程数（默认 CPU-1）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略 _partial.jsonl 的断点，从头跑")
    ap.add_argument("--from-library", action="store_true",
                    help="直接读因子库现成的 ic_h*.parquet（秒级，优先用这个）")
    args = ap.parse_args()
    if args.from_library:
        run_from_library()
    else:
        run(quick=args.quick, workers=args.workers, resume=not args.fresh)


if __name__ == "__main__":
    main()