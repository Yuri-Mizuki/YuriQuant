"""全A公开因子面板重建（alpha101/158/191/360）→ all_a_2018_2026 数据集。

背景（2026-09-01）：因子库现有面板只覆盖 HS300 2022-2025；全A 2018~now 的
多年度滚动训练实验需要全截面、全历史的公因子面板。公因子全部为量价因子，
仅需日线 + 后复权因子即可重建，不依赖财务表。

与 build_alpha_factors.py 的差异（为何单开脚本）：
- 全A ~5500 股 × 2015~now 一次进内存逐因子计算（截面算子 cs_rank/行业中性化
  要求全截面在场，不能按股票分块）；每算完一个因子立即落盘 float32，
  峰值内存 ~2-3GB/进程，而不是 compute_alphaXXX 的全量 dict（~45GB 不可行）；
- **多进程并行**（--workers N）：按因子轮转分片，各进程独立加载基础面板
  （截面语义在每个因子内部不变，分片只是因子间并行）；
- 不做逐因子四配置回测评估（重），改为预计算 4 个 horizon 的日频 IC 缓存
  （ic_h{1,5,10,20}.parquet，date×factor），供实验的特征漏斗零 IO 复用；
- 轻量 registry.csv（覆盖/IC 均值），不进正式库监控流程。

口径与 build_alpha_factors 严格一致：
- 价格一律后复权（raw × backward_factor），vwap = amount/volume × bf；
- warmup：公式最大回看 250 日，数据从 2015-01 起，面板保留 keep_from(20160701)
  之后（首日即充分预热）；
- alpha101 IndNeutralize 用申万一级行业 PIT 面板（已有全A覆盖）；
- 与 alpha101 重复的 alpha191 因子在公式层已去重（skip 集逻辑沿用）。

用法:
    python scripts/builders/build_alla_alpha_panels.py                 # 单进程
    python scripts/builders/build_alla_alpha_panels.py --workers 6     # 6 进程并行
    python scripts/builders/build_alla_alpha_panels.py --workers 6 --resume
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_alpha")

HORIZONS = (1, 5, 10, 20)
# IC 缓存抽稀：每 3 只取 1（截面 ~1800 只），均值 IC 精度足够特征筛选用
IC_CODE_STRIDE = 3


def load_alla_panels(warmup: int):
    """读 daily_all_a + backward_factor → 后复权 AlphaData 输入面板（float32）。"""
    from config import Config
    from data.cache import DataCache
    from data.industry import IndustryClassification
    from data.offline import OfflineQuietDataSource

    cache_root = Path(str(Config.cache()["root"]))
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(
        daily.index.levels[0].normalize(), level=0)
    w0 = pd.Timestamp(str(warmup))
    daily = daily[daily.index.get_level_values(0) >= w0]
    log.info("日线 %d 行, %d 日 × %d 股（%s ~ %s）", len(daily),
             daily.index.get_level_values(0).nunique(),
             daily.index.get_level_values(1).nunique(),
             daily.index.get_level_values(0).min().date(),
             daily.index.get_level_values(0).max().date())

    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, hi, lo, c = _panel("open"), _panel("high"), _panel("low"), _panel("close")
    v, amt = _panel("volume"), _panel("amount")

    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf = bf[[x for x in c.columns if x in bf.columns]]
    f = bf.reindex(index=c.index, columns=c.columns).ffill()
    n_no_bf = int(f.iloc[-1].isna().sum())
    for pnl in (o, hi, lo, c):
        pnl[:] = (pnl.values * f.values).astype(np.float32)
    vwap = ((amt / v).replace([np.inf, -np.inf], np.nan) * f).astype(np.float32)
    log.info("后复权完成: %d 股 × %d 日（%d 股无复权因子被置 NaN）",
             c.shape[1], len(c), n_no_bf)

    industry = None
    try:
        cache = DataCache(OfflineQuietDataSource())
        industry = IndustryClassification(cache, level=1).get_industry_panel(
            list(c.columns), c.index)
        if industry.isna().all().all():
            industry = None
        else:
            log.info("行业面板: 覆盖率 %.2f", industry.notna().mean().mean())
    except Exception as e:  # noqa: BLE001
        log.warning("行业面板不可用（%s），IndNeutralize 退化为恒等", str(e)[:80])

    panels = {"open": o.astype(np.float32), "high": hi.astype(np.float32),
              "low": lo.astype(np.float32), "close": c.astype(np.float32),
              "volume": v.astype(np.float32), "amount": amt.astype(np.float32),
              "vwap": vwap}
    return panels, industry, c.astype(np.float32)


def collect_factor_fns() -> dict[str, tuple[str, str]]:
    """{name: (setname, 注册名)}——函数对象在 worker 内按名解析（spawn 安全）。"""
    from factor.alpha101 import ALPHA101, SKIPPED_101
    from factor.alpha158 import ALPHA158, ALPHA360
    from factor.alpha191 import ALPHA191, SKIPPED_191

    out: dict[str, tuple[str, str]] = {}
    for name in ALPHA101:
        if name not in SKIPPED_101:
            out[name] = ("alpha101", name)
    for name in ALPHA191:
        if name not in SKIPPED_191:
            out[name] = ("alpha191", name)
    for name in ALPHA158:
        out[name] = ("alpha158", name)
    for name in ALPHA360:
        out[name] = ("alpha360", name)
    return out


def registry_lookup(setname: str, name: str):
    """按 (因子集, 注册名) 解析因子函数（spawn 进程内按名查找，不跨进程序列化）。"""
    from factor.alpha101 import ALPHA101
    from factor.alpha158 import ALPHA158, ALPHA360
    from factor.alpha191 import ALPHA191
    table = {"alpha101": ALPHA101, "alpha191": ALPHA191,
             "alpha158": ALPHA158, "alpha360": ALPHA360}[setname]
    return table[name]


def _worker(wid: int, n_workers: int, warmup: int, keep_from: int,
            dataset: str, resume: bool):
    """worker 进程：轮转分片计算因子 → panels/*.parquet + 分片统计/IC。"""
    from config import Config
    from factor.alpha_base import SET_LABELS, AlphaData
    from stats.ic import calc_ic_series

    t0 = time.time()
    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / dataset
    panels_dir = ds_dir / "panels"
    parts_dir = ds_dir / "ic_parts"
    panels_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    all_fns = collect_factor_fns()
    names = sorted(all_fns)
    mine = [n for i, n in enumerate(names) if i % n_workers == wid]

    stats_path = ds_dir / f"factor_stats_w{wid}.jsonl"
    done: set[str] = set()
    if resume and stats_path.exists():
        done = {json.loads(line)["name"]
                for line in stats_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    todo = [n for n in mine if n not in done]
    log.info("[w%d] 分片 %d 因子, 待算 %d（resume 跳过 %d）", wid, len(mine),
             len(todo), len(done) - len(todo) if done else 0)
    if not todo:
        return

    panels_px, industry, close_adj = load_alla_panels(warmup)
    d = AlphaData(panels_px, industry=industry)
    k0 = pd.Timestamp(str(keep_from))
    close_kept = close_adj.loc[close_adj.index >= k0]
    fwd = {h: close_kept.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_stride_codes = close_kept.columns[::IC_CODE_STRIDE]

    ic_frames: dict[int, dict[str, pd.Series]] = {h: {} for h in HORIZONS}

    def _flush():
        for h in HORIZONS:
            df = pd.DataFrame(ic_frames[h])
            if len(df):
                df.index.name = "date"
                df.astype(np.float32).to_parquet(parts_dir / f"w{wid}_h{h}.parquet")

    n_done = 0
    for i, name in enumerate(todo, 1):
        setname, reg_name = all_fns[name]
        try:
            fn = registry_lookup(setname, reg_name)
            p = fn(d)
        except Exception as e:  # noqa: BLE001
            log.warning("[w%d] 因子 %s 计算失败: %s（跳过）", wid, name, str(e)[:120])
            continue
        # 先 replace 再 astype 会漏掉 float64 巨值（astype 溢出成 float32 inf），
        # 故 astype 之后再 replace（2026-09-01 alpha101_084/alpha191_017 教训）
        p = p.loc[p.index >= k0].astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan)
        p.to_parquet(panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        ic_means = {}
        for h in HORIZONS:
            ic = calc_ic_series(p[ic_stride_codes], fwd[h]).astype(np.float32)
            ic_frames[h][name] = ic
            ic_means[h] = float(ic.mean())
        row = {"name": name, "set": setname,
               "label": f"{SET_LABELS.get(setname, setname)} #{name.split('_')[-1]}",
               "coverage": cov,
               **{f"ic_mean_h{h}": ic_means[h] for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_done += 1
        if i % 25 == 0:
            _flush()
            log.info("[w%d] %d/%d 最近: %s cov=%.2f ic_h1=%+.4f | %.0fs",
                     wid, i, len(todo), name, cov, ic_means[1], time.time() - t0)
    _flush()
    log.info("[w%d] 完成 %d 因子, 耗时 %.0fs", wid, n_done, time.time() - t0)


def _merge_outputs(dataset: str):
    """合并各 worker 的统计与 IC 分片 → registry.csv + ic_h{h}.parquet。"""
    from config import Config
    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / dataset

    rows = []
    for p in sorted(ds_dir.glob("factor_stats_w*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        log.warning("无统计产出")
        return
    reg = pd.DataFrame(rows).drop_duplicates(subset="name", keep="last")
    reg.to_csv(ds_dir / "registry.csv", index=False, encoding="utf-8-sig")
    for h in HORIZONS:
        parts = sorted(ds_dir.glob(f"ic_parts/w*_h{h}.parquet"))
        if not parts:
            continue
        merged = pd.concat([pd.read_parquet(p) for p in parts], axis=1)
        merged.index.name = "date"
        merged.astype(np.float32).to_parquet(ds_dir / f"ic_h{h}.parquet")
        log.info("ic_h%d: %d 因子 × %d 日", h, merged.shape[1], len(merged))
    log.info("registry: %d 因子 | IC(h1) 中位数 %.4f, |IC|>0.01 的 %d 个",
             len(reg), reg["ic_mean_h1"].median(),
             int((reg["ic_mean_h1"].abs() > 0.01).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20150101, help="计算起点（含预热）")
    ap.add_argument("--keep-from", type=int, default=20160701,
                    help="面板保留起点（之后为可用因子值）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--resume", action="store_true", help="跳过已完成的因子")
    ap.add_argument("--workers", type=int, default=1, help="并行进程数")
    args = ap.parse_args()

    t0 = time.time()
    if args.workers <= 1:
        _worker(0, 1, args.warmup, args.keep_from, args.dataset, args.resume)
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker_proc, name=f"alpha-w{w}",
                             args=(w, args.workers, args.warmup, args.keep_from,
                                   args.dataset, args.resume))
                 for w in range(args.workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        failed = [p.name for p in procs if p.exitcode not in (0, None)]
        if failed:
            raise RuntimeError(f"worker 异常退出: {failed}")
    _merge_outputs(args.dataset)
    log.info("全部完成, 耗时 %.0fs", time.time() - t0)


def _worker_proc(wid, n_workers, warmup, keep_from, dataset, resume):
    """spawn 入口：独立进程内跑 worker（异常时非零退出）。"""
    try:
        _worker(wid, n_workers, warmup, keep_from, dataset, resume)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
