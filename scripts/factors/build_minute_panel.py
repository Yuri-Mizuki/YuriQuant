"""
分钟面板构建入口
================

把 parquet 分钟长表缓存物化为内存映射稠密面板（data.intraday.MinutePanelStore），
并可顺带跑统计特征层（factor.intraday_features）做初步 IC 快照。这是日内
因子挖掘 pipeline 的数据层入口（TODO「分钟频因子挖掘」第一阶段）。

用法
----
    python -m scripts.factors.build_minute_panel --offline                      # 面板 + 覆盖统计
    python -m scripts.factors.build_minute_panel --offline --features           # 另存统计特征长表
    python -m scripts.factors.build_minute_panel --offline --features --demo-ic # 打印特征次日IC快照
    python -m scripts.factors.build_minute_panel --mock                         # 合成数据端到端演示
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

from config import Config  # noqa: E402
from data.cache import DataCache  # noqa: E402
from data.intraday import MinutePanelStore  # noqa: E402
from factor.intraday_features import FEATURE_DOCS, extract_all  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402
from stats.significance import mean_inference  # noqa: E402

log = setup_logging("build_minute_panel")


def gen_mock_minute(n_days: int = 60, n_codes: int = 8, period: int = 5,
                    seed: int = 7) -> pd.DataFrame:
    """合成分钟长表（演示/冒烟用，与真实缓存同构）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    bars = pd.date_range("09:35", "11:30", freq=f"{period}min").tolist() + \
        pd.date_range("13:05", "15:00", freq=f"{period}min").tolist()
    times = pd.DatetimeIndex([d.replace(hour=t.hour, minute=t.minute)
                              for d in dates for t in bars])
    rows = []
    for c in codes:
        n = len(times)
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.001, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
        vol = rng.lognormal(10, 0.4, n)
        rows.append(pd.DataFrame({
            "kline_time": times, "code": c, "open": open_, "high": high,
            "low": low, "close": close, "volume": vol, "amount": vol * close,
        }))
    df = pd.concat(rows, ignore_index=True).set_index(["kline_time", "code"])
    return df.sort_index()


def demo_ic(features: dict[str, pd.DataFrame], cache: "DataCache", pool: str) -> pd.DataFrame:
    """特征 vs 次日收益的日度 spearman IC 快照（演示口径，非入库结论）。"""
    daily = cache.read_daily(pool)
    if daily is None or daily.empty:
        raise SystemExit("daily 缓存缺失，--demo-ic 需要先 python -m scripts.ingest.update_data")
    close = daily["close"].unstack("code")
    fwd = close.shift(-1) / close - 1.0
    rows = []
    for name, panel in features.items():
        aligned_p, aligned_r = panel.align(fwd, join="inner")
        # 逐日截面 spearman：秩相关
        pr = aligned_p.rank(axis=1)
        rr = aligned_r.rank(axis=1)
        n = pr.notna().sum(axis=1)
        cov = (pr * rr).mean(axis=1, skipna=True) - pr.mean(axis=1) * rr.mean(axis=1)
        ic = cov / (pr.std(axis=1) * rr.std(axis=1))
        ic = ic[n >= 10]
        if ic.empty or ic.isna().all():
            continue
        # t 统一走 stats.significance（2026-09-11 口径统一）。顺带修正旧写法
        # "分子 skipna、分母却用含 NaN 的 len(ic)" 的口径不匹配，并剔除截面秩
        # 标准差为 0 时产生的 ±inf —— 故**此处数值会变**（t 相对旧值偏大）。
        ic_v = ic.replace([np.inf, -np.inf], np.nan).dropna()
        _inf = mean_inference(ic_v, robust=False)
        t = _inf["t_stat"] if _inf["n"] >= 2 else 0.0
        rows.append({"feature": name, "ic_mean": ic.mean(), "ic_ir": ic.mean() / ic.std(ddof=1),
                     "t": t, "n_days": len(ic)})
    return pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)


def main():
    ap = argparse.ArgumentParser(description="分钟长表 → MemMap 稠密面板（+可选统计特征/IC快照）")
    ap.add_argument("--mock", action="store_true", help="合成数据端到端演示")
    ap.add_argument("--real", action="store_true", help="先经增量缓存拉数再物化（默认只读缓存）")
    ap.add_argument("--period", type=int, default=5, choices=[1, 3, 5, 10, 15, 30, 60, 120])
    ap.add_argument("--universe", default=None, help="池名（默认 config.universe.default）")
    ap.add_argument("--begin", type=int, default=None, help="特征提取起始日（面板构建始终全年份）")
    ap.add_argument("--end", type=int, default=None, help="特征提取截止日")
    ap.add_argument("--features", action="store_true", help="提取统计特征并落盘长表 parquet")
    ap.add_argument("--demo-ic", action="store_true", help="打印特征 vs 次日收益 IC 快照")
    args = ap.parse_args()

    pool = args.universe or Config.universe().get("default", "hs300")
    cache_root = None

    if args.mock:
        import tempfile
        cache_root = Path(tempfile.mkdtemp(prefix="mock_intraday_"))
        minute = gen_mock_minute(period=args.period)
        log.info("mock 分钟长表 %d 行（临时根 %s）", len(minute), cache_root)
    else:
        if args.real:
            cache = _real_cache()
        else:
            from data.offline import OfflineDataSource
            cache = DataCache(OfflineDataSource())
        minute = cache.read_minute_kline(pool, args.period)
        if minute is None or minute.empty:
            raise SystemExit(
                f"min{args.period}_{pool}.parquet 缓存不存在——先运行 "
                f"`python -m scripts.ingest.update_data` 拉取分钟数据。")
        log.info("读取缓存长表 %d 行", len(minute))

    store = MinutePanelStore.build(minute, args.period, pool, cache_root=cache_root)
    print(store.coverage_stats().to_string(index=False))

    if not (args.features or args.demo_ic):
        log.info("完成（面板根: %s）。统计特征可用 --features 提取。", store.root)
        return

    log.info("提取统计特征（%d 个）...", len(FEATURE_DOCS))
    feats = extract_all(store, begin_date=args.begin, end_date=args.end)
    log.info("特征面板: %d 个 × %s", len(feats),
             next(iter(feats.values())).shape if feats else (0, 0))

    if args.features:
        frames = []
        for name, p in feats.items():
            s = p.stack(future_stack=True)
            frames.append(pd.DataFrame({
                "date": s.index.get_level_values(0),
                "code": s.index.get_level_values(1),
                "feature": name,
                "value": s.to_numpy(),
            }))
        long = pd.concat(frames, ignore_index=True)
        out = store.root / "features_long.parquet"
        long.to_parquet(out, compression="snappy")
        log.info("特征长表落盘: %s（%d 行）", out, len(long))

    if args.demo_ic:
        if args.mock:
            log.info("--mock 下跳过 IC（无日线缓存）")
        else:
            ic = demo_ic(feats, cache, pool)
            with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
                print(ic.head(20).to_string(index=False))


def _real_cache() -> "DataCache":
    from data.datasource import create_datasource
    return DataCache(create_datasource())


if __name__ == "__main__":
    main()
