"""
ETF 轮动信号链端到端耗时探针
================================

用途：判断「盘中出信号」改造（模式 B）的时间预算是否够用。14:55 决策点只剩
120 秒到 14:57 收盘集合竞价，14:50 决策点有 420 秒 —— 但**耗时本身不随决策点
变化**，变的是预算。本探针把生产信号链（``etf_champ_signals.main`` 的完整路径）
按段打点，给出冷启动 / 热数据 / 尾部增量三种口径。

分段（与生产代码一一对应）：

  s0 login        ``create_datasource()`` → ``AmazingDataSource._login``
  s1 bundle       ``load_bundle``：全样本复权收盘 + 成交额 + 基准（2018+，暖机）
  s2 raw_close    第二次 ``cache.read_daily("etf")`` 取未复权收盘
  s3 panels       ``build_panels`` + ``build_extra_panels``（全样本因子）
  s4 backtest     ``run_variant``：全样本 2018–2026 周频回测（最重）
  s5 output       出榜 + HTML

  s1n/s2n/s3n/s4n  同段在**同进程第二次**执行（OS 文件缓存已热）的耗时
  tail            尾部增量：panels 已在内存，只重算 sig_date 一天的因子 +
                  ``strategy.get_weights_at``（盘中真正需要的部分）

用法::

    python -m scripts.portfolio.etf_signal_latency                 # 默认 contest3
    python -m scripts.portfolio.etf_signal_latency --profile champ --json out.json

⚠️ 会把结果同时打印到 stdout 并（可选）落 JSON；不写 reports/ 以外的路径。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class _Stopwatch:
    """累计式分段计时（同一 key 多次 record 则累加，兼作次数统计）。"""

    def __init__(self):
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self._t0: float | None = None
        self._key: str | None = None

    def start(self, key: str):
        if self._key is not None:
            self.record(self._key)
        self._key = key
        self._t0 = time.perf_counter()

    def record(self, key: str):
        if self._t0 is None:
            return
        dt = time.perf_counter() - self._t0
        self.t[key] = self.t.get(key, 0.0) + dt
        self.n[key] = self.n.get(key, 0) + 1
        self._t0 = None
        self._key = None

    def as_dict(self) -> dict:
        return {k: {"sec": round(v, 3), "n": self.n.get(k, 1)}
                for k, v in self.t.items()}


def run(profile: str = "contest3") -> dict:
    sw = _Stopwatch()
    import pandas as pd

    from data.cache import DataCache
    from data.datasource import create_datasource
    from scripts.portfolio.etf_champ_signals import PROFILES
    from scripts.portfolio.etf_contest_spec import EXCLUDED_5
    from scripts.portfolio.etf_rotation_panels import (
        build_defense,
        build_extra_panels,
        level_from_returns,
    )
    from scripts.portfolio.etf_rotation_probe import (
        WARMUP_BEGIN,
        build_panels,
        load_bundle,
        run_variant,
    )

    spec = dict(PROFILES[profile])

    # ---- s0: 登录 ----
    sw.start("s0_login")
    cache = DataCache(create_datasource())
    sw.record("s0_login")

    # ---- s1/s2/s3/s4: 第一遍（冷） ----
    sw.start("s1_bundle")
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    sw.record("s1_bundle")
    uni, close, amount, _bench = bundle

    sw.start("s2_raw_close")
    raw_close = (cache.read_daily("etf")["close"].unstack("code")
                 .reindex(index=close.index, columns=close.columns))
    sw.record("s2_raw_close")

    sw.start("s3_panels")
    panels = build_extra_panels(close, build_panels(close, amount), raw_close)
    sw.record("s3_panels")
    excluded = list(EXCLUDED_5) if profile.startswith("contest") else []
    for c in excluded:
        panels["mask"].loc[:, c] = False

    extra = (pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})
             if spec.get("defense") else None)

    sw.start("s4_backtest")
    res = run_variant(bundle, panels, spec, None, None, extra_close=extra)
    sw.record("s4_backtest")
    first = dict(sw.t)

    # ---- s1n~s4n: 第二遍（热 OS 缓存 + 已暖 import），只用来看数据 IO 的占比 ----
    sw.start("s1n_bundle")
    bundle2 = load_bundle(cache, WARMUP_BEGIN, None)
    sw.record("s1n_bundle")
    sw.start("s2n_raw_close")
    raw2 = (cache.read_daily("etf")["close"].unstack("code")
            .reindex(columns=close.columns))
    sw.record("s2n_raw_close")
    sw.start("s3n_panels")
    panels2 = build_extra_panels(bundle2[1], build_panels(bundle2[1], bundle2[2]), raw2)
    sw.record("s3n_panels")
    sw.start("s4n_backtest")
    run_variant(bundle2, panels2, spec, None, None, extra_close=extra)
    sw.record("s4n_backtest")

    # ---- tail: 尾部增量（盘中真实需要的那一段） ----
    # 前提：T-1 收盘后已完整跑过一次，panels 在内存；盘中只把 sig_date 的收盘价
    # 换成实时快照价，重算该日因子行 + 取权重。
    sig_date = close.index[-1]
    sw.start("tail_factor_row")
    fac_now = panels[spec["signal"]].where(panels["mask"]).loc[sig_date].dropna()
    sw.record("tail_factor_row")

    sw.start("tail_weights")
    tw = res["strategy"].get_weights_at(sig_date, fac_now)
    sw.record("tail_weights")

    # 尾部增量里真正必须重算的**只有** pre10 那类依赖 T 日价的筛子；动量类
    # （shift(21)/shift(252)）在 14:50 也要重算，但它们只是一次向量化切片。
    sw.start("tail_pre10")
    pre10 = (close / close.shift(20) - 1.0).loc[sig_date]
    keep = set(pre10[~(pre10.abs() > 0.10)].index)
    sw.record("tail_pre10")

    return {
        "profile": profile,
        "asof": str(sig_date.date()),
        "n_codes": int(close.shape[1]),
        "n_days": int(close.shape[0]),
        "n_holdings": int((tw > 1e-9).sum()) if hasattr(tw, "sum") else None,
        "first_pass": {k: round(v, 3) for k, v in first.items()
                       if not k.endswith("_login")},
        "login_sec": round(sw.t.get("s0_login", 0.0), 3),
        "raw": sw.as_dict(),
        "pre10_kept": len(keep),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 信号链各段耗时")
    ap.add_argument("--profile", default="contest3",
                    choices=("contest3", "contest5", "contest5h", "champ"))
    ap.add_argument("--json", default=None, help="把结果写到该 JSON")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    out = run(args.profile)
    out["total_wall_sec"] = round(time.perf_counter() - t0, 2)

    raw = out["raw"]
    print("\n=== 第一遍（冷启动）===", flush=True)
    for k in ("s0_login", "s1_bundle", "s2_raw_close", "s3_panels", "s4_backtest"):
        if k in raw:
            print(f"  {k:14s} {raw[k]['sec']:8.2f}s", flush=True)
    print("\n=== 第二遍（OS 缓存已热）===", flush=True)
    for k in ("s1n_bundle", "s2n_raw_close", "s3n_panels", "s4n_backtest"):
        if k in raw:
            print(f"  {k:14s} {raw[k]['sec']:8.2f}s", flush=True)
    print("\n=== 尾部增量（panels 已在内存）===", flush=True)
    for k in ("tail_factor_row", "tail_pre10", "tail_weights"):
        if k in raw:
            print(f"  {k:14s} {raw[k]['sec']:8.4f}s", flush=True)
    print(f"\n  total_wall  {out['total_wall_sec']:.2f}s  "
          f"({out['n_codes']} 只 × {out['n_days']} 日, profile={out['profile']})",
          flush=True)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {p}", flush=True)


if __name__ == "__main__":
    main()
