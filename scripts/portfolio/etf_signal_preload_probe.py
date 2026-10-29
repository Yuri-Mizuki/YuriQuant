"""
ETF 信号链「预加载」路径耗时探针
===================================

模式 B（盘中出信号）的核心工程方案：**盘前把 panels 落 pickle，盘中只加载不重算**。
本探针测量这条路径的两端：

  ``build``  一次完整装载 + 建 panels + 落 pickle（盘前 08:30 跑一次）
  ``load``   **新进程**加载 pickle + 重跑全样本回测（盘中真正要做的）

为什么必须拆两个进程：pickle 的收益正是「跳过 ``load_bundle``/``build_panels``」，
同进程测会被已暖的 import 和内存态污染。代价是两次 login（SDK 单账号互斥，**不能并行**）。

参照数字（2026-10-29，50 只 × 2136 日）：build 23.3 秒（其中 ``get_backward_factor`` 20 秒）、
dump 0.03 秒 / 26.7 MB；load 端 login+universe 3.2 秒 + pickle.load 0.009 秒 +
``run_variant`` 1.7 秒 = **wall 4.9 秒**。加上快照拉取 53~71 秒 ⇒ 盘中全链 ≈ 61 秒。

用法::

    python -m scripts.portfolio.etf_signal_preload_probe build   # 盘前
    python -m scripts.portfolio.etf_signal_preload_probe load    # 盘中

⚠️ `EtfUniverse` 含 module 引用、不可 pickle，故 load 端重建它（读日线 0.08 秒）。
⚠️ 本探针**只测耗时**，未实现「用快照价替换当日行」——那要乘 T-1 复权因子
（快照是原始价，panels 是后复权），属改造本体。
"""
from __future__ import annotations

import pickle
import sys
import tempfile
import time
from pathlib import Path

#: pickle 落盘位置。放系统临时目录：它是**可重建的中间态**不是实验证据，
#: 且 tracked 脚本不得硬编码盘符（`tests/test_no_hardcoded_drive_paths.py`）。
PK = Path(tempfile.gettempdir()) / "yuriquant_panels_cache.pkl"
mode = sys.argv[1]

if mode == "build":
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
    )

    t0 = time.perf_counter()
    t = time.perf_counter()
    cache = DataCache(create_datasource())
    print(f"login {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    print(f"load_bundle {time.perf_counter() - t:.2f}s")

    uni, close, amount, bench = bundle
    raw_close = (cache.read_daily("etf")["close"].unstack("code")
                 .reindex(index=close.index, columns=close.columns))
    t = time.perf_counter()
    panels = build_extra_panels(close, build_panels(close, amount), raw_close)
    print(f"panels {time.perf_counter() - t:.2f}s")

    spec = dict(PROFILES["contest3"])
    for c in EXCLUDED_5:
        panels["mask"].loc[:, c] = False
    extra = pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})
    print(f"build total {time.perf_counter() - t0:.1f}s")

    PK.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    with open(PK, "wb") as f:
        pickle.dump({"close": close, "amount": amount, "bench": bench,
                     "raw_close": raw_close, "panels": panels, "extra": extra,
                     "spec": spec}, f, protocol=5)
    print(f"dump {time.perf_counter() - t:.2f}s  {PK.stat().st_size / 1e6:.1f}MB")

else:
    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.etf_universe import EtfUniverse
    from scripts.portfolio.etf_rotation_probe import run_variant

    t0 = time.perf_counter()
    t = time.perf_counter()
    cache = DataCache(create_datasource())
    uni = EtfUniverse(cache)
    print(f"login+universe {time.perf_counter() - t:.2f}s")
    t = time.perf_counter()
    with open(PK, "rb") as f:
        d = pickle.load(f)
    t_load = time.perf_counter() - t
    t = time.perf_counter()
    res = run_variant((uni, d["close"], d["amount"], d["bench"]),
                      d["panels"], d["spec"], None, None, extra_close=d["extra"])
    print(f"pickle.load {t_load:.3f}s | run_variant {time.perf_counter() - t:.2f}s "
          f"| asof {res['rebalance_dates'][-1].date()} "
          f"| wall {time.perf_counter() - t0:.2f}s")
