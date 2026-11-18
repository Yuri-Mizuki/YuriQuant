"""调仓频率 × 持有期 **全扫描**（PIT 动态池）
================================================

问题
----
`pit_weekly_probe` 只测了 4 个点 ``H ∈ {1,5,10,20}``，且每个 H 只取一个因子。
「频率-收益曲线到底什么形状」和「H=17 那种单点高 SR 是真信号还是相位幸运」
都答不了。

本脚本做三件事：

1. **H = 1..20 全扫**，每个 H 跑满 H 个调仓相位 ⇒ 消除相位幸运
   （`off in range(0,H)`，`pit_weekly_probe.run_periodic` 只吃单一相位）
2. **两种选因子方式对照**，这是本脚本最重要的一件事：

   - ``--select best``：每个 H 各自挑 ``|t|`` 最大的因子（**有选择偏差**）
   - ``--select fixed``：固定同一因子，只变 H（**无选择偏差**）

   🚨 **两者的结论会彻底相反**。上一轮只跑 ``best`` 得到「U 形曲线」
   （H=1 峰 → H=2~5 死亡谷 → H=14~20 高原），固定单因子后高原整体消失：
   `mom_last_bar` 在 H≥4 全部回到 0 附近。**高原是每档换因子换出来的**。
   同一张图既可以读成「长持有期有 alpha」，也可以读成「短持有期因子
   在长持有期全废、只有另一族因子在长持有期勉强活着」，
   ``best`` 方式分不清这两件事，所以**结论只许引用 fixed 列**。

3. **毛/成本/净三段分解**：净是二者之差，缺了分段就看不出「是信号掉了
   还是成本涨了」。

为什么 ``t/√h`` 必须一起看
--------------------------
H 日收益序列逐日采样时相邻样本重叠 ``H-1`` 天 ⇒ 有效独立样本只有 ``n/H``，
``t`` 会被机械放大 ``√H``。所以横比不同 H 的 ``|t|`` **不能直接比**，
``|t|/√H`` 才是可比的量级。实测 ``evt_down_tail_share``：
``t`` 5.40→12.02（H=1→20），但 ``t/√H`` **5.40→2.69 反而降**。
⇒ 用 ``|t|`` 挑因子会**系统性偏向长 H**，这正是 ``best`` 方式产假 U 形的机制。

口径
----
- 池：PIT 动态池（`load_context`，每日在册）
- 收益：``fwd[h] = op.shift(-(1+h))/op.shift(-1) - 1``（T+1 open 买 → T+1+h open 卖）
- 调仓：``idx[::h]``，持有 h 日；每个相位 ``off ∈ [0,h)`` 各跑一次
- 年化：``ppy = 252/h``（`stats_periodic`）
- 成本：单边 13.5bp（往返 27bp），按调仓日换手扣

用法::

    .venv/Scripts/python.exe -m scripts.evaluation.pit_freq_sweep
    .venv/Scripts/python.exe -m scripts.evaluation.pit_freq_sweep --select fixed
    .venv/Scripts/python.exe -m scripts.evaluation.pit_freq_sweep --decompose

⚠️ 特征缓存 ``xs_nl_feats.pkl`` 由 **pandas 3.0.5（venv）** 写出，
系统 Python 3.12 的 pandas 2.3.3 **读不了**（``datetime64[us]`` 未实现）
⇒ 本脚本必须用 ``.venv`` 跑。
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

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation.pit_intraday_probe import (  # noqa: E402
    OUT,
    ic_stats,
    load_context,
    orient,
)
from scripts.evaluation.pit_weekly_probe import (  # noqa: E402
    daily_ic,
    fwd_returns,
    run_periodic,
    stats_periodic,
)

# 长 H 高原的贡献者，与 mom_last_bar 分开报（方向相反，混在一张表会误导）
FIXED_DEFAULT = ("mom_last_bar", "evt_down_tail_share", "vol_extreme_bar_share")

log = setup_logging("pit_freq_sweep")


def load_feats(store, fwds: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """读日内特征缓存，并按可交易池对齐列。"""
    path = OUT / "xs_nl_feats.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"缺特征缓存 {path}；先跑 pit_xs_nl_probe 生成（约 17 分钟）"
        )
    feats = pd.read_pickle(path)
    pool = [c for c in store.codes if c in fwds[min(fwds)].columns]
    return {k: v.reindex(columns=pool) for k, v in feats.items()}


def sweep_one(p: pd.DataFrame, fwds: dict[int, pd.DataFrame], h: int,
              top: int = 10) -> dict | None:
    """固定因子、固定 H，跑满该 H 的**全部相位**，返回均值统计。"""
    ic = ic_stats(daily_ic(p, fwds[h]))
    if not ic:
        return None
    sig = orient(p, float(ic["ic"]))
    g, n, s, turns = [], [], [], []
    for off in range(0, h):
        r = run_periodic(sig.iloc[off:], fwds[h].iloc[off:], hold=h, top=top)
        sg = stats_periodic(r["gross"], h)
        sn = stats_periodic(r["net"], h)
        if sg and sn:
            g.append(sg["ann"])
            n.append(sn["ann"])
            s.append(sn["sr"])
            turns.append(float(r["turn"].mean()) * 252 / h)
    if not n:
        return None
    g, n, s = np.array(g), np.array(n), np.array(s)
    return {
        "H": h, "ic": ic["ic"], "t": ic["t"],
        "t_over_sqrt_h": ic["t"] / np.sqrt(h),
        "gross_mean": g.mean(), "cost_mean": (g - n).mean(),
        "net_mean": n.mean(), "net_std": n.std(), "net_min": n.min(),
        "net_max": n.max(), "sr_mean": s.mean(), "sr_std": s.std(),
        "turn": float(np.mean(turns)) if turns else np.nan,
        "n_phase": len(n),
    }


def run_best(feats: dict[str, pd.DataFrame], fwds: dict[int, pd.DataFrame],
             horizons: list[int], top: int) -> pd.DataFrame:
    """**有选择偏差**的对照组：每个 H 挑 |t| 最大的因子。"""
    rows = []
    for h in horizons:
        ics = {n: ic_stats(daily_ic(p, fwds[h])) for n, p in feats.items()}
        ics = {n: v for n, v in ics.items() if v}
        if not ics:
            continue
        best = max(ics.items(), key=lambda kv: abs(kv[1]["t"]))
        r = sweep_one(feats[best[0]], fwds, h, top)
        if r:
            r["factor"] = best[0]
            rows.append(r)
    cols = ["H", "factor", "ic", "t", "t_over_sqrt_h", "gross_mean", "cost_mean",
            "net_mean", "net_std", "net_min", "net_max", "sr_mean", "sr_std",
            "turn", "n_phase"]
    return pd.DataFrame(rows)[cols]


def run_fixed(feats: dict[str, pd.DataFrame], fwds: dict[int, pd.DataFrame],
              names: tuple[str, ...], horizons: list[int],
              top: int) -> pd.DataFrame:
    """**无选择偏差**：固定单因子，只变 H。结论只许引用这张。"""
    rows = []
    for name in names:
        if name not in feats:
            print(f"  [skip] 缓存里没有 {name}")
            continue
        for h in horizons:
            r = sweep_one(feats[name], fwds, h, top)
            if r:
                r["factor"] = name
                rows.append(r)
    cols = ["factor", "H", "ic", "t", "t_over_sqrt_h", "gross_mean", "cost_mean",
            "net_mean", "net_std", "net_min", "net_max", "sr_mean", "sr_std",
            "turn", "n_phase"]
    return pd.DataFrame(rows)[cols]


def _print(df: pd.DataFrame, tag: str) -> None:
    print(f"\n=== {tag}（21 相位 / 该 H 的全相位）===")
    print("  %-22s %-4s %8s %8s %8s %8s %8s %7s %6s" % (
        "factor", "H", "IC", "毛", "成本", "净", "net min", "SR均", "换手"))
    for _, r in df.iterrows():
        print("  %-22s %-4d %+8.4f %+7.2f%% %+7.2f%% %+7.2f%% %+7.2f%% %+7.3f %6.1f" % (
            str(r["factor"])[:22], r["H"], r["ic"],
            r["gross_mean"] * 100, r["cost_mean"] * 100,
            r["net_mean"] * 100, r["net_min"] * 100, r["sr_mean"], r["turn"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="调仓频率 × 持有期全扫描")
    ap.add_argument("--select", default="both", choices=["both", "best", "fixed"])
    ap.add_argument("--h-max", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--fixed", default=",".join(FIXED_DEFAULT),
                    help="fixed 模式下固定的因子名，逗号分隔")
    args = ap.parse_args()

    setup_logging("pit_freq_sweep")
    horizons = list(range(1, args.h_max + 1))

    store, _codes, daily, _ = load_context()
    fwds = fwd_returns(daily, horizons)
    feats = load_feats(store, fwds)

    if args.select in ("both", "best"):
        best = run_best(feats, fwds, horizons, args.top)
        best.to_csv(OUT / "pit_freq_sweep_phase.csv", index=False, encoding="utf-8-sig")
        _print(best, "best：每个 H 各挑 |t| 最大因子（有选择偏差，勿引结论）")
        print("  已写 pit_freq_sweep_phase.csv")

    if args.select in ("both", "fixed"):
        names = tuple(s.strip() for s in args.fixed.split(",") if s.strip())
        fixed = run_fixed(feats, fwds, names, horizons, args.top)
        fixed.to_csv(OUT / "pit_freq_fixed_phase.csv", index=False, encoding="utf-8-sig")
        _print(fixed, "fixed：固定单因子只变 H（结论以此为准）")
        print("  已写 pit_freq_fixed_phase.csv")

        # t/√H：横比不同 H 唯一可比的量级
        print("\n=== |t|/√H（消重叠膨胀；横比不同 H 只能看这个）===")
        piv = fixed.pivot(index="H", columns="factor", values="t_over_sqrt_h")
        print(piv.round(2).to_string())


if __name__ == "__main__":
    main()