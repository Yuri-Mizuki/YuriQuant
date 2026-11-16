"""日内因子日频组合探针 —— 期限对齐：h1 信号 × 持有 1 天。

**与既有判负实验的差别（这是本脚本存在的理由）**：
项目里所有「日频」负结论（`-52pp/年`、`freq_tune -22.2%`、`h1 组合 -17.32%`）
测的都是 **月频信号每日调仓**（期限错配）。本脚本用 **日频信号 + 日频调仓**，
让 `h1` 的 IC 与 1 天持有期真正对齐。

口径
----
- 池：`hs300_2022_2025` 库的 339 只（静态快照，**含幸存者偏差**，只做可行性判断）
- 信号：`im_*` 日内因子，T 日收盘可得（当日 48 根 5min bar 算出）
- 执行：**T+1 开盘**买入，**T+2 开盘**卖出（持有 1 天；T+1 制度允许 T+2 卖）
- 成本：往返 27bp（13.5bp 单边）× 换手 `Σ|Δw|/2`
- 权重：等权 TopN，可选 buffer 降换手

用法:
    python -m scripts.evaluation.intraday_daily_probe --factor im_mom_last_bar
    python -m scripts.evaluation.intraday_daily_probe --scan-top
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(r"E:/data/factor_library/hs300_2022_2025")
DAILY = Path(r"E:/data/parquet/daily_all_a.parquet")
OUT = Path(r"E:/YuriQuant/reports/intraday_daily")

ONE_WAY_BP = 13.5           # 27bp 往返 / 2


def load_returns(period_days: int = 1) -> pd.DataFrame:
    """信号日 T → 执行 T+1 open 买 → T+(1+period) open 卖的收益。

    🚨 **前视边界（2026-11-16 踩坑，本函数第一版写错）**：
    信号 T 依赖 T 日全部 48 根 5min bar，**T 日收盘才可得**。所以买入价必须是
    **T+1 的开盘**，不能用 T 日开盘（那已经过去了）。

    错写 `op.shift(-period)/op - 1` 再 `.shift(1)` 会把「T-1 收盘信号 → T-1→T
    的隔夜收益」配对起来，而那段收益在 T-1 开盘后就锁定了 ⇒ 相当于把
    「信号日前一日的跌幅」记成可赚收益，是纯前视。

    正确写法：索引 T 上放 `op[T+1+period] / op[T+1] - 1`。
    """
    px = pd.read_parquet(DAILY, columns=["open"])
    op = px["open"].unstack("code").sort_index()
    buy = op.shift(-1)                    # T+1 open（执行价）
    sell = op.shift(-(1 + period_days))   # T+1+period open（平仓价）
    return (sell / buy - 1.0).replace([np.inf, -np.inf], np.nan)


def load_panel(name: str) -> pd.DataFrame:
    reg = pd.read_csv(LIB / "registry.csv")
    row = reg[reg["name"] == name]
    if row.empty:
        raise KeyError(f"未知因子 {name!r}")
    return pd.read_parquet(row.iloc[0]["panel_path"])


def run(sig: pd.DataFrame, fwd: pd.DataFrame, top: int = 20,
        buffer_mult: float = 1.0, cost_bp: float = ONE_WAY_BP,
        min_names: int = 30, higher_is_better: bool = False) -> dict:
    """日频等权 TopN + 可选 buffer。

    ``higher_is_better=False``（默认）：取**最小**的一端 —— 90/95 个 `im_*` 因子
    IC 为负（日内反转），信号值小 = 后续涨。
    复合因子若已统一成「越大越好」，必须传 ``True``，否则方向会被翻两次
    （2026-11-16 踩坑：复合臂全期 −47%，比单因子 −1.9 SR 还差）。
    """
    idx = sig.index.intersection(fwd.index)
    s, r = sig.loc[idx], fwd.loc[idx]
    codes = s.columns.intersection(r.columns)
    s, r = s[codes], r[codes]

    prev: set[str] = set()
    rets, turns, nhold = [], [], []
    for dt in idx:
        sv = s.loc[dt].dropna()
        if len(sv) < min_names:
            rets.append(np.nan)
            turns.append(0.0)
            nhold.append(0)
            continue
        # 方向：IC 为负的因子「信号越小越好」；复合因子已统一方向则传 True
        asc = not higher_is_better
        rank = sv.rank(ascending=asc, method="first")
        head = rank.sort_values(ascending=True)      # 名次 1 = 最优
        keep = head[head <= top * buffer_mult].index
        # 缓冲：上期持仓只要还在缓冲区内就保留
        held = [c for c in prev if c in keep]
        new = [c for c in head.index if c not in held]
        picked = (held + new)[:top]
        if not picked:
            rets.append(np.nan)
            turns.append(0.0)
            nhold.append(0)
            continue

        w = pd.Series(1.0 / len(picked), index=picked)
        wp = (pd.Series(1.0 / len(prev), index=sorted(prev))
              if prev else pd.Series(dtype=float))
        allc = w.index.union(wp.index)
        dw = w.reindex(allc, fill_value=0) - wp.reindex(allc, fill_value=0)
        turn = float(dw.abs().sum() / 2)
        rr = r.loc[dt, picked].dropna()
        rets.append(float(rr.mean()) if len(rr) else np.nan)
        turns.append(turn)
        nhold.append(len(picked))
        prev = set(picked)

    net = pd.Series(rets, index=idx) - pd.Series(turns, index=idx) * cost_bp / 1e4
    gross = pd.Series(rets, index=idx)
    return {"gross": gross.dropna(), "net": net.dropna(),
            "turn": pd.Series(turns, index=idx), "nhold": pd.Series(nhold, index=idx)}


def stats(x: pd.Series, ppy: int = 252) -> dict:
    x = x.dropna()
    if len(x) < 30:
        return {}
    nav = (1 + x).cumprod()
    cagr = nav.iloc[-1] ** (ppy / len(x)) - 1
    vol = x.std(ddof=1) * np.sqrt(ppy)
    dd = (nav / nav.cummax() - 1).min()
    return {"ann": cagr, "vol": vol, "sr": x.mean() / x.std(ddof=1) * np.sqrt(ppy),
            "mdd": dd, "n": len(x), "calmar": cagr / abs(dd) if dd < 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", default="im_mom_last_bar")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--buffer", type=float, default=1.0)
    ap.add_argument("--scan-top", action="store_true")
    ap.add_argument("--scan-buffer", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fwd = load_returns(1)
    sig = load_panel(args.factor)

    if args.scan_top:
        rows = []
        for bm in (1.0, 1.5, 2.0, 3.0):
            for top in (10, 20, 30, 50, 100):
                res = run(sig, fwd, top=top, buffer_mult=bm)
                g, n = stats(res["gross"]), stats(res["net"])
                if not g or not n:
                    continue
                rows.append({"buffer": bm, "top": top,
                             "gross_ann": g["ann"], "net_ann": n["ann"],
                             "sr": n["sr"], "mdd": n["mdd"],
                             "turn": float(res["turn"].mean()),
                             "hold": float(res["nhold"].replace(0, np.nan).mean())})
        t = pd.DataFrame(rows)
        t.to_csv(OUT / f"scan_{args.factor}.csv", index=False, encoding="utf-8-sig")
        pd.set_option("display.width", 200)
        print(f"=== {args.factor} · 日频调仓 · T+1 open→T+2 open · 27bp 往返 ===")
        print(t.to_string(index=False, float_format=lambda x: "%.4f" % x))
        return

    res = run(sig, fwd, top=args.top, buffer_mult=args.buffer)
    g, n = stats(res["gross"]), stats(res["net"])
    print(f"=== {args.factor} top={args.top} buffer={args.buffer} ===")
    print(f"  单边换手/日 {res['turn'].mean():.3f}  年均换手 {res['turn'].mean()*252:.1f}")
    print(f"  毛年化 {g['ann']*100:+.2f}%  净年化 {n['ann']*100:+.2f}%  "
          f"SR {n['sr']:.3f}  MDD {n['mdd']*100:.2f}%  Calmar {n['calmar']:.2f}")
    print(f"  成本拖累 {(g['ann']-n['ann'])*100:.2f}pp/年")
    json.dump({"factor": args.factor, "gross": g, "net": n,
               "turn_mean": float(res["turn"].mean())},
              open(OUT / "last_run.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
