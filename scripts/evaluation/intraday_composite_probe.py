"""im_ 因子复合的方向纠错 —— 上一轮「复合更差」是实现 bug 还是真结论。

背景
----
`reports/intraday_daily/` §2.4 记录：10 个高 IR 因子 rank 平均后，毛年化从
**55.5% 掉到 14.0%**，结论写成「信息同质：95 个 im_* 里 71 个显著，但全是
同一件事（日内反转）的不同切法，最强那个已把信息吃干净，平均进去只加噪声」。

**这个解释站不住**，本脚本给三条反证：

1. **n_eff = 22.14（23.3%）**：95 个因子的有效自由度不是 1，是有 22 个独立方向；
   PC1 只解释 13.1% 方差。（对照：全库 924 因子 n_eff 仅 5.73）
2. **方向不止一侧**：71 个显著因子里 **44 个 IC<0（反转）、27 个 IC>0（延续）**。
   上一轮复合按 `ic` 升序取前 10 ⇒ **全部落在反转侧**，把 27 个延续方向的因子
   （`im_vol_seg*_rvshare` / `im_pv_seg*_volshare` / `im_candle_upper_shadow` /
   `im_pv_zero_ret_share`）全丢了。
3. **选法本身有偏**：按 ``|IC|`` 排序偏向**数值大**，按 ``t`` 排序才是统计强度。
   两者重合仅 6/10；``im_pv_zero_ret_share``（IC 0.044 / t 6.56）按 |IC| 进前十，
   按 |t| 进不了。

正确做法
--------
先把每个因子按 IC 符号定向（``sign(ic) × 截面 rank``），让所有因子「越大越好」，
再 rank 平均。这样延续与反转两侧都能贡献，不会相互抵消。

对照臂（同 pool / 同 top / 同 buffer / 同成本）
---------------------------------------------
- `single`：最强单因子（复现 55.5%/14.4% 的基线）
- `wrong_side10`：上一轮的选法（按 ``ic`` 升序前 10，全落在反转侧）⇒ **复现 bug**
- `both_dir10`：按 ``|t|`` 前 10、**逐因子定向**后平均 ⇒ 正确做法
- `both_dir_all71`：全部 71 个显著因子定向后平均 ⇒ 覆盖上限

用法:
    python -m scripts.evaluation.intraday_composite_probe --arms all
    python -m scripts.evaluation.intraday_composite_probe --arms both_dir10 --top 10
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
    """T+1 open 买 → T+1+period open 卖。🚨 前视边界见 intraday_daily_probe。"""
    px = pd.read_parquet(DAILY, columns=["open"])
    op = px["open"].unstack("code").sort_index()
    buy = op.shift(-1)
    sell = op.shift(-(1 + period_days))
    return (sell / buy - 1.0).replace([np.inf, -np.inf], np.nan)


def load_ic_table() -> pd.DataFrame:
    """逐因子 h1 IC（直接从 evals 读，registry 的 ic_mean_h1 全为空）。"""
    reg = pd.read_csv(LIB / "registry.csv")
    im = reg[reg["name"].str.startswith("im_")].copy()
    rows = []
    for _, r in im.iterrows():
        d = pd.read_parquet(r["eval_path"])
        ic = d["ic"].dropna()
        if len(ic) < 100 or ic.std(ddof=1) == 0:
            continue
        rows.append({"name": r["name"], "n": len(ic), "ic": ic.mean(),
                     "t": ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))})
    return pd.DataFrame(rows)


def load_panel(name: str) -> pd.DataFrame:
    reg = pd.read_csv(LIB / "registry.csv")
    row = reg[reg["name"] == name]
    if row.empty:
        raise KeyError(name)
    return pd.read_parquet(row.iloc[0]["panel_path"])


def orient(panel: pd.DataFrame, ic: float) -> pd.DataFrame:
    """按 IC 符号定向：IC<0 的取负 ⇒ 之后一律「越大越好」。

    🚨 上一轮的 bug 就在这里：只取 rank 最小端 + 复合时再乘 −1 ⇒ 方向被翻两次。
    正解是**每个因子各自定向**，复合后统一 `higher_is_better=True`。
    """
    return -panel if ic < 0 else panel


def composite(names: list[str], ics: dict[str, float],
              min_panels: int = 2) -> pd.DataFrame:
    """定向后逐日截面 rank 平均。全 NaN 因子自动降级（按可用数平均）。"""
    panels = [orient(load_panel(n), ics[n]) for n in names]
    idx = panels[0].index
    cols = panels[0].columns
    for p in panels[1:]:
        idx = idx.intersection(p.index)
        cols = cols.intersection(p.columns)
    num = None
    cnt = None
    for p in panels:
        r = p.loc[idx, cols].rank(axis=1, pct=True)
        num = r if num is None else num.add(r, fill_value=0)
        c = r.notna().astype(np.int16)
        cnt = c if cnt is None else cnt.add(c)
    cnt = cnt.where(cnt >= min_panels)
    return (num / cnt).where(cnt.notna())


def run(sig: pd.DataFrame, fwd: pd.DataFrame, top: int = 10,
        buffer_mult: float = 2.0, cost_bp: float = ONE_WAY_BP,
        min_names: int = 30, higher_is_better: bool = True) -> dict:
    """日频等权 TopN + buffer。复合后默认 higher_is_better=True。"""
    idx = sig.index.intersection(fwd.index)
    s, r = sig.loc[idx], fwd.loc[idx]
    codes = s.columns.intersection(r.columns)
    s, r = s[codes], r[codes]
    asc = not higher_is_better

    prev: set[str] = set()
    rets, turns, nhold = [], [], []
    for dt in idx:
        sv = s.loc[dt].dropna()
        if len(sv) < min_names:
            rets.append(np.nan)
            turns.append(0.0)
            nhold.append(0)
            continue
        rank = sv.rank(ascending=asc, method="first")
        head = rank.sort_values(ascending=True)
        keep = head[head <= top * buffer_mult].index
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
    gross = pd.Series(rets, index=idx)
    net = gross - pd.Series(turns, index=idx) * cost_bp / 1e4
    return {"gross": gross.dropna(), "net": net.dropna(),
            "turn": pd.Series(turns, index=idx)}


def stats(x: pd.Series, ppy: int = 252) -> dict:
    x = x.dropna()
    if len(x) < 30:
        return {}
    nav = (1 + x).cumprod()
    cagr = nav.iloc[-1] ** (ppy / len(x)) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"ann": cagr, "sr": x.mean() / x.std(ddof=1) * np.sqrt(ppy),
            "mdd": dd, "n": len(x),
            "calmar": cagr / abs(dd) if dd < 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--buffer", type=float, default=2.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ic = load_ic_table()
    ic.to_csv(OUT / "im_h1_ic.csv", index=False, encoding="utf-8-sig")
    ics = dict(zip(ic["name"], ic["ic"], strict=True))
    sig_ic = ic[ic["t"].abs() > 1.96]
    print(f"95 个 im_ 因子：显著 {len(sig_ic)} 个 "
          f"(IC<0 {int((sig_ic['ic']<0).sum())} / IC>0 {int((sig_ic['ic']>0).sum())})")

    fwd = load_returns(1)
    arms: dict[str, tuple[pd.DataFrame, bool]] = {}

    best = ic.reindex(ic["t"].abs().sort_values(ascending=False).index).iloc[0]["name"]
    arms["single"] = (orient(load_panel(best), ics[best]), True)

    # 上一轮选法：按 ic 升序前 10（全落反转侧），且不逐因子定向 ⇒ 复现 bug
    wrong = ic.sort_values("ic").head(10)["name"].tolist()
    arms["wrong_side10"] = (composite(wrong, ics), True)

    t10 = ic.reindex(ic["t"].abs().sort_values(ascending=False).index).head(10)["name"].tolist()
    arms["both_dir10"] = (composite(t10, ics), True)

    all71 = sig_ic["name"].tolist()
    arms["both_dir_all71"] = (composite(all71, ics), True)

    want = list(arms) if args.arms == "all" else args.arms.split(",")
    rows = []
    for name in want:
        if name not in arms:
            continue
        sig, hib = arms[name]
        res = run(sig, fwd, top=args.top, buffer_mult=args.buffer,
                  higher_is_better=hib)
        g, n = stats(res["gross"]), stats(res["net"])
        if not g or not n:
            continue
        rows.append({"arm": name, "n_factors": 1 if name == "single" else
                     len(wrong) if name == "wrong_side10" else
                     len(t10) if name == "both_dir10" else len(all71),
                     "gross_ann": g["ann"], "net_ann": n["ann"],
                     "sr": n["sr"], "mdd": n["mdd"],
                     "turn_day": float(res["turn"].mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "composite_arms.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"\n=== 复合方向对照 · top{args.top} buffer{args.buffer} · 27bp ===")
    print(t.to_string(index=False, float_format=lambda x: "%.4f" % x))
    json.dump({"top": args.top, "buffer": args.buffer,
               "wrong_side10": wrong, "both_dir10": t10},
              open(OUT / "composite_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()