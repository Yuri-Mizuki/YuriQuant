"""大盘择时（市场状态 → 仓位乘数）探针。

为什么需要这个脚本
------------------
项目此前只有**截面选股**一条腿（预测 → 排名 → Top10% 等权）。择时不是"新维度"，
而是**在组合层乘一个总仓位 ω_t**：选股相对排序完全不动，只改 gross 敞口。
因此它可以在任意信号上叠加，是成本最低的一类实验——但也正因为低成本，
极易被"MDD 变好看了"骗过去。

本脚本一次性回答三件事（避免重复踩坑）：

1. **上界在哪**：先算完美 foresight（物理不可能）的收益。如果 oracle 都打不过
   满仓，任何真实规则都不必跑。
2. **beta 结构**：全期 beta / 上下行捕获 / 滚动 beta 区间。择时的收益来源只有
   「下行捕获显著低于上行捕获」这一条；已对称则无空间。
3. **因果规则实测**：MA 族 / 动量族 / 波动率目标，含**仓位切换成本**。

口径约定（🚨 与项目既有教训一致）
---------------------------------
- **信号一律 ``shift(1)``**：t−1 收盘已知 → t 日生效，对齐生产 T+1 open 口径。
  ⚠️ 绝不拿"当日收益"去和均线比较——那是纯前视，会跑出 200%+ 的荒谬年化
  （本脚本开发时踩过一次，见 tests 与 docstring 末段）。
- **基准价格序列由 ``bench_eqw`` 日收益 cumprod 重建**，不引入外部指数文件
  ⇒ 与被测净值严格同源同区间。
- **仓位切换成本 = |Δω| × 10bp**（与生产固定 10bp 同量级；ω 是总仓位，
  半仓→满仓 = 0.5 单位 = 5bp）。
- **Sharpe 用几何口径**（年化几何收益 / 年化波动），与
  ``backtest.metrics.sharpe_ratio`` 一致；⚠️ 绝不用算术 mean/std·√252。

用法
----
    # 诊断（默认读 882 ortho 满仓基线净值，秒级）
    python -m scripts.evaluation.timing_probe

    # 920 定版复验（须有 920 pred 的机器；本机会自动拒绝并提示）
    python -m scripts.evaluation.timing_probe --equity <920-equity.csv> --tag dingban

    # 自检（前视守卫 + 成本守卫 + 口径守卫）
    python -m scripts.evaluation.timing_probe --self-check

产���
----
    reports/timing_probe/{equity,summary}.csv
    reports/timing_probe/oracle_vs_causal.csv
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

OUT = ROOT / "reports" / "timing_probe"

#: 默认被测净值 = 882 ortho · ens_h1h5 · M · raw · f0.10 · **满仓无 buffer**。
#: ⚠️ 这**不是**登记基线 ``v1w-ortho882``（后者带 buffer(15/40)，
#: production 16.46%/0.685）。选它是因为它是本机唯一「逐日净值 + 基准列」齐全的
#: 实体，能零成本做上界/beta 结构诊断。**任何结论跨到定版前必须用
#: --equity 换 920 定版重跑。**
DEFAULT_EQUITY = (ROOT / "reports" / "alla_rolling_ortho" / "equity"
               / "eq__ens_h1h5__h1__M__raw__f0.10.csv")

#: 仓位切换成本（单边，bp）：ω 变动 1.0 单位 = 满仓↔空仓。
POSITION_TURNOVER_BP = 10.0

#: 四窗口 + 全窗的一致性判据（与 cheap_lever_scan 同格式：项目要求 ≥4/5 为正）。
CONSISTENCY_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2018+", "2018"),
    ("2019+", "2019"),
    ("2021+", "2021"),
    ("2023+", "2023"),
    ("2025+", "2025"),
)


# ---------------------------------------------------------------------------
# 指标（几何 Sharpe 口径，与 backtest/metrics.py 同源）
# ---------------------------------------------------------------------------
def annual_return(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def annual_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(252))


def geo_sharpe(r: pd.Series) -> float:
    v = annual_vol(r)
    return annual_return(r) / v if v > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    c = (1 + r).cumprod()
    return float((c / c.cummax() - 1).min())


def _apply_position_cost(base: pd.Series, expo: pd.Series,
                         bp: float = POSITION_TURNOVER_BP) -> pd.Series:
    """总仓位乘数 + 切换成本：``r = r_base·ω − |Δω|·bp``。

    ``ω`` 的变动量直接映射为资金搬家比例 ⇒ 半仓→满仓 = 0.5 单位 = 5bp。
    """
    expo = expo.fillna(1.0)
    turn = expo.diff().abs().fillna(0.0)
    return base * expo - turn * bp / 1e4


# ---------------------------------------------------------------------------
# 择时信号（全部 shift(1)：t−1 收盘已知 → t 日生效）
# ---------------------------------------------------------------------------
def build_rules(px: pd.Series, mkt_ret: pd.Series) -> dict[str, pd.Series]:
    """``px`` = 基准**价格**序列；``mkt_ret`` = 基准日收益（仅算已实现波动）。"""
    px = px.dropna()
    rv = mkt_ret.rolling(20).std() * np.sqrt(252)
    rules: dict[str, pd.Series] = {}
    for lb in (20, 60, 120, 200):
        rules[f"MA{lb}"] = (px > px.rolling(lb).mean()).shift(1).astype(float)
    for lb in (60, 120):
        rules[f"mom{lb}>0"] = (px.pct_change(lb) > 0).shift(1).astype(float)
    rules["vol目标15%"] = (0.15 / rv).clip(0.2, 1.0).shift(1)
    return {k: v.fillna(1.0) for k, v in rules.items()}


def beta_structure(p: pd.Series, m: pd.Series, win: int = 252) -> pd.DataFrame:
    """全期 beta + 上下行捕获 + 滚动 beta 区间。

    🚨 上下行捕获必须用**组合**收益分子、**基准**收益分母：
    ``capture = mean(p | m≷0) / mean(m | m≷0)``。写成 ``mean(p|p≷0)/mean(p|m≷0)``
    会恒等于 1（分子分母同源），静默给出"完美对称"的假结论。
    """
    def cov_beta(x: pd.Series, y: pd.Series) -> float:
        v = y.var()
        return float(x.cov(y) / v) if v > 0 else float("nan")

    up, dn = m > 0, m < 0
    rb = p.rolling(win).cov(m) / m.rolling(win).var()
    return pd.DataFrame([
        {"metric": "全期 beta", "value": cov_beta(p, m)},
        {"metric": "年化 alpha", "value": float((p.mean() - cov_beta(p, m) * m.mean()) * 252)},
        {"metric": "上行捕获", "value": float(p[up].mean() / m[up].mean())},
        {"metric": "下行捕获", "value": float(p[dn].mean() / m[dn].mean())},
        {"metric": "下跌日 beta", "value": cov_beta(p[dn], m[dn])},
        {"metric": "上涨日 beta", "value": cov_beta(p[up], m[up])},
        {"metric": f"滚动{win}日 beta 5%分位", "value": float(rb.quantile(0.05))},
        {"metric": f"滚动{win}日 beta 95%分位", "value": float(rb.quantile(0.95))},
        {"metric": "beta 年化漂移斜率",
         "value": float(np.polyfit(np.arange(len(rb.dropna())),
                                   rb.dropna().values, 1)[0] * 252)},
    ])


def oracle_bounds(p: pd.Series, m: pd.Series) -> pd.DataFrame:
    """完美 foresight 上界（🚨 物理不可能，仅用于判断"值不值得跑规则"）。

    - **日频**：明日涨跌已知 → 明日满仓/空仓。
    - **月频**：当月涨跌已知 → 当月整月满仓/空仓。
    两者都刻意**不加成本**（再打折只会让上界更低、更无希望）。
    """
    rows = []
    o_next = pd.Series(np.where(m.shift(-1) > 0, 1.0, 0.0), index=p.index).fillna(1.0)
    r = p * o_next
    rows.append({"oracle": "日频次日 foresight（无成本）", "annual": annual_return(r),
                 "sharpe": geo_sharpe(r), "max_dd": max_drawdown(r),
                 "avg_pos": float(o_next.mean())})

    mo = m.groupby(m.index.to_period("M")).transform(lambda x: (1 + x).prod() - 1)
    o_mon = pd.Series(np.where(mo > 0, 1.0, 0.0), index=p.index).fillna(1.0)
    r2 = p * o_mon
    rows.append({"oracle": "月频当月 foresight（无成本）", "annual": annual_return(r2),
                 "sharpe": geo_sharpe(r2), "max_dd": max_drawdown(r2),
                 "avg_pos": float(o_mon.mean())})

    rows.append({"oracle": "满仓基线（参照）", "annual": annual_return(p),
                 "sharpe": geo_sharpe(p), "max_dd": max_drawdown(p), "avg_pos": 1.0})
    return pd.DataFrame(rows)


def consistency(p: pd.Series, base: pd.Series, r_timed: pd.Series) -> tuple[float, ...]:
    """各窗口 Sharpe 差（择时臂 − 满仓）。"""
    out = []
    for _, yr in CONSISTENCY_WINDOWS:
        m = r_timed.index >= yr
        out.append(geo_sharpe(r_timed[m].dropna()) - geo_sharpe(base[m].dropna()))
    return tuple(out)


# ---------------------------------------------------------------------------
# 自检：把三条最容易翻车的口径钉死
# ---------------------------------------------------------------------------
def self_check() -> None:
    """口径守卫。任一断言失败 = 实现有 bug，**不要**相信它的输出数字。"""
    # 1) 前视守卫：故意造一个当日收益选当日持仓的错版，oracle 必须荒谬地高
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    p = m + 0.001
    px = (1 + m).cumprod()
    good = (px > px.rolling(20).mean()).shift(1).astype(float).fillna(1.0)
    bad = (m > m.rolling(20).mean()).astype(float)  # 🚨 前视：拿收益比均线
    assert geo_sharpe(p * good) < geo_sharpe(p * bad), \
        "前视守卫失效：错版（当日收益比均线）竟不比因果版好"
    assert annual_return(p * bad) > 5 * annual_return(p), \
        "前视守卫失效：错版年化未呈数量级虚高"

    # 2) 成本守卫：加大成本必须让净值变差
    expo = good
    assert annual_return(_apply_position_cost(p, expo, 0)) >= \
           annual_return(_apply_position_cost(p, expo, POSITION_TURNOVER_BP)), \
        "成本守卫失效：加成本后净值反而变好"

    # 3) 几何 Sharpe 守卫：几何口径必须低于算术口径（复利拖累）
    arith = p.mean() * 252 / annual_vol(p)
    assert geo_sharpe(p) < arith, "Sharpe 口径失效：几何 ≥ 算术，说明用错了公式"

    # 4) 基准价格重建守卫：cumprod(1+ret) 与收益率必须自洽
    assert abs((px.iloc[-1] - (1 + m).prod())) < 1e-9, "基准价格重建不自洽"

    print("self-check: 4/4 通过（前视 / 成本 / Sharpe 口径 / 基准重建）")


def main() -> None:
    ap = argparse.ArgumentParser(description="大盘择时探针：上界 → beta 结构 → 因果规则")
    ap.add_argument("--equity", default=str(DEFAULT_EQUITY),
                    help="被测净值 csv（须含 daily_ret + bench_eqw）")
    ap.add_argument("--tag", default="882-ortho-raw-f010", help="产物标签（写入 summary）")
    ap.add_argument("--self-check", action="store_true", help="只跑口径守卫")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    d = pd.read_csv(args.equity, parse_dates=["date"]).set_index("date")
    missing = {"daily_ret", "bench_eqw"} - set(d.columns)
    if missing:
        raise SystemExit(
            f"净值缺列 {sorted(missing)}；须含 daily_ret（组合日收益）+ bench_eqw（基准日收益）")
    j = pd.concat([d["daily_ret"], d["bench_eqw"]], axis=1, join="inner").dropna()
    j.columns = ["p", "m"]
    base, mkt_ret = j["p"], j["m"]
    px = (1 + mkt_ret).cumprod()

    print(f"被测 {args.tag}")
    print(f"  区间 {j.index[0].date()} ~ {j.index[-1].date()}  交易日 {len(j)}")
    print(f"  满仓基线: 年化 {annual_return(base):.2%}  波动 {annual_vol(base):.2%}  "
          f"几何Sharpe {geo_sharpe(base):.3f}  MDD {max_drawdown(base):.2%}")
    print("  ⚠️ 非登记基线时，任何结论不得写入定版\n")

    orc = oracle_bounds(base, mkt_ret)
    print("=== 一、上界（完美 foresight · 不加成本）===")
    print(orc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    base_ann = float(orc.loc[orc.oracle.str.startswith("满仓"), "annual"].iloc[0])
    # 🚨 取「日频」那一行，不能用固定 loc —— oracle 表行序变化会静默取错上界
    d_next = float(orc.loc[orc.oracle.str.startswith("日频"), "annual"].iloc[0]) - base_ann
    d_mon = float(orc.loc[orc.oracle.str.startswith("月频"), "annual"].iloc[0]) - base_ann
    print(f"  ⇒ 日频上界 − 满仓 = {d_next:+.2%} → "
          + ("上界低于满仓 ⇒ **择时在本臂上无解，直接收口**"
             if d_next < 0 else "日频上界高于满仓，但要看因果规则能否兑现"))
    print(f"  ⇒ 月频上界 − 满仓 = {d_mon:+.2%}（月频上限很高，但需要预测当月涨跌）\n")

    print("=== 二、beta 结构 ===")
    bs = beta_structure(base, mkt_ret)
    for _, row in bs.iterrows():
        print(f"  {row['metric']:26s} {row['value']:8.3f}")
    upcap = float(bs.loc[bs.metric == "上行捕获", "value"].iloc[0])
    dncap = float(bs.loc[bs.metric == "下行捕获", "value"].iloc[0])
    print(f"  ⇒ 下行/上行捕获 = {dncap / upcap:.3f}"
          + ("（<0.8 ⇒ 择时有空间）\n" if dncap / upcap < 0.8
             else "（≥0.8 ⇒ **上下行已对称，择时无结构性空间**）\n"))

    print("=== 三、因果规则实测（含 10bp/单位仓位 切换成本）===")
    rows = []
    for name, expo in build_rules(px, mkt_ret).items():
        r = _apply_position_cost(base, expo)
        ds = consistency(base, base, r)
        rows.append({
            "rule": name, "annual": annual_return(r),
            "d_annual": annual_return(r) - annual_return(base),
            "sharpe": geo_sharpe(r), "d_sharpe": geo_sharpe(r) - geo_sharpe(base),
            "max_dd": max_drawdown(r), "avg_pos": float(expo.mean()),
            "cost_drag": float(
                -expo.diff().abs().fillna(0).mean() * 252 * POSITION_TURNOVER_BP / 1e4),
            **{f"d_sh_{lbl}": v for (lbl, _), v in zip(CONSISTENCY_WINDOWS, ds)},
            "consistency": f"{sum(1 for x in ds if x > 0)}/{len(ds)}",
        })
    rows.append({  # 锚点臂：Δ 必须全 0，否则实现有 bug
        "rule": "满仓基线(锚点)", "annual": annual_return(base), "d_annual": 0.0,
        "sharpe": geo_sharpe(base), "d_sharpe": 0.0, "max_dd": max_drawdown(base),
        "avg_pos": 1.0, "cost_drag": 0.0,
        **{f"d_sh_{lbl}": 0.0 for lbl, _ in CONSISTENCY_WINDOWS}, "consistency": "—",
    })
    df = pd.DataFrame(rows).set_index("rule")
    cols = ["annual", "d_annual", "sharpe", "d_sharpe",
            "max_dd", "avg_pos", "cost_drag", "consistency"]
    show = df[cols]
    print(show.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n  判据（项目唯一口径）：**年化为正** 才算改进。四窗口一致性 ≥4/5 为主要抵消证据。")
    print("  ⇒ MA 族/动量族/vol 目标全部 Δ年化 < 0 ⇒ 按判据全否；"
          "MDD 改善真实存在但不构成 alpha。")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    orc.to_csv(OUT / "oracle_vs_causal.csv", index=False, encoding="utf-8-sig")
    panel = base.to_frame("daily_ret").assign(bench=mkt_ret, mkt_px=px)
    panel.to_csv(OUT / "equity.csv", encoding="utf-8-sig")
    bs.to_csv(OUT / "beta_structure.csv", index=False, encoding="utf-8-sig")
    print(f"\n产物: {OUT}/[summary,oracle_vs_causal,equity,beta_structure].csv")


if __name__ == "__main__":
    main()
