"""ETF 轮动 · 策略级过拟合审计（CSCV/PBO + DSR，2026-10-30）
================================================================

背景（`TODO.md` §一 赛后残余 ③，预注册判读）
    「33 条判否」+ 80 个重叠窗 + 池子都是**看过结果之后**选的。目前对外只
    说「30.63% 是上界」，这是定性描述，没有量化「上界里有多少是选择偏差」。
    预注册判读：**PBO ≤ 0.3 可继续对外引用；> 0.5 须在 `etf_contest_spec`
    加过拟合折扣标注。**

本脚本只做一件事：把「我们实际在 N 个候选里挑了一个」这件事，按 CSCV 的
对称切分量化成概率。**零改动**——不碰 `FINAL`、不碰宇宙、不碰成本口径。

四个候选网格（每个网格内部才可比，PBO 的语义就是「这个候选集里挑 IS 最优」）：

| 网格 | 候选集 | 我们最终选了谁 | 来源 |
|---|---|---|---|
| `phase5` | 周内 5 个调仓相位 | k=1（引擎默认，但也是 5 档里最好的） | 现跑（`run_variant`） |
| `weight13` | 13 档权重规则 | hrp | 现跑 |
| `buffer6` | contest3 / contest5h / 4 个 buffer 档 | contest3 | `eval_windows*.csv` |
| `pool4` | 池 13 / 32 / 45 / 50 只 | 45 只（合规口径） | `eval_pool*.csv` |

两个时间网格各跑一遍：
- **重叠 80 个 2 月窗**（t→t+2 月，相邻共享 1 月）⇒ 切 S=16 块；
- **非重叠 41 个 2 月期**（`resample("2ME")`，赛制真实计分单位）⇒ 切 S=8 块。

🚨 为什么两个都要看：重叠窗的观测高度相关，CSCV 的随机块切分只能部分吸收，
**非重叠期是主判据**（与 README §4.1–4.2 的教训同源：曾据重叠窗「单点最差
窗口」判否 buffer，换非重叠后方向反转）。

DSR 的读法（与 PBO 互补）
- PBO 回答「挑出来的那个 OOS 掉队的概率」；
- DSR 回答「挑出来的那个 SR，在 N 次试验的最大值零分布下还剩多少显著性」。
- 🚨 **DSR 的 N 是保守下界**：联合网格只装得下 28 个臂（有逐期收益的），
  而实际测过的变体是 **33 条判否 + 各档位** ⇒ 真实 SR₀ 更高、真实 DSR 更低。
  故 `dsr_union` 是**上界**，引用时不得说成「计入全部试验后仍显著」。

用法::

    python -m scripts.portfolio.etf_pbo_audit              # 全部四个网格
    python -m scripts.portfolio.etf_pbo_audit --grids buffer6,pool4   # 只读现成 CSV

产物（落 `reports/etf_contest5/`）：
    pbo_audit_2026-10-30.txt   人读报告（含预注册判读）
    pbo_summary.csv            每网格汇总
    pbo_detail.json            逐网格 logits / omegas / trial SR（可复算）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stats.pbo import cscv_pbo, deflate_best  # noqa: E402

OUT = Path("reports/etf_contest5")
TXT = OUT / "pbo_audit_2026-10-30.txt"

#: 周内相位（第 k 个交易日，0-indexed）
PHASES = [0, 1, 2, 3, 4]
#: 13 档权重规则（与 `etf_weight_probe.ALL_RULES` 逐字一致）
WEIGHT_RULES = ("hrp", "erc", "inv_var", "inverse_vol", "equal", "rank_decay",
                "min_var", "max_div", "risk_budget",
                "hrp_shrunk", "hrp_avgcorr", "hrp_ew", "hrp_anticorr")
POOLS = ("pool13", "pool32", "pool45", "pool50")

#: 各时间网格的分块数（S 必须偶数；每块至少 2 期）
S_OVERLAP = 16
S_PERIOD = 8


# --------------------------------------------------------------- 收益矩阵来源
def _read_csv_grid(cols_map: dict[str, str], path: Path) -> pd.DataFrame:
    """从评测 CSV 取若干列；``cols_map`` = {输出名: CSV 原列名}。"""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    src = list(cols_map.values())
    missing = [c for c in src if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} 缺列 {missing}，实际列 {list(df.columns)}")
    out = df[src].rename(columns={v: k for k, v in cols_map.items()})
    return out.astype(float).dropna()


def grid_buffer6() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = {"contest3": "contest3", "contest5h": "contest5h",
         "buf10": "c3_buffer10", "buf15_40": "c3_buffer15_40",
         "buf5_20": "c3_buffer5_20", "buf30": "c3_buffer30"}
    return (_read_csv_grid(m, OUT / "eval_windows.csv"),
            _read_csv_grid(m, OUT / "eval_windows_periods.csv"))


def grid_pool4() -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个池一个独立 CSV（列名都是 `contest3`）⇒ 逐文件取单列再拼。"""

    def _load(suffix: str) -> pd.DataFrame:
        cols = {}
        for p in POOLS:
            f = OUT / f"eval_{p}{suffix}.csv"
            if not f.exists():
                continue
            s = pd.read_csv(f, index_col=0, parse_dates=True)["contest3"]
            cols[p] = s.astype(float)
        if len(cols) < 2:
            raise RuntimeError(f"池候选 CSV{suffix} 不足 2 个")
        return pd.DataFrame(cols).dropna()

    return _load(""), _load("_periods")


# ------------------------------------------------------------ 现跑（需数据）
def _week_groups(days: pd.DatetimeIndex, begin: int) -> list[list[pd.Timestamp]]:
    """按自然周分组（口径同 `etf_weight_probe`/`phase_rule_20261030`）。

    🚨 必须先用**回测起点之后**的交易日分组：engine 的 `_get_rebalance_days`
    是在 `begin` 切片后的 dates 上 groupby；拿全索引（含 2018–2019 暖机）
    分组会让 2020 首周锚点前移，k=1 复现掉 1.41pp（2026-10-30 实测）。
    """
    days = pd.DatetimeIndex(days)
    days = days[days >= pd.Timestamp(str(begin))]
    s = pd.Series(days, index=days)
    return [list(g) for _, g in s.groupby(days.to_period("W"))]


def _rb_days(groups: list[list[pd.Timestamp]], k: int) -> set:
    """第 k 个交易日调仓的集合；短周（交易日 ≤ k）跳过。"""
    return {g[k] for g in groups if len(g) > k}


def _periods(eq: pd.Series) -> pd.Series:
    """非重叠 2 月期收益（与 `etf_contest5_eval` 逐字同实现）。"""
    return eq.resample("2ME").last().pct_change().dropna()


def run_rerun_grids(models: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """现跑 phase5 / weight13，返回 {grid: (重叠窗, 非重叠期)}。

    🚨 指纹守卫：k=1 与 hrp 必须各自复现 30.63% 基线，否则整份审计作废
    （说明取数路径或策略默认值被改过，PBO 算出来也没意义）。
    """
    from scripts.portfolio.etf_contest5_eval import BEGIN, two_month_windows
    from scripts.portfolio.etf_contest_spec import FINAL
    from scripts.portfolio.etf_weight_probe import load_env, run

    env = load_env()
    spec = dict(FINAL)
    grids: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _collect(arms: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
        w = pd.DataFrame({k: two_month_windows(v["equity"], env["mfs"])
                          for k, v in arms.items()}).dropna()
        p = pd.DataFrame({k: _periods(v["equity"]) for k, v in arms.items()}).dropna()
        return w, p

    if "phase5" in models:
        groups = _week_groups(env["close"].index, BEGIN)
        arms = {}
        for k in PHASES:
            _, r1 = run(env, spec, rebalance_days=_rb_days(groups, k))
            arms[f"k{k + 1}"] = r1
            print(f"  [phase5] k={k + 1} 年化 {r1['annual_return']:.2%} "
                  f"SR {r1['sharpe']:.2f} 调仓 {r1['n_rebalance']}")
        base = arms["k1"]["annual_return"]
        if abs(base - 0.3063) > 0.005:
            raise SystemExit(f"🚨 指纹守卫失败：k=1 年化 {base:.2%} 偏离基线 30.63%")
        print(f"  ✅ 指纹守卫：k=1 = {base:.2%}")
        grids["phase5"] = _collect(arms)

    if "weight13" in models:
        arms = {}
        for wt in WEIGHT_RULES:
            s = dict(spec)
            s["weighting"] = wt
            _, r1 = run(env, s)
            arms[wt] = r1
            print(f"  [weight13] {wt:<12} 年化 {r1['annual_return']:.2%} "
                  f"SR {r1['sharpe']:.2f}")
        base = arms["hrp"]["annual_return"]
        if abs(base - 0.3063) > 0.005:
            raise SystemExit(f"🚨 指纹守卫失败：hrp 年化 {base:.2%} 偏离基线 30.63%")
        print(f"  ✅ 指纹守卫：hrp = {base:.2%}")
        grids["weight13"] = _collect(arms)

    return grids


# ------------------------------------------------------------------- 审计
def _mean_pair_corr(mat: pd.DataFrame) -> float:
    """臂间平均两两相关（去对角）。

    🚨 读 PBO 必须配这一列：候选臂高度相关时，「挑 IS 最优」几乎不会 OOS 掉队
    （因为大家路径一样），PBO 会被机械压低 —— **低 PBO 不等于选择过程稳健**。
    """
    c = mat.corr().to_numpy(dtype=float)
    n = c.shape[0]
    if n < 2:
        return float("nan")
    off = c[~np.eye(n, dtype=bool)]
    return float(np.nanmean(off))


def audit_one(name: str, win: pd.DataFrame, per: pd.DataFrame,
              pad: list[str], dump: bool = True) -> dict:
    """一个网格同时出重叠窗 / 非重叠期两套 PBO + DSR。"""
    res = {"grid": name, "n_arms": int(win.shape[1]),
           "arms": list(win.columns), "overlap_n": int(win.shape[0]),
           "period_n": int(per.shape[0])}
    for tag, mat, S in (("overlap", win, S_OVERLAP), ("period", per, S_PERIOD)):
        if dump:
            mat.to_csv(OUT / f"pbo_grid_{name}_{tag}.csv", encoding="utf-8-sig")
        if mat.shape[0] < 2 * S or mat.shape[1] < 2:
            res[f"{tag}_pbo"] = float("nan")
            continue
        r = cscv_pbo(mat, n_partitions=S, metric="sharpe")
        r_mean = cscv_pbo(mat, n_partitions=S, metric="mean")
        b = deflate_best(mat)
        res.update({
            f"{tag}_pbo": r["pbo"],
            f"{tag}_pbo_meanmetric": r_mean["pbo"],
            f"{tag}_omega": r["omega_mean"],
            f"{tag}_logit_med": r["logit_median"],
            f"{tag}_best_arm": b["best_column"],
            f"{tag}_best_sr": b["best_sr"],
            f"{tag}_dsr": b["dsr"],
            f"{tag}_n_comb": r["n_combinations"],
            f"{tag}_arm_corr": _mean_pair_corr(mat),
        })
        pad.append(f"  {name:<9}{tag:<8}{mat.shape[0]:>4}{mat.shape[1]:>4}{S:>4}"
                   f"{r['pbo']:>8.3f}{r_mean['pbo']:>9.3f}{r['omega_mean']:>9.3f}"
                   f"{b['best_column']:>12}{b['best_sr']:>9.3f}{b['dsr']:>8.3f}"
                   f"{_mean_pair_corr(mat):>9.2f}")
    return res


def verdict(pbo: float) -> str:
    """预注册判读（TODO.md §一 ③，不得事后改阈值）。"""
    if not np.isfinite(pbo):
        return "不可算"
    if pbo <= 0.30:
        return "✅ 可继续对外引用"
    if pbo <= 0.50:
        return "⚠️ 需标注「含选择偏差」"
    return "🚨 须加过拟合折扣标注"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ETF 轮动 CSCV/PBO 过拟合审计")
    ap.add_argument("--grids", default="phase5,weight13,buffer6,pool4",
                    help="逗号分隔；不带 phase5/weight13 时完全不碰数据源（纯读 CSV）")
    ap.add_argument("--out-txt", default=str(TXT))
    args = ap.parse_args(argv)
    want = [g.strip() for g in args.grids.split(",") if g.strip()]

    grids: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    need_rerun = [g for g in want if g in ("phase5", "weight13")]
    if need_rerun:
        print(f"== 现跑网格：{need_rerun}（DataCache 单账号互斥，勿与其他任务并发）==")
        grids.update(run_rerun_grids(",".join(need_rerun)))
    if "buffer6" in want:
        grids["buffer6"] = grid_buffer6()
    if "pool4" in want:
        grids["pool4"] = grid_pool4()

    lines: list[str] = []
    rows: list[dict] = []
    pad: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 100)
    say("ETF 轮动 · 策略级过拟合审计（CSCV/PBO + DSR）  2026-10-30")
    say("=" * 100)
    say("口径：与 `etf_contest_spec.FINAL` 完全一致（Top3 / HRP / 周频 / dedup 0.9 /")
    say("      T+1 open / 80 重叠窗 + 41 非重叠期）；**零改动**，只量化「在 N 个候选里挑一个」。")
    say("预注册判读：PBO ≤ 0.3 可继续对外引用；0.3–0.5 需标注；> 0.5 须加过拟合折扣。")
    say()
    say("  " + f"{'网格':<9}{'口径':<8}{'T':>4}{'N':>4}{'S':>4}"
              f"{'PBO':>8}{'PBOμ':>9}{'ω̄':>9}{'IS最优':>12}{'IS期SR':>9}"
              f"{'DSR':>8}{'臂相关':>8}")
    for name, (win, per) in grids.items():
        rows.append(audit_one(name, win, per, pad))
    for s in pad:
        say(s)

    # 联合网格：所有臂拼在一起 = 「实际测过的那一堆里挑一个」
    common = None
    for win, _ in grids.values():
        common = win.index if common is None else common.intersection(win.index)
    union_win = pd.concat({k: v[0].loc[common] for k, v in grids.items()}, axis=1)
    union_win.columns = [f"{g}::{c}" for g, c in union_win.columns]
    union_per = pd.concat({k: v[1] for k, v in grids.items()}, axis=1).dropna()
    union_per.columns = [f"{g}::{c}" for g, c in union_per.columns]
    say()
    say(f"== 联合网格（{union_win.shape[1]} 个臂 × {union_win.shape[0]} 窗 / "
        f"{union_per.shape[0]} 非重叠期）==")
    u = audit_one("union", union_win, union_per, [])
    rows.append(u)
    say(f"  overlap : PBO {u['overlap_pbo']:.3f}（μ 口径 {u['overlap_pbo_meanmetric']:.3f}）"
        f" ω̄ {u['overlap_omega']:.3f}  IS 最优 {u['overlap_best_arm']}"
        f"  DSR {u['overlap_dsr']:.3f}")
    say(f"  period  : PBO {u['period_pbo']:.3f}（μ 口径 {u['period_pbo_meanmetric']:.3f}）"
        f" ω̄ {u['period_omega']:.3f}  IS 最优 {u['period_best_arm']}"
        f"  DSR {u['period_dsr']:.3f}")

    say()
    say("== 逐网格判读 ==")
    for r in rows:
        say(f"  {r['grid']:<9} 重叠 PBO {r['overlap_pbo']:.3f} → {verdict(r['overlap_pbo'])}"
            f"  |  非重叠 PBO {r['period_pbo']:.3f} → {verdict(r['period_pbo'])}")
    say()
    say("== 必读边界 ==")
    say("  ① PBO/DSR 只回答「选择过程是否过拟合」，不回答策略本身是否有效；")
    say("  ② 80 重叠窗的观测高度相关，CSCV 的随机块切分只部分吸收 ⇒ **非重叠 41 期为主判据**；")
    say("  ③ DSR 的 N 只含「有逐期收益的臂」（联合网格 "
        f"{union_win.shape[1]} 个），而实际测过的变体是 33 条判否 + 各档位 ⇒")
    say("     真实 SR₀ 更高、真实 DSR 更低 ⇒ **本表 DSR 是上界**，")
    say("     不得表述成「计入全部试验后仍显著」；")
    say("  ④ 网格候选集本身含「先看过结果再决定测哪几档」的成分，PBO 无法修正这一层；")
    say("  ⑤ 🚨 **臂相关列是必读的**：候选臂高度相关时，挑 IS 最优几乎不会 OOS 掉队")
    say("     （大家走同一条路径）⇒ PBO 被机械压低。**低 PBO 只说明「换个候选也没差」，")
    say("     不等于「选择过程稳健」**，两者不可混读。")
    say("  ⑥ 逐网格收益矩阵已落盘 `pbo_grid_<网格>_<口径>.csv`（可离线复算，不必再跑数据源）。")

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pbo_summary.csv", index=False, encoding="utf-8-sig")
    detail = {r["grid"]: {k: (v if isinstance(v, (int, float, str, list, type(None)))
                              else float(v))
                          for k, v in r.items()} for r in rows}
    (OUT / "pbo_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘 → {args.out_txt}\n落盘 → {OUT / 'pbo_summary.csv'}"
          f"\n落盘 → {OUT / 'pbo_detail.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
