"""cheap_lever 胜者的 920 定版信号复验（好机器批次 11，2026-10-17）。

882 扫出并四窗口复验的两个微调胜者（buf(0.08/0.40) +0.78pp / h1:h5=0.25:0.75
+0.31pp / 叠加 +0.85pp，登记 ``baselines.yaml::v1w2-cheap``）——**参数迁移 920
未验证是唯一 caveat**。本脚本在 **920 fundind 定版信号**（h1020⊕s0.5，
``buffer_arm_920 --variant defv`` 同口径）上复验：全窗 + 2019+/2021+/2023+/2025+
四段，比锚点 vs 各臂逐窗一致性。只截信号窗口；fwd/mask/fill/slow 保留全历史
（Σ 与 IC 暖水不截断）。

h1 加权在 920 信号上的映射（882 → 920，操作迁移而非数字迁移）：
  882 胜者 = ens_h1h5 里 h1 权重**减半**（0.5→0.25，h5 得其余）；
  920 快信号 = h1/h5/h10/h20 等权秩平均（h1 份额 0.25）。
  ⇒ 920 忠实迁移 = h1 份额**减半 0.25→0.125**，其余均摊给 h5/h10/h20：
  ``fast(w) = w·rank(h1) + (1−w)·mean(rank(h5), rank(h10), rank(h20))``，
  w=0.25 与 ``_rank_average`` **逐位相同**（脚本内断言）。另设剂量-反应诊断臂
  （w=0 / w=0.375）与字面迁移诊断臂（0.25·rank(h1)+0.75·rank(h5) 两面板⊕慢信号，
  弃用 h10/h20 与 h1020 定版裁决冲突，**不参与晋升判定**）。

锚点纪律：锚点臂（buf(15/40)+w0.25）须**逐位复现**同环境
``buffer_arm_920 --variant defv --entry 0.15 --exit 0.40`` 的输出
（``reports/cheap_lever_scan/_anchor_check/buffer_arm.csv``）。与登记数
18.07/0.805（§⑯）的差额 = 本机因子库缺 10 张基本面面板（慢信号 62 因子 vs
好机器 81）——环境漂移如实记录，不回退数据换数字。

裁决门槛（TODO_ARCHIVE 册Ⅲ 批次 11）：叠加臂 5 窗（全窗+四段）≥4/5 一致
（Δ年化 > 0）且量级 ≥+0.5pp ⇒ ``v1w2-cheap`` 晋升登记。

用法:
    python -m scripts.evaluation.cheap_lever_revalidate_920
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("cheap_lever_920")

# (tag, frac_entry, frac_exit, w_h1)；w_h1=None → 字面 h1:h5 两面板诊断臂
ARMS: tuple[tuple[str, float, float, float | None], ...] = (
    ("anchor_buf15_40_w0.25", 0.15, 0.40, 0.25),
    ("buf08_40", 0.08, 0.40, 0.25),
    ("h1w0.125_buf15_40", 0.15, 0.40, 0.125),
    ("stack_buf08_40_h1w0.125", 0.08, 0.40, 0.125),
    ("diag_w0_纯长窗", 0.15, 0.40, 0.0),
    ("diag_w0.375_h1升权", 0.15, 0.40, 0.375),
    ("diag_字面h1h5_25_75", 0.15, 0.40, None),
)
BEGINS: tuple[int | None, ...] = (None, 2019, 2021, 2023, 2025)  # None = 全窗（2018+）
ANCHOR_REF = ROOT / "reports" / "cheap_lever_scan" / "_anchor_check" / "buffer_arm.csv"


def wrank_avg(ranked: dict[str, pd.DataFrame], weights: dict[str, float],
               min_panels: int = 2) -> pd.DataFrame:
    """逐日截面**加权**秩平均（严格交集口径，等权时与 ``_rank_average`` 逐位相同）。

    ``_rank_average`` 对当日全部面板取有效代码**交集**后求均值；加权版在交集上
    取 ``Σw_k·r_k / Σw_k``（交集内每股票全面板有效 ⇒ 分母恒为 Σw_k）。当日可用
    面板数 < ``min_panels`` 的行整行置 NaN，与原实现一致。
    """
    panels = [ranked[k] for k in ranked]
    weights_l = [weights[k] for k in ranked]
    inter = None
    n_panel_row = None
    for p in panels:
        v = p.notna()
        inter = v if inter is None else inter & v
        has = v.any(axis=1).astype(np.int8)
        n_panel_row = has if n_panel_row is None else n_panel_row + has
    wsum = float(sum(weights_l))
    num = None
    for p, w in zip(panels, weights_l):
        t = p.where(inter) * w
        num = t if num is None else num + t
    out = (num / wsum).astype(np.float32)
    out.loc[n_panel_row < min_panels, :] = np.nan
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho920fundind",
                    help="920 定版 pred 目录（v2 数据态冻结面板）")
    ap.add_argument("--out", default="reports/cheap_lever_scan/revalidate_920.csv")
    ap.add_argument("--anchor-ref", default=str(ANCHOR_REF),
                    help="锚点交叉核对参照（buffer_arm_920 同环境输出）；"
                         "环境变更（如补面板）后须先重跑 _anchor_check 再指新文件")
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest, build_execution_split
    from scripts.evaluation.fundamental_blend import build_slow_panel
    from scripts.evaluation.horizon_mix import load
    from scripts.common import alla_common as AC
    from strategy.examples import BufferedTopFracLongOnly

    t0 = time.time()
    pred_dir = ROOT / args.pred_dir

    # ---- 1) 快信号四面板 + 慢信号（全窗构造，逐字对齐 buffer_arm_920 --variant defv）
    h = {k: load(pred_dir, f"gbdt__h{k}") for k in (1, 5, 10, 20)}
    ranked = {k: p.rank(axis=1, pct=True) for k, p in h.items()}
    fast_eq = AC.rank_average([h[1], h[5], h[10], h[20]])

    base = AC.load_base()
    close = base["close"]
    sig_idx = fast_eq.index
    slow, _mx, _w = build_slow_panel(base, sig_idx, 24, 2, 0.5, 12, log=print)
    slow_rank = slow.rank(axis=1, pct=True)

    # ---- 2) 构造各臂信号：defv(w) = 0.5·rank(fast(w)) + 0.5·rank(slow)
    def make_defv(w: float | None) -> pd.DataFrame:
        if w is None:  # 字面 h1:h5 两面板诊断臂
            fast = wrank_avg({1: ranked[1], 5: ranked[5]}, {1: 0.25, 5: 0.75})
        else:          # h1 份额 w，其余 (1−w) 均摊给 h5/h10/h20
            rest = (1.0 - w) / 3.0
            fast = wrank_avg(ranked, {1: w, 5: rest, 10: rest, 20: rest})
        return 0.5 * fast.rank(axis=1, pct=True) + 0.5 * slow_rank

    # 构造自检①：w=0.25 的 fast 与 _rank_average 逐位相同（NaN 掩码一致 +
    # 实值差 < 1e-6——float64 求和顺序 ULP 差容忍，回测影响为零）
    fast_w025 = wrank_avg(ranked, {1: 0.25, 5: 0.25, 10: 0.25, 20: 0.25})
    if not fast_w025.isna().equals(fast_eq.isna()):
        n_diff = int((fast_w025.isna() != fast_eq.isna()).sum().sum())
        raise AssertionError(f"w=0.25 加权重构与 _rank_average NaN 掩码不同（{n_diff} 格）")
    d = float((fast_w025 - fast_eq).abs().max().max())
    if not np.isfinite(d) or d > 1e-6:
        raise AssertionError(f"w=0.25 加权重构与 _rank_average 数值不符 (maxΔ={d})")
    log.info("构造自检①通过：w=0.25 加权重构 == _rank_average（NaN 掩码一致，maxΔ=%.2e）", d)

    sigs = {}
    for tag, _fe, _fx, w in ARMS:
        if w not in sigs:
            sigs[w] = make_defv(w)

    mask = base["mask"].astype(bool)
    fwd = close.pct_change(fill_method=None).reindex(close.index)
    costs = default_costs()
    fp = pred_dir / "_base" / "open_adj.parquet"
    if not fp.exists():
        raise SystemExit(f"{fp} 缺失：先跑 rolling_grid_alla --stage prep")
    fill = pd.read_parquet(fp)

    # ---- 3) 逐窗 × 逐臂回测（buffer_arm_920 同执行路径：T+1 open + 预掩码）
    def eval_arm(sig_full: pd.DataFrame, fe: float, fx: float,
                 begin: int | None) -> dict:
        sig = sig_full if begin is None else sig_full.loc[f"{begin}-01-01":]
        oos = sig.index
        mask_oos = mask.reindex(index=oos, columns=close.columns).fillna(True)
        bench = base["bench_index"].reindex(oos).fillna(0.0)
        s_ = pd.Series(oos, index=oos)
        firsts = s_.groupby(s_.index.to_period("M")).first()
        pos_ = {d: i for i, d in enumerate(oos)}
        rb_signal = {t for t in firsts if pos_[t] + 1 < len(oos)}
        rb_exec = {oos[pos_[t] + 1] for t in rb_signal}
        # build 收信号日 T，run 收执行日 E=T+1（2026-10-19 口径修复）
        split = build_execution_split(fill, close, rb_signal)
        bt = VectorBacktest(strategy=BufferedTopFracLongOnly(fe, fx),
                            rebalance_freq="M", initial_capital=1_000_000.0,
                            costs=costs)
        res = bt.run(sig.reindex(index=oos, columns=close.columns)
                     .shift(1).where(mask_oos),
                     fwd.loc[oos], horizon=1, rebalance_days=rb_exec,
                     execution_split=split)
        m = AC.res_metrics(res.daily_returns, bench, res.turnover_series)
        return dict(ann=m["annual"], sharpe=m["sharpe"], max_dd=m["max_dd"],
                    turnover=m["turnover"], n_days=len(res.daily_returns.dropna()))

    rows = []
    anchors = {}
    for begin in BEGINS:
        wlabel = "全窗2018+" if begin is None else f"{begin}+"
        for tag, fe, fx, w in ARMS:
            m = eval_arm(sigs[w], fe, fx, begin)
            if tag.startswith("anchor"):
                anchors[begin] = m["ann"]
            rows.append(dict(window=wlabel, arm=tag, fe=fe, fx=fx, w_h1=w,
                             ann=round(m["ann"] * 100, 2),
                             sharpe=round(m["sharpe"], 3),
                             max_dd=round(m["max_dd"] * 100, 1),
                             turnover=round(m["turnover"] * 100, 1),
                             n_days=m["n_days"]))
            log.info("[%s] %-24s 年化=%+6.2f%% Sharpe=%.3f 换手=%5.1f%%",
                     wlabel, tag, rows[-1]["ann"], rows[-1]["sharpe"],
                     rows[-1]["turnover"])

    for r in rows:  # 同窗内对锚点的 Δ
        begin = None if r["window"] == "全窗2018+" else int(r["window"][:4])
        r["delta_ann"] = round(r["ann"] - anchors[begin] * 100, 2)

    # ---- 4) 锚点交叉核对：全窗锚点臂 == buffer_arm_920 同环境输出（逐位）
    anchor_ref = Path(args.anchor_ref)
    if anchor_ref.exists():
        ref = pd.read_csv(anchor_ref)
        ref_row = ref.loc[ref["variant"] == "buffer_15_40"].iloc[0]
        got = anchors[None]
        if abs(got * 100 - ref_row["annual"] * 100) > 0.02 or \
                abs(next(r["sharpe"] for r in rows
                         if r["window"] == "全窗2018+" and r["arm"].startswith("anchor"))
                    - ref_row["sharpe"]) > 0.005:
            raise AssertionError(
                f"锚点臂与 buffer_arm_920 同环境输出不符：{got * 100:.2f}/"
                f"{ref_row['annual'] * 100:.2f} —— 先修口径再谈结论（环境变更后"
                f"重跑 _anchor_check 并 --anchor-ref 指新文件）")
        log.info("构造自检②通过：锚点臂逐位复现 buffer_arm_920 同环境输出 "
                 "%.2f%%/%.3f（参照 %s；与登记数 18.07/0.805 的差额 = "
                 "环境数据态漂移，见 README）",
                 got * 100, ref_row["sharpe"], anchor_ref.name)
    else:
        log.warning("锚点参照 %s 不存在，跳过交叉核对", anchor_ref)

    df = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, out)

    log.info("==== 裁决表（Δ年化 vs 同窗锚点，pp）====")
    piv = df.pivot(index="arm", columns="window", values="delta_ann")
    log.info("\n%s", piv.to_string())
    stack = df[df["arm"] == "stack_buf08_40_h1w0.125"]
    n_pos = int((stack["delta_ann"] > 0).sum())
    full_delta = float(stack.loc[stack["window"] == "全窗2018+", "delta_ann"].iloc[0])
    log.info("叠加臂：%d/5 窗一致为正；全窗 Δ=%+.2fpp（门槛 ≥4/5 且 ≥+0.5pp）",
             n_pos, full_delta)


if __name__ == "__main__":
    main()
