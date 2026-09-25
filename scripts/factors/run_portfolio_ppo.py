"""stage2 Phase 1：hs300 平价验证（PPO vs QP vs 等权基准 vs TopN）。

设计依据：``reports/docs/research_notes/银河0706_华安226_研读_
RL组合优化全景与stage2设计定稿.md``。Phase 1 只验证链路——
**预期结果：PPO 与 QP 打平或小输**（银河证据：成分多/偏离空间小的池子
RL 无优势），判定标准是四臂评估链路自洽、无口径 bug，而非 RL 跑赢。

信号代理口径（Phase 1 简化，Phase 2 换 GBDT h=10 预测，TODO 已注明）：
- f（收益信号）= ``rev5``（5 日反转，``factor.classic`` 已截面标准化）；
- z（风险信号，**值越大风险越低**）= ``-vol20``；
- wb（基准）= HS300 PIT 在册成分**等权**（简化口径：项目无指数权重面板，
  全臂同 wb 内部自洽；引用须注明非自由流通市值加权）。

用法::

    python -m scripts.factors.run_portfolio_ppo --ppo-timesteps 30000
    python -m scripts.factors.run_portfolio_ppo --begin 2023-01-01 --arms ppo,qp,ew,topn

输出：``reports/portfolio_ppo_phase1/``（summary.csv / equity.csv / report.md）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("portfolio_ppo")


def rl_device() -> str:
    """RL 训练设备：环境变量 ``YQ_RL_DEVICE`` 控制。

    - ``auto``（默认）：有 CUDA 就用 GPU，否则 CPU；
    - ``cpu`` / ``cuda`` / ``cuda:0``：显式指定。

    2026-09-20 加入：此前 PPO 训练硬编码 ``device="cpu"``，在无 GPU 机器上
    是唯一可行选择；迁到 RTX 4060（cu126）后 MlpPolicy 的小网络在 GPU 上
    收益有限，但长 timesteps 训练仍有提速，故默认改为 auto 并允许显式回退。
    """
    import os

    want = os.environ.get("YQ_RL_DEVICE", "auto").strip().lower()
    if want != "auto":
        return want
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


#: 决策频率（交易日）——与 h=10 预测期对齐，非重叠持有期
DECISION_FREQ = 10
#: TopN 臂持仓只数（hs300 的 20%）
TOPN = 60


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def load_panels(begin: str, end: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """日线 OHLCV 宽表 + HS300 PIT 成分掩码（begin 前多留 60 日预热窗）。"""
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from data.universe import Universe

    cache = DataCache(OfflineDataSource())
    raw = pd.read_parquet(cache.root / "daily_hs300.parquet")
    # 存储为 (date, code) MultiIndex 长表 → reset 后按日期过滤再透视
    raw = raw.reset_index()
    date_col = "date" if "date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    start = (pd.Timestamp(begin) - pd.Timedelta(days=130)).strftime("%Y-%m-%d")
    raw = raw[(raw[date_col] >= start)
              & (raw[date_col] <= pd.Timestamp(end))]
    px = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        px[col] = (raw.pivot(index=date_col, columns="code",
                             values=col).sort_index())
    dates = px["close"].index
    mask = Universe(cache).get_membership_mask("000300.SH", dates)
    log.info("面板 %d 日 × %d 码（%s ~ %s）",
             len(dates), mask.shape[1], dates[0].date(), dates[-1].date())
    return px, mask


def build_signals(px: dict[str, pd.DataFrame], codes: list[str]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """f = rev5（5 日反转）；z = -vol20（值越大风险越低）。"""
    from factor.classic import compute_classic_features

    feats = compute_classic_features(px)
    f = feats["rev5"][codes]
    z = -feats["vol20"][codes]
    return f, z


def make_decision_frames(
    px: dict[str, pd.DataFrame],
    mask: pd.DataFrame,
    codes: list[str],
    begin: str,
) -> tuple[list[pd.Timestamp], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """决策日（每 DECISION_FREQ 个交易日）+ 期间收益（末端日索引）+ 基准面板。

    期间收益约定（PortfolioEnv docstring）：``period_returns.loc[d_i]`` =
    close[d_i]/close[d_{i-1}] − 1（d_i 为期间末端日）。基准期间收益 = 成分
    等权组合的同期间收益（在册掩码内等权、逐期重算）。
    """
    close = px["close"][codes]
    dates = close.index
    in_mkt = dates >= pd.Timestamp(begin)
    eval_dates = dates[in_mkt]
    # 决策日：评估期内每第 DECISION_FREQ 个交易日（首日为决策起点）
    decision_dates = list(eval_dates[::DECISION_FREQ])
    if len(decision_dates) < 2:
        raise ValueError("评估期太短，凑不满两个决策日")

    prev = close.shift(DECISION_FREQ)
    period_returns = (close / prev - 1.0).loc[decision_dates[1:]]

    bench = mask.reindex(index=decision_dates, columns=codes).fillna(False)
    bw = bench.div(bench.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    idx_ret = pd.Series({
        d: float(bw.loc[d] @ (close.loc[d] / prev.loc[d] - 1.0).fillna(0.0))
        for d in decision_dates[1:]
    })
    return decision_dates, period_returns, bw, idx_ret


# ---------------------------------------------------------------------------
# 四臂
# ---------------------------------------------------------------------------
def run_arm_ppo(env_kwargs: dict, timesteps: int, seed: int, out_dir: Path
                ) -> tuple[pd.Series, pd.Series, dict]:
    """PPO 臂：训练（env 奖励）→ 确定性 rollout 收集权重 → 统一口径重算收益。"""
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(max(1, torch.get_num_threads()))
    env = PortfolioEnv(**env_kwargs)
    dev = rl_device()
    log.info("PPO 训练设备：%s", dev)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                device=dev, **PPO_KW)
    model.learn(total_timesteps=timesteps, progress_bar=False)

    obs, _ = env.reset()
    weights: dict[pd.Timestamp, np.ndarray] = {}
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        weights[info["date"]] = info["weights"]
        done = term or trunc
    log.info("PPO 训练完成（%d steps，%d 个决策日权重）", timesteps, len(weights))
    return weights, pd.Series(dtype=float), {}


def run_arm_qp(px: dict[str, pd.DataFrame], codes: list[str],
               decision_dates: list[pd.Timestamp], bw: pd.DataFrame,
               f: pd.DataFrame) -> tuple[dict, pd.Series]:
    """QP 臂：solve_portfolio（mvo + 个股上限 + 组合换手），与 PPO 同信号同基准。

    协方差估计的口径（为何不用 rolling_covariance 整面板调用）：该函数内部
    ``dropna(how="any")`` 会在含数据不全历史成分的宽面板上把窗口行全数剔除
    （实测 513 码全部 None → 整臂退化为基准）。故逐期取「当期成分 ∩ 120 日
    窗内数据完整」子集，在子集内解小稠密 QP，解散回全码表（子集外恒 0）；
    求解失败或样本不足的期回退持有基准。
    """
    from optimize.solver import estimate_covariance, solve_portfolio

    close = px["close"][codes]
    rets = close.pct_change(fill_method=None)
    w_prev = bw.iloc[0].to_numpy(dtype=float).copy()
    weights: dict[pd.Timestamp, np.ndarray] = {}
    n_fallback = 0
    for t in decision_dates[:-1]:                      # 末期决策无持有期，跳过
        hist = rets.loc[:t].iloc[:-1].tail(120)        # 防前视：不含当日
        valid = (bw.loc[t] > 0) & hist.notna().all()
        cols = valid.index[valid]
        if len(cols) < 60:                             # 风险模型样本不足 → 持有基准
            weights[t] = bw.loc[t].to_numpy(dtype=float)
            n_fallback += 1
            continue
        sigma_sub = estimate_covariance(rets[cols].loc[:t].iloc[:-1].tail(120),
                                        method="ledoit_wolf", shrinkage=0.5)
        # 在有效子集内解小稠密 QP（全尺寸带零块的 Σ 会让 OSQP 数值退化），
        # 解再散回全码表；子集外权重恒 0
        alpha_sub = f.loc[t].reindex(cols)
        prev_sub = pd.Series(w_prev, index=codes).reindex(cols).fillna(0.0)
        try:
            w_sub = solve_portfolio(alpha_sub, sigma_sub, method="mvo",
                                    risk_aversion=5.0, max_weight=0.05,
                                    prev_weights=prev_sub, max_turnover=0.5)
        except RuntimeError:
            n_fallback += 1
            weights[t] = bw.loc[t].to_numpy(dtype=float)
            continue
        w = np.zeros(len(codes))
        w[valid.to_numpy(dtype=bool)] = w_sub.reindex(cols).fillna(0.0).to_numpy(dtype=float)
        weights[t] = w
        w_prev = w
    if n_fallback:
        log.warning("QP 臂 %d/%d 期风险样本不足，回退基准", n_fallback,
                    len(decision_dates) - 1)
    return weights, pd.Series(dtype=float)


def run_arm_topn(f: pd.DataFrame, bw: pd.DataFrame, codes: list[str],
                 decision_dates: list[pd.Timestamp]
                 ) -> tuple[dict[pd.Timestamp, np.ndarray], pd.Series]:
    """TopN 臂：决策日信号前 TOPN 只等权（不可持仓=非成分处剔除）。"""
    weights: dict[pd.Timestamp, np.ndarray] = {}
    for t in decision_dates[:-1]:
        sig = f.loc[t].reindex(codes)
        tradable = bw.loc[t].to_numpy(dtype=float) > 0
        sig = sig.where(tradable)
        pick = sig.nlargest(TOPN).index
        w = pd.Series(0.0, index=codes)
        w[pick] = 1.0 / max(len(pick), 1)
        weights[t] = w.to_numpy(dtype=float)
    return weights, pd.Series(dtype=float)


def evaluate_arm(name: str, weights: dict[pd.Timestamp, np.ndarray],
                 codes: list[str], decision_dates: list[pd.Timestamp],
                 period_returns: pd.DataFrame, idx_ret: pd.Series
                 ) -> tuple[dict, pd.DataFrame]:
    """统一口径评估：组合期间收益链（所有臂同一段代码，防口径漂移）。"""
    rows = []
    v = 1.0
    for t_prev, t_next in zip(decision_dates[:-1], decision_dates[1:]):
        w = weights.get(t_prev)
        if w is None:
            continue
        r = period_returns.loc[t_next].reindex(codes)
        port = float(np.nansum(w * np.nan_to_num(r.to_numpy(dtype=float))))
        idx = float(idx_ret.loc[t_next])
        to = 0.0                                        # 首期换手不定义，置 0
        if rows:
            to = float(np.nansum(np.abs(w - rows[-1]["w"])))
        v *= (1.0 + port)
        d = w - bw_row_cache.get(t_prev, np.zeros(len(codes)))
        rows.append({"date": t_next, "w": w, "port_ret": port, "idx_ret": idx,
                     "excess": port - idx, "turnover": to, "nav": v,
                     "te": float(np.linalg.norm(d))})
    eq = pd.DataFrame({r["date"]: {"nav": r["nav"]} for r in rows}).T
    eq.index.name = "date"
    if not rows:
        return {}, eq
    ex = np.array([r["excess"] for r in rows])
    n = len(rows)
    years = n * DECISION_FREQ / 244.0
    cagr = v ** (1.0 / years) - 1.0
    bench_cagr = float(np.prod([1.0 + r["idx_ret"] for r in rows])) ** (1.0 / years) - 1.0
    vol = float(np.std([r["port_ret"] for r in rows], ddof=1) * np.sqrt(244.0 / DECISION_FREQ))
    ex_vol = float(np.std(ex, ddof=1) * np.sqrt(244.0 / DECISION_FREQ))
    metrics = {
        "n_periods": n,
        "nav": float(v),
        "cagr": cagr,
        "bench_cagr": bench_cagr,
        "excess_cagr": cagr - bench_cagr,
        "vol": vol,
        "sharpe": (cagr - 0.0) / vol if vol > 0 else np.nan,
        "ir": float(np.mean(ex) / ex_vol) if ex_vol > 0 else np.nan,
        "avg_turnover": float(np.mean([r["turnover"] for r in rows[1:]])) if n > 1 else 0.0,
        "avg_te": float(np.mean([r["te"] for r in rows])),
        "max_dd": float((eq["nav"] / eq["nav"].cummax() - 1.0).min()),
    }
    return metrics, eq


# 全局缓存：评估时需要每期基准行（计算 TE 用）
bw_row_cache: dict[pd.Timestamp, np.ndarray] = {}

PPO_KW: dict = {"n_steps": 256, "batch_size": 256, "gamma": 0.99,
                "gae_lambda": 0.95, "clip_range": 0.2,
                "ent_coef": 0.01, "learning_rate": 3e-4}

from factor.rl.portfolio_env import PortfolioEnv  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--begin", default="2023-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--arms", default="ppo,qp,ew,topn")
    ap.add_argument("--ppo-timesteps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "portfolio_ppo_phase1"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    px, mask = load_panels(args.begin, args.end)
    # 成分掩码列含退市/长期停牌股（价格面板无数据），取交集；T 开头等脏码一并剔除
    codes = sorted(set(mask.columns) & set(px["close"].columns))
    log.info("成分掩码 %d 列 ∩ 价格面板 %d 列 → %d 码",
             mask.shape[1], px["close"].shape[1], len(codes))
    mask = mask.reindex(columns=codes).fillna(False)
    f, z = build_signals(px, codes)
    decision_dates, period_returns, bw, idx_ret = make_decision_frames(
        px, mask, codes, args.begin)
    bw_row_cache.update({d: bw.loc[d].to_numpy(dtype=float)
                         for d in decision_dates[:-1]})
    log.info("决策日 %d 个（每 %d 交易日）| codes=%d",
             len(decision_dates), DECISION_FREQ, len(codes))

    env_kwargs = dict(
        factor=f, aux=z, bench=bw,
        period_returns=period_returns,
        idx_period_returns=idx_ret,
        decision_dates=decision_dates,
        cost_rate=0.0, seed=args.seed,
        tradable_masks=bw.gt(0),
    )

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results: dict[str, dict] = {}
    equities: dict[str, pd.DataFrame] = {}
    for arm in arms:
        log.info("=== 臂：%s ===", arm)
        if arm == "ppo":
            weights, _, _ = run_arm_ppo(env_kwargs, args.ppo_timesteps,
                                        args.seed, out_dir)
        elif arm == "qp":
            weights, _ = run_arm_qp(px, codes, decision_dates, bw, f)
        elif arm == "topn":
            weights, _ = run_arm_topn(f, bw, codes, decision_dates)
        elif arm == "ew":
            weights = {d: bw.loc[d].to_numpy(dtype=float)
                       for d in decision_dates[:-1]}
        else:
            raise ValueError(f"未知臂 {arm!r}（ppo/qp/ew/topn）")
        metrics, eq = evaluate_arm(arm, weights, codes, decision_dates,
                                   period_returns, idx_ret)
        results[arm] = metrics
        equities[arm] = eq
        log.info("%s: 超额年化 %.2f%% | IR %.2f | 换手 %.2f | TE %.2f%%",
                 arm, metrics.get("excess_cagr", np.nan) * 100,
                 metrics.get("ir", np.nan), metrics.get("avg_turnover", np.nan),
                 metrics.get("avg_te", np.nan) * 100)

    summary = pd.DataFrame(results).T.drop(columns=[], errors="ignore")
    summary.to_csv(out_dir / "summary.csv")
    pd.concat({k: v["nav"] for k, v in equities.items()}, axis=1) \
        .to_csv(out_dir / "equity.csv")
    lines = ["# stage2 Phase 1：hs300 平价验证（信号代理=rev5，基准=成分等权）", "",
             "| 臂 | 年化 | 超额年化 | IR | 波动 | 平均换手 | 平均TE | 最大回撤 |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, m in results.items():
        lines.append(
            f"| {arm} | {m['cagr']:.2%} | {m['excess_cagr']:.2%} | {m['ir']:.2f} "
            f"| {m['vol']:.2%} | {m['avg_turnover']:.2f} | {m['avg_te']:.2%} "
            f"| {m['max_dd']:.2%} |")
    lines += ["", f"- 决策日 {len(decision_dates)} 个 / 每 {DECISION_FREQ} 交易日；"
              f"PPO timesteps={args.ppo_timesteps}, seed={args.seed}",
              "- 判定：Phase 1 只验证链路自洽（预期 PPO 打平或小输），不判 RL 有效性。",
              ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("已写出 %s", out_dir)


if __name__ == "__main__":
    main()
