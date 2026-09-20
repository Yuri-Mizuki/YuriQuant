"""stage2 Phase 2：zz1000 主实验（信号管线 + PPO 滚动集成 / 银河口径 QP / 基准）。

设计依据（RESEARCH_TODO stage2 条目）：
- 信号 = L1 产出：三标签 GBDT 滚动预测（alpha_22/sharpe_22/mdd_66）+
  验证集 RankIC>0.05 质量门槛 + 回退（f=alpha_z+sharpe_z 按门槛逐年择用、
  全败回退零信号；z=mdd 预测，值越大风险越低，L1 已验证稳定）。
- PPO 臂 = 银河 0706 滚动协议：逐年训练 + 验证年双窗口双门槛 +
  多 seed 候选 Softmax 集成 + 全败回退。
- QP 臂 = 银河 0608 原始口径（主动空间、TE≤10%、换手≤20%、单股≤20%、
  风险中性软惩罚、换手软惩罚；top5≤50% 非凸→排序后处理投影近似）。
- 消融 = 银河 6 组（去 alpha / 去风险 / 去实际超额 / 去 TE+换手 / 完整 /
  pipeline 集成参照）；结果接 ``stats/pbo.py``。

用法（本机冒烟）::

    python -m scripts.factors.run_portfolio_phase2 --model-years 2024 \\
        --ppo-seeds 2 --ppo-retries 1 --ppo-timesteps 2048 \\
        --arms ppo,qp,ew --out reports/portfolio_phase2_smoke

生产（好机器，命令见 RESEARCH_TODO）：默认全量参数。
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

log = logging.getLogger("phase2")

DECISION_FREQ = 10
GATE = 0.05                     # 验证集 RankIC 质量门槛（研报口径）


# ===========================================================================
# 数据与信号
# ===========================================================================
def load_pool_panels(pool: str, begin: str, end: str
                     ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """行情宽表 + PIT 成分掩码（begin 前留 130 日预热）。"""
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from data.universe import Universe

    cache = DataCache(OfflineDataSource())
    fname = "daily_hs300.parquet" if pool == "hs300" else "daily_all_a.parquet"
    raw = pd.read_parquet(Path(str(cache.root)) / fname).reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    start = (pd.Timestamp(begin) - pd.Timedelta(days=130)).strftime("%Y-%m-%d")
    raw = raw[(raw["date"] >= start) & (raw["date"] <= pd.Timestamp(end))]
    px = {col: raw.pivot(index="date", columns="code", values=col).sort_index()
          for col in ("open", "high", "low", "close", "volume", "amount")}
    index_code = {"hs300": "000300.SH", "zz1000": "000852.SH"}[pool]
    dates = px["close"].index
    mask = Universe(cache).get_membership_mask(index_code, dates)
    codes = sorted(set(mask.columns) & set(px["close"].columns))
    mask = mask.reindex(index=dates, columns=codes).fillna(False)
    log.info("%s 面板 %d 日 × %d 码（并集∩价格）", pool, len(dates), len(codes))
    return px, mask


def build_frame(px: dict, mask: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """特征（classic+galaxy）+ 三标签的行式样本（成分内、标签非缺失）。"""
    from factor.classic import (
        build_galaxy_labels,
        compute_classic_features,
        compute_galaxy_features,
    )

    member = mask.reindex(px["close"].index).fillna(False)
    rets = px["close"][codes].pct_change(fill_method=None)
    mkt_ret = rets.where(member).mean(axis=1).fillna(0.0)
    idx_close = (1.0 + mkt_ret).cumprod()
    feats = compute_classic_features(px)
    feats.update(compute_galaxy_features(px, idx_close))
    labels = build_galaxy_labels(px, idx_close)
    # 内存友好：逐特征 stack 后立刻过滤到成分格（整面板 stack 的峰值
    # 在 zz1000 全历史 ≈ 48×4.5M float64 → 数 GB，实测会 OOM）
    member_stack = member.stack()
    cols: dict[str, pd.Series] = {}
    for k, v in feats.items():
        s = v.astype("float32").stack()
        cols[f"f_{k}"] = s[member_stack.reindex(s.index).fillna(False)]
    for k, v in labels.items():
        s = v.astype("float32").stack()
        cols[k] = s[member_stack.reindex(s.index).fillna(False)]
    df = pd.concat(cols, axis=1)
    df = df[df[[c for c in labels]].notna().all(axis=1)]
    df.index.names = ["date", "code"]
    return df


def fit_label_models(df: pd.DataFrame, year: int, seed: int,
                     cache_dir: Path) -> dict[str, dict]:
    """单模型年：3 年窗 80/20 训练三标签 LightGBM → 门控 + 测试年预测面板。

    Returns:
        {label: {"gate": bool, "valid_ic": float,
                 "pred": Series(date×code stack, 测试年)}}
    """
    import lightgbm as lgb
    from stats.robust_stats import nw_tstat

    cache_file = cache_dir / f"signals_{year}_seed{seed}.parquet"
    if cache_file.exists():
        log.info("信号缓存命中 %s", cache_file.name)
        out = {}
        gate_df = pd.read_parquet(cache_file / ".." / f"gates_{year}_seed{seed}.csv")
        for lab in ("alpha", "sharpe", "mdd"):
            # 写入时是 Series.to_frame()（MultiIndex+单列），读回取列即可，
            # 不能再 stack（会把列名压成第三层索引 → 索引错位）
            pred = pd.read_parquet(cache_file / f"pred_{lab}_{year}.parquet")["pred"]
            out[lab] = {"gate": bool(gate_df.loc[lab, "gate"]),
                        "valid_ic": float(gate_df.loc[lab, "valid_ic"]),
                        "pred": pred}
        return out

    dates = df.index.get_level_values(0)
    win = df[(dates >= pd.Timestamp(f"{year - 3}-01-01"))
             & (dates <= pd.Timestamp(f"{year}-12-31"))]
    udates = win.index.get_level_values(0).unique().sort_values()
    n_tr = int(len(udates) * 0.8)
    tr = win[win.index.get_level_values(0).isin(udates[:n_tr])]
    va = win[win.index.get_level_values(0).isin(udates[n_tr:])]
    te_dates = dates[(dates > pd.Timestamp(f"{year}-12-31"))
                     & (dates <= pd.Timestamp(f"{year + 1}-12-31"))]
    te = df[dates.isin(te_dates)]
    fcols = [c for c in df.columns if c.startswith("f_")]

    out: dict[str, dict] = {}
    for lab in ("alpha", "sharpe", "mdd"):
        col = f"label_{lab}_22" if lab != "mdd" else "label_mdd_66"
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1)
        model.fit(tr[fcols], tr[col], eval_set=[(va[fcols], va[col])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        va_pred = pd.Series(model.predict(va[fcols]), index=va.index)
        icdf = pd.concat({"p": va_pred, "y": va[col]}, axis=1).dropna()

        def _ic(g):
            return g["p"].corr(g["y"], method="spearman") if len(g) >= 15 else np.nan
        ic = icdf.groupby(level=0).apply(_ic).dropna()
        valid_ic = float(ic.mean()) if len(ic) else np.nan
        gate = bool(np.isfinite(valid_ic) and valid_ic > GATE)
        pred = pd.Series(model.predict(te[fcols]), index=te.index)
        out[lab] = {"gate": gate, "valid_ic": valid_ic, "pred": pred}
        log.info("  [%d] %s valid_ic=%.4f gate=%s", year, lab, valid_ic, gate)

    cache_dir.mkdir(parents=True, exist_ok=True)
    for lab, d in out.items():
        d["pred"].rename("pred").to_frame().to_parquet(
            cache_dir / f"pred_{lab}_{year}.parquet")
    pd.DataFrame({lab: {"gate": d["gate"], "valid_ic": d["valid_ic"]}
                  for lab, d in out.items()}).T.to_csv(
        cache_dir / f"gates_{year}_seed{seed}.csv")
    return out


def assemble_fz(models: dict[str, dict], index: pd.MultiIndex
                ) -> tuple[pd.Series, pd.Series, dict]:
    """门控装配：f = alpha_z(+sharpe_z)（按门槛，全败回退 0）；z = mdd 预测。"""
    parts, used = [], []
    for lab in ("alpha", "sharpe"):
        d = models.get(lab)
        if d is None:
            continue                       # 消融组显式剔除该信号
        if d["gate"]:
            parts.append(d["pred"].rename(lab))
            used.append(lab)
    if parts:
        f = pd.concat(parts, axis=1).mean(axis=1)
        f.index = index.intersection(f.index)
        f = f.reindex(index)
    else:
        f = pd.Series(0.0, index=index)            # 全败回退：零信号
    z = models["mdd"]["pred"].reindex(index)
    meta = {"used": used, "fallback": not parts}
    return f, z, meta


# ===========================================================================
# PPO 滚动集成（银河 0706 协议）
# ===========================================================================
def make_env(px, mask, codes, f, z, decision_dates, reward_kw, seed):
    from factor.rl.portfolio_env import PortfolioEnv
    # 信号可能是 stack 后的 (date,code) Series（逐年 GBDT 预测）→ 转面板，
    # 并 reindex 到本 env 的决策日/码表（缺失=0 中性）
    if isinstance(f, pd.Series):
        f = (f.unstack().reindex(index=decision_dates, columns=codes)
             .fillna(0.0))
    if isinstance(z, pd.Series):
        z = (z.unstack().reindex(index=decision_dates, columns=codes)
             .fillna(0.0))
    close = px["close"][codes]
    prev = close.shift(DECISION_FREQ)
    pr = (close / prev - 1.0).loc[[d for d in decision_dates[1:]
                                   if d in close.index]]
    bw = mask.reindex(index=decision_dates, columns=codes).fillna(False)
    bwm = bw.div(bw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    idx_ret = pd.Series({
        d: float(bwm.loc[d] @ (close.loc[d] / prev.loc[d] - 1.0).fillna(0.0))
        for d in decision_dates[1:] if d in close.index})
    return PortfolioEnv(
        factor=f, aux=z, bench=bwm,
        period_returns=pr, idx_period_returns=idx_ret,
        decision_dates=decision_dates, cost_rate=0.0, seed=seed,
        tradable_masks=bwm.gt(0), reward_kw=reward_kw,
        action_kw={"max_holding_pct": 0.4}), bwm


def ppo_arm_rolling(px, mask, codes, signals_by_year, all_dates, model_year, *,
                    timesteps, n_seeds, n_retries, seed0, reward_kw,
                    gate_reward=0.02, gate_excess=0.08):
    """银河滚动协议：验证年双窗口双门槛 → 候选 Softmax 集成 → 预测年权重。

    信号逐年口径：训练窗（Y-2 年）用 signal_model[Y-2] 的样本外预测、
    验证年（Y-1）双窗用 signal_model[Y-1]、预测年（Y）用 signal_model[Y]
    ——每个日历年的信号来自该年自己的模型（构造上即样本外），与银河
    「DL 模型逐年滚动、RL 在其输出的历年信号上训练」同构。
    """
    from stable_baselines3 import PPO

    def _sig(y):
        if y not in signals_by_year:
            raise KeyError(f"缺 {y} 年信号（--model-years 需覆盖 Y-2..Y）")
        f, z, _meta = signals_by_year[y]
        return f, z

    year_dates = [d for d in all_dates
                  if pd.Timestamp(f"{model_year}-01-01") <= d
                  <= pd.Timestamp(f"{model_year}-12-31")]
    if len(year_dates) < 4:
        return {}, {"status": "skip: no dates"}
    f_tr, z_tr = _sig(model_year - 2)
    f_va, z_va = _sig(model_year - 1)
    f_te, z_te = _sig(model_year)
    # 双窗口：验证年对半
    va_dates = [d for d in all_dates
                if pd.Timestamp(f"{model_year - 1}-01-01") <= d
                <= pd.Timestamp(f"{model_year - 1}-12-31")]
    half = len(va_dates) // 2
    windows = [va_dates[:half], va_dates[half:]]

    candidates = []
    attempts = 0
    for si in range(n_seeds):
        for retry in range(n_retries):
            attempts += 1
            seed = seed0 + si * 100 + retry
            tr_dates = [d for d in all_dates
                        if pd.Timestamp(f"{model_year - 2}-01-01") <= d
                        < pd.Timestamp(f"{model_year}-01-01")]
            env, _ = make_env(px, mask, codes, f_tr, z_tr, tr_dates,
                              reward_kw, seed)
            model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                        **PPO_KW | {"n_steps": min(256, max(64, len(tr_dates)))})
            model.learn(total_timesteps=timesteps)

            scores, excess = [], []
            for wdates in windows:
                ev, _ = make_env(px, mask, codes, f_va, z_va,
                                 wdates, reward_kw, seed)
                obs, _ = ev.reset()
                rew, ex = [], []
                done = False
                while not done:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = ev.step(a)
                    rew.append(r)
                    ex.append(info["reward"]["er"])
                    done = term or trunc
                scores.append(float(np.mean(rew)))
                excess.append(float(np.sum(ex)))
            sc = sum(1 for s in scores if s > gate_reward) \
                + sum(1 for e in excess if e > gate_excess)
            log.info("  PPO seed%d retry%d: score=%d/4 (rew=%s ex=%s)",
                     si, retry, sc, [f"{s:.3f}" for s in scores],
                     [f"{e:.2%}" for e in excess])
            if sc >= 2:
                candidates.append({"seed": seed, "model": model,
                                   "excess": float(np.sum(excess))})
                break                                  # 达标提前结束该 seed 区间
    if not candidates:
        return {}, {"status": "fallback", "attempts": attempts}

    tau = 0.05
    exs = np.array([c["excess"] for c in candidates])
    wts = np.exp((exs - exs.max()) / tau)
    wts /= wts.sum()
    # 集成：各候选对预测年 rollout，逐期 Softmax 加权平均权重
    env, bwm = make_env(px, mask, codes, f_te, z_te, year_dates, reward_kw, seed0)
    per_model = []
    for c in candidates:
        ev, _ = make_env(px, mask, codes, f_te, z_te, year_dates, reward_kw, c["seed"])
        obs, _ = ev.reset()
        seq, done = [], False
        while not done:
            a, _ = c["model"].predict(obs, deterministic=True)
            obs, _r, term, trunc, info = ev.step(a)
            seq.append((info["date"], info["weights"]))
            done = term or trunc
        per_model.append(dict(seq))
    merged = {}
    for date, _w in per_model[0]["seq"]:
        acc = np.zeros(len(codes))
        for w, c in zip(wts, per_model):
            dm = dict(c["seq"])
            acc = acc + w * dm[date]
        merged[date] = acc / acc.sum()
    return merged, {"status": "ok", "n_candidates": len(candidates),
                    "attempts": attempts}


PPO_KW: dict = {"n_steps": 256, "batch_size": 256, "gamma": 0.99,
                "gae_lambda": 0.95, "clip_range": 0.2,
                "ent_coef": 0.01, "learning_rate": 3e-4}


# ===========================================================================
# 银河口径 QP（0608 原始：主动空间 + TE/换手硬约束 + 风险中性软惩罚）
# ===========================================================================
def solve_galaxy_qp(alpha, risk_t, sigma, w_b, w_prev, *, lam_risk=1.0,
                    lam_aux=2.0, lam_turn=0.01, max_weight=0.2, te=0.10,
                    turnover=0.20, tradable=None, shrink_eqv=0.2):
    import cvxpy as cp

    n = len(alpha)
    mv = float(np.mean(np.diag(sigma)))
    sigma = shrink_eqv * mv * np.eye(n) + (1.0 - shrink_eqv) * sigma
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= max_weight,
            cp.sum_squares(w - w_b) <= te ** 2,
            cp.sum(cp.abs(w - w_prev)) <= 2.0 * turnover]
    if tradable is not None:
        nm = np.flatnonzero(~np.asarray(tradable, dtype=bool))
        if len(nm):
            cons.append(w[nm] == 0)
    # 银河目标 max α'w − λr·wΣw + soft(−λaux(r̃'w)² − λturn‖Δw‖₁)
    # 等价最小化形式（soft 惩罚项均为正号凸项）：
    obj = (cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))
                       - float(lam_risk) * (alpha @ w)
                       + float(lam_aux) * cp.square(risk_t @ w)
                       + float(lam_turn) * cp.sum(cp.abs(w - w_prev))))
    prob = cp.Problem(obj, cons)
    # norm1 约束+惩罚与 quad_form 组合 OSQP 不接受（cvxpy reduction 限制），
    # SCS 锥求解器可解；N≈1000 子集秒级
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=200000)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"galaxy QP status={prob.status}")
    wv = np.asarray(w.value, dtype=float).reshape(-1)
    # top5≤50%（非凸，排序后处理近似）：超标则向基准方向收缩
    for _ in range(8):
        top5 = np.sort(wv)[::-1][:5].sum()
        if top5 <= 0.5:
            break
        wv = 0.8 * wv + 0.2 * w_b
        wv /= wv.sum()
    return wv


def qp_arm_galaxy(px, mask, codes, f, z, decision_dates, bw, reward_kw):
    from optimize.solver import estimate_covariance

    close = px["close"][codes]
    rets = close.pct_change(fill_method=None)
    w_prev = bw.iloc[0].to_numpy(dtype=float).copy()
    weights, n_fallback = {}, 0
    for t in decision_dates[:-1]:
        hist = rets.loc[:t].iloc[:-1].tail(120)
        valid = (bw.loc[t] > 0) & hist.notna().all()
        cols = valid.index[valid]
        if len(cols) < 60:
            weights[t] = bw.loc[t].to_numpy(dtype=float)
            n_fallback += 1
            continue
        sigma = estimate_covariance(hist[cols], method="ledoit_wolf",
                                    shrinkage=0.5)
        # 风险中性项的 r̃ = z − w_bᵀz（相对基准平均暴露中心化）
        z_t = z.loc[t].reindex(cols).fillna(0.0).to_numpy(dtype=float)
        wb = bw.loc[t].reindex(cols).fillna(0.0).to_numpy(dtype=float)
        wbm = wb / wb.sum() if wb.sum() > 0 else np.full(len(cols), 1 / len(cols))
        r_tilde = z_t - float(wbm @ z_t)
        a = f.loc[t].reindex(cols).fillna(0.0).to_numpy(dtype=float)
        try:
            w_sub = solve_galaxy_qp(a, r_tilde, sigma, wbm,
                                    w_prev[valid.to_numpy(dtype=bool)])
        except RuntimeError:
            n_fallback += 1
            weights[t] = bw.loc[t].to_numpy(dtype=float)
            continue
        w = np.zeros(len(codes))
        w[valid.to_numpy(dtype=bool)] = w_sub
        weights[t] = w
        w_prev = w
    if n_fallback:
        log.warning("QP 臂 %d 期回退基准", n_fallback)
    return weights


# ===========================================================================
# 评估（与 Phase 1 同口径）
# ===========================================================================
def evaluate(name, weights, codes, decision_dates, period_returns, bw):
    from scripts.factors.run_portfolio_ppo import evaluate_arm
    bw_row_cache = {d: bw.loc[d].to_numpy(dtype=float)
                    for d in decision_dates[:-1]}
    import scripts.factors.run_portfolio_ppo as p1
    p1.bw_row_cache.update(bw_row_cache)
    return p1.evaluate_arm(name, weights, codes, decision_dates,
                           period_returns, None) \
        if False else _evaluate(name, weights, codes, decision_dates,
                                period_returns, bw)


def _evaluate(name, weights, codes, decision_dates, period_returns, bw):
    rows, v = [], 1.0
    for t_prev, t_next in zip(decision_dates[:-1], decision_dates[1:]):
        w = weights.get(t_prev)
        if w is None:
            continue
        r = period_returns.loc[t_next].reindex(codes)
        port = float(np.nansum(w * np.nan_to_num(r.to_numpy(dtype=float))))
        wb = bw.loc[t_prev].to_numpy(dtype=float)
        idx = float(np.nansum(wb * np.nan_to_num(r.to_numpy(dtype=float))))
        to = 0.0
        if rows:
            to = float(np.nansum(np.abs(w - rows[-1]["w"])))
        v *= (1.0 + port)
        rows.append({"date": t_next, "w": w, "port_ret": port, "idx_ret": idx,
                     "excess": port - idx, "turnover": to, "nav": v})
    eq = pd.DataFrame({r["date"]: {"nav": r["nav"]} for r in rows}).T
    if not rows:
        return {"n_periods": 0}, eq
    ex = np.array([r["excess"] for r in rows])
    years = len(rows) * DECISION_FREQ / 244.0
    cagr = v ** (1.0 / years) - 1.0
    bcagr = float(np.prod([1 + r["idx_ret"] for r in rows])) ** (1 / years) - 1
    vol = float(np.std([r["port_ret"] for r in rows], ddof=1)
                * np.sqrt(244.0 / DECISION_FREQ))
    exv = float(np.std(ex, ddof=1) * np.sqrt(244.0 / DECISION_FREQ))
    return {
        "n_periods": len(rows), "nav": float(v), "cagr": cagr,
        "excess_cagr": cagr - bcagr,
        "ir": float(np.mean(ex) / exv) if exv > 0 else np.nan,
        "avg_turnover": float(np.mean([r["turnover"] for r in rows[1:]]))
            if len(rows) > 1 else 0.0,
        "max_dd": float((eq["nav"] / eq["nav"].cummax() - 1).min()),
    }, eq


# ===========================================================================
# 主流程
# ===========================================================================
ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_alpha": {"drop_alpha": True},
    "no_risk": {"reward": {"lambda_aux": 0.0}},
    "no_excess": {"reward": {"lambda_excess": 0.0}},
    "no_te_turn": {"reward": {"lambda_tracking": 0.0, "lambda_turnover": 0.0}},
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="zz1000", choices=["zz1000", "hs300"])
    ap.add_argument("--begin", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--model-years", default="2024,2025")
    ap.add_argument("--arms", default="ppo,qp,ew")
    ap.add_argument("--ablations", default="full",
                    help="逗号分隔：full,no_alpha,no_risk,no_excess,no_te_turn")
    ap.add_argument("--ppo-seeds", type=int, default=5)
    ap.add_argument("--ppo-retries", type=int, default=3)
    ap.add_argument("--ppo-timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal-cache", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "portfolio_phase2"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.signal_cache) if args.signal_cache \
        else out_dir / "signals"

    px, mask = load_pool_panels(args.pool, args.begin, args.end)
    codes = list(mask.columns)
    df = build_frame(px, mask, codes)
    log.info("信号样本 %d 行", len(df))

    # ---- 信号：三标签 GBDT 按日历年装配 ----
    # 约定：fit_label_models(Y) 训练到 Y 年底、预测 Y+1 年（te 即 Y+1）。
    # 因此日历年 T 的样本外信号 = 模型年 T-1 的输出。RL 模型年 Y 需要
    # 日历年 Y-2（训练窗）/Y-1（验证年）/Y（预测年）三年的信号
    # → 拟合模型年 Y-3..Y-1。首年模型窗不足 3 年自动缩短（与研报
    # 「2020 年模型仅两年数据」同口径）。
    model_years = [int(y) for y in args.model_years.split(",")]
    needed_cal_years = sorted({y for my in model_years
                               for y in (my - 2, my - 1, my)})
    year_signals: dict[int, tuple[pd.Series, pd.Series, dict]] = {}
    for cal_year in needed_cal_years:
        models = fit_label_models(df, cal_year - 1, args.seed, cache_dir)
        f_pred = pd.concat({lab: d["pred"] for lab, d in models.items()}, axis=1)
        f, z, meta = assemble_fz(models, f_pred.index)
        year_signals[cal_year] = (f, z, meta)
        log.info("信号日历年 %d 装配（模型年 %d）：%s",
                 cal_year, cal_year - 1, meta)

    # ---- 决策日框架（预测年逐臂复用）----
    dates = px["close"].index
    eval_dates = dates[dates >= pd.Timestamp(args.begin)]
    decision_dates = list(eval_dates[::DECISION_FREQ])

    results, equities = {}, {}
    for year in model_years:
        f, z, _meta = year_signals[year]
        y_dates = [d for d in decision_dates
                   if pd.Timestamp(f"{year}-01-01") <= d
                   <= pd.Timestamp(f"{year}-12-31")]
        if len(y_dates) < 4:
            log.warning("模型年 %d 决策日不足，跳过", year)
            continue
        close = px["close"][codes]
        prev = close.shift(DECISION_FREQ)
        period_returns = (close / prev - 1.0).loc[y_dates[1:]]
        bw = mask.reindex(index=y_dates, columns=codes).fillna(False)
        bwm = bw.div(bw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        # 消融组只作用于 PPO 臂（银河消融=RL 奖励函数消融）；QP/EW/TopN 只跑 full
        for abl in [a.strip() for a in args.ablations.split(",")]:
            cfg = ABLATIONS[abl]
            reward_kw = {**cfg.get("reward", {})}
            f_use = f
            if cfg.get("drop_alpha"):
                # 去 alpha 信号：门控装配剔除 alpha 后重装配（sharpe 若未过门槛
                # 则为零信号=回退）
                # 信号年口径：日历年 T 的信号 = 模型年 T-1（与 year_signals 一致，
                # 此处命中缓存不再重训）
                sub_models = {k: v for k, v in
                              (fit_label_models(df, year - 1, args.seed,
                                                cache_dir) or {}).items()}
                sub_models.pop("alpha", None)
                f_use, _z2, _m2 = assemble_fz(sub_models, f.index)

            arms_for_abl = ([a.strip() for a in args.arms.split(",")]
                            if abl == "full"
                            else ["ppo"]) if "ppo" in args.arms else                            [a.strip() for a in args.arms.split(",")]
            for arm in arms_for_abl:
                key = f"{year}_{abl}_{arm}"
                log.info("=== %s ===", key)
                if arm == "ppo":
                    weights, info = ppo_arm_rolling(
                        px, mask, codes, year_signals, eval_dates, year,
                        timesteps=args.ppo_timesteps, n_seeds=args.ppo_seeds,
                        n_retries=args.ppo_retries, seed0=args.seed,
                        reward_kw=reward_kw)
                    results[key + "|meta"] = {"info": json.dumps(info)}
                    if info.get("status") != "ok":
                        weights = {d: bwm.loc[d].to_numpy(dtype=float)
                                   for d in y_dates[:-1]}      # 回退持有基准
                elif arm == "qp":
                    weights = qp_arm_galaxy(px, mask, codes, f_use, z,
                                            y_dates, bwm, reward_kw)
                elif arm == "ew":
                    weights = {d: bwm.loc[d].to_numpy(dtype=float)
                               for d in y_dates[:-1]}
                elif arm == "topn":
                    weights = {}
                    for t in y_dates[:-1]:
                        sig = f_use.loc[t].reindex(codes)
                        sig = sig.where(bwm.loc[t] > 0)
                        pick = sig.nlargest(max(1, len(codes) // 10)).index
                        w = pd.Series(0.0, index=codes)
                        w[pick] = 1.0 / max(len(pick), 1)
                        weights[t] = w.to_numpy(dtype=float)
                else:
                    raise ValueError(arm)
                m, eq = _evaluate(key, weights, codes, y_dates,
                                  period_returns, bwm)
                results[key] = m
                equities[key] = eq
                log.info("%s: 超额 %.2f%% | IR %.2f", key,
                         m.get("excess_cagr", np.nan) * 100,
                         m.get("ir", np.nan))

    pd.DataFrame({k: v for k, v in results.items()
                  if "meta" not in k}).T.to_csv(out_dir / "summary.csv")
    pd.concat({k: v["nav"] for k, v in equities.items()}, axis=1) \
        .to_csv(out_dir / "equity.csv")
    (out_dir / "results.json").write_text(
        json.dumps({k: {kk: (vv if not isinstance(vv, float) else round(vv, 6))
                        for kk, vv in v.items()} for k, v in results.items()},
                   ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    log.info("已写出 %s", out_dir)


if __name__ == "__main__":
    main()
