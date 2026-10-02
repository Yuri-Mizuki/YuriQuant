"""phase2 PPO 深负超额诊断（09-27/10-02 全档 0/4 门槛事故）。

单 seed 短训练（10k 步）→ 回答三个问题：
1. 学出的权重形态是否病态（集中度/换手/方向性）？
2. 零动作基线（a=0 ≈ 贴基准）超额是否 ≈0（env 无 bug 的 sanity）？
3. 训练年(Y-2)与验证年(Y-1)的信号分布是否漂移（策略过反应的候选根因）？
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_YEAR = 2024
POOL, BEGIN, END = "zz1000", "2019-06-01", "2026-06-30"


def main():
    from stable_baselines3 import PPO
    from scripts.factors.run_portfolio_phase2 import (
        PPO_KW, DECISION_FREQ, make_env, load_pool_panels, build_frame,
        fit_label_models, assemble_fz)
    from factor.rl.portfolio_env import map_actions_to_weights

    px, mask = load_pool_panels(POOL, BEGIN, END)
    codes = list(mask.columns)
    df = build_frame(px, mask, codes)
    cache_dir = Path("reports/portfolio_phase2_full/signals")
    signals = {}
    for y in (MODEL_YEAR - 2, MODEL_YEAR - 1, MODEL_YEAR):
        models = fit_label_models(df, y - 1, 0, cache_dir)
        f_pred = pd.concat({lab: d["pred"] for lab, d in models.items()}, axis=1)
        signals[y] = assemble_fz(models, f_pred.index)

    all_dates = px["close"].index
    f_tr, z_tr = signals[MODEL_YEAR - 2][:2]
    f_va, z_va = signals[MODEL_YEAR - 1][:2]

    # Q3: 信号分布漂移
    for nm, f in [("tr(Y-2)", f_tr), ("va(Y-1)", f_va)]:
        s = pd.Series(f).dropna()
        print(f"[Q3] {nm} 信号 mean={s.mean():.4f} std={s.std():.4f} "
              f"p99={s.quantile(0.99):.3f} p1={s.quantile(0.01):.3f}")

    tr_dates = [d for d in all_dates
                if pd.Timestamp(f"{MODEL_YEAR-2}-01-01") <= d
                < pd.Timestamp(f"{MODEL_YEAR}-01-01")]
    env, _ = make_env(px, mask, codes, f_tr, z_tr, tr_dates, {}, 0)
    model = PPO("MlpPolicy", env, seed=0, device="cpu", verbose=0,
                **PPO_KW | {"n_steps": min(256, max(64, len(tr_dates)))})
    model.learn(total_timesteps=10_000)

    va_dates = [d for d in all_dates
                if pd.Timestamp(f"{MODEL_YEAR-1}-01-01") <= d
                <= pd.Timestamp(f"{MODEL_YEAR-1}-12-31")]
    half = len(va_dates) // 2
    for wi, wdates in enumerate([va_dates[:half], va_dates[half:]]):
        ev, _ = make_env(px, mask, codes, f_va, z_va, wdates, {}, 0)
        obs, _ = ev.reset()
        top5s, acts, ers = [], [], []
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = ev.step(a)
            w = info["weights"]
            top5s.append(float(np.sort(w)[::-1][:5].sum()))
            acts.append(float(np.abs(a).mean()))
            ers.append(float(info["reward"]["er"]))
            done = term or trunc
        cum_ex = float(np.sum(ers))
        print(f"[Q1] 窗{wi+1}: 累计超额={cum_ex*100:+.2f}% | "
              f"Top5集中度均值={np.mean(top5s):.2f} | 动作幅度均值={np.mean(acts):.3f} | "
              f"单步超额 min={min(ers)*100:+.2f}% max={max(ers)*100:+.2f}%")

        # Q2: 零动作基线（同一 env，a=0）
        ev2, _ = make_env(px, mask, codes, f_va, z_va, wdates, {}, 0)
        obs2, _ = ev2.reset()
        zero_a = np.zeros(ev2.action_space.shape, dtype=np.float32)
        ers0, done2 = [], False
        while not done2:
            obs2, r, term, trunc, info = ev2.step(zero_a)
            ers0.append(float(info["reward"]["er"]))
            done2 = term or trunc
        print(f"[Q2] 窗{wi+1} 零动作累计超额={float(np.sum(ers0))*100:+.2f}% "
              f"（≈0 则 env 无 bug）")


if __name__ == "__main__":
    main()
