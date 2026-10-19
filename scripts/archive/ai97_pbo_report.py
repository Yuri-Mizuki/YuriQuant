"""AI97 三臂 ×3 seed 的策略级过拟合检验（GOOD_MACHINE_TASKS 批次 2 收尾）。

读取 9 个 ``reports/alphapool_ppo_real_mlp_{arm}_seed{seed}.json`` 的评估段
（evaluation.test）逐因子日 IC 序列合成的**期间收益代理**不可得时，退化用
``equity``（若产物含逐日收益序列则直接用）。当前产物结构里可用的 T×N 矩阵是
**测试段逐日组合超额收益**：evaluation.test.portfolio.daily_excess（若存在）。

输出：CSCV PBO + deflate_best（DSR），落 reports/ai97_pbo/pbo_report.json。
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("none", "random", "llm_openai")
SEEDS = (0, 1, 2)


def _load_returns() -> pd.DataFrame:
    """T×N 矩阵：9 个臂的测试段逐日复合 IC（run_alphapool_ppo 补丁后产物）。"""
    cols = {}
    for arm, seed in product(ARMS, SEEDS):
        p = (ROOT / "reports" /
             f"alphapool_ppo_real_mlp_{arm}_seed{seed}.json")
        if not p.exists():
            print(f"[跳过] 缺产物: {p.name}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ev = d.get("evaluation", {}).get("test", {})
        ics = ev.get("composite_daily_ic")
        if isinstance(ics, list) and len(ics) >= 32:
            cols[f"{arm}_s{seed}"] = pd.Series(ics, dtype=float)
        else:
            print(f"[跳过] {p.name} 无 composite_daily_ic（旧代码产物，需重跑）")
    if len(cols) < 2:
        raise SystemExit(f"可用臂不足 2：{list(cols)}；先用补丁版重跑 9 臂")
    df = pd.DataFrame(cols).dropna(axis=0, how="any")
    print(f"PBO 矩阵: T={len(df)} × N={df.shape[1]}（列: {list(df.columns)}）")
    return df


def main() -> None:
    from stats.pbo import cscv_pbo, deflate_best

    rets = _load_returns()
    n_part = 16 if len(rets) >= 32 else 8
    pbo = cscv_pbo(rets, n_partitions=n_part)
    best = deflate_best(rets)
    out = {
        "pbo": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                for k, v in pbo.items() if k in ("pbo", "omega_mean", "logit_median",
                                                 "n_combinations", "n_trials")},
        "deflate_best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                         for k, v in best.items()},
    }
    dest = ROOT / "reports" / "ai97_pbo"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "pbo_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
