"""因子判据**唯一真源** —— 准入 / 晋升 / 组合三段的判据都在这里取（2026-11-19 建立）。

## 为什么有这个模块

在此之前，「一条因子有没有效」在仓里有 **13 处独立实现的截面秩 IC**、**26 处残差
正交**、**24 处裸调 Newey-West**，阈值散在 ``gate_core.DEFAULTS`` /
原晋升门槛（已退役）/ 各探针的 argparse 默认值等多处。后果是同一个概念
能给出两个答案，且无法横向比较。

本模块把**判据**收口成一份纯函数实现。它**不做 IO**（不读面板、不读 registry），
所有输入是已经对齐的 DataFrame —— 数据装载留给调用方（``scripts/evaluation/gate.py``、
``scripts/evaluation/gate_core.py``）。

## 三段流水线（**不是一个入口**，见下）

| 段 | 问的问题 | 函数 | 通过后状态 |
|---|---|---|---|
| ① 准入 | 对**现有体系**有没有增量 | :func:`verdict` L1~L5 | 可入库，``experimental`` |
| ~~② 晋升~~ | （2026-11-22 随晋升系统退役；原「独立成腿」问题由看板双尾/信号层 A/B 承接） | — | — |
| ③ 组合 | 换成钱还剩多少 | ``caliber.evaluate_weights``（本模块不碰） | 进主链 |

🚨 **为什么三段的差异不是"没统一"**：①问的是「相对主信号的增量」，②问的是
「独立成腿的绝对业绩」（``sharpe_ls_M ≥ 0.5``）。两个答案经常相反——单腿夏普 0.8
但和主信号 ρ=0.9 的因子，②放行、①判死；单腿夏普 −0.04 但能稳定识别"该躲那一端"
的因子，②判死、①能给出"SIGNFLIP，方向不可交易"这类有用信息（2026-11-18 cyq 实证）。
把①③合并成②会系统性偏好"自己能赚但和主信号重复"的因子。

🚨 **也不是"所有脚本一个入口"**：仓里 60 个 ``_probe`` 是一次性实验装置（事件窗口 /
择时条件 / 执行相位），各有各的样本构造，产物是 ``reports/`` 里的结论不是工具。
收口的是**判据函数**，不是流程。

## 阈值在哪

- 准入阈值：本模块 :data:`ADMISSION_GATES`（**唯一处**）
- （原晋升阈值：随晋升系统 2026-11-22 退役；:func:`promotion_gates` 现为
  :data:`SEGMENT_DEFAULTS` 的兼容名）
- 重复性阈值：``research.dup_criteria``（**唯一处**，本模块只转发）
- 口径窗口 ``caliber.WINDOWS`` **不在此列** —— 那是数据口径定义，改了历史结论作废，
  与"筛子松紧"性质不同，故意分开。

## 🚨 关于 t 值（2026-11-18 实测，最贵的一条）

日频 2 年半、5000 只截面的残差 IC 序列自相关很强，``t = IC × √T`` 会把 0.003 的
噪声放大成 t≈4。cyq 探针里 **逐列循环平移的安慰剂拿到 ``resfac_t = 4.24``，比真实
因子的 3.85 还高**。所以：

- ❌ 不看绝对 t 值；
- ✅ 看 :func:`effect_size`（真实 IC ÷ 安慰剂 IC），**两个 p 都要报**；
- ✅ 过了 L5 也只是"可以进组合层"，不等于赚钱（IC ≠ 经济价值）。
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from research.dup_criteria import (
    CORR_TRIGGER,
    MIN_SPAN,
    REL_STD_DEGENERATE,
    REL_STD_TINY,
    RESID_IC_NEGLIGIBLE,
    Verdict,
    classify_independent,
    should_check,
    tier_of,
)
from research.dup_criteria import (
    classify as dup_classify,
)
from research.dup_criteria import (
    explain as dup_explain,
)
from stats.ic import ic_stats

# ---------------------------------------------------------------------------
# 阈值（准入段唯一处；晋升/重复性段的真源在别处，这里只转发）
# ---------------------------------------------------------------------------
#: 准入（L1~L5）门槛。**这是筛子不是判据** —— 过了线只代表"值得跑组合层"。
#:
#: * ``min_cov`` = 0.30：面板非空覆盖率下限。低于它说明数据没铺开，不是因子无效。
#: * ``min_n`` = 1000：有效格子数下限（掩码内、有前瞻收益、有因子值）。
#: * ``ic_floor`` = 0.005：无条件 IC 入场线。**与 ``dup_criteria.RESID_IC_NEGLIGIBLE``
#:   （0.02）不可比** —— 后者是"残差小到可删"的量级，用途相反；也**与
#:   亦与旧晋升系统的 ``min_resid_ic``（0.01，2026-11-22 随系统退役）不可比。
#: * ``gap_floor`` / ``eff_ratio``：与安慰剂的绝对差与倍数，两者都要过。
ADMISSION_GATES: dict[str, float] = {
    "min_cov": 0.30,
    "min_n": 1000,
    "ic_floor": 0.005,
    "gap_floor": 0.003,
    "eff_ratio": 2.0,
}


#: IS/OOS 分段参数（2026-11-22 起**唯一真源在本模块**）。原转发
#: ``factor_library.MATURITY_GATES``，晋升系统已于 2026-11-22 退役
#: （零消费方 + 口径变动下静默腐烂）；这三个数只服务 :func:`seg_stability`
#: 的 IS/OOS 切分，与晋升无关，故留此承载。
SEGMENT_DEFAULTS: dict[str, float] = {
    "split_frac": 0.7,
    "recent_days": 250,
    "min_seg_days": 60,
}


def promotion_gates() -> dict[str, float]:
    """兼容名：返回 :data:`SEGMENT_DEFAULTS`（原晋升门槛已随系统退役）。"""
    return dict(SEGMENT_DEFAULTS)


def all_thresholds() -> dict[str, dict]:
    """全部判据阈值的一站式视图（便于审计"有没有第二份定义"）。"""
    return {
        "admission": dict(ADMISSION_GATES),
        "segmentation": dict(SEGMENT_DEFAULTS),
        "dup": {
            "corr_trigger": CORR_TRIGGER,
            "rel_std_degenerate": REL_STD_DEGENERATE,
            "rel_std_tiny": REL_STD_TINY,
            "resid_ic_negligible": RESID_IC_NEGLIGIBLE,
            "min_span": MIN_SPAN,
        },
    }


def qualifier() -> str:
    """一行版判据说明（日志/报错引用，避免两处各写一套措辞）。"""
    return (f"准入：cov≥{ADMISSION_GATES['min_cov']:g} 且 |IC|≥"
            f"{ADMISSION_GATES['ic_floor']:g} 且 与安慰剂 gap≥"
            f"{ADMISSION_GATES['gap_floor']:g}、比值≥{ADMISSION_GATES['eff_ratio']:g}×。"
            + dup_explain())


# ---------------------------------------------------------------------------
# L1 覆盖
# ---------------------------------------------------------------------------
def panel_coverage(x: pd.DataFrame, fwd: pd.DataFrame,
                   mask: pd.DataFrame) -> dict:
    """覆盖率与有效格子数。

    ``cov`` 是**原始**非空率（不看掩码与前瞻收益），``cov_eff`` 才是三者交集
    —— 判 DATA 要看后者，前者只说明面板本身铺得开不开。
    """
    valid = x.notna() & fwd.notna() & mask
    return {
        "cov": float(x.notna().to_numpy(dtype=float).mean()),
        "cov_eff": float(valid.to_numpy(dtype=bool).mean()),
        "n_valid": int(valid.to_numpy(dtype=bool).sum()),
        "n_days": int(x.notna().any(axis=1).sum()),
    }


# ---------------------------------------------------------------------------
# L2/L3 IC 四口径
# ---------------------------------------------------------------------------
def resid_rank(sig_rank: pd.DataFrame, main_rank: pd.DataFrame) -> pd.DataFrame:
    """逐日一元 OLS 残差（对主信号正交）。

    与 ``residual_ic_probe.resid_rank`` / ``text_factor_probe.residual_panel``
    同一实现；``tests/test_factor_gate.py`` 锁死三者逐位一致。

    🚨 **只在秩空间做**：原始值 OLS 会给纯单调变换留下曲线残差（伪增量，
    ``dup_criteria`` 实测 ``VSUMN30`` vs ``VSUMD30`` 秩相关恒 −1 但 raw t=+25.4）。
    """
    mm = main_rank.reindex(index=sig_rank.index, columns=sig_rank.columns)
    v = sig_rank.notna() & mm.notna()
    a = sig_rank.where(v)
    b = mm.where(v)
    ya = a.sub(a.mean(axis=1), axis=0)
    yb = b.sub(b.mean(axis=1), axis=0)
    var = (yb ** 2).mean(axis=1)
    cov = (ya * yb).mean(axis=1)
    beta = cov / var.replace(0.0, np.nan)
    return ya.sub(yb.mul(beta, axis=0)).where(v)


def yearly_ic(ic: pd.Series) -> dict:
    """逐年 IC 均值 + 正年数（方向稳定性；8/9 也可能是噪声，见 L5）。"""
    vals = [float(sub.mean()) for _y, sub in ic.dropna().groupby(
        ic.dropna().index.year)]
    n = len(vals)
    pos = int(sum(v > 0 for v in vals))
    return {"years_pos": pos, "years_n": n,
            "years_sign_consistent": int(max(pos, n - pos))}


def ic_calibers(x: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
                main: pd.DataFrame) -> dict | None:
    """单因子四口径 IC（与 ``residual_ic_probe.one_factor`` 同定义）。

    | 口径 | 因子 | 目标 | 用途 |
    |---|---|---|---|
    | ``raw`` | 原值 | 原值 | L2 原始预测力 |
    | ``resfac`` | 正交主信号 | 原值 | **L3 增量（主判据）** |
    | ``restgt`` | 原值 | 正交主信号 | 旁证（收益里已被解释的部分） |
    | ``both`` | 正交 | 正交 | 双正交（最保守，会低估） |

    ``resfac`` 是主判据：它问"扣掉主信号能解释的那部分，这条因子还剩多少预测力"。

    🚨 返回值里的 ``rank_corr_main`` 是**与主信号的共线度**，不是换皮检测。
    真正的换皮（vs 全库 950 因子）走 :func:`research.dup_criteria.classify`。
    """
    valid = x.notna() & fwd.notna() & mask
    if int(valid.to_numpy(dtype=bool).sum()) < 1:
        return None
    v = valid & main.notna()
    if int(v.to_numpy(dtype=bool).sum()) < 1000:
        return None

    xr_raw = x.where(valid).rank(axis=1)
    yr_raw = fwd.where(valid).rank(axis=1)

    xr = x.where(v).rank(axis=1)
    mr = main.where(v).rank(axis=1)
    yr = fwd.where(v).rank(axis=1)

    xres = resid_rank(xr, mr)
    xres_r = xres.where(v).rank(axis=1)
    fres = resid_rank(yr, mr)
    fres_r = fres.where(v).rank(axis=1)

    series = {
        "raw": xr_raw.corrwith(yr_raw, axis=1).astype(float),
        "resfac": xres_r.corrwith(yr, axis=1).astype(float),
        "restgt": xr.corrwith(fres_r, axis=1).astype(float),
        "both": xres_r.corrwith(fres_r, axis=1).astype(float),
    }
    row: dict = {"rank_corr_main": float(xr.corrwith(mr, axis=1).mean())}
    for tag, s in series.items():
        for k, val in ic_stats(s).items():
            row[f"{tag}_{k}"] = val
    row.update(yearly_ic(series["resfac"]))
    return row


def rank_corr(a: pd.DataFrame, b: pd.DataFrame, mask: pd.DataFrame) -> float:
    """逐日截面秩相关的均值（``a``/``b`` 需事先对齐）。"""
    v = a.notna() & b.notna() & mask
    if not v.to_numpy(dtype=bool).any():
        return float("nan")
    return float(a.where(v).rank(axis=1)
                 .corrwith(b.where(v).rank(axis=1), axis=1).mean())


# ---------------------------------------------------------------------------
# 时间稳定性（复用 factor_library 的实现，不另写）
# ---------------------------------------------------------------------------
def seg_stability(dret: pd.Series, ic: pd.Series, *,
                  split_frac: float | None = None,
                  recent_days: int | None = None) -> dict:
    """前段 / 后段 / 近 N 日三段统计（准入后段稳定性判据用）。

    直接转发 ``factor_library.window_stats``（**唯一实现**），默认值取
    :data:`SEGMENT_DEFAULTS`（晋升系统退役后分段常量的唯一住址）。

    Args:
        dret: 净多空日收益（``eval`` 的 ``dret_ls_M``）；拿不到就传空 Series，
            此时 ``sharpe_*`` 全 NaN，但 ``ic_back`` / ``ic_fwd`` 仍可用。
        ic: 逐日 IC（NaN = 当日无有效截面）。
    """
    g = SEGMENT_DEFAULTS
    from research.factor_library import window_stats

    return window_stats(
        pd.Series(dret).astype(float), pd.Series(ic).astype(float),
        split_frac=float(g["split_frac"] if split_frac is None else split_frac),
        recent_days=int(g["recent_days"] if recent_days is None else recent_days),
        min_days=int(g["min_seg_days"]),
    )


# ---------------------------------------------------------------------------
# L5 安慰剂
# ---------------------------------------------------------------------------
def cyclic_placebo(x: pd.DataFrame, seed: int, name: str = "",
                   lo: int = 120, hi: int = 900) -> pd.DataFrame:
    """逐列**循环平移**：保留每条序列的分布与自相关，打掉与收益的时点对应。

    每列独立随机平移 ``U[lo, hi)`` 个交易日。平移量由 ``(seed, name)`` 唯一确定
    ⇒ 同一因子每次重跑得到**同一份**安慰剂（可复现），且不随同批其他因子变化。

    🚨 ``hi`` 必须显著小于样本长度：平移量接近 T 时序列几乎回到原位，安慰剂会
    泄漏真实信息。``roll`` 是循环的，``k`` 若是 ``T`` 的整数倍就绕回原位、
    安慰剂等于原因子（实现在 ``hi`` 超过 ``T/2`` 时自动收紧，T 太短直接报错）。
    1864 日的面板上 ``[120, 900)`` 是安全的。

    🚨 平移量用 ``crc32(name)`` 而非 ``hash(name)``：后者带进程级随机盐
    （PYTHONHASHSEED），同一因子换个进程就换一份安慰剂，不可复现。
    """
    n = len(x)
    if n < 2 * lo + 1:
        raise ValueError(
            f"序列长度 {n} 太短，无法做循环平移（需 ≥ {2 * lo + 1} 日）；"
            f"短样本请改用其它安慰剂形式")
    hi = min(int(hi), n // 2)
    if hi <= lo:
        raise ValueError(f"平移区间 [{lo}, {hi}) 为空（n={n}）")
    seed_name = zlib.crc32(name.encode("utf-8"))
    rng = np.random.default_rng([int(seed), int(seed_name)])
    arr = x.to_numpy(dtype=np.float32, copy=True)
    for j in range(arr.shape[1]):
        k = int(rng.integers(lo, hi))
        arr[:, j] = np.roll(arr[:, j], k)
    return pd.DataFrame(arr, index=x.index, columns=x.columns)


def effect_size(real_ic: float, plc_ic: float) -> dict:
    """效应量：真实 IC 相对安慰剂 IC。

    - ``eff_ratio``：同号时 = ``real/plc``；异号或 plc≈0 时为 NaN（比值无意义）。
    - ``eff_gap``：``real − plc``（绝对差，不受比值爆炸影响）。

    两者都要报：比值说明"比噪声强几倍"，差值说明"绝对值够不够大"。
    """
    gap = real_ic - plc_ic
    same_sign = np.sign(real_ic) == np.sign(plc_ic) and abs(plc_ic) > 1e-9
    ratio = float(real_ic / plc_ic) if same_sign else float("nan")
    return {"eff_ratio": ratio, "eff_gap": float(gap)}


# ---------------------------------------------------------------------------
# ① 准入判定（筛子，不是结论）
# ---------------------------------------------------------------------------
def admission_verdict(row: dict, th: dict | None = None) -> tuple[str, str]:
    """按 L1~L5 给出**准入**标记与理由。返回 ``(tag, why)``。

    🚨 这里**不下最终结论**。``PASS`` 只代表"值得进组合层"，
    因为 IC ≠ 经济价值 —— 必须再跑 Top-N。

    🚨 本函数**不检查 sharpe**：IC ≠ 经济价值，组合层见 ``caliber``；
    原晋升段已于 2026-11-22 退役，见模块 docstring。
    """
    t = {**ADMISSION_GATES, **(th or {})}
    # 🚨 门槛按 ``cov``（原始非空率）标定，不用 ``cov_eff``：后者还叠了掩码与
    # 前瞻收益（掩码覆盖率约 0.96），换口径等于换门槛松紧，必须重新标定。
    # ``cov_eff`` 照样输出到表里，供人工判断"面板铺得开但有效格少"这种情形。
    if not np.isfinite(row.get("cov", np.nan)) or row["cov"] < t["min_cov"]:
        return "DATA", f"覆盖率 {row.get('cov', float('nan')):.2f} < {t['min_cov']}"
    if row.get("n_valid", 0) < t["min_n"]:
        return "DATA", f"有效格 {row.get('n_valid', 0)} < {t['min_n']:.0f}"

    raw, res = row.get("raw_ic", np.nan), row.get("resfac_ic", np.nan)
    if not np.isfinite(raw) or abs(raw) < t["ic_floor"]:
        return "NOIC", f"raw IC {raw:+.4f} 不足 {t['ic_floor']}"
    if not np.isfinite(res) or abs(res) < t["ic_floor"]:
        return "ABSORBED", f"残差 IC {res:+.4f} 不足（信息被主信号吸收）"

    plc = row.get("plc_ic", np.nan)
    if not np.isfinite(plc):
        return "NOPLC", "安慰剂缺失 —— 按铁律不得判增益"
    # 符号翻转：raw 与残差方向相反 ⇒ 原始预测力全部来自与主信号共线的那部分，
    # 扣掉之后方向反转。**不是自动拒绝**，而是"方向只能取残差那一侧"，
    # 且残差侧必须单独过组合层（2026-11-18 cyq_benefit 实证：残差方向打平市场）。
    if np.sign(raw) != np.sign(res):
        return "SIGNFLIP", (f"raw {raw:+.4f} 与残差 {res:+.4f} 反号 ⇒ "
                            f"raw 方向不可交易，只能测残差方向 {np.sign(res):+.0f} 侧")
    ratio, gap = row.get("eff_ratio", np.nan), row.get("eff_gap", np.nan)
    if np.sign(plc) != np.sign(res):
        return "PASS*", f"安慰剂反号（plc {plc:+.4f}），效应量比不可算，看 gap {gap:+.4f}"
    if abs(gap) < t["gap_floor"]:
        return "NOISE", f"真实与安慰剂差 {gap:+.4f} < {t['gap_floor']}"
    if np.isfinite(ratio) and ratio < t["eff_ratio"]:
        return "WEAK", f"效应量比 {ratio:.1f}× < {t['eff_ratio']}×"
    return "PASS", f"效应量比 {ratio:.1f}×，gap {gap:+.4f}（仍需 Top-N 组合验证）"


#: 兼容别名（旧调用方写 ``gc.verdict`` / ``schema.verdict_admission``）。
verdict = admission_verdict
verdict_admission = admission_verdict

__all__ = [
    "ADMISSION_GATES", "CORR_TRIGGER", "MIN_SPAN", "REL_STD_DEGENERATE",
    "admission_verdict",
    "REL_STD_TINY", "RESID_IC_NEGLIGIBLE", "Verdict", "all_thresholds",
    "classify_independent", "cyclic_placebo", "dup_classify", "dup_explain",
    "effect_size", "ic_calibers", "ic_stats", "panel_coverage",
    "promotion_gates", "qualifier", "rank_corr", "resid_rank", "seg_stability",
    "should_check", "tier_of", "verdict", "verdict_admission", "yearly_ic",
]
