"""因子检验公共件 —— L1~L5（2026-11-18 建立，**本层是单一真源**）。

## 为什么有这个模块

项目此前有 102 个探针脚本，其中 14 个各自实现了一遍「截面秩 IC + Newey-West t
+ 残差 IC」，口径靠人肉对齐。后果是：同一个问题在不同探针里给出不可比的结果，
而判据（什么算"有效"）只写在 `.workbuddy/memory/MEMORY-stats.md` §6 的文字里，
执行时全凭当班的人记不记得。

本模块把「判定一条因子需要哪几步、每步怎么算」变成代码，供
:mod:`scripts.evaluation.factor_gate`（唯一入口）调用。

## 五道关（每道都可能否掉因子）

| 关 | 问的问题 | 函数 |
|---|---|---|
| L1 | 面板覆盖够不够、有没有死面板 | :func:`panel_coverage` |
| L2 | 有没有原始预测力（期限 = 持有期） | :func:`ic_calibers` → ``raw_*`` |
| L3 | 是不是被主信号/风格冒充了 | :func:`ic_calibers` → ``resfac_*`` |
| L4 | 和生产因子池是不是同一个东西 | :func:`rank_corr` |
| L5 | **纯噪声能不能拿到同样的数** | :func:`cyclic_placebo` |

## 🚨 关于 t 值（2026-11-18 实测，最贵的一条）

2 年半日频、5000 只截面的残差 IC 序列自相关很强，`t = IC × √T` 会把 0.003 的
噪声放大成 t≈4。cyq 探针里，**逐列循环平移的安慰剂拿到 ``resfac_t = 4.24``，
比真实因子的 3.85 还高**。所以：

- ❌ 不看绝对 t 值；
- ✅ 看 :func:`effect_size`（真实 IC ÷ 安慰剂 IC），**两个 p 都要报**；
- ✅ 过了 L5 也只是"可以进组合层"，不等于赚钱（IC ≠ 经济价值）。

## 口径

复用 :mod:`scripts.evaluation.caliber` 的窗口与掩码约定：
``prod`` 窗口（2019-01 起）、日频 h 期前瞻收益、生产掩码。
🚨 报数必须带 horizon —— IC 的期限必须等于持有期，跨期限 IC 不可并列。

本模块**不是**入口，可被 scripts 下的入口脚本 import。
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

#: 判定用的默认门槛。**这是筛子不是判据** —— 过了线只代表"值得跑组合层"。
DEFAULTS: dict[str, float] = {
    "min_cov": 0.30,      # 面板非空覆盖率下限
    "min_n": 1000,        # 有效格子数下限
    "ic_floor": 0.005,    # |IC| 下限（0.005 ≈ 年化 1pp 量级的量级门）
    "gap_floor": 0.003,   # |真实 IC − 安慰剂 IC| 下限
    "eff_ratio": 2.0,     # 效应量比下限
}


# ---------------------------------------------------------------------------
# L0 数据装载
# ---------------------------------------------------------------------------
def load_ic_context(horizon: int = 1, window: str = "prod",
                    pred_dir: str = "reports/alla_rolling_ortho/pred",
                    main_signal: str = "ens_h1h5__h1.parquet",
                    end: int | None = None,
                    begin: int | str | None = None,
                    stride: int = 1) -> dict:
    """装载 IC 检验所需的全部面板（与 ``residual_ic_probe`` 同一口径）。

    返回 ``{idx, cols, close, fwd, mask, main, main_rank}``。

    - ``fwd``：h 期前瞻收益 ``close.pct_change(h).shift(-h)``，即 t 收盘 → t+h 收盘。
    - ``mask``：生产可交易掩码（涨跌停 / 停牌 / ST），NaN 按不可交易。
    - ``idx``：**主信号**的日期轴（OOS 区间），不是数据轴 —— IC 只在有预测的日子算。
      ``begin`` / ``end``（``YYYYMMDD``）从这里再截一刀，起点缺省取自
      ``caliber.WINDOWS``，保证与组合层同窗口。
    """
    from scripts.common.alla_common import load_base

    base = load_base()
    close = base["close"]
    mask_all = base["mask"].astype(bool)

    sig_path = Path(main_signal)
    main = pd.read_parquet(sig_path if sig_path.is_absolute()
                           else Path(pred_dir) / main_signal)
    main = main.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    idx = main.index
    if begin is not None:
        lo = pd.to_datetime(str(int(begin)), format="%Y%m%d")
        idx = idx[idx >= lo]
    if end is not None:
        cut = pd.to_datetime(str(int(end)), format="%Y%m%d")
        idx = idx[idx <= cut]
    if stride > 1:
        idx = idx[:: stride]
    cols = main.columns.intersection(close.columns)

    fwd = (close.pct_change(horizon, fill_method=None)
           .shift(-horizon)
           .reindex(index=idx, columns=cols)
           .replace([np.inf, -np.inf], np.nan)
           .astype(np.float32))
    main_c = main.reindex(index=idx, columns=cols)
    return {
        "idx": idx,
        "cols": cols,
        "fwd": fwd,
        "mask": mask_all.reindex(index=idx, columns=cols).fillna(False).astype(bool),
        "main": main_c,
        "main_rank": main_c.rank(axis=1, pct=True),
        "main_signal": main_signal,
        "horizon": int(horizon),
        "window": window,
    }


def load_panel(path, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame | None:
    """读一张因子面板并对齐；非数值 / 空面板返回 None。"""
    raw = pd.read_parquet(path)
    if not isinstance(raw.index, pd.DatetimeIndex):
        try:
            raw.index = pd.to_datetime(raw.index)
        except Exception:
            return None
    if raw.columns.has_duplicates:
        raw = raw.loc[:, ~raw.columns.duplicated()]
    raw = raw.select_dtypes(include=[np.number])
    if raw.shape[1] == 0:
        return None
    x = raw.reindex(index=index, columns=columns)
    return x.replace([np.inf, -np.inf], np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# L1 覆盖
# ---------------------------------------------------------------------------
def panel_coverage(x: pd.DataFrame, fwd: pd.DataFrame,
                   mask: pd.DataFrame) -> dict:
    """覆盖率与有效格子数。"""
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


def ic_stats(ic: pd.Series) -> dict:
    """IC 均值 / IR / 胜率 / Newey-West t / 有效天数。"""
    from stats.robust_stats import nw_tstat

    s = ic.dropna()
    if len(s) < 2:
        return dict(ic=float("nan"), ir=float("nan"), win=float("nan"),
                    nw_t=float("nan"), n=0)
    sd = float(s.std())
    t_nw, _se, _lag = nw_tstat(s.to_numpy(dtype=float))
    return dict(ic=float(s.mean()),
                ir=float(s.mean() / sd) if sd else float("nan"),
                win=float((s > 0).mean()), nw_t=float(t_nw), n=int(len(s)))


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


# ---------------------------------------------------------------------------
# L4 换皮
# ---------------------------------------------------------------------------
def rank_corr(a: pd.DataFrame, b: pd.DataFrame, mask: pd.DataFrame) -> float:
    """逐日截面秩相关的均值（``a``/``b`` 都需事先对齐）。"""
    v = a.notna() & b.notna() & mask
    if not v.to_numpy(dtype=bool).any():
        return float("nan")
    return float(a.where(v).rank(axis=1)
                 .corrwith(b.where(v).rank(axis=1), axis=1).mean())


# ---------------------------------------------------------------------------
# L5 安慰剂
# ---------------------------------------------------------------------------
def cyclic_placebo(x: pd.DataFrame, seed: int, name: str = "",
                   lo: int = 120, hi: int = 900) -> pd.DataFrame:
    """逐列**循环平移**：保留每条序列的分布与自相关，打掉与收益的时点对应。

    每列独立随机平移 ``U[lo, hi)`` 个交易日。平移量由 ``(seed, name)`` 唯一确定
    ⇒ 同一因子每次重跑得到**同一份**安慰剂（可复现），且不随同批其他因子变化。

    🚨 ``hi`` 必须显著小于样本长度：平移量接近 T 时序列几乎回到原位，安慰剂会
    泄漏真实信息（``hi=900`` 相对 2100 日是安全的）。
    🚨 **平移量必须显著小于序列长度**：``roll`` 是循环的，``k`` 若是 ``T`` 的整数倍
    就绕回原位、安慰剂等于原因子（实现在 ``hi`` 超过 ``T/2`` 时自动收紧，``T`` 太短
    直接报错）。1864 日的面板上 ``[120, 900)`` 是安全的。
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
# 判定（筛子，不是结论）
# ---------------------------------------------------------------------------
def verdict(row: dict, th: dict | None = None) -> tuple[str, str]:
    """按 L1~L5 给出**筛选标记**与理由。返回 ``(tag, why)``。

    🚨 这里**不下最终结论**。``PASS`` 只代表"值得进组合层"，
    因为 IC ≠ 经济价值 —— 必须再跑 Top-N（见 ``cyq_panel.py topn``）。
    """
    t = {**DEFAULTS, **(th or {})}
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
