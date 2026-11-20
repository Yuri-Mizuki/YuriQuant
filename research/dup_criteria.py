"""因子重复性判据——**全仓唯一真源**。

> 2026-11-19 从 ``scripts/factors/dup_criteria.py`` **下沉到 research 层**。
> 原因：判据库 :mod:`research.factor_criteria` 需要引用它，而 research 反向
> import scripts 是分层倒挂（此前 ``factor_library`` 已在函数级这么做了两处）。
> 原路径保留兼容转出口，两者**必须是同一对象**（``tests/test_factor_criteria.py``
> 有断言）。**改口径只改本文件**，不要在调用点另抄阈值。

为什么要有这个模块
------------------
2026-11-12 之前，同一件事在三个地方各写了一遍，口径互不相同：

| 位置 | 用的判据 | 后果 |
|---|---|---|
| `library_dedup_plan`（决策） | `ρ≥0.95` + raw 值残差 `t` | 「高相关」被当成「等价」，误退 91 个 |
| `library_dedup_apply`（执行） | 同上（`COVER_RHO_MIN=0.95`） | 校验通过 = 按错口径批 |
| `factor_library.precheck_dup`（入库） | 同型 | 新因子按错口径判重 |

现在全部改为调用本模块。**改口径只改这里**，禁止在各调用点另抄一份阈值。

判据本体（一句话）
------------------
**判「两个因子是否重复」只能看秩空间残差是否为 0。`ρ ≥ 0.95` 什么也不算。**

原因是生产链消费的是截面秩/z（`panels_neu` → z-score → DPP），
两个因子在秩空间完全等价 ⇔ 对模型是同一个输入。而 `ρ` 只是「像不像」：
本库 `|ρ|≥0.95` 有 2554 对、涉及 1142/950 个因子，拿它当等价等于不筛。

🔑 三个必须记住的实测事实（2026-11-12）
---------------------------------------
1. **纯单调变换在秩空间恒等价**：`alpha158_OPEN0` 是 `alpha101_033` 的截面 rank，
   2511 天秩相关**恒 1.000000**；而原始值 OLS 给 `t=−7.40`（曲线残差，伪增量）。
   `VSUMN30` vs `VSUMD30` 秩相关恒 **−1.000**，raw OLS 给 `t=+25.4`
   —— 当年据此写下的「量比信号在残差里」**是错的**。
2. **`ρ ≥ 0.95` 不等于冗余**：129 个退登记里只有 34 个 `cover_rho=1.000` 真退化，
   其余 95 个 ρ 仅 0.96~0.999、秩空间仍留 7%~27% 残差标准差。
3. **反向也成立**：秩空间**有**残差 ≠ 该保留，还得过组合层 A/B（`caliber` 口径）。

所以本模块只给「等价 / 有增量」的分级，**不直接决定删不删**；
删不删要看是否进过定版、以及组合层 A/B。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 阈值（唯一真源）
#: 秩空间残差标准差 / 因变量秩标准差。< 此值视为**精确退化**：
#: 该因子是覆盖者的单调变换（含镜像），对模型没有任何新信息。
REL_STD_DEGENERATE = 1e-3

#: 残差仍存在但极小时的上界。配合 ``RESID_IC_NEGLIGIBLE`` 一起用——
#: 两个条件**同时**满足才判「微小增量」（可删），避免单条件误判。
REL_STD_TINY = 0.05

#: 残差与未来收益的 Rank IC 均值，小于此值视为「残差里没有可用信号」。
#: 0.02 是本库的 IC 量级门槛（与 `ic_h20` 的分档一致）。
RESID_IC_NEGLIGIBLE = 0.02

#: 秩相关绝对值达到此值才值得做残差检验（ρ 更低的因子之间不存在"等价"问题）。
#: ⚠️ 这是**触发检验**的门槛，不是判定重复的门槛——别混用。
#:
#: 🚨 2026-11-13 下午：从 0.85 **改回 0.95**（等价于只验 T1/T2，T3 直接 keep）。
#: 依据是产出比而非理论洁癖：把触发线压到 0.85 后全量补测 156 个 T3，
#: **154 个判「有独立增量」、仅 1 个真等价**，而决策表耗时 6 → 9 分钟。
#: 花 +55% 计算量换 1 个删除决定，在这个因子上不成立。
#: 唯一代价：`alpha191_098`（ρ=0.8975 / rel_std=7.9e-16，与 `alpha101_024`
#: 严格等价）会作为 T3 被放行——已在 `reports/library_governance/dedup_execution.md`
#: §十三 记为已知例外，不靠这条阈值兜。
CORR_TRIGGER = 0.95

#: 标注用的分档（只用于报告可读性，**不参与删除判定**）。
TIER_T1 = 0.999      # 硬重复（含镜像）
TIER_T2 = 0.95       # 近重复
TIER_T3 = 0.85       # 同族

#: 有效截面数下限。低于此值一律「无法检验」，**不得**当成「无增量」。
MIN_SPAN = 20

# 判定结果（字符串常量，避免各处手写拼错）
V_DEGENERATE = "等价(秩空间退化)"
V_TINY = "微小增量"
V_INCREMENT = "有独立增量"
V_UNTESTED = "无法检验"
#: 与库内所有因子的 |ρ| 都低于触发线 ⇒ **不相似，不需要做残差检验**。
#: 与 ``V_UNTESTED`` 必须区分：后者是"该测没测成"，前者是"不需要测"。
V_INDEPENDENT = "独立(无需检验)"

#: |ρ| 落在 [TIER_T3, CORR_TRIGGER)（即 T3 同族档）⇒ **相似但不检验，按产出比放行**。
#: 🚨 必须与 ``V_INDEPENDENT`` 分开：ρ=0.90 的两个因子**并不"不相似"**，
#: 只是实测「补测 156 个、154 个有独立增量、仅 1 个真等价」⇒ 检验的边际产出
#: 不抵 +55% 计算量（2026-11-13）。把它写成"独立"会在日志里输出
#: 「ρ=0.897 与库内不相似」这种自相矛盾的话，也会让人误以为测过了。
#: 语义：「没测，因为整档不值得测」——**不是**「测了没问题」。
V_BELOW_TRIGGER = "同族(未达检验线)"


@dataclass(frozen=True)
class Verdict:
    """一次重复性判定的完整结论。

    Attributes:
        level: 四档之一（见 ``V_*`` 常量）。
        rel_std: 残差秩标准差 / 因变量秩标准差。
        resid_ic: 残差对覆盖者的未来收益 Rank IC 均值（退化时为 NaN）。
        resid_t: 上述 IC 序列的 NW t（退化或样本不足时为 NaN）。
        n_span: 参与回归的截面数（**不等于** IC 有效日）。
        n_days: IC 有效日（精确退化时为 0，因为残差恒 0 无从算 IC）。
        reason: 人读的判定理由。
    """

    level: str
    rel_std: float = float("nan")
    resid_ic: float = float("nan")
    resid_t: float = float("nan")
    n_span: int = 0
    n_days: int = 0
    reason: str = ""

    @property
    def is_equivalent(self) -> bool:
        """精确等价：秩空间没有残差 ⇒ 对模型是同一个输入 ⇒ 可以判重。"""
        return self.level == V_DEGENERATE

    @property
    def removable(self) -> bool:
        """**仅凭重复性**可删：等价，或残差小到没有可用信号。

        ⚠️ 可删 ≠ 该删。调用方还须检查「从未进定版」等约束。
        """
        return self.level in (V_DEGENERATE, V_TINY)

    @property
    def tested(self) -> bool:
        """是否真的测了。False 时**不得**当成「无增量」。

        ``V_BELOW_TRIGGER``（T3 档）也是 False —— 它是「没测」，只是原因是
        「整档不值得测」。写成 True 会让调用方以为有残差证据。
        """
        return self.level not in (V_UNTESTED, V_INDEPENDENT, V_BELOW_TRIGGER)

    def as_dict(self) -> dict:
        return {
            "verdict": self.level, "rel_std": self.rel_std,
            "resid_ic": self.resid_ic, "resid_t": self.resid_t,
            "n_span": self.n_span, "n_days": self.n_days,
            "reason": self.reason,
        }


def classify(stats: dict) -> Verdict:
    """把 :func:`rank_resid_stats` 的原始输出（2026-11-20 起同在本模块）转成 :class:`Verdict`。

    🔑 **判定顺序有讲究**：必须先判退化再判样本量。
    精确退化时残差恒 0 ⇒ IC 无从计算 ⇒ ``n_days=0``，但回归可能跑满 2511 个截面。
    若先查 ``n_days < 20 ⇒ 无法检验``，等价因子会被判成「必须保留」，**方向完全反**
    （2026-11-12 实测 34 个被这样误保）。

    Args:
        stats: ``rank_resid_stats`` 的返回，需含 ``rel_std`` / ``ic`` / ``t`` /
            ``n_days`` / ``n_span`` / ``degenerate``。

    Returns:
        Verdict: 四档结论之一。
    """
    rel_std = float(stats.get("rel_std", np.nan))
    ic = float(stats.get("ic", np.nan))
    t = float(stats.get("t", np.nan))
    n_span = int(stats.get("n_span", 0) or 0)
    n_days = int(stats.get("n_days", 0) or 0)

    if n_span < MIN_SPAN or not np.isfinite(rel_std):
        return Verdict(V_UNTESTED, rel_std, ic, t, n_span, n_days,
                       f"有效截面 {n_span} < {MIN_SPAN}")

    # ① 退化：只需 rel_std，不需要 IC
    if bool(stats.get("degenerate")) or rel_std < REL_STD_DEGENERATE:
        return Verdict(V_DEGENERATE, rel_std, ic, t, n_span, n_days,
                       f"秩空间退化 rel_std={rel_std:.2e}（单调变换/镜像）")

    # ② 残差存在但没有 IC 可比 ⇒ 不判微小增量，宁可留
    if n_days < MIN_SPAN or not np.isfinite(ic):
        return Verdict(V_UNTESTED, rel_std, ic, t, n_span, n_days,
                       f"残差存在但 IC 有效日仅 {n_days}")

    # ③ 微小增量：两个条件同时满足
    if rel_std < REL_STD_TINY and abs(ic) < RESID_IC_NEGLIGIBLE:
        return Verdict(V_TINY, rel_std, ic, t, n_span, n_days,
                       f"残差小 rel_std={rel_std:.3f} 且 |ic|={abs(ic):.3f}")

    return Verdict(V_INCREMENT, rel_std, ic, t, n_span, n_days,
                   f"残差有信号 rel_std={rel_std:.3f} ic={ic:+.4f} t={t:+.2f}")


def tier_of(max_abs_corr: float) -> str:
    """按 |ρ| 分档，**仅用于报告标注**，不参与删除判定。"""
    if not np.isfinite(max_abs_corr):
        return "未知"
    if max_abs_corr >= TIER_T1:
        return "T1硬重复"
    if max_abs_corr >= TIER_T2:
        return "T2近重复"
    if max_abs_corr >= TIER_T3:
        return "T3同族"
    return "独立"


def should_check(max_abs_corr: float) -> bool:
    """|ρ| 达到 ``CORR_TRIGGER`` 才值得做残差检验（省掉无谓的 8~13 s/因子）。"""
    return bool(np.isfinite(max_abs_corr) and max_abs_corr >= CORR_TRIGGER)


def classify_independent(max_abs_corr: float) -> Verdict:
    """未达触发线时的结论：**不需要残差检验**（但分两种，别混）。

    - ``|ρ| < TIER_T3``（0.85）⇒ ``V_INDEPENDENT``：真的**不相似**。
    - ``TIER_T3 ≤ |ρ| < CORR_TRIGGER``（0.85~0.95，T3 档）⇒ ``V_BELOW_TRIGGER``：
      **相似，但按产出比不检验**。实测该档 156 个里 154 个有独立增量、仅 1 个真等价
      （2026-11-13），补测不值。⚠️ 这是「没测」不是「测了没问题」。

    两者都与 ``V_UNTESTED`` 严格区分：后者是"该测没测成"（面板缺失/截面不足）。
    """
    if not np.isfinite(max_abs_corr):
        return Verdict(V_UNTESTED, reason="最近邻 |ρ| 非有限值")
    if max_abs_corr >= TIER_T3:
        return Verdict(V_BELOW_TRIGGER,
                       reason=f"|ρ|={max_abs_corr:.3f} 在 T3 档"
                              f"（{TIER_T3:g}~{CORR_TRIGGER:g}），整档不检验")
    return Verdict(V_INDEPENDENT, reason=f"最近邻 |ρ|={max_abs_corr:.3f} < "
                                        f"{TIER_T3:g}，与库内不相似")


def explain() -> str:
    """一行版判据，供日志/报错信息引用。"""
    return (f"判重口径：秩空间残差 rel_std<{REL_STD_DEGENERATE:g} ⇒ 等价；"
            f"rel_std<{REL_STD_TINY:g} 且 |ic|<{RESID_IC_NEGLIGIBLE:g} ⇒ 微小增量；"
            f"否则有独立增量。ρ 只用于触发检验（≥{CORR_TRIGGER:g}），不参与判定。")


# ---------------------------------------------------------------- 秩空间增量统计
#: 2026-11-20 自 ``scripts/factors/dup_audit`` 下沉（research.factor_library 需要，
#: 属判据的计算核心；原处不得留副本）。
# ---------------------------------------------------------------- 秩空间增量
def rank_resid_stats(panel: pd.DataFrame, against: dict[str, pd.DataFrame],
                     returns: pd.DataFrame, *, min_obs: int = 20,
                     stride: int = 1) -> dict:
    """**秩空间**多因子增量检验：把 y 的截面秩对 covering 因子的截面秩回归取残差。

    为什么不用原始值 OLS（2026-11-12 复核改口径）
    --------------------------------------------
    ``_resid_for`` 在**原始值**上做逐日截面 OLS。纯单调变换（例如
    ``alpha101_033`` 就是 ``alpha158_OPEN0`` 的截面 rank）在原始值空间不是线性
    关系，OLS 残差是一个**关于 x 的系统性曲线**（x² 项等），它当然与未来收益
    相关 ⇒ 稳定的大 t 值。实测：

    - ``OPEN0`` vs ``101_033``：2511 天秩相关**恒 = 1.000000**，raw OLS 给 **t = −7.40**
    - ``VSUMN30`` vs ``VSUMD30``：2511 天秩相关**恒 = −1.000**，raw OLS 给 **t = +25.4**

    后者曾被当作「量比信号就在残差里」的证据写进报告——**那个结论是错的**：
    两因子在秩空间完全等价，而生产链（``panels_neu`` → z-score → DPP）消费的
    正是截面秩/z，原始值的非线性差异进不了模型。

    本函数在秩空间做同一件事：先对每日截面取 rank，再对 covering 因子的 rank
    回归。单调变换下残差**精确为 0**，不会产生任何伪增量。

    Args:
        panel: 待检验因子（date × code）。
        against: ``{name: 面板}``，要控制掉的因子（通常是簇代表 + 已判留者）。
        returns: 未来收益面板（与 ``default_returns_panel`` 同口径）。

    Returns:
        dict:
        - ``rel_std``：残差标准差 / 因变量秩标准差。**< 1e-3 即「秩空间退化」**
          （等价于 covering 的单调变换），可直接删。**只取决于因子间关系，
          与收益面板无关**——收益不覆盖的因子照样能判等价。
        - ``ic`` / ``t``：残差与未来收益的 Rank IC 均值与 NW t。
        - ``n_days``：IC 有效截面数。< 20 时``ic``/``t`` 记 NaN，**必须与「无增量」
          区分开**（见 ``_verify_retire_resid_on_cover`` 的 8/8 假通过事故）。
        - ``n_span``：参与回归的截面数。**与 ``n_days`` 分开报**：精确退化时
          残差恒 0 ⇒ ``n_days=0`` 但 ``n_span`` 可能是 2511。
        - ``degenerate``：``rel_std < 1e-3``。

    ⚠️ 掩码分两个（2026-11-12 修）：``rel_std`` 用「因子 × covering 均有效」，
    ``ic`` 再叠「收益有效」。曾经共用一个含收益的掩码，导致收益面板不覆盖的
    因子被判「无法检验（n_span=0）」，而它其实**已经可以判等价**。
    """
    import scipy.stats as ss_mod

    from stats.robust_stats import nw_tstat

    cd = panel.index
    cc = panel.columns
    for p in against.values():
        cd = cd.intersection(p.index)
        cc = cc.intersection(p.columns)
    if len(cd) < min_obs or len(cc) < min_obs:
        return {"rel_std": float("nan"), "ic": float("nan"),
                "t": float("nan"), "n_days": 0, "n_span": 0,
                "degenerate": None}

    cd = cd[::max(1, int(stride))]
    Y = panel.reindex(index=cd, columns=cc).to_numpy(dtype=np.float64)
    # 收益**不进交集**：不覆盖的日子在下面按 NaN 掩掉，只影响 ic，不影响 rel_std。
    R = returns.reindex(index=cd, columns=cc).to_numpy(dtype=np.float64)
    Xs = [p.reindex(index=cd, columns=cc).to_numpy(dtype=np.float64)
          for p in against.values()]

    rel: list[float] = []
    ics: list[float] = []
    for ti in range(len(cd)):
        y = Y[ti]
        r = R[ti]
        cols = [x[ti] for x in Xs]
        # ⚠️ 两个掩码**必须分开**（2026-11-12 修）：
        # ``rel_std`` 是纯**因子间**的秩空间关系，与收益无关；只有 ``ic`` 需要收益。
        # 原先用同一个 ``ok``（含 isfinite(r)）⇒ 收益面板不覆盖的因子被判「无法检验」，
        # 而它其实**已经能判等价**。真实场景：新因子窗口比收益面板长/短、
        # 或单测里没给收益面板。这与我此前犯的「精确退化写成样本不足」是同一类错。
        ok = np.isfinite(y)
        for c in cols:
            ok &= np.isfinite(c)
        if ok.sum() < min_obs:
            continue
        ry = ss_mod.rankdata(y[ok])
        sy = float(np.std(ry))
        if sy <= 0:
            continue
        # 秩空间设计矩阵：[1, rank(covering)...]
        M = np.column_stack([np.ones(int(ok.sum()))]
                            + [ss_mod.rankdata(c[ok]) for c in cols])
        beta, *_ = np.linalg.lstsq(M, ry, rcond=None)
        resid = ry - M @ beta
        rs = float(np.std(resid))
        rel.append(rs / sy)
        if rs / sy < 1e-6:            # 秩空间精确退化 ⇒ 无残差可言
            continue
        # 到这里才需要收益：在 ok 的索引空间里再筛一次
        ok_ic = np.isfinite(r[ok])
        if int(ok_ic.sum()) < min_obs:
            continue
        rr = r[ok][ok_ic]
        if np.std(rr) == 0:
            continue
        ic = float(np.corrcoef(ss_mod.rankdata(resid[ok_ic]),
                               ss_mod.rankdata(rr))[0, 1])
        if not np.isnan(ic):
            ics.append(ic)

    if not rel:
        return {"rel_std": float("nan"), "ic": float("nan"),
                "t": float("nan"), "n_days": 0, "n_span": 0, "degenerate": None}
    rel_std = float(np.mean(rel))
    # ⚠️ ``n_span``（参与回归的截面数）与 ``n_days``（IC 有效日）**必须分开报**：
    # 秩空间精确退化时残差恒 0，IC 无从计算 ⇒ ``n_days=0`` 但 ``n_span`` 可能是 2511。
    # 混用会让"精确等价"被当成"样本不足"（2026-11-12 实测 34 个）。
    n_span = len(rel)
    if len(ics) < 20:
        return {"rel_std": rel_std, "ic": float("nan"), "t": float("nan"),
                "n_days": len(ics), "n_span": n_span,
                "degenerate": rel_std < 1e-3}
    t, _se, _lag = nw_tstat(np.asarray(ics))
    return {"rel_std": rel_std, "ic": float(np.mean(ics)), "t": float(t),
            "n_days": len(ics), "n_span": n_span,
            "degenerate": rel_std < 1e-3}


# ---------------------------------------------------------------- 回填
def rank_resid_stats_from_sig(z_new: np.ndarray, z_old: np.ndarray, *,
                              min_obs: int = 20) -> dict:
    """索引签名兜底：只用两个因子的 rank-z 签名做秩空间回归（无收益 ⇒ 无 ic）。

    为什么需要它
    ------------
    ``precheck_dup`` 判等价要拿最近邻的**面板**做回归，但库内 912 个批量入库
    （``merge_outputs``）的行 ``panel_path`` 为空、``get_panel`` 返回 None。
    而索引（``dup_signatures.npy``）覆盖**全库**，于是「新因子 vs 库内 builder
    因子」这条最主要的路径全部落到「无法检验 ⇒ 放行」——等价检查静默失效。
    签名本来就在内存里（memmap），拿它兜底不需要多读一个字节。

    ⚠️ 口径差异（引用时必须带上）：签名里 NaN 已记 0（= 当日中位），
    无法还原「缺失」。后果分两种：

    - 缺失模式**相同** ⇒ 0 的位置也相同 ⇒ 残差仍**精确为 0**，判等价不受影响；
    - 缺失模式**不同** ⇒ ``rel_std`` 偏大 ⇒ 只会**漏判**等价，不会误判。

    方向是保守的（宁留冗余，不误杀真信号），因此本函数**只回答"是否等价"**，
    不产出 ``ic``/``t``——没有收益就绝不假装有。

    Returns:
        同 :func:`rank_resid_stats`，但 ``ic``/``t`` 恒 NaN、``n_days`` 恒 0。
    """
    import scipy.stats as ss_mod

    rel: list[float] = []
    n_span = 0
    for ti in range(z_new.shape[0]):
        a, b = z_new[ti], z_old[ti]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_obs:
            continue
        ra = ss_mod.rankdata(a[ok])
        sy = float(np.std(ra))
        if sy <= 0:
            continue
        M = np.column_stack([np.ones(int(ok.sum())), ss_mod.rankdata(b[ok])])
        beta, *_ = np.linalg.lstsq(M, ra, rcond=None)
        rel.append(float(np.std(ra - M @ beta)) / sy)
        n_span += 1
    if not rel:
        return {"rel_std": float("nan"), "ic": float("nan"),
                "t": float("nan"), "n_days": 0, "n_span": 0, "degenerate": None}
    rel_std = float(np.mean(rel))
    return {"rel_std": rel_std, "ic": float("nan"), "t": float("nan"),
            "n_days": 0, "n_span": n_span, "degenerate": rel_std < 1e-3}
