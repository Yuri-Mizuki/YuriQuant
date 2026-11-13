"""因子重复性判据——**全仓唯一真源**。

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
    """把 ``dup_audit.rank_resid_stats()`` 的原始输出转成 :class:`Verdict`。

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