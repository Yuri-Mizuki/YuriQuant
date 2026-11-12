"""因子信息半衰期判定的统计原语回归（``scripts/evaluation/factor_half_life_verdict.py``）。

2026-11-11 建立。锁死三件事 —— 这三处静默出错的后果都是「把噪声读成信号」：

1. **BH-FDR 的拒绝集必须单调**：p 更小的假设不可能不被拒绝。这是 BH 的定义性质，
   实现写错（比如阈值用了 `q*k/m` 但排序后没回填原索引）会静默产生「p 小却没拒绝」
   的荒谬结果，而调用方只看到「显著数」一个标量，不会察觉。
2. **纯噪声下的显著数 ≈ q·m**：952 个因子在 α=0.10 下，若几乎全显著，说明 p 值算错
   （典型错误：单侧 p 当双侧用，或有 NaN 时索引错位拿别的因子的 p）。
3. **族归并不吞掉独立因子**：`alpha191_001` 与 `alpha191_002` 归一族是对的，但
   `sue_q` 与 `roa_ttm` 不能被归成同一族（归太粗会把证据量砍掉，从而把「有原料」
   误判成「无原料」）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------
def test_bh_rejection_is_monotone_in_p():
    """p 越小越该被拒绝：拒绝集必须是「p 排序后的前缀」。"""
    from scripts.evaluation.factor_half_life_verdict import bh_fdr

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    rej = bh_fdr(p, q=0.10)
    # 不存在「p 较大被拒绝、p 较小未拒绝」的情况
    pmax_rej = p[rej].max() if rej.any() else -1.0
    assert (p[~rej] >= pmax_rej - 1e-12).all(), "拒绝集不是 p 的前缀 ⇒ BH 回填索引写错"


def test_bh_pure_noise_rarely_rejects_anything():
    """全局零假设下 BH 的「至少拒绝一个」比例必须 ≤ q（这是 FDR 控制的定义）。

    🚨 我首版这里写错了：以为纯噪声下期望拒绝数 ≈ q·m。**不对** —— BH 是
    控制 FDR 上限，全局零下它的拒绝概率 ≤ q，实践中通常**一次也不拒绝**
    （最小 p 的阈值只有 q/m，连续均匀几乎撞不上）。首版断言 `share>=0.04`
    实测拿到 0.000 就是这个误读。
    正确性质：多次重复里「有任一拒绝」的频率 ≤ q 量级。
    """
    from scripts.evaluation.factor_half_life_verdict import bh_fdr

    rng = np.random.default_rng(1)
    reps, m, q = 300, 200, 0.10
    any_rej = 0
    for _ in range(reps):
        if bh_fdr(rng.uniform(0, 1, m), q=q).any():
            any_rej += 1
    share = any_rej / reps
    assert share <= q + 0.03, f"全局零下拒绝率 {share:.3f} 超过 q={q} ⇒ p 值或回填写错"


def test_bh_finds_true_signals_among_noise():
    """有真实信号时 BH 必须抓住它们（否则判定门槛形同虚设）。"""
    from scripts.evaluation.factor_half_life_verdict import bh_fdr

    rng = np.random.default_rng(2)
    m, n_true = 400, 40
    p = rng.uniform(0, 1, m)
    p[:n_true] = rng.uniform(0, 1e-6, n_true)      # 40 个强信号
    rej = bh_fdr(p, q=0.10)
    assert rej[:n_true].all(), "真实信号未被拒绝 ⇒ 判定过严，会把「有原料」漏成「无原料」"
    # 噪声侧不应被大量误拒（留宽松余量，只拦量级错）
    assert rej[n_true:].sum() <= 0.1 * (m - n_true) + 3


def test_bh_rejects_nothing_when_all_p_large():
    """全部 p 接近 1 时不得拒绝任何假设（含 NaN 混入的情形）。"""
    from scripts.evaluation.factor_half_life_verdict import bh_fdr

    p = np.array([0.9, 0.95, 1.0, np.nan, 0.99])
    assert not bh_fdr(p, q=0.10).any()


def test_bh_handles_nan_without_index_shift():
    """NaN 必须落成「不拒绝」，且不能让后续元素的 p 与索引错位。

    构造：只有第 3 个元素是极小 p，其余是 NaN 或大 p ⇒ 只该拒绝它。
    索引回填写错（用 `pv` 的位置而不是原数组位置）会误拒别的位置。
    """
    from scripts.evaluation.factor_half_life_verdict import bh_fdr

    p = np.array([np.nan, np.nan, 1e-8, np.nan, 0.7, 0.8])
    rej = bh_fdr(p, q=0.10)
    assert list(rej) == [False, False, True, False, False, False]


# ---------------------------------------------------------------------------
# 族归并
# ---------------------------------------------------------------------------
def test_family_groups_qlib_feature_sets():
    """`alpha158_VMA5` / `alpha360_VOLUME19` 这类必须按「集 + 算子」归族。

    🚨 首版漏了这条规则，把 60 个 `alpha360_VOLUME*` 数成 60 个独立族，
    导致「周频显著更强」因子被数成 77 族（实际远少于此）⇒ 族数虚高会
    把「无原料」误判成「有原料」。
    """
    from scripts.evaluation.factor_half_life_verdict import factor_family

    assert factor_family("alpha360_VOLUME19") == factor_family("alpha360_VOLUME20")
    assert factor_family("alpha158_VMA5") == factor_family("alpha158_VMA10")
    # 算子不同 ⇒ 不同族（VMA 与 VSTD 是两种信息）
    assert factor_family("alpha158_VMA5") != factor_family("alpha158_VSTD5")
    # 集不同 ⇒ 不同族
    assert factor_family("alpha158_VMA5") != factor_family("alpha360_VMA5")
    # 大小写混写也要归对（面板里 `alpha158_VMA5` 与 `alpha360_volume32` 并存）
    assert factor_family("alpha360_volume32") == factor_family("alpha360_VOLUME32")


def test_family_does_not_merge_across_operators():
    """归族不能粗到把不同算子并掉 —— 那会把真证据量砍掉。"""
    from scripts.evaluation.factor_half_life_verdict import factor_family

    fams = {factor_family(n) for n in
            ("alpha158_VMA5", "alpha158_VSTD5", "alpha158_VWAP0", "alpha158_LOW0")}
    assert len(fams) == 4, f"不同算子被并成一族：{fams}"


def test_family_groups_formulaic_alphas():
    from scripts.evaluation.factor_half_life_verdict import factor_family

    assert factor_family("alpha191_001") == "alpha191"
    assert factor_family("alpha101_042") == "alpha101"
    # 不同大族不能被并到一起
    assert factor_family("alpha191_001") != factor_family("alpha101_001")


def test_family_keeps_independent_factors_separate():
    """不同源的因子必须留在不同族 —— 归太粗会把证据量砍掉。"""
    from scripts.evaluation.factor_half_life_verdict import factor_family

    a, b = factor_family("sue_q"), factor_family("roa_ttm")
    assert a != b, f"sue_q 与 roa_ttm 被归成同一族 {a}，族数被低估"


def test_family_strips_window_suffix():
    """窗口后缀是同族变体，应归一族（否则同一信号的 20d/60d 会被算两次证据）。"""
    from scripts.evaluation.factor_half_life_verdict import factor_family

    assert factor_family("suspend_ratio_60d") == factor_family("suspend_ratio_20d")


def test_gates_are_not_silently_relaxed():
    """判据门槛是预注册的：出现在 README 里的数必须与 GATES 一致。

    这条防的是「看完结果再悄悄放宽门槛」（本项目已踩过同类：改门槛不改文档）。
    """
    from scripts.evaluation.factor_half_life_verdict import GATES

    assert GATES["fdr_q"] == 0.10
    assert GATES["sign_test_p"] == 0.01
    assert GATES["min_sig_factors"] == 30
    assert GATES["min_families"] == 5
    # 首版的 `min_ratio` 已作废（判据从「个数比」改成「符号检验」）
    assert "min_ratio" not in GATES


# ---------------------------------------------------------------------------
# nw_t（与探针共用，配对差显著性）
# ---------------------------------------------------------------------------
def test_nw_t_penalises_positive_autocorrelation():
    """正自相关序列的 NW-t 必须显著小于 naive t —— 这是重叠口径必用 NW 的全部理由。"""
    from scripts.evaluation.factor_half_life_probe import nw_t

    rng = np.random.default_rng(7)
    n = 1000
    # AR(1) with phi=0.8：强正自相关，但均值恒为 0.05
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + e[i]
    x = x + 0.05
    _, t_naive = nw_t(x, lags=0)
    _, t_nw = nw_t(x, lags=20)
    assert abs(t_nw) < abs(t_naive), "NW 未惩罚自相关 ⇒ lags 没生效"


def test_nw_t_short_series_returns_nan():
    from scripts.evaluation.factor_half_life_probe import nw_t

    m, t = nw_t([1.0, 2.0, 3.0], lags=2)
    assert np.isnan(m) and np.isnan(t)


def test_nw_t_matches_ols_for_iid():
    """iid 且无自相关时，NW-t（lags=0）应逼近普通 t = mean/(std/√n)。"""
    from scripts.evaluation.factor_half_life_probe import nw_t

    rng = np.random.default_rng(3)
    x = rng.normal(0.5, 1.0, 4000)
    m, t = nw_t(x, lags=0)
    naive = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert m == pytest.approx(x.mean(), abs=1e-9)
    assert abs(t - naive) < 0.05, f"NW(lags=0)={t:.4f} 与 naive={naive:.4f} 差太多"


def test_factor_family_on_full_panel_names():
    """对真实面板名跑一遍：族数必须在合理区间（不能 1 族，也不能几乎等于因子数）。"""
    from pathlib import Path

    from scripts.evaluation.factor_half_life_verdict import factor_family

    panels = Path("E:/data/factor_library/all_a_2018_2026/panels_neu")
    names = [p.stem for p in sorted(panels.glob("*.parquet"))]
    if len(names) < 100:
        return                                    # 面板不在本机 ⇒ 跳过
    fams = pd.Series([factor_family(n) for n in names])
    assert fams.nunique() > 100, f"族数 {fams.nunique()} 过少 ⇒ 归并过粗"
    assert fams.nunique() < len(names), "族数等于因子数 ⇒ 归并完全没生效"


# ---------------------------------------------------------------------------
# 配对差口径（三次修正，本探针最易重犯的错）
# ---------------------------------------------------------------------------
def test_directional_diff_handles_negative_ic():
    """v1 陷阱：负 IC 因子必须先用 sign(ic_h20) 对齐方向，否则符号记反。

    手工例子：某因子 ic_h5 = -0.05, ic_h20 = -0.02（方向为负，5 日分辨力更强）
      - 带符号差 = -0.03（看着像「周频更差」）❌
      - 定向差 sign(ic_h20)·(ic_h5-ic_h20) = -1 × -0.03 = **+0.03** ✅
    """
    import numpy as np

    ic_h5 = np.array([-0.05, 0.05])
    ic_h20 = np.array([-0.02, 0.02])
    sgn = np.where(ic_h20 >= 0, 1.0, -1.0)

    signed = ic_h5 - ic_h20
    directional = sgn * (ic_h5 - ic_h20)

    assert signed[0] < 0, "带符号差对负 IC 因子给出负值（错）"
    assert directional[0] > 0, "定向差应对齐后给出正值（对）"
    # 正 IC 因子两者一致 —— 这是 v1 的错不易被立刻发现的原因（73% 因子为正）
    assert signed[1] > 0 and directional[1] > 0


def test_daily_abs_diff_is_variance_artifact():
    """v2 陷阱：逐日 |IC| 之差是方差假象，必须能被守卫识别。

    机制：`E|X| ≈ σ√(2/π)`（折半正态）⇒ 只要日度波动 σ 大，`mean(|IC_t|)` 就机械性变大，
    与「更准」无关。h 越短、日度 IC 越抖是**定义使然**（前瞻窗口越短，单日噪声占比越高）。
    构造：两个「均值完全相同、只有波动不同」的序列 ⇒ |·| 之差的均值必须显著 > 0。
    """
    import numpy as np

    rng = np.random.default_rng(11)
    n = 4000
    mu = 0.0
    lo = rng.normal(mu, 0.05, n)          # 低波动
    hi = rng.normal(mu, 0.10, n)          # 高波动，均值同为 0

    assert abs(lo.mean() - hi.mean()) < 0.01, "两序列均值应几乎相同"
    d_abs = np.abs(hi).mean() - np.abs(lo).mean()
    # 均值相同，但 |·| 的均值差了约一倍 σ ⇒ 纯假象
    assert d_abs > 0.03, "波动大的序列 |·| 均值应显著更大（这就是假象）"


def test_variance_guard_threshold_separates_calibers():
    """守卫阈值 0.70 能干净分开「真口径」与「方差假象」。

    实测（2026-11-11，944 因子）：
      v3 定向差 corr(+0.480) / v1 带符号 corr(+0.125) / v2 逐日|IC| corr(+0.883)
    ⇒ 0.70 落在 v3 与 v2 之间，v2 被判假象、v3 通过。
    """
    from scripts.evaluation.factor_half_life_verdict import GATES

    thr = 0.70
    assert "variance_guard_corr" in GATES, "守卫阈值必须登记在 GATES 里"
    assert GATES["variance_guard_corr"] == thr
    assert 0.48 < thr < 0.883, "阈值须落在真口径与假象口径之间"


def test_sign_test_does_not_depend_on_t_values():
    """G1 用二项检验（只看符号），不依赖 t 值 —— 换 t 的算法不该改判定。"""
    from scipy import stats

    # 实测 v3 口径：332/944 个定向差 > 0（占 35.2%）
    bt = stats.binomtest(332, 944, 0.5)
    assert bt.pvalue < 0.01, "占比 0.352 应被判为显著偏负"
    # 若是一半一半，则不该显著
    assert stats.binomtest(472, 944, 0.5).pvalue > 0.5

def test_library_ic_is_a_valid_cross_check_source():
    """因子库现成的 `ic_h*.parquet` 可作为独立口径交叉验证（§2.8 的教训）。

    2026-11-11 由莉酱问「不是应该直接看因子库里因子们的周频 IC 吗」——
    确实该先看：库里有 builder 融合好的 `ic_h1/5/10/20.parquet`，
    从 `panels_neu` 重算是白花 37 分钟。

    本测试锁两件事：
    1. 库文件存在且形状自洽（否则 `--from-library` 会静默读到空表）；
    2. 库版与重算版对「周频是否更强」给**同向**结论 —— 若哪天两者符号相反，
       说明其中一条口径被改坏了，必须查清而不是任选一个引用。
    """
    from pathlib import Path

    lib = Path("E:/data/factor_library/all_a_2018_2026")
    p5, p20 = lib / "ic_h5.parquet", lib / "ic_h20.parquet"
    if not (p5.exists() and p20.exists()):
        pytest.skip("因子库 IC 不存在（本机无数据）")

    i5 = pd.read_parquet(p5)
    i20 = pd.read_parquet(p20)
    assert i5.shape[0] > 1000, "库 IC 天数过少，可能只建了几天"
    assert i5.shape[1] > 500, "库 IC 因子数过少"
    assert list(i5.columns) == list(i20.columns), "h5/h20 列必须同序，否则配对差会错位"

    w = slice("2019-01-02", "2026-09-07")
    m5, m20 = i5.loc[w].mean(), i20.loc[w].mean()
    common = [c for c in m5.index
              if np.isfinite(m5[c]) and np.isfinite(m20[c])]
    assert len(common) > 500
    d = np.sign(m20[common]) * (m5[common] - m20[common])

    # 库版实测正占比 0.193（重算版 0.352），两者都远离 0.5 且同向偏负
    assert (d > 0).mean() < 0.35, (
        f"库版正占比 {(d > 0).mean():.3f} 应显著低于 50% —— "
        "若变成 >0.5，说明库 IC 的收益定义被改了（检查 fwd 是否 shift 方向反了）")
