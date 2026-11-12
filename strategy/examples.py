"""
策略示例
========

常用策略实现，继承 Strategy。

tie-break 语义（2026-09-11 统一，全仓单一规则）
----------------------------------------------

因子值出现并列时（离散型因子、GBDT 同一叶子的样本）"取哪 k 只"必须**确定且
可解释**。此前三处 top-k 直接对原始因子值排序取值——pandas 默认
``kind="quicksort"``（introsort，**不稳定**）：并列时谁入选取决于底层排序实现
细节，同一版本内可复现，但**跨 pandas 版本/平台不保证**，语义上也解释不了。

现统一为 ``rank(ascending=False, method="first")`` —— 并列时**列序靠前者优先**，
与 :func:`strategy.constraints.build_signal_weights` 以及
:class:`BufferedTopFracLongOnly` 的内部实现是**同一语义**。

实测影响面（``scripts/oneoff/probe_tie_stable_sort.py``，HS300 2025 真实因子库
40 面板 / 46930 截面）：约 **4.4%** 的截面在 top-k 边界存在并列，而且几乎每个
并列截面新旧实现选出的持仓**集合都不同**（最坏 Jaccard = 0，即完全无交集）。
故本次改动会让受影响实验的持仓与换手发生可辨识变化，需按新口径重跑。
"""
from __future__ import annotations

import pandas as pd

from strategy.base import Strategy


# ---------------------------------------------------------------------------
# 确定性 top-k 选取（模块内共用，tie 语义的唯一实现）
# ---------------------------------------------------------------------------
def rank_desc(vals: pd.Series) -> pd.Series:
    """降序名次；并列按**列序**（``method="first"``）确定性打破。"""
    return vals.rank(ascending=False, method="first")


def _top_k_indices(vals: pd.Series, k: int) -> pd.Index:
    """因子值最大的 k 个 index（按值降序返回）；并列时列序靠前者优先。

    ``k <= 0`` 返回空——注意旧实现 ``index[-0:]`` 会返回**全部**索引
    （Python 的 ``-0 == 0``），k=0 时给出 ``inf`` 权重，属修复。
    """
    if k <= 0:
        return vals.index[:0]
    r = rank_desc(vals)
    return r[r <= k].sort_values().index


def _bottom_k_indices(vals: pd.Series, k: int) -> pd.Index:
    """因子值最小的 k 个 index（按值升序返回）；并列时列序靠前者优先。"""
    if k <= 0:
        return vals.index[:0]
    r = rank_desc(vals)
    n = len(vals)
    return r[r > n - k].sort_values(ascending=False).index


class TopKLongShort(Strategy):
    """Top-K 多空策略: 做多因子值最大的 K 只，做空最小的 K 只。

    Args:
        k: 每边持仓数量
        weight_mode: 'equal' 等权 / 'factor' 按因子值加权
    """

    def __init__(self, k: int = 30, weight_mode: str = "equal"):
        self.k = k
        self.weight_mode = weight_mode
        self.name = f"topk_ls_{k}_{weight_mode}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) < 2:
            # 只有 1 只：纯多头，避免 long/short 重叠导致权重 index 重复
            return pd.Series(1.0, index=vals.index)
        if len(vals) < 2 * self.k:
            # 标的不足：每边最多取 len//2，保证多空不重叠
            k = max(1, min(self.k, len(vals) // 2))
        else:
            k = self.k

        long_codes = _top_k_indices(vals, k)
        short_codes = _bottom_k_indices(vals, k)

        if self.weight_mode == "equal":
            w_long = pd.Series(1.0 / k, index=long_codes)
            w_short = pd.Series(-1.0 / k, index=short_codes)
        else:
            # 按因子值绝对值归一化
            long_vals = vals.loc[long_codes]
            short_vals = vals.loc[short_codes]
            w_long = long_vals / long_vals.abs().sum()
            w_short = -short_vals.abs() / short_vals.abs().sum()

        return pd.concat([w_long, w_short])


class TopKLongOnly(Strategy):
    """Top-K 纯多头策略: 只做多因子值最大的 K 只。"""

    def __init__(self, k: int = 30, weight_mode: str = "equal"):
        self.k = k
        self.weight_mode = weight_mode
        self.name = f"topk_lo_{k}_{weight_mode}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        k = min(self.k, len(vals))
        if k <= 0:
            # 空截面：旧实现会走到 `1.0 / 0` 抛 ZeroDivisionError，这里显式返回空。
            return pd.Series(dtype=float)
        long_codes = _top_k_indices(vals, k)

        if self.weight_mode == "equal":
            return pd.Series(1.0 / k, index=long_codes)
        else:
            long_vals = vals.loc[long_codes]
            return long_vals / long_vals.abs().sum()


class SectorCappedTopKLongOnly(Strategy):
    """Top-K 等权多头 + **每行业最多 N 只**（贪心填充）。

    2026-10-09：可交付组合口径。定版榜单的 ``picks`` 是 band 上限
    （全A Top10% ≈ 500+ 只），不是能下单的组合；实盘按
    「45~50 只 + 每申万一级 ≤2 只」执行。约束用贪心保证：
    分数从高到低逐只纳入，行业计数已满则跳过，凑满 k 即停
    （行业数 × max_per_sector < k 时实际持仓会不足 k）。

    行业截面需按 date 定位，故实现 :meth:`get_weights_at`；
    :meth:`get_weights` 拿不到日期，直接抛 ``NotImplementedError``。
    """

    def __init__(self, sectors: pd.DataFrame, k: int = 50,
                 max_per_sector: int = 2, weight_mode: str = "equal"):
        if k <= 0:
            raise ValueError(f"k 必须为正: {k}")
        if max_per_sector <= 0:
            raise ValueError(f"max_per_sector 必须为正: {max_per_sector}")
        self.sectors = sectors
        self.k = int(k)
        self.max_per_sector = int(max_per_sector)
        self.weight_mode = weight_mode
        self.name = f"sector_cap_topk_{k}_m{max_per_sector}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        raise NotImplementedError(
            "行业约束需按日期定位行业截面，请用 get_weights_at(date, vals)")

    def get_weights_at(self, date, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            return pd.Series(dtype=float)
        sect = (self.sectors.loc[date] if date in self.sectors.index
                else pd.Series(index=vals.index, dtype=object)).reindex(vals.index)
        # 与全仓唯一 tie-break 口径一致：降序名次（并列按列序）
        order = rank_desc(vals).sort_values().index
        counts: dict[str, int] = {}
        picked: list = []
        for code in order:
            s = sect.get(code)
            key = "__NA__" if s is None or pd.isna(s) else str(s)
            if counts.get(key, 0) >= self.max_per_sector:
                continue
            counts[key] = counts.get(key, 0) + 1
            picked.append(code)
            if len(picked) >= self.k:
                break
        if not picked:
            return pd.Series(dtype=float)
        if self.weight_mode == "equal":
            return pd.Series(1.0 / len(picked), index=picked)
        sub = vals.loc[picked]
        return sub / sub.abs().sum()


class SectorCappedBufferedTopKLongOnly(Strategy):
    """TopK 等权 + **排名缓冲带** + **每行业最多 N 只**（小资金集中组合）。

    2026-11-10 新增。三个约束叠加，各管一件事：

    - ``k`` / ``exit_mult``：控制**换手**。保留门槛 = 排名 ≤ ``k·exit_mult``，
      新进门槛 = 排名 ≤ ``k``（与 :class:`BufferedTopKLongOnly` 同规则）。
    - ``max_per_sector``：控制**行业押注**。分数从高到低贪心纳入，行业已满则跳过；
      设为 1 即「每个申万一级行业只留最强的一只」。实测这一步在 10 万整手组合上
      把年化从 31.3% 提到 38.3%（同持仓数），且通过随机行业标签安慰剂
      （10 次打乱 max 34.6%，真实 38.3%，100% 分位）。
    - ``min_holdings``：控制**尾部集中**。行业去重会让实际持仓少于 ``k``
      （TopK 内常有 2~3 只同行业），若某期只剩 2 只，等权就是单票 50%。
      设 ``min_holdings=m`` 后，持仓数 < m 时每只按 ``1/m`` 配、差额**留现金**，
      不硬凑满仓。m=8 时回撤从 36% 降到 26%，代价是年化同比例降。
    - ``gross_exposure``：整体仓位系数（``<1`` 即常驻现金）。与 ``min_holdings``
      不同，它是**无条件**按比例降杠杆 ⇒ Sharpe 基本不变、回撤与收益等比缩。

    .. note:: 行业截面需按 date 定位，故实现 :meth:`get_weights_at`。
       ``__NA__``（无行业分类，约 10%）**共用一个名额** —— 实测这些股票从不进
       TopK（把它们直接排除后数字逐位相同），故不影响结论。

    - ``fill_to_k``：被行业挡掉的空缺要不要往下补满 k。默认 ``False``（只在 TopK 内选）。

    - ``min_sectors``：控制**行业数下限**（2026-11-10 新增）。``max_per_sector=1``
      只保证「同行业不重复」，**不保证行业数**——TopK 内可能只剩 2~3 个行业，
      行业 beta 暴露不受控。设 ``min_sectors=n`` 后，若去重后行业数 < n，则从
      TopK **之外**（排名 ≤ ``k·sector_rank_mult``）按排名补「新行业」各 1 只，
      补到达标或持仓达到 ``k`` 为止。默认 0 = 关闭（与历史数字逐位一致）。

    .. note:: **被行业挡掉的空缺默认不往下补**（只在 TopK 内选，所以实际持仓 < k）。
       2026-11-10 实测过「补满 k」的版本（候选放宽到 ``k·exit_mult``）：K=6 b1.5
       38.30%→36.38%、**K=8 b2.0 35.08%→29.04%**（持仓 5.9→7.7）⇒ TopK 之外的
       alpha 衰减很快，欠仓比凑数好。故保留「不补」语义。``min_sectors`` 是**例外但受限**
       的补位：只补「行业数缺口」，不补「持仓数缺口」，且候选有排名上限
       （``sector_rank_mult``），不会去捞全市场第 2000 名。

    实测口径见 ``reports/focus_micro/README.md``。
    """

    def __init__(self, sectors: pd.DataFrame, k: int = 8, exit_mult: float = 1.0,
                 max_per_sector: int = 1, min_holdings: int = 1,
                 gross_exposure: float = 1.0, fill_to_k: bool = False,
                 min_sectors: int = 0, sector_rank_mult: float = 3.0):
        if k <= 0:
            raise ValueError(f"k 必须为正: {k}")
        if exit_mult < 1.0:
            raise ValueError(f"exit_mult 必须 ≥ 1: {exit_mult}")
        if max_per_sector <= 0:
            raise ValueError(f"max_per_sector 必须为正: {max_per_sector}")
        if min_holdings <= 0:
            raise ValueError(f"min_holdings 必须为正: {min_holdings}")
        if not 0 < gross_exposure <= 1.0:
            raise ValueError(f"gross_exposure 必须在 (0,1]: {gross_exposure}")
        if min_sectors < 0:
            raise ValueError(f"min_sectors 不能为负: {min_sectors}")
        if min_sectors > k:
            raise ValueError(f"min_sectors({min_sectors}) 不能超过 k({k})")
        if sector_rank_mult < 1.0:
            raise ValueError(f"sector_rank_mult 必须 ≥ 1: {sector_rank_mult}")
        self.sectors = sectors
        self.k = int(k)
        self.exit_mult = float(exit_mult)
        self.max_per_sector = int(max_per_sector)
        self.min_holdings = int(min_holdings)
        self.gross_exposure = float(gross_exposure)
        self.fill_to_k = bool(fill_to_k)
        self.min_sectors = int(min_sectors)
        self.sector_rank_mult = float(sector_rank_mult)
        self._prev: set = set()
        self.name = (f"sector_buffered_topk_{k}_{exit_mult}_m{max_per_sector}"
                     f"_h{min_holdings}_g{gross_exposure:g}"
                     f"{'_fillk' if fill_to_k else ''}"
                     f"{f'_ms{min_sectors}x{sector_rank_mult:g}' if min_sectors else ''}")

    def seed_holdings(self, codes) -> None:
        """把上一期的实际持仓灌入缓冲带状态（**仅单期推理/生产出榜用**）。

        回测由引擎按时序逐次调用 :meth:`get_weights_at` 自动维护 ``_prev``；生产
        只跑一个截面、没有时序驱动，故须由调用方从上一期落盘的持仓显式灌入。
        ⚠️ 回测中不要调用 —— 会覆盖引擎已建立的时序持仓，等价于制造前视。
        """
        self._prev = set(codes)

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        raise NotImplementedError(
            "行业约束需按日期定位行业截面，请用 get_weights_at(date, vals)")

    def get_weights_at(self, date: pd.Timestamp,
                       factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            self._prev = set()
            return pd.Series(dtype=float)
        sect = (self.sectors.loc[date] if date in self.sectors.index
                else pd.Series(index=vals.index, dtype=object)).reindex(vals.index)

        def _key(code):
            s = sect.get(code)
            return "__NA__" if s is None or pd.isna(s) else str(s)

        ranks = rank_desc(vals)
        n_exit = max(self.k, int(round(self.k * self.exit_mult)))

        # 1) 保留：上期持仓中排名仍在缓冲带内、且行业名额未满的
        counts: dict[str, int] = {}
        keep: list = []
        for code in sorted(self._prev, key=lambda c: ranks.get(c, float("inf"))):
            if code not in ranks.index or ranks[code] > n_exit:
                continue
            key = _key(code)
            if counts.get(key, 0) >= self.max_per_sector:
                continue
            counts[key] = counts.get(key, 0) + 1
            keep.append(code)
            if len(keep) >= self.k:
                break

        # 2) 补位：从排名最高开始贪心，行业已满则跳过。
        #    候选上限默认 = k（只在 TopK 内选）；fill_to_k=True 时放宽到保留带外沿
        #    ``k·exit_mult`` —— 实测更差（k8 b2.0: 35.08%→29.04%），默认关。
        limit = n_exit if self.fill_to_k else self.k
        fills: list = []
        for code in ranks.sort_values().index:
            if len(keep) + len(fills) >= self.k:
                break
            if ranks[code] > limit:
                break
            if code in self._prev and ranks[code] <= n_exit:
                continue                    # 已在 keep 名单（或刚被行业约束拒掉）
            key = _key(code)
            if counts.get(key, 0) >= self.max_per_sector:
                continue
            counts[key] = counts.get(key, 0) + 1
            fills.append(code)

        port = keep + fills

        # 3) 行业数地板：``max_per_sector`` 管「同质」，不管「只剩几个行业」。
        #    去重后行业数 < min_sectors 时，从 TopK 之外按排名补「新行业」各 1 只，
        #    候选上限 k·sector_rank_mult（不无限下探），补到行业达标或持仓满 k 为止。
        #    默认 min_sectors=0 ⇒ 整段不执行，历史数字逐位不变。
        if self.min_sectors > 0:
            have = {_key(c) for c in port}
            if len(have) < self.min_sectors:
                cap = int(round(self.k * self.sector_rank_mult))
                for code in ranks.sort_values().index:
                    if len(have) >= self.min_sectors or len(port) >= self.k:
                        break
                    if ranks[code] > cap:
                        break
                    if code in port:
                        continue
                    key = _key(code)
                    if key in have:
                        continue
                    have.add(key)
                    port.append(code)

        self._prev = set(port)
        if not port:
            return pd.Series(dtype=float)
        per = self.gross_exposure / max(len(port), self.min_holdings)
        return pd.Series(per, index=port)


class TopFracLongOnly(Strategy):
    """Top-Frac 重仓多头: 按比例 selected top-股做等权多头。

    与 TopKLongOnly（固定 k）的区别：k = round(frac * 当日有效信号数)，
    随股票池规模自适应。研究结论：中性化后的模型信号用 Top20% 重仓多头
    是 alpha 变现充分的组合形式（gbdt_tune 实验：frac=0.20,horizon=1
    于 2025 test 段成本后超额 +4.85%）。
    """

    name = "topfrac_lo"

    def __init__(self, frac: float = 0.20, weight_mode: str = "equal"):
        if not 0 < frac <= 1:
            raise ValueError(f"frac 必须在 (0,1]: {frac}")
        self.frac = float(frac)
        self.weight_mode = weight_mode
        self.name = f"topfrac_lo_{frac:.2f}_{weight_mode}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            return pd.Series(dtype=float)
        k = max(1, min(round(self.frac * len(vals)), len(vals)))
        top = _top_k_indices(vals, k)

        if self.weight_mode == "equal":
            return pd.Series(1.0 / k, index=top)
        top_vals = vals.loc[top]
        return top_vals / top_vals.abs().sum()

    def get_weights_at(self, date: pd.Timestamp, factor_values: pd.Series) -> pd.Series:
        return self.get_weights(factor_values)


class BufferedTopFracLongOnly(Strategy):
    """带排名缓冲带的 Top-Frac 多头（低换手变体）。

    指数式缓冲带规则（每期目标持仓数 N = round(frac_entry × 有效截面数)）：
    - 现有持仓：排名仍在前 frac_exit × n 名以内则保留（[entry, exit) 缓冲带内不换出）；
    - 空缺名额由排名最高的非持仓补足（仅考虑排名在 frac_entry × n 以内的候选）；
    - 截面超量收缩导致保留数 > N 时，按排名裁到 N；
    - 信号缺失（当日截面无值）的持仓卖出（无法排名；停牌场景由引擎
      executable_mask 另行处理）；
    - 等权 1/N。

    目的：TopFracLongOnly 每期按分数硬切 TopN，截止线附近的排名噪声直接变成
    交易（2025 实测月频单次单边换手 66%，其中大量无信息增量）。缓冲带把
    "进出门槛"分离，换手随 (frac_exit − frac_entry) 带宽下降。

    **有状态**：引擎按调仓日时序逐次调用 get_weights，实例内部维护上一期持仓。
    每个 VectorBacktest 必须使用全新实例——跨回测复用同一实例会把上一次回测的
    期末持仓泄漏为本次期初持仓（前视）。

    缺省 (0.15, 0.40) = 生产定版形态（2026-10-15 起与 dingban preset / caliber
    对齐；此前缺省 (0.20, 0.30) 是历史实验值，曾靠"勿用缺省"的口头纪律防错）。
    """

    def __init__(self, frac_entry: float = 0.15, frac_exit: float = 0.40):
        if not 0 < frac_entry <= 1:
            raise ValueError(f"frac_entry 必须在 (0,1]: {frac_entry}")
        if not frac_entry <= frac_exit <= 1:
            raise ValueError(
                f"需要 0 < frac_entry <= frac_exit <= 1: entry={frac_entry}, exit={frac_exit}")
        self.frac_entry = float(frac_entry)
        self.frac_exit = float(frac_exit)
        self.name = f"buffered_topfrac_lo_{frac_entry:.2f}_{frac_exit:.2f}"
        self._prev: set = set()

    def seed_holdings(self, codes) -> None:
        """把上一期的实际持仓灌入缓冲带状态（**仅单期推理/生产出榜用**）。

        回测由引擎按时序逐次调用 ``get_weights`` 自动维护 `_prev`；生产每天只跑
        一个截面，没有时序驱动，故须由调用方从上一期落盘的 picks 显式灌入。
        ⚠️ 回测中不要调用 —— 会覆盖引擎已建立的时序持仓，等价于制造前视。
        """
        self._prev = set(codes)

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            self._prev = set()
            return pd.Series(dtype=float)
        n = len(vals)
        n_entry = max(1, min(round(self.frac_entry * n), n))
        n_exit = max(n_entry, min(round(self.frac_exit * n), n))
        # tie 语义：method="first" 按列序确定性打破，与模块内 rank_desc 及
        # strategy.constraints.build_signal_weights 完全一致（本类一直是确定的）。
        ranks = rank_desc(vals)

        keep = [c for c in self._prev if c in ranks.index and ranks[c] <= n_exit]
        candidates = [c for c in ranks.sort_values().index
                      if c not in set(keep) and c not in self._prev and ranks[c] <= n_entry]
        portfolio = keep + candidates[:max(0, n_entry - len(keep))]
        if len(portfolio) > n_entry:
            portfolio = sorted(portfolio, key=lambda c: ranks[c])[:n_entry]

        self._prev = set(portfolio)
        return pd.Series(1.0 / len(portfolio), index=portfolio)


class BufferedTopKLongOnly(Strategy):
    """固定 K 多头的排名缓冲带（低换手变体；**有状态**，每回测须用新实例）。

    与 :class:`BufferedTopFracLongOnly` 同规则，但目标持仓数**固定为 K**，而非按截面
    比例 —— 截面数从 2019 年的 3400 只涨到 2025 年的 4900 只时，Top-Frac 的只数会跟着
    漂（411→734 只），TopK 不会。集中组合（小资金）用它。

    保留门槛 = 排名 ≤ K×``exit_mult``；新进候选门槛 = 排名 ≤ K。
    ``exit_mult=1.0`` 即无缓冲带（等价硬切 TopK，但实现仍有状态，可安全替换）。

    .. note:: 2026-11-10 从 ``scripts/evaluation/portfolio_weighting.py`` 迁入
       （策略定义不该住在评测脚本里），原位置保留同名再导出以兼容既有调用方。
       迁移未改任何逻辑，只把 ``vals.rank(ascending=False, method="first")``
       换成等价的 :func:`rank_desc`，与模块内其余策略共用同一个 tie-break。
    """

    def __init__(self, k: int = 45, exit_mult: float = 1.5):
        if k < 1:
            raise ValueError(f"k 必须 ≥ 1: {k}")
        if exit_mult < 1.0:
            raise ValueError(f"exit_mult 必须 ≥ 1: {exit_mult}")
        self.k = int(k)
        self.exit_mult = float(exit_mult)
        self._prev: set = set()
        self.name = f"buffered_topk_lo_{k}_{exit_mult}"

    def seed_holdings(self, codes) -> None:
        """把上一期实际持仓灌入缓冲带（仅生产单期推理用，回测勿调，否则制造前视）。"""
        self._prev = set(codes)

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if len(vals) == 0:
            self._prev = set()
            return pd.Series(dtype=float)
        n_exit = max(self.k, int(round(self.k * self.exit_mult)))
        ranks = rank_desc(vals)
        keep = [c for c in self._prev if c in ranks.index and ranks[c] <= n_exit]
        cands = [c for c in ranks.sort_values().index
                 if c not in self._prev and ranks[c] <= self.k]
        port = keep + cands[:max(0, self.k - len(keep))]
        if len(port) > self.k:
            port = sorted(port, key=lambda c: ranks[c])[:self.k]
        self._prev = set(port)
        return pd.Series(1.0 / max(len(port), 1), index=port)


class QuantileLongShort(Strategy):
    """分位多空策略: 做多最高分位，做空最低分位。"""

    def __init__(self, n_quantiles: int = 5):
        self.n = n_quantiles
        self.name = f"quantile_ls_{n_quantiles}"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            return pd.Series(dtype=float)

        # 分位分组
        try:
            groups = pd.qcut(vals, self.n, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(dtype=float)
        n_actual = groups.nunique()
        if n_actual < 2:
            return pd.Series(dtype=float)

        long_mask = groups == groups.max()
        short_mask = groups == groups.min()

        long_codes = vals[long_mask].index
        short_codes = vals[short_mask].index

        w_long = pd.Series(1.0 / len(long_codes), index=long_codes)
        w_short = pd.Series(-1.0 / len(short_codes), index=short_codes)
        return pd.concat([w_long, w_short])


def build_strategy(name: str, k: int = 30, frac: float | None = None):
    """按名称构造策略实例（run_backtest / select_stocks 共用工厂，2026-08-05 统一）。

    注（2026-09-21）：`run_backtest` 已删除、`select_stocks` 已归档到
    ``scripts/archive/``；工厂本身仍为活跃调用方（`model_portfolio` /
    `multi_period` 等）服务。

    Args:
        name: topk_ls（TopK 多空）| topk_lo（TopK 纯多）| quantile（分位多空）
              | topfrac_lo（Top-Frac 比例多头）
        k: TopK 持仓数（quantile 忽略；topfrac_lo 忽略）。
        frac: topfrac_lo 持仓比例（默认 0.20）。
    """
    if name == "topk_ls":
        return TopKLongShort(k=k)
    elif name == "topk_lo":
        return TopKLongOnly(k=k)
    elif name == "quantile":
        return QuantileLongShort(n_quantiles=5)
    elif name == "topfrac_lo":
        return TopFracLongOnly(frac=frac if frac is not None else 0.20)
    raise ValueError(f"未知策略: {name}（可选 topk_ls / topk_lo / quantile / topfrac_lo）")
