"""
因子库（Factor Library / Factor Zoo）
====================================

把挖掘/合成产出的因子**持久化**为一个可查询、可对比、可迭代的因子库。

解决的问题（用户需求）：
1. 每个因子都能快速查看**某一段时间**的回测结果 —— 通过预存 IC 序列与多套
   canonical 回测的净值/日收益，``evaluate_period`` 按时间段切片、重算指标，秒级返回。
2. 所有因子可以**统一对比**绩效指标、挑选好因子 —— ``compare`` 按任意指标/配置排序。
3. 复合因子**入库并参与下一轮迭代** —— 复合因子记录 ``parents`` 血缘；``load_library_features``
   把它们作为新特征喂给下一轮挖掘（``mine_factors --use-library``）。

持久化结构（root 默认 e:/data/factor_library/）：
    registry.csv              # 所有因子的元数据 + 各配置绩效指标
    panels/<slug>.parquet     # 预计算因子面板 (date × code)，已截面标准化
    evals/<slug>.parquet      # ic 序列 + 各 canonical 配置的 equity/dret（按日期索引）

按数据集分库根（dataset）：
    传入 dataset=<name> 时，上述三件套落在 root/<dataset>/ 子目录下，互不串扰。
    FactorLibrary.list_datasets() 可列出全部数据集；CLI 用 --dataset 指定，
    mine/synthesize 用 --library-dataset（不传则自动推导：真实→<指数>_<年>，mock→mock）。

canonical 回测配置（注册时一次性算好，之后查看/对比都不重算）：
    ls_M  : TopK 多空, k=30, 月调仓
    lo_M  : TopK 纯多, k=30, 月调仓
    ls_W  : TopK 多空, k=30, 周调仓

两段式入库（2026-11-02）：
    ``ingest``   面板落盘 + IC 序列 + 多周期 IC + 标签     ≈ 17 s/因子（关 evidence；
                 再关 horizons 约 8 s；开 evidence 约 102 s，那三列不参与门槛）
    ``evaluate`` 三套 canonical 回测 + eval 落盘 + 回填指标 ≈ 3.3 s/因子（all_a）
    ``register`` 两者的兼容包装（900+ 处调用点，行为不变）

    拆分的理由：合在一起时"便宜入库"没有入口，批量通道只好绕过库入口直写
    registry，于是库分成"有评估的"和"只有面板的"两半。状态用
    ``eval_status``（ingested/evaluated）+ ``evaluated_at`` 显式表达，
    契约测试见 ``tests/test_factor_library_contract.py``。

    ⚠️ 耗时结构反直觉（2026-11-02 实测）：三套回测只要 3.3 s，131 s 里
    90% 是 ``ic_decay5``/``monotonicity``/``pfs`` 三个证据列。所以"补评估"
    在并行下是十分钟级动作，不要被 register 的总耗时吓住。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import ShortCostModel
from backtest.engine import BacktestResult, VectorBacktest
from backtest.metrics import PERIODS_PER_YEAR, calc_all_metrics, sharpe_ratio
from research.factor_analysis import calc_ic_decay, calc_ic_series, calc_ir, factor_autocorr
from strategy.examples import TopKLongOnly, TopKLongShort

log = logging.getLogger("factor_library")


# ===========================================================================
# canonical 回测配置
# ===========================================================================
class _Config:
    key: str          # 列后缀，如 "ls_M"
    label: str        # 人类可读
    strategy: type    # TopKLongShort / TopKLongOnly
    freq: str         # D / W / M
    k: int = 30

    def __init__(self, key, label, strategy, freq, k=30):
        self.key = key
        self.label = label
        self.strategy = strategy
        self.freq = freq
        self.k = k


CANONICAL_CONFIGS: list = [
    _Config("ls_M", "TopK多空·月", TopKLongShort, "M", 30),
    _Config("lo_M", "TopK纯多·月", TopKLongOnly, "M", 30),
    _Config("ls_W", "TopK多空·周", TopKLongShort, "W", 30),
]

#: 因子库**预计算的多周期截面 IC 面板**（``ic_h{h}.parquet``，date × factor）。
#:
#: builder 每次入库时由 ``scripts.builders.common`` 的 IC 融合步骤写入，
#: **是已落盘的事实**：全A 库 936 因子 × 2511 日，四个 horizon 合计读盘 **0.34 s**。
#:
#: 🔑 **要 IC 先读这里，不要重算**。2026-11-11 踩过：问「因子周频 IC」时从
#: ``panels_neu`` 逐因子重算，花了 37 分钟，而库里有现成的（0.34 s）。
#: 同口径重算的正当理由只有两个：要 t 值/非重叠对照，或要换一份复权价。
#:
#: ⚠️ 与 :meth:`FactorLibrary.evaluate` 现算的 IC **不同源**：库版用
#: ``IC_CODE_STRIDE=3`` 把截面抽稀到 1/3 只股票、且 close_adj 是另一份
#: （复权价最大差 3.5%，``ic_h20`` 逐因子相关 0.983）。引用时必须写明用哪一份。
IC_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)

_METRIC_COLS = ["annual_return", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "avg_turnover",
                "avg_margin_usage", "borrow_fee_drag_annual"]

#: 因子族（family）受控词表——**唯一真源**（2026-09-12 统一规范）。
#:
#: 历史上库的 family 是两个维度的并集：风格族（动量/反转/波动率/价值/质量/成长）
#: 与来源族（情绪/非线性组合/量价…）混用，且出现过英文缩写漂移
#: （rnd/liq/qual/cfq/lev/pro）。这里把两者并成一个受控词表，值必须落在词表内。
#: 比 family 更细的分类用 ``subfamily`` 列承载（不受控，自由文本）。
FAMILY_WHITELIST: frozenset[str] = frozenset({
    # 风格族
    "动量", "反转", "波动率", "价值", "质量", "成长", "流动性", "拥挤度",
    # 来源/主题族
    "量价", "技术", "日内", "基本面", "事件", "资金流", "股东", "状态",
    "情绪", "文本", "论文", "模型", "合成", "非线性组合",
    # 兜底
    "其他",
})

#: 信号频率受控词表。
FREQUENCY_WHITELIST: frozenset[str] = frozenset({"日内", "日频", "周频", "月频", "季频"})

#: 成熟度状态枚举（小写）。
#:
#: ``invalid`` = **该因子本身没有可排名的截面信息**（面板全空 / 整列同值 /
#: 时序变量误入截面库 / 可排名率 <5%），不是"还没验证"，而是"验证不了"。
#: 与 ``retired`` 的区别：retired 是曾经可用、现在停用；invalid 是从来没成立过。
#: 二者都不得进入任何选因子/组合链路。
MATURITY_VALUES: frozenset[str] = frozenset(
    {"invalid", "experimental", "oos_verified", "active", "retired"})

#: 成熟度晋升门槛（2026-11-05 落码，**改阈值等于改"什么算可用"，只在这里改**）。
#:
#: 之所以要门槛而不是"看排名"：库里 965 个因子，只要按 sharpe 排序就总能有前 10 名，
#: 而"前 10 名"是相对量、不是合格量。这组阈值是绝对量，任一条不过就是不过。
#:
#: 阈值来历（都对着实测分布定的，不是拍脑袋）：
#:
#: * ``min_sharpe`` = 0.5：新口径（复权 + 可交易掩码）下全库中位 **−0.29**、
#:   ≥0.5 的只有 83 个（前 8.6%）。对照量级：`top10_sharpe` 榜单最好 0.52~0.55、
#:   4,843 篇文献复现中位 0.37 ⇒ 0.5 意味着"明显好于文献基线"，而不是"库里靠前"。
#: * ``min_recent_sharpe`` = 0.3：近 250 日窗口的夏普标准误约 ±0.25（√252/√250），
#:   0.3 是"仍为正且不至于落在噪声带里"的下限；低于它不能说因子还活着。
#: * ``max_dup_corr`` / ``min_resid_ic``：652 个近重复里，相关 ≥0.95 但残差 IC
#:   ≥0.02 的（如 `alpha158_VSUMN30` vs `alpha360_VOLUME30`）确实有增量，
#:   所以"相关高"本身不判死，要"高相关 **且** 残差没信息"才算重复。
#: * ``max_active_corr`` = 0.85：`active` 是**生产白名单**，内部不许留近亲。
#:   0.95 的完全链接簇太松（VOLUME 家族彼此 0.83~0.87 不在同簇却近乎同源），
#:   改用"按 sharpe 降序贪心接受、与已选任一相关 ≥0.85 则拒"——结果可解释为
#:   "active 集合内任意两因子 |corr| < 0.85"。
#: * ``max_per_family`` = 2：**相关性抓不住的重复，靠命名族兜底**。实测要紧：
#:   `alpha360_VOLUME36` 与 `alpha360_VOLUME9` 截面相关只有 0.4~0.6（跨越长短窗口
#:   确实不重合），只靠相关去重会放进 **43 个** active，其中 40 个是同一公式的
#:   窗口网格——它们是"一个想法调 60 次参数"，不是 60 个发现。平均链接聚类也救不了
#:   （@corr≥0.6 仍有 6 个 VOLUME 簇）。族键 = 剥掉末尾连续数字（编号型命名如
#:   `alpha191_166` 剥完以 ``_`` 结尾，判为编号不剥，否则会把 180 个 alpha191 归一族）。
MATURITY_GATES: dict[str, float] = {
    "min_sharpe": 0.5,
    "min_recent_sharpe": 0.3,
    "max_dup_corr": 0.95,
    "min_resid_ic": 0.01,
    "max_active_corr": 0.85,
    "max_per_family": 2,
    "recent_days": 250,
    "split_frac": 0.7,
    "min_seg_days": 60,
}

#: 晋升失败原因码 → 人话（报告与 note 共用，别在两处各写一套措辞）。
MATURITY_REASONS: dict[str, str] = {
    "no_eval": "无 eval 或 sharpe 为空",
    "stale_caliber": "收益口径非复权/未套掩码",
    "dup_unknown": "未做冗余检查（含死因子）",
    "low_sharpe": "净多空夏普未达线",
    "redundant": "与最近邻重复且残差无增量",
    "unstable": "后段 IC 反向或后段夏普为负",
    "fading": "近 250 日夏普未达线",
    "crowded": "与已选 active 高相关",
    "family_cap": "同族代表已达上限",
    "uncovered": "不在相关矩阵内，无法证明独立性",
}

#: ``source`` 命名规范：``<prefix>:<detail>[:<scope>]``。
#:
#: prefix 取自本集合；detail 建议为因子集名（alpha101/evt/moneyflow…）；
#: scope 为数据集或区间（如 ``all_a_2018_2026``）。**下游用
#: ``source.split(":")[0]`` 做路由**（见 ``scripts/ingest/extend_factor_library.py``
#: 的 gp/model/alpha 因子集判定），改 prefix 等于改路由键，不要随手改。
SOURCE_PREFIXES: frozenset[str] = frozenset({
    # 产出方（无法归到具体因子集时的兜底前缀）
    "builders", "textmining", "paper", "mining", "synthesis", "synthesis_library",
    # 库内既有来源前缀
    "model", "gp", "alpha101", "alpha158", "alpha191", "alpha360", "technical",
    "intraday", "fundamental", "events", "moneyflow", "margin", "holder", "status",
    # 因子集名——builder 通道的**首选前缀**，与 hs300 既有口径一致
    # （``alpha101:build_alpha_factors:20220101-20260821``）
    "evt", "event", "sentiment", "constructed", "style", "pledge", "sue_pledge",
    "disc_holder_dyn", "holder_dyn", "significant_synthesis", "p5",
    # 另类数据因子集（2026-10-09 首次入库；insider/macro/news 三族）
    "altf",
    # 构造/搜索产出的两个集（2026-11-07 补登：两者在 registry 里都有 15/14 行，
    # 但集名一直不在册，`source_for` 每次调用都会告警，family 只能留空）
    "e4x", "llmmcts",
})

#: 无法识别产出脚本时 ``source_for`` 的默认 producer 段。
DEFAULT_PRODUCER = "builders"

#: 因子集名（registry ``set`` 列 / builder 的 ``factor_stats_{set}.jsonl`` 名）
#: → 因子族（``family`` 列）默认映射。
#:
#: 用途有二：① 批量入库（builder）自动补 family，免得新因子再落进"有来源无分类"；
#: ② 历史数据回填的**唯一口径**（回填脚本 import 本表，不另立一套）。
#: 未收录的 set 返回空串——宁可留空，也不猜。
SET_TO_FAMILY: dict[str, str] = {
    # 量价公式族
    "alpha101": "量价", "alpha158": "量价", "alpha191": "量价", "alpha360": "量价",
    # 财务基本面（含构造型/风格型）
    "fundamental": "基本面", "constructed": "基本面", "style": "基本面",
    "sue_pledge": "基本面", "p5": "基本面",
    # 股东（持股/质押/折价；holder_dyn = 增减持/内部人交易等**动态**持股，2026-09-20 另类数据 P0）
    "holder": "股东", "pledge": "股东", "disc_holder_dyn": "股东", "holder_dyn": "股东",
    # 事件与资金
    "evt": "事件", "event": "事件",
    "moneyflow": "资金流", "margin": "资金流",
    # 其余
    "status": "状态",
    # 构造型：E4 = 基本面分桶 × 量价因子的组内 zscore，本质是**两个父因子的组合**
    # （与 synthesize 同性质）；LLM-MCTS 是**公式搜索**产物（与 gp/gflownet 同性质）。
    # 两个集名 2026-11-07 才补进册，此前 29 行只能留空。
    "e4x": "合成",
    "llmmcts": "非线性组合",
    # 另类数据三族统一归「事件」（增减持公告 / 宏观日历 surprise / 快讯情绪，
    # 2026-10-09 首次入库；此前 merge_outputs 传 family_tag="其他"）
    "altf": "事件",
    "technical": "技术", "intraday": "日内",
    "paper": "论文", "model": "模型", "synthesis": "合成", "textmining": "文本",
    "sentiment": "情绪",
}


def family_for_set(set_name: str) -> str:
    """按因子集名推断因子族；未收录返回 ``""``（不猜）。"""
    return SET_TO_FAMILY.get((set_name or "").strip(), "")


def source_for(set_name: str, *, producer: str = DEFAULT_PRODUCER,
               scope: str = "") -> str:
    """生成规范 ``source``：``<因子集名>:<产出脚本>[:<数据集/区间>]``。

    **前缀优先取因子集名，而不是通用 ``builders``**——``source.split(":")[0]``
    在下游是被当路由键用的：

    - ``scripts/ingest/extend_factor_library.py`` 用 ``src == "gp"`` /
      ``src == "model"`` 决定要不要重算这批因子；
    - ``scripts/archive/factor_usability_stats.py`` 按前缀做来源分组；
    - ``research/factor_report.py`` 用前缀判定 alpha/gp/model 族。

    全库统一写成 ``builders:*`` 会让这些消费者塌缩成单一来源桶（gp/model
    路由直接失效）。因子集名不在 :data:`SOURCE_PREFIXES` 时由调用方告警。

    Args:
        set_name: 因子集名（registry ``set`` 列值）。
        producer: 产出脚本名（如 ``build_alla_alpha_panels``）。
        scope: 数据集名或日期区间（如 ``all_a_2018_2026``）。

    >>> source_for("alpha101", producer="build_alla_alpha_panels",
    ...            scope="all_a_2018_2026")
    'alpha101:build_alla_alpha_panels:all_a_2018_2026'
    """
    parts = [x for x in ((set_name or "").strip(), (producer or "").strip(),
                         (scope or "").strip()) if x]
    return ":".join(parts)


def normalize_tags(*, family: str = "", frequency: str = "", maturity: str = "",
                   where: str = "") -> tuple[str, str, str]:
    """把因子标签收敛到受控词表。

    越界值**只告警、不静默改写**——静默改写会掩盖真实的口径漂移，而直接拒绝
    会让历史数据无法入库。告警里带上 ``where`` 便于定位写入方。
    """
    fam = (family or "").strip()
    freq = (frequency or "").strip()
    mat = (maturity or "").strip()
    if fam and fam not in FAMILY_WHITELIST:
        log.warning("[%s] family=%r 不在 FAMILY_WHITELIST；新增族请同步词表", where, fam)
    if freq and freq not in FREQUENCY_WHITELIST:
        log.warning("[%s] frequency=%r 不在 FREQUENCY_WHITELIST", where, freq)
    if mat and mat not in MATURITY_VALUES:
        log.warning("[%s] maturity=%r 不在 MATURITY_VALUES", where, mat)
    return fam, freq, mat


#: registry 的规范列序——**唯一 schema 真源**（2026-09-12 统一规范）。
#:
#: 此前 ``_load_registry`` 用内联清单，缺 ``set`` / ``label`` / ``coverage`` /
#: ``ic_mean_h1..h20`` / ``p_value_nw``；而批量入库（``merge_outputs``）只写
#: 后几列，于是 ``all_a_2018_2026`` 有 ``set`` 列、``hs300_2025`` 没有——
#: 同一套库出现了两套 schema。列清单收在这里，`_load_registry` 与 `_save_registry`
#: 共用，未知列一律保留在尾部（不丢数据）。
CANONICAL_COLUMNS: list[str] = [
    # —— 标识与来源 ——
    "name", "kind", "formula", "source", "dataset", "parents", "created_at",
    # —— 六维标签子集（家族/子族/频率/成熟度；其余维放 note）——
    # ``subfamily`` 是**细粒度**子类（不受控词表），用于 family 粗粒度化后
    # 仍保留原始分类信息。典型场景：``family=基本面`` 下的
    # 研发投入/财务流动性/杠杆/盈利能力/现金流质量——它们不能直接塞进
    # ``family``，因为白名单里的「流动性」指**市场流动性**（换手率/Amihud），
    # 与「流动比率」这类**财务流动性**不是一回事，混用即语义碰撞。
    "family", "subfamily", "frequency", "maturity", "note",
    # —— 批量入库（merge_outputs）自带列 ——
    "set", "label", "coverage", "neutralized_coverage",
    "ic_mean_h1", "ic_mean_h5", "ic_mean_h10", "ic_mean_h20",
    # —— 检验统计 ——
    "n_dates", "n_codes", "ic_mean", "ic_std", "ic_ir", "t_stat", "t_stat_nw",
    "p_value_nw", "ic_win_rate", "ic_decay5", "autocorr", "significant",
    # —— 结构化证据（2026-09-22 evals 改造包：报告列，均不参与入库门槛）——
    "monotonicity", "pfs",
    # —— 入库前冗余预检（check_dup）——
    "dup_checked", "dup_corr_max", "dup_top", "resid_ic", "resid_t_nw",
] + [f"{m}_{c.key}" for c in CANONICAL_CONFIGS for m in _METRIC_COLS] + [
    "best_sharpe", "best_config", "panel_path", "eval_path",
    # —— 评估状态（2026-11-02 两段式入库）——
    # ``eval_status``: ingested（只登记+IC）/ evaluated（跑过 canonical）
    # ``evaluated_at``: 最后一次 evaluate 的时间戳（空=从未评估）
    # ``ret_basis``: 收益口径标记（``adj_close_pct_v1`` = 复权收盘 1 日收益；
    #   2026-11-04 之前是 ``close_pct_v1`` = 未复权）
    # ``mask_basis``: 回测是否套了可交易掩码（``tradable_t0`` = T 日状态，
    #   与 shift(-1) 收益的 T 日收盘建仓自洽；空 = 未套）
    "eval_status", "evaluated_at", "ret_basis", "mask_basis",
]


#: **计算型统计列**——每次 builder 重跑都会重算，必须允许覆盖旧值。
#:
#: 与 :data:`CANONICAL_COLUMNS` 里的其余列分两类处理（2026-10-27）：
#:
#: - **标签/标识类**（``name`` / ``kind`` / ``source`` / ``family`` /
#:   ``subfamily`` / ``frequency`` / ``maturity`` / ``note`` / ``set`` /
#:   ``label`` / ``formula`` / ``parents`` / ``panel_path`` / ``eval_path``）：
#:   默认模式（``fill_missing_only=True``）下**只补空缺、不覆盖**；其中
#:   :data:`TAG_COLUMNS` 那 5 列即使在 ``False`` 模式下也保持旧值优先。
#: - **统计类**（本清单）：由面板/IC 现算，**两种模式都刷新**。
#:
#: 历史坑（2026-09-20 首次暴露，10-27 收口）：``upsert_rows`` 曾对整个行做
#: ``old.combine_first(new)`` —— 旧值非空即胜，于是**任何计算型字段一旦写过，
#: 后续修正值永远进不去 registry**。holder_dyn 踩得最狠：dtype bug 态跑出
#: ``active_day_frac=0.008``，修好重跑、stats 已是正确值，registry 仍是 0.008
#: （当时只能在 ``scripts/oneoff/_fix_holder_dyn_registry.py`` 里外科回填）。
#: 更隐蔽的是 ``coverage``：数据天天长，重建后面板覆盖变了，registry 却冻结在
#: 首次入库态 —— 而 ``MIN_COVERAGE`` / ``MIN_NEU_COVERAGE`` / ``MIN_IC_COVERAGE``
#: 三道过滤正是读它。
STAT_COLUMNS: frozenset[str] = frozenset([
    # 面板覆盖（raw 口径；中性化覆盖另立 neutralized_coverage）
    "coverage", "neutralized_coverage",
    # 事件类因子的活跃度（altf / holder_dyn）
    "active_day_frac", "active_first", "active_last",
    # 多周期 IC
    "ic_mean_h1", "ic_mean_h5", "ic_mean_h10", "ic_mean_h20",
    # 检验统计
    "n_dates", "n_codes", "ic_mean", "ic_std", "ic_ir", "t_stat", "t_stat_nw",
    "p_value_nw", "ic_win_rate", "ic_decay5", "autocorr", "significant",
    # 结构化证据
    "monotonicity", "pfs",
    # 冗余预检
    "dup_checked", "dup_corr_max", "dup_top", "resid_ic", "resid_t_nw",
    # 分组回测
    "best_sharpe", "best_config",
    # 评估状态（两段式入库：ingest/evaluate 之间的状态机）
    "eval_status", "evaluated_at",
]).union({f"{m}_{c.key}" for c in CANONICAL_CONFIGS for m in _METRIC_COLS})


#: **人工可编辑的标签列**——与 :data:`STAT_COLUMNS` 互补。
#:
#: 清单 = :meth:`FactorLibrary.set_tag` 能改的那几列。这些列一旦被人工改过
#: （例如把某个 alpha101 因子标成「动量」而非默认的「量价」），任何 builder
#: 重跑都不该把它冲回去，因此**两种 upsert 模式下都旧值优先**，只补空缺。
#: ``subfamily`` 不在其中：它由 builder 从统计行写入（见 altf），属机器派生。
TAG_COLUMNS: frozenset[str] = frozenset([
    "family", "frequency", "maturity", "note", "source",
])


def _cmp_parts(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """把一列拆成 (数值视图, 归一化文本视图)。

    文本视图里 ``NaN`` / 空串 / 字面 "nan"/"none"/"<na>" 全部归一成 ``""``
    —— registry CSV 里"空"有两种落盘形态（``""`` 和 NaN），读回来必须视为
    同一状态，否则统计"改了几格"会把格式差异算成内容变化。

    .. warning::
       文本视图**只用于非数值对**。pandas 3 的 ``astype(str)`` 保留完整
       repr，同一个 float 经 CSV 往返会变成 ``-0.0221045296639204`` vs
       ``-0.022104529663920403`` —— 拿它比较数值对必然误报。
    """
    t = s.astype(str).where(s.notna(), "").str.strip()
    t = t.where(~t.str.lower().isin(["", "nan", "none", "<na>"]), "")
    num = pd.to_numeric(t.where(t != "", np.nan), errors="coerce")
    return num, t


def _count_changed(old: pd.DataFrame, new: pd.DataFrame) -> int:
    """统计 ``new`` 相对 ``old`` 真正变化的单元格数（浮点带容差）。

    用于 ``upsert_rows`` 的刷新计数日志：只想知道"这次重跑到底改了多少"。
    数值对用 ``isclose(atol=1e-12)``（受得住 repr 抖动）；其余按归一化文本比。
    """
    n = 0
    for col in new.columns:
        a = old[col] if col in old.columns else pd.Series(np.nan,
                                                          index=old.index)
        b = new[col]
        an, at = _cmp_parts(a)
        bn, bt = _cmp_parts(b)
        both = an.notna() & bn.notna()
        n += int((both & ~np.isclose(an.where(both, 0.0), bn.where(both, 0.0),
                                     rtol=0.0, atol=1e-12)).sum())
        n += int((~both & (at != bt)).sum())
    return n


def _align_registry(reg: pd.DataFrame) -> pd.DataFrame:
    """把 registry 的列集/列序对齐到 :data:`CANONICAL_COLUMNS`。

    缺列补 ``NaN``（**不补 pd.NA**：那会让数值列变 object，下游
    ``reg["ic_mean"].abs()`` 这类调用会直接炸），多余列保留在尾部不丢。
    已对齐时原样返回——``list_all`` / ``get_panel`` 会高频走这条快速路径。

    .. warning::
       补列的循环**必须在算 ``head`` 之前**完成。2026-09-12 的写法是
       ``head = [c for c in CANONICAL_COLUMNS if c in reg.columns]`` 先算、
       再补列、最后 ``reg[head + tail]`` 取列——于是补出来的列（``head`` 里
       没有、又不在 ``tail`` 里）被静默丢弃：新加的 canonical 列在旧库上
       **永远无法落地**，表现为 ``list_all()`` 少一列却毫无告警。
    """
    cur = list(reg.columns)
    n = len(CANONICAL_COLUMNS)
    if len(cur) >= n and cur[:n] == CANONICAL_COLUMNS:
        return reg
    for c in CANONICAL_COLUMNS:
        if c not in reg.columns:
            reg[c] = np.nan
    tail = [c for c in reg.columns if c not in CANONICAL_COLUMNS]
    return reg[list(CANONICAL_COLUMNS) + tail]


def _collapse_duplicate_names(reg: pd.DataFrame, *, where: str = "") -> pd.DataFrame:
    """同一 ``name`` 出现多行时收敛为一行（**保留最后一行**）。

    重复登记的成因见 :func:`scripts.builders.common.read_stats` 的注释：builder 的
    ``factor_stats_*.jsonl`` 是**追加**写的，读回时不按 name 去重，整批同名行
    原样进 registry（``upsert_rows`` 只挡「库里已有的」，不挡「批内重复」）。
    2026-10-27 实测 ``all_a_2018_2026``：1112 行 / 950 唯一 —— 152 个名字被登记
    2~3 次，全部集中在非 alpha 族（alpha 四族因 ``_merge_outputs`` 全量覆盖而幸免）。

    为什么"保留最后一行"就是"保留最新统计"：追加顺序即运行顺序，且实测 152 组里
    ``active_last`` 可比的 26 组 **0 例倒退**、``coverage`` 144/152 单调不降。

    收敛放在**读入口**而不是写入口，是为了让所有消费者（``list_all`` /
    ``has`` / 下游直接 ``set_index("name")`` 的脚本）一次治干净：重复索引会让
    ``reg["family"].get(n)`` 返回 Series（静默污染）、``float(reg.loc[n, "coverage"])``
    抛 ``TypeError``。落盘由下一次 :meth:`_save_registry` 顺带完成，不额外改盘。

    Args:
        reg: 已对齐列序的 registry。
        where: 日志标识（``load:<dataset>`` / ``upsert``），便于定位来源。

    Returns:
        pd.DataFrame: 去重后的 registry（原位序保持首次出现顺序，索引已重置）。
    """
    if reg.empty or "name" not in reg.columns:
        return reg
    dup = reg["name"].duplicated()
    if not bool(dup.any()):
        return reg
    n_before, n_dup_names = len(reg), int(reg.loc[dup, "name"].nunique())
    out = reg.drop_duplicates(subset="name", keep="last").reset_index(drop=True)
    log.warning("registry 重复登记（%s）：%d 行 → %d 个唯一因子（%d 个名字重复，"
                "已保留最后一次统计）。根因 = factor_stats jsonl 追加写 + "
                "read_stats 未去重", where, n_before, len(out), n_dup_names)
    return out


def _slug(name: str) -> str:
    """把因子名（可能是长公式）映射为安全的文件名片段。"""
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def _write_parquet_robust(df: pd.DataFrame, path: Path, retries: int = 4) -> None:
    """健壮写盘：**临时文件 + 原子替换**，不做 unlink。

    历史实现是"先删旧文件再写"，为绕过 Windows 上 pyarrow 覆盖写的偶发
    PermissionError。但 2026-11-02 的回填暴露了它的两个问题：
    ① 批量回填要重写几百个 eval，触发沙箱的"批量删除需确认"保护，进程挂住；
    ② 删了再写不是原子操作，中途失败会留下缺文件的因子。

    改成写 ``<name>.tmp.parquet`` 再 ``os.replace``：既原子、又不产生删除，
    Windows 上目标文件被占用时用退避重试兜住。
    """
    tmp = path.with_suffix(".tmp.parquet")
    for attempt in range(retries):
        try:
            df.to_parquet(tmp, compression="snappy")
            tmp.replace(path)
            return
        except (PermissionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (2 ** attempt))


def _coerce_date(d) -> pd.Timestamp | None:
    if d is None:
        return None
    if isinstance(d, (int, np.integer)):
        s = str(int(d))
        return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return pd.Timestamp(str(d))


def _sanitize_dataset(ds: str) -> str:
    """把数据集名映射为安全的子目录名。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(ds).strip())


# ===========================================================================
# 因子库
# ===========================================================================
def _finite(v) -> bool:
    """数值有限即为真；NaN/None/空串/all-NA 一律 False（registry 里三者都有）。"""
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _f(v) -> float:
    """:func:`_finite` 的取值版：非有限值一律折成 ``nan``（便于统一比较）。"""
    return float(v) if _finite(v) else float("nan")


def _evidence_brief(rec) -> str:
    """一行证据摘要，写进 registry 的 ``note``（达标因子用）。

    目的：让 ``active`` / ``oos_verified`` 行在表里就能自证，不用回翻报告。
    """
    def g(k, fmt, scale=1.0):
        try:
            v = float(rec.get(k))
        except (TypeError, ValueError):
            return "-"
        return "-" if not np.isfinite(v) else format(v * scale, fmt)
    return (f"夏普 {g('sharpe', '.2f')}；后段 {g('sharpe_fwd', '.2f')}；"
            f"近250日 {g('sharpe_recent', '.2f')}；"
            f"IC后段 {g('ic_fwd', '+.1f', 100)}%")


def _family_key(name: str) -> str:
    """命名族键：剥掉末尾连续数字，得到"同一公式的窗口网格"。

    为什么需要它：截面相关抓不住"窗口网格"这种重复——`alpha360_VOLUME36` 与
    `alpha360_VOLUME9` 相关只有 0.4~0.6，统计上算"两个因子"，构造上却是同一个
    字段的同一个公式。只靠相关阈值去重，全库会放进 43 个 active，其中 40 个是
    VOLUME 网格（实测），白名单等于没有。

    编号型命名（`alpha191_166`）不能剥：剥完是 ``alpha191_``，会把 180 个互不
    相同的 alpha191 公式并成一族。判据是"剥完以分隔符结尾"——那说明数字紧跟在
    分隔符后，是编号；窗口型命名数字紧跟字母（`VOLUME36`），才是参数。
    """
    m = re.match(r"^(.*?)(\d+)$", str(name))
    if not m:
        return str(name)
    stem = m.group(1)
    if not stem or stem.endswith(("_", "-", ".")):
        return str(name)
    return stem


def _window_stats(dret: pd.Series, ic: pd.Series, *, split_frac: float = 0.7,
                  recent_days: int = 250, min_days: int = 60) -> dict:
    """把一条日收益序列 + 逐日 IC 切成"前段 / 后段 / 近 250 日"三段统计。

    分段的用途是**时间稳健性**（不是严格 OOS）：因子在全样本上被挑出来，
    所以后段表现带有选择偏差；但"前段挣钱、后段反向"仍能筛掉一批靠单段行情
    撑起来的因子，而且不需要重跑回测——eval 序列本来就是逐日的。

    Args:
        dret: 净多空日收益（``eval`` 的 ``dret_ls_M``）。
        ic: 逐日 IC（同文件 ``ic``），NaN 表示当日无有效截面。
        split_frac: 前段占比（默认 0.7，即后段为最近 30%）。
        recent_days: "近期"窗口交易日数。
        min_days: 少于这么多样本不算（返回 NaN，宁可判"证据不足"也不给噪声数字）。

    Returns:
        ``{"sharpe_back", "sharpe_fwd", "sharpe_recent", "ic_back", "ic_fwd"}``。
        夏普与库内 ``sharpe_ls_M`` **同口径**（:func:`backtest.metrics.sharpe_ratio`
        的几何年化），否则分段数字与 registry 的全样本数字不可比。
    """
    out = {"sharpe_back": np.nan, "sharpe_fwd": np.nan, "sharpe_recent": np.nan,
           "ic_back": np.nan, "ic_fwd": np.nan}
    r = pd.Series(dret).dropna()
    if len(r) < min_days:
        return out
    k = int(len(r) * split_frac)
    for key, seg in (("sharpe_back", r.iloc[:k]), ("sharpe_fwd", r.iloc[k:]),
                     ("sharpe_recent", r.iloc[-int(recent_days):])):
        if len(seg) >= min_days and seg.std(ddof=1) > 0:
            out[key] = float(sharpe_ratio(seg))
    i = pd.Series(ic).dropna()
    if len(i) >= min_days:
        ki = int(len(i) * split_frac)
        out["ic_back"] = float(i.iloc[:ki].mean())
        out["ic_fwd"] = float(i.iloc[ki:].mean())
    return out


# --- 冗余索引：全库 pairwise 相关预计算（构建见 scripts/factors/dup_audit.py）---
_DUP_INDEX_CACHE: dict[str, dict] = {}

# --- 复权收盘：按 cache root 缓存（133 MB/份，进程内只读一次）---
# 放在模块级而不是实例级：测试里每个用例都新建 FactorLibrary，实例缓存等于没有，
# 77 个用例会重复读 20 次盘（实测把 30 s 的测试拖到 322 s）。
_ADJ_CACHE: dict[str, tuple] = {}

# --- 可交易掩码与 1 日收益面板：同样按 cache root 缓存 ---
# 掩码建一次 13 s（2851×5822），收益面板 ~2 s。二者都只取决于市场数据，
# 与库实例无关，没理由每个实例重算一遍。
_MASK_CACHE: dict[str, object] = {}
_RP_CACHE: dict[str, pd.DataFrame] = {}
_RP_BASIS: dict[str, str] = {}


def _row_rank_zscore(a: np.ndarray) -> np.ndarray:
    """(R, C) 原始值 → 逐行截面 rank 后 z-score，NaN 记 0。

    冗余审计与入库门禁共用这一份变换：只有两边口径完全一致，
    「入库时查到的相关」才等于「审计时入表的那个数」。
    """
    from scipy.stats import rankdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = rankdata(a, axis=1, nan_policy="omit")
        mu = np.nanmean(r, axis=1, keepdims=True)
        sd = np.nanstd(r, axis=1, keepdims=True)
        z = (r - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def _forward_returns(returns_panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """h 日远期收益：由**同一张** 1 日远期收益面板累乘得到。

    这样多周期 IC 与 1 日 IC 严格同源（同一收益面板、同一 NaN 模式），
    不会出现「h20 列来自复权收盘、ic_mean 列来自未复权收盘」的双真源。
    窗口内任一缺失即整段 NaN，与 ``close.pct_change(h)`` 的行为一致。
    """
    if horizon <= 1:
        return returns_panel
    g = np.log1p(returns_panel.replace([np.inf, -np.inf], np.nan))
    return np.expm1(g.rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1)))


def _ic_summary(ic: pd.Series) -> dict:
    """IC 序列 → registry 的检验统计列（均值/IR/显著性/胜率）。

    从 ``_panel_ic_stats`` 抽出，供两处复用：ingest（算完就写）与
    evaluate（回填老因子时只有 eval 文件里的 ic 列，不必重算面板）。
    """
    from stats.significance import mean_inference

    ok = ic.notna()
    ic_mean = float(ic.mean()) if ok.any() else float("nan")
    ic_std = float(ic.std()) if ok.any() else float("nan")
    ic_valid = ic.dropna()
    n = len(ic_valid)
    _inf = mean_inference(ic_valid) if n >= 2 else {
        "t_stat": 0.0, "p_value": float("nan"),
        "t_stat_nw": 0.0, "p_value_nw": float("nan"),
    }
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": calc_ir(ic) if n >= 2 else 0.0,
        "t_stat": _inf["t_stat"],
        "t_stat_nw": _inf["t_stat_nw"],
        "p_value_nw": _inf["p_value_nw"],
        "ic_win_rate": float((ic_valid > 0).mean()) if n else float("nan"),
        # |t_nw| > 2：ingest/evaluate 一次只处理一个因子，调用内不存在
        # 可言的"检验族"；整库校正走 significance_table(correction="fdr")。
        "significant": bool(abs(_inf["t_stat_nw"]) > 2.0),
    }


def _panel_ic_stats(panel: pd.DataFrame, returns_panel: pd.DataFrame, *,
                    ic_method: str = "spearman",
                    horizons: tuple[int, ...] | None = (1, 5, 10, 20),
                    evidence: bool = True) -> dict:
    """一次算齐登记所需的全部统计量（IC 序列 + 显著性 + 多周期 IC）。

    从 ``register`` 抽出（2026-11-02），供 :meth:`FactorLibrary.ingest` 调用；
    返回 dict 中 ``_ic`` 为 IC 序列本身（落 eval 文件用），其余为 registry 列。

    耗时分布（all_a，2487 日 × 5822 只，2026-11-02 实测）：
    IC 3.1 s、autocorr 3.2 s、ic_decay 30.4 s、monotonicity 13.2 s、
    pfs 47.4 s。后三者是**报告列**（不参与入库门槛），``evidence=False``
    时不计算、置 NaN——批量入库靠这个开关把 109 s 压到 7 s。
    """
    ic = calc_ic_series(panel, returns_panel, method=ic_method)
    summary = _ic_summary(ic)
    ic_mean = summary["ic_mean"]

    if evidence:
        try:
            with np.errstate(all="ignore"):
                _decay = calc_ic_decay(panel, returns_panel, max_lag=10)
            ic_decay5 = float(_decay.get(5, float("nan")))
        except Exception:  # noqa: BLE001
            ic_decay5 = float("nan")
    else:
        ic_decay5 = float("nan")
    try:
        autocorr = factor_autocorr(panel)
    except Exception:  # noqa: BLE001
        autocorr = float("nan")

    # 结构化证据（2026-09-22 evals 改造包）：只做报告列、**不参与入库门槛**
    # （门槛仍是 |t_nw|>2 + FDR）。三列合计约 91 s，evidence=False 时跳过。
    monotonicity = float("nan")
    pfs = float("nan")
    if evidence:
        try:
            from stats.ic import monotonicity_ratio
            monotonicity = monotonicity_ratio(panel, returns_panel, n_quantiles=5)
        except Exception:  # noqa: BLE001
            pass
        try:
            from stats.ic import perturbation_fidelity
            pfs = perturbation_fidelity(panel)
        except Exception:  # noqa: BLE001
            pass

    h_stats = {}
    for h in (horizons or ()):
        if h == 1:
            h_stats["ic_mean_h1"] = ic_mean
            continue
        try:
            h_ic = calc_ic_series(panel, _forward_returns(returns_panel, h),
                                  method=ic_method)
            h_stats[f"ic_mean_h{h}"] = float(h_ic.mean()) if h_ic.notna().any() else float("nan")
        except Exception:  # noqa: BLE001
            h_stats[f"ic_mean_h{h}"] = float("nan")

    return {
        "_ic": ic,
        **summary,
        "ic_decay5": ic_decay5,
        "autocorr": autocorr,
        "monotonicity": monotonicity,
        "pfs": pfs,
        **h_stats,
    }


class FactorLibrary:
    """因子持久化仓库。"""

    def __init__(self, root: str | Path | None = None, dataset: str | None = None):
        if root is None:
            # 多机适配：缺省根走 config 单一真源，**不硬编码盘符**
            # （两台机器数据盘不同：一台 D:/data、一台 E:/data）
            from config import Config
            root = Config.paths().get("factor_library_root") or ""
            if not str(root).strip():
                raise RuntimeError(
                    "factor_library.root 未配置：请检查 config/settings.yaml")
        self.base_root = Path(root)
        self.dataset = dataset
        # 按数据集分库根：dataset 给定时落到 base_root/<dataset>/，否则用 legacy 默认库（向后兼容）
        self.root = self.base_root / _sanitize_dataset(dataset) if dataset else self.base_root
        self.panels_dir = self.root / "panels"
        self.evals_dir = self.root / "evals"
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.root / "registry.csv"

    @classmethod
    def list_datasets(cls, root: str | Path | None = None) -> list[str]:
        """列出 base_root 下所有含 registry.csv 的数据集名（按数据集分库根）。"""
        if root is None:
            # 同上：缺省根走 config 单一真源，不硬编码盘符
            from config import Config
            root = Config.paths().get("factor_library_root") or ""
            if not str(root).strip():
                raise RuntimeError(
                    "factor_library.root 未配置：请检查 config/settings.yaml")
        base = Path(root)
        if not base.exists():
            return []
        return sorted(d.name for d in base.iterdir()
                      if d.is_dir() and (d / "registry.csv").exists())

    # ---- registry IO ----
    def _load_registry(self) -> pd.DataFrame:
        if self._registry_path.exists():
            reg = _align_registry(
                pd.read_csv(self._registry_path, dtype={"parents": str}))
            return _collapse_duplicate_names(reg, where=f"load:{self.dataset}")
        # schema 真源 = 模块级 CANONICAL_COLUMNS：空库也给完整列集，免得
        # "新建库少几列 → 第一次批量入库后又变成另一套"
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    def _save_registry(self, df: pd.DataFrame) -> None:
        # 写临时文件再原子替换，避免 read-modify-write 并发入库时损坏 CSV。
        # 列序统一按 CANONICAL_COLUMNS（未知列留尾部）——两条入库通道
        # （register / merge_outputs）写出的文件因此共享同一套 schema；
        # 编码统一 utf-8-sig，与历史 merge_outputs 产物一致（Excel 可直接读）。
        # 走 _align_registry 而非在此重算 head/tail：列对齐只有一处实现，
        # 免得两个函数对"缺列怎么办"理解不一致（2026-09-12 补列被丢的坑）。
        df = _align_registry(df)
        tmp = self._registry_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        # Windows 上 os.replace 会因"目标文件被别的句柄打开"（Excel/编辑器/
        # 杀软扫描/另一个进程正在读）抛 PermissionError[WinError 5]。这在
        # 2026-11-02 的 10 进程回填里真实打断过整批（957 个算到一半全丢）。
        # 重试 5 次、退避 0.2~3.2 s，足够熬过瞬时占用。
        for attempt in range(5):
            try:
                tmp.replace(self._registry_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (2 ** attempt))

    # ---- 注册 ----
    def ingest(
        self,
        name: str,
        panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        kind: str = "raw",
        formula: str | None = None,
        parents: list[str] | None = None,
        source: str = "",
        ic_method: str = "spearman",
        family: str = "",
        frequency: str = "",
        maturity: str = "experimental",
        note: str = "",
        horizons: tuple[int, ...] | None = (1, 5, 10, 20),
        evidence: bool = True,
        write_panel: bool = True,
        check_dup: bool = False,
        dup_corr: float = 0.7,
        reject_dup: bool = False,
        dup_include_builder_rows: bool = False,
        persist: bool = True,
        ret_basis: str = "close_pct_v1",
    ) -> dict:
        """因子登记的**便宜半段**：面板落盘 + IC 序列 + 多周期 IC + registry 行。

        2026-11-02 从 :meth:`register` 拆出。拆之前实测 all_a 规模
        （2487 日 × 5822 只）单因子 **131.4 s**，但耗时结构反直觉：

        =========================  =======  ============================
        环节                       耗时      性质
        =========================  =======  ============================
        IC 序列                       3.1 s   入门必需
        多周期 IC（h5/h10/h20）       9.3 s   可选（``horizons=None`` 关）
        ``autocorr``                  3.2 s   入门必需（换手代理）
        面板写盘                     ~2.0 s   入门必需
        ``ic_decay5``                30.4 s   报告列
        ``monotonicity``             13.2 s   报告列
        ``pfs``（扰动保真）           47.4 s   报告列
        三套 canonical 回测            3.3 s   入门必需（在 evaluate 里）
        =========================  =======  ============================

        也就是说**贵的不是回测，是三个明确"不参与入库门槛"的证据列**
        （见 :func:`_panel_ic_stats` 注释）。实测本方法：默认参数 17.5 s，
        ``evidence=False`` 同价，``evidence=False, horizons=None`` 约 8 s。
        要这三列时再单独补。

        评估状态写在 ``eval_status`` / ``evaluated_at`` 两列，**显式**表达，
        不再靠 ``sharpe_ls_M`` 是否为 NaN 去猜。

        Args:
            name: 因子唯一名。
            panel: date×code 因子面板（建议已截面标准化）。
            returns_panel: date×code **1 日**远期收益面板（``shift(-1)`` 口径）。
                多周期 IC 由本面板累乘推导，保证与 1 日 IC 严格同源。
            horizons: 需补算的多周期 IC；``None`` 或 ``()`` 表示只算 1 日。
            evidence: 是否计算 ``ic_decay5`` / ``monotonicity`` / ``pfs`` 三列
                （合计约 91 s，all_a 规模）。批量入库关掉，逐因子精读时开。
            write_panel: ``False`` 时不动已有面板文件，只更新 registry（回填用）。
            persist: ``False`` 时不写 registry，供 :meth:`register` 合并成一次写入。
            ret_basis: 收益口径标记，写入 ``ret_basis`` 列供追溯（当前全库均为
                ``close_pct_v1`` = 未复权收盘的 1 日收益）。
        Returns:
            该因子的 registry 行（dict）；canonical 指标列为空。
        """
        # 标签先过受控词表（越界只告警）：登记是唯一入口，入口不校验，
        # 后面每个消费者都得各自防漂移。
        family, frequency, maturity = normalize_tags(
            family=family, frequency=frequency, maturity=maturity,
            where=f"ingest({name})")
        slug = _slug(name)
        panel_path = self.panels_dir / f"{slug}.parquet"
        eval_path = self.evals_dir / f"{slug}.parquet"

        stats = _panel_ic_stats(panel, returns_panel, ic_method=ic_method,
                                horizons=horizons, evidence=evidence)
        ic = stats.pop("_ic")

        if write_panel:
            _write_parquet_robust(panel, panel_path)

        # IC 序列先单列落盘，evaluate 之后在保留该列的前提下补回测列。
        # 这样只 ingest 过的因子也能做时间段 IC 切片（evaluate_period 的部分功能）。
        _write_parquet_robust(
            pd.DataFrame({"ic": ic}).sort_index(), eval_path)

        dup_row = self._run_dup_check(name, panel, returns_panel, check_dup, dup_corr,
                                      reject_dup,
                                      include_builder_rows=dup_include_builder_rows)
        row = {
            "name": name,
            "kind": kind,
            "formula": formula if formula is not None else name,
            "source": source,
            "dataset": self.dataset or "",
            "parents": "|".join(parents) if parents else "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "family": family,
            "frequency": frequency,
            "maturity": maturity,
            "note": note,
            "n_dates": panel.shape[0],
            "n_codes": panel.shape[1],
            "eval_status": "ingested",
            "evaluated_at": "",
            "ret_basis": ret_basis,
            **stats,
            **dup_row,
            "panel_path": str(panel_path),
            "eval_path": str(eval_path),
        }
        if persist:
            self._upsert_row(row)
        log.info("因子登记(ingest): %s IC=%+.4f t_nw=%.2f best=待评估",
                 name, row["ic_mean"], row["t_stat_nw"])
        return row

    def evaluate(
        self,
        name: str,
        returns_panel: pd.DataFrame | None = None,
        *,
        panel: pd.DataFrame | None = None,
        short_costs: ShortCostModel | None = None,
        deleverage: bool = False,
        persist: bool = True,
        use_mask: bool = True,
        mask_lag: int = 0,
        recompute_ic: bool = False,
    ) -> dict:
        """补算 canonical 回测（**评估那半段**），回填 registry 指标。

        实测 3.3 s/因子（all_a 2487 日 × 5822 只）——比 :meth:`ingest` 便宜，
        所以"把库补成整齐的"在并行下是十分钟级动作，不是几十小时。

        与 :meth:`ingest` 配套：ingest 负责"有因子"，本方法负责"有数"。
        必须 ingest 过（registry 里有行），否则抛 ``KeyError``——刻意不隐式
        建行，避免出现"评估了但没登记"的第三态。

        Args:
            name: 已登记因子名。
            returns_panel: 1 日远期收益面板；省略时按缓存日线现算
                （:meth:`default_returns_panel`，``close.pct_change().shift(-1)``）。
            panel: 显式传入面板（``register`` 合并写入时用，避免读盘回环）。
            short_costs: 空头腿成本模型（默认沿用引擎默认：启用借券费）。
            deleverage: 1 倍资金约束。
            persist: ``False`` 时只返回指标、不写 registry。
            use_mask: 回测前把**不可交易**（停牌/收盘封板/ST）的格子因子值置空，
                即"选不到的票不参与排序"。2026-11-04 起默认开：不开的话，
                涨停封板当天买不进的那批票照样计入多空，头部因子会被系统性高估
                （实测 `alpha191_015` sharpe 2.75 → **−0.71**）。
                ⚠️ 只作用于**回测**，不改落盘面板、不改 IC —— IC 是预测力统计量，
                口径保留全截面。
            mask_lag: 掩码取样日，必须与成交日一致。库内收益是 ``shift(-1)``
                ⇒ T 日收盘建仓 ⇒ 默认 0。
            recompute_ic: 强制按当前收益面板重算 IC 序列（换口径迁移时用），
                否则沿用 eval 文件里已有的 ic 列。
        Returns:
            指标 dict（``*_ls_M`` / ``best_sharpe`` / ``evaluated_at`` …）。
        """
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            raise KeyError(f"未登记因子 {name}：先 ingest() 再 evaluate()")
        if panel is None:
            panel = self.get_panel(name)
        if panel is None:
            raise FileNotFoundError(f"因子面板缺失：{name}（registry 的 panel_path 无效）")
        rp = returns_panel if returns_panel is not None else self.default_returns_panel()
        # 收益口径只如实登记，不猜：调用方自带的面板无从判断是哪一种；
        # 库内面板由 default_returns_panel() 自己记在实例上（含缓存命中路径）。
        ret_basis = ("caller_supplied" if returns_panel is not None
                     else getattr(self, "_ret_basis", "close_pct_v1"))

        # 回测用可交易面板；掩码不可用（缺状态表/复权文件）时退化为全可交易，
        # 但 mask_basis 会如实标成 ""，不让"没套上"看起来像"套上了"。
        #
        # 股票池重叠检查：掩码来自**真实市场**日线，若因子面板的股票池与它不
        # 重叠（mock 库、单元测试的合成代码），套上去只会全 NaN→全 True，白花
        # 13 秒建表。重叠不足一半就不套。
        eval_panel, mask_basis = panel, ""
        if use_mask:
            adj = self._adjusted_close()
            if adj is not None:
                ov = panel.columns.intersection(adj.columns)
                if len(ov) >= 0.5 * max(1, len(panel.columns)):
                    mask = self.tradable_mask(mask_lag)
                    if mask is not None:
                        mm = mask.reindex(index=panel.index, columns=panel.columns)
                        eval_panel = panel.where(mm.fillna(True))
                        mask_basis = f"tradable_t{mask_lag}"

        eval_cols, metric_rows, best_sharpe, best_config = self._canonical_eval(
            eval_panel, rp, short_costs=short_costs, deleverage=deleverage)

        eval_path = self.evals_dir / f"{_slug(name)}.parquet"
        # 保留 ingest 阶段落的 ic 列；回填的老因子没有 ic 列，现算一次补上
        # （3 s 级），保证「eval 文件一定有 ic」这个不变量对两条路径都成立。
        # ``recompute_ic`` 用于口径迁移：换收益面板后旧 ic 已经不是同一个数。
        ic_series = None
        if not recompute_ic and eval_path.exists():
            try:
                old = pd.read_parquet(eval_path)
                if "ic" in old.columns:
                    ic_series = old["ic"]
            except Exception:  # noqa: BLE001
                log.warning("旧 eval 读取失败，按全新写入：%s", eval_path)
        if ic_series is None:
            try:
                ic_series = calc_ic_series(panel, rp)
            except Exception:  # noqa: BLE001
                log.warning("IC 序列重算失败，eval 将缺 ic 列：%s", name)
        if ic_series is not None:
            eval_cols = {"ic": ic_series, **eval_cols}
        _write_parquet_robust(pd.DataFrame(eval_cols).sort_index(), eval_path)

        # 顺手把 IC 检验统计写回 registry：老因子（builder 通道）的
        # ic_mean / t_stat_nw 一直是空的，只有 ic_mean_h*，导致"按显著性
        # 排序"在半个库上失效。ic 序列就在手上，重算代价可忽略。
        ic_stats = _ic_summary(ic_series) if ic_series is not None else {}

        upd = {
            **ic_stats,
            **metric_rows,
            "best_sharpe": best_sharpe,
            "best_config": best_config,
            "eval_path": str(eval_path),
            "eval_status": "evaluated",
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # 口径标记：老行没有这两列，一眼能看出哪些数是在哪个口径下算的
            # （2026-11-04 之前 = 未复权 + 无掩码）。
            "ret_basis": ret_basis,
            "mask_basis": mask_basis,
            # 回填场景下 registry 的 n_dates/n_codes 往往从没写过（builder 通道
            # 只写 ic 统计），评估时顺手用真实面板尺寸补齐，避免体检永远报空。
            "n_dates": panel.shape[0],
            "n_codes": panel.shape[1],
        }
        if persist:
            self.apply_eval(name, upd)
        log.info("因子评估(evaluate): %s best_sharpe=%.3f @%s",
                 name, best_sharpe, best_config)
        return upd

    def apply_eval(self, name: str, metrics: dict) -> None:
        """把 :meth:`evaluate` 的结果写回 registry（单条）。"""
        self.apply_eval_many({name: metrics})

    def apply_eval_many(self, mapping: dict[str, dict]) -> int:
        """批量把 :meth:`evaluate` 的结果写回 registry，**只落盘一次**。

        回填脚本的唯一写盘点：worker 只算（``evaluate(persist=False)``）、
        父进程汇总后调本方法，避免多进程同时写 ``registry.csv`` 互相截断。
        """
        if not mapping:
            return 0
        reg = self._load_registry()
        for name, metrics in mapping.items():
            m = reg["name"] == name
            if not m.any():
                raise KeyError(f"未登记因子 {name}")
            for k, v in metrics.items():
                if k not in reg.columns:
                    reg[k] = pd.NA
                # 全 NaN 的数值列（例如还没有任何 evaluate 过）默认是 float64，
                # 直接塞 str/bool 会抛 LossySetitemError；先升到 object 再写。
                if isinstance(v, (str, bool)) and reg[k].dtype != object:
                    reg[k] = reg[k].astype(object)
                reg.loc[m, k] = v
        self._save_registry(reg)
        return len(mapping)

    def _upsert_row(self, row: dict) -> None:
        reg = self._load_registry()
        reg = reg[reg["name"] != row["name"]]  # 覆盖同名
        new_df = pd.DataFrame([row])
        reg = new_df if reg.empty else pd.concat([reg, new_df], ignore_index=True)
        self._save_registry(reg)

    def _canonical_eval(self, panel, returns_panel, *, short_costs=None,
                        deleverage=False):
        """跑三套 canonical 配置，返回 (eval 列, 指标列, 最优 sharpe, 最优配置)。"""
        eval_cols: dict = {}
        metric_rows: dict = {}
        best_sharpe = -np.inf
        best_config = CANONICAL_CONFIGS[0].key
        for cfg in CANONICAL_CONFIGS:
            bt = VectorBacktest(cfg.strategy(k=cfg.k), rebalance_freq=cfg.freq,
                                short_costs=short_costs, deleverage=deleverage)
            # 库内收益口径为 shift(-1)（第 i 行 = i→i+1 收益，调仓日 t 权重赚
            # rp[t]=t→t+1，无前视）；canonical 回测只算绝对指标、不对齐指数
            # 算 beta/IR，故显式声明关闭引擎的 shift 指纹守卫。
            res = bt.run(panel, returns_panel, check_convention=False)
            dret = res.daily_returns
            eval_cols[f"dret_{cfg.key}"] = dret
            eval_cols[f"equity_{cfg.key}"] = res.equity_curve
            m = res.metrics()
            for mc in _METRIC_COLS:
                if mc in m:
                    metric_rows[f"{mc}_{cfg.key}"] = m[mc]
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_config = cfg.key
        return eval_cols, metric_rows, best_sharpe, best_config

    def _adjusted_close(self) -> pd.DataFrame | None:
        """复权收盘面板（未复权收盘 × 累积后复权因子）；缺因子文件时 None。

        2026-11-04 起作为库内收益口径。修之前用的是**未复权**收盘，除权日会产生
        假跌幅（拆股当天 −90% 这类）。实测影响很小但不是零：2018+ 全样本 1234 万
        格里 |Δ|>0.5pp 占 0.195%、均 |Δ| 1.6bp，对 6 个因子的 IC 影响 ≤0.0004
        ——是**卫生问题**，不是结论问题，但没有理由保留。

        同时缓存后复权因子面板（``self._bwd_cache``），可交易掩码要用它把复权价
        还原成未复权价才能跟涨跌停价比较。
        """
        if getattr(self, "_adj_ready", False):
            return self._adj_cache
        from config import Config

        root = Path(str(Config.cache()["root"]))
        ck = str(root)
        if ck in _ADJ_CACHE:
            adj, bwd = _ADJ_CACHE[ck]
            self._adj_cache, self._bwd_cache = adj, bwd
            self._adj_ready = True
            return adj
        bwd_f = root / "backward_factor.parquet"
        if not bwd_f.exists():
            log.warning("缺 backward_factor.parquet，收益回退未复权收盘")
            self._adj_cache = None
            self._bwd_cache = None
            self._adj_ready = True
            return None
        try:
            raw = pd.read_parquet(root / "daily_all_a.parquet", columns=["close"])["close"]
            raw = raw.unstack()
            raw.index = pd.to_datetime(raw.index).normalize()
            bwd = pd.read_parquet(bwd_f)
        except Exception as exc:  # noqa: BLE001
            log.warning("复权数据读取失败，收益回退未复权收盘：%s", exc)
            self._adj_cache = None
            self._bwd_cache = None
            self._adj_ready = True
            return None
        cols = raw.columns.intersection(bwd.columns)
        if len(cols) < 100:
            log.warning("复权因子与日线股票交集过小（%d），回退未复权", len(cols))
            self._adj_cache = None
            self._bwd_cache = None
            self._adj_ready = True
            return None
        bwd = bwd.reindex(index=raw.index, columns=cols).ffill()
        adj = raw[cols] * bwd
        self._adj_cache = adj
        self._bwd_cache = bwd
        self._adj_ready = True
        _ADJ_CACHE[ck] = (adj, bwd)
        return adj

    def tradable_mask(self, execution_lag: int = 0) -> pd.DataFrame | None:
        """可交易性掩码（bool 宽表，True=该日可进入组合）；不可用返回 None。

        ``execution_lag`` 必须与**成交日**一致：库内收益面板是
        ``close.pct_change().shift(-1)``，第 T 行 = T→T+1 收益 ⇒ 建仓发生在
        **T 日收盘**，故取 ``execution_lag=0``（T 日状态）。

        ``data/tradability.py`` 的 🚨 段记过这个坑：掩码取样日与成交日错配时，
        过剔与漏剔方向相反、净效应在月频是噪声、周频 4/4 一致为正（+3.76pp）。
        默认值从 1 改成 0 是**为了与库内收益口径自洽**，不是随手改的。

        实例内按 lag 缓存（建一次约 16 s / 2851×5822）。
        """
        adj = self._adjusted_close()
        if adj is None:
            return None
        from config import Config

        key = f"{Config.cache()['root']}|{execution_lag}"
        if key in _MASK_CACHE:
            return _MASK_CACHE[key]
        try:
            from data.tradability import build_tradable_mask

            mask = build_tradable_mask(adj, bwd=getattr(self, "_bwd_cache", None),
                                       execution_lag=execution_lag)
        except Exception as exc:  # noqa: BLE001
            log.warning("可交易掩码构建失败，按全可交易处理：%s", exc)
            mask = None
        _MASK_CACHE[key] = mask
        return mask

    def default_returns_panel(self) -> pd.DataFrame:
        """缓存日线现算的 1 日远期收益面板（``close.pct_change().shift(-1)``）。

        2026-11-04 起用**复权**收盘（``_adjusted_close``），缺复权文件时回退
        未复权并在日志里告警。据此保证库内所有因子的 IC 与 canonical 回测
        用的是同一张收益面板。

        实例内缓存，回填脚本每个 worker 只读一次盘。
        """
        cached = getattr(self, "_rp_cache", None)
        if cached is not None:
            return cached
        from config import Config

        rk = str(Config.cache()["root"])
        if rk in _RP_CACHE:
            self._rp_cache = _RP_CACHE[rk]
            # 口径标记必须跟着面板走：缓存命中时不经过 _adjusted_close()，
            # 靠 self._adj_cache 判断会误报成未复权（2026-11-04 实测踩到，
            # 表现为"每个复用进程的第一个因子"口径标错）。
            self._ret_basis = _RP_BASIS.get(rk, "close_pct_v1")
            return self._rp_cache
        adj = self._adjusted_close()
        if adj is not None:
            self._rp_cache = adj.pct_change(fill_method=None).shift(-1)
            _RP_CACHE[rk] = self._rp_cache
            _RP_BASIS[rk] = "adj_close_pct_v1"
            self._ret_basis = "adj_close_pct_v1"
            return self._rp_cache
        from config import Config

        root = Path(str(Config.cache()["root"]))
        close = pd.read_parquet(root / "daily_all_a.parquet", columns=["close"])
        wide = close["close"].unstack()
        wide.index = pd.to_datetime(wide.index).normalize()
        self._rp_cache = wide.pct_change(fill_method=None).shift(-1)
        self._ret_basis = "close_pct_v1"
        return self._rp_cache

    def contract_violations(self) -> list[dict]:
        """因子库**契约**检查：把"什么算一个合格的库"变成可执行断言。

        三条规则（2026-11-02 与两段式入库同时建立）：

        1. ``maturity ∈ {oos_verified, active}`` 的因子必须真的评估过：
           ``evaluated_at`` 非空、eval 文件存在、``t_stat_nw`` 有限、
           ``dup_checked=True``（正式因子不许没查过冗余）。
        2. ``eval_status == "evaluated"`` ⇒ eval 文件在盘上且 ``sharpe_ls_M``
           非空（防止"标了已评估但指标是 NaN"的半写态）。
        3. ``eval_status == "ingested"`` ⇒ canonical 指标必须全为空
           （防止只登记却残留旧回测数字，造成"看起来评估过"）。

        Returns:
            违规清单 ``[{"name", "rule", "detail"}, ...]``；空列表 = 合规。
        """
        reg = self._load_registry()
        bad: list[dict] = []
        metric_cols = [f"{m}_{c.key}" for c in CANONICAL_CONFIGS for m in _METRIC_COLS]
        for _, r in reg.iterrows():
            name = r["name"]
            status = str(r.get("eval_status") or "").strip()
            evaluated = str(r.get("evaluated_at") or "").strip()
            evaluated = evaluated not in ("", "nan", "None")

            if evaluated:
                p = self._resolve_eval_path(r) or (self.evals_dir / f"{_slug(name)}.parquet")
                if not Path(p).exists():
                    bad.append({"name": name, "rule": "eval_file_missing",
                                "detail": str(p)})
            if status == "evaluated":
                if not _finite(r.get("sharpe_ls_M")):
                    bad.append({"name": name, "rule": "evaluated_but_nan",
                                "detail": "sharpe_ls_M 为空"})
            elif status == "ingested":
                leaked = [c for c in metric_cols if _finite(r.get(c))]
                if leaked:
                    bad.append({"name": name, "rule": "ingested_with_metrics",
                                "detail": ",".join(leaked[:3])})

            mat = str(r.get("maturity") or "").strip()
            if mat in ("oos_verified", "active"):
                if not evaluated:
                    bad.append({"name": name, "rule": "formal_without_eval",
                                "detail": f"maturity={mat}"})
                if not _finite(r.get("t_stat_nw")):
                    bad.append({"name": name, "rule": "formal_without_t",
                                "detail": f"maturity={mat}"})
                if r.get("dup_checked") is not True and str(r.get("dup_checked")) != "True":
                    bad.append({"name": name, "rule": "formal_without_dup_check",
                                "detail": f"maturity={mat}"})
        return bad

    # ---- 成熟度评定（晋升 / 降级） ----
    def _eval_series(self, row) -> tuple[pd.Series, pd.Series] | None:
        """读某因子的 ``(日收益, 逐日 IC)`` 序列；缺 eval 文件返回 ``None``。

        两列都可能含 NaN（IC 在无截面日为空、收益在末日为空），调用方自己 dropna。
        """
        name = str(row.get("name") or "")
        p = self._resolve_eval_path(row)
        if p is None:
            p = self.evals_dir / f"{_slug(name)}.parquet"
        if not Path(p).exists():
            return None
        try:
            ev = pd.read_parquet(p)
        except Exception as e:                       # 坏文件不该拖垮整批评定
            log.warning("[%s] eval 读取失败 %s: %s", name, p, e)
            return None
        if "dret_ls_M" not in ev.columns:
            return None
        ic = ev["ic"] if "ic" in ev.columns else pd.Series(np.nan, index=ev.index)
        return ev["dret_ls_M"], ic

    def assess_maturity(
        self,
        *,
        names: list[str] | None = None,
        corr: np.ndarray | None = None,
        corr_names: np.ndarray | list[str] | None = None,
        demote: bool = False,
        gates: dict | None = None,
        stamp: str | None = None,
    ) -> pd.DataFrame:
        """按 :data:`MATURITY_GATES` 评定每个因子的成熟度，返回**含证据的判定表**。

        阶梯（每一档都是"全满足才进"，不是打分）：

        1. ``invalid`` —— 面板没有可排名的截面信息（由 ``dup_audit`` 判，本方法不改）。
        2. ``experimental`` —— 默认档；入库但证据不全或未达线。
        3. ``oos_verified`` —— 六条硬门槛全过：
           证据存在、口径正确（复权 + 掩码）、查过冗余、净多空夏普达标、
           相对最近邻有增量、前后两段同号。
        4. ``active`` —— 在 ``oos_verified`` 上再加"队列唯一性"：按夏普降序贪心选取，
           与已选 ``active`` 相关 ≥ ``max_active_corr`` 的拒（``crowded``），
           同族代表已达 ``max_per_family`` 个的拒（``family_cap``）。
           **必须给 ``corr``/``corr_names``**——证不了独立性就不给 active，
           没有相关矩阵时本方法的上限是 ``oos_verified``。
        5. ``retired`` —— 只在 ``demote=True`` 时产生，且**仅对"曾经达标、现在失效"**
           （``unstable`` / ``fading``）。若是证据缺失（没 eval、旧口径、没查冗余）
           则降回 ``experimental``：那不是因子失效，是我们自己的账没记清。

        Args:
            names: 只评定这些因子（默认全库）。
            corr: 截面相关矩阵（``dup_matrix.npz`` 的 ``corr``），用于队列去重。
            corr_names: ``corr`` 的行列名，顺序须与矩阵一致。
            demote: 允许把已达标的因子降级（默认只升不降，便于对照）。
            gates: 覆盖 :data:`MATURITY_GATES` 里的阈值（测试/敏感性分析用）。
            stamp: 写进 ``note`` 的日期戳（默认今天）。

        Returns:
            行索引为 ``name``，列含 ``current`` / ``verdict`` / ``reasons`` /
            ``sharpe`` / ``sharpe_back`` / ``sharpe_fwd`` / ``sharpe_recent`` /
            ``ic_back`` / ``ic_fwd`` / ``dup_corr_max`` / ``resid_ic`` / ``queue``。
            ``queue`` 为 1 表示成为 ``active`` 队列成员。
        """
        g = dict(MATURITY_GATES)
        if gates:
            g.update(gates)
        stamp = stamp or datetime.now().strftime("%Y-%m-%d")
        reg = self._load_registry()
        if names is not None:
            keep = set(names)
            reg = reg[reg["name"].isin(keep)]

        # 相关矩阵的 name → 行号（缺矩阵时留空，active 一律不给）
        cidx: dict[str, int] = {}
        if corr is not None and corr_names is not None:
            cidx = {str(n): i for i, n in enumerate(list(corr_names))}

        rows: list[dict] = []
        for _, r in reg.iterrows():
            name = str(r["name"])
            # ⚠️ 不能写 ``str(r.get("maturity") or "")``：NaN 是 truthy，空档会变成
            # 字符串 "nan"，后续 ``cur in ("oos_verified", ...)`` 与报告全都跟着错。
            _m = r.get("maturity")
            cur = "" if self._is_blank(_m) or str(_m).strip() in ("nan", "None") \
                else str(_m).strip()
            rec = {
                "name": name, "current": cur, "family": _family_key(name),
                "sharpe": _f(r.get("sharpe_ls_M")),
                "dup_corr_max": _f(r.get("dup_corr_max")),
                "resid_ic": _f(r.get("resid_ic")),
                "sharpe_back": np.nan, "sharpe_fwd": np.nan,
                "sharpe_recent": np.nan, "ic_back": np.nan, "ic_fwd": np.nan,
            }
            fails: list[str] = []

            # —— 门槛 1：证据存在
            series = self._eval_series(r)
            if (str(r.get("eval_status") or "").strip() != "evaluated"
                    or series is None or not np.isfinite(rec["sharpe"])):
                fails.append("no_eval")
            # —— 门槛 2：口径（旧口径的数字不许晋级：未复权 + 无掩码会把
            #     "买涨停"当成 alpha，2026-11-04 实测过 2.86 → −0.71）
            if (str(r.get("ret_basis") or "").strip() != "adj_close_pct_v1"
                    or not str(r.get("mask_basis") or "").strip()):
                fails.append("stale_caliber")
            # —— 门槛 3：查过冗余（死因子 dup_checked=False，别拿 True 掩盖）
            if not (r.get("dup_checked") is True or str(r.get("dup_checked")) == "True"):
                fails.append("dup_unknown")
            # —— 门槛 4：绝对业绩
            if not (np.isfinite(rec["sharpe"]) and rec["sharpe"] >= g["min_sharpe"]):
                fails.append("low_sharpe")
            # —— 门槛 5：相对最近邻有增量（高相关 + 残差无信息才是重复）
            if (np.isfinite(rec["dup_corr_max"]) and rec["dup_corr_max"] >= g["max_dup_corr"]
                    and not (np.isfinite(rec["resid_ic"]) and rec["resid_ic"] >= g["min_resid_ic"])):
                fails.append("redundant")

            # —— 门槛 6：时间稳健（前 70% / 后 30% 分段，样本外近似）
            if series is not None:
                st = _window_stats(series[0], series[1],
                                   split_frac=g["split_frac"],
                                   recent_days=int(g["recent_days"]),
                                   min_days=int(g["min_seg_days"]))
                rec.update({k: st[k] for k in
                            ("sharpe_back", "sharpe_fwd", "sharpe_recent",
                             "ic_back", "ic_fwd")})
                if fails == []:      # 前面已废的不用重复算判据
                    bad_seg = (np.isfinite(rec["ic_fwd"]) and np.isfinite(rec["ic_back"])
                               and np.sign(rec["ic_fwd"]) != np.sign(rec["ic_back"]))
                    if bad_seg or not (np.isfinite(rec["sharpe_fwd"]) and rec["sharpe_fwd"] > 0):
                        fails.append("unstable")
                # —— 门槛 7：没在衰（近 250 日）
                if fails == [] and not (np.isfinite(rec["sharpe_recent"])
                                        and rec["sharpe_recent"] >= g["min_recent_sharpe"]):
                    fails.append("fading")

            rec["fails"] = tuple(fails)
            rec["queue"] = 0
            rows.append(rec)

        out = pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame()
        if out.empty:
            out["reasons"] = pd.Series(dtype=str)
            out["verdict"] = pd.Series(dtype=str)
            return out

        # —— 队列去重：只在"前七条全过"的候选里，按夏普降序贪心。
        #    两道闸：① 与已选 active 相关 ≥ max_active_corr → crowded；
        #    ② 同族代表 ≥ max_per_family → family_cap（相关抓不住窗口网格，见 _family_key）。
        cand = out[out["fails"].str.len() == 0].sort_values("sharpe", ascending=False)
        picked: list[str] = []
        fam_n: dict[str, int] = {}
        for name in cand.index:
            fk = str(out.at[name, "family"])
            if not cidx or name not in cidx:
                out.at[name, "fails"] = ("uncovered",)
                out.at[name, "queue"] = -1        # −1 = 无法验证独立性
                continue
            i = cidx[name]
            if any(abs(float(corr[i, cidx[p]])) >= g["max_active_corr"] for p in picked):
                out.at[name, "fails"] = ("crowded",)
                continue
            if fam_n.get(fk, 0) >= int(g["max_per_family"]):
                out.at[name, "fails"] = ("family_cap",)
                continue
            picked.append(name)
            fam_n[fk] = fam_n.get(fk, 0) + 1
            out.at[name, "queue"] = 1

        # —— 定档
        verdicts, reasons = [], []
        for name, rec in out.iterrows():
            f = rec["fails"]
            cur = rec["current"]
            if cur == "invalid":
                verdicts.append("invalid")
                reasons.append("面板无可排名信息（人工复核，不由本方法改判）")
                continue
            if not f:
                verdicts.append("active" if rec["queue"] == 1 else "oos_verified")
                reasons.append("")
                continue
            # 队列类失败（crowded / family_cap / uncovered）只挡 active，不否定个体证据
            hard = [k for k in f if k not in ("crowded", "family_cap", "uncovered")]
            if not hard:
                verdicts.append("oos_verified")
                reasons.append("；".join(MATURITY_REASONS.get(k, k) for k in f))
                continue
            # 降级判定：只有"失效"才 retired，"账没记清"退回 experimental
            if demote and cur in ("oos_verified", "active") and (
                    {"unstable", "fading"} & set(f)):
                verdicts.append("retired")
            else:
                verdicts.append("experimental")
            reasons.append("；".join(MATURITY_REASONS.get(k, k) for k in f))
        out["verdict"] = verdicts
        out["reasons"] = reasons
        out["stamp"] = stamp
        return out

    def apply_maturity(self, assess: pd.DataFrame, *, note: bool = True,
                       only_change: bool = True) -> int:
        """把 :meth:`assess_maturity` 的判定写回 registry，**一次落盘**。

        ``only_change=True`` 只写"判定变了"或"``note`` 里还没有本次证据标签"的行
        （外加 maturity 为空的补实，空值既不属于任何档、在过滤时等于永久隐身，
        比 ``experimental`` 更糟）。判定没变时不写 note，是为了幂等——否则每跑一次
        note 就长一截。
        ``note`` 为真时追加标签，保留原有内容不覆盖。
        """
        if assess.empty:
            return 0
        reg = self._load_registry().set_index("name")
        payload: dict[str, dict] = {}
        for name, r in assess.iterrows():
            if name not in reg.index:
                continue
            cur = str(reg.at[name, "maturity"] or "").strip()
            new = str(r["verdict"])
            blank = cur in ("", "nan", "None")
            same = (not blank) and cur == new
            old = "" if pd.isna(reg.at[name, "note"]) else str(reg.at[name, "note"])
            # 不达标的写失败原因；达标的写证据摘要——否则 registry 里 active 的
            # note 全空，回头审计只能翻报告，没法在表里自证。
            body = (str(r.get("reasons") or "").strip() or _evidence_brief(r)) if note else ""
            tag = f"[{r['stamp']} {new}] {body}" if body else ""
            if only_change and same and (not tag or tag in old):
                continue
            upd: dict = {"maturity": new}
            if tag and tag not in old:
                upd["note"] = (old + " | " + tag).strip(" |")
            payload[name] = upd
        if payload:
            # 逐条写：apply_eval_many 已处理"数值列改写字符串"的 dtype 问题
            self.apply_eval_many(payload)
        return len(payload)

    def register(
        self,
        name: str,
        panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        kind: str = "raw",
        formula: str | None = None,
        parents: list[str] | None = None,
        source: str = "",
        ic_method: str = "spearman",
        short_costs: ShortCostModel | None = None,
        deleverage: bool = False,
        family: str = "",
        frequency: str = "",
        maturity: str = "experimental",
        note: str = "",
        check_dup: bool = False,
        dup_corr: float = 0.7,
        reject_dup: bool = False,
        dup_include_builder_rows: bool = False,
    ) -> dict:
        """完整登记 = :meth:`ingest` + :meth:`evaluate`（**兼容入口**，行为不变）。

        保留它是因为有 900+ 处调用点，且"我要一个能跑回测的因子"仍是最常见
        诉求。只想入库、暂不评估的批量场景走 :meth:`ingest`。

        参数含义见两个被调方法；``ic_method`` 只影响 ingest，``short_costs``
        与 ``deleverage`` 只影响 evaluate。
        """
        row = self.ingest(
            name, panel, returns_panel, kind=kind, formula=formula, parents=parents,
            source=source, ic_method=ic_method, family=family, frequency=frequency,
            maturity=maturity, note=note, check_dup=check_dup, dup_corr=dup_corr,
            reject_dup=reject_dup,
            dup_include_builder_rows=dup_include_builder_rows, persist=True)
        upd = self.evaluate(name, returns_panel, panel=panel,
                            short_costs=short_costs, deleverage=deleverage,
                            persist=True)
        row.update(upd)
        self._upsert_row(row)
        log.info("因子入库: %s (kind=%s, IC=%+.4f, best_sharpe=%.3f @%s)",
                 name, kind, row["ic_mean"], row["best_sharpe"], row["best_config"])
        return row

    def upsert_rows(self, rows: list[dict], *, source: str = "", family: str = "",
                    kind: str = "raw", maturity: str = "experimental",
                    frequency: str = "", set_name: str = "",
                    fill_missing_only: bool = True) -> dict:
        """批量**轻量登记**：把已有统计行并入 registry，不重算任何指标。

        这是 builder 通道（``scripts/builders/common.py`` 的 ``merge_outputs``）
        的唯一写入口——替代此前"``pd.read_csv`` → concat → ``to_csv``"的直写。
        直写绕过库入口的后果在 2026-09-12 暴露：all_a 的 912 个因子
        ``source`` / ``family`` / ``kind`` 全空，库 API 按这些字段筛选时
        完全看不到它们（``select_stocks --all-raw``、``list_all(kind=)``、
        ``extend_factor_library`` 的因子集路由全部漏掉这一批）。

        与 :meth:`register` 的分工：``register`` 是完整登记（IC 序列 + 3 组
        canonical 回测 + panel/eval 双落盘，实测约 53.7 s/因子）；本方法是
        轻量登记，**只写 registry 行**，用于几百个候选因子的批量入库。

        Args:
            rows: 逐因子统计行（至少含 ``name``）。
            source: 本批来源标注，形如 ``alpha101:builders:all_a_2018_2026``
                （用 :func:`source_for` 生成，前缀必须是因子集名——见该函数
                关于路由键的说明）。
            family: 本批因子族（受控词表，见 :data:`FAMILY_WHITELIST`）。
            kind: raw / composite。
            maturity: experimental / oos_verified / active / retired。
            frequency: 日频 / 周频 / 月频 / 日内。
            set_name: 因子集名（alpha101/evt/moneyflow…），写入 ``set`` 列。
            fill_missing_only: True（默认）时**只有** :data:`STAT_COLUMNS`
                （计算型统计）允许用新值覆盖旧值，其余列只补空缺——builder
                重跑不该把人工补过的标签冲掉。False 时本批带来的所有列都以
                新值优先（:data:`TAG_COLUMNS` 仍旧值优先）。**两种模式都是
                "非空才写"**：空值永远不会清空已有字段。

        Returns:
            dict: ``{"before": n, "after": m, "added": a, "updated": u}``。
        """
        if not rows:
            return {"before": 0, "after": 0, "added": 0, "updated": 0}
        family, frequency, maturity = normalize_tags(
            family=family, frequency=frequency, maturity=maturity,
            where="upsert_rows")
        source = (source or "").strip()
        if source:
            prefix = source.split(":")[0]
            if prefix not in SOURCE_PREFIXES:
                log.warning("upsert_rows: source 前缀 %r 不在 SOURCE_PREFIXES: %s",
                            prefix, source)
        else:
            log.warning("upsert_rows: source 为空——extend_factor_library 用 "
                        "source.split(':')[0] 路由因子集，空来源等于这批因子进不了"
                        "「延长数据集」；请在调用处显式传入")

        new_df = pd.DataFrame(rows)
        if "name" not in new_df.columns:
            raise ValueError("upsert_rows: rows 缺少 name 列")
        # 🚨 批内同名去重（2026-10-27）。rows 多来自追加式 jsonl（见
        # ``scripts.builders.common.read_stats``）：同一因子重跑几次就有几行。
        # 下面的 ``new_only`` 分支只排除"库里已有的"，**不排除批内重复** ⇒ 首次
        # 入库就把 2~3 行同名原样写进 registry（实测 all_a 1112 行 / 950 唯一，
        # ``added`` 也虚报）。这里按 keep="last" 收敛（追加顺序 = 运行顺序）。
        if new_df["name"].duplicated().any():
            n_dup = int(new_df["name"].duplicated().sum())
            log.warning("upsert_rows: 批内 %d 行同名重复，按最后一次统计收敛"
                        "（来源 jsonl 追加写，见 read_stats）", n_dup)
            new_df = new_df.drop_duplicates(subset="name", keep="last")

        batch = {"source": source, "family": family, "kind": kind,
                 "maturity": maturity, "frequency": frequency, "set": set_name}
        for col, val in batch.items():
            if not val:
                continue
            if col not in new_df.columns:
                new_df[col] = val
            else:
                new_df[col] = new_df[col].astype(object)
                blank = new_df[col].isna() | (new_df[col].astype(str).str.strip() == "")
                new_df.loc[blank, col] = val

        reg = self._load_registry()
        before = len(reg)
        known = set(reg["name"])
        n_add = int((~new_df["name"].isin(known)).sum())
        dup = reg["name"].isin(new_df["name"])
        keep = reg[~dup]
        n_refresh = 0
        if dup.any():
            # 已有行：**按列分组合并**（2026-10-27 重写，见 STAT_COLUMNS）。
            #   - 统计列：非空新值胜出 —— 重跑重算过就必须刷新。
            #   - 标签列：fill_missing_only=True 时旧值优先（保住人工标签），
            #     False 时非空新值胜出（builder 明确要求整批覆盖）。
            # 两种模式都是「非空才写」：空值不再清空任何已有字段。此前
            # False 分支走 concat 整行替换，stats 行没有的列（source /
            # formula / created_at…）会被洗成 NaN，这正是 holder_dyn 只敢用
            # oneoff 脚本外科回填的原因。
            old = (reg[dup].set_index("name")
                   .replace(r"^\s*$", np.nan, regex=True))
            # 空串一律视同缺失：否则 ``stats`` 里的 "" 会盖掉旧值（写回 CSV
            # 又变成空，等于把统计抹掉一格），且刷新计数会把 ""↔NaN 当成变化
            add = (new_df.set_index("name").reindex(old.index)
                   .replace(r"^\s*$", np.nan, regex=True))
            # 旧值优先打底（= 只补空缺）
            upd = old.combine_first(add)
            refresh = [c for c in add.columns if c in STAT_COLUMNS]
            if not fill_missing_only:
                # 显式覆盖：本批带了的列全部以新值优先（仍不动空值）。
                # 但 :data:`TAG_COLUMNS`（set_tag 可人工改的列）始终旧值优先——
                # builder 重跑不能把人工分类冲掉。
                refresh = [c for c in add.columns if c not in TAG_COLUMNS]
            if refresh:
                new_wins = add[refresh].combine_first(old.reindex(columns=refresh))
                # 计数：真正发生变化的单元格（浮点容差，避免 repr 抖动误计）
                n_refresh = int(_count_changed(old.reindex(columns=refresh),
                                               new_wins))
                upd[refresh] = new_wins
            upd = upd.reset_index()
            # 2026-09-22 修：混合批次（新+已有）时 upd 只覆盖交集，
            # 纯新增行会被 keep 过滤后静默丢失——必须显式并回
            new_only = new_df[~new_df["name"].isin(known)]
            out = pd.concat([keep, upd, new_only], ignore_index=True)
        else:
            out = pd.concat([keep, new_df], ignore_index=True)
        self._save_registry(out)
        log.info("轻量登记: %d -> %d 因子（新增 %d, 更新 %d, 统计刷新 %d 格）",
                 before, len(out), n_add, int(dup.sum()), n_refresh)
        return {"before": before, "after": len(out), "added": n_add,
                "updated": int(dup.sum())}

    # ---- 查询 ----
    def has(self, name: str) -> bool:
        reg = self._load_registry()
        return name in set(reg["name"])

    def list_all(self, kind: str | None = None, family: str | None = None,
                 maturity: str | None = None) -> pd.DataFrame:
        reg = self._load_registry()
        if kind is not None:
            reg = reg[reg["kind"] == kind]
        if family is not None:
            fam = reg.get("family", pd.Series("", index=reg.index)).fillna("")
            reg = reg[fam == family]
        if maturity is not None:
            mat = reg.get("maturity", pd.Series("", index=reg.index)).fillna("")
            reg = reg[mat == maturity]
        return reg

    def list_composites(self) -> pd.DataFrame:
        return self.list_all(kind="composite")

    # ---- 面板/评估路径解析（兼容两条入库通道） ----
    @staticmethod
    def _is_blank(v) -> bool:
        """registry 里的空路径：NaN / None / "" / 字符串化的 "nan"。"""
        if v is None:
            return True
        if isinstance(v, float) and np.isnan(v):
            return True
        return str(v).strip().lower() in ("", "nan", "none")

    def _resolve_panel_path(self, row) -> Path | None:
        """解析该行的因子面板路径，两条入库通道都能命中。

        优先 registry 的 ``panel_path``（``register()`` 写
        ``panels/<md5>.parquet``）；为空或文件缺失时回退
        ``panels/<name>.parquet``——``merge_outputs()`` 的落盘口径
        （全A builder 批量入库只落原名文件、从不写 ``panel_path``）。
        两者都不存在返回 None，不抛异常。

        历史教训（2026-09-12）：此前直接 ``Path(hit.iloc[0]["panel_path"])``，
        扁平行该列为 NaN，``Path(nan)`` 抛 ``TypeError``，导致 912 个因子
        在库 API 上整片不可用。
        """
        v = row.get("panel_path") if hasattr(row, "get") else None
        if not self._is_blank(v):
            p = Path(str(v))
            if p.exists():
                return p
        fb = self.panels_dir / f"{row['name']}.parquet"
        return fb if fb.exists() else None

    def _resolve_eval_path(self, row) -> Path | None:
        """同 :meth:`_resolve_panel_path`，作用于 ``evals/``。

        builder 批量入库不产 eval，故这批因子返回 None（是缺数据，不是异常）。
        """
        v = row.get("eval_path") if hasattr(row, "get") else None
        if not self._is_blank(v):
            p = Path(str(v))
            if p.exists():
                return p
        fb = self.evals_dir / f"{row['name']}.parquet"
        return fb if fb.exists() else None

    def get_panel(self, name: str) -> pd.DataFrame | None:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return None
        p = self._resolve_panel_path(hit.iloc[0])
        return pd.read_parquet(p) if p is not None else None

    def _load_eval(self, name: str) -> pd.DataFrame | None:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return None
        p = self._resolve_eval_path(hit.iloc[0])
        return pd.read_parquet(p) if p is not None else None

    # ---- 多周期 IC 面板（库内预计算，见 IC_HORIZONS） ----
    def ic_matrix(self, horizon: int, *, names=None) -> pd.DataFrame | None:
        """读库内预计算的**逐日截面 Rank IC** 面板（date × factor），不重算。

        这是"库里本来就有"的那份东西：builder 入库时顺手融合好的
        ``ic_h{1,5,10,20}.parquet``。全库四个 horizon 读盘合计约 0.34 s，
        而按同口径从 ``panels_neu`` 重算是分钟级。

        Parameters
        ----------
        horizon : IC 的前瞻期数（交易日）。取 :data:`IC_HORIZONS` 里的值，
            非库内预计算的期限会抛 ``KeyError``（不静默返回空表）。
        names : 只取指定因子列；缺列的因子**补 NaN 列**（不丢列、不报错），
            便于直接与 registry 名单对齐。

        Returns
        -------
        面板；该 horizon 文件不存在时返回 ``None``。

        Notes
        -----
        引用数字时必须写明口径：库版 = ``IC_CODE_STRIDE=3`` 抽稀 + 库内 close_adj；
        与 :meth:`evaluate` 现算的 IC 有 ~3.5% 复权价差，两者不可互换。
        """
        if int(horizon) not in IC_HORIZONS:
            raise KeyError(
                f"ic_h{horizon}.parquet 不是库内预计算的期限，可用 {IC_HORIZONS}；"
                "其他期限请用 factor_analysis / stats.ic 现算")
        p = self.root / f"ic_h{int(horizon)}.parquet"
        if not p.exists():
            log.warning("缺 %s（该数据集未建多周期 IC）", p)
            return None
        df = pd.read_parquet(p)
        if names is not None:
            missing = [n for n in names if n not in df.columns]
            if missing:
                df = df.reindex(columns=[*df.columns, *missing])
            df = df[list(dict.fromkeys(names))]
        return df

    def ic_summary(self, horizons: tuple[int, ...] | None = None, *,
                   window: tuple[str, str] | None = None) -> pd.DataFrame:
        """逐因子 × horizon 的 IC 均值 / IR / 正占比 / 有效天数（秒级）。

        「周频 IC 是什么水平」这类问题**一行答完**，不必逐因子重算：

        >>> lib = FactorLibrary(dataset="all_a_2018_2026")
        >>> lib.ic_summary().sort_values("ic_h5", ascending=False).head()

        Parameters
        ----------
        horizons : 默认全取 :data:`IC_HORIZONS`。
        window : ``(start, end)`` 日期切片（含端点，字符串或 Timestamp）。
            默认用面板全段——**跨数据态引用时务必显式给窗口**。

        Returns
        -------
        DataFrame，index = 因子名，列：``ic_h{h}`` / ``ir_h{h}``（IC 均值 ÷ IC 标准差）
        / ``pos_h{h}``（IC 为正的交易日占比）/ ``n_h{h}``（有效天数）。
        """
        hs = tuple(horizons or IC_HORIZONS)
        mats = {}
        for h in hs:
            m = self.ic_matrix(h)
            if m is None:
                continue
            if window is not None:
                m = m.loc[window[0]:window[1]]
            mats[h] = m
        if not mats:
            raise RuntimeError(f"{self.root} 下没有任何 ic_h*.parquet，库未建多周期 IC")

        out = {}
        for h, m in mats.items():
            mean, std = m.mean(), m.std()
            out[f"ic_h{h}"] = mean
            out[f"ir_h{h}"] = mean / std.replace(0.0, np.nan)
            out[f"pos_h{h}"] = (m > 0).sum() / m.notna().sum().replace(0, np.nan)
            out[f"n_h{h}"] = m.notna().sum()
        return pd.DataFrame(out)

    def load_library_features(self, kind: str | None = None) -> dict:
        """返回 {name: panel}，作为下一轮挖掘的特征集（迭代用）。

        面板解析走 :meth:`_resolve_panel_path`，``register()`` 与
        ``merge_outputs()`` 两条通道入库的因子都可见。个别因子缺面板文件时
        跳过并记 warning（而非整库抛异常）。
        """
        reg = self.list_all(kind=kind)
        out: dict = {}
        missing: list[str] = []
        for _, r in reg.iterrows():
            p = self._resolve_panel_path(r)
            if p is None:
                missing.append(str(r["name"]))
                continue
            out[str(r["name"])] = pd.read_parquet(p)
        if missing:
            log.warning("load_library_features: %d/%d 个因子无面板文件，已跳过（示例: %s）",
                        len(missing), len(reg), ", ".join(missing[:5]))
        return out

    def significance_table(self, q: float = 0.05, exclude_model: bool = True) -> pd.DataFrame:
        """在**整库这个检验族**上做 BH-FDR 多重检验校正，返回逐因子判定表。

        与 registry 里 ``significant`` 列的分工（**族不同，答案本就不同**）：

        - ``significant`` 列 = **单因子原始** NW 显著性（|t_nw| > 2，双侧 α≈0.05），
          刻意不校正——``register()`` 一次只登记一个因子，调用内不存在"检验族"。
        - 本方法 = 把库里已登记的全部因子当作**一个族**做 BH-FDR，回答的是另一个
          问题："把整库当成一个筛选池时，哪些因子值得信？"

        两者都保留、都显式，不用一个去冒充另一个（真源同为 ``stats.significance``）。

        Args:
            q: 目标 FDR 水平（默认 0.05）。
            exclude_model: 排除 model:* 来源（与 ``load_significant_features`` 一致）。
        Returns:
            DataFrame[name, source, t_stat_nw, p_value_nw, raw_significant,
                      fdr_significant]，按 ``p_value_nw`` 升序；无可用 t 的行不参与族。
        """
        from stats.significance import benjamini_hochberg, t_pvalue

        reg = self.list_all()
        if reg.empty:
            return pd.DataFrame(columns=["name", "source", "t_stat_nw", "p_value_nw",
                                         "raw_significant", "fdr_significant"])

        def _col(name, default):
            return reg[name] if name in reg.columns else pd.Series([default] * len(reg),
                                                                   index=reg.index)

        if exclude_model:
            reg = reg[~_col("source", "").fillna("").str.startswith("model:")]
        d = pd.DataFrame({"name": reg["name"].values,
                          "source": _col("source", "").values})
        d["t_stat_nw"] = pd.to_numeric(_col("t_stat_nw", np.nan), errors="coerce").values
        d["raw_significant"] = _col("significant", False).fillna(False).astype(bool).values
        p = pd.to_numeric(_col("p_value_nw", np.nan), errors="coerce").values.astype(float)

        # 旧行（本列引入前入库）没有 p 值：用 t 与 n_dates-1（IC 观测数的上界代理）
        # 补算，并明确记录近似范围——不静默编数。
        miss = ~np.isfinite(p) & np.isfinite(d["t_stat_nw"].values)
        if miss.any():
            if "n_dates" in reg.columns:
                nd = pd.to_numeric(reg["n_dates"], errors="coerce").fillna(250).values
            else:                                # 极旧库缺列：退回 250（≈一年交易日）
                nd = np.full(len(reg), 250.0)
            p[miss] = np.asarray(t_pvalue(d["t_stat_nw"].values[miss],
                                          df=np.maximum(nd[miss] - 1, 1)), dtype=float)
            log.warning("significance_table: %d 行缺 p_value_nw，已用 t 与 n_dates-1 近似补算"
                        "（重跑 register 可获得精确值）", int(miss.sum()))
        d["p_value_nw"] = p
        d["fdr_significant"] = benjamini_hochberg(p, q) if len(d) else np.array([], bool)
        return d.sort_values("p_value_nw", na_position="last").reset_index(drop=True)

    def load_significant_features(self, exclude_model: bool = True,
                                  correction: str = "raw", q: float = 0.05) -> dict:
        """加载 significant 因子面板（2026-08-31 从 e2e_common 下沉）。

        Args:
            exclude_model: 排除 model:* 来源（模型预测回写因子，面板通常滞后，
                且与预测/回测工作流自身循环引用）。默认 True——这是 e2e
                预测日能到数据末端的关键（model:* 面板截至 2025-12-31）。
            correction: 显著性口径。
                - ``"raw"``（默认，行为与历史一致）：用 registry 的 ``significant``
                  列（单因子原始 NW 显著性，未做多重检验校正）。
                - ``"fdr"``：把**整库**当作一个族做 BH-FDR（见
                  ``significance_table``），只留 q 水平下仍显著的因子。
                切换口径会改变喂给下游的因子池，进而改变回测数字——要做横向
                比较时两端口径须一致，别一半 raw 一半 fdr。
            q: ``correction="fdr"`` 时的目标 FDR 水平。
        """
        reg = self.list_all()

        def _col(name, default):
            return reg[name] if name in reg.columns else pd.Series([default] * len(reg),
                                                                   index=reg.index)

        # 列缺失是合法的历史形态：批量入库（merge_outputs()）的库可能压根没有
        # significant / source 列，退回默认值（不显著 / 无来源），而不是 KeyError。
        sig = _col("significant", False).fillna(False).astype(bool)
        if correction == "fdr":
            tbl = self.significance_table(q=q, exclude_model=exclude_model)
            keep = set(tbl.loc[tbl["fdr_significant"], "name"])
            mask = reg["name"].isin(keep)
            log.info("因子库: %d 个因子, raw significant %d 个, BH-FDR(q=%.2f) %d 个",
                     len(reg), int(sig.sum()), q, len(keep))
        elif correction == "raw":
            mask = sig.copy()
            if exclude_model:
                mask &= ~_col("source", "").fillna("").astype(str).str.startswith("model:")
        else:
            raise ValueError(f"未知 correction: {correction}（可选 'raw' | 'fdr'）")
        sig_names = set(reg[mask]["name"])
        if correction == "raw":
            log.info("因子库: %d 个因子, significant %d 个（排除 model:* 后 %d）",
                     len(reg), int(sig.sum()), len(sig_names))

        # 只解析显著因子自己的面板——不无条件遍历整库：既不为 900+ 个用不到的
        # 因子付读盘成本，也不会因个别因子面板缺失把整库拖崩。
        # 历史教训（2026-09-12）：原先写 `self.load_library_features()` 后过滤，
        # 等价于遍历全部 938 行，扁平行 panel_path 为 NaN 时整库抛 TypeError。
        feats: dict = {}
        missing: list[str] = []
        for _, r in reg[mask].iterrows():
            p = self._resolve_panel_path(r)
            if p is None:
                missing.append(str(r["name"]))
                continue
            feats[str(r["name"])] = pd.read_parquet(p)
        if missing:
            log.warning("load_significant_features: %d 个显著因子无面板文件，已跳过（示例: %s）",
                        len(missing), ", ".join(missing[:5]))
        if feats:
            sample = next(iter(feats.values()))
            log.info("加载面板 %d 个, 日期范围 %s ~ %s",
                     len(feats), sample.index[0].date(), sample.index[-1].date())
        return feats

    # ---- 时间段回测查看 ----
    def evaluate_period(self, name: str, start=None, end=None, config: str = "ls_M") -> dict:
        """切片查看某因子在 [start, end] 的回测绩效（秒级，不重算）。

        Returns:
            dict: 含 metrics、ic 统计量、子区间 series、可直接喂报告的 BacktestResult。
        """
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        if config not in [c.key for c in CANONICAL_CONFIGS]:
            raise ValueError(f"未知 config: {config}，可选 {[c.key for c in CANONICAL_CONFIGS]}")
        s, e = _coerce_date(start), _coerce_date(end)
        sub = eval_df.loc[s:e]

        dret = sub[f"dret_{config}"].dropna()
        ic_sub = sub["ic"].dropna()
        metrics = calc_all_metrics(dret)
        ic_mean = float(ic_sub.mean()) if len(ic_sub) else float("nan")
        ic_std = float(ic_sub.std()) if len(ic_sub) else float("nan")
        ic_win = float((ic_sub > 0).mean()) if len(ic_sub) else float("nan")
        ic_ir = calc_ir(ic_sub) if len(ic_sub) >= 2 else 0.0

        bt = BacktestResult(
            daily_returns=dret,
            weights_history=pd.DataFrame(index=dret.index),
            equity_curve=(1 + dret).cumprod(),
            turnover_series=pd.Series(dtype=float),
            cost_series=pd.Series(dtype=float),
            config={"strategy": config, "period": f"{s} ~ {e}"},
        )
        return {
            "name": name, "config": config,
            "start": str(s) if s is not None else None,
            "end": str(e) if e is not None else None,
            "metrics": metrics,
            "ic_mean": ic_mean, "ic_std": ic_std, "ic_win_rate": ic_win, "ic_ir": ic_ir,
            "n_days": len(dret),
            "ic_series": ic_sub,
            "equity_curve": bt.equity_curve,
            "daily_returns": dret,
            "backtest_result": bt,
        }

    def reconstruct_backtest(self, name: str, config: str = "ls_M") -> BacktestResult:
        """从存储的序列重建完整 BacktestResult（用于报告/绘图）。"""
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        dret = eval_df[f"dret_{config}"].dropna()
        return BacktestResult(
            daily_returns=dret,
            weights_history=pd.DataFrame(index=dret.index),
            equity_curve=eval_df[f"equity_{config}"].reindex(dret.index),
            turnover_series=pd.Series(dtype=float),
            cost_series=pd.Series(dtype=float),
            config={"strategy": config},
        )

    # ---- 统一对比 ----
    def compare(
        self,
        metric: str = "ir",
        config: str | None = None,
        ascending: bool = False,
        kind: str | None = None,
        topn: int | None = None,
    ) -> pd.DataFrame:
        """所有因子按统一指标排序，方便挑好因子。

        默认按 **IR（ic_ir，业界统一主轴）** 排序；IR 尺度无关、跨因子可比。
        metric 也支持 ic_mean / sharpe / annual_return / max_drawdown / calmar /
        avg_turnover 等（sharpe/return/calmar/sortino 默认取 best_<metric>）。

        Args:
            metric: 排序指标。
            config: 指定配置列（如 "ls_M"）；为 None 对 sharpe 等用 best_<metric>。
            kind: 仅比较某类（raw/composite）。
            topn: 仅返回前 N 行。
        """
        reg = self.list_all(kind=kind).copy()
        if reg.empty:
            return reg
        # IC 类指标无 config 后缀（单一口径，不随回测配置变，2026-08-05 修复：
        # 原实现把 ic_mean 也拼成 ic_mean_ls_M 导致 KeyError）
        _IC_COLS = {"ic_mean", "ic_std", "ic_ir", "t_stat", "t_stat_nw", "p_value_nw",
                    "ic_win_rate", "ic_decay5", "autocorr", "significant",
                    "monotonicity", "pfs"}
        if metric in ("ir", "ic_ir"):
            col = "ic_ir"
        elif metric in _IC_COLS:
            col = metric
            if col not in reg.columns:
                raise ValueError(f"列不存在: {col}")
        elif config is not None:
            col = f"{metric}_{config}"
            if col not in reg.columns:
                raise ValueError(f"列不存在: {col}")
        elif metric in ("sharpe", "annual_return", "calmar", "sortino"):
            col = f"best_{metric}" if f"best_{metric}" in reg.columns else f"{metric}_{CANONICAL_CONFIGS[0].key}"
        else:
            col = f"{metric}_{CANONICAL_CONFIGS[0].key}"
        reg = reg.sort_values(col, ascending=ascending).reset_index(drop=True)
        reg["_sort_col"] = col
        if topn is not None:
            reg = reg.head(topn)
        return reg

    # ---- 删除 ----
    def delete(self, name: str) -> bool:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return False
        r = hit.iloc[0]
        # 用解析器取路径：扁平行 panel_path/eval_path 为 NaN，`Path(nan)` 抛
        # TypeError，而 `if p` 对 NaN 恰为真——两处都得绕开（2026-09-12 修）。
        for p in (self._resolve_panel_path(r), self._resolve_eval_path(r)):
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    # 文件可能被占用/沙箱回收站不可用：保留文件，仅从 registry 移除
                    log.warning("因子文件删除失败（保留文件）: %s", p)
        reg = reg[reg["name"] != name]
        self._save_registry(reg)
        log.info("已删除因子: %s", name)
        return True

    # ---- 血缘 ----
    def lineage(self, name: str) -> list:
        """返回该因子的父因子链（复合因子才有）。"""
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return []
        parents = hit.iloc[0].get("parents", "")
        return [p for p in str(parents).split("|") if p]

    # ---- 入库前冗余预检（对齐因子工程实践：防"因子库一锅粥"） ----
    def dup_index(self) -> dict | None:
        """加载 ``dup_audit.py`` 预计算的全库冗余索引；不可用返回 None。

        索引三件套（都落在库目录下，随库走，不属于代码仓库）：
        ``dup_matrix.npz``（相关矩阵 + 因子名）· ``dup_signatures.npy``（采样签名，
        ``(K,F,C) float32``，memmap 打开）· ``dup_meta.json``（对齐日历与股票池）。

        有了它，冗余预检从「逐候选读全库 58 MB 面板、965 个约 167 分钟」变成
        「一次矩阵乘、秒级」——因为"因子 i 与 j 有多像"是只与库内容有关的静态
        事实，不该每来一个新因子重算一遍。
        """
        if not self.dataset:
            # 老调用路径：FactorLibrary() 不指定数据集时没有库目录，索引自然不存在
            return None
        base = Path(self.base_root) / self.dataset
        npz_f = base / "dup_matrix.npz"
        sig_f = base / "dup_signatures.npy"
        meta_f = base / "dup_meta.json"
        if not (npz_f.exists() and sig_f.exists() and meta_f.exists()):
            return None
        key = str(base)
        hit = _DUP_INDEX_CACHE.get(key)
        if hit is not None:
            return hit
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            z = np.load(npz_f)
            names = [str(n) for n in z["names"]]
            sig = np.load(sig_f, mmap_mode="r")
            codes = list(meta["codes"])
            if sig.shape[1:] != (len(names), len(codes)):
                log.warning("冗余索引形状不匹配：signature %s vs %d 因子 × %d 股票，"
                            "回退逐面板比对", sig.shape, len(names), len(codes))
                return None
            norms_f = base / "dup_norms.npy"
            if norms_f.exists():
                norms = np.load(norms_f)
            else:
                norms = np.zeros(len(names), dtype=np.float64)
                for k in range(sig.shape[0]):
                    norms += (sig[k].astype(np.float64) ** 2).sum(axis=1)
                np.save(norms_f, norms)
            idx = {"sig": sig, "names": names, "norms": norms, "corr": z["corr"],
                   "dates": pd.DatetimeIndex(meta["dates"]), "codes": codes,
                   "built_at": meta.get("built_at", "?")}
            _DUP_INDEX_CACHE[key] = idx
            log.info("冗余索引已加载：%d 因子 × %d 采样日 × %d 股票（构建于 %s）",
                     len(names), sig.shape[0], len(codes), idx["built_at"])
            return idx
        except Exception as e:  # noqa: BLE001
            log.warning("冗余索引不可用（回退逐面板比对）：%s: %s", type(e).__name__, e)
            return None

    def _dup_check_fast(self, panel, dup_corr, reject_dup, name="") -> dict | None:
        """签名查表版冗余预检；索引不可用或样本不重叠时返回 None（调用方回退）。

        精度：索引只用抽样日历，与全样本逐日相关实测差 ≤0.02（4 对已知因子
        对照），用于 0.7/0.85 这种阈值判断足够；报告口径见
        ``reports/library_governance/dup_audit.md``。
        """
        idx = self.dup_index()
        if idx is None:
            return None
        dates, codes = idx["dates"], idx["codes"]
        arr = panel.reindex(index=dates, columns=codes).to_numpy(dtype=np.float64)
        if arr.size == 0 or np.isfinite(arr).mean() < 0.05:
            return None                      # 与索引日历/股票池几乎不重叠
        zq = _row_rank_zscore(arr)
        sig = idx["sig"]
        num = np.zeros(len(idx["names"]), dtype=np.float64)
        for k in range(len(dates)):
            zk = zq[k]
            if not zk.any():
                continue
            num += sig[k] @ zk
        snew = float((zq.astype(np.float64) ** 2).sum())
        if snew <= 0:
            return None
        den = np.sqrt(np.clip(idx["norms"] * snew, 1e-12, None))
        corr = np.clip(num / den, -1.0, 1.0)
        j = int(np.argmax(corr))
        top_corr, top_name = float(corr[j]), idx["names"][j]
        resid_ic, resid_t = self._residual_ic(panel, self.default_returns_panel(), top_name)
        msg = (f"因子 {name} 冗余预检（签名查表）: 与 {top_name} 相关 {top_corr:.2f}"
               + (f"（>{dup_corr} 疑似冗余）" if top_corr > dup_corr else "")
               + f"；正交残差 IC={resid_ic:.4f} t_nw={resid_t:.2f}"
               + ("（增量信息不足）" if abs(resid_t) < 2.0 else "（含增量信息）"))
        if top_corr > dup_corr:
            log.warning(msg)
            if reject_dup:
                raise ValueError(f"入库被拒（reject_dup）: {msg}")
        else:
            log.info(msg)
        return {
            "dup_checked": True,
            "dup_corr_max": round(top_corr, 3),
            "dup_top": top_name,
            "resid_ic": round(resid_ic, 4),
            "resid_t_nw": round(resid_t, 2),
        }

    def _run_dup_check(self, name, panel, returns_panel, check_dup, dup_corr, reject_dup,
                       include_builder_rows: bool = False) -> dict:
        """冗余预检：与库内已有因子算截面相关 + 对最相关因子做正交残差 IC。

        Args:
            include_builder_rows: 是否把批量入库（``merge_outputs()``，没有
                ``panel_path``）的因子也纳入比较。默认 False——它们的面板要经
                扩展路径才找得到，而全库比较实测约 **167 分钟**（全A 938 个
                ≈57MB 面板，单候选 10.7s，2026-09-12 实测）。开 True 即完整
                覆盖，代价是每次入库都要等这么久。

        Returns:
            dict: dup_checked / dup_corr_max / dup_top / resid_ic / resid_t_nw。
            check_dup=False 时 dup_checked=False，其余为空。

        历史教训（2026-09-12）：原先用 ``Path(str(r.get("panel_path", "")))``，
        扁平行上得到字符串 ``"nan"``、``exists()`` 为 False → ``continue``，
        912 个批量入库因子被**静默跳过**，预检实际只覆盖 20 余个正规因子，
        而 registry 写的是 ``dup_checked=True``。现在覆盖面**一定会打印**，
        不再用静默跳过冒充全量预检。
        """
        empty = {"dup_checked": False, "dup_corr_max": "", "dup_top": "", "resid_ic": "", "resid_t_nw": ""}
        if not check_dup:
            return empty
        fast = self._dup_check_fast(panel, dup_corr, reject_dup, name=name)
        if fast is not None:
            return fast
        hits = []
        reg = self._load_registry()
        candidates = [r for _, r in reg.iterrows() if r["name"] != name]
        n_total = len(candidates)
        if not include_builder_rows:
            candidates = [r for r in candidates if not self._is_blank(r.get("panel_path"))]
        n_pool = len(candidates)
        for r in candidates:
            p = self._resolve_panel_path(r)
            if p is None:
                continue
            try:
                old = pd.read_parquet(p)
            except Exception:
                continue
            common_dates = panel.index.intersection(old.index)
            common_codes = panel.columns.intersection(old.columns)
            if len(common_dates) < 30 or len(common_codes) < 10:
                continue
            a = panel.loc[common_dates, common_codes]
            b = old.loc[common_dates, common_codes]
            valid = a.notna() & b.notna()
            ra = a.where(valid).rank(axis=1)
            rb = b.where(valid).rank(axis=1)
            corr = ra.corrwith(rb, axis=1, method="pearson").mean()
            if not np.isnan(corr):
                hits.append((float(corr), r["name"]))
        if n_pool < n_total:
            log.warning("冗余预检覆盖 %d/%d 个库内因子：%d 个批量入库因子（无 panel_path）"
                        "未参与比较。要全量覆盖用 include_builder_rows=True"
                        "（全库实测约 167 分钟）。",
                        n_pool, n_total, n_total - n_pool)
        if not hits:
            if n_pool:
                log.warning("冗余预检可比对数为 0/%d（候选面板缺失或无重叠样本），"
                            "dup_checked=True 只表示预检已执行、不代表已充分比较", n_pool)
            return {**empty, "dup_checked": True}
        hits.sort(reverse=True)
        log.info("冗余预检覆盖: 实际比较 %d/%d 个库内因子", len(hits), n_total)
        top_corr, top_name = hits[0]
        resid_ic, resid_t = self._residual_ic(panel, returns_panel, top_name)
        msg = (f"因子 {name} 冗余预检: 与 {top_name} 相关 {top_corr:.2f}"
               + (f"（>{dup_corr} 疑似冗余）" if top_corr > dup_corr else "")
               + f"；正交残差 IC={resid_ic:.4f} t_nw={resid_t:.2f}"
               + ("（增量信息不足）" if abs(resid_t) < 2.0 else "（含增量信息）"))
        if top_corr > dup_corr:
            log.warning(msg)
            if reject_dup:
                raise ValueError(f"入库被拒（reject_dup）: {msg}")
        else:
            log.info(msg)
        return {
            "dup_checked": True,
            "dup_corr_max": round(top_corr, 3),
            "dup_top": top_name,
            "resid_ic": round(resid_ic, 4),
            "resid_t_nw": round(resid_t, 2),
        }

    def _residual_ic(self, panel, returns_panel, top_name) -> tuple[float, float]:
        """对新因子做「对 top 因子逐日截面回归取残差」，算残差 IC 与 NW t。

        残差 IC 显著（|t|>2）→ 新因子相对库内最相似因子仍含增量信息；
        不显著 → 只是旧因子的（近似）线性组合，入库价值低。

        口径：逐日 **Rank IC**（残差与未来收益先截面 rank 再 Pearson，与
        ``calc_ic_series`` 的 spearman 默认一致）。2026-09-11 前为原始
        Pearson——与库内 "IC" 一词的其余用法不同义且未声明；存量 registry 行
        的 ``resid_ic`` / ``resid_t_nw`` 为旧口径，重新注册后按新口径覆盖。
        """
        old = self.get_panel(top_name)
        if old is None:
            return float("nan"), 0.0
        common_dates = panel.index.intersection(old.index).intersection(returns_panel.index)
        common_codes = panel.columns.intersection(old.columns).intersection(returns_panel.columns)
        resid_dates, resid_vals = [], []
        for d in common_dates:
            y = panel.loc[d, common_codes]
            x = old.loc[d, common_codes]
            r = returns_panel.loc[d, common_codes]
            ok = y.notna() & x.notna() & r.notna()
            if ok.sum() < 20:
                continue
            yv = y[ok].values.astype(float)
            xv = x[ok].values.astype(float)
            rv = r[ok].values.astype(float)
            X = np.column_stack([np.ones(len(xv)), xv])
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ coef
            y_scale = float(np.std(yv))
            # 完全共线判零用**相对**阈值：面板是 float32，线性相关时残差不会精确
            # 为 0，而是 1e-7 量级的浮点噪声。原来的绝对阈值 1e-10 永远触发不了，
            # 于是"同一个因子换了个名字"会被算出一个纯噪声的伪残差 IC
            # （2026-11 实测 rev_1m 类重复因子给过 +0.0165 的假信号）。
            if y_scale <= 0 or float(np.std(resid)) / y_scale < 1e-6:
                resid_dates.append(d)
                resid_vals.append(0.0)
                continue
            if len(resid) < 5 or np.std(rv) == 0:
                continue
            ic_d = float(np.corrcoef(
                pd.Series(resid).rank().values,
                pd.Series(rv).rank().values,
            )[0, 1])
            if not np.isnan(ic_d):
                resid_dates.append(d)
                resid_vals.append(ic_d)
        if len(resid_dates) < 20:
            return float("nan"), 0.0
        from stats.robust_stats import nw_tstat
        t_nw, _se, _lag = nw_tstat(pd.Series(resid_vals, index=resid_dates).values)
        return float(np.mean(resid_vals)), float(t_nw)

    # ---- 标签管理（六维标签子集：家族/频率/成熟度） ----
    def set_tag(self, name: str, family: str | None = None, frequency: str | None = None,
                maturity: str | None = None, note: str | None = None,
                source: str | None = None) -> bool:
        """给因子补打/更新标签（不重算任何指标）。

        ``source`` 也走这里补写：批量入库（``merge_outputs()``）的因子从未经过
        ``register()``，source 为空、又没有别的回填入口——只补 family/frequency
        却留着 source 空着，等于"来源管理"没做。
        """
        # 补标签同样过受控词表；None 表示"不动这一列"，不能被规范化成空串，
        # 否则会把已有值洗掉。
        if any(v is not None for v in (family, frequency, maturity)):
            _f, _fq, _m = normalize_tags(
                family=family or "", frequency=frequency or "", maturity=maturity or "",
                where=f"set_tag({name})")
            family = _f if family is not None else None
            frequency = _fq if frequency is not None else None
            maturity = _m if maturity is not None else None
        reg = self._load_registry()
        hit = reg["name"] == name
        if not hit.any():
            return False
        for col, val in (("family", family), ("frequency", frequency),
                         ("maturity", maturity), ("note", note), ("source", source)):
            if val is not None:
                if col not in reg.columns:
                    reg[col] = ""
                reg[col] = reg[col].astype(object)  # 字符串写入避免 float64 dtype 冲突
                reg.loc[hit, col] = val
        self._save_registry(reg)
        log.info("已更新标签: %s", name)
        return True

    # ---- 生命周期监控（滚动 IC 漂移） ----
    def monitor(self, window: int = 60) -> pd.DataFrame:
        """库内因子生命周期监控：全期 vs 近期 IC 漂移（复用 evals 已存 IC 序列）。

        Returns:
            DataFrame(name/maturity/ic_mean_full/ic_mean_recent/ic_drift/
                      ic_ir_recent/ic_t_nw_recent/status)，warning 排前。
        """
        from stats.monitor import monitor_ic_series
        reg = self._load_registry()
        rows = []
        n_no_eval = 0
        n_bad = 0
        for _, r in reg.iterrows():
            p = self._resolve_eval_path(r)
            if p is None:
                n_no_eval += 1
                continue
            try:
                ic = pd.read_parquet(p)["ic"].dropna()
            except Exception:
                n_bad += 1
                continue
            if len(ic) < 20:
                n_bad += 1
                continue
            m = monitor_ic_series(ic, window=window)
            rows.append({
                "name": r["name"],
                "kind": r.get("kind", ""),
                "maturity": str(r.get("maturity", "")),
                "family": str(r.get("family", "")),
                "ic_mean_full": m["ic_mean_full"],
                "ic_mean_recent": m["ic_mean_recent"],
                "ic_drift": m["ic_drift"],
                "ic_ir_recent": m["ic_ir_recent"],
                "ic_t_nw_recent": m["ic_t_nw_recent"],
                "n_days": m["n_days"],
                "status": m["status"],
            })
        if n_no_eval or n_bad:
            log.warning("monitor: 覆盖 %d/%d 个因子（跳过 %d 个无 eval 数据——未走 register "
                        "入库的批量因子；%d 个 IC 样本不足或不可读）",
                        len(rows), len(reg), n_no_eval, n_bad)
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows)
                .sort_values(["status", "ic_mean_recent"], ascending=[False, False])
                .reset_index(drop=True))

    # ---- 分市场状态检验（八维检验之一） ----
    def regime_analysis(self, name: str, market_returns: pd.Series | None = None,
                        n_tiles: int = 3) -> dict:
        """按市场状态（牛/熊/震荡，按市场收益分位）分段看因子 IC。

        Args:
            name: 因子名。
            market_returns: 市场日收益 Series（默认从日线缓存读等权市场）。
            n_tiles: 分段数（3=熊/震荡/牛）。
        Returns:
            dict: {段名: {ic_mean, ir, win_rate, n_days, market_ann}}。
        """
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        ic = eval_df["ic"].dropna()
        if len(ic) < 30:
            raise ValueError(f"IC 样本过少（{len(ic)} < 30），无法分市场状态")
        if market_returns is None:
            market_returns = self._default_market_returns(ic.index)
        if market_returns is None or market_returns.empty:
            raise RuntimeError("无法获取市场收益（需要日线缓存或显式传入 market_returns）")
        mr = market_returns.reindex(ic.index).dropna()
        ic = ic.reindex(mr.index).dropna()
        mr = mr.reindex(ic.index)
        if len(ic) < 30:
            raise ValueError("市场收益与 IC 对齐后样本过少")
        seg = pd.cut(mr.rank(pct=True), bins=n_tiles,
                     labels=["熊/弱市", "震荡市", "牛/强市"] if n_tiles == 3 else None)
        out = {}
        for label in seg.cat.categories:
            mask = (seg == label)
            s = ic[mask]
            m = mr[mask]
            out[str(label)] = {
                "ic_mean": float(s.mean()),
                "ir": float(calc_ir(s)) if len(s) >= 2 else 0.0,
                "win_rate": float((s > 0).mean()),
                "n_days": int(len(s)),
                "market_ann": float((1 + m.mean()) ** PERIODS_PER_YEAR - 1),
            }
        return out

    def _default_market_returns(self, dates: pd.Index) -> pd.Series | None:
        """从日线缓存构造等权市场日收益（次期口径，与 IC 对齐）。"""
        try:
            from data.cache import DataCache
            from data.cache_helpers import returns_from_cache
            from data.offline import OfflineDataSource
            cache = DataCache(OfflineDataSource())
            begin = int(str(dates.min().date()).replace("-", ""))
            end = int(str(dates.max().date()).replace("-", ""))
            returns = returns_from_cache(cache, begin, end)
            return returns.mean(axis=1)
        except Exception:
            return None

    # ---- 集合级多样性筛选（DPP，研报系列之二十四 §3.1） ----
    def select_diverse(self, names: list[str] | None = None, k: int | None = None,
                       method: str = "cross", sigma: float = 0.2,
                       quality_col: str | None = "ic_mean",
                       min_overlap_dates: int = 30,
                       min_overlap_codes: int = 10) -> dict:
        """对库内因子做 DPP 集合级多样性筛选（log-det 最大化，去冗余）。

        对比现有两两去重（check_dup / select_low_corr）：DPP 是集合级全局判据，
        不会因三角相关结构（A~B、B~C 高相关，A~C 独立）连锁误杀；结果与顺序无关。

        Args:
            names: 候选因子名（None=库内全部）；可按 source/family 预过滤后传入。
            k: 目标入选数（None=ceil(0.7 × n_pool)，对齐研报 800/1134≈0.7）。
            method: 相关口径，"cross"（逐日截面相关均值）或 "flat"（flatten）。
            sigma: 相似度核带宽（小→对高相关惩罚强，默认 0.2）。
            quality_col: registry 质量列（默认 ic_mean → 质量=|IC| 归一化）；
                         None=纯多样性（对齐研报 DPP 口径）。
            min_overlap_dates / min_overlap_codes: 面板公共样本下限。

        Returns:
            dict: selected(list[str]) / k / n_pool / 各 summary 指标（含质量保留率）。
        """
        from research.dpp_selection import corr_matrix, dpp_select
        reg = self._load_registry()
        if names is None:
            names = list(reg["name"])
        name_set = set(names)
        panels: dict[str, pd.DataFrame] = {}
        meta: dict[str, dict] = {}
        for _, r in reg[reg["name"].isin(name_set)].iterrows():
            p = self._resolve_panel_path(r)
            if p is None:
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            if df.empty:
                continue
            panels[r["name"]] = df
            meta[r["name"]] = dict(r)
        if not panels:
            raise RuntimeError("无可用因子面板（请先入库）")
        corr = corr_matrix(panels, method=method,
                           min_overlap_dates=min_overlap_dates,
                           min_overlap_codes=min_overlap_codes)
        quality = None
        if quality_col is not None:
            ic = pd.Series({n: meta.get(n, {}).get(quality_col, float("nan"))
                            for n in corr.index}, dtype=float)
            ic = ic.fillna(0.0)
            from research.dpp_selection import quality_from_ic
            quality = quality_from_ic(ic)
        if k is None:
            k = int(np.ceil(0.7 * len(corr)))
        res = dpp_select(corr, k=k, quality=quality, sigma=sigma)
        # 质量保留率（入选子集 vs 候选池的 |IC| 均值）
        def _mean_abs(col: str) -> float:
            vals = [meta.get(n, {}).get(col) for n in res["selected"]]
            pool = [meta.get(n, {}).get(col) for n in corr.index]
            v = pd.Series(vals, dtype=float).abs().dropna()
            p = pd.Series(pool, dtype=float).abs().dropna()
            return float(v.mean()) if len(v) else float("nan"), \
                   float(p.mean()) if len(p) else float("nan")
        if quality_col is not None and quality_col in reg.columns:
            sel_m, pool_m = _mean_abs(quality_col)
            res["quality_mean_selected"] = sel_m
            res["quality_mean_pool"] = pool_m
        log.info("DPP 筛选: 池 %d → %d 因子, max|corr| %.3f→%.3f, mean|corr| %.3f→%.3f",
                 res["n_pool"], res["k"], res["max_abs_corr_pool"],
                 res["max_abs_corr_selected"], res["mean_abs_corr_pool"],
                 res["mean_abs_corr_selected"])
        return res
