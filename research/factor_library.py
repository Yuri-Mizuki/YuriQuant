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
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import ShortCostModel
from backtest.engine import BacktestResult, VectorBacktest
from backtest.metrics import PERIODS_PER_YEAR, calc_all_metrics
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
MATURITY_VALUES: frozenset[str] = frozenset(
    {"experimental", "oos_verified", "active", "retired"})

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
    "disc_holder_dyn", "holder_dyn", "significant_synthesis",
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
]


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


def _slug(name: str) -> str:
    """把因子名（可能是长公式）映射为安全的文件名片段。"""
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def _write_parquet_robust(df: pd.DataFrame, path: Path, retries: int = 3) -> None:
    """健壮写盘：先删旧文件再写 + 失败重试。

    Windows 上 pyarrow 覆盖写已存在 parquet 偶发 PermissionError——外部服务
    （Defender 实时扫描 / 索引）对特定文件的瞬态锁（2026-08-05 实测：文件可
    删除但覆盖写被拒，且只发生在个别文件）。先 unlink 再 to_parquet 绕过
    "打开已存在文件"路径；仍失败则退避重试。
    """
    import time
    for attempt in range(retries):
        try:
            try:
                if path.exists():
                    path.unlink(missing_ok=True)
            except OSError:
                pass  # 沙箱 safe-delete / 回收站不可用时退回覆盖写
            df.to_parquet(path, compression="snappy")
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(1 + 2 * attempt)


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
class FactorLibrary:
    """因子持久化仓库。"""

    def __init__(self, root: str | Path | None = None, dataset: str | None = None):
        if root is None:
            try:
                from config import Config
                root = Config.get().get("factor_library", {}).get("root", "e:/data/factor_library/")
            except Exception:
                root = "e:/data/factor_library/"
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
            try:
                from config import Config
                root = Config.get().get("factor_library", {}).get("root", "e:/data/factor_library/")
            except Exception:
                root = "e:/data/factor_library/"
        base = Path(root)
        if not base.exists():
            return []
        return sorted(d.name for d in base.iterdir()
                      if d.is_dir() and (d / "registry.csv").exists())

    # ---- registry IO ----
    def _load_registry(self) -> pd.DataFrame:
        if self._registry_path.exists():
            return _align_registry(
                pd.read_csv(self._registry_path, dtype={"parents": str}))
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
        tmp.replace(self._registry_path)

    # ---- 注册 ----
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
        """注册一个因子：预计算 IC 序列 + 各 canonical 回测，落盘面板/评估/registry。

        Args:
            name: 因子唯一名（建议用公式字符串）。
            panel: date×code 因子面板（建议已截面标准化）。
            returns_panel: date×code 未来一期收益面板。
            kind: 'raw' | 'composite'。
            formula: 人类可读公式（默认=name）。
            parents: 复合因子的父因子名列表（血缘）。
            source: 来源标注（如 "mining:factor_mining_xxx.csv" / "synthesis:ic_weighted"）。
            ic_method: IC 计算方式。
            short_costs: 空头腿成本模型（默认 None=引擎默认：从配置读并启用借券费，
                         修正空头腿乐观偏差；传 ShortCostModel(borrow_rate=0) 关闭）。
            deleverage: 1 倍资金约束（总保证金需求 > 1 时降杠杆）。
            family: 因子族标签，取值应落在 :data:`FAMILY_WHITELIST`（风格族
                    ＋来源族的并集，如 动量/反转/波动率/价值/质量/成长/情绪/
                    量价/基本面/事件/资金流/股东/状态…）；越界只告警、不改写。
            frequency: 信号频率（日频/周频/月频/日内…）。
            maturity: 成熟度状态（experimental / oos_verified / active / retired）。
            note: 备注（设计动机、差异化贡献说明等）。
            check_dup: 入库前冗余预检——与库内已有因子算截面相关性 + 对最相关
                       因子做正交残差 IC 检验（对齐因子工程实践：防止因子库
                       "一锅粥"、判断新因子是否只是旧因子的线性组合）。
            dup_corr: 相关性阈值（> 该值判定疑似冗余；默认 0.7，与业界一致）。
            reject_dup: True 时冗余直接抛 ValueError 拒绝入库；False 仅记 warning
                        （个人研究场景默认警告，保留变体对比的灵活性）。
            dup_include_builder_rows: 冗余预检是否把批量入库（无 ``panel_path``）
                的因子也纳入比较。默认 False = 只比库里走过 ``register()`` 的行，
                成本与历史上一致；True = 完整覆盖，但全库比较实测约 167 分钟。
                无论哪种，覆盖面都会打印出来，不会静默跳过。
        Returns:
            该因子的 registry 行（dict）。
        """
        # 标签先过受控词表（越界只告警）：登记是唯一入口，入口不校验，
        # 后面每个消费者都得各自防漂移。
        family, frequency, maturity = normalize_tags(
            family=family, frequency=frequency, maturity=maturity,
            where=f"register({name})")
        slug = _slug(name)
        panel_path = self.panels_dir / f"{slug}.parquet"
        eval_path = self.evals_dir / f"{slug}.parquet"

        # 1) IC 序列
        ic = calc_ic_series(panel, returns_panel, method=ic_method)
        ic_mean = float(ic.mean()) if ic.notna().any() else float("nan")
        ic_std = float(ic.std()) if ic.notna().any() else float("nan")
        ic_valid = ic.dropna()
        n = len(ic_valid)
        ic_ir = calc_ir(ic) if n >= 2 else 0.0
        # 显著性：一步式给出 OLS 与 NW 两套 t/p（真源 = stats.significance；
        # IC 序列强自相关时 OLS t 会虚高，判显著一律用 NW 那一列）。
        from stats.significance import mean_inference
        _inf = mean_inference(ic_valid) if n >= 2 else {
            "t_stat": 0.0, "p_value": float("nan"),
            "t_stat_nw": 0.0, "p_value_nw": float("nan"),
        }
        t_stat = _inf["t_stat"]
        t_stat_nw = _inf["t_stat_nw"]
        p_value_nw = _inf["p_value_nw"]
        ic_win_rate = float((ic_valid > 0).mean()) if n else float("nan")
        # **原始** NW 显著性（双侧 α=0.05 的惯用等价写法 |t_nw| > 2）：
        # 刻意不做多重检验校正——register() 一次只登记一个因子，调用内不存在
        # 可言的"检验族"，假装存在会让 significant 依赖入库顺序/批次大小。
        # 批量挖掘走 factor/mining 的 BH-FDR；若要在**整库**这个族上校正，用
        # FactorLibrary.significance_table() / load_significant_features(correction="fdr")。
        significant = bool(abs(t_stat_nw) > 2.0)

        # 1b) IC 衰减 + 截面排名自相关（换手率代理）
        try:
            with np.errstate(all="ignore"):
                _decay = calc_ic_decay(panel, returns_panel, max_lag=10)
            ic_decay5 = float(_decay.get(5, float("nan")))
        except Exception:
            ic_decay5 = float("nan")
        try:
            autocorr = factor_autocorr(panel)
        except Exception:
            autocorr = float("nan")

        # 1c) 结构化证据（2026-09-22 evals 改造包，t+单调性论文 / AlphaEval）：
        # 只做报告列、**不参与入库门槛**（门槛仍是可交易口径 |t_nw|>2 + FDR）。
        # 与样本外的相关性验证成立之前，这两列仅用于观察分布。
        try:
            from stats.ic import monotonicity_ratio
            monotonicity = monotonicity_ratio(panel, returns_panel, n_quantiles=5)
        except Exception:
            monotonicity = float("nan")
        try:
            from stats.ic import perturbation_fidelity
            pfs = perturbation_fidelity(panel)
        except Exception:
            pfs = float("nan")

        # 2) canonical 回测
        eval_cols: dict = {"ic": ic}
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

        # 3) 落盘（先删旧文件再写，绕过 Windows 覆盖写偶发锁）
        _write_parquet_robust(panel, panel_path)
        eval_df = pd.DataFrame(eval_cols).sort_index()
        _write_parquet_robust(eval_df, eval_path)

        # 4) registry 行
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
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "t_stat": t_stat,
            "t_stat_nw": t_stat_nw,
            "p_value_nw": p_value_nw,
            "ic_win_rate": ic_win_rate,
            "ic_decay5": ic_decay5,
            "autocorr": autocorr,
            "monotonicity": monotonicity,
            "pfs": pfs,
            "significant": significant,
            **dup_row,
            **metric_rows,
            "best_sharpe": best_sharpe,
            "best_config": best_config,
            "panel_path": str(panel_path),
            "eval_path": str(eval_path),
        }
        reg = self._load_registry()
        reg = reg[reg["name"] != name]  # 覆盖同名
        new_df = pd.DataFrame([row])
        reg = new_df if reg.empty else pd.concat([reg, new_df], ignore_index=True)
        self._save_registry(reg)
        log.info("因子入库: %s (kind=%s, IC=%.4f, best_sharpe=%.3f @%s)",
                 name, kind, ic_mean, best_sharpe, best_config)
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
            fill_missing_only: True（默认）只补空缺字段、不覆盖已有非空值——
                builder 重跑不该把人工补过的标签冲掉。

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
        if dup.any() and fill_missing_only:
            # 已有行：逐字段只补空缺（空串视同空缺），新行整行并入
            old = reg[dup].set_index("name").replace(r"^\s*$", np.nan, regex=True)
            add = new_df.set_index("name").reindex(old.index)
            upd = old.combine_first(add).reset_index()
            # 2026-09-22 修：混合批次（新+已有）时 upd 只覆盖交集，
            # 纯新增行会被 keep 过滤后静默丢失——必须显式并回
            new_only = new_df[~new_df["name"].isin(known)]
            out = pd.concat([keep, upd, new_only], ignore_index=True)
        else:
            out = pd.concat([keep, new_df], ignore_index=True)
        self._save_registry(out)
        log.info("轻量登记: %d -> %d 因子（新增 %d, 更新 %d）",
                 before, len(out), n_add, int(dup.sum()))
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
            if np.std(resid) < 1e-10:
                # 完全共线（残差为浮点噪声）：无增量信息 → IC=0（而非跳过/随机）
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
