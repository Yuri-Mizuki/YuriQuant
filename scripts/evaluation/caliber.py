"""统一评估口径 —— 全项目**唯一真源**（2026-10-13 建立）。

背景（为什么需要这个模块）
--------------------------
2026-10-13 一天之内，同一批「改进 vs 基线」的对照结论发生了**三次反转**，根因
**都不在模型里**，而在**评估口径不同源**：

============  ==========================================================  =====================
口径要素       事故形态                                                     量级
============  ==========================================================  =====================
① 基线形态     拿 ``TopFracLongOnly(0.10)``（研究口径、423 只、换手 72%）   ~2.0pp
               当基线；生产执行层实为 ``BufferedTopFracLongOnly(0.15,0.40)``
② 窗口        探针默认 2019+，生产定版 2018+（2018 是纯 β 年）              ~6.4pp
               → 2026-11-05 已统一：主口径一律 2019+（2018 缺可执行性掩码）
③ Sharpe 定义  算术 ``mean/std·√252`` vs 生产几何 ``年化/年化波动``          0.05~0.13
④ 协方差暖机   收益面板被截断到回测窗口 ⇒ 早期调仓日 Σ 不足 ⇒ QP 静默空仓    ~2.7pp/年
⑤ 成本模型     QP 臂漏传 ``adv/vol`` ⇒ 冲击项恒 0、只扣佣金印花               ~6500bp
⑥ 调仓日       ``.last()`` vs 引擎 ``.first()`` ⇒ 查表落空、权重冻结          —
⑦ 口径组合     拿 `fair`（impact/close/1e8）的数字去比生产 `production`        ~0.75pp
               （固定10bp/T+1open/1e6）—— 2026-10-14 发现：v1w 与 v1-dingban
               被跨口径并列，差 0.75pp 里 0.70pp 来自「close vs T+1 open」
============  ==========================================================  =====================

⇒ 本模块把「一次评估必须对齐的全部要素」收敛到一处。**任何改进都必须经由**
:func:`load_context` + :func:`evaluate_strategy` / :func:`evaluate_weights`，
不得各自重写加载/成本/指标逻辑。**口径组合（成本 × 成交价 × 资金）必须显式**：
按 :data:`PROFILES` 的 `production`（= 登记表 `status: production` 同源）或
`fair`（研究压力口径），且 `describe()` 的横幅要随结论一起引用。
配套的定版登记见 ``config/baselines.yaml``，设计说明见
``reports/docs/design/定版主模型与统一口径.md``。

用法::

    from scripts.evaluation.caliber import get_profile, load_context
    from scripts.evaluation.caliber import cost_profile, evaluate_strategy
    from strategy.cross_section import BufferedTopFracLongOnly

    p = get_profile("production")                          # 生产口径（默认）
    ctx = load_context(signal="ens_h1h5", window="prod", execution=p.execution)
    cost = cost_profile(p.cost)
    m = evaluate_strategy(BufferedTopFracLongOnly(0.15, 0.40), ctx, cost, p.capital)

命令行（复算 / 校验登记表）::

    D:/python/Python312/python.exe scripts/evaluation/caliber.py --check
    D:/python/Python312/python.exe scripts/evaluation/caliber.py --profile fair --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.costs import ImpactCostModel, TransactionCosts  # noqa: E402
from backtest.engine import (  # noqa: E402
    BacktestResult,
    VectorBacktest,
    build_execution_split,
)
from backtest.metrics import (  # noqa: E402
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe_ratio,
    years_to_prove_from_sharpe,
)
from config import Config, baselines_path, load_baselines  # noqa: E402
from optimize.multi_period import PrecomputedWeightsStrategy  # noqa: E402
from stats.robust_stats import nw_tstat  # noqa: E402

# ---------------------------------------------------------------------------
# 1) 口径常量（改这里 = 改全项目口径，慎动）
# ---------------------------------------------------------------------------
#: 回测窗口。
#:
#: **主口径 = ``prod``（2019-01 起，2026-11-05 由莉酱拍板）**。
#: 理由不是「数字好看」（剔 2018 抬高的 7.2pp 年化全是 β 红利，超额只动 0.33pp），
#: 而是**口径不同源**：可执行性掩码（涨跌停封板 / 停牌 / ST）只有 2019 起有状态表，
#: 2018 年按「所有股票都能买进」处理 ⇒ 它与 2019+ 测的不是同一件事。
#: 见 ``load_context`` 里 ``mask`` 的加载分支。
#:
#: ``prod2018`` = 历史口径（2018 起，含缺掩码年份），**仅用于追溯
#: ``config/baselines.yaml`` 里 2018+ 的旧登记值**，不得用于新结论。
#: ``iso2019`` = ``prod`` 的兼容别名（``cost_penalty_probe`` /
#: ``portfolio_weighting`` 等旧脚本仍按此名调用），值相同。
WINDOWS: dict[str, str] = {
    "prod": "20190101",
    "prod2018": "20180101",
    "iso2019": "20190101",
}
DEFAULT_WINDOW = "prod"

#: 信号源登记。``path`` 为**本机** predict 宽表面板（date × code）。
#: ``reproducible`` = 本机是否已具备该面板实体。
SIGNALS: dict[str, dict] = {
    "ens_h1h5": {
        "path": "reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet",
        "desc": "882 ortho · gbdt h1+h5 截面秩平均（**本机工作基线**）",
        "reproducible": True,
    },
    "gbdt_h1": {
        "path": "reports/alla_rolling_ortho/pred/gbdt__h1.parquet",
        "desc": "882 ortho · gbdt 单周期 h1（主基线单模型）",
        "reproducible": True,
    },
    # 单周期分量：2026-11-10 起作为复合信号（gbdt_avg4 / gbdt_mid13）的基座登记。
    # 它们本身就是落盘实体（非复合），单独引用时注意 h1 单用的 K 曲线是灾难性的
    # （`yqcore-focus` 升级研究：h1 权重 1/3 是过滤器、1/2 就开始亏）。
    "gbdt_h5": {
        "path": "reports/alla_rolling_ortho/pred/gbdt__h5.parquet",
        "desc": "882 ortho · gbdt 单周期 h5",
        "reproducible": True,
    },
    "gbdt_h10": {
        "path": "reports/alla_rolling_ortho/pred/gbdt__h10.parquet",
        "desc": "882 ortho · gbdt 单周期 h10",
        "reproducible": True,
    },
    "gbdt_h20": {
        "path": "reports/alla_rolling_ortho/pred/gbdt__h20.parquet",
        "desc": "882 ortho · gbdt 单周期 h20（长周期选择器；单用 K≤30 都只有 ~23%）",
        "reproducible": True,
    },
    "ens_all": {
        "path": "reports/alla_rolling_ortho/pred/ens_all__h1.parquet",
        "desc": "882 ortho · 全模型族集成",
        "reproducible": True,
    },
    # 生产定版信号（fundind 快信号 h1020 ⊕ 慢信号 s0.5）。
    # 🚨 2026-10-14 更正：该信号**不是** ``stage_ensemble`` 的产物 —— 该阶段只写
    # ``ens_rgr`` / ``ens_h1h5`` / ``ens_all``（源码 ``rolling_grid_alla.py:1588-1596``），
    # **没有** ``ens_h1020``。它由 ``scripts/evaluation/buffer_arm_920.py
    # --variant defv`` **现场构造**：``_rank_average([gbdt__h1, h5, h10, h20])``
    # ⊕ ``0.5·rank(slow)``（slow 来自 ``fundamental_blend.build_slow_panel``）。
    # ✅ 2026-10-21（批次 12/13）：pred 四件套已随 920 产物同步到本机，defv 面板
    # 已构造落盘（同 buffer_arm_920 defv 逐口径，72 因子慢信号当前数据态），
    # 登记转 ``reproducible=True``；数字属 income 补齐后数据态（≠ 登记表 18.07
    # 旧数据态，见 baselines.yaml data_state）。
    "dingban_fundind": {
        "path": "reports/frontier_audit_920_reval/defv_sig__h1.parquet",
        "desc": "920 fundind · h1/5/10/20 秩平均 ⊕ 慢信号 s0.5（**生产定版**）",
        "reproducible": True,
        "construct": "scripts/evaluation/buffer_arm_920.py --variant defv",
        "regen_inputs": ("reports/alla_rolling_ortho920fundind/pred/"
                         "gbdt__h{1,5,10,20}.parquet"),
    },
    # 本机代理：2026-11-09 用当前数据态重跑 select+predict 后现场构造的 defv，
    # 落在滚动产物目录里（登记路径 reports/frontier_audit_920_reval/ 在本机已不存在）。
    # 🚨 **不是**登记指纹 2bc26c866512 那份：窗口止 2026-11-04（1900 日）、
    #    特征清单全部重选。v2-dingban（21.57%、36bp 旧基数、1876 日）已于
    #    2026-11-12 并入 yqcore-breadth ⇒ 本键就是定版基线的唯一信号源。
    #    `yqcore` 族的报数用本键并显式截到 2026-09-07（与 882 同末日）。
    "dingban_fundind_local": {
        "path": "reports/alla_rolling_ortho920fundind/pred/defv__h1.parquet",
        "desc": "920 fundind defv（**本机重建代理**，非登记指纹）",
        "reproducible": True,
        "construct": "scripts/evaluation/buffer_arm_920.py --variant defv --sig-only",
        "regen_inputs": ("reports/alla_rolling_ortho920fundind/pred/"
                         "gbdt__h{1,5,10,20}.parquet"),
    },
    # ---- 复合信号（多份 pred 面板的加权截面秩平均；无落盘实体，`load_context` 现场构造）
    # 2026-11-10 新增：`yqcore-focus` 从 `ens_h1h5`（只用 h1+h5）升到 h1/h5/h10/h20
    # 集成。依据见 `reports/focus_upgrade/README.md`：只用 h1+h5 时 K 曲线整体落在
    # 22.1%（11 格均值），任何含 h20 的组合都在 28~30% ⇒ h1+h5 是**漏了长周期选择器**。
    # 🚨 复合信号的指纹是**构造后**面板的指纹，与任一分量不同；引用必须带构造式。
    "gbdt_avg4": {
        "components": {"gbdt_h1": 1.0, "gbdt_h5": 1.0,
                       "gbdt_h10": 1.0, "gbdt_h20": 1.0},
        "desc": "882 ortho · gbdt h1/h5/h10/h20 **等权**秩平均",
        "reproducible": True,
    },
    "gbdt_mid13": {
        "components": {"gbdt_h1": 2.0, "gbdt_h5": 1.0,
                       "gbdt_h10": 1.0, "gbdt_h20": 2.0},
        "desc": ("882 ortho · gbdt h1:h5:h10:h20 = 2:1:1:2 秩平均"
                 "（h1/h20 各 1/3，两个中期合计 1/3）"),
        "reproducible": True,
    },
}

#: 默认资金规模（冲击成本的量纲依赖它）。**注意**：生产口径是 ``1e6``（见 PROFILES），
#: 本常量仅作 ``fair`` 口径与向后兼容的缺省。
DEFAULT_CAPITAL = 1e8

#: 活跃仓位判定阈值。OSQP 会在数百只上留 1e-8~1e-20 级数值微仓，用 ``!= 0``
#: 会把「10 只满仓的顶点解」误报成「264 只活跃」（2026-10-13 实测）。
N_ACTIVE_TOL = 1e-4

#: 成交价口径 —— 决定「信号日 → 成交日」的结算方式：
#:   ``close`` = T 收盘成交（引擎默认；**乐观上限**，比生产高约 +0.7pp）；
#:   ``open`` / ``vwap`` = T+1 开盘 / T+1 VWAP 成交（**生产定版口径**，经
#:   ``build_execution_split`` + 信号 ``shift(1)`` 预掩码）。
EXECUTIONS: tuple[str, ...] = ("close", "open", "vwap")


@dataclass(frozen=True)
class Profile:
    """一次评估的**完整口径组合**（成本 × 成交价 × 资金）。

    2026-10-14 新增。此前 caliber 只硬编码了「研究压力口径」
    （impact / close / 1e8），而生产定版 ``v1-dingban`` 的口径是
    「固定 10bp / T+1 open / 1e6」（源：``scripts/evaluation/buffer_arm_920.py``）
    —— **三项都不同**，导致两条基线的绝对数字被跨口径并列（实测差 0.75pp）。
    现在口径组合显式化，`describe()` 会把 profile 打进横幅，随结论一起引用。
    """
    name: str
    cost: str          # :func:`cost_profile` 的 kind
    execution: str     # :data:`EXECUTIONS` 之一
    capital: float
    note: str


PROFILES: dict[str, Profile] = {
    "production": Profile(
        name="production", cost="production", execution="open", capital=1e6,
        note="生产定版口径（buffer_arm_920.py：27bp 时变成本 + T+1 open + 1e6 + 预掩码）。"
             "与登记表 `status: production` 条目同源，绝对数字可直接比。"
             "掩码口径 = 方案② mask-lag 1（T+1 状态，由莉酱 2026-10-15 拍板："
             "报数一律本口径；mask-lag 0 + T 收盘仅研究上限，数字不得并列）。"
             "🚨 成本 2026-11-11 修正：原「固定 10bp」文案已作废 —— 实收往返 27bp"
             "（佣金万1 双边 2bp + 印花税万5 单边 5bp + 滑点 10bp 双边 20bp），"
             "且**印花税按调仓日时变**（2023-08-28 前千1、之后万5，见"
             "`backtest.costs.stamp_duty_on`）。历史登记的 10bp 数字不可与新数字并列。"),
    "fair": Profile(
        name="fair", cost="impact", execution="close", capital=1e8,
        note="研究压力口径（逐股冲击 + T 收盘 + 1e8）：成本稳健性 / 容量 / QP 对照用。"
             "绝对数字**不可**与登记表 production 并列（比生产高约 +0.7pp）。"),
}
#: 缺省口径组合 = **生产口径**（用户 2026-10-13 指令：改进都在「定版主模型 + 统一口径」上做）。
DEFAULT_PROFILE = "production"

BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"


def _daily_candidates() -> tuple[Path, ...]:
    """daily_all_a 缓存（Config.cache root；禁止硬编码盘符——多机适配约定）。

    2026-10-19 审计：原实现并列 E:/D: 两个绝对路径取覆盖最长者；换机器/迁移
    数据根后会静默读旧副本。现统一走配置根。
    """
    from config import Config

    return (Path(str(Config.cache()["root"])) / "daily_all_a.parquet",)


# ---------------------------------------------------------------------------
# 2) 基础原语
# ---------------------------------------------------------------------------
#: 支持的调仓频率（与 ``VectorBacktest._get_rebalance_days`` 同名同义）。
#: 2026-11-10 起 ``rebalance_dates`` / ``next_day_rebalance`` / ``run_once`` 参数化，
#: 周频实验不再需要在探针里另写一套调仓日（那会与引擎静默漂移）。
REBALANCE_FREQS: tuple[str, ...] = ("D", "W", "2W", "M")


def rebalance_dates(idx: pd.DatetimeIndex, freq: str = "M",
                    phase: int = 0) -> pd.DatetimeIndex:
    """调仓**信号日**（T）。

    Args:
        freq: ``D`` / ``W`` / ``2W`` / ``M``。``phase=0`` 时**必须与
            ``VectorBacktest._get_rebalance_days`` 逐位一致**（引擎用
            ``groupby(to_period(freq)).first()`` / ``2W`` 取 ``[::2]``）；否则
            ``PrecomputedWeightsStrategy`` 按日期查表落空、退化为「恒用最后一期
            权重」，回测数字失真（2026-10-13 踩坑）。
        phase: 周内/月内相位 —— 取该周期第 ``phase`` 个交易日（0 = 首日）。
            引擎不认这个参数（它只会取首日），故 ``phase>0`` 只能配合
            ``run(rebalance_days=...)`` 显式传日期使用；周内交易日不足时回退到
            该周期**最后**一个交易日（防节假日把整周丢掉）。

    ``phase`` 存在的理由（2026-11-10）：周频的毛超额在四个周内相位上差异极大
    （既有记录：单相位可达四相位均值 1.9 倍），固定「周一」会把结论锁死在一个
    相位上，必须能扫。
    """
    if freq not in REBALANCE_FREQS:
        raise KeyError(f"未知调仓频率 {freq!r}，可选 {list(REBALANCE_FREQS)}")
    if freq == "D":
        return pd.DatetimeIndex(idx)
    s = pd.Series(idx, index=idx)
    per = s.index.to_period(freq)
    if phase <= 0:
        picked = s.groupby(per).first()
        if freq == "2W":
            picked = picked.iloc[::2]
        return pd.DatetimeIndex(picked.values)
    # 周期内第 phase 个交易日；不足则回退该周期最后一日（防节假日丢整周）
    pos = pd.Series(s.groupby(per).cumcount().to_numpy(), index=s.index)
    nth = s[pos == phase]
    have = set(nth.index.to_period(freq))
    last = pd.DatetimeIndex(s.groupby(per).last().to_numpy())
    # groupby(PeriodIndex).last() 的索引是 Period ⇒ 必须先还原 Timestamp
    # （2026-11-10 踩坑：混着 concat 会 TypeError）。
    tail = last[~last.to_period(freq).isin(have)]
    picked = pd.DatetimeIndex(sorted(set(nth.values) | set(tail.values)))
    if freq == "2W":
        picked = picked.iloc[::2]
    return pd.DatetimeIndex(picked.values)


def sharpe_pair(dr: pd.Series) -> dict:
    """两套 Sharpe 口径并列。

    ``sharpe`` = **生产口径**（:func:`backtest.metrics.sharpe_ratio`，几何年化/年化波动）；
    ``sharpe_arith`` = 算术 ``mean/std·√252``（**仅供追溯旧产物**，2026-10-14 前的
    ``summary_*.csv`` 的 ``sharpe`` 列是算术口径；两者差 0.05~0.13）。
    """
    dr = dr.dropna()
    if dr.empty or dr.std() == 0:
        return {"sharpe": float("nan"), "sharpe_arith": float("nan")}
    return {
        "sharpe": float(sharpe_ratio(dr)),
        "sharpe_arith": float(dr.mean() / dr.std() * np.sqrt(252.0)),
    }


def _load_daily_amount() -> pd.DataFrame:
    """取 ``Config.cache()`` 下的 ``daily_all_a``，返回 amount 宽表（date × code，元）。"""
    candidates = _daily_candidates()
    best, best_n = None, -1
    for p in candidates:
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        lv = d.index.levels[0] if isinstance(d.index, pd.MultiIndex) else d.index
        if len(lv) > best_n:
            best, best_n = d, len(lv)
    if best is None:
        raise FileNotFoundError(f"无可用 daily_all_a：{candidates}")
    out = best["amount"].unstack("code").astype(float)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def fingerprint_panel(panel: pd.DataFrame, n: int = 12) -> str:
    """信号面板指纹（sha1 前 ``n`` 位）。用于登记表核对「跑的是不是同一份信号」。"""
    h = hashlib.sha1()
    h.update(str(panel.shape).encode())
    h.update(str(panel.index.min()).encode())
    h.update(str(panel.index.max()).encode())
    arr = np.nan_to_num(panel.to_numpy(dtype=np.float64), nan=-1e308)
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:n]


# ---------------------------------------------------------------------------
# 3) 上下文
# ---------------------------------------------------------------------------
@dataclass
class CaliberContext:
    """一次评估所需的全部面板 + 口径元信息。

    🚨 ``fwd``（窗口内前瞻收益）与 ``ret_hist``（**不截断**的收益，供 Σ 暖机）
    是两个不同面板 —— 混用会造成「协方差历史不足 ⇒ 静默空仓」。
    """
    window: str
    signal: str
    begin: str
    pred: pd.DataFrame          # 窗口内信号（date × code）
    close: pd.DataFrame         # 窗口内后复权收盘
    fwd: pd.DataFrame           # 窗口内日收益（close.pct_change, fill_method=None）
    mask: pd.DataFrame          # 可交易掩码（窗口内，NaN→True）
    amt: pd.DataFrame           # 窗口内成交额（ADV 用）
    vol60: pd.DataFrame         # 窗口内 60 日滚动波动（冲击模型用）
    ret_hist: pd.DataFrame      # **不截断**收益（协方差暖机专用）
    fingerprint: str
    execution: str = "close"    # 成交价口径（EXECUTIONS 之一）
    fill: pd.DataFrame | None = None   # 窗口内成交价（open/vwap 专用；close 时为 None）

    @property
    def n_days(self) -> int:
        return len(self.pred)


def weighted_rank_average(panels: list[pd.DataFrame],
                          weights: list[float] | None = None) -> pd.DataFrame:
    """多面板**加权**截面秩平均 —— 复合信号的唯一构造口径。

    ``Σ w_k · rank_pct_k / Σ w_k``，只在**各面板都有值**的格上输出（取交集）。
    与 ``scripts/common/alla_common.rank_average``（逐日循环实现）同口径，但
    **向量化**（``rank(axis=1)``）：2000 日 × 5800 列从 ~10 秒降到 ~1 秒。

    为什么放在 caliber 而不是复用实验侧：复合信号必须能被 ``load_context``
    构造，否则登记表里会出现「只有某探针脚本能算出来的信号」；两边口径一致由
    ``tests/test_caliber.py::test_weighted_rank_average_matches_experiment`` 锁死。

    Args:
        panels: 同形面板列表（date × code）。
        weights: 与 ``panels`` 对齐；``None`` = 等权。给 ``None`` 与给全 1 结果相同。
    """
    if not panels:
        raise ValueError("weighted_rank_average 需要至少一份面板")
    w = np.ones(len(panels)) if weights is None else np.asarray(weights, float)
    if len(w) != len(panels):
        raise ValueError(f"权重数 {len(w)} 与面板数 {len(panels)} 不一致")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError(f"权重须非负且不全为 0：{w}")

    base = panels[0]
    ranks, valid = [], None
    for p, wi in zip(panels, w):
        if wi == 0:                        # 零权重分量不参与，也不该约束交集
            continue
        p = p.reindex(index=base.index, columns=base.columns)
        ranks.append(wi * p.rank(axis=1, pct=True))
        v = p.notna()
        valid = v if valid is None else (valid & v)
    if not ranks:
        raise ValueError("所有分量权重都是 0，无法构造复合信号")
    out = sum(ranks) / w.sum()
    return out.where(valid).astype("float32")


def _composite_panel(signal: str) -> pd.DataFrame:
    """按 ``SIGNALS[signal]["components"]`` 现场构造复合信号面板。"""
    spec = SIGNALS[signal]
    panels, weights = [], []
    for key, wt in spec["components"].items():
        sub = SIGNALS[key]
        fp = ROOT / sub["path"]
        if not fp.exists():
            raise FileNotFoundError(
                f"复合信号 {signal!r} 的分量 {key!r} 面板缺失：{fp}")
        panels.append(pd.read_parquet(fp))
        weights.append(float(wt))
    return weighted_rank_average(panels, weights)


def load_context(signal: str = "ens_h1h5", window: str = DEFAULT_WINDOW,
                 begin: int | None = None, end: int | None = None,
                 pred_path: str | Path | None = None,
                 execution: str = "close") -> CaliberContext:
    """加载统一口径上下文。

    Args:
        signal: ``SIGNALS`` 的键（决定读哪份 predict 面板）。
        window: ``WINDOWS`` 的键（决定回测起点）。
        begin/end: 显式覆盖窗口（int ``YYYYMMDD`` 或 None）。
        pred_path: 直接指定 signal panel（覆盖 ``signal`` 的 path）。
        execution: 成交价口径（``EXECUTIONS``）。非 ``close`` 时加载对应
            ``_base/<execution>_adj.parquet``（``open``/``vwap`` 已在本机）。
    """
    if window not in WINDOWS:
        raise KeyError(f"未知窗口 {window!r}，可选 {sorted(WINDOWS)}")
    if execution not in EXECUTIONS:
        raise KeyError(f"未知成交价口径 {execution!r}，可选 {list(EXECUTIONS)}")
    if signal not in SIGNALS and pred_path is None:
        raise KeyError(f"未知信号 {signal!r}，可选 {sorted(SIGNALS)}")

    spec = SIGNALS.get(signal, {})
    if pred_path is not None:
        p = Path(pred_path)
    elif spec.get("components"):
        p = None                       # 复合信号：无落盘实体，下面现场构造
    elif spec.get("path") is None:
        raise FileNotFoundError(
            f"信号 {signal!r} 无落盘面板，须**现场构造**：{spec.get('construct')}\n"
            f"（基座输入 {spec.get('regen_inputs')}；"
            f"重建步骤见 config/baselines.yaml 的 regen_cmd）")
    else:
        p = ROOT / spec["path"]
    if p is not None and not p.exists():
        raise FileNotFoundError(
            f"信号面板不存在：{p}\n"
            f"（信号 {signal!r} 本机 reproducible={spec.get('reproducible')}；"
            f"重建命令见 config/baselines.yaml 的 regen_cmd）")

    close_full = pd.read_parquet(BASE_DIR / "close_adj.parquet")
    mask_full = pd.read_parquet(BASE_DIR / "tradable_mask.parquet")
    pred_full = (_composite_panel(signal) if p is None
                 else pd.read_parquet(p))
    amt_full = _load_daily_amount()

    lo = str(begin) if begin is not None else WINDOWS[window]
    hi = str(end) if end is not None else None
    idx = pred_full.index
    keep = idx[idx >= pd.Timestamp(lo)]
    if hi is not None:
        keep = keep[keep <= pd.Timestamp(hi)]

    cols = pred_full.columns
    pred = pred_full.reindex(index=keep, columns=cols)
    close = close_full.reindex(index=keep, columns=cols)
    mask = mask_full.reindex(index=keep, columns=cols).fillna(True)
    amt = amt_full.reindex(index=keep, columns=cols)

    fwd = (close.pct_change(fill_method=None)
           .replace([np.inf, -np.inf], np.nan).fillna(0.0))
    vol60 = fwd.rolling(60, min_periods=20).std()
    # Σ 暖机：用**完整历史**（不截断），否则窗口最早几个调仓日历史不足 ⇒
    # rolling_covariance 返回 None ⇒ 组合静默空仓（2026-10-13 实测 −2.7pp/年）。
    ret_hist = (close_full.reindex(columns=cols).pct_change(fill_method=None)
                .replace([np.inf, -np.inf], np.nan).fillna(0.0))

    fill = None
    if execution != "close":
        fp = BASE_DIR / f"{execution}_adj.parquet"
        if not fp.exists():
            raise FileNotFoundError(
                f"成交价面板缺失：{fp}\n先跑 rolling_grid_alla --stage prep 重建 _base。")
        fill = pd.read_parquet(fp).reindex(index=keep, columns=cols)

    return CaliberContext(
        window=window, signal=signal, begin=lo, pred=pred, close=close, fwd=fwd,
        mask=mask, amt=amt, vol60=vol60, ret_hist=ret_hist,
        fingerprint=fingerprint_panel(pred), execution=execution, fill=fill,
    )


# ---------------------------------------------------------------------------
# 4) 成本口径
# ---------------------------------------------------------------------------
def cost_profile(kind: str = "impact", *, eta: float | None = None,
                 buy_mult: float | None = None,
                 slippage_bp: float | None = None) -> TransactionCosts:
    """标准成本口径。费率一律取 :meth:`Config.costs`（全项目唯一真源）。

    Args:
        kind: ``"production"`` = **生产定版口径**，直接转调
              :func:`backtest.costs.default_costs`（与 ``buffer_arm_920.py`` /
              ``rolling_grid_alla`` 逐字同源，零漂移）；
              ``"impact"`` = 逐股平方根律冲击（**研究压力口径**，`fair` profile 用）；
              ``"flat"`` = 固定 ``slippage_bp``（项目历史假设，零回归对照）。
    """
    if kind == "production":
        from backtest.costs import default_costs
        return default_costs()
    c = Config.costs()
    # 三个口径统一开启时变印花税：窗口跨 2023-08-28 政策变更（千1→万5），
    # 单一常数会让一半窗口用错税率。见 backtest.costs.stamp_duty_on。
    base = dict(commission_rate=c["commission_rate"],
                commission_min=c["commission_min"],
                stamp_duty=c["stamp_duty"],
                stamp_duty_schedule=True)
    if kind == "impact":
        return TransactionCosts(
            **base, slippage_bp=0.0,
            impact=ImpactCostModel(
                eta=float(eta if eta is not None else c["impact_eta"]),
                buy_multiplier=float(buy_mult if buy_mult is not None
                                     else c["impact_buy_multiplier"])))
    if kind == "flat":
        return TransactionCosts(
            **base,
            slippage_bp=float(slippage_bp if slippage_bp is not None
                              else c["slippage_bp"]))
    raise KeyError(f"未知成本口径 {kind!r}，可选 'production' / 'impact' / 'flat'")


# ---------------------------------------------------------------------------
# 5) 评估
# ---------------------------------------------------------------------------
def metrics_row(res: BacktestResult, capital: float) -> dict:
    dr = res.daily_returns
    imp = getattr(res, "impact_cost_series", None)
    sp = sharpe_pair(dr)
    return {
        "ann": float(annual_return(dr)),
        **sp,
        "vol": float(annual_volatility(dr)),
        "max_dd": float(max_drawdown(dr)),
        "turnover": float(res.turnover_series.dropna().mean()),
        "cost_bp": float(res.cost_series.sum() / float(capital) * 1e4),
        # 冲击成本单列（cost_bp 已含佣金+印花+冲击）—— 容量表的 AUM 敏感项。
        "impact_bp": (float(imp.sum() / float(capital) * 1e4)
                      if imp is not None and len(imp) else 0.0),
        "n_active": float((res.weights_history > N_ACTIVE_TOL).sum(axis=1).mean()),
        "n_days": int(len(dr)),
        # ---- 显著性（2026-10-28 补）----
        # 此前本函数自己拼 dict、**从不调 `calc_all_metrics`** ⇒ 走 caliber 的每条
        # 结论都不带 t / 证明年数，「指标在 metrics 里算、却从没进评估主路径」。
        # t 恒等式（`t = SR·√T`）只认**算术** SR ⇒ 两个 years_to_prove 并列，
        # 引用显著性一律用 `_arith` 那个；`years_to_prove` 保留是为了与
        # `backtest.metrics.calc_all_metrics` 的既有产物对得上。
        "sharpe_t_stat": float(nw_tstat(dr.dropna())[0]),
        "years_to_prove": years_to_prove_from_sharpe(sp["sharpe"]),
        "years_to_prove_arith": years_to_prove_from_sharpe(sp["sharpe_arith"]),
    }


def next_day_rebalance(idx: pd.DatetimeIndex, freq: str = "M",
                       phase: int = 0) -> set:
    """T+1 成交口径的**执行日**集合：每个调仓信号日（:func:`rebalance_dates`）的
    **次一交易日**。

    与 ``scripts/evaluation/buffer_arm_920.py`` 及 ``rolling_grid_alla`` 的
    ``execution != "close"`` 分支逐字一致（信号 T 收盘产生、T+1 开盘执行）。

    2026-10-19 口径修复：本集合只用于 ``run(rebalance_days=...)``；
    ``build_execution_split`` 必须收**信号日 T**（:func:`rebalance_dates`），
    否则 fill 会被写到 E+1、执行价失效（详见 :func:`run_once`）。

    2026-11-10：加 ``freq`` / ``phase`` 透传（周频实验用；缺省月频，逐位兼容）。
    """
    pos = {d: i for i, d in enumerate(idx)}
    return {idx[pos[t] + 1]
            for t in rebalance_dates(idx, freq=freq, phase=phase)
            if pos[t] + 1 < len(idx)}


def run_once(bt: VectorBacktest, ctx: CaliberContext, execution: str,
             signal: pd.DataFrame | None = None,
             price_panel: pd.DataFrame | None = None,
             lot_size: int | None = None,
             freq: str = "M", phase: int = 0):
    """按成交价口径执行一次回测。

    ``close``：T 收盘成交（引擎默认路径，传 ``executable_mask``）。
    ``open`` / ``vwap``：**生产 T+1 口径** —— 信号 ``shift(1)`` 预掩码、
    执行日 = 信号日次一交易日、经 ``build_execution_split`` 结算。与
    ``buffer_arm_920.py`` 的三行调用逐字对齐。

    2026-10-19 修复（审计 P0-1）：``build_execution_split`` 收**信号日 T**
    （:func:`rebalance_dates`），``run`` 收**执行日 E=T+1**
    （:func:`next_day_rebalance`）。此前两者都传 E，fill 被写到 E+1、
    实际变成 T+1 收盘成交。

    Args:
        price_panel: **未复权**收盘价（date × code），与 ``lot_size`` 一起启用
            **整手约束**（2026-11-10）。不传（默认）= 无整手，与历史数字逐位一致。
            🚨 面板必须是 ``close_raw.ffill().shift(1)`` —— 引擎在**执行日 E** 取
            ``price_values[i]``，shift 后才是信号日 T 的价格 ⇒ **无前视**
            （T 收盘选股、T+1 开盘成交，用 T 收盘价估算手数）。直接传未 shift 的
            raw 会用当日收盘价定当日成交股数，属前视（实测虚高 ~1pp）。
        lot_size: 最小交易单位（A 股 100）。
        freq / phase: 调仓频率与相位（2026-11-10）。``freq="M"`` 逐位等于历史行为。
            ``close`` 路径下会就地把 ``bt.rebalance_freq`` 改成 ``freq``（引擎按
            ``_get_rebalance_days`` 自算调仓日）；``phase>0`` 仅对
            ``open``/``vwap`` 生效（显式传 ``rebalance_days``）。
    """
    if execution == "close":
        if freq != "M":
            bt.rebalance_freq = freq
        panel = ctx.pred if signal is None else signal
        return bt.run(panel, ctx.fwd, executable_mask=ctx.mask,
                      adv_panel=ctx.amt, vol_panel=ctx.vol60,
                      price_panel=price_panel, lot_size=lot_size)
    if ctx.fill is None:
        raise ValueError(
            f"execution={execution!r} 需要 ctx.fill；请用 "
            f"load_context(..., execution={execution!r}) 加载成交价面板。")
    panel = ctx.pred if signal is None else signal
    sig_used = panel.shift(1).where(ctx.mask)
    rb_signal = set(rebalance_dates(ctx.pred.index, freq=freq, phase=phase))
    rb_exec = next_day_rebalance(ctx.pred.index, freq=freq, phase=phase)
    split = build_execution_split(ctx.fill, ctx.close, rb_signal)
    return bt.run(sig_used, ctx.fwd, horizon=1, rebalance_days=rb_exec,
                  execution_split=split,
                  adv_panel=ctx.amt, vol_panel=ctx.vol60,
                  price_panel=price_panel, lot_size=lot_size)


def evaluate_strategy(strategy, ctx: CaliberContext, cost: TransactionCosts,
                      capital: float = DEFAULT_CAPITAL,
                      execution: str | None = None,
                      signal: pd.DataFrame | None = None,
                      yearly: bool = False,
                      price_panel: pd.DataFrame | None = None,
                      lot_size: int | None = None,
                      freq: str = "M", phase: int = 0) -> dict:
    """在统一口径下回测一个 ``Strategy``（真正走引擎的选股路径）。

    Args:
        execution: 覆盖 ``ctx.execution``（``close`` / ``open`` / ``vwap``）。
        signal: 覆盖选股信号面板（date × code，与 ``ctx.pred`` 同 index/columns；
            overlay / 信号变换实验用。None = ctx.pred）。
        yearly: 附带逐年日收益（`yearly_ret_{yyyy}` 列），对照臂逐年差用。
        price_panel / lot_size: 见 :func:`run_once`（整手约束；默认不启用）。
        freq / phase: 调仓频率与相位，见 :func:`rebalance_dates`（缺省月频）。
    """
    ex = execution or ctx.execution
    bt = VectorBacktest(strategy=strategy, rebalance_freq=freq,
                        initial_capital=float(capital), costs=cost)
    res = run_once(bt, ctx, ex, signal=signal, freq=freq, phase=phase,
                   price_panel=price_panel, lot_size=lot_size)
    out = metrics_row(res, capital)
    if yearly:
        dr = res.daily_returns
        for y, sub in dr.groupby(dr.index.year):
            out[f"yearly_ret_{y}"] = float((1 + sub).prod() - 1)
    return out


def evaluate_weights(target_w: pd.DataFrame, ctx: CaliberContext,
                     cost: TransactionCosts,
                     capital: float = DEFAULT_CAPITAL,
                     execution: str | None = None) -> dict:
    """在统一口径下回测**给定权重表**（如 QP / 任意组合层）。

    ⚠️ 必须传 ``adv_panel / vol_panel``：否则 ``ImpactCostModel`` 收不到逐股
    ADV/波动 ⇒ 冲击项恒 0，与等权臂不可比（2026-10-13 实测低估 ~6500bp）。
    ⚠️ 目前仅支持 ``execution="close"``（T+1 口径下 ``PrecomputedWeightsStrategy``
    的按日查表语义需另行定义）；QP 类对照请走 `fair` profile。
    """
    ex = execution or ctx.execution
    if ex != "close":
        raise NotImplementedError(
            f"evaluate_weights 暂不支持 execution={ex!r}（T+1 查表语义未定义）；"
            f"请用 execution='close'（即 `fair` profile）。")
    bt = VectorBacktest(strategy=PrecomputedWeightsStrategy(target_w),
                        rebalance_freq="M", initial_capital=float(capital),
                        costs=cost)
    return metrics_row(run_once(bt, ctx, ex, signal=ctx.fwd), capital)


def get_profile(name: str | None = None) -> Profile:
    """取口径组合（``PROFILES``）；``None`` → :data:`DEFAULT_PROFILE`（生产口径）。"""
    key = name or DEFAULT_PROFILE
    if key not in PROFILES:
        raise KeyError(f"未知口径组合 {key!r}，可选 {sorted(PROFILES)}")
    return PROFILES[key]


def describe(ctx: CaliberContext, cost_kind: str, capital: float | None = None,
             profile: str | None = None) -> str:
    """口径横幅 —— **每条结论都必须带上它**，防跨口径并列。

    2026-10-14 起含 `profile` / 成交价 / 资金三项（此前缺，导致
    `v1w`(fair) 与 `v1-dingban`(production) 的数字被跨口径并列）。
    """
    sig = SIGNALS.get(ctx.signal, {}).get("desc", ctx.signal)
    cap = capital if capital is not None else DEFAULT_CAPITAL
    tag = f"[{profile}] " if profile else ""
    warns = ""
    if ctx.window == "prod2018":
        warns += ("\n🚨 窗口 = 2018+，**含缺掩码年份**（2018 无涨跌停/停牌/ST 状态表，"
                  "按「全部可交易」处理）⇒ 与主口径 prod(2019+) **不同源，绝对数字不可并列**。"
                  "此窗口只用于追溯登记表旧值，勿用于新结论。")
    if ctx.execution == "close":
        warns += ("\n⚠️ 成交价 = close（T 收盘成交，**乐观上限**）—— 比生产 T+1 open "
                  "高约 +0.7pp。")
    if profile == "fair":
        warns += ("\n🚨 fair 口径（impact / close / 1e8）与登记表 production"
                  "（27bp 时变 / T+1 open / 1e6）**不同源**，绝对数字不可并列。")
    return (f"{tag}口径：信号={sig} | 窗口={ctx.window}({ctx.begin}起, {ctx.n_days}日) | "
            f"成本={cost_kind} | 成交价={ctx.execution} | 资金={cap:.0e} | "
            f"Sharpe=生产几何 | 指纹={ctx.fingerprint}{warns}")


# ---------------------------------------------------------------------------
# 6) CLI
# ---------------------------------------------------------------------------
def _registry_path() -> Path:
    return baselines_path()


def load_registry() -> dict:
    """定版登记表（读盘逻辑在 :func:`config.load_baselines`，此处只做门面）。

    2026-11-12：原来这里自带一份 yaml 读取，与 ``config`` 重复；登记表读取入口
    统一收到 ``config.load_baselines``，``strategy.registry`` 也走同一入口
    （它从登记表取构造参数，不再在代码里存第二份数字）。
    """
    return load_baselines()


def _arm_strategies(buffer: str, eqw_topk: str) -> list[tuple[str, object]]:
    from strategy.cross_section import BufferedTopFracLongOnly, TopFracLongOnly
    arms: list[tuple[str, object]] = [("eqw_top10pct", TopFracLongOnly(frac=0.10))]
    for spec in [x for x in buffer.split(",") if x.strip()]:
        a_s, b_s = spec.split(":")
        ent, ext = float(a_s), float(b_s)
        arms.append((f"eqw_buf{ent:g}_{ext:g}", BufferedTopFracLongOnly(ent, ext)))
    return arms


def _check(signal: str, window: str, prof_name: str, cost_kind: str,
           execution: str, capital: float, buffer: str,
           tol_ann: float, tol_sharpe: float,
           baseline_id: str = "", end: int | None = None) -> int:
    """复算登记表中该 (signal, window) 的基线，核对年化/Sharpe。

    ``baseline_id`` 非空时**按 id 精确匹配**，绕过 (signal_key, window) 匹配，
    并**从登记条目自身**取「信号 / 资金 / 整手 / 调仓频率」四件事。
    必要性（2026-11-10）：`yqcore` 三档与既有条目共用 signal_key + window
    （`gbdt_mid13+prod` / `dingban_fundind+prod`），按旧逻辑只会命中列表里更早的
    那条；且三档的资金（1e5 / 1e6）与整手（100 / 无）各不相同，用 `--profile`
    的缺省值复算必然对不上。
    """
    reg = load_registry()
    all_b = reg.get("baselines", [])
    hit: list = []
    spec = None
    if baseline_id:
        from strategy.registry import get_tier

        hit = [b for b in all_b if b.get("id") == baseline_id]
        if not hit:
            print(f"  ⚠️ 登记表无 id={baseline_id!r} 条目，跳过核对")
        try:
            spec = get_tier(baseline_id)
        except KeyError:
            spec = None
        if hit:
            # 登记条目自带的信号键优先（`--signal` 缺省是 ens_h1h5，与 yqcore 不同源）；
            # 有本机代理（`signal_key_local`）时用代理，否则 breadth 会因登记路径缺失报错。
            signal = ((spec.runnable_signal if spec is not None else None)
                      or hit[0].get("signal_key") or signal)
            blk = hit[0].get("caliber", {}) or {}
            if "capital" in blk:
                capital = float(blk["capital"])
            # 2026-11-12：条目可声明 `check_end`（末日对齐日）。yqcore 三档报数统一截到
            # 882 pred 末日 2026-09-07，而 breadth 的本机代理信号到 11-04 ⇒ 不截就把
            # 36 个 2026 交易日混进窗口，复算值必对不上（读数差 ~1.4pp）。
            # 显式 `--end` 优先级更高，便于做窗口敏感性对照。
            if end is None and hit[0].get("check_end"):
                end = int(hit[0]["check_end"])
        ctx = load_context(signal=signal, window=window, execution=execution,
                           end=end)
    else:
        ctx = load_context(signal=signal, window=window, execution=execution,
                           end=end)
        hit = [b for b in all_b
               if b.get("signal_key") == signal and b.get("window") == window]
    cost = cost_profile(cost_kind)
    print(describe(ctx, cost_kind, capital, prof_name))
    registered: dict = {}
    if hit:
        mreg = hit[0].get("metrics", {}) or {}
        registered = ((mreg.get("profiles", {}) or {}).get(prof_name)
                      or mreg.get("arms", {}) or {})
    else:
        print(f"  ⚠️ 登记表无 (signal={signal}, window={window}) 条目，跳过核对")
    rc = 0
    arms = _arm_strategies(buffer, "")
    lot_size = None
    price_panel = None
    freq = "M"
    if baseline_id and spec is not None:
        # 按 id 核对时只跑该档位自己的那一臂（yqcore 的档位不在默认臂里）。
        # 🚨 2026-11-10 修：原实现用 `except KeyError: pass` 包住 get_tier + 参数拼装 +
        # build 三步，而「非 BufferedTopKLongOnly 档位」在拼 tag 时会 KeyError
        # （`p['frac_entry']`）⇒ **静默退回默认的两条 eqw 臂、rc 仍为 0**，
        # 即「口径核对」对 yqcore 三档一直空跑。现在只捕获 get_tier 的 KeyError，
        # 面板注入与构造失败一律上抛。
        kw: dict = {}
        if spec.panel is not None:
            pf = BASE_DIR / f"{spec.panel}.parquet"
            if not pf.exists():
                raise FileNotFoundError(f"档位 {baseline_id} 需要面板 {pf}")
            kw[spec.panel] = (pd.read_parquet(pf)
                              .reindex(index=ctx.pred.index,
                                       columns=ctx.pred.columns))
        blk = (hit[0].get("caliber", {}) or {}) if hit else {}
        if blk.get("lot_size"):
            lot_size = int(blk["lot_size"])
            raw = pd.read_parquet(BASE_DIR / "close_raw.parquet")
            price_panel = (raw.reindex(index=ctx.pred.index,
                                       columns=ctx.pred.columns)
                           .ffill().shift(1))
        freq = ((hit[0] if hit else {}).get("execution", {}) or {}).get(
            "rebalance_freq", "M") or "M"
        tag = (hit[0].get("check_arm") if hit else None) or baseline_id
        arms = [(tag, spec.build(**kw))]
    for tag, strat in arms:
        m = evaluate_strategy(strat, ctx, cost, capital, execution=execution,
                              price_panel=price_panel, lot_size=lot_size,
                              freq=freq)
        line = (f"  {tag:<18s} 年化 {m['ann']*100:6.2f}%  Sharpe {m['sharpe']:.3f}  "
                f"换手 {m['turnover']:.3f}  成本 {m['cost_bp']:.0f}bp  活跃 {m['n_active']:.0f}只")
        print(line)
        exp = registered.get(tag)
        if exp is None and "ann" in registered:
            # 部分条目把 metrics.profiles.production 直接写成「一条臂」的指标
            # （yqcore-focus/breadth），没有臂名这一层 ⇒ 按单臂处理。
            exp = registered
        if exp:
            d_ann = abs(m["ann"] - exp["ann"])
            d_sh = abs(m["sharpe"] - exp["sharpe"])
            ok = d_ann <= tol_ann and d_sh <= tol_sharpe
            note = hit[0].get("check_note") if hit else None
            if note:
                # 已知数据态差异（如 breadth 的本机代理信号 ≠ 登记面板）：
                # 打印差异但不算失败，否则这条守卫永远是红的、等于没有。
                print(f"      ↔ 登记 {exp['ann']*100:.2f}% / {exp['sharpe']:.3f}  "
                      f"Δann={d_ann*100:+.2f}pp Δsharpe={d_sh:+.3f}  ⚠️ 已知差异：{note}")
                continue
            rc |= 0 if ok else 1
            print(f"      ↔ 登记 {exp['ann']*100:.2f}% / {exp['sharpe']:.3f}  "
                  f"Δann={d_ann*100:+.2f}pp Δsharpe={d_sh:+.3f}  "
                  f"{'✅ PASS' if ok else '❌ FAIL'}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal", default="ens_h1h5", choices=sorted(SIGNALS))
    ap.add_argument("--window", default=DEFAULT_WINDOW, choices=sorted(WINDOWS))
    ap.add_argument("--signal-path", default="",
                    help="直接指定信号面板，覆盖 --signal 的登记路径。用途："
                         "同一构造在**不同数据态**下的对照（如重建前后各存一份信号，"
                         "分别跑同一窗口），或未登记的临时面板。"
                         "⚠️ 用此参数时结果**不是**登记信号，不得当作条目值")
    ap.add_argument("--profile", default=DEFAULT_PROFILE,
                    choices=sorted(PROFILES) + ["custom"],
                    help="口径组合：production（生产口径，默认）/ fair（研究压力）/ custom")
    ap.add_argument("--cost", default=None, choices=["production", "impact", "flat"],
                    help="覆盖 profile 的成本口径（给定即转 custom）")
    ap.add_argument("--execution", default=None, choices=list(EXECUTIONS),
                    help="覆盖 profile 的成交价（给定即转 custom）")
    ap.add_argument("--capital", type=float, default=None,
                    help="覆盖 profile 的资金规模（给定即转 custom）")
    ap.add_argument("--buffer", default="0.15:0.40",
                    help="附加的 buffer 等权臂（逗号分隔，如 `0.15:0.40,0.20:0.50`）")
    ap.add_argument("--check", action="store_true",
                    help="核对 config/baselines.yaml 的登记值")
    ap.add_argument("--baseline-id", default="",
                    help="配合 --check：按 id 精确匹配登记条目（同 signal+window 有多条"
                         "并列档位时必须用它，如 yqcore-breadth / yqcore-focus）")
    ap.add_argument("--end", type=int, default=None,
                    help="窗口末日 YYYYMMDD（覆盖 WINDOWS 上界）。用于让两条 pred 末日"
                         "不同的信号**对齐到同一天**——不对齐时长短差会被读成业绩差"
                         "（yqcore 两档固定截 20260907）")
    ap.add_argument("--tol-ann", type=float, default=0.005)
    ap.add_argument("--tol-sharpe", type=float, default=0.02)
    ap.add_argument("--json", default="", help="把结果另存为 JSON（相对 ROOT）")
    args = ap.parse_args()

    overridden = (args.cost is not None or args.execution is not None
                  or args.capital is not None)
    if args.profile == "custom" or overridden:
        prof_name = "custom"
        cost_kind = args.cost or "impact"
        execution = args.execution or "close"
        capital = args.capital if args.capital is not None else DEFAULT_CAPITAL
        note = "手工覆盖口径（未登记，结论须自带横幅）"
    else:
        p = get_profile(args.profile)
        prof_name, cost_kind, execution, capital, note = (
            p.name, p.cost, p.execution, p.capital, p.note)

    ctx = load_context(signal=args.signal, window=args.window, execution=execution,
                       pred_path=args.signal_path or None, end=args.end)
    cost = cost_profile(cost_kind)
    print(describe(ctx, cost_kind, capital, prof_name))
    if args.signal_path:
        print(f"  ⚠️ 信号面板被 --signal-path 覆盖：{args.signal_path}"
              "（非同登记面板，结果仅作数据态对照）")
    print(f"  profile 说明：{note}")
    if args.check:
        # 早退：--check 自己会复算目标臂，别再按 --buffer 跑一遍（原来跑两遍，耗时翻倍）
        return _check(args.signal, args.window, prof_name, cost_kind, execution,
                      capital, args.buffer, args.tol_ann, args.tol_sharpe,
                      args.baseline_id, args.end)
    out = {}
    for tag, strat in _arm_strategies(args.buffer, ""):
        m = evaluate_strategy(strat, ctx, cost, capital, execution=execution)
        out[tag] = m
        print(f"  {tag:<18s} 年化 {m['ann']*100:6.2f}%  Sharpe {m['sharpe']:.3f}  "
              f"换手 {m['turnover']:.3f}  成本 {m['cost_bp']:.0f}bp  活跃 {m['n_active']:.0f}只")
    if args.json:
        p = ROOT / args.json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"context": {"signal": ctx.signal, "window": ctx.window,
                         "begin": ctx.begin, "n_days": ctx.n_days,
                         "fingerprint": ctx.fingerprint, "profile": prof_name,
                         "cost": cost_kind, "execution": execution,
                         "capital": capital},
             "arms": out}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"产物: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
