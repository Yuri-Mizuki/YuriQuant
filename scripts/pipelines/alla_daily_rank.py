"""全A滚动模型每日选股排名 —— 主实验信号形态（gbdt h1+h5 秩平均 × M raw Top10%）的生产化推理。

实验背景（reports/alla_rolling、reports/alla_rolling_ortho、reports/prod_pipeline_gap）：
- 主实验（ortho 臂）口径 = 全A + 因子层正交化 + 每年特征选择 +
  **gbdt h1+h5 截面秩平均** + 500 日滚动训练窗 + raw Top10% 等权、月频调仓，
  2018–2026 OOS 年化 15.52% / 超额上证 +13.29% / Sharpe 0.631。
- 本脚本对齐其中**不依赖面板层**的两项（2026-09-17 实测 +3.06pp 缺口拆解）：
  ① h1+h5 秩平均（**+1.28pp/年**，且降换手 78%→71%）；② 硬剔除 `limit_pos`。
  因子层正交化（+1.77pp）**已实现、2026-09-17 起为默认**（`--preproc ortho`）：在实时
  面板上跑同一套 `factor.preprocessing::preprocess_factor`（MAD → 行业哑变量 +
  log 市值中性化 → zscore → ±10），并自动切到 ortho 臂特征清单；**不依赖离线
  `panels_neu`**（那份落后生产链 11 个交易日、且不含 09-12 接入的另类族）。
  ✅ 口径一致性已实测（2026-09-17，见 reports/prod_pipeline_gap/align_check_20260917.md）：
  ① 市值口径差异（实时 vs `_base`）对**全部 85 个在产特征**的面板影响
     秩相关 median 0.999936 / min 0.9517（唯一显著异常为 `ln_mktcap` 本身）；
  ② 出榜链 ortho 榜与主实验 ortho 臂 pred 末日非 NaN 数及板块构成吻合
     （5195 vs 5196；两者 BJ 均为 0）。
  ⚠️ 边界：行业面板 `cov_industry` 不覆盖北交所 → 中性化时 BJ 股被整体排除，
  ortho 口径下北交所（337 只）不出现在榜上 —— **与主实验 ortho 臂一致**（非本链特有）。
  ⚠️ 缺口分解（report.md §2）的 2×2 收益数字仍基于离线 `panels_neu` 回测；
  本链是"同流程、不同数据源"，面板等价性已实测但**年化未重跑**。

本脚本每天做的事（与实验同一代码路径，无新增调参）：

  1. 数据增量更新（scripts.ingest.update_data --pool all_a，--skip-update 跳过）；
  2. 尾部面板重算：最近 (训练窗 + 260 日预热 + 缓冲) 个交易日的后复权 OHLCV/vwap；
  3. 只重算当年入选的 ~50 个因子（量价走 factor.alphaXXX 注册表；基本面/风格/
     股东/质押/构造/事件/SUE/两融/资金流/折价动态/停牌状态等族复用
     scripts.builders 的 build_panels 入口），截面 zscore + ±10 剪裁（与
     FeatureStore 同口径）；h1/h5 清单取**并集只算一次**再切分；
  4. 每个 horizon 各自重训 gbdt（500 日窗）并预测最新截面，再逐股取截面百分位
     秩平均（`_rank_average_single_day`，与实验 `_rank_average` 同口径）；
  5. 幽灵股守卫（existence_mask，逐 horizon）+ 信号日可交易性标注（停牌/ST/封板）；
  6. 输出全A排名 CSV + Top10% 持仓候选 + history 追加（reports/alla_daily/）。

口径披露（与实验的差异，均为如实可知的边界）：
- **尾部重算而非全历史拼接**：后复权因子随分红除权漂移，全历史拼接会产生复权
  基准断层。本脚本对训练窗与预测截面用同一次重算（同一复权基准），内部自洽；
  不改写冻结的 all_a_2026 实验数据集。
- **面板层默认已正交化**（`--preproc ortho`，2026-09-17 起）：特征走实时重算面板 +
  `preprocess_factor`（MAD → 行业+log市值中性化 → zscore → ±10），**与离线
  `panels_neu` 同流程、非同数据源**；实测面板等价（85 特征秩相关 median 0.999936）。
  传 `--preproc zscore` 回到原始面板 + zscore（旧生产口径，2026-09-16 前默认）。
- **北交所不出现在 ortho 榜上**：`cov_industry` 无 BJ 行业分类 → 中性化整体排除
  （实测：主实验 ortho 臂 pred 末日 BJ 计数同样为 0）。需要 BJ 时用
  `--preproc zscore`（该口径 BJ 保留，337 只有分）。
- **训练段用发布时点已知的全部标签**（最后有效标签日 = 预测日前一交易日）。
  实验的 embargo 是回测防污染隔离；实时预测不存在窥探未来，故取到最新。
- **可交易性是信号日状态估计**：回测掩码用 T+1 状态（T+1 成交口径），实时排名
  发布时 T+1 未知，改用 T 日停牌/ST/收盘封板标注，T+1 一字板仍可能买不进。
- **训练标签默认不含可交易掩码**（`--tradable-labels` 可开，2026-09-17 接入）：
  默认与主实验同口径（全样本前瞻收益截面 rank）。开启后用
  `data.tradability::build_tradable_mask` 把"买不进的样本"标签置 NaN，等价于
  从训练集剔除 —— 这是 P0 报告 §六 P1-a 的治本项（纸面关系不该进损失函数）。
  默认关闭的理由：它改变模型训练集，属口径变更而非缺陷修复，须先对照实验。
- **`limit_pos` 已被硬剔除**（`EXCLUDE_FEATURES`）：该特征收益 100% 来自次日买
  不进的封板股（可交易口径正向选股 −55.6%/年），生产形态剔除后 4/4 格全项改善。
- 跨年无当年选择文件时回退最近年份并告警（可重跑 rolling_grid_alla --stage select）。

用法:
    python scripts/pipelines/alla_daily_rank.py                     # 全流程（含数据更新）
    python scripts/pipelines/alla_daily_rank.py --skip-update       # 离线/数据已更新
    python scripts/pipelines/alla_daily_rank.py --date 20260904     # 指定预测日（默认最新）
    python scripts/pipelines/alla_daily_rank.py --horizons 1        # 回退 h1 单模型口径
    python scripts/pipelines/alla_daily_rank.py --exclude-features ""  # 关闭硬剔除
    python scripts/pipelines/alla_daily_rank.py --out-tag _h1h5     # 输出到旁路目录
    python scripts/pipelines/alla_daily_rank.py --preproc ortho --out-tag _ortho  # 正交化口径
    python scripts/pipelines/alla_daily_rank.py --tradable-labels --out-tag _tl    # 标签掩码口径
    python scripts/pipelines/alla_daily_rank.py --window 750        # gbdt_w750 变体
    python scripts/pipelines/alla_daily_rank.py --install-task 17:30   # 注册每日计划任务
    python scripts/pipelines/alla_daily_rank.py --remove-task
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

if TYPE_CHECKING:  # 仅用于类型标注；运行时在 train_and_predict 内延迟导入
    from model.predictor import LGBMPredictor

log = setup_logging("alla_daily_rank")

# ---------------------------------------------------------------------------
# 配置（参数与 rolling_grid_alla 实验单一真源对齐）
# ---------------------------------------------------------------------------
HORIZON = 1                    # 单模型臂口径（--horizons 1）；也是 tail_n_days 的视野项
# 集成臂口径：gbdt h1 + h5 截面秩平均（主实验信号形态 `ens_h1h5`）。
# 实验实测（reports/prod_pipeline_gap/report.md）：zscore 臂内 h1+h5 比 h1 单模型
# +1.28pp/年，且降换手（78%→71%）；信号秩相关仅 0.828，是两套互补信号。
HORIZONS: tuple[int, ...] = (1, 5)
# 硬剔除的特征（生产 selection 层面，先于四条入选路径）；空元组 = 不剔除。
# `limit_pos`（封板位置）实测：可交易口径 IC 为负、正向选股全池 +71.1%/年 vs
# 池内 −55.6%/年（纸面缺口 +126.7pp，reports/limit_pos_tradable_p0/report.md），
# 且它的收益 100% 来自买不到的封板股。生产形态（h1 单模型）剔除后 4/4 格
# 全项改善（+0.3~1.2pp，reports/limit_pos_ablation_p1b/report.md）。
EXCLUDE_FEATURES: tuple[str, ...] = ("limit_pos",)
# 训练标签口径（2026-09-17 接入，P0 报告 §六 P1-a 的治本项）：
#   False（默认）= 标签 = 全样本前瞻收益截面 rank —— 现行口径，零回归；
#   True         = 标签掩掉"买不进的样本"（``data.tradability::build_tradable_mask``
#                  的 T+1 成交口径掩码置 NaN，等价于从训练集剔除）。
# 为什么：无掩码时标签把"T 日封涨停 → T+1 继续封板"的**买不进**收益当监督信号
# （reports/limit_pos_tradable_p0/report.md）。掩码在尾部面板上实时构建 ——
# ``tail`` 已含 ``bwd``（后复权因子）与 ``close_adj``，无需离线面板。
DEFAULT_TRADABLE_LABELS = False
DEFAULT_WINDOW = 500           # gbdt 滚动训练窗（--window 750 = gbdt_w750 变体）
MIN_TRAIN = 250                # 最少训练日（同实验）
WARMUP = 260                   # alpha 公式最大回看 250 日 + 缓冲
TAIL_BUFFER = 40               # 停牌缺口/月末等额外缓冲
DEFAULT_FRAC = 0.10            # Top10% 持仓候选口径
# 面板预处理口径（`--preproc`）：
#   `ortho`（默认，2026-09-17 起）= 因子层正交化：MAD 去极值 → 行业哑变量 +
#               log 市值中性化 → zscore → ±10，与主实验 `panels_neu` 同流程
#               （factor.preprocessing::preprocess_factor），自动配 ortho 臂特征清单；
#   `zscore` = 原始面板 + 截面 zscore + ±10 剪裁（2026-09-16 前默认，同 `panels`）。
DEFAULT_PREPROC = "ortho"
SELECTION_DIR = ROOT / "reports" / "alla_rolling" / "selection"
# ortho 臂的特征清单（按 `panels_neu` 的质量窗 IC 选）—— 必须与 `--preproc ortho` 同用，
# 否则「清单按中性化面板选、输入却是原始面板」口径错配（reports/prod_pipeline_gap §5.3）。
ORTHO_SELECTION_DIR = ROOT / "reports" / "alla_rolling_ortho" / "selection"
OUT_DIR = ROOT / "reports" / "alla_daily"
TASK_NAME = "YuriQuant AllaDailyRank"
# 计划任务用的 Python（沿用 monitor_performance 约定：YQ_SYSTEM_PY 可配置）
SYSTEM_PY = Path(os.environ.get("YQ_SYSTEM_PY") or sys.executable)

_ALPHA_PREFIXES = ("alpha101_", "alpha158_", "alpha191_", "alpha360_")
# 股东族因子名（scripts/builders/build_alla_holder_factors 的两组输出键）
_HOLDER_NUM_KEYS = {"holder_num_chg", "holder_num_yoy"}
_HOLDER_TOP_KEYS = {"top1_holding", "top5_holding", "top10_holding",
                    "top10_hhi", "inst_holding"}
# 商誉/质押/业绩预告快报族（build_alla_pledge_factors 输出键）
_PLEDGE_KEYS = {"goodwill_ratio", "pledge_ratio", "pledge_holder_ratio",
                "frozen_ratio", "profit_notice_chg", "profit_express_np_yoy",
                "profit_express_rev_yoy"}
# 构造型基本面族（build_alla_constructed_factors.build_panels 输出键）
_CONSTRUCTED_KEYS = {"np_ded_ratio", "main_profit_ratio", "ebit_margin",
                     "altman_zscore", "debt_to_ebitda", "interest_coverage",
                     "np_rev_gap", "rev_recv_gap", "cfo_to_np",
                     "margin_delta_ttm", "np_accel_sq", "roe_vol_pit",
                     "risky_asset_ratio", "div_growth_yoy",
                     "div_consecutive_years"}

# ---------------------------------------------------------------------------
# 另类数据族输出键（2026-09-12 接入日频主链）
# 每组 = 对应 builders 的 build_panels() 产出键集合，是 compute_features 分派
# 白名单的单一真源：构建器新增因子必须同步登记，否则分派直接报「未知来源」。
# 未登记因子不再被兜底塞进基本面路径（旧行为会在深层 KeyError，指向性差）。
# ---------------------------------------------------------------------------
# 风格族（build_alla_style_factors.build_panels）
_STYLE_KEYS = {"rd_exp_ratio_ttm", "current_ratio", "cfo_to_rev_ttm",
               "rd_to_assets_ttm", "selling_exp_ratio_ttm",
               "admin_exp_ratio_ttm", "debt_to_assets", "noncur_liab_ratio",
               "quick_ratio", "cash_ratio", "ebitda_margin_ttm",
               "op_income_margin_ttm", "goodwill_to_equity",
               "tangible_asset_ratio", "intangible_ratio",
               "cfo_to_assets_ttm", "cfo_to_debt_ttm"}
# 事件族（build_alla_event_factors.build_panels）
_EVENT_KEYS = {"notice_sue", "notice_profit_ttm", "notice_forecast_hit",
               "express_profit_yoy", "express_rev_yoy", "express_reporting",
               "unlock_ratio_20d"}
# SUE 意外 + 质押深度（build_alla_sue_pledge_factors.build_panels）
_SUE_PLEDGE_KEYS = {"sue_notice_cs", "sue_notice_20d", "sue_express_cs",
                    "sue_express_20d", "pledge_chg_20d", "pledge_frn_density",
                    "pledge_holder_density"}
# 两融族（build_alla_margin_factors.build_panels）
_MARGIN_KEYS = {"margin_bal_chg_5d", "margin_bal_chg_20d",
                "financing_chg_1d", "securities_chg_1d"}
# 龙虎榜/大宗额量族（build_alla_moneyflow_factors.build_panels）
_MONEYFLOW_KEYS = {"lhb_net_buy", "lhb_count_20d", "lhb_net_20d",
                   "block_amt_20d", "block_vol_20d"}
# 大宗折溢价 + 股东动态（build_alla_disc_holder_dyn.build_panels）
_DISC_HOLDER_DYN_KEYS = {"block_disc_1d", "block_disc_20d", "block_prem_20d",
                         "top10_hold_chg", "holder_stability"}
# 停牌/ST 状态族（build_alla_status_factors.build_panels）
_STATUS_KEYS = {"suspend_ratio_60d", "suspend_count_60d", "is_st", "st_days",
                "limit_pos"}

# ---------------------------------------------------------------------------
# 因子释义（选股解释用：特征名 → (中文名, 释义)）
# ---------------------------------------------------------------------------
_FUND_VALUE_KEYS = {"bp", "sp_ttm", "div_yield", "ep_ttm"}

FACTOR_GLOSSARY: dict[str, tuple[str, str]] = {
    "bp": ("账面市值比", "市净率倒数，估值因子；值越高，股价相对每股净资产越便宜"),
    "sp_ttm": ("市销率倒数", "TTM 营收/市值，估值因子；值越高，营收相对市值越厚实"),
    "div_yield": ("股息率", "TTM 每股分红/股价，红利收益因子"),
    "ep_ttm": ("盈利收益率", "TTM 净利/市值（PE 倒数），估值因子；值越高越便宜"),
    "ln_mktcap": ("对数市值", "总市值取对数，规模因子"),
    "holder_num_yoy": ("股东户数同比", "股东户数较去年同期变化，筹码集中度（下降=筹码集中）"),
    "holder_num_chg": ("股东户数环比", "股东户数较上期变化，筹码集中度（下降=筹码集中）"),
    "altman_zscore": ("Altman Z 分数", "破产风险评分，越高财务越健康"),
    "alpha158_KLEN": ("K线长度", "(最高-最低)/开盘，日内振幅相对开盘价，波动/情绪"),
    "alpha158_VMA60": ("60日量能水平", "60日均量/最新成交量，>1 表示近期缩量"),
    "alpha158_QTLD60": ("60日低位分位", "60日20%分位价/最新收盘价，价格处于历史低位区"),
    "alpha158_MIN10": ("10日支撑", "10日最低价/最新收盘价，短期超跌/支撑位"),
    "alpha158_CORR10": ("10日量价相关", "10日收盘价与对数成交量相关性，量价配合度"),
    "alpha158_QTLD30": ("30日低位分位", "30日20%分位价/最新收盘价"),
    "alpha158_STD5": ("5日波动率", "5日收盘价标准差/最新收盘价"),
    "alpha158_CORD10": ("10日量价变动相关", "10日价格变动与量变动相关性，量价配合变化"),
    "alpha158_STD10": ("10日波动率", "10日收盘价标准差/最新收盘价"),
    "alpha158_VSUMN60": ("60日量缩占比", "60日中成交量减少天数占比，缩量程度"),
    "alpha158_MIN5": ("5日支撑", "5日最低价/最新收盘价，超短支撑位"),
    "alpha101_040": ("高价波动×量价相关", "-rank(10日高价标准差)×10日量价相关，WorldQuant #040"),
    "alpha191_041": ("VWAP 短期加速反向", "-rank(5日窗口内 VWAP 3日变动的最大值)，#041"),
    "alpha191_070": ("6日成交额波动", "6日成交额标准差，短期资金活跃度，#070"),
    "alpha191_095": ("20日成交额波动", "20日成交额标准差，资金活跃度，#095"),
    "alpha191_118": ("多空实体强度", "20日阳线实体和/阴线实体和，多头占优程度，#118"),
    "alpha191_159": ("多尺度区间位置", "6/12/24日高低区间位置加权复合，RSV 类趋势位置，#159"),
    "alpha191_167": ("12日上行动量", "12日上涨日涨幅累和，短期动量，#167"),
    # —— 状态族（build_alla_status_factors）——
    "limit_pos": ("封板位置", "（收盘-前收）/（涨停价-前收）：1.0=收盘封涨停，"
                             "0=跌停，0.5=区间中部；打板热度/连板惯性。"
                             "⚠ 收益主要来自次日跳空（T 日收盘已封板、纸面口径买入"
                             "不可成交），且极偏态易被截面 zscore 推到极端叶子，"
                             "分数不宜按绝对值解读"),
    "st_days": ("连续ST天数", "当前已连续处于 ST 状态的天数，治理恶化持续时间"),
    # —— 事件族（build_alla_event_factors / build_alla_sue_pledge_factors）——
    "notice_forecast_hit": ("预告相对上年净利同比",
                            "预告净利润上限/上年同期净利 - 1，业绩预告同比增速"),
    "sue_notice_20d": ("业绩预告SUE 20日累积",
                       "过去 20 交易日标准化预期外盈余（SUE）累积，预告窗口脉冲"),
    # —— 资金族（build_alla_moneyflow_factors）——
    "lhb_count_20d": ("龙虎榜20日上榜次数",
                      "过去 20 交易日龙虎榜上榜次数，事件热度"),
    # —— 两融族（build_alla_margin_factors）——
    "margin_bal_chg_20d": ("两融余额20日变化率",
                           "融资融券余额 20 日变化率，中期杠杆趋势"),
}
# ⚠ 本表是**报告层副本**：真源在各 builder 的中文标签（如 build_alla_*.
#   *_ROUTES / build_panels docstring）与 factor/alpha*.py 的 docstring。
#   新增被选中的因子须同步补表，守卫见
#   tests/test_alla_daily_rank.py::test_glossary_covers_selection_2026。


def lookup_glossary(name: str) -> tuple[str, str]:
    """特征名 → (中文名, 释义)；alpha360 量能/最低价族按参数通配。"""
    if name in FACTOR_GLOSSARY:
        return FACTOR_GLOSSARY[name]
    m = re.fullmatch(r"alpha360_VOLUME(\d+)", name)
    if m:
        return (f"{m.group(1)}日量能回溯",
                f"{m.group(1)}日前成交量/最新成交量，量能回溯比；>1 表示较"
                f"{m.group(1)}日前缩量")
    m = re.fullmatch(r"alpha360_LOW(\d+)", name)
    if m:
        return (f"{m.group(1)}日最低价回溯",
                f"{m.group(1)}日前最低价/最新收盘价")
    m = re.fullmatch(r"alpha360_(OPEN|HIGH|CLOSE|VWAP)(\d+)", name)
    if m:
        field = {"OPEN": "开盘价", "HIGH": "最高价",
                 "CLOSE": "收盘价", "VWAP": "VWAP 均价"}[m.group(1)]
        return (f"{m.group(1)}日{field}回溯",
                f"{m.group(1)}日前{field}/最新收盘价")
    # Alpha158 价格相对族（factor/alpha158.py 的 [OPEN/HIGH/LOW/VWAP]0）
    m = re.fullmatch(r"alpha158_(OPEN|HIGH|LOW|VWAP)(\d+)", name)
    if m:
        field = {"OPEN": "开盘价", "HIGH": "最高价",
                 "LOW": "最低价", "VWAP": "VWAP 均价"}[m.group(1)]
        return (f"价格相对·{field}", f"{field}/收盘价（Alpha158 价格相对因子）")
    return (name, "未收录释义")


def factor_family(name: str) -> str:
    """因子族：估值/规模/筹码/财务质量/量价。"""
    if name.startswith(_ALPHA_PREFIXES):
        return "量价因子"
    if name in _FUND_VALUE_KEYS:
        return "估值因子"
    if name == "ln_mktcap":
        return "规模因子"
    if name.startswith("holder_"):
        return "筹码因子"
    if name == "altman_zscore":
        return "财务质量"
    return "其他"


# ---------------------------------------------------------------------------
# 纯函数（单测覆盖）
# ---------------------------------------------------------------------------
def tail_n_days(window: int) -> int:
    """尾部加载交易日数：训练窗 + 因子预热 + 标签视野 + 缓冲。"""
    return window + WARMUP + HORIZON + TAIL_BUFFER


# 面板变换钩子：默认 None = 旧行为（zscore + ±10）。`--preproc ortho` 时由 run()
# 换成 `make_ortho_transform(...)` 的闭包 —— 9 条特征构建路径的 `preprocess_panel`
# 调用点无需逐个改（分支多必漏，见 reports/prod_pipeline_gap §4）。
_PANEL_TRANSFORM = None  # Callable[[pd.DataFrame], pd.DataFrame] | None


def set_panel_transform(fn) -> None:
    """设置/清除面板变换（None = 恢复默认 zscore 口径）。零回归：不设置即旧行为。"""
    global _PANEL_TRANSFORM
    _PANEL_TRANSFORM = fn


def make_ortho_transform(market_cap_panel: pd.DataFrame,
                         industry_panel: pd.DataFrame | None):
    """构造因子层正交化变换（与主实验 `panels_neu` 同流程）。

    逐日截面：MAD 去极值 → 行业哑变量 + log 市值中性化（取残差）→ zscore → ±10 剪裁。
    与 ``build_alla_factor_neutralized::_process_one`` 逐字同口径，差别仅在市值/行业
    面板来源（出榜链实时构造 vs 离线 `_base`，见 reports/prod_pipeline_gap §5.5）。
    """
    from factor.preprocessing import preprocess_factor

    def _transform(p: pd.DataFrame) -> pd.DataFrame:
        p = p.astype(np.float32).replace([np.inf, -np.inf], np.nan)
        # 协变量必须与因子面板**逐列对齐**：neutralize 按行取 panel.loc[d] 与
        # market_cap_panel.loc[d] 做布尔掩码，列不一致会 IndexingError ——
        # 2026-09-17 实测踩到（_base 市值面板 5801 列 vs 因子面板 5683 列）。
        # 与 build_alla_factor_neutralized::_process_one 的 reindex 同口径。
        idx = p.index.intersection(market_cap_panel.index)
        mc = market_cap_panel.reindex(index=idx, columns=p.columns)
        ind = (industry_panel.reindex(index=idx, columns=p.columns)
               if industry_panel is not None else None)
        x = preprocess_factor(p.reindex(index=idx), market_cap_panel=mc,
                              industry_panel=ind)
        x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        return x.clip(-10.0, 10.0)

    return _transform


def preprocess_panel(p: pd.DataFrame) -> pd.DataFrame:
    """因子面板统一口径：float32 → inf→NaN → 截面 zscore → ±10 剪裁。

    与 FeatureStore 读取冻结面板时的变换一致（astype 先于 replace，防 float64
    巨值溢出成 float32 inf 后漏杀，见 build_alla_alpha_panels 2026-09-01 教训）。
    ``--preproc ortho`` 下由 `_PANEL_TRANSFORM` 换成正交化，签名与调用点不变。
    """
    if _PANEL_TRANSFORM is not None:
        return _PANEL_TRANSFORM(p)
    from factor.preprocessing import standardize_zscore
    p = p.astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return standardize_zscore(p).clip(-10.0, 10.0).astype(np.float32)


def load_selection(predict_date: pd.Timestamp,
                   sel_dir: Path | None = None,
                   horizon: int = HORIZON,
                   exclude: tuple[str, ...] = ()) -> tuple[list[str], int]:
    """当年 `y{year}__h{horizon}.json` 特征选择；跨年缺失时回退最近年份（告警）。

    ``exclude`` 在**读入后硬剔除**（不走实验侧 `select_features_for_year(exclude=)`，
    即不做"保留席位轮转回填"）—— 生产只消费落盘清单，剔除后特征数会少于 50。
    h1/h5 各有独立清单（实验口径：每个 horizon 用各自的质量窗 IC 选出的特征）。
    """
    d = sel_dir or SELECTION_DIR
    p = d / f"y{predict_date.year}__h{horizon}.json"
    if p.exists():
        names, sel_year = json.loads(p.read_text(encoding="utf-8")), predict_date.year
    else:
        cands = sorted(d.glob(f"y*__h{horizon}.json"))
        if not cands:
            raise FileNotFoundError(
                f"{d} 下无任何 y*__h{horizon}.json 选择文件")
        p = cands[-1]
        sel_year = int(p.name[1:5])
        log.warning("y%d__h%d.json 不存在，回退最近年份 %d 的特征选择"
                    "（如需当年口径：python scripts/pipelines/"
                    "rolling_grid_alla.py --stage select）",
                    predict_date.year, horizon, sel_year)
        names = json.loads(p.read_text(encoding="utf-8"))

    if exclude:
        dropped = [n for n in names if n in set(exclude)]
        if dropped:
            names = [n for n in names if n not in set(exclude)]
            log.info("h%d 特征硬剔除 %s → %d 个（原 %d 个）",
                     horizon, dropped, len(names), len(names) + len(dropped))
        # 剔除名单里的名字若不在清单中，视为空操作（不告警：跨 horizon 名单本就不同）
    if not names:
        raise ValueError(f"h{horizon} 特征清单为空（剔除后）")
    return names, sel_year


def _limit_from_daily(close_raw: pd.DataFrame, predict_date: pd.Timestamp,
                      cache_root: str | None = None):
    """状态表缺失时用日线行情估算涨跌停价（降级守卫）。

    规则：北交所 ±30%、创业板/科创板 ±20%、主板 ±10%（ST 无法识别，按板块估）。
    停牌：当日成交量为 0/NaN。返回 (high_limited, low_limited, suspended)。
    行情也无预测日时返回 None（调用方无法判断则放行）。
    """
    from config import Config

    root = Path(cache_root or str(Config.cache()["root"]))
    daily = pd.read_parquet(root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(
        daily.index.levels[0].normalize(), level=0)
    dates = daily.index.get_level_values(0).unique().sort_values()
    if predict_date not in dates:
        return None
    codes = close_raw.columns
    prev_dates = dates[dates < predict_date]
    if len(prev_dates) == 0:
        return None
    cur = daily.xs(predict_date, level="date").reindex(codes)
    pre = daily.xs(prev_dates[-1], level="date").reindex(codes)
    pct = pd.Series(
        [0.30 if c.startswith(("8", "4", "92"))
         else 0.20 if c.startswith(("300", "301", "688"))
         else 0.10 for c in codes], index=codes)
    pre_close = pre["close"].astype(float)
    hi = (pre_close * (1 + pct)).round(2)
    lo = (pre_close * (1 - pct)).round(2)
    suspended = cur["volume"].fillna(0) <= 0
    return hi, lo, suspended


def signal_day_tradable(close_raw: pd.DataFrame, predict_date: pd.Timestamp,
                        cache_root: str | None = None) -> pd.Series:
    """信号日（T 日）已知状态的可交易性估计：非停牌/非ST/收盘未封板。

    区别于回测用的 ``data.tradability.build_tradable_mask``（行 T 取 T+1 状态，
    「T 日信号 → T+1 成交」口径）：实时排名
    发布时 T+1 未知，只能用 T 日状态做保守标注——T 日收盘封板的股票 T+1 大概率
    一字板买不进；T+1 才停牌/封板的极端情形无法预知，如实披露。

    状态表（history_stock_status）缺失预测日时**降级用日线行情推断**涨跌停
    （2026-09-08 实测：状态表停在 09-02 时此前版本全部放行，一字板漏标进 picks）。
    """
    from config import Config

    codes = close_raw.columns
    p = Path(cache_root or str(Config.cache()["root"])) / "history_stock_status.parquet"
    if not p.exists():
        return pd.Series(True, index=codes, dtype=bool)
    st = pd.read_parquet(p)
    st = st[st.index.get_level_values("code").isin(codes)]
    if st.empty:
        return pd.Series(True, index=codes, dtype=bool)
    try:
        st_t = st.xs(predict_date, level="date")
    except KeyError:
        log.warning("状态表无 %s（最新 %s），降级用日线行情推断涨跌停",
                    predict_date.date(),
                    st.index.get_level_values("date").max().date())
        est = _limit_from_daily(close_raw, predict_date, cache_root)
        if est is None:
            return pd.Series(True, index=codes, dtype=bool)
        hi, lo, susp = est
        raw = pd.to_numeric(close_raw.loc[predict_date],
                            errors="coerce").reindex(codes)
        tol = 1e-4
        sealed_up = (raw >= hi * (1 - tol)) & hi.notna()
        sealed_dn = (raw <= lo * (1 + tol)) & lo.notna()
        bad = susp | sealed_up | sealed_dn
        return (~bad).fillna(True).astype(bool)

    def _row(col: str, default=None):
        if col not in st_t.columns:
            return pd.Series(default, index=codes)
        return pd.to_numeric(st_t[col], errors="coerce").reindex(codes)

    susp = _row("is_suspended").fillna(False).astype(bool)
    is_st = _row("is_st").fillna(False).astype(bool)
    hi, lo = _row("high_limited"), _row("low_limited")
    raw = pd.to_numeric(close_raw.loc[predict_date], errors="coerce").reindex(codes)
    tol = 1e-6
    sealed_up = (raw >= hi * (1 - tol)) & hi.notna()
    sealed_dn = (raw <= lo * (1 + tol)) & lo.notna()
    bad = susp | is_st | sealed_up | sealed_dn
    return (~bad).fillna(True).astype(bool)


def load_stock_names(codes: pd.Index, cache_root: str | None = None) -> pd.Series:
    """股票中文名：本地 code_info.parquet 优先，缺失时在线拉一次（顺手落盘）。

    code_info 是"每日最新"全表覆盖缓存（DataCache.get_code_info），股票改名
    极少、退市股保留旧名，故读到稍旧的本地表也够用；两处都失败时返回空名
    （排名照常产出，name 列为空并告警）。
    """
    from config import Config

    root = Path(cache_root or str(Config.cache()["root"]))
    p = root / "code_info.parquet"
    df = pd.read_parquet(p) if p.exists() else None
    if (df is None or df.empty) and p.exists() is False:
        try:
            from data.cache import DataCache
            from data.datasource import create_datasource
            log.info("code_info 缓存缺失，在线拉取证券名称 ...")
            df = DataCache(create_datasource()).get_code_info()
        except Exception as e:  # noqa: BLE001
            log.warning("股票名称不可用（在线拉取失败）: %s", str(e)[:80])
    if df is None or df.empty or "symbol" not in getattr(df, "columns", []):
        return pd.Series("", index=codes, dtype=object)
    names = df["symbol"].astype(str)
    names.index = df.index.astype(str)
    return names.reindex(codes).fillna("")


def load_industry_names(cache_root: str | None = None, level: int = 1) -> pd.Series:
    """行业代码 → 申万行业中文名（industry_classification 事件表自带名称列）。"""
    from config import Config

    p = Path(cache_root or str(Config.cache()["root"])) / \
        f"industry_classification_level{level}.parquet"
    if not p.exists():
        return pd.Series(dtype=object)
    df = pd.read_parquet(p)
    if "industry_code" not in df.columns or "industry_name" not in df.columns:
        return pd.Series(dtype=object)
    return (df.drop_duplicates("industry_code")
              .set_index("industry_code")["industry_name"].astype(str))


def industry_series(panel: pd.DataFrame | None, predict_date: pd.Timestamp,
                    name_map: pd.Series,
                    panel_l2: pd.DataFrame | None = None,
                    name_map_l2: pd.Series | None = None) -> pd.DataFrame:
    """行业归属面板预测日行 → 每股行业标注（两列：二级 + 一级）。

    panel: (date × code) → industry_code（load_tail 的 industry 键，申万一级）；
    name_map: industry_code → 中文名（load_industry_names，可为空）；
    panel_l2 / name_map_l2: 申万二级同构输入（可选）。
    返回 DataFrame(index=code)，列：
      - ``industry_l2``: 申万二级行业中文名（无二级输入/该股缺失 → 空串）；
      - ``industry_l1``: 申万一级行业中文名（缺名表时回退行业代码）。
    无一级归属的股票不在返回中（下游 reindex 后落为 NaN → "未知"）。
    """
    if panel is None or panel.empty or predict_date not in panel.index:
        return pd.DataFrame()
    codes = panel.loc[predict_date].dropna().astype(str)
    if name_map is None or name_map.empty:
        l1 = codes
    else:
        l1 = codes.map(name_map.astype(str)).fillna(codes)
    out = pd.DataFrame({"industry_l1": l1})

    has_l2 = (panel_l2 is not None and not panel_l2.empty
              and predict_date in panel_l2.index
              and name_map_l2 is not None and not name_map_l2.empty)
    if not has_l2:
        out["industry_l2"] = ""
        return out
    l2_codes = panel_l2.loc[predict_date].dropna().astype(str)
    l2 = l2_codes.map(name_map_l2.astype(str)).fillna(l2_codes)
    out["industry_l2"] = l2.reindex(out.index).fillna("")
    return out


def build_ranking(scores: pd.Series, tradable: pd.Series, frac: float,
                  names: pd.Series | None = None,
                  industry: pd.DataFrame | pd.Series | None = None,
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分数 → 排名表 + 持仓候选（top-frac 且信号日可交易，等权参考权重）。

    names 为可选「股票代码→简称」标注列；industry 可选：
      - DataFrame（industry_series 新口径）：须含 ``industry_l2``（二级，可空串）
        与 ``industry_l1``（一级）两列，输出同名列；
      - Series（旧口径，兼容历史调用）：单列行业标注，输出为 ``industry_l1``。
    """
    df = pd.DataFrame({"score": scores.dropna().sort_values(ascending=False)})
    df["rank"] = np.arange(1, len(df) + 1)
    df["pct_rank"] = df["score"].rank(pct=True)
    k = max(1, int(round(len(df) * frac)))
    df["top_frac"] = df["rank"] <= k
    df["tradable"] = tradable.reindex(df.index).fillna(False).astype(bool)
    # 展示列：把裸 bool 变成一眼可辨的中文标记（"" = 可交易；"不可交易" =
    # 信号日封板/停牌/ST 等被剔，实盘买不进）。不参与任何计算。
    df["flag"] = np.where(df["tradable"], "", "不可交易")
    if names is not None:
        df["name"] = names.reindex(df.index).fillna("")
    if industry is not None:
        if isinstance(industry, pd.DataFrame):
            df["industry_l2"] = industry["industry_l2"].reindex(df.index) \
                .fillna("").replace("", "二级未知")
            df["industry_l1"] = industry["industry_l1"].reindex(df.index) \
                .fillna("未知").replace("", "未知")
        else:
            df["industry_l2"] = ""
            df["industry_l1"] = industry.reindex(df.index) \
                .fillna("未知").replace("", "未知")
    cols = [c for c in ("rank", "name", "industry_l2", "industry_l1", "score",
                        "pct_rank", "top_frac", "tradable", "flag")
            if c in df.columns]
    ranking = df[cols].copy()
    picks = ranking[ranking["top_frac"] & ranking["tradable"]].copy()
    if len(picks):
        picks["weight"] = 1.0 / len(picks)
    return ranking, picks


def build_industry_table(ranking: pd.DataFrame) -> pd.DataFrame:
    """行业板块排名：按申万二级行业聚合个股模型分数。

    行业排序按 mean_score（行业内全部有分股票的截面 z 分数均值——即模型
    对该板块的整体看多程度）；top_frac_share = 行业内进入全A Top-frac 的
    股票占比，与全局 frac（默认 10%）比较可知板块的超/低配强度。
    输出行索引为二级中文名（无二级归属的股票并入 "未知"），并附 ``industry_l1``
    一级归属列（取组内众数），便于二级→一级上卷。
    """
    df = ranking.copy()
    if "industry_l2" not in df.columns:
        df["industry_l2"] = ""
    df["industry_l2"] = df["industry_l2"].fillna("未知").replace("", "未知")
    g = df.groupby("industry_l2")
    out = pd.DataFrame({
        "n_stocks": g.size(),
        "mean_score": g["score"].mean(),
        "median_score": g["score"].median(),
        "mean_pct_rank": g["pct_rank"].mean(),
        "n_top_frac": g["top_frac"].sum().astype(int),
    })
    # 一级归属：组内众数（二级在申万体系下唯一挂靠一级，众数即真值）
    if "industry_l1" in df.columns:
        out["industry_l1"] = g["industry_l1"] \
            .agg(lambda s: s.dropna().mode().iat[0]
                 if len(s.dropna().mode()) else "未知")
    out["top_frac_share"] = (out["n_top_frac"] / out["n_stocks"]).round(4)
    # 行业内第一名（全A排名最高的成员）
    order = df.sort_values("score", ascending=False)
    firsts = order.drop_duplicates("industry_l2", keep="first")
    has_name = "name" in firsts.columns
    labels = pd.Series(
        [f"{code} {nm}" if has_name and isinstance(nm, str) and nm else str(code)
         for code, nm in zip(firsts.index,
                             firsts["name"] if has_name else [""] * len(firsts))],
        index=firsts["industry_l2"].values)
    out["top_stock"] = labels
    out = out.sort_values("mean_score", ascending=False)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.index.name = "industry_l2"
    return out


def explain_features(importance: pd.Series) -> pd.DataFrame:
    """模型级：gain 特征重要性 → 排序表（中文名/释义/因子族/归一化贡献）。"""
    df = pd.DataFrame({"feature": importance.index.astype(str)})
    df["importance_gain"] = importance.values.astype(float)
    total = float(df["importance_gain"].sum())
    df["share"] = (df["importance_gain"] / total).round(4) if total > 0 else 0.0
    df["cum_share"] = df["share"].cumsum().round(4)
    gl = [lookup_glossary(f) for f in df["feature"]]
    df["name_cn"] = [g[0] for g in gl]
    df["description"] = [g[1] for g in gl]
    df["family"] = df["feature"].map(factor_family)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def explain_stocks(contrib: pd.DataFrame, z_scores: dict[str, pd.Series],
                   ranking: pd.DataFrame, n_factors: int = 3) -> pd.DataFrame:
    """个股级：每股取 |SHAP 贡献| 最大的 n 个因子，附截面 z 与中文释义。

    contrib: 预测日 SHAP 截面（index=code，列=因子+bias，见 predict_contrib）
    z_scores: {因子名: 预测日截面 z}（preprocess_panel 后值，0=截面均值）
    ranking: 已排序排名表（含 name/industry/score 标注列）

    输出列：code/name/industry_l2/industry_l1/rank/score
    + drv{i}_feature/name_cn/contrib/z + summary（"主要驱动：A(z=+2.1)、
    B(z=+1.8)；拖累：C(z=-1.2)"）。
    """
    feat_cols = [c for c in contrib.columns if c != "bias"]

    def _driver(code: str, f: str) -> dict:
        cval = float(contrib.loc[code, f])
        z = float(z_scores.get(f, pd.Series(dtype=float)).get(code, np.nan))
        name_cn, _desc = lookup_glossary(f)
        return {"feature": f, "name_cn": name_cn,
                "contrib": round(cval, 4), "z": round(z, 2)}

    rows = []
    for code, row in ranking.iterrows():
        if code not in contrib.index:
            continue
        top = contrib.loc[code, feat_cols].abs().nlargest(n_factors).index
        drv = [_driver(code, f) for f in top]

        def _note(p: dict) -> str:
            # SHAP 贡献方向与因子值方向相反时点明（非线性/反向关系）：
            # z 低但推高预测 = 低值看多（如超跌反弹）；z 高但压低预测 = 高值看空
            if p["contrib"] > 0 and p["z"] < -1:
                return f"{p['name_cn']} 低值看多(z={p['z']:+.2f})"
            if p["contrib"] < 0 and p["z"] > 1:
                return f"{p['name_cn']} 高值看空(z={p['z']:+.2f})"
            return f"{p['name_cn']}(z={p['z']:+.2f})"

        ups = [p for p in drv if p["contrib"] > 0]
        dns = [p for p in drv if p["contrib"] < 0]
        bits = []
        if ups:
            bits.append("主要驱动：" + "、".join(_note(p) for p in ups))
        if dns:
            bits.append("拖累：" + "、".join(_note(p) for p in dns))
        out = {"code": code,
               **{k: row.get(k, "") for k in
                  ("name", "industry_l2", "industry_l1", "rank", "score")}}
        for i, p in enumerate(drv, 1):
            out[f"drv{i}_feature"] = p["feature"]
            out[f"drv{i}_name"] = p["name_cn"]
            out[f"drv{i}_contrib"] = p["contrib"]
            out[f"drv{i}_z"] = p["z"]
        out["summary"] = "；".join(bits)
        rows.append(out)
    cols = ["code", "name", "industry_l2", "industry_l1", "rank", "score"]
    for i in range(1, n_factors + 1):
        cols += [f"drv{i}_feature", f"drv{i}_name", f"drv{i}_contrib", f"drv{i}_z"]
    cols.append("summary")
    return pd.DataFrame(rows, columns=cols)


def append_history(row: dict, out_dir: Path | None = None) -> None:
    """history.csv 追加一行（同 predict_date 重跑则覆盖，幂等）。"""
    p = (out_dir or OUT_DIR) / "history.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = pd.read_csv(p, dtype={"predict_date": str})
        old = old[old["predict_date"] != row["predict_date"]]
        out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        out = out.sort_values("predict_date", ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(p, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 数据与因子（复用实验同一代码路径）
# ---------------------------------------------------------------------------
def run_data_update(max_attempts: int = 3) -> None:
    """增量更新全A缓存（水位线推进：日线/复权/状态/股本/财务/股东表）。

    - --no-minute：日频排名不需要分钟线，且分钟表水位在 2026-09-07 退市股修复时
      被重置——不带此开关会触发数年分钟全量重拉（小时级）。分钟数据由独立的
      update_data 调用维护。
    - 盘中安全由 update_data 自身的守卫负责（cli_common.complete_day_target：
      未到收盘确认时刻自动把终点回退到上一交易日，防半拉日入缓存）。
    - 失败重试：SDK 拉大表偶发瞬时失败（2026-09-08 实测资产负债表 retry×3 挂）；
      各表水位独立推进，重试即增量续拉，代价小。
    - **总预算 + 降级不阻塞（2026-09-16 事故后新增）**：整条数据更新有墙钟预算
      （``fetch.update_total_timeout_s``，默认 1800s）。超时/连续失败时**不再 raise**，
      而是改用已推进的缓存继续出榜——数据更新的卡死不应阻塞当日排名产出。
      等价于手工 ``--skip-update``（09-16 当晚即靠此路径 7 分钟出榜）。
    """
    from config import Config

    cmd = [sys.executable, "-m", "scripts.ingest.update_data", "--pool", "all_a",
           "--no-minute"]
    timeout_s = int(Config.get().get("fetch", {}).get("update_total_timeout_s", 1800))
    for attempt in range(1, max_attempts + 1):
        log.info("数据更新(第 %d/%d 次，预算 %ds): %s",
                 attempt, max_attempts, timeout_s, " ".join(cmd))
        try:
            proc = subprocess.run(cmd, cwd=ROOT, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("数据更新超过 %ds 预算已被终止；改用其已推进的缓存继续出榜"
                        "（各表水位独立，下次运行自动续拉）", timeout_s)
            return
        if proc.returncode == 0:
            return
        log.warning("数据更新失败（exit=%d）", proc.returncode)
        if attempt < max_attempts:
            time.sleep(60)
    log.warning("数据更新连续 %d 次失败；改用现有缓存继续出榜（predict_date 以缓存"
                "最新交易日为准，见后续日志与 history.csv）", max_attempts)


def load_tail(predict_date: pd.Timestamp | None, n_days: int) -> dict:
    """尾部基础面板：后复权 OHLCV/vwap + 未复权 close + 复权因子 + 行业。

    与 build_alla_alpha_panels.load_alla_panels 同口径，只取尾部 n_days 日。
    """
    from config import Config
    from data.cache import DataCache
    from data.industry import IndustryClassification
    from data.offline import OfflineQuietDataSource

    cache_root = Path(str(Config.cache()["root"]))
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(
        daily.index.levels[0].normalize(), level=0)
    dates = daily.index.get_level_values(0).unique().sort_values()
    if predict_date is not None:
        dates = dates[dates <= predict_date]
    if len(dates) == 0:
        raise ValueError("daily_all_a 缓存为空（先跑 scripts.ingest.update_data --pool all_a）")
    tail_dates = dates[-n_days:]
    predict_date = tail_dates[-1]
    daily = daily[daily.index.get_level_values(0).isin(tail_dates)]
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, hi, lo, c = _panel("open"), _panel("high"), _panel("low"), _panel("close")
    v, amt = _panel("volume"), _panel("amount")
    raw_close = c.copy()

    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf = bf[[x for x in c.columns if x in bf.columns]]
    f = bf.reindex(index=c.index, columns=c.columns).ffill()
    for pnl in (o, hi, lo, c):
        pnl[:] = pnl.values * f.values
    vwap = ((amt / v).replace([np.inf, -np.inf], np.nan) * f).astype(np.float32)
    log.info("尾部面板: %d 日 × %d 股（%s ~ %s，预测日=%s）", len(c), c.shape[1],
             tail_dates[0].date(), tail_dates[-1].date(), predict_date.date())

    industry = None
    industry_l2 = None
    try:
        cache = DataCache(OfflineQuietDataSource())
        industry = IndustryClassification(cache, level=1).get_industry_panel(
            list(c.columns), c.index)
        if industry.isna().all().all():
            industry = None
        # 二级行业面板（细分行排名用）：文件不存在时 IndustryClassification
        # 返回全 NaN 面板 → 置 None 退回一级口径。
        ind2 = IndustryClassification(cache, level=2).get_industry_panel(
            list(c.columns), c.index)
        if not ind2.isna().all().all():
            industry_l2 = ind2
    except Exception as e:  # noqa: BLE001
        log.warning("行业面板不可用（%s），IndNeutralize 退化为恒等", str(e)[:80])

    px = {"open": o.astype(np.float32), "high": hi.astype(np.float32),
          "low": lo.astype(np.float32), "close": c.astype(np.float32),
          "volume": v.astype(np.float32), "amount": amt.astype(np.float32),
          "vwap": vwap}
    return {"px": px, "close_adj": c.astype(np.float32),
            "close_raw": raw_close, "bwd": f, "industry": industry,
            "industry_l2": industry_l2,
            "predict_date": predict_date}


def compute_alpha_features(names: list[str], px: dict,
                           industry: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """量价因子：factor.alphaXXX 注册表 + AlphaData（与实验面板构建同一路径）。"""
    from factor.alpha_base import AlphaData
    from scripts.builders.build_alla_alpha_panels import collect_factor_fns, registry_lookup

    fns = collect_factor_fns()
    missing = [n for n in names if n not in fns]
    if missing:
        raise KeyError(f"选择文件包含未知量价因子: {missing}")
    data = AlphaData(px, industry=industry)
    out: dict[str, pd.DataFrame] = {}
    for i, n in enumerate(names, 1):
        setname, reg_name = fns[n]
        p = registry_lookup(setname, reg_name)(data)
        out[n] = preprocess_panel(p)
        if i % 10 == 0:
            log.info("量价因子 %d/%d（最近 %s）", i, len(names), n)
    return out


_LONG_TABLE_FILES = {
    "income": "income.parquet",
    "balance": "balance_sheet.parquet",
    "cashflow": "cash_flow.parquet",
    "equity": "equity_structure.parquet",
    "dividend": "dividend.parquet",
    "pledge": "equity_pledge_freeze.parquet",
    "notice": "profit_notice.parquet",
    "express": "profit_express.parquet",
    # ---- 另类数据族源表（2026-09-12 接入日频主链）----
    "restricted": "equity_restricted.parquet",
    "margin": "margin_detail.parquet",
    "long_hu_bang": "long_hu_bang.parquet",
    "block_trading": "block_trading.parquet",
    "stock_status": "history_stock_status.parquet",
    "share_holder": "share_holder.parquet",
}


def _load_long_tables(need: list[str]) -> dict[str, pd.DataFrame]:
    """按需加载财务/股东/质押长表（多个基本面构建路径共用一次 IO）。"""
    from config import Config

    cache_root = Path(str(Config.cache()["root"]))
    out = {}
    for key in need:
        p = cache_root / _LONG_TABLE_FILES[key]
        out[key] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return out


def compute_fundamental_features(
        names: list[str], close_raw: pd.DataFrame,
        tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """基本面/分红因子（B族）：复用 build_alla_fundamental_factors 构建器（PIT）。"""
    from scripts.builders.build_alla_fundamental_factors import build_fundamental_panels

    panels = build_fundamental_panels(
        close_raw, close_raw.index, close_raw.columns,
        tables["income"], tables["balance"], tables["cashflow"],
        tables["equity"], tables["dividend"])
    missing = [n for n in names if n not in panels]
    if missing:
        raise KeyError(f"选择文件包含基本面构建器不产出的因子: {missing}")
    log.info("基本面因子(B族): %d 个（构建器产出 %d 个）", len(names), len(panels))
    return {n: preprocess_panel(panels[n]) for n in names}


def compute_market_cap(close_raw: pd.DataFrame,
                       long_tables: dict[str, pd.DataFrame] | None = None,
                       ) -> pd.DataFrame:
    """预测截面市值面板（亿元）：TOT_SHARE(总股本) × 未复权收盘，PIT 展开。

    与 ln_mktcap 因子同源（build_alla_fundamental_factors 的 cap = TOT_SHARE×close），
    但只取市值本身、不跑完整基本面构建器，供龙头股视图按市值分层。
    返回 date×code 的市值面板（亿元，NaN=缺股本/缺价）。
    """
    from data.financials import build_pit_panel

    need = ["income", "balance", "cashflow", "equity", "dividend"]
    tables = long_tables if long_tables is not None \
        else _load_long_tables(need)
    balance = tables.get("balance")
    if balance is None or "TOT_SHARE" not in balance.columns:
        return pd.DataFrame(np.nan, index=close_raw.index, columns=close_raw.columns)
    cal_idx = close_raw.index
    codes = close_raw.columns
    tot_share = build_pit_panel(balance, cal_idx, "TOT_SHARE").reindex(
        index=cal_idx, columns=codes)
    cap = tot_share.astype(float) * close_raw.astype(float)
    return cap / 1e8  # 元 → 亿元


def build_leaders(ranking: pd.DataFrame, mktcap: pd.Series,
                  n_leaders: int = 200, top_k: int = 20,
                  min_cap: float | None = None) -> pd.DataFrame:
    """龙头股视图：市值最大的 n_leaders 只中，模型分数最高的 top_k 只。

    主排名（raw 不中性化）天然偏向小市值，此视图按市值分层后单独排序，
    回答"市值前 n 的龙头里，模型最看好谁"。列：rank(市值内模型名次)/
    cap_rank(全市场市值名次)/mktcap(亿元)/score 及排名表原有标注。
    """
    cap = mktcap.reindex(ranking.index).dropna().sort_values(ascending=False)
    if min_cap is not None:
        cap = cap[cap >= min_cap]
    cap_rank = cap.rank(ascending=False, method="first").astype(int)
    leaders = cap.head(n_leaders)
    sub = ranking.loc[leaders.index].copy()
    if "rank" in sub.columns:
        sub = sub.drop(columns="rank")  # 全A名次，换成市值层内名次
    sub["mktcap"] = leaders
    sub["cap_rank"] = cap_rank.reindex(sub.index)
    sub = sub.sort_values("score", ascending=False)
    sub.insert(0, "rank", np.arange(1, len(sub) + 1))
    cols = [c for c in ("rank", "cap_rank", "name", "industry_l2", "industry_l1",
                        "mktcap", "score", "tradable") if c in sub.columns]
    return sub.head(top_k)[cols].copy()


def compute_constructed_features(
        names: list[str], close_adj: pd.DataFrame, close_raw: pd.DataFrame,
        tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """构造型基本面（B++族）：复用 build_alla_constructed_factors 构建器（PIT）。"""
    from scripts.builders.build_alla_constructed_factors import build_panels

    panels = build_panels(close_adj, close_raw, tables["income"],
                           tables["balance"], tables["cashflow"],
                           tables["dividend"])
    missing = [n for n in names if n not in panels]
    if missing:
        raise KeyError(f"选择文件包含构造型基本面不产出的因子: {missing}")
    log.info("构造型基本面(B++族): %d 个", len(names))
    return {n: preprocess_panel(panels[n]) for n in names}


def compute_pledge_features(
        names: list[str], close_adj: pd.DataFrame, close_raw: pd.DataFrame,
        tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """商誉/质押/业绩预告快报（B+族）：复用 build_alla_pledge_factors 构建器。"""
    from scripts.builders.build_alla_pledge_factors import build_pledge_factors

    panels = build_pledge_factors(close_adj, close_raw, tables["balance"],
                                  pledge=tables.get("pledge"),
                                  notice=tables.get("notice"),
                                  express=tables.get("express"))
    missing = [n for n in names if n not in panels]
    if missing:
        raise KeyError(f"选择文件包含 B+ 族构建器不产出的因子: {missing}")
    log.info("B+族(商誉/质押/业绩): %d 个", len(names))
    return {n: preprocess_panel(panels[n]) for n in names}


def compute_holder_features(names: list[str],
                            index: pd.DatetimeIndex,
                            columns: pd.Index) -> dict[str, pd.DataFrame]:
    """股东族因子（A族）：复用 build_alla_holder_factors 的 PIT 构建器。"""
    from config import Config
    from scripts.builders.build_alla_holder_factors import pit_holder_num, pit_share_holder

    cache_root = Path(str(Config.cache()["root"]))
    out: dict[str, pd.DataFrame] = {}
    need_num = [n for n in names if n in _HOLDER_NUM_KEYS]
    need_top = [n for n in names if n in _HOLDER_TOP_KEYS]
    if need_num:
        p = cache_root / "holder_num.parquet"
        hn = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        out.update(pit_holder_num(hn, index, columns))
    if need_top:
        p = cache_root / "share_holder.parquet"
        sh = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        out.update(pit_share_holder(sh, index, columns))
    missing = [n for n in names if n not in out]
    if missing:
        raise KeyError(f"选择文件包含未知股东族因子: {missing}")
    log.info("股东族因子(A族): %d 个", len(names))
    return {n: preprocess_panel(out[n]) for n in names}


def _check_missing(names: list[str], panels: dict, tag: str) -> None:
    """构建器产出覆盖校验：选择文件里的因子必须有对应构建路径。"""
    missing = [n for n in names if n not in panels]
    if missing:
        raise KeyError(f"选择文件包含{tag}构建器不产出的因子: {missing}")


def compute_style_features(names: list[str], close_adj: pd.DataFrame,
                           close_raw: pd.DataFrame,
                           tables: dict[str, pd.DataFrame]
                           ) -> dict[str, pd.DataFrame]:
    """风格族财务比率（17 个）：复用 build_alla_style_factors.build_panels（PIT）。"""
    from scripts.builders.build_alla_style_factors import build_panels

    panels = build_panels(close_adj, close_raw, tables["income"],
                          tables["balance"], tables["cashflow"],
                          tables["dividend"])
    _check_missing(names, panels, "风格族")
    log.info("风格族(财务比率): %d 个", len(names))
    return {n: preprocess_panel(panels[n]) for n in names}


def compute_alt_extra_features(
        groups: dict[str, list[str]], close_adj: pd.DataFrame,
        close_raw: pd.DataFrame, tables: dict[str, pd.DataFrame]
        ) -> dict[str, pd.DataFrame]:
    """另类数据族统一分派：事件/SUE质押/两融/资金流/折价股东动态/状态。

    groups: {构建器标识: 该路径下的因子名列表}，标识取
    ``event``/``sue_pledge``/``margin``/``moneyflow``/``disc_dyn``/``status``。
    每个路径都复用对应 builders 的 ``build_panels``（与全量构建同一代码路径），
    只对尾部 (cal_idx, codes) 重算。
    """
    cal_idx, codes = close_adj.index, close_adj.columns
    out: dict[str, pd.DataFrame] = {}
    for key, names in groups.items():
        if not names:
            continue
        if key == "event":
            from scripts.builders.build_alla_event_factors import build_panels as _bp
            panels = _bp(cal_idx, codes, tables={
                k: tables.get(k) for k in ("notice", "express", "restricted")})
        elif key == "sue_pledge":
            from scripts.builders.build_alla_sue_pledge_factors import build_panels as _bp
            panels = _bp(cal_idx, codes, close_raw=close_raw, tables={
                k: tables.get(k) for k in ("notice", "express", "pledge")})
        elif key == "margin":
            from scripts.builders.build_alla_margin_factors import build_panels as _bp
            panels = _bp(cal_idx, codes, margin=tables.get("margin"))
        elif key == "moneyflow":
            from scripts.builders.build_alla_moneyflow_factors import build_panels as _bp
            panels = _bp(cal_idx, codes, tables={
                k: tables.get(k) for k in ("long_hu_bang", "block_trading")})
        elif key == "disc_dyn":
            from scripts.builders.build_alla_disc_holder_dyn import build_panels as _bp
            panels = _bp(cal_idx, codes, close_raw=close_raw, tables={
                "block": tables.get("block_trading"),
                "holder": tables.get("share_holder")})
        elif key == "status":
            from scripts.builders.build_alla_status_factors import build_panels as _bp
            panels = _bp(cal_idx, codes, status=tables.get("stock_status"),
                         close_raw=close_raw)
        else:
            raise ValueError(f"未知另类数据族: {key}")
        _check_missing(names, panels, key)
        out.update({n: preprocess_panel(panels[n]) for n in names})
        log.info("另类族(%s): %d 个", key, len(names))
    return out


def compute_features(names: list[str], tail: dict,
                     long_tables: dict[str, pd.DataFrame] | None = None
                     ) -> dict[str, pd.DataFrame]:
    """按因子名分派到 12 条构建路径（与实验同一代码路径）。

    路径：量价 / 基本面 / 风格 / 质押(B+) / 构造(B++) / 股东(A) / 事件 /
    SUE+质押深度 / 两融 / 资金流 / 折价+股东动态 / 停牌状态。
    未登记因子直接抛 KeyError，不再兜底塞进基本面路径（旧行为会在构建器
    深处报错，指向性差且掩盖「因子库新增来源未接线」的真实问题）。

    long_tables 可由调用方预先加载（_load_long_tables）供多个构建路径复用，
    避免重复 IO；不传或不全时按需补齐。
    """
    def _pick(keys: set[str]) -> list[str]:
        return [n for n in names if n in keys]

    alpha = [n for n in names if n.startswith(_ALPHA_PREFIXES)]
    holder = _pick(_HOLDER_NUM_KEYS | _HOLDER_TOP_KEYS)
    pledge = _pick(_PLEDGE_KEYS)
    constructed = _pick(_CONSTRUCTED_KEYS)
    style = _pick(_STYLE_KEYS)
    event = _pick(_EVENT_KEYS)
    sue_pledge = _pick(_SUE_PLEDGE_KEYS)
    margin = _pick(_MARGIN_KEYS)
    moneyflow = _pick(_MONEYFLOW_KEYS)
    disc_dyn = _pick(_DISC_HOLDER_DYN_KEYS)
    status = _pick(_STATUS_KEYS)
    claimed = set(alpha) | set(holder) | set(pledge) | set(constructed) \
        | set(style) | set(event) | set(sue_pledge) | set(margin) \
        | set(moneyflow) | set(disc_dyn) | set(status)
    fund = [n for n in names if n not in claimed]
    log.info("特征分派: 量价 %d / 基本面 %d / 风格 %d / 质押 %d / 构造 %d / "
             "股东 %d / 事件 %d / SUE质押 %d / 两融 %d / 资金流 %d / "
             "折价动态 %d / 状态 %d（共 %d）",
             len(alpha), len(fund), len(style), len(pledge), len(constructed),
             len(holder), len(event), len(sue_pledge), len(margin),
             len(moneyflow), len(disc_dyn), len(status), len(names))

    long_keys: set[str] = set()
    if fund:
        long_keys |= {"income", "balance", "cashflow", "equity", "dividend"}
    if style:
        long_keys |= {"income", "balance", "cashflow", "dividend"}
    if pledge:
        long_keys |= {"balance", "pledge", "notice", "express"}
    if constructed:
        long_keys |= {"income", "balance", "cashflow", "dividend"}
    if event:
        long_keys |= {"notice", "express", "restricted"}
    if sue_pledge:
        long_keys |= {"notice", "express", "pledge"}
    if margin:
        long_keys |= {"margin"}
    if moneyflow:
        long_keys |= {"long_hu_bang", "block_trading"}
    if disc_dyn:
        long_keys |= {"block_trading", "share_holder"}
    if status:
        long_keys |= {"stock_status"}
    # 预加载只作复用优化：缺的键在这里补齐，避免调用方漏传导致静默缺表
    tables = dict(long_tables) if long_tables else {}
    todo = sorted(k for k in long_keys if k not in tables)
    if todo:
        tables.update(_load_long_tables(todo))

    feats: dict[str, pd.DataFrame] = {}
    if alpha:
        feats.update(compute_alpha_features(alpha, tail["px"], tail["industry"]))
    if fund:
        feats.update(compute_fundamental_features(
            fund, tail["close_raw"], tables))
    if style:
        feats.update(compute_style_features(
            style, tail["close_adj"], tail["close_raw"], tables))
    if pledge:
        feats.update(compute_pledge_features(
            pledge, tail["close_adj"], tail["close_raw"], tables))
    if constructed:
        feats.update(compute_constructed_features(
            constructed, tail["close_adj"], tail["close_raw"], tables))
    if holder:
        feats.update(compute_holder_features(holder, tail["close_adj"].index,
                                             tail["close_adj"].columns))
    feats.update(compute_alt_extra_features(
        {"event": event, "sue_pledge": sue_pledge, "margin": margin,
         "moneyflow": moneyflow, "disc_dyn": disc_dyn, "status": status},
        tail["close_adj"], tail["close_raw"], tables))
    got = set(feats)
    if got != set(names):
        raise RuntimeError(f"特征缺失: {sorted(set(names) - got)}")
    return feats


# ---------------------------------------------------------------------------
# 训练与预测
# ---------------------------------------------------------------------------
def _rank_average_single_day(panels: list[pd.DataFrame],
                             min_panels: int = 2) -> pd.DataFrame:
    """单日截面秩平均（集成臂 h1+h5 合成）。

    与实验侧 ``rolling_grid_alla::_rank_average`` 的**单日**情形同口径：对每只
    股票取其在各模型预测中的截面百分位秩，仅在**有效模型数 >= min_panels** 且
    该股在所有入选模型里都有预测（取交集）时求均值，否则不输出。

    生产链只需要预测日一个截面，故不复用实验侧逐日实现（那会 import 跨模块
    私有名，被 ``tests/test_layering`` 拦截）；一致性由
    ``tests/test_alla_daily_rank.py::test_rank_average_matches_experiment`` 锁死。
    """
    valid_codes = None
    ranks = []
    for p in panels:
        row = p.iloc[0] if len(p) else None
        if row is None:
            continue
        r = row.rank(pct=True)
        r = r[~row.isna()]
        ranks.append(r)
        valid_codes = r.index if valid_codes is None else valid_codes.intersection(r.index)
    # 返回全列（未入选者 NaN），与实验侧面板同形 —— 下游 build_ranking 会 dropna
    out = pd.Series(np.nan, index=panels[0].columns, dtype="float32")
    if len(ranks) < min_panels or valid_codes is None or not len(valid_codes):
        return out
    sub = pd.concat([r.reindex(valid_codes) for r in ranks], axis=1).dropna(axis=0)
    out.loc[sub.index] = sub.mean(axis=1).astype(np.float32)
    return out


def train_and_predict(feats: dict, close_adj: pd.DataFrame,
                      predict_date: pd.Timestamp, window: int,
                      horizon: int = HORIZON,
                      tradable_mask: pd.DataFrame | None = None
                      ) -> tuple[pd.Series, dict, "LGBMPredictor"]:
    """gbdt 在最近 window 个有效标签日重训 → 预测 predict_date 截面。

    标签 = h{horizon} 未来收益截面 rank；最后有效标签日 = 预测日前一交易日（其收益到
    预测日收盘，发布时点已知，无窥探）。返回 (score 截面, 训练段元信息, 预测器)
    ——预测器供解释功能取特征重要性与 SHAP 归因。

    ``tradable_mask``（T+1 成交口径）：给定时把买不进的样本标签置 NaN，等价于
    从训练集剔除（``LGBMPredictor.fit`` 内 ``~np.isnan(y)`` 自然跳过）。
    默认 None = 现行口径。
    """
    from model.labels import build_labels
    from model.params import DEFAULT_MODEL_PARAMS
    from model.predictor import LGBMPredictor

    labels, _embargo = build_labels(close_adj, horizon=horizon, mode="rank",
                                    tradable_mask=tradable_mask)
    valid = labels.index[labels.notna().any(axis=1)]
    valid = valid[valid < predict_date]
    tr = valid[-window:]
    if len(tr) < MIN_TRAIN:
        raise ValueError(f"训练段不足 {MIN_TRAIN} 日（现 {len(tr)}）")
    log.info("训练 h%d: %d 日（%s ~ %s，窗=%d）→ 预测 %s", horizon, len(tr),
             tr[0].date(), tr[-1].date(), window, predict_date.date())

    p = LGBMPredictor(**DEFAULT_MODEL_PARAMS["gbdt"])
    p.fit({k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
    pred = p.predict({k: v.loc[[predict_date]] for k, v in feats.items()})
    meta = {"n_train_days": len(tr), "train_begin": str(tr[0].date()),
            "train_end": str(tr[-1].date()), "window": window,
            "horizon": horizon}
    return pred.iloc[0], meta, p


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _atomic_or_direct(path: Path, write) -> bool:
    """优先原子替换（tmp + os.replace），被占用时降级为直接覆写。

    Windows 陷阱：目标被其他进程以共享读打开（IDE / Excel / 预览面板）且未
    授予 FILE_SHARE_DELETE 权限时，``os.replace`` 报 WinError 5；**但文件本身
    仍可写**（实测 ``cp`` / ``to_csv`` 直写成功）。旧行为是整轮运行失败——
    6 分钟计算只为最后一步写盘白跑（2026-09-17 实测：编辑器持有 ranking /
    picks 句柄 → 出榜流程 raise）。故降级为直接覆写（牺牲原子性换取不中断），
    并告警提示存在半写风险。
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        write(path)
        log.warning("%s 原子替换被占用阻挡，已降级为直接覆写（存在半写风险）",
                    path.name)
        return True
    except OSError:
        return False


def _write_latest(p: Path, src: Path) -> bool:
    """latest 稳定副本写入（原子优先、占用降级）；彻底失败时告警不中断。

    latest_*.csv 只是当日主输出（ranking_{ds}.csv 等）的稳定路径副本，
    被外部进程（如预览/同步）短暂独占时不应拖垮整次运行。
    """
    ok = _atomic_or_direct(
        p, lambda t: t.write_text(src.read_text(encoding="utf-8-sig"),
                                  encoding="utf-8-sig"))
    if not ok:
        log.warning("latest 副本 %s 未更新（不影响当日主输出）", p.name)
    return ok


def _write_table(path: Path, df: pd.DataFrame) -> bool:
    """主 CSV 写入（原子优先、占用降级）；彻底失败时返回 False。"""
    return _atomic_or_direct(path, lambda t: df.to_csv(t, encoding="utf-8-sig"))


def write_outputs(ranking: pd.DataFrame, picks: pd.DataFrame,
                  predict_date: pd.Timestamp, meta: dict,
                  industry_table: pd.DataFrame | None = None,
                  feature_importance: pd.DataFrame | None = None,
                  explain_top: pd.DataFrame | None = None,
                  leaders: pd.DataFrame | None = None,
                  out_dir: Path | None = None):
    """排名/持仓/行业排名/解释 CSV + latest 稳定路径副本，返回文件路径 dict。"""
    d = out_dir or OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    ds = predict_date.strftime("%Y%m%d")

    rank_path = d / f"ranking_{ds}.csv"
    picks_path = d / f"picks_{ds}.csv"
    ranking.index.name = "code"
    picks.index.name = "code"
    if not (_write_table(rank_path, ranking) and _write_table(picks_path, picks)):
        raise PermissionError(
            f"主输出写入失败：{rank_path.name} / {picks_path.name} 被其他进程占用"
            "（如 IDE/Excel 打开着文件，请关闭后重跑）")
    _write_latest(d / "latest_ranking.csv", rank_path)
    _write_latest(d / "latest_picks.csv", picks_path)

    ind_path = None
    if industry_table is not None and len(industry_table):
        ind_path = d / f"industry_rank_{ds}.csv"
        industry_table.index.name = "industry"
        if not _write_table(ind_path, industry_table):
            raise PermissionError(
                f"行业排名写入失败：{ind_path.name} 被其他进程占用，请关闭后重跑")
        _write_latest(d / "latest_industry_rank.csv", ind_path)

    imp_path = None
    if feature_importance is not None and len(feature_importance):
        imp_path = d / f"feature_importance_{ds}.csv"
        if not _write_table(imp_path, feature_importance):
            raise PermissionError(
                f"特征重要性写入失败：{imp_path.name} 被其他进程占用，请关闭后重跑")
        _write_latest(d / "latest_feature_importance.csv", imp_path)

    exp_path = None
    if explain_top is not None and len(explain_top):
        exp_path = d / f"explain_top_{ds}.csv"
        if not _write_table(exp_path, explain_top):
            raise PermissionError(
                f"个股归因写入失败：{exp_path.name} 被其他进程占用，请关闭后重跑")
        _write_latest(d / "latest_explain_top.csv", exp_path)

    lead_path = None
    if leaders is not None and len(leaders):
        lead_path = d / f"leaders_{ds}.csv"
        if not _write_table(lead_path, leaders):
            raise PermissionError(
                f"龙头股视图写入失败：{lead_path.name} 被其他进程占用，请关闭后重跑")
        _write_latest(d / "latest_leaders.csv", lead_path)

    append_history({"predict_date": ds, **meta}, out_dir=d)
    paths = {"ranking": str(rank_path), "picks": str(picks_path)}
    if ind_path is not None:
        paths["industry"] = str(ind_path)
    if imp_path is not None:
        paths["feature_importance"] = str(imp_path)
    if exp_path is not None:
        paths["explain_top"] = str(exp_path)
    if lead_path is not None:
        paths["leaders"] = str(lead_path)
    return paths


# ---------------------------------------------------------------------------
# 计划任务（沿用 monitor_performance 的 schtasks 模式）
# ---------------------------------------------------------------------------
def _run_schtasks(cmd: str) -> str:
    """跑 schtasks 并回显输出。

    中文 Windows 下 schtasks 输出为 GBK，``text=True``（默认 utf-8）会
    UnicodeDecodeError —— 2026-09-17 实测：任务其实已建好，但异常栈把成功回显吞了，
    看起来像注册失败。故按编码逐个尝试解码。
    """
    proc = subprocess.run(cmd, shell=True, capture_output=True)
    raw = (proc.stdout or b"") + (proc.stderr or b"")
    for enc in ("utf-8", "gbk", "cp1252"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").strip()


def install_task(time_str: str) -> str:
    """注册/更新每日计划任务（默认跑全流程含数据更新）。"""
    py = Path(SYSTEM_PY).resolve()
    script = (ROOT / "scripts" / "pipelines" / "alla_daily_rank.py").resolve()
    tr = f'\\"{py}\\" \\"{script}\\"'
    cmd = (f'schtasks /Create /F /TN "{TASK_NAME}" /SC DAILY /ST {time_str} '
           f'/TR "{tr}"')
    return f"cmd: {cmd}\n{_run_schtasks(cmd)}"


def remove_task() -> str:
    cmd = f'schtasks /Delete /F /TN "{TASK_NAME}"'
    return f"cmd: {cmd}\n{_run_schtasks(cmd)}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args) -> dict:
    from scripts.pipelines.rolling_grid_alla import existence_mask

    t0 = time.time()
    if not args.skip_update:
        run_data_update()

    tail = load_tail(pd.Timestamp(str(args.date)) if args.date else None,
                     tail_n_days(args.window))
    predict_date = tail["predict_date"]
    horizons = tuple(args.horizons)
    exclude = tuple(args.exclude_features)
    out_dir = OUT_DIR if not args.out_tag \
        else OUT_DIR.with_name(OUT_DIR.name + args.out_tag)

    preproc = str(getattr(args, "preproc", DEFAULT_PREPROC))
    sel_dir = ORTHO_SELECTION_DIR if preproc == "ortho" else SELECTION_DIR

    # 每个 horizon 读各自的落盘清单（实验口径：h1/h5 的质量窗 IC 不同，清单不同）
    sel: dict[int, dict] = {}
    for h in horizons:
        names_h, year_h = load_selection(predict_date, sel_dir=sel_dir,
                                         horizon=h, exclude=exclude)
        sel[h] = {"names": names_h, "year": year_h}
        log.info("h%d 特征选择: %d 个（选择年份 %d）", h, len(names_h), year_h)
    names_all = sorted({n for d_ in sel.values() for n in d_["names"]})
    log.info("口径: preproc=%s | horizons=%s | 硬剔除=%s | 特征并集 %d 个 | 输出 → %s",
             preproc, list(horizons), list(exclude) or "无", len(names_all),
             out_dir.name)

    # 预加载基本面长表：compute_features 与龙头股市值面板共用一次 IO
    long_keys: set[str] = set()
    if any(n in (_HOLDER_NUM_KEYS | _HOLDER_TOP_KEYS) for n in names_all):
        long_keys |= {"income", "balance", "cashflow", "equity", "dividend"}
    if any(n in _PLEDGE_KEYS for n in names_all):
        long_keys |= {"balance", "pledge", "notice", "express"}
    if any(n in _CONSTRUCTED_KEYS for n in names_all):
        long_keys |= {"income", "balance", "cashflow", "dividend"}
    if preproc == "ortho":
        long_keys.add("balance")   # 正交化需要 TOT_SHARE × 未复权收盘 的市值面板
    long_tables = _load_long_tables(sorted(long_keys))

    # 正交化协变量面板：行业取自 tail["industry"]（与实验 cov_industry 同源
    # IndustryClassification level=1）；市值实时构造 —— 股本源与离线 `_base` 不同源
    # （balance.TOT_SHARE vs equity_structure 事件表），见 prod_pipeline_gap §5.5。
    cap_panel = None
    if preproc == "ortho":
        cap_panel = compute_market_cap(tail["close_raw"], long_tables)
        set_panel_transform(make_ortho_transform(cap_panel, tail.get("industry")))
        log.info("面板口径: 因子层正交化（MAD→行业+log市值中性化→zscore）| "
                 "市值面板 %d×%d | 行业面板 %s",
                 *cap_panel.shape,
                 "可用" if tail.get("industry") is not None else "缺失(退化为仅市值)")
    else:
        set_panel_transform(None)

    # 特征按并集**只算一次**，再按各 horizon 的子集切分（IO/计算与单模型臂同量级）
    feats_all = compute_features(names_all, tail, long_tables=long_tables)
    close_adj = tail["close_adj"]

    # 训练标签口径（默认关闭）：可交易掩码在**尾部面板**上实时构建（T+1 成交口径），
    # 与实验侧 build_tradable_mask 同一函数、同一掩码语义（防两套口径漂移）。
    lab_mask = None
    if args.tradable_labels:
        from data.tradability import build_tradable_mask
        lab_mask = build_tradable_mask(close_adj, bwd=tail.get("bwd"))
        log.info("+++ 训练标签：可交易掩码口径（不可交易占比 %.2f%%，掩掉 %s 个 日×股）",
                 100.0 * float((~lab_mask).to_numpy().mean()),
                 f"{int((~lab_mask).to_numpy().sum()):,}")

    preds, per_h_meta, predictors = [], [], {}
    for h in horizons:
        feats_h = {k: feats_all[k] for k in sel[h]["names"] if k in feats_all}
        missing_h = [k for k in sel[h]["names"] if k not in feats_all]
        if missing_h:
            log.warning("h%d 有 %d 个特征未产出（构建失败），已忽略: %s",
                        h, len(missing_h), missing_h)
        pred_h, meta_h, mdl_h = train_and_predict(
            feats_h, close_adj, predict_date, args.window, horizon=h,
            tradable_mask=lab_mask)
        # 幽灵股守卫：掩掉未上市/无行情/特征不可用的股票（同实验 stage_predict）
        valid_h = existence_mask(feats_h, close_adj, pd.DatetimeIndex([predict_date]))
        pred_h = pred_h.where(valid_h.iloc[0])
        log.info("h%d 可用股票 %d", h, int(pred_h.notna().sum()))
        preds.append(pred_h.to_frame().T)
        per_h_meta.append(meta_h)
        predictors[h] = (mdl_h, feats_h)

    if len(horizons) == 1:
        scores = preds[0].iloc[0]
        ens_tag = "none"
    else:
        scores = _rank_average_single_day(preds)
        ens_tag = "+".join(f"h{h}" for h in horizons) + "_rank_avg"
        log.info("集成 %s → 单日截面秩平均，有效股票 %d（各臂交集口径）",
                 ens_tag, int(scores.notna().sum()))
    # 解释口径固定 h1（不在集成里时取首个 horizon）：SHAP/重要性只对单模型有意义
    expl_h = 1 if 1 in predictors else horizons[0]
    predictor, feats = predictors[expl_h]
    train_meta = {"horizons": list(horizons), "excluded_features": list(exclude),
                  "ensemble": ens_tag, "explain_horizon": expl_h,
                  "per_horizon": per_h_meta}
    n_raw = int(scores.notna().sum())

    tradable = signal_day_tradable(tail["close_raw"], predict_date)
    # 变量名必须与上面的特征名 names 区分：此处是「股票代码→简称」映射。
    # 曾因同名覆盖，meta["n_features"] 被写成股票数 5689（2026-09-07 history.csv 可见）。
    stock_names = load_stock_names(tail["close_raw"].columns)
    ind_panel = tail.get("industry")
    name_map = load_industry_names() if ind_panel is not None \
        else pd.Series(dtype=object)
    ind_panel_l2 = tail.get("industry_l2")
    name_map_l2 = load_industry_names(level=2) if ind_panel_l2 is not None \
        else pd.Series(dtype=object)
    if name_map_l2.empty:
        ind_panel_l2 = None  # 名表缺失 → 拼合无意义，整体退回一级口径
    industry = industry_series(ind_panel, predict_date, name_map,
                               panel_l2=ind_panel_l2, name_map_l2=name_map_l2)
    industry = industry if len(industry) else None
    ranking, picks = build_ranking(scores, tradable, args.frac,
                                   names=stock_names, industry=industry)
    ind_table = build_industry_table(ranking) if "industry_l2" in ranking.columns \
        else pd.DataFrame()

    # 选股解释：模型级（当日特征重要性）+ 个股级（Top20 SHAP 归因）
    imp_table = explain_features(predictor.feature_importance("gain"))
    contrib = predictor.predict_contrib(
        {k: v.loc[[predict_date]] for k, v in feats.items()})
    contrib_t = contrib.xs(predict_date, level="date")
    z_scores = {f: feats[f].loc[predict_date] for f in feats}
    explain_top = explain_stocks(contrib_t, z_scores, ranking.head(20))
    log.info("解释: 特征重要性 %d 个 | Top%d SHAP 归因完成",
             len(imp_table), len(explain_top))

    # 龙头股视图：市值前 200 中模型分最高的 20 只（ortho 口径下前面已算过，复用）
    if cap_panel is None:
        cap_panel = compute_market_cap(tail["close_raw"], long_tables)
    mktcap = cap_panel.loc[predict_date]
    leaders_table = build_leaders(ranking, mktcap)
    log.info("龙头视图: 市值前 200 中模型 Top%d（市值口径 TOT_SHARE×收盘）",
             len(leaders_table))

    meta = {"model": "gbdt", **train_meta, "frac": args.frac, "preproc": preproc,
            "selection_dir": sel_dir.name,
            "n_features": len(sel[expl_h]["names"]),
            "n_features_union": len(names_all),
            "selection_year": sel[expl_h]["year"],
            "n_scored": n_raw, "n_top_frac": int(ranking["top_frac"].sum()),
            "n_picks": len(picks),
            "n_industries": 0 if ind_table.empty else len(ind_table),
            "top5": "|".join(ranking.index[:5].astype(str)),
            "runtime_sec": round(time.time() - t0, 1)}
    paths = write_outputs(ranking, picks, predict_date, meta,
                          industry_table=ind_table,
                          feature_importance=imp_table,
                          explain_top=explain_top,
                          leaders=leaders_table,
                          out_dir=out_dir)

    log.info("=" * 70)
    log.info("预测日 %s: 有分股票 %d | top%.0f%% %d | 可交易候选 %d | 行业 %d | 耗时 %.0fs",
             predict_date.date(), n_raw, args.frac * 100,
             meta["n_top_frac"], len(picks), meta["n_industries"],
             time.time() - t0)
    log.info("Top 20 预览 (code, name, 行业, score, 可交易):")
    for i, (code, row) in enumerate(ranking.head(20).iterrows(), 1):
        nm = row.get("name", "")
        ind = row.get("industry_l2", "") or row.get("industry_l1", "")
        log.info("%4d  %s  %-8s  %-6s  %+.4f  %s", i, code, nm, ind,
                 row["score"], "√" if row["tradable"] else "×")
    # P0 告警：rank 头部若混入被 tradable 剔除的股票（信号日封板/停牌等），
    # 「排名第一」并不等于「最看好」——它们实盘买不进，分数也常落在极端叶子上。
    _head = ranking.head(20)
    _rej = _head[~_head["tradable"]]
    if len(_rej):
        log.warning(
            "⚠ Top20 中 %d 只被 tradable 过滤（未进 picks、实盘不可买）：%s",
            len(_rej),
            "、".join(f"{c} {r.get('name', '')}".strip()
                     for c, r in _rej.iterrows()))
        log.warning(
            "  提示：头部被剔多为信号日封板/停牌股，分数受极端叶子驱动，"
            "不等于更强的选股能力；实际选股请以 picks 为准。")
    log.info("当日模型 Top 因子 (gain 占比):")
    for _, r in imp_table.head(6).iterrows():
        log.info("  %-24s %-12s %-6s %6.2f%%  %s", r["feature"], r["name_cn"],
                 r["family"], r["share"] * 100, r["description"])
    log.info("Top 5 驱动摘要:")
    for _, r in explain_top.head(5).iterrows():
        log.info("  %-10s %-8s %s", r["code"], r.get("name", ""), r["summary"])
    out_msg = f"输出: {paths['ranking']} | {paths['picks']}"
    if "industry" in paths:
        out_msg += f" | 行业排名: {paths['industry']}"
    if "feature_importance" in paths:
        out_msg += f" | 特征重要性: {paths['feature_importance']}"
    if "explain_top" in paths:
        out_msg += f" | 个股归因: {paths['explain_top']}"
    if "leaders" in paths:
        out_msg += f" | 龙头视图: {paths['leaders']}"
    log.info(out_msg)
    log.info("=" * 70)
    return {"predict_date": str(predict_date.date()), "meta": meta, "paths": paths}


def main() -> None:
    ap = argparse.ArgumentParser(description="全A滚动模型每日选股排名")
    ap.add_argument("--skip-update", action="store_true", help="跳过数据增量更新")
    ap.add_argument("--date", default=None, help="预测日 YYYYMMDD（默认缓存最新交易日）")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help=f"滚动训练窗（默认 {DEFAULT_WINDOW}；750 = gbdt_w750 变体）")
    ap.add_argument("--frac", type=float, default=DEFAULT_FRAC,
                    help=f"持仓候选分位（默认 {DEFAULT_FRAC}）")
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS),
                    metavar="H[,H...]",
                    help="集成 horizon 列表（逗号分隔）。默认 "
                         f"{'+'.join('h' + str(h) for h in HORIZONS)} 截面秩平均"
                         "（主实验信号形态）；传 1 回到 h1 单模型口径")
    ap.add_argument("--exclude-features", default=",".join(EXCLUDE_FEATURES),
                    metavar="F[,F...]",
                    help=f"硬剔除的特征名（逗号分隔）。默认 "
                         f"{','.join(EXCLUDE_FEATURES) or '无'}；传空串 '' 关闭剔除")
    ap.add_argument("--preproc", choices=("zscore", "ortho"),
                    default=DEFAULT_PREPROC,
                    help="面板预处理口径：ortho（默认，因子层正交化 "
                         "MAD→行业+log市值中性化→zscore，与主实验 panels_neu 同流程、"
                         "自动改用 ortho 臂特征清单；北交所无行业分类故被排除）或 "
                         "zscore（原始面板 + 截面 zscore，2026-09-16 前默认）")
    ap.add_argument("--tradable-labels", action="store_true",
                    default=DEFAULT_TRADABLE_LABELS,
                    help="训练标签掩掉买不进的样本（T+1 成交口径可交易掩码，"
                         "P0 §六 P1-a 治本项）；默认关闭 = 现行口径")
    ap.add_argument("--out-tag", default="", metavar="TAG",
                    help="输出目录后缀（如 _h1h5 → reports/alla_daily_h1h5）；"
                         "默认空 = 写生产目录 reports/alla_daily")
    ap.add_argument("--install-task", nargs="?", const="17:30", default=None,
                    metavar="HH:MM", help="注册每日 Windows 计划任务并退出")
    ap.add_argument("--remove-task", action="store_true", help="删除计划任务并退出")
    args = ap.parse_args()
    args.horizons = tuple(int(x) for x in str(args.horizons).split(",") if x.strip())
    args.exclude_features = tuple(
        x.strip() for x in str(args.exclude_features).split(",") if x.strip())
    if not args.horizons:
        ap.error("--horizons 不能为空")
    if args.install_task:
        print(install_task(args.install_task))
        return
    if args.remove_task:
        print(remove_task())
        return
    run(args)


if __name__ == "__main__":
    main()
