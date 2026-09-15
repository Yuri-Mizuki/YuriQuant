"""全A滚动模型每日选股排名 —— alla_rolling 最优方案（gbdt h1×M raw Top10%）的生产化推理。

实验背景（reports/alla_rolling，2026-09-01 交付 + 2026-09-07 退市股修复后重跑）：
最优方案 = 全A + 每年特征选择（50个量价/基本面因子，reports/alla_rolling/selection/
y{year}__h1.json）+ gbdt（超参沿用 model.params.DEFAULT_MODEL_PARAMS）+
h=1 rank 标签 + 500 日滚动训练窗 + raw（不中性化）Top10% 等权、月频调仓，
年超额 vs 上证 +9.1%（9 年中 7 年为正，README 正典方案）。

本脚本每天做的事（与实验同一代码路径，无新增调参）：

  1. 数据增量更新（scripts.ingest.update_data --pool all_a，--skip-update 跳过）；
  2. 尾部面板重算：最近 (训练窗 + 260 日预热 + 缓冲) 个交易日的后复权 OHLCV/vwap；
  3. 只重算当年入选的 ~50 个因子（量价走 factor.alphaXXX 注册表；基本面/风格/
     股东/质押/构造/事件/SUE/两融/资金流/折价动态/停牌状态等族复用
     scripts.builders 的 build_panels 入口），截面 zscore + ±10 剪裁（与
     FeatureStore 同口径）；
  4. gbdt 在最近 500 个有效标签日上重训，预测最新截面；
  5. 幽灵股守卫（existence_mask）+ 信号日可交易性标注（停牌/ST/封板）；
  6. 输出全A排名 CSV + Top10% 持仓候选 + history 追加（reports/alla_daily/）。

口径披露（与实验的差异，均为如实可知的边界）：
- **尾部重算而非全历史拼接**：后复权因子随分红除权漂移，全历史拼接会产生复权
  基准断层。本脚本对训练窗与预测截面用同一次重算（同一复权基准），内部自洽；
  不改写冻结的 all_a_2018_2026 实验数据集。
- **训练段用发布时点已知的全部标签**（最后有效标签日 = 预测日前一交易日）。
  实验的 embargo 是回测防污染隔离；实时预测不存在窥探未来，故取到最新。
- **可交易性是信号日状态估计**：回测掩码用 T+1 状态（T+1 成交口径），实时排名
  发布时 T+1 未知，改用 T 日停牌/ST/收盘封板标注，T+1 一字板仍可能买不进。
- 跨年无当年选择文件时回退最近年份并告警（可重跑 rolling_grid_alla --stage select）。

用法:
    python scripts/pipelines/alla_daily_rank.py                     # 全流程（含数据更新）
    python scripts/pipelines/alla_daily_rank.py --skip-update       # 离线/数据已更新
    python scripts/pipelines/alla_daily_rank.py --date 20260904     # 指定预测日（默认最新）
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
HORIZON = 1                    # 实验最优 horizon（h1×M raw Top10%）
DEFAULT_WINDOW = 500           # gbdt 滚动训练窗（--window 750 = gbdt_w750 变体）
MIN_TRAIN = 250                # 最少训练日（同实验）
WARMUP = 260                   # alpha 公式最大回看 250 日 + 缓冲
TAIL_BUFFER = 40               # 停牌缺口/月末等额外缓冲
DEFAULT_FRAC = 0.10            # Top10% 持仓候选口径
SELECTION_DIR = ROOT / "reports" / "alla_rolling" / "selection"
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


def preprocess_panel(p: pd.DataFrame) -> pd.DataFrame:
    """因子面板统一口径：float32 → inf→NaN → 截面 zscore → ±10 剪裁。

    与 FeatureStore 读取冻结面板时的变换一致（astype 先于 replace，防 float64
    巨值溢出成 float32 inf 后漏杀，见 build_alla_alpha_panels 2026-09-01 教训）。
    """
    from factor.preprocessing import standardize_zscore
    p = p.astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return standardize_zscore(p).clip(-10.0, 10.0).astype(np.float32)


def load_selection(predict_date: pd.Timestamp,
                   sel_dir: Path | None = None) -> tuple[list[str], int]:
    """当年 h1 特征选择文件；跨年缺失时回退最近年份（告警）。"""
    d = sel_dir or SELECTION_DIR
    p = d / f"y{predict_date.year}__h{HORIZON}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), predict_date.year
    cands = sorted(d.glob(f"y*__h{HORIZON}.json"))
    if not cands:
        raise FileNotFoundError(f"{d} 下无任何 y*__h{HORIZON}.json 选择文件")
    p = cands[-1]
    sel_year = int(p.name[1:5])
    log.warning("y%d__h%d.json 不存在，回退最近年份 %d 的特征选择"
                "（如需当年口径：python scripts/pipelines/rolling_grid_alla.py --stage select）",
                predict_date.year, HORIZON, sel_year)
    return json.loads(p.read_text(encoding="utf-8")), sel_year


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
                    name_map: pd.Series) -> pd.Series:
    """行业归属面板预测日行 → 每股行业标注（中文名优先，缺名表时回退行业代码）。

    panel: (date × code) → industry_code（load_tail 的 industry 键）；
    name_map: industry_code → 中文名（load_industry_names，可为空）。
    返回 index=code、values=中文名|行业代码 的 Series；无行业归属的股票
    不在返回中（下游 reindex 后落为 NaN → "未知"）。
    """
    if panel is None or panel.empty or predict_date not in panel.index:
        return pd.Series(dtype=object)
    codes = panel.loc[predict_date].dropna().astype(str)
    if name_map is None or name_map.empty:
        return codes
    return codes.map(name_map.astype(str)).fillna(codes)


def build_ranking(scores: pd.Series, tradable: pd.Series, frac: float,
                  names: pd.Series | None = None,
                  industry: pd.Series | None = None,
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分数 → 排名表 + 持仓候选（top-frac 且信号日可交易，等权参考权重）。

    names / industry 为可选标注列（每股中文名 / 申万一级行业中文名）。
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
        df["industry"] = industry.reindex(df.index).fillna("未知").replace("", "未知")
    cols = [c for c in ("rank", "name", "industry", "score", "pct_rank",
                        "top_frac", "tradable", "flag") if c in df.columns]
    ranking = df[cols].copy()
    picks = ranking[ranking["top_frac"] & ranking["tradable"]].copy()
    if len(picks):
        picks["weight"] = 1.0 / len(picks)
    return ranking, picks


def build_industry_table(ranking: pd.DataFrame) -> pd.DataFrame:
    """行业板块排名：按申万一级行业聚合个股模型分数。

    行业排序按 mean_score（行业内全部有分股票的截面 z 分数均值——即模型
    对该板块的整体看多程度）；top_frac_share = 行业内进入全A Top-frac 的
    股票占比，与全局 frac（默认 10%）比较可知板块的超/低配强度。
    """
    df = ranking.copy()
    if "industry" not in df.columns:
        df["industry"] = "未知"
    df["industry"] = df["industry"].fillna("未知")
    g = df.groupby("industry")
    out = pd.DataFrame({
        "n_stocks": g.size(),
        "mean_score": g["score"].mean(),
        "median_score": g["score"].median(),
        "mean_pct_rank": g["pct_rank"].mean(),
        "n_top_frac": g["top_frac"].sum().astype(int),
    })
    out["top_frac_share"] = (out["n_top_frac"] / out["n_stocks"]).round(4)
    # 行业内第一名（全A排名最高的成员）
    order = df.sort_values("score", ascending=False)
    firsts = order.drop_duplicates("industry", keep="first")
    has_name = "name" in firsts.columns
    labels = pd.Series(
        [f"{code} {nm}" if has_name and isinstance(nm, str) and nm else str(code)
         for code, nm in zip(firsts.index,
                             firsts["name"] if has_name else [""] * len(firsts))],
        index=firsts["industry"].values)
    out["top_stock"] = labels
    out = out.sort_values("mean_score", ascending=False)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.index.name = "industry"
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

    输出列：code/name/industry/rank/score + drv{i}_feature/name_cn/contrib/z
    + summary（"主要驱动：A(z=+2.1)、B(z=+1.8)；拖累：C(z=-1.2)"）。
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
               **{k: row.get(k, "") for k in ("name", "industry", "rank", "score")}}
        for i, p in enumerate(drv, 1):
            out[f"drv{i}_feature"] = p["feature"]
            out[f"drv{i}_name"] = p["name_cn"]
            out[f"drv{i}_contrib"] = p["contrib"]
            out[f"drv{i}_z"] = p["z"]
        out["summary"] = "；".join(bits)
        rows.append(out)
    cols = ["code", "name", "industry", "rank", "score"]
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
    """
    cmd = [sys.executable, "-m", "scripts.ingest.update_data", "--pool", "all_a",
           "--no-minute"]
    for attempt in range(1, max_attempts + 1):
        log.info("数据更新(第 %d/%d 次): %s", attempt, max_attempts, " ".join(cmd))
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode == 0:
            return
        log.warning("数据更新失败（exit=%d）", proc.returncode)
        if attempt < max_attempts:
            time.sleep(60)
    raise RuntimeError(f"数据更新连续 {max_attempts} 次失败：{' '.join(cmd)}")


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
    try:
        cache = DataCache(OfflineQuietDataSource())
        industry = IndustryClassification(cache, level=1).get_industry_panel(
            list(c.columns), c.index)
        if industry.isna().all().all():
            industry = None
    except Exception as e:  # noqa: BLE001
        log.warning("行业面板不可用（%s），IndNeutralize 退化为恒等", str(e)[:80])

    px = {"open": o.astype(np.float32), "high": hi.astype(np.float32),
          "low": lo.astype(np.float32), "close": c.astype(np.float32),
          "volume": v.astype(np.float32), "amount": amt.astype(np.float32),
          "vwap": vwap}
    return {"px": px, "close_adj": c.astype(np.float32),
            "close_raw": raw_close, "bwd": f, "industry": industry,
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
    cols = [c for c in ("rank", "cap_rank", "name", "industry", "mktcap", "score",
                        "tradable") if c in sub.columns]
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
def train_and_predict(feats: dict, close_adj: pd.DataFrame,
                      predict_date: pd.Timestamp, window: int
                      ) -> tuple[pd.Series, dict, "LGBMPredictor"]:
    """gbdt 在最近 window 个有效标签日重训 → 预测 predict_date 截面。

    标签 = h1 未来收益截面 rank；最后有效标签日 = 预测日前一交易日（其收益到
    预测日收盘，发布时点已知，无窥探）。返回 (score 截面, 训练段元信息, 预测器)
    ——预测器供解释功能取特征重要性与 SHAP 归因。
    """
    from model.labels import build_labels
    from model.predictor import LGBMPredictor
    from model.params import DEFAULT_MODEL_PARAMS

    labels, _embargo = build_labels(close_adj, horizon=HORIZON, mode="rank")
    valid = labels.index[labels.notna().any(axis=1)]
    valid = valid[valid < predict_date]
    tr = valid[-window:]
    if len(tr) < MIN_TRAIN:
        raise ValueError(f"训练段不足 {MIN_TRAIN} 日（现 {len(tr)}）")
    log.info("训练: %d 日（%s ~ %s，窗=%d）→ 预测 %s", len(tr), tr[0].date(),
             tr[-1].date(), window, predict_date.date())

    p = LGBMPredictor(**DEFAULT_MODEL_PARAMS["gbdt"])
    p.fit({k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
    pred = p.predict({k: v.loc[[predict_date]] for k, v in feats.items()})
    meta = {"n_train_days": len(tr), "train_begin": str(tr[0].date()),
            "train_end": str(tr[-1].date()), "window": window}
    return pred.iloc[0], meta, p


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _write_latest(p: Path, src: Path) -> bool:
    """latest 稳定副本：临时文件 + 原子替换；目标被占用时告警不中断。

    latest_*.csv 只是当日主输出（ranking_{ds}.csv 等）的稳定路径副本，
    被外部进程（如预览/同步）短暂独占时不应拖垮整次运行。
    """
    try:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
        os.replace(tmp, p)
        return True
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        log.warning("latest 副本 %s 未更新（不影响当日主输出）: %s",
                    p.name, str(e)[:100])
        return False


def _write_table(path: Path, df: pd.DataFrame) -> bool:
    """主 CSV 原子写入：临时文件 + os.replace；目标被占用时返回 False。"""
    try:
        tmp = path.with_name(path.name + ".tmp")
        df.to_csv(tmp, encoding="utf-8-sig")
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


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
def install_task(time_str: str) -> str:
    """注册/更新每日计划任务（默认跑全流程含数据更新）。"""
    py = Path(SYSTEM_PY).resolve()
    script = (ROOT / "scripts" / "pipelines" / "alla_daily_rank.py").resolve()
    tr = f'\\"{py}\\" \\"{script}\\"'
    cmd = (f'schtasks /Create /F /TN "{TASK_NAME}" /SC DAILY /ST {time_str} '
           f'/TR "{tr}"')
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return f"cmd: {cmd}\n{out.strip()}"


def remove_task() -> str:
    cmd = f'schtasks /Delete /F /TN "{TASK_NAME}"'
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return f"cmd: {cmd}\n{out.strip()}"


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
    names, sel_year = load_selection(predict_date)
    log.info("特征选择: %d 个（选择年份 %d）", len(names), sel_year)

    # 预加载基本面长表：compute_features 与龙头股市值面板共用一次 IO
    long_keys: set[str] = set()
    if any(n in (_HOLDER_NUM_KEYS | _HOLDER_TOP_KEYS) for n in names):
        long_keys |= {"income", "balance", "cashflow", "equity", "dividend"}
    if any(n in _PLEDGE_KEYS for n in names):
        long_keys |= {"balance", "pledge", "notice", "express"}
    if any(n in _CONSTRUCTED_KEYS for n in names):
        long_keys |= {"income", "balance", "cashflow", "dividend"}
    long_tables = _load_long_tables(sorted(long_keys))

    feats = compute_features(names, tail, long_tables=long_tables)
    close_adj = tail["close_adj"]

    scores, train_meta, predictor = train_and_predict(feats, close_adj,
                                                      predict_date, args.window)
    # 幽灵股守卫：掩掉未上市/无行情/特征不可用的股票（同实验 stage_predict）
    valid = existence_mask(feats, close_adj, pd.DatetimeIndex([predict_date]))
    scores = scores.where(valid.iloc[0])
    n_raw = int(scores.notna().sum())

    tradable = signal_day_tradable(tail["close_raw"], predict_date)
    # 变量名必须与上面的特征名 names 区分：此处是「股票代码→简称」映射。
    # 曾因同名覆盖，meta["n_features"] 被写成股票数 5689（2026-09-07 history.csv 可见）。
    stock_names = load_stock_names(tail["close_raw"].columns)
    ind_panel = tail.get("industry")
    name_map = load_industry_names() if ind_panel is not None \
        else pd.Series(dtype=object)
    industry = industry_series(ind_panel, predict_date, name_map)
    industry = industry if len(industry) else None
    ranking, picks = build_ranking(scores, tradable, args.frac,
                                   names=stock_names, industry=industry)
    ind_table = build_industry_table(ranking) if "industry" in ranking.columns \
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

    # 龙头股视图：市值前 200 中模型分最高的 20 只
    cap_panel = compute_market_cap(tail["close_raw"], long_tables)
    mktcap = cap_panel.loc[predict_date]
    leaders_table = build_leaders(ranking, mktcap)
    log.info("龙头视图: 市值前 200 中模型 Top%d（市值口径 TOT_SHARE×收盘）",
             len(leaders_table))

    meta = {"model": "gbdt", **train_meta, "frac": args.frac,
            "n_features": len(names), "selection_year": sel_year,
            "n_scored": n_raw, "n_top_frac": int(ranking["top_frac"].sum()),
            "n_picks": len(picks),
            "n_industries": 0 if ind_table.empty else len(ind_table),
            "top5": "|".join(ranking.index[:5].astype(str)),
            "runtime_sec": round(time.time() - t0, 1)}
    paths = write_outputs(ranking, picks, predict_date, meta,
                          industry_table=ind_table,
                          feature_importance=imp_table,
                          explain_top=explain_top,
                          leaders=leaders_table)

    log.info("=" * 70)
    log.info("预测日 %s: 有分股票 %d | top%.0f%% %d | 可交易候选 %d | 行业 %d | 耗时 %.0fs",
             predict_date.date(), n_raw, args.frac * 100,
             meta["n_top_frac"], len(picks), meta["n_industries"],
             time.time() - t0)
    log.info("Top 20 预览 (code, name, 行业, score, 可交易):")
    for i, (code, row) in enumerate(ranking.head(20).iterrows(), 1):
        nm = row.get("name", "")
        ind = row.get("industry", "")
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
    ap.add_argument("--install-task", nargs="?", const="17:30", default=None,
                    metavar="HH:MM", help="注册每日 Windows 计划任务并退出")
    ap.add_argument("--remove-task", action="store_true", help="删除计划任务并退出")
    args = ap.parse_args()
    if args.install_task:
        print(install_task(args.install_task))
        return
    if args.remove_task:
        print(remove_task())
        return
    run(args)


if __name__ == "__main__":
    main()
