"""全A多年度滚动训练实验：horizon × 调仓频率 × 模型/超参 × 年份（2018~now）。

模拟真实预测场景：对每个目标年份 Y，特征选择与模型训练只使用 Y 年之前的
数据（walk-forward 前推 + embargo 隔离），逐年滚动产生 OOS 预测，拼接成
2018~今的连续样本外信号，再按不同组合参数（调仓频率/中性化/持仓集中度）
做向量化回测，产出分年收益曲线与绩效指标。

数据与因子（2026-09-01 就位）：
- 全A日线 2015~now（scripts/oneoff/fetch_alla_history.py 回填）；
- 公因子面板 alpha101/158/191/360 共 ~790 个（scripts/oneoff/build_alla_alpha_panels.py
  重建于 all_a_2018_2026 数据集，含 4 个 horizon 的日频 IC 缓存）；
- 风格中性化协变量：市值（equity_structure × 未复权 close）+ 申万一级行业
  + 动量/波动/换手（build_style_covariates）；
- 可执行性掩码：涨跌停封板/停牌/ST（2019 起，2018 缺状态表按可交易处理）。

实验网格：
- horizon ∈ {1, 5, 10, 20}；调仓频率：h1×{D,W,M}，h5/h10×M，h20×双月(2M)；
- 模型：{ridge, gbdt, ranker} 全 horizon；{gbdt_fast, gbdt_deep, 窗口750 变体}
  仅 h=1（超参/训练窗口敏感性）；
- 组合变体：风格中性化 on/off × TopFrac {20%, 10%}；
- 基准：上证指数 000001.SH + 全A等权。

纪律与诚实披露：
- 每年特征选择只用当年之前 500 交易日 IC 窗口 + 覆盖率 + 相关去冗余；
- 滚动训练 embargo=horizon，训练段严格早于测试折（forward_roll_folds 保证）；
- 本实验为**验证型网格**（全量报告，不挑赢家）；模型超参沿用项目固化默认，
  不在本数据上重新调参，避免二次数据窥探。

用法:
    python scripts/rolling_grid_alla.py --stage all        # 全流程（断点续跑）
    python scripts/rolling_grid_alla.py --stage predict --horizons 1
    python scripts/rolling_grid_alla.py --stage backtest
    python scripts/rolling_grid_alla.py --stage all --quick   # 冒烟：1年×h1×gbdt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402

log = setup_logging("rolling_grid_alla")

# ---------------------------------------------------------------------------
# 实验配置（单一真源）
# ---------------------------------------------------------------------------
DATASET = "all_a_2018_2026"
OUT = Path("reports") / "alla_rolling"
PANELS_DIR: Path | None = None   # --preproc ortho 时指向 panels_neu 中性化面板
# --preproc mixed：基本面/股东族逐因子指向 panels_neu，量价仍用 panels（混合口径）
NAME_DIR: dict[str, Path] | None = None
PANEL_BEGIN = 20160701          # 因子面板起点（预热后）
YEARS = list(range(2018, 2027))  # 目标年份（最后一年到数据末端，为部分年）
HORIZONS = [1, 5, 10, 20]

N_FOLDS = 4                     # 年内再训练折数（h<=5 季度再训练）
N_FOLDS_LONG = 2                # h>=10 用半年度再训练（长视野信号无需季度重训）
MIN_TRAIN = 250                 # 最少训练日（~1 年）
DEFAULT_WINDOW = 500            # 默认滚动训练窗（~2 年）
QUALITY_WINDOW = 500            # 特征选择 IC 质量窗（~2 年）
MAX_FEATURES = 50
MIN_COVERAGE = 0.5
DPP_SIGMA = 0.2                    # DPP 相似度核带宽（研报/项目正典默认）
DEDUP_TOP_CANDIDATES = 150      # 进入 DPP 去冗余的候选数
# 基本面慢变量族（A族股东结构 + B族财务报表）保留席位：IC 弱于量价、会被 DPP 质量项
# 边缘化而静默缺席，故每期为基本面族保留固定席位（按当期 IC 强度择优），其余席位由 DPP 填充。
RESERVED_FUNDAMENTAL_SLOTS = 8
HOLDER_SETS = {"holder_num_chg", "holder_num_yoy", "inst_holding",
               "top10_hhi", "top10_holding", "top1_holding", "top5_holding"}
# B族：财务报表基本面（价值/质量/成长/规模/杠杆/周转/分红）——慢变量、与量价低相关，
# 同样需保护进入 DPP 候选池（build_alla_fundamental_factors 构建）。
FUNDAMENTAL_SETS = {
    "ln_mktcap", "float_ratio",
    "ep_ttm", "bp", "sp_ttm", "pcf_ttm", "pcf", "fcf_yield", "netcf_yield", "peg",
    "div_yield", "div_yield_ttm", "div_payout_ratio",
    "roe_ttm", "roa_ttm", "gross_margin", "net_margin", "accruals", "roic_ttm",
    "fin_exp_ratio_ttm", "asset_turnover_ttm", "inv_turnover_ttm", "recv_turnover_ttm",
    "leverage", "rev_growth_yoy", "np_growth_yoy", "np_growth_sq_yoy", "np_growth_sq_qoq",
    "rev_growth_sq_yoy", "rev_growth_sq_qoq", "oppro_growth_sq_yoy", "oppro_growth_sq_qoq",
    "np_ded_growth_ttm_yoy", "cfo_growth_ttm_yoy",
    # B+ 族：商誉/股权质押/业绩预告/业绩快报（build_alla_pledge_factors 构建）
    "goodwill_ratio", "pledge_ratio", "pledge_holder_ratio", "frozen_ratio",
    "profit_notice_chg", "profit_express_np_yoy", "profit_express_rev_yoy",
    # B++ 族：构造型基本面（build_alla_constructed_factors 构建）
    "np_ded_ratio", "main_profit_ratio", "ebit_margin",
    "altman_zscore", "debt_to_ebitda", "interest_coverage",
    "np_rev_gap", "rev_recv_gap", "cfo_to_np",
    "margin_delta_ttm", "np_accel_sq", "roe_vol_pit",
    "risky_asset_ratio", "div_growth_yoy", "div_consecutive_years",
}
# 注：float_mktcap（流通市值）剔除——其 float_share 底层在 2021-01~04 覆盖 0.000 有数据洞，
# 且与 ln_mktcap 规模信息冗余；规模/流通维度由 ln_mktcap + float_ratio 覆盖。
# 低相关基本面慢变量族（A族股东结构 + B族财务报表），统合进 DPP 候选保护
FUNDAMENTAL_FAMILY_SETS = HOLDER_SETS | FUNDAMENTAL_SETS
# 消融开关：False 时 A+B 完全退出候选池/保留席位（对照组=纯量价）。--ablation 置 False，
# 并把 OUT 重定向到独立目录，保证两组除「是否有基本面族」外全部同口径。
INCLUDE_FUNDAMENTAL = True
FREQ_BY_HORIZON = {1: ["D", "W", "M"], 5: ["M"], 10: ["M"], 20: ["2M"]}
NEUT_VARIANTS = [True, False]
FRACS = [0.20, 0.10]
BENCH_INDEX = "000001.SH"       # 上证指数（全A 组合对照）

# 子网格 smallcap：聚焦验证 A股小盘溢价 + 集中度 + 信号加权。
SC_MODEL = "gbdt"               # 全网格核心最优形态
SC_HORIZON = 1
SC_FREQ = "M"
SC_FRACS = [0.20, 0.10, 0.05]
SC_WEIGHTS = ["equal", "factor"]


def _build_model_grid() -> dict[str, dict]:
    """模型网格：name -> (PREDICTORS key, params, train window)。

    超参单一真源：scripts/run_model_portfolio.DEFAULT_MODEL_PARAMS（项目固化值）。
    h=1 跑全网格（模型/超参/窗口变体）；h>1 只跑核心三模型（gbdt/ridge/ranker），
    控制全A规模下的总训练时长（模型×horizon 交互弱，主效应各自可辨）。
    """
    from scripts.run_model_portfolio import DEFAULT_MODEL_PARAMS
    return {
        "ridge":      dict(key="ridge", params={}, window=DEFAULT_WINDOW,
                           h1_only=False),
        "gbdt":       dict(key="gbdt",  params=dict(DEFAULT_MODEL_PARAMS["gbdt"]),
                           window=DEFAULT_WINDOW, h1_only=False),
        "ranker":     dict(key="ranker", params=dict(DEFAULT_MODEL_PARAMS["ranker"]),
                           window=DEFAULT_WINDOW, h1_only=False),
        # ---- 以下仅 h=1 ----
        "gbdt_fast":  dict(key="gbdt",  params=dict(n_estimators=80, learning_rate=0.05,
                                                    num_leaves=31, min_child_samples=50,
                                                    seed=0),
                           window=DEFAULT_WINDOW, h1_only=True),
        "gbdt_deep":  dict(key="gbdt",  params=dict(n_estimators=300, learning_rate=0.02,
                                                    num_leaves=31, min_child_samples=100,
                                                    seed=0),
                           window=DEFAULT_WINDOW, h1_only=True),
        # 训练窗口变体（验证 500 日 vs 750 日滚动窗）
        "gbdt_w750":  dict(key="gbdt",  params=dict(DEFAULT_MODEL_PARAMS["gbdt"]),
                           window=750, h1_only=True),
        "ridge_w750": dict(key="ridge", params={}, window=750, h1_only=True),
    }


# ---------------------------------------------------------------------------
# Phase A: 基础面板（一次构建，落盘复用）
# ---------------------------------------------------------------------------
def _shares_panel(eq: pd.DataFrame, index, columns) -> pd.DataFrame:
    """equity_structure 事件表 → 日频总股本面板（单位：股，groupby 向量化）。"""
    out = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    if eq is None or eq.empty or "tot_share" not in eq.columns:
        return out
    df = eq[eq["code"].isin(columns)][["code", "change_date", "tot_share"]].copy()
    df["change_date"] = pd.to_datetime(df["change_date"])
    df = df.dropna(subset=["tot_share"]).drop_duplicates(
        subset=["code", "change_date"], keep="last")
    wide = df.pivot(index="change_date", columns="code", values="tot_share").sort_index()
    wide = wide.reindex(index=sorted(set(wide.index) | set(index))).ffill()
    wide = wide.reindex(columns=columns)
    common = index.intersection(wide.index)
    out.loc[common] = wide.loc[common].to_numpy()
    return out * 10000.0


def stage_prep():
    """构建回测所需基础面板：复权价、风格协变量、可执行掩码、基准。"""
    from config import Config
    from data.cache import DataCache
    from data.cache_helpers import load_index_returns
    from data.tradability import build_tradable_mask
    from data.industry import IndustryClassification
    from data.offline import OfflineQuietDataSource
    from factor.preprocessing import build_style_covariates

    t0 = time.time()
    base = OUT / "_base"
    base.mkdir(parents=True, exist_ok=True)
    cache_root = Path(str(Config.cache()["root"]))

    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(
        daily.index.levels[0].normalize(), level=0)
    daily = daily[daily.index.get_level_values(0) >= pd.Timestamp(str(PANEL_BEGIN))]
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()

    def _panel(col):
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, hi, lo, c = _panel("open"), _panel("high"), _panel("low"), _panel("close")
    v = _panel("volume")
    raw_close = c.copy()

    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf = bf[[x for x in c.columns if x in bf.columns]]
    f = bf.reindex(index=c.index, columns=c.columns).ffill()
    for pnl in (o, hi, lo, c):
        pnl[:] = pnl.values * f.values
    log.info("复权面板: %d 日 × %d 股（%s ~ %s）", len(c), c.shape[1],
             c.index[0].date(), c.index[-1].date())
    c.astype(np.float32).to_parquet(base / "close_adj.parquet")
    raw_close.astype(np.float32).to_parquet(base / "close_raw.parquet")

    # 市值（未复权 close × 总股本，市场真实口径）
    eq = pd.read_parquet(cache_root / "equity_structure.parquet")
    shares = _shares_panel(eq, c.index, c.columns)
    mktcap = (shares * raw_close).astype(np.float32)
    mktcap.to_parquet(base / "market_cap.parquet")
    log.info("市值面板完成（覆盖 %.2f）", float(mktcap.notna().mean().mean()))

    # 风格协变量（中性化用）
    cache = DataCache(OfflineQuietDataSource())
    industry = IndustryClassification(cache, level=1).get_industry_panel(
        list(c.columns), c.index)
    industry.to_parquet(base / "industry.parquet")
    cov = build_style_covariates(
        {"close": c, "volume": v, "tot_share": shares},
        market_cap_panel=mktcap, industry_panel=industry)
    for k, pnl in cov.items():
        if k == "industry":
            pnl.to_parquet(base / f"cov_{k}.parquet")   # 字符串行业代码，原样保存
        else:
            pnl.astype(np.float32).to_parquet(base / f"cov_{k}.parquet")
    log.info("风格协变量: %s", list(cov))

    # 可执行性掩码（涨跌停封板/停牌/ST；状态表 2019 起，2018 视为可交易）
    mask = build_tradable_mask(c, bwd=f, cache_root=str(cache_root))
    mask.to_parquet(base / "tradable_mask.parquet")
    log.info("可执行掩码: 不可交易占比 %.4f", float((~mask).mean().mean()))

    # 基准：上证指数 + 全A等权
    idx_ret = load_index_returns(BENCH_INDEX, begin=int(str(PANEL_BEGIN)[:8]),
                                 real=True)
    if idx_ret is None:
        raise RuntimeError("上证指数基准不可用（index_daily 缓存缺失）")
    idx_ret.to_frame("ret").to_parquet(base / "bench_index.parquet")
    eqw = c.pct_change(fill_method=None).mean(axis=1)
    eqw.to_frame("ret").to_parquet(base / "bench_eqw.parquet")
    log.info("基准: 指数 %d 日 / 等权 %d 日（%s ~ %s）", len(idx_ret), len(eqw),
             idx_ret.index[0].date(), idx_ret.index[-1].date())
    log.info("prep 完成 %.0fs", time.time() - t0)


def load_base() -> dict:
    base = OUT / "_base"
    need = ["close_adj", "market_cap", "tradable_mask", "bench_index", "bench_eqw"]
    missing = [n for n in need if not (base / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"基础面板缺失 {missing}，先跑 --stage prep")
    close_adj = pd.read_parquet(base / "close_adj.parquet")
    cov = {}
    for k in ("size", "industry", "mom", "vol", "turn"):
        p = base / f"cov_{k}.parquet"
        if p.exists():
            cov[k] = pd.read_parquet(p)
    return {
        "close": close_adj,
        "market_cap": pd.read_parquet(base / "market_cap.parquet"),
        "cov": cov,
        "mask": pd.read_parquet(base / "tradable_mask.parquet"),
        "bench_index": pd.read_parquet(base / "bench_index.parquet")["ret"],
        "bench_eqw": pd.read_parquet(base / "bench_eqw.parquet")["ret"],
    }


def ds_root() -> Path:
    from config import Config
    return Path(str(Config.get()["factor_library"]["root"])) / DATASET


# ---------------------------------------------------------------------------
# Phase B: 特征筛选（每年 × horizon，仅用过去数据）
# ---------------------------------------------------------------------------
class FeatureStore:
    """因子面板 LRU 读取（float32，全A面板单个 ~50MB）。

    加载时统一做截面 zscore + ±10σ 剪裁（与因子库"入库即 zscore"及
    multiyear_oos 的加载口径对齐；全A原始公式值含爆炸量级——alpha191_017
    最大 ~3e38，直接进 Ridge 的稠密矩阵会 SVD 崩溃，2026-09-01 实证）。
    IC/去冗余均为秩相关，不受该单调变换影响。
    """

    def __init__(self, panels_dir: Path, cap: int = 130,
                 name_dir: dict[str, Path] | None = None):
        self.dir = panels_dir
        self.name_dir = name_dir or {}   # 逐因子目录覆盖（mixed 口径）
        self.cap = cap
        self._cache: dict[str, pd.DataFrame] = {}
        self._order: list[str] = []

    def get(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            if len(self._order) >= self.cap:
                evict = self._order.pop(0)
                self._cache.pop(evict, None)
            from factor.preprocessing import standardize_zscore
            src = self.name_dir.get(name, self.dir)
            p = pd.read_parquet(src / f"{name}.parquet")
            p = standardize_zscore(p).clip(-10.0, 10.0).astype(np.float32)
            self._cache[name] = p
            self._order.append(name)
        return self._cache[name]

    def get_many(self, names: list[str]) -> dict[str, pd.DataFrame]:
        return {n: self.get(n) for n in names}


def select_features_for_year(year: int, horizon: int, ic_cache: pd.DataFrame,
                             registry: pd.DataFrame, store: FeatureStore,
                             all_days: pd.DatetimeIndex,
                             cut: pd.Timestamp | None = None) -> list[str]:
    """某年 OOS 用的特征：质量窗（过去 QUALITY_WINDOW 日）IC + 覆盖率 + DPP 集合去冗余。

    去冗余用项目正典 DPP（research.dpp_selection::dpp_select，log-det 最大化）——
    研报国金 AlphaEval 框架确认：DPP 是收益端增益来源（图表40），两两贪心去重
    （pairwise_dedup）遇三角相关结构会连锁误杀、长 horizon 只剩 3~16 个特征，
    替换为 DPP 后在 h≥5 能跨族保留互补特征。
    """
    from research.dpp_selection import corr_matrix, dpp_select

    # 实验口径：质量窗截止=上年年末（年内冻结，防前视选择）；
    # 生产口径（run_model_portfolio）：传 cut=最新完整交易日，质量窗随数据滚动更新。
    cut = cut if cut is not None else pd.Timestamp(f"{year - 1}-12-31")
    q_days = all_days[all_days <= cut][-QUALITY_WINDOW:]
    ic_q = ic_cache.loc[ic_cache.index.intersection(q_days)]
    quality = ic_q.mean().abs().sort_values(ascending=False)

    cov_ok = registry.set_index("name")["coverage"]
    cands = [n for n in quality.index
             if n in cov_ok.index and cov_ok[n] >= MIN_COVERAGE][:DEDUP_TOP_CANDIDATES]
    # 基本面慢变量族（A族股东结构 + B族财务报表）IC 弱于量价：先为其保留固定席位
    # （按当期 |IC| 择优），其余席位交 DPP 去冗余填充，避免照 IC 排序或 DPP 质量项
    # 将其整体挤出而静默缺席。消融（INCLUDE_FUNDAMENTAL=False）时整个族退出。
    reserved: list[str] = []
    _extra: list[str] = []
    if INCLUDE_FUNDAMENTAL:
        ab_ok = [n for n in quality.index
                 if n in FUNDAMENTAL_FAMILY_SETS and cov_ok[n] >= MIN_COVERAGE]
        reserved = ab_ok[:min(len(ab_ok), RESERVED_FUNDAMENTAL_SLOTS)]
        _extra = [n for n in cov_ok.index
                  if n in FUNDAMENTAL_FAMILY_SETS and cov_ok[n] >= MIN_COVERAGE
                  and n not in cands and n not in reserved]
    cands = cands + _extra
    k_remain = max(0, MAX_FEATURES - len(reserved))
    if k_remain == 0:
        return reserved
    if reserved:
        cands = [n for n in cands if n not in reserved]

    sample_days = q_days[::4]
    sample_codes = store.get(cands[0]).columns[::2]
    sample = {n: store.get(n).reindex(index=sample_days, columns=sample_codes)
              for n in cands}
    corr = corr_matrix(sample, method="cross")
    res = dpp_select(corr, k=min(len(cands), k_remain),
                     quality=quality.reindex(cands).fillna(0.0), sigma=DPP_SIGMA)
    return reserved + res["selected"]


def stage_select(quick: bool = False):
    t0 = time.time()
    root = ds_root()
    registry = pd.read_csv(root / "registry.csv")
    store = FeatureStore(PANELS_DIR or root / "panels", name_dir=NAME_DIR)
    base = load_base()
    all_days = base["close"].index
    sel_dir = OUT / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)

    horizons = [1] if quick else HORIZONS
    years = [2019] if quick else YEARS
    for h in horizons:
        ic_cache = pd.read_parquet(root / f"ic_h{h}.parquet")
        for year in years:
            out = sel_dir / f"y{year}__h{h}.json"
            if out.exists():
                continue
            feats = select_features_for_year(year, h, ic_cache, registry,
                                             store, all_days)
            out.write_text(json.dumps(feats, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            log.info("选择 y%d h%d: %d 特征（%.0fs）", year, h, len(feats),
                     time.time() - t0)
    log.info("select 完成 %.0fs", time.time() - t0)


# ---------------------------------------------------------------------------
# Phase C: 滚动训练 → OOS 预测
# ---------------------------------------------------------------------------
def rolling_window_oos(predictor_cls, params, features: dict, labels: pd.DataFrame,
                       test_days, all_days, horizon: int, max_train: int | None):
    """前推滚动：test 年等分折；训练=折前历史（可选截尾 max_train 日）。"""
    from factor.cv import forward_roll_folds
    n_folds = N_FOLDS_LONG if horizon >= 10 else N_FOLDS
    folds = forward_roll_folds(all_days, test_days, n_folds, embargo_days=horizon)
    out = pd.DataFrame(np.nan, index=test_days, columns=labels.columns)
    for fold in folds:
        tr = fold.train_days
        if max_train:
            tr = tr[-max_train:]
        if len(tr) < MIN_TRAIN:
            log.warning("  折训练段不足（%d < %d），跳过", len(tr), MIN_TRAIN)
            continue
        p = predictor_cls(**params)
        p.fit({k: v.loc[tr] for k, v in features.items()}, labels.loc[tr])
        pred = p.predict({k: v.loc[fold.test_days] for k, v in features.items()})
        out.loc[fold.test_days] = pred.reindex(index=fold.test_days,
                                               columns=labels.columns)
        log.info("  折: train %d 日 -> 预测 %s ~ %s", len(tr),
                 fold.test_days[0].date(), fold.test_days[-1].date())
    if not out.notna().any().any():
        raise RuntimeError("无 OOS 预测产出")
    return out


def _existence_mask(feats: dict, close: pd.DataFrame,
                    test_days: pd.DatetimeIndex) -> pd.DataFrame:
    """股票在信号日的可用性掩码：当日有行情 且 >=1/4 特征非 NaN。

    全A滚动 IPO 场景的关键守卫（2026-09-01 quick 冒烟发现）：LightGBM 对
    全 NaN 特征行照样输出预测，未上市/新股会占满信号顶部（买入 1110 只
    幽灵股 → 组合近似空仓）。特征可用率掩码把预测限制在"当时真实存在、
    因子已有值"的股票上（PIT 正确）。
    """
    valid = close.loc[test_days].notna()
    if feats:
        avail = None
        for p in feats.values():
            a = p.loc[test_days].notna() if test_days.isin(p.index).all() \
                else p.reindex(index=test_days).notna()
            # 注意 bool+bool 在 pandas 2.x 下仍为 bool（0/1），须显式整型累加
            avail = a.astype(np.int16) if avail is None \
                else avail + a.astype(np.int16)
        valid = valid & (avail >= max(1, len(feats) // 4))
    return valid


def stage_predict(quick: bool = False, only_horizons: list[int] | None = None):
    from model.labels import build_labels
    from model.predictor import PREDICTORS
    from stats.ic import calc_ic_series

    t0 = time.time()
    root = ds_root()
    base = load_base()
    close = base["close"]
    all_days = close.index
    store = FeatureStore(PANELS_DIR or root / "panels", name_dir=NAME_DIR)
    models = _build_model_grid()
    if quick:
        models = {"gbdt": models["gbdt"]}
        horizons, years = [1], [2019]
    else:
        horizons = only_horizons or HORIZONS
        years = YEARS

    pred_dir = OUT / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for h in horizons:
        labels, _embargo = build_labels(close, horizon=h, mode="rank")
        fwd = close.pct_change(h, fill_method=None).shift(-h)
        for mname, mcfg in models.items():
            if mcfg.get("h1_only") and h != 1:
                continue  # 模型/超参/窗口变体只在 h=1 上验证
            out_path = pred_dir / f"{mname}__h{h}.parquet"
            if out_path.exists():
                continue
            t1 = time.time()
            parts = []
            for year in years:
                sel_path = OUT / "selection" / f"y{year}__h{h}.json"
                if not sel_path.exists():
                    # 兜底：predict 单独重跑时缺选择文件则现算（同口径无前视）
                    ic_cache = pd.read_parquet(root / f"ic_h{h}.parquet")
                    registry = pd.read_csv(root / "registry.csv")
                    sel_dir = OUT / "selection"
                    sel_dir.mkdir(parents=True, exist_ok=True)
                    feats_names = select_features_for_year(
                        year, h, ic_cache, registry, store, all_days)
                    sel_path.write_text(json.dumps(feats_names, ensure_ascii=False),
                                        encoding="utf-8")
                feats_names = json.loads(sel_path.read_text(encoding="utf-8"))
                feats = store.get_many(feats_names)
                feats = {k: v.reindex(index=all_days, columns=close.columns)
                         for k, v in feats.items()}
                test_days = all_days[(all_days >= pd.Timestamp(f"{year}-01-01")) &
                                     (all_days <= pd.Timestamp(f"{year}-12-31"))]
                log.info("[%s h%d] 年 %d: %d 特征, %d 测试日", mname, h, year,
                         len(feats), len(test_days))
                pred_y = rolling_window_oos(
                    PREDICTORS[mcfg["key"]], mcfg["params"], feats, labels,
                    test_days, all_days, h, mcfg["window"])
                # 幽灵股守卫：掩掉未上市/无行情/特征不可用的股票（见 _existence_mask）
                valid = _existence_mask(feats, close, test_days)
                n_before = int(pred_y.notna().sum().sum())
                pred_y = pred_y.where(valid)
                log.info("  掩码: 可用股票/日 中位 %d（掩前 %d）",
                         int(valid.sum(axis=1).median()), n_before // max(1, len(test_days)))
                parts.append(pred_y)
            pred = pd.concat(parts).astype(np.float32)
            pred.to_parquet(out_path)
            ic = calc_ic_series(pred, fwd)
            ic_by_year = ic.groupby(ic.index.year).mean()
            log.info("[%s h%d] 完成: OOS IC=%.4f | 分年 %s | %.0fs", mname, h,
                     float(ic.mean()),
                     {int(k): round(float(v), 4) for k, v in ic_by_year.items()},
                     time.time() - t1)
            _append_ic_stats(mname, h, ic)
    log.info("predict 完成 %.0fs", time.time() - t0)


def _append_ic_stats(model: str, horizon: int, ic: pd.Series) -> None:
    """追加/覆盖一行模型 OOS IC 统计（ic_stats.csv，报告用；幂等）。"""
    from backtest.metrics import PERIODS_PER_YEAR
    p = OUT / "ic_stats.csv"
    ic = ic.dropna()
    row = {
        "model": model, "horizon": horizon,
        "ic_mean": float(ic.mean()), "ic_std": float(ic.std()),
        "ic_ir": float(ic.mean() / ic.std() * np.sqrt(PERIODS_PER_YEAR))
        if ic.std() > 0 else 0.0,
        "ic_win": float((ic > 0).mean()),
        **{f"ic_{int(y)}": float(v) for y, v in ic.groupby(ic.index.year).mean().items()},
    }
    if p.exists():
        old = pd.read_csv(p)
        old = old[~((old["model"] == model) & (old["horizon"] == horizon))]
        old.to_csv(p, index=False, encoding="utf-8-sig")
    pd.DataFrame([row]).to_csv(p, mode="a", index=False, header=not p.exists(),
                               encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Phase D: 组合回测
# ---------------------------------------------------------------------------
def _rebalance_days_validated(dates: pd.DatetimeIndex, horizon: int,
                              freq: str) -> set:
    """月频/双月频调仓日集合，剔除区间跨度 < horizon 的调仓日。

    引擎对 h>1 要求每个调仓区间跨度 ≥ horizon（防重复结算）。样本末日的
    月首交易日（如 2026-09-01）到序列末跨度为 0，会触发守卫误杀整个回测
    ——该调仓日本就无收益可结（引擎也会跳过末日活动），剔除即可。
    """
    s = pd.Series(dates, index=dates)
    firsts = sorted(set(s.groupby(s.index.to_period("M")).first()))
    if freq == "2M":
        firsts = firsts[::2]
    pos = {d: i for i, d in enumerate(dates)}
    out = []
    for i, d in enumerate(firsts):
        nxt = firsts[i + 1] if i + 1 < len(firsts) else None
        span = (pos[nxt] - pos[d]) if nxt is not None else len(dates) - 1 - pos[d]
        if span >= horizon:
            out.append(d)
    return set(out)


def _ann(dr: pd.Series) -> float:
    dr = dr.dropna()
    if len(dr) < 20:
        return 0.0
    from backtest.metrics import PERIODS_PER_YEAR
    return float((1 + dr).prod() ** (PERIODS_PER_YEAR / len(dr)) - 1)


def _yearly_metrics(dr: pd.Series, bench: pd.Series) -> dict[int, dict]:
    from backtest.metrics import PERIODS_PER_YEAR
    out = {}
    for year, sub in dr.groupby(dr.index.year):
        sub = sub.dropna()
        if len(sub) < 20:
            continue
        b = bench.reindex(sub.index).fillna(0.0)
        ann = _ann(sub)
        b_ann = _ann(b)
        sd = sub.std()
        eq = (1 + sub).cumprod()
        out[int(year)] = {
            "annual": ann, "excess": ann - b_ann, "bench_annual": b_ann,
            "sharpe": float(sub.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else 0.0,
            "max_dd": float(abs((eq / eq.cummax() - 1).min())),  # 正幅度（同 metrics 约定）
        }
    return out


def stage_backtest(quick: bool = False):
    from backtest.engine import VectorBacktest
    from scripts.run_model_portfolio import default_costs, neutralize_panel
    from strategy.examples import TopFracLongOnly

    t0 = time.time()
    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    cov = base["cov"]
    costs = default_costs()
    eq_dir = OUT / "equity"
    eq_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted((OUT / "pred").glob("*.parquet"))
    if quick:
        pred_files = [p for p in pred_files if p.stem in ("gbdt__h1",)]
    rows_overall, rows_yearly = [], []
    for pf in pred_files:
        mname, hs = pf.stem.split("__")
        h = int(hs[1:])
        pred = pd.read_parquet(pf)
        oos_days = pred.index
        fwd = close.pct_change(fill_method=None) if h == 1 else \
            close.pct_change(h, fill_method=None).shift(-h)
        fwd = fwd.loc[oos_days]
        mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)

        sig_variants = {"raw": pred}
        try:
            sig_variants["neut"] = neutralize_panel(pred, cov)
        except Exception as e:  # noqa: BLE001
            log.warning("中性化失败（退化为 raw）: %s", str(e)[:80])

        bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
        bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)
        freqs = ["M"] if quick else FREQ_BY_HORIZON.get(h, ["M"])
        for freq in freqs:
            for sname, sig in sig_variants.items():
                for frac in FRACS:
                    stag = "neut" if sname == "neut" else "raw"
                    run_id = f"{mname}__h{h}__{freq}__{stag}__f{frac:.2f}"
                    eq_path = eq_dir / f"eq__{run_id}.csv"
                    if not eq_path.exists():
                        strat = TopFracLongOnly(frac=frac, weight_mode="equal")
                        rb = None
                        if h > 1:
                            # h>1 一律显式调仓日（月频/双月频 + 跨度守卫剔除末段）
                            rb = _rebalance_days_validated(oos_days, h, freq)
                        bt = VectorBacktest(strategy=strat,
                                            rebalance_freq=("M" if freq == "2M" else freq),
                                            initial_capital=1_000_000.0, costs=costs)
                        try:
                            res = bt.run(sig, fwd, executable_mask=mask_oos,
                                         horizon=h, rebalance_days=rb)
                        except ValueError as e:
                            log.warning("[%s] 不可行: %s", run_id, str(e)[:100])
                            continue
                        pd.DataFrame({
                            "equity": res.equity_curve,
                            "daily_ret": res.daily_returns,
                            "bench_index": bench_idx,
                            "bench_eqw": bench_eqw,
                            "turnover": res.turnover_series,
                        }).to_csv(eq_path, encoding="utf-8-sig")
                    eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
                    dr = eq["daily_ret"]
                    m = res_metrics(dr, bench_idx, eq.get("turnover"))

                    row = {
                        "run_id": run_id, "model": mname, "horizon": h,
                        "freq": freq, "neut": sname == "neut", "frac": frac,
                        "annual": m["annual"], "excess_idx": m["excess"],
                        "excess_eqw": m["annual"] - _ann(bench_eqw),
                        "sharpe": m["sharpe"],
                        "sharpe_t_stat": m["sharpe_t_stat"],
                        "years_to_prove": m["years_to_prove"],
                        "excess_t_stat": m["excess_t_stat"],
                        "ir": m["ir"],
                        "max_dd": m["max_dd"], "turnover": m["turnover"],
                        "beta": m["beta"], "n_days": len(eq),
                    }
                    rows_overall.append(row)
                    for year, ym in _yearly_metrics(dr, bench_idx).items():
                        rows_yearly.append({**row, "year": year, **ym})
                    log.info("[%s] 年化=%.2f%% 超额(指数)=%+.2f%% Sharpe=%.2f 换手=%.1f%%",
                             run_id, row["annual"] * 100, row["excess_idx"] * 100,
                             row["sharpe"], row["turnover"] * 100)

    pd.DataFrame(rows_overall).to_csv(OUT / "metrics_overall.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_yearly).to_csv(OUT / "metrics_yearly.csv",
                                     index=False, encoding="utf-8-sig")
    log.info("backtest 完成 %.0fs（%d 组合）", time.time() - t0, len(rows_overall))


def res_metrics(dr: pd.Series, bench: pd.Series, turnover: pd.Series | None) -> dict:
    """从日收益序列计算整体指标（首跑与断点续跑同口径）。"""
    from backtest.metrics import calc_all_metrics
    dr = dr.dropna()
    m = calc_all_metrics(dr, bench.reindex(dr.index).fillna(0.0), None)
    to = turnover.dropna() if turnover is not None else pd.Series(dtype=float)
    return {
        "annual": m.get("annual_return", 0.0),
        "excess": m.get("excess_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "sharpe_t_stat": m.get("sharpe_t_stat", np.nan),
        "years_to_prove": m.get("years_to_prove", np.nan),
        "excess_t_stat": m.get("excess_t_stat", np.nan),
        "ir": m.get("information_ratio", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "beta": m.get("beta", np.nan),
        "turnover": float(to.mean()) if len(to) else 0.0,
    }


# ---------------------------------------------------------------------------

def stage_smallcap(quick: bool = False):
    """聚焦子网格：验证 A股小盘溢价 × 集中度 × 信号加权 的增厚空间。

    锁全网格核心最优形态（SC_MODEL×SC_HORIZON×SC_FREQ），扫描：
      - 中性化口径：raw（不中性）/ inds（**只行业中性**、保留市值/小盘暴露）
        —— 当前主线 neut 把 log市值一同回归清零，抹平小盘溢价；inds 只压平
        行业集中、保留小盘，是本子网格验证的核心改动。
      - frac：{0.20, 0.10, 0.05}，越集中 alpha 浓度越高、波动也越大；
      - weight：equal（等权）/ factor（按信号强度加权，强信号更大仓位）。
    共 2×3×2=12 个组合。产出 equity_smallcap/*。csv + metrics_smallcap.csv。

    推断对齐：h=1 日频收益，月频调仓（M），小盘换手高——成本用项目固化
    default_costs（对小盘滑点略乐观，结论仅作方向参考）。
    """
    from backtest.engine import VectorBacktest
    from factor.preprocessing import neutralize
    from scripts.run_model_portfolio import default_costs
    from strategy.examples import TopFracLongOnly

    t0 = time.time()
    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    cov = base["cov"]
    costs = default_costs()

    mname, h, freq = SC_MODEL, SC_HORIZON, SC_FREQ
    pf = OUT / "pred" / f"{mname}__h{h}.parquet"
    if not pf.exists():
        raise FileNotFoundError(f"缺少预测 {pf}，先跑 --stage predict")

    pred = pd.read_parquet(pf)
    oos_days = pred.index
    fwd = close.pct_change(fill_method=None)   # h=1 单日收益，引擎约定
    fwd = fwd.loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)

    # 中性化口径：raw 原信号；inds 只回归行业哑变量（跳过 size/动量/波动/换手）
    sig_variants = {"raw": pred}
    if cov.get("industry") is not None:
        sig_variants["inds"] = neutralize(
            pred, market_cap_panel=None, industry_panel=cov["industry"])
    log.info("[smallcap] 信号变体: %s", list(sig_variants))

    rb = None  # h=1 用引擎默认月频日历调仓
    eq_dir = OUT / "equity_smallcap"
    eq_dir.mkdir(parents=True, exist_ok=True)
    rows_overall, rows_yearly = [], []
    for sname, sig in sig_variants.items():
        for frac in SC_FRACS:
            for wmode in SC_WEIGHTS:
                run_id = f"{mname}__h{h}__{freq}__{sname}__f{frac:.2f}__{wmode}"
                eq_path = eq_dir / f"eq__{run_id}.csv"
                strat = TopFracLongOnly(frac=frac, weight_mode=wmode)
                bt = VectorBacktest(strategy=strat, rebalance_freq=freq,
                                    initial_capital=1_000_000.0, costs=costs)
                try:
                    res = bt.run(sig, fwd, executable_mask=mask_oos,
                                 horizon=h, rebalance_days=rb)
                except ValueError as e:
                    log.warning("[%s] 不可行: %s", run_id, str(e)[:100])
                    continue
                pd.DataFrame({
                    "equity": res.equity_curve,
                    "daily_ret": res.daily_returns,
                    "bench_index": bench_idx,
                    "bench_eqw": bench_eqw,
                    "turnover": res.turnover_series,
                }).to_csv(eq_path, encoding="utf-8-sig")

                dr = res.daily_returns
                m = res_metrics(dr, bench_idx, res.turnover_series)
                row = {
                    "run_id": run_id, "model": mname, "horizon": h,
                    "freq": freq, "neut": sname, "frac": frac, "weight": wmode,
                    "annual": m["annual"], "excess_idx": m["excess"],
                    "excess_eqw": m["annual"] - _ann(bench_eqw),
                    "sharpe": m["sharpe"], "ir": m["ir"],
                    "max_dd": m["max_dd"], "turnover": m["turnover"],
                    "beta": m["beta"], "n_days": len(dr.dropna()),
                }
                rows_overall.append(row)
                for year, ym in _yearly_metrics(dr, bench_idx).items():
                    rows_yearly.append({**row, "year": year, **ym})
                log.info("[%s] 年化=%.2f%% 超额(指数)=%+.2f%% 超额(等权)=%+.2f%% "
                         "Sharpe=%.2f 换手=%.1f%%", run_id, row["annual"] * 100,
                         row["excess_idx"] * 100, row["excess_eqw"] * 100,
                         row["sharpe"], row["turnover"] * 100)

    pd.DataFrame(rows_overall).to_csv(OUT / "metrics_smallcap.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_yearly).to_csv(OUT / "metrics_smallcap_yearly.csv",
                                     index=False, encoding="utf-8-sig")
    log.info("smallcap 完成 %.0fs（%d 组合）", time.time() - t0, len(rows_overall))


def _rank_average(panels: list[pd.DataFrame], min_panels: int = 2) -> pd.DataFrame:
    """逐日截面秩平均（与 calc_ic 的 Spearman 同口径的单调变换）。

    对每个 (日期, 股票)：在 k 个模型预测里取截面百分位秩，仅当当日
    有效面板数 >= min_panels 时求平均，否则置 NaN —— 避免单一模型
    的离群值/全缺失股票凭凑数进入合成信号。秩平均使不同模型量纲
    （ridge 值 / gbdt 分 / ranker 分）可比，集成在"排序信息"层面。
    """
    out = pd.DataFrame(np.nan, index=panels[0].index, columns=panels[0].columns)
    if not panels:
        return out
    codes = panels[0].columns
    for d in panels[0].index:
        ranks = []
        for p in panels:
            if d not in p.index:
                continue
            row = p.loc[d]
            r = row.rank(pct=True)
            r = r[~row.isna()]
            ranks.append(r)
        # 对齐当日所有面板的有效代码，取交集计数
        valid_codes = ranks[0].index if ranks else codes[:0]
        for r in ranks[1:]:
            valid_codes = valid_codes.intersection(r.index)
        if len(valid_codes) < 1 or len(ranks) < min_panels:
            continue
        sub = [r.reindex(valid_codes) for r in ranks]
        ok = pd.concat(sub, axis=1).dropna(axis=0)
        if len(ok) < 30:
            out.loc[d, valid_codes] = pd.concat(sub, axis=1).mean(axis=1)
            continue
        out.loc[d, ok.index] = ok.mean(axis=1).astype(np.float32)
    return out


def stage_ensemble(quick: bool = False):
    """多模型集成 + 跨 horizon 合成子网格。

    用户确认方向（2026-09-03）：做 ridge+gbdt+ranker 秩平均、h1+h5 信号合成，
    检验信号端集成能否突破 single-model 的超额天花板（smallcap 最优 9.7%）。

    合成变体：
      ens_rgr  : ridge+gbdt+ranker 的 h1 截面秩平均（跨模型去单一噪声）
      ens_h1h5 : gbdt 的 h1 与 h5 信号截面秩平均（long+h5 视野叠加）
      ens_all  : ens_rgr 再与 gbdt h5/h10 合成（模型×horizon 全集成）
    回测形态：跟 smallcap 最优口径（M 月频, inds 只行业中性, equal 等权），
    frac 用 0.10 / 0.20 两组；对照 raw（不中性）同 frac。产出
    pred/ens_*.parquet + equity_ensemble/*.csv + metrics_ensemble.csv。
    """
    from backtest.engine import VectorBacktest
    from factor.preprocessing import neutralize
    from scripts.run_model_portfolio import default_costs
    from stats.ic import calc_ic_series
    from strategy.examples import TopFracLongOnly

    t0 = time.time()
    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    cov = base["cov"]
    costs = default_costs()
    h, freq, wmode = 1, "M", "equal"
    fwd = close.pct_change(fill_method=None).reindex(close.index)

    def _load(m):
        return pd.read_parquet(OUT / "pred" / f"{m}.parquet")

    if not (OUT / "pred" / "ridge__h1.parquet").exists():
        raise FileNotFoundError("缺 ridge__h1，先跑 --stage predict")

    variants = {}
    variants["ens_rgr"] = _rank_average([_load("ridge__h1"),
                                         _load("gbdt__h1"), _load("ranker__h1")])
    variants["ens_h1h5"] = _rank_average([_load("gbdt__h1"),
                                          _load("gbdt__h5")])
    vars_all = [_load("ridge__h1"), _load("gbdt__h1"), _load("ranker__h1"),
                _load("gbdt__h5"), _load("gbdt__h10")]
    variants["ens_all"] = _rank_average(vars_all)

    pred_dir = OUT / "pred"
    for vname, v in variants.items():
        v.reindex(columns=close.columns).astype(np.float32).to_parquet(
            pred_dir / f"{vname}__h{h}.parquet")
        log.info("[ensemble] %s -> OOS IC=%.4f",
                 vname, float(calc_ic_series(v, fwd.shift(-1)).mean()))

    # 回测窗口 = 集成信号自身的索引（2018+）。不可用 close 全索引：预热期
    # （2016-2017）无信号空仓、收益恒 0，却计入基准对比，会把年化与超额
    # 系统性稀释（2026-09-08 ortho 集成实测 +9.9% vs 主口径 +13.3%）。
    oos_days = variants["ens_rgr"].index
    fwd = fwd.loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)

    eq_dir = OUT / "equity_ensemble"
    eq_dir.mkdir(parents=True, exist_ok=True)
    rows_overall, rows_yearly = [], []
    for hname, sig in variants.items():
        sig = sig.reindex(index=oos_days, columns=close.columns)
        neut_variants = {"raw": sig}
        if cov.get("industry") is not None:
            neut_variants["inds"] = neutralize(
                sig, market_cap_panel=None, industry_panel=cov["industry"])
        for sname, sg in neut_variants.items():
            for frac in ([0.10] if quick else [0.10, 0.20]):
                run_id = f"ens_{hname}__h{h}__{freq}__{sname}__f{frac:.2f}__{wmode}"
                eq_path = eq_dir / f"eq__{run_id}.csv"
                strat = TopFracLongOnly(frac=frac, weight_mode=wmode)
                bt = VectorBacktest(strategy=strat, rebalance_freq=freq,
                                    initial_capital=1_000_000.0, costs=costs)
                try:
                    res = bt.run(sg, fwd, executable_mask=mask_oos,
                                 horizon=h, rebalance_days=None)
                except ValueError as e:
                    log.warning("[%s] 不可行: %s", run_id, str(e)[:100])
                    continue
                pd.DataFrame({
                    "equity": res.equity_curve, "daily_ret": res.daily_returns,
                    "bench_index": bench_idx, "bench_eqw": bench_eqw,
                    "turnover": res.turnover_series,
                }).to_csv(eq_path, encoding="utf-8-sig")
                dr = res.daily_returns
                m = res_metrics(dr, bench_idx, res.turnover_series)
                row = {
                    "run_id": run_id, "ensemble": hname, "horizon": h,
                    "freq": freq, "neut": sname, "frac": frac, "weight": wmode,
                    "annual": m["annual"], "excess_idx": m["excess"],
                    "excess_eqw": m["annual"] - _ann(bench_eqw),
                    "sharpe": m["sharpe"], "ir": m["ir"],
                    "max_dd": m["max_dd"], "turnover": m["turnover"],
                    "beta": m["beta"], "n_days": len(dr.dropna()),
                }
                rows_overall.append(row)
                for year, ym in _yearly_metrics(dr, bench_idx).items():
                    rows_yearly.append({**row, "year": year, **ym})
                log.info("[%s] 年化=%.2f%% 超额(指数)=%+.2f%% 超额(等权)=%+.2f%% "
                         "Sharpe=%.2f 换手=%.1f%%", run_id, row["annual"] * 100,
                         row["excess_idx"] * 100, row["excess_eqw"] * 100,
                         row["sharpe"], row["turnover"] * 100)

    pd.DataFrame(rows_overall).to_csv(OUT / "metrics_ensemble.csv",
                                      index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_yearly).to_csv(OUT / "metrics_ensemble_yearly.csv",
                                     index=False, encoding="utf-8-sig")
    log.info("ensemble 完成 %.0fs（%d 组合）", time.time() - t0, len(rows_overall))


def main():
    ap = argparse.ArgumentParser(description="全A多年度滚动训练实验")
    ap.add_argument("--stage", default="all",
                    choices=["prep", "select", "predict", "backtest", "smallcap",
                             "ensemble", "all"])
    ap.add_argument("--quick", action="store_true", help="冒烟模式（1年×h1×gbdt×M）")
    ap.add_argument("--horizons", default=None,
                    help="predict 阶段限定 horizon，逗号分隔")
    ap.add_argument("--ablation", action="store_true",
                    help="消融对照组：排除 A+B 基本面族，输出到 alla_rolling_nofund")
    ap.add_argument("--preproc", default="zscore",
                    choices=["zscore", "ortho", "mixed"],
                    help="特征预处理口径：zscore(全局截面z) / ortho(因子层行业+市值"
                         "中性化) / mixed(基本面族中性化+量价zscore)")
    args = ap.parse_args()

    global INCLUDE_FUNDAMENTAL, OUT, PANELS_DIR, NAME_DIR
    if args.ablation:
        INCLUDE_FUNDAMENTAL = False
        OUT = Path("reports") / "alla_rolling_nofund"
        log.info("+++ 消融模式：关闭 A+B 基本面族，输出 -> %s", OUT)
    if args.preproc == "ortho":
        OUT = Path("reports") / "alla_rolling_ortho"
        PANELS_DIR = ds_root() / "panels_neu"
        log.info("+++ 因子层正交化：特征源 -> panels_neu，输出 -> %s", OUT)
    if args.preproc == "mixed":
        OUT = Path("reports") / "alla_rolling_mixed"
        _neu = ds_root() / "panels_neu"
        NAME_DIR = {n: _neu for n in FUNDAMENTAL_FAMILY_SETS | {"float_mktcap"}}
        log.info("+++ 混合口径：基本面/股东族 %d 因子 -> panels_neu，量价 -> panels，"
                 "输出 -> %s", len(NAME_DIR), OUT)

    OUT.mkdir(parents=True, exist_ok=True)
    stages = [args.stage] if args.stage != "all" else \
        ["prep", "select", "predict", "backtest"]
    only_h = [int(x) for x in args.horizons.split(",")] if args.horizons else None
    for st in stages:
        log.info("===== 阶段: %s =====", st)
        if st == "prep":
            stage_prep()
        elif st == "select":
            stage_select(args.quick)
        elif st == "predict":
            stage_predict(args.quick, only_h)
        elif st == "backtest":
            stage_backtest(args.quick)
        elif st == "smallcap":
            stage_smallcap(args.quick)
        elif st == "ensemble":
            stage_ensemble(args.quick)


if __name__ == "__main__":
    main()
