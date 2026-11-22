"""飞书《因子字典》候选因子 → 因子库体检入库
=============================================

对 ``factor/factor_dict_factors.py`` 里的 18 个候选跑与 ``build_intraday_stat_factors``
同一套体检：截面 zscore 入库 ``FactorLibrary``（IC / NW-t / IC 衰减 / canonical
回测 / 冗余查重全套自动算），再补一份**安慰剂对照**（``research.caliber`` 的
循环平移）。

候选来源与筛选理由见 ``reports/factor_dict_probe/README.md``。

用法
----
    python -m scripts.factors.build_fd_factors --offline
    python -m scripts.factors.build_fd_factors --offline --no-save
    python -m scripts.factors.build_fd_factors --offline --features fd_tcc_g,fd_ent30
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache import DataCache  # noqa: E402
from data.cache_helpers import (  # noqa: E402
    load_backward_factor,
    load_daily,
    load_financial_tables,
)
from data.financials import build_pit_panel  # noqa: E402
from data.intraday import MinutePanelStore  # noqa: E402
from data.offline import OfflineDataSource  # noqa: E402
from data.universe import Universe  # noqa: E402
from factor.factor_dict_factors import (  # noqa: E402
    abnormal_ret_max_panel,
    amt_lead_daily,
    attention_spill_panel,
    corr_lagret,
    cs_spread_panel,
    csad_std21_panel,
    dazzle_daily,
    ds_significance_daily,
    elasticity_daily,
    emo_spill_panel,
    entropy_amount_share,
    ex_rtn_max_daily,
    fear_panel,
    foc_autocorr,
    group_abnormal_panel,
    ifle_daily,
    illiq_panel,
    ind_joint_momentum_panel,
    jump_daily,
    lcp_daily,
    mainforce_vol_daily,
    mmd_asym_panel,
    net_support_panel,
    network_centrality_panel,
    neutralize_panel,
    peak_count_daily,
    peak_ridge_daily,
    price_delay_panel,
    rev_imp_pos_daily,
    rnbr_panel,
    sigup_pv_daily,
    smart_vwap_panel,
    tcc_panel,
    traj_illiq_daily,
    turnover_resid_vol_panel,
    umr_panel,
    undercurrent_daily,
    utr_panel,
    wind_tide_daily,
)
from research.factor_criteria import (  # noqa: E402
    ADMISSION_GATES,
    cyclic_placebo,
    effect_size,
    panel_coverage,
)
from scripts.common.cli_common import (  # noqa: E402
    record_experiment_safe,
    register_panels,
    setup_logging,
)
from stats.ic import calc_ic_series, ic_stats  # noqa: E402

log = setup_logging("build_fd_factors")

FACTOR_DEFS: dict[str, str] = {
    "fd_tcc": "时间网络相对中心度(全截面) = 1/mean_21( 日收益全截面zscore² )",
    "fd_tcc_g": "时间网络相对中心度(邪修) = 1/mean_21( 日收益在[申万一级×市值三档]组内zscore² )",
    "fd_grp_abn": "分组异常收益 = max_21( |日收益 − [行业×市值三档]组内均值| )",
    "fd_rnbr": "自信溢出 = std_21( 日收益对代码邻接10只收益均值做带截距截面OLS的残差 )",
    "fd_ent30": "单一成交额占比熵(30min) = −Σ p·ln p，p=(close_t/Σclose)×(vol_t/Σvol)",
    "fd_csad21": "CSAD模型 std_21 = std_21( 过去120日相关性Top10板块内 |r_i − r_self| 的截面均值 )",
    "fd_mmd21": "收益率不对称性(日频近似) = mean_21( MMD(过去60日日收益, −x+2μ) )",
    "fd_tr_resid": "特质换手波动率 = std_21( 换手率对[市场/SMB/HML/MOM换手]时序回归的残差 )",
    "fd_illiq_ds": "非流动性(DS版) = |0.6·(H−L)/O + 0.4·(O_t/C_{t−1}−1)|^1.5 / (成交额^0.7 + 5e6)",
    "fd_umr_tr": "UMR2.0(换手风险) = mean_21( 超额收益 × (mean_21(换手) − 换手) )",
    "fd_umr_trv": "UMR2.0(真实波动风险) = mean_21( 超额收益 × (mean_21(TR) − TR) )",
    "fd_fear_std": "原始惊恐因子 = std_21( 惊恐度 × 个股收益 )，惊恐度=|r−r_m|/(|r|+|r_m|+0.1)",
    "fd_hf_cs": "CS价差(分钟) = std_21( 日内 Corwin-Schultz 价差估计的均值 )",
    "fd_smart_vwap": "聪明钱(跟随vwap) = 跟随时刻vwap / 全日vwap（成交量Top10时刻的后一根）",
    "fd_net_support": "净支撑成交量 = (支撑量 − 阻力量) / 总股本，支撑=分钟收盘价低于日均价",
    "fd_utr": "优加换手率 = (Score1 + Score2)/N，量小/量稳分别市值中性化后按 STR 分两组反向排名",
    "fd_ind_mom": "行业联合动量CMC = ICM − ICR（过去20日最大5日/最小15日对应的市值加权行业收益）",
    "fd_price_delay": "价格时滞 = 1 − R²受限/R²全（个股收益对当期+滞后3期市场收益回归）",
    # ---- 第五轮（2026-11-22）：字典新候选 ----
    "fd_pk_ridge": "峰岭成交比 = 21日量峰成交额和 / 21日量岭成交额和（过去20日同时点量均值+1σ为喷发；孤立=峰，连续=岭；5min）",
    "fd_pk_cnt": "成交额波峰计数 = 日内成交额 ≥ 当日均值+1σ 且前一根未喷发的次数（5min）",
    "fd_main_vol_up": "主力波动率(放量上涨) = std_20( 两次截面标准化后的|放量上涨时刻分钟收益均值| )（5min，放量=量>前一根）",
    "fd_ds_std": "intraDSStd_byT = max_8段 |σ_s−mean(σ)|/(σ_s+mean|σ|)，σ=30分钟段内5min收益标准差",
    "fd_ds_vol": "intraDSVol_byT = max_8段 |v_s−mean(v)|/(v_s+mean|v|)，v=30分钟段成交量占全日比",
    "fd_sigup_pvr": "价格显著上行价稳量比 = 21日均值( std(r|显著上行)·mean(vol|显著上行)/mean(vol) )",
    "fd_sigup_pvr_s": "价格显著上行价稳量比稳定 = sqrt(21日均值(日值²))，日值同上",
    "fd_attn_spill": "注意力溢出 = 过去21日异常收益平方均值 − [申万一级×市值30/40/30]同组均值",
    "fd_dazzle_vol": "耀眼波动率 = std_21( 成交量激增时刻后20分钟的|5min收益|均值 )",
    "fd_dazzle_ret": "耀眼收益率 = std_21( 成交量激增时刻的5min收益均值 )",
    "fd_undercurrent": "暗流涌动 = 21日均值( |日内成交量分布熵 − 截面均值| )（5min 48桶）",
    # ---- 第六轮（2026-11-22）：Fitness 1.7~2.0 去重候选 ----
    "fd_abn_ret_max": "最大异常日收益率 = max_21( |日收益 − 当日截面中位数| )（abnormal_rtn 同式合并）",
    "fd_net_cent": "空间网络相对中心度 = 1/(2(1−ρ))，ρ=过去21日收益两两相关去自身后的均值（不含当日）",
    "fd_mod_amp": "修正的振幅 = mean_21( 振幅 × sign(日内跳跃度 > 当日截面均值) )",
    "fd_jump": "日内跳跃度 = mean_21( 2·mean(s−c) − mean(c²) )，s=单利 c=复利，5min 去首尾",
    "fd_traj_illiq": "轨迹非流动性 = mean_21( ln Π(|close/open−1|+1) / 当日换手 )（5min）",
    "fd_corr_lagret_tr": "滞后|收益|与换手率相关 = corr( |ln(1+r)|.shift(1), 同分钟21日zscore(换手) )",
    "fd_corr_lagret_amt": "滞后|收益|与成交额相关 = corr( |ln(1+r)|.shift(1), 同分钟21日zscore(成交额) )",
    "fd_foc_vol": "成交量一阶自相关 = mean_20( 当日量喷发01序列的一阶自相关 )（21日同时点 μ+σ）",
    "fd_foc_amt": "成交额一阶自相关 = mean_20( 当日额喷发01序列的一阶自相关 )（21日同时点 μ+σ）",
    "fd_ifle": "日内分形流动熵 = std_21( FD·(1−LE)·AD·sign(收盘−开盘) )（5min）",
    "fd_elastic": "高频弹性 = mean_21( 对数价HP周期项FFT幅度加权频率均值 )（5min）",
    "fd_lcp": "卖出反弹偏离 = mean_21( Σ卖出反弹额/Σ量 ÷ 当日收盘 − 1 )（5min）",
    "fd_rev_imp_pos": "改进的正收益反转 = mean_21( −mean(放量上涨分钟的分钟收益) )（5min）",
    "fd_amt_lead": "领先成交额异动 = mean_21( mean(m_t|次bar收益<10%分位)/mean(m_t) )（5min）",
    "fd_ex_rtn_max5": "收益率极大值幅度 = mean_21( mean(r|r>行业×市值组内95%分位)/分位阈值 )（5min）",
    "fd_emo_spill": "情绪溢出 = std_21( 换手率对代码邻接10只换手均值做带截距截面OLS的残差 )",
    "fd_wind_tide": "风起潮随 = mean_21( 量潮上行×涨幅到收盘共振的日值 )，5×MAD+行业+市值中性",
}

FAMILY = {"fd_tcc": "波动率", "fd_tcc_g": "波动率", "fd_grp_abn": "波动率",
          "fd_rnbr": "波动率", "fd_ent30": "日内",
          "fd_csad21": "波动率", "fd_mmd21": "波动率", "fd_tr_resid": "流动性",
          "fd_illiq_ds": "流动性", "fd_umr_tr": "量价", "fd_umr_trv": "量价",
          "fd_fear_std": "波动率",
          "fd_hf_cs": "流动性", "fd_smart_vwap": "资金流", "fd_net_support": "量价",
          "fd_utr": "流动性", "fd_ind_mom": "动量", "fd_price_delay": "动量",
          "fd_pk_ridge": "日内", "fd_pk_cnt": "日内", "fd_main_vol_up": "波动率",
          "fd_ds_std": "波动率", "fd_ds_vol": "日内",
          "fd_sigup_pvr": "量价", "fd_sigup_pvr_s": "量价",
          "fd_attn_spill": "情绪", "fd_dazzle_vol": "波动率",
          "fd_dazzle_ret": "量价", "fd_undercurrent": "日内",
          "fd_abn_ret_max": "波动率", "fd_net_cent": "动量",
          "fd_mod_amp": "波动率", "fd_jump": "波动率",
          "fd_traj_illiq": "流动性", "fd_corr_lagret_tr": "流动性",
          "fd_corr_lagret_amt": "流动性", "fd_foc_vol": "流动性",
          "fd_foc_amt": "流动性", "fd_ifle": "日内",
          "fd_elastic": "波动率", "fd_lcp": "动量",
          "fd_rev_imp_pos": "动量", "fd_amt_lead": "动量",
          "fd_ex_rtn_max5": "波动率", "fd_emo_spill": "情绪",
          "fd_wind_tide": "动量"}

#: 从 5 分钟面板出发的因子（池由 `--minute-pool` / `--entropy-pool` 决定；
#: 2026-11-22 起默认全 A，hs300 仅用于复现第四轮历史口径）
MINUTE_FACTORS = frozenset({"fd_ent30", "fd_hf_cs", "fd_smart_vwap",
                            "fd_net_support",
                            "fd_pk_ridge", "fd_pk_cnt", "fd_main_vol_up",
                            "fd_ds_std", "fd_ds_vol", "fd_sigup_pvr",
                            "fd_sigup_pvr_s", "fd_dazzle_vol", "fd_dazzle_ret",
                            "fd_undercurrent",
                            "fd_jump", "fd_traj_illiq", "fd_corr_lagret_tr",
                            "fd_corr_lagret_amt", "fd_foc_vol", "fd_foc_amt",
                            "fd_ifle", "fd_elastic", "fd_lcp",
                            "fd_rev_imp_pos", "fd_amt_lead", "fd_ex_rtn_max5"})

#: 第六轮新增的分钟频因子（统一走 5 分钟流式块；fd_mod_amp 由 jump 派生、
#: fd_wind_tide 在主流程再做中性化，二者不入此集合，但内核仍在此块计算）
ROUND6_MINUTE = frozenset({"fd_jump", "fd_mod_amp", "fd_traj_illiq",
                           "fd_corr_lagret_tr", "fd_corr_lagret_amt",
                           "fd_foc_vol", "fd_foc_amt", "fd_ifle",
                           "fd_elastic", "fd_lcp", "fd_rev_imp_pos",
                           "fd_amt_lead", "fd_ex_rtn_max5", "fd_wind_tide"})

#: 第五轮新增的分钟频因子（走统一的 5 分钟流式块）
ROUND5_MINUTE = frozenset({"fd_pk_ridge", "fd_pk_cnt", "fd_main_vol_up",
                           "fd_ds_std", "fd_ds_vol", "fd_sigup_pvr",
                           "fd_sigup_pvr_s", "fd_dazzle_vol", "fd_dazzle_ret",
                           "fd_undercurrent"})


# ----------------------------------------------------------------------
# 数据装载
# ----------------------------------------------------------------------
def load_daily_context(begin: int, end: int, *, with_prices: bool = False):
    """全 A 日线 → ``(ret, market_cap, industry, returns_fwd, dates, codes, extra)``。

    ``extra`` 含 ``turnover`` / ``pb`` / ``tot_share`` / ``adj_close`` / ``bf``；
    ``with_prices=True`` 时再补 ``open`` / ``high`` / ``low`` / ``amount``
    （非流动性、UMR 真实波动用，按需装载避免白占内存）。
    换手率的口径是 **成交量 / PIT 总股本**（缓存无自由流通股，是总股本口径的
    近似）；``pb = 市值 / PIT 股东权益(含少数股东)``。
    """
    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    codes, _cal, daily = load_daily(cache, uni, "000300.SH", begin, end,
                                    pool="all_a")
    d = daily.reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    close_raw, volume = _panel("close"), _panel("volume")
    bf = load_backward_factor(cache, list(close_raw.columns))
    if bf.empty:
        log.warning("无后复权因子，退化为原始价（除权日收益有跳变）")
        close, bf_adj = close_raw, None
    else:
        # 对齐到日线面板再 ffill：原始 bf 索引含缓存全历史（8767 日），
        # 直接拿它做乘法会把面板膨胀成并集索引
        bf_adj = bf.reindex(index=close_raw.index,
                            columns=close_raw.columns).ffill()
        close = close_raw * bf_adj

    # 市值 = PIT 总股本 × 未复权 close（与生产/回测同口径）
    fin = load_financial_tables(cache, list(close_raw.columns))
    balance = fin.get("balance_sheet", pd.DataFrame())
    tot_share = build_pit_panel(balance, _cal, "TOT_SHARE").reindex(
        index=close_raw.index, columns=close_raw.columns)
    market_cap = (tot_share * close_raw).where(close_raw.notna())

    industry = None
    try:
        from data.industry import IndustryClassification
        industry = IndustryClassification(cache, level=1).get_industry_panel(
            list(close.columns), close.index)
        if industry.isna().all().all():
            industry = None
    except Exception as e:  # noqa: BLE001
        log.warning("行业面板不可用（%s），分组类因子跳过", str(e)[:80])

    ret = close.pct_change(fill_method=None)
    returns_fwd = ret.shift(-1)

    # 特质换手波动率所需：换手率（成交量/PIT 总股本）+ pb（市值/PIT 股东权益）
    equity = build_pit_panel(balance, _cal, "TOT_SHARE_EQUITY_INCL_MIN_INT") \
        .reindex(index=close_raw.index, columns=close_raw.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = (volume / tot_share.where(tot_share > 0)) \
            .where(volume.notna())
        pb = (market_cap / equity.where(equity > 0)).where(market_cap.notna())
    extra = {"turnover": turnover, "pb": pb, "tot_share": tot_share,
             "adj_close": close, "bf": bf_adj}
    if with_prices:
        for col in ("open", "high", "low", "amount"):
            extra[col] = _panel(col)

    log.info("日线 %d 日 × %d 股 / 行业%s / 市值非空 %.0f%% / 换手非空 %.0f%%",
             close.shape[0], close.shape[1],
             "可用" if industry is not None else "缺失",
             100 * market_cap.notna().mean().mean(),
             100 * turnover.notna().mean().mean())
    return ret, market_cap, industry, returns_fwd, close.index, list(close.columns), extra


def build_entropy(begin: int, end: int, pool: str = "all_a",
                  block_days: int = 8) -> pd.DataFrame:
    """5 分钟面板重采样 30 分钟 → 单一成交额占比熵（``pool`` 选池）。

    2026-11-22 起默认全 A（新增版本，不覆盖 hs300 历史探针）。用
    ``iter_blocks`` 逐块计算，峰值内存只与块大小相关——整面板
    ``load`` 转 float64 会到 20GB+，低内存机器直接被打爆。
    """
    store = MinutePanelStore.open(5, pool)
    frames: list[pd.DataFrame] = []
    n_days = 0
    for chunk, block in store.iter_blocks(
            ("close", "volume"), block_days=block_days,
            begin_date=begin, end_date=end):
        frames.append(entropy_amount_share(block, chunk, store.codes,
                                           period=5, target=30)
                      .astype(np.float32))
        n_days += len(chunk)
    if not frames:
        raise RuntimeError(f"分钟面板 min5_{pool} 在 {begin}-{end} 无交易日")
    panel = pd.concat(frames)
    log.info("分钟熵面板 %d 日 × %d 股（%d→%d，pool=%s，%d 天/块）",
             panel.shape[0], panel.shape[1], int(panel.index[0].strftime("%Y%m%d")),
             int(panel.index[-1].strftime("%Y%m%d")), pool, block_days)
    return panel


def build_minute_factors(begin: int, end: int, want: set[str] | None,
                         tot_share: pd.DataFrame,
                         pool: str = "all_a") -> dict[str, pd.DataFrame]:
    """5 分钟面板 → CS 价差 / 聪明钱 vwap / 净支撑成交量（``--minute-pool`` 选池）。

    第四轮（2026-11-20）在 hs300 分钟面板上首建；全 A 分钟数据落库后默认
    ``pool="all_a"`` 重建全 A 口径面板并覆盖同名重注册。``tot_share`` 来自
    资产负债表 PIT 总股本（全 A 日频上下文），与分钟池无关。
    """
    try:
        store = MinutePanelStore.open(5, pool)
    except FileNotFoundError as e:
        log.warning("分钟面板 min5_%s 不存在（先 fetch_minute_all --compact + "
                    "build_minute_panel --universe %s），分钟频 3 因子跳过：%s",
                    pool, pool, e)
        return {}
    days = [d for d in store.all_days() if begin <= d <= end]
    if not days:
        raise RuntimeError(f"分钟面板在 {begin}-{end} 无交易日")
    pick = (lambda n: want is None or n in want)
    fields_needed: set[str] = set()
    if pick("fd_hf_cs"):
        fields_needed |= {"high", "low"}
    if pick("fd_smart_vwap"):
        fields_needed |= {"volume", "amount"}
    if pick("fd_net_support"):
        fields_needed |= {"close", "volume"}
    fields = store.load(tuple(sorted(fields_needed)),
                        begin_date=days[0], end_date=days[-1])
    out: dict[str, pd.DataFrame] = {}
    if pick("fd_hf_cs"):
        out["fd_hf_cs"] = cs_spread_panel(fields, days, store.codes)
    if pick("fd_smart_vwap"):
        out["fd_smart_vwap"] = smart_vwap_panel(fields, days, store.codes)
    if pick("fd_net_support"):
        dates = pd.to_datetime([str(d) for d in days])
        shares = tot_share.reindex(index=dates, columns=list(store.codes)) \
            .ffill().to_numpy(dtype=float)
        out["fd_net_support"] = net_support_panel(
            fields, days, store.codes, shares)
    for n, p in out.items():
        log.info("  %-14s 面板 %d 日 × %d 股 | 非空率 %.0f%%", n,
                 p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
    return out


def _mainvol_transform(daily: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """主力波动率低频化：截面 zscore → |·| → 再截面 zscore → 20 日标准差。"""
    mu = daily.mean(axis=1)
    sd = daily.std(axis=1)
    z1 = daily.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    a = z1.abs()
    mu2, sd2 = a.mean(axis=1), a.std(axis=1)
    z2 = a.sub(mu2, axis=0).div(sd2.where(sd2 > 0), axis=0)
    return z2.rolling(window, min_periods=window).std()


def build_round5_minute(begin: int, end: int, want: set[str] | None,
                        pool: str = "all_a", block_days: int = 8,
                        warmup_days: int = 50) -> dict[str, pd.DataFrame]:
    """第五轮 10 个分钟频因子：一次流式遍历 5 分钟面板。

    原文均为 1 分钟数据，本仓库只有 5 分钟 bar；各内核的近似口径见
    ``factor.factor_dict_factors`` 对应函数的 docstring。峰岭成交比需要
    过去 20 日**同时点**量分布，故用滑动窗口保存最近 20 日 volume，
    并前推 ``warmup_days`` 天，保证 2019 年初就有 20 日历史 + 21 日平滑。
    """
    store = MinutePanelStore.open(5, pool)
    picks = ROUND5_MINUTE if want is None else (ROUND5_MINUTE & set(want))
    if not picks:
        return {}
    days = store.all_days()
    first = next((i for i, d in enumerate(days) if d >= begin), None)
    if first is None:
        raise RuntimeError(f"分钟面板 min5_{pool} 在 {begin}-{end} 无交易日")
    start = days[max(0, first - int(warmup_days))]
    codes = list(store.codes)
    raw: dict[str, list[np.ndarray]] = {n: [] for n in picks
                                        if n != "fd_pk_ridge"}
    pk_parts: list[tuple[np.ndarray, np.ndarray]] = []
    dates: list[int] = []
    hist: deque = deque(maxlen=20)
    for chunk, block in store.iter_blocks(
            ("open", "close", "volume", "amount"), block_days=block_days,
            begin_date=start, end_date=end):
        o_all, c_all = block["open"], block["close"]
        v_all, a_all = block["volume"], block["amount"]
        for k, d in enumerate(chunk):
            # 内核统一吃 [D, B, C]，这里按单日切片保持 3 维
            o = o_all[k:k + 1]
            c = c_all[k:k + 1]
            v = v_all[k:k + 1]
            a = a_all[k:k + 1]
            if "fd_pk_ridge" in picks:
                pk_parts.append(peak_ridge_daily(v, a, hist))
            if "fd_pk_cnt" in picks:
                raw["fd_pk_cnt"].append(peak_count_daily(a))
            if "fd_main_vol_up" in picks:
                raw["fd_main_vol_up"].append(mainforce_vol_daily(c, v))
            if "fd_ds_std" in picks or "fd_ds_vol" in picks:
                ds_std, ds_vol = ds_significance_daily(c, o, v)
                if "fd_ds_std" in picks:
                    raw["fd_ds_std"].append(ds_std)
                if "fd_ds_vol" in picks:
                    raw["fd_ds_vol"].append(ds_vol)
            if "fd_sigup_pvr" in picks or "fd_sigup_pvr_s" in picks:
                pvr, pvr_s = sigup_pv_daily(c, o, v)
                if "fd_sigup_pvr" in picks:
                    raw["fd_sigup_pvr"].append(pvr)
                if "fd_sigup_pvr_s" in picks:
                    raw["fd_sigup_pvr_s"].append(pvr_s)
            if "fd_dazzle_vol" in picks or "fd_dazzle_ret" in picks:
                dz_vol, dz_ret = dazzle_daily(c, v)
                if "fd_dazzle_vol" in picks:
                    raw["fd_dazzle_vol"].append(dz_vol)
                if "fd_dazzle_ret" in picks:
                    raw["fd_dazzle_ret"].append(dz_ret)
            if "fd_undercurrent" in picks:
                raw["fd_undercurrent"].append(undercurrent_daily(v))
            hist.append(np.array(v[0], dtype=np.float32, copy=True))
            dates.append(d)

    idx = pd.to_datetime([str(int(d)) for d in dates])
    codes_idx = codes

    def frame(seq: list[np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame(np.vstack(seq), index=idx, columns=codes_idx)

    out: dict[str, pd.DataFrame] = {}
    if "fd_pk_ridge" in picks:
        pk = frame([x[0] for x in pk_parts])
        rg = frame([x[1] for x in pk_parts])
        ps = pk.rolling(21, min_periods=21).sum()
        rs = rg.rolling(21, min_periods=21).sum()
        out["fd_pk_ridge"] = ps.div(rs.where(rs > 0))
    for name in ("fd_pk_cnt", "fd_ds_std", "fd_ds_vol"):
        if name in picks:
            out[name] = frame(raw[name])
    if "fd_main_vol_up" in picks:
        out["fd_main_vol_up"] = _mainvol_transform(frame(raw["fd_main_vol_up"]))
    if "fd_sigup_pvr" in picks:
        out["fd_sigup_pvr"] = frame(raw["fd_sigup_pvr"]).rolling(
            21, min_periods=21).mean()
    if "fd_sigup_pvr_s" in picks:
        d = frame(raw["fd_sigup_pvr_s"])
        out["fd_sigup_pvr_s"] = (d * d).rolling(21, min_periods=21).mean().pow(0.5)
    for name in ("fd_dazzle_vol", "fd_dazzle_ret"):
        if name in picks:
            out[name] = frame(raw[name]).rolling(21, min_periods=21).std()
    if "fd_undercurrent" in picks:
        out["fd_undercurrent"] = frame(raw["fd_undercurrent"]).rolling(
            21, min_periods=21).mean()

    keep = out[next(iter(out))].index >= pd.Timestamp(str(begin))
    for name, p in out.items():
        out[name] = p.loc[keep]
        log.info("  %-16s 面板 %d 日 × %d 股 | 非空率 %.0f%%", name,
                 p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
    return out


def build_round6_minute(begin: int, end: int, want: set[str] | None,
                        tot_share: pd.DataFrame, industry: pd.DataFrame | None,
                        market_cap: pd.DataFrame, pool: str = "all_a",
                        block_days: int = 8,
                        warmup_days: int = 45) -> dict[str, pd.DataFrame]:
    """第六轮 14 个分钟派生因子：一次流式遍历 5 分钟面板。

    需要过去 21 日同时点分布的（foc / 滞后相关）用 20 日滑动 deque；前推
    ``warmup_days=45`` 天保证 2019 年初同时具备 21 日历史与 21 日平滑。

    Returns:
        日频面板字典；``_raw_jump`` 为未平滑的日跳跃度，仅供主流程派生
        ``fd_mod_amp``，注册前会被移除。
    """
    store = MinutePanelStore.open(5, pool)
    picks = ROUND6_MINUTE if want is None else (ROUND6_MINUTE & set(want))
    if not picks:
        return {}
    days = store.all_days()
    first = next((i for i, d in enumerate(days) if d >= begin), None)
    if first is None:
        raise RuntimeError(f"分钟面板 min5_{pool} 在 {begin}-{end} 无交易日")
    start = days[max(0, first - int(warmup_days))]
    codes = list(store.codes)
    n_code = len(codes)
    dates_all = pd.to_datetime([str(int(d)) for d in days])
    shares_all = tot_share.reindex(index=dates_all, columns=codes) \
        .ffill().to_numpy(dtype=np.float32)
    cap_all = market_cap.reindex(index=dates_all, columns=codes) \
        .ffill().to_numpy(dtype=np.float32)
    ind_all = None if industry is None else industry \
        .reindex(index=dates_all, columns=codes).ffill().to_numpy()
    day_pos = {int(d): i for i, d in enumerate(days)}

    want_corr = bool({"fd_corr_lagret_tr", "fd_corr_lagret_amt"} & picks)
    want_foc = bool({"fd_foc_vol", "fd_foc_amt"} & picks)
    want_jump = bool({"fd_jump", "fd_mod_amp"} & picks)

    hist_v: deque = deque(maxlen=20)
    hist_a: deque = deque(maxlen=20)
    hist_lag: deque = deque(maxlen=20)
    prev_close = np.full(n_code, np.nan, dtype=np.float32)
    rec_days: list[int] = []
    raw: dict[str, list[np.ndarray]] = {n: [] for n in picks}
    if want_jump:
        raw["_jump"] = []

    def nan_row() -> np.ndarray:
        return np.full((1, n_code), np.nan, dtype=np.float32)

    for chunk, block in store.iter_blocks(
            ("open", "close", "high", "low", "volume", "amount"),
            block_days=block_days, begin_date=start, end_date=end):
        for k, d in enumerate(chunk):
            o = block["open"][k][None].astype(np.float64)
            c = block["close"][k][None].astype(np.float64)
            h = block["high"][k][None].astype(np.float64)
            lo = block["low"][k][None].astype(np.float64)
            v = block["volume"][k][None].astype(np.float64)
            a = block["amount"][k][None].astype(np.float64)
            v32 = block["volume"][k].astype(np.float32)
            a32 = block["amount"][k].astype(np.float32)
            p = day_pos[int(d)]

            if want_jump:
                jj = jump_daily(c)
                raw["_jump"].append(jj)
                if "fd_jump" in picks:
                    raw["fd_jump"].append(jj)
            if "fd_traj_illiq" in picks:
                raw["fd_traj_illiq"].append(
                    traj_illiq_daily(c, o, v, shares_all[p][None, :]))
            if "fd_lcp" in picks:
                raw["fd_lcp"].append(lcp_daily(c, v, a, prev_close))
            if "fd_rev_imp_pos" in picks:
                raw["fd_rev_imp_pos"].append(rev_imp_pos_daily(c, o, v))
            if "fd_elastic" in picks:
                raw["fd_elastic"].append(elasticity_daily(c))
            if "fd_ifle" in picks:
                raw["fd_ifle"].append(ifle_daily(o, h, lo, c, a))
            if "fd_wind_tide" in picks:
                raw["fd_wind_tide"].append(
                    wind_tide_daily(c, v, prev_close))
            if "fd_amt_lead" in picks:
                raw["fd_amt_lead"].append(amt_lead_daily(c, a))
            if "fd_ex_rtn_max5" in picks:
                if ind_all is None:
                    raw["fd_ex_rtn_max5"].append(nan_row())
                else:
                    raw["fd_ex_rtn_max5"].append(ex_rtn_max_daily(
                        c, ind_all[p][None], cap_all[p][None]))

            lag = None
            if want_corr:
                r_in = np.full_like(c, np.nan)
                r_in[:, 1:, :] = c[:, 1:, :] / c[:, :-1, :] - 1.0
                lag = np.abs(np.log1p(r_in))
                lag[:, 0, :] = np.nan
            if want_corr or want_foc:
                if len(hist_v) < 20:
                    for name in ("fd_corr_lagret_tr", "fd_corr_lagret_amt",
                                 "fd_foc_vol", "fd_foc_amt"):
                        if name in picks:
                            raw[name].append(nan_row())
                else:
                    stack_v = np.stack(list(hist_v) + [v32], axis=0)
                    stack_a = np.stack(list(hist_a) + [a32], axis=0)
                    if want_foc:
                        mu = np.where(np.isfinite(stack_v),
                                      stack_v, 0.0).sum(axis=0) \
                            / np.maximum(np.isfinite(stack_v).sum(axis=0), 1)
                        dev = np.where(np.isfinite(stack_v), stack_v - mu, 0.0)
                        sd = np.sqrt((dev * dev).sum(axis=0) / np.maximum(
                            np.isfinite(stack_v).sum(axis=0) - 1, 1))
                        flag_v = np.isfinite(v32) & np.isfinite(mu + sd) \
                            & (v32 >= mu + sd)
                        if "fd_foc_vol" in picks:
                            raw["fd_foc_vol"].append(foc_autocorr(flag_v[None]))
                        mu_a = np.where(np.isfinite(stack_a),
                                        stack_a, 0.0).sum(axis=0) \
                            / np.maximum(np.isfinite(stack_a).sum(axis=0), 1)
                        dev_a = np.where(np.isfinite(stack_a),
                                         stack_a - mu_a, 0.0)
                        sd_a = np.sqrt((dev_a * dev_a).sum(axis=0) / np.maximum(
                            np.isfinite(stack_a).sum(axis=0) - 1, 1))
                        flag_a = np.isfinite(a32) & np.isfinite(mu_a + sd_a) \
                            & (a32 >= mu_a + sd_a)
                        if "fd_foc_amt" in picks:
                            raw["fd_foc_amt"].append(foc_autocorr(flag_a[None]))
                    if want_corr:
                        stack_lag = np.stack(list(hist_lag) + [lag[0]], axis=0)
                        if "fd_corr_lagret_amt" in picks:
                            raw["fd_corr_lagret_amt"].append(
                                corr_lagret(stack_a, stack_lag)[None])
                        if "fd_corr_lagret_tr" in picks:
                            flow = stack_v / shares_all[p][None, None, :]
                            raw["fd_corr_lagret_tr"].append(
                                corr_lagret(flow, stack_lag)[None])

            if want_corr:
                hist_lag.append(lag[0].astype(np.float32))
            hist_v.append(v32.copy())
            hist_a.append(a32.copy())
            prev_close = c[0, -1, :].astype(np.float32)
            rec_days.append(int(d))

    idx = pd.to_datetime([str(int(d)) for d in rec_days])

    def frame(seq: list[np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame(np.vstack(seq), index=idx, columns=codes,
                            dtype=np.float32)

    out: dict[str, pd.DataFrame] = {}
    if want_jump:
        jf = frame(raw["_jump"])
        if "fd_mod_amp" in picks:
            out["_raw_jump"] = jf
        if "fd_jump" in picks:
            out["fd_jump"] = jf.rolling(21, min_periods=10).mean()
    for name in ("fd_traj_illiq", "fd_lcp", "fd_rev_imp_pos", "fd_elastic",
                 "fd_amt_lead", "fd_ex_rtn_max5", "fd_wind_tide"):
        if name in picks:
            out[name] = frame(raw[name]).rolling(21, min_periods=10).mean()
    if "fd_ifle" in picks:
        out["fd_ifle"] = frame(raw["fd_ifle"]).rolling(
            21, min_periods=10).std()
    for name in ("fd_corr_lagret_tr", "fd_corr_lagret_amt"):
        if name in picks:
            out[name] = frame(raw[name])
    for name in ("fd_foc_vol", "fd_foc_amt"):
        if name in picks:
            out[name] = frame(raw[name]).rolling(20, min_periods=10).mean()

    keep = idx >= pd.Timestamp(str(begin))
    for name, panel in out.items():
        out[name] = panel.loc[keep]
        log.info("  %-18s 面板 %d 日 × %d 股 | 非空率 %.0f%%", name,
                 out[name].shape[0], out[name].shape[1],
                 100 * out[name].notna().mean().mean())
    return out


# ----------------------------------------------------------------------
# 安慰剂对照（L5）
# ----------------------------------------------------------------------
def placebo_check(name: str, panel: pd.DataFrame,
                  returns_panel: pd.DataFrame) -> dict:
    """真实 IC vs 循环平移安慰剂：报 eff_ratio / eff_gap（L2/L5）。"""
    real = ic_stats(calc_ic_series(panel, returns_panel))
    try:
        plc = cyclic_placebo(panel, seed=20261119, name=name)
        plc_ic = ic_stats(calc_ic_series(plc, returns_panel))["ic"]
    except ValueError as e:
        log.warning("安慰剂跳过 %s：%s", name, e)
        return {"real_ic": real["ic"], "plc_ic": np.nan,
                "eff_ratio": np.nan, "eff_gap": np.nan}
    return {"real_ic": real["ic"], "plc_ic": plc_ic,
            **effect_size(real["ic"], plc_ic)}


def coverage_row(panel: pd.DataFrame, returns_panel: pd.DataFrame) -> dict:
    """覆盖率/有效格：先按**本因子自己的**行列裁剪前瞻收益与掩码再统计。

    ``panel_coverage`` 内部直接做 ``x & fwd & mask``，不裁剪会让 pandas 按并集
    展开（熵因子只有 hs300，并集到全 A 后 n_valid 会被 fwd 的非空撑成 8.9M 假数）。
    """
    fwd = returns_panel.reindex(index=panel.index, columns=panel.columns)
    mask = pd.DataFrame(True, index=panel.index, columns=panel.columns)
    return panel_coverage(panel, fwd, mask)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="因子字典候选复现体检入库")
    ap.add_argument("--offline", action="store_true", default=True,
                    help="只读本地缓存（默认）")
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20251231)
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--minute-pool", default="all_a",
                    help="分钟频 3 因子用的分钟面板池（hs300 为第四轮历史口径）")
    ap.add_argument("--entropy-pool", default="all_a",
                    help="fd_ent30 用的分钟面板池（hs300 为 11-19 历史探针口径）")
    ap.add_argument("--features", default=None)
    ap.add_argument("--no-save", action="store_true", help="只算不入库")
    ap.add_argument("--report-only", action="store_true",
                    help="跳过入库，只重算面板并出体检报告")
    args = ap.parse_args()

    want = ({x.strip() for x in args.features.split(",") if x.strip()}
            if args.features else None)

    def need(*names: str) -> bool:
        return want is None or any(n in want for n in names)

    ret, cap, ind, returns_fwd, _dates, _codes, extra = load_daily_context(
        args.begin, args.end,
        with_prices=need("fd_illiq_ds", "fd_umr_trv", "fd_mod_amp"))

    # 只算被点名的因子：CSAD 每日一个全截面相关矩阵、MMD 是逐日核矩阵，
    # 没必要为跑 3 个因子先把整套（含 30 分钟熵）算一遍
    panels: dict[str, pd.DataFrame] = {}
    if need("fd_tcc"):
        panels["fd_tcc"] = tcc_panel(ret, ind, cap, by_group=False)
    if ind is not None:
        if need("fd_tcc_g"):
            panels["fd_tcc_g"] = tcc_panel(ret, ind, cap, by_group=True)
        if need("fd_grp_abn"):
            panels["fd_grp_abn"] = group_abnormal_panel(ret, ind, cap)
    elif need("fd_tcc_g", "fd_grp_abn"):
        log.warning("无行业面板：fd_tcc_g / fd_grp_abn 跳过")
    if need("fd_rnbr"):
        panels["fd_rnbr"] = rnbr_panel(ret)
    if need("fd_csad21"):
        panels["fd_csad21"] = csad_std21_panel(ret)
    if need("fd_mmd21"):
        panels["fd_mmd21"] = mmd_asym_panel(ret)
    if need("fd_tr_resid"):
        panels["fd_tr_resid"] = turnover_resid_vol_panel(
            extra["turnover"], extra["pb"], cap, ret)
    if need("fd_utr"):
        panels["fd_utr"] = utr_panel(extra["turnover"], cap)
    if need("fd_ind_mom"):
        if ind is None:
            log.warning("无行业面板：fd_ind_mom 跳过")
        else:
            panels["fd_ind_mom"] = ind_joint_momentum_panel(ret, cap, ind)
    if need("fd_price_delay"):
        panels["fd_price_delay"] = price_delay_panel(ret)
    # ---- 第六轮日频候选 ----
    if need("fd_abn_ret_max"):
        panels["fd_abn_ret_max"] = abnormal_ret_max_panel(ret)
        log.info("  fd_abn_ret_max 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                 *panels["fd_abn_ret_max"].shape,
                 100 * panels["fd_abn_ret_max"].notna().mean().mean())
    if need("fd_net_cent"):
        panels["fd_net_cent"] = network_centrality_panel(ret)
        log.info("  fd_net_cent 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                 *panels["fd_net_cent"].shape,
                 100 * panels["fd_net_cent"].notna().mean().mean())
    if need("fd_emo_spill"):
        panels["fd_emo_spill"] = emo_spill_panel(extra["turnover"])
        log.info("  fd_emo_spill 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                 *panels["fd_emo_spill"].shape,
                 100 * panels["fd_emo_spill"].notna().mean().mean())

    # 非流动性 / UMR / 惊恐：价格类，复权后再算（比值在乘性复权下不变）
    if need("fd_illiq_ds", "fd_umr_tr", "fd_umr_trv", "fd_fear_std"):
        bf = extra["bf"]
        mul = (lambda x: x) if bf is None else (lambda x: x * bf)
        if need("fd_illiq_ds"):
            panels["fd_illiq_ds"] = illiq_panel(
                mul(extra["open"]), mul(extra["high"]), mul(extra["low"]),
                extra["adj_close"], extra["amount"])
        if need("fd_umr_tr", "fd_umr_trv"):
            mc = cap.reindex_like(ret)
            w = mc.div(mc.sum(axis=1, min_count=1), axis=0)
            excess = ret.sub((ret * w).sum(axis=1), axis=0)
        if need("fd_umr_tr"):
            panels["fd_umr_tr"] = umr_panel(excess, extra["turnover"])
        if need("fd_umr_trv"):
            high, low = mul(extra["high"]), mul(extra["low"])
            prev = extra["adj_close"].shift(1)
            tr = (np.maximum(np.maximum(high - low, (high - prev).abs()),
                             (low - prev).abs()) / prev)
            panels["fd_umr_trv"] = umr_panel(excess, tr)
        if need("fd_fear_std"):
            panels["fd_fear_std"] = fear_panel(ret)
    if need("fd_ent30"):
        panels["fd_ent30"] = build_entropy(args.begin, args.end,
                                           pool=args.entropy_pool)
    if need("fd_hf_cs", "fd_smart_vwap", "fd_net_support"):
        panels.update(build_minute_factors(args.begin, args.end, want,
                                           extra["tot_share"],
                                           pool=args.minute_pool))
    if want is None or (ROUND5_MINUTE & set(want)):
        panels.update(build_round5_minute(args.begin, args.end, want,
                                          pool=args.minute_pool))
    if want is None or (ROUND6_MINUTE & set(want)):
        panels.update(build_round6_minute(
            args.begin, args.end, want, extra["tot_share"], ind, cap,
            pool=args.minute_pool))
        raw_jump = panels.pop("_raw_jump", None)
        if raw_jump is not None and need("fd_mod_amp"):
            bf = extra["bf"]
            mul = (lambda x: x) if bf is None else (lambda x: x * bf)
            amp = (mul(extra["high"]) - mul(extra["low"])) \
                / extra["adj_close"].shift(1)
            jump = raw_jump.reindex(index=amp.index, columns=amp.columns)
            sign = np.where(jump.gt(jump.mean(axis=1), axis=0), 1.0, -1.0)
            panels["fd_mod_amp"] = (amp * sign).rolling(
                21, min_periods=10).mean()
            log.info("  fd_mod_amp 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                     *panels["fd_mod_amp"].shape,
                     100 * panels["fd_mod_amp"].notna().mean().mean())
        if need("fd_wind_tide") and ind is not None:
            panels["fd_wind_tide"] = neutralize_panel(
                panels["fd_wind_tide"], ind, cap)
            log.info("  fd_wind_tide 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                     *panels["fd_wind_tide"].shape,
                     100 * panels["fd_wind_tide"].notna().mean().mean())
    if need("fd_attn_spill"):
        if ind is None:
            log.warning("无行业面板：fd_attn_spill 跳过")
        else:
            panels["fd_attn_spill"] = attention_spill_panel(ret, ind, cap)
            log.info("  fd_attn_spill 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                     *panels["fd_attn_spill"].shape,
                     100 * panels["fd_attn_spill"].notna().mean().mean())

    if args.no_save:
        for n, p in panels.items():
            log.info("  %-10s 面板 %s 日 × %d 股 | 非空率 %.0f%%",
                     n, p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
        return

    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset=args.dataset)
    if not args.report_only:
        src_daily = f"paper:build_fd_factors:{args.begin}-{args.end}"
        # 收益面板不传：走库标准口径（复权收盘 1 日远期 + 可交易掩码）。
        # 自带一份 pct_change 只会被记成 caller_supplied，卡住晋升第 2 关。
        rows = register_panels(
            lib, panels, {k: FACTOR_DEFS[k] for k in panels}, None,
            source=src_daily, check_dup=True, families=FAMILY,
        )
        for r in rows:
            lib.set_tag(r["name"],
                        frequency="日内" if r["name"] in MINUTE_FACTORS else "日频")
        # formula 即中文定义（label 口径）。不标 formula_src 会被
        # ``formula_extract --apply`` 当手写保护起来，源码改了公式永不刷新。
        if rows:
            reg0 = lib._load_registry()
            reg0.loc[reg0["name"].isin([r["name"] for r in rows]), "formula_src"] = "label"
            lib._save_registry(reg0)

    # ---- 体检报告：库指标 + 覆盖 + 安慰剂 ----
    reg = lib.list_all()
    reg = reg[reg["name"].isin(list(panels))].copy()
    extra = []
    for name, panel in panels.items():
        row = {"name": name}
        row.update(coverage_row(panel, returns_fwd))
        row.update(placebo_check(name, panel, returns_fwd))
        extra.append(row)
    rep = reg.merge(pd.DataFrame(extra), on="name", how="left")
    rep = rep[["name", "cov", "cov_eff", "n_valid", "n_days", "ic_mean",
               "ic_ir", "t_stat_nw", "best_sharpe", "best_config",
               "dup_corr_max", "dup_top", "plc_ic", "eff_ratio", "eff_gap"]]
    # ---- 批内多重检验（BH）+ 有效独立维数 + 溯源（2026-11-22）----
    # 用 p_value_nw（NW t 的双侧正态近似 p）做 BH；eff_ratio 门槛不动——
    # BH 只回答「这批里哪些是多重比较下的幸存者」，不改变单因子判定。
    from stats.significance import t_pvalue, benjamini_hochberg, effective_dims

    rep["p_value_nw"] = rep["t_stat_nw"].map(
        lambda v: float(t_pvalue(float(v))) if np.isfinite(v) else np.nan)
    surv = benjamini_hochberg(rep["p_value_nw"].to_numpy(), q=0.05)
    rep["fdr_pass"] = surv
    ic_frame = pd.DataFrame(
        {n: calc_ic_series(panels[n], returns_fwd) for n in rep["name"]})
    n_eff = effective_dims(ic_frame)
    data_end = str(pd.to_datetime(returns_fwd.index.max()).date())
    basis = (f"returns=adj_close_shift(-1)@{data_end}; "
             "placebo=cyclic_crc32(name); nw=andrews; batch_bh_q=0.05")
    rep["admission_basis"] = basis
    rep["panel_end"] = data_end
    n_pass_bh = int(surv.sum()) if len(surv) else 0
    log.info("批内多重检验: m=%d → BH(FDR 5%%) 存活 %d | 批内有效独立维数 "
             "n_eff=%.1f（名义 m=%d）| 溯源: %s",
             len(surv), n_pass_bh, n_eff, len(surv), basis)

    # ① 区分力验证（并列报告，不进 ladder）：|t|≥2 与安慰剂门槛（eff≥2×）
    # 的一致率——t 已在表里，先看它对安慰剂的区分力再谈是否立阈值。
    ok_eff = rep["eff_ratio"].ge(2.0) & rep["eff_ratio"].notna()
    ok_t = rep["t_stat_nw"].abs().ge(2.0)
    both = ok_eff & ok_t
    neither = (~ok_eff) & (~ok_t)
    agree = float((both | neither).mean()) if len(rep) else float("nan")
    log.info("t_nw 区分力验证: |t|≥2 与 eff_ratio≥2 一致率 %.0f%%（n=%d；"
             "仅 t 过 %d / 仅 eff 过 %d / 双过 %d / 双不过 %d）",
             agree * 100, len(rep),
             int((ok_t & ~ok_eff).sum()), int((ok_eff & ~ok_t).sum()),
             int(both.sum()), int(neither.sum()))

    rep["_abs_ic"] = rep["ic_mean"].abs()
    rep = rep.sort_values("_abs_ic", ascending=False).drop(columns="_abs_ic")

    out_dir = ROOT / "reports" / "factor_dict_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "checkup.csv"
    # 按因子名合并：分轮跑（--features 只跑子集）时不把上一轮的体检行冲掉
    if out.exists():
        old = pd.read_csv(out, encoding="utf-8-sig")
        rep = pd.concat([old[~old["name"].isin(rep["name"])], rep],
                        ignore_index=True)
        rep = rep.sort_values("name")
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_colwidth", 26,
                           "display.float_format", "{:.4f}".format):
        print(f"\n==== 因子字典候选体检（全 A 日频 / 熵为 {args.entropy_pool} 30min） ====")
        print(rep.to_string(index=False))
    log.info("报告落盘: %s", out)

    n_pass = int((rep["eff_ratio"].abs() >= ADMISSION_GATES["eff_ratio"]).sum())
    log.info("摘要: %d/%d 满足安慰剂效应量比 ≥%.1f×（准入 L5）",
             n_pass, len(rep), ADMISSION_GATES["eff_ratio"])

    record_experiment_safe(
        kind="factor_dict_probe",
        command=" ".join(sys.argv),
        params={"begin": args.begin, "end": args.end, "dataset": args.dataset,
                "n_factors": len(panels)},
        fingerprint=None, result_path=str(out),
        metrics={"n_pass_placebo": n_pass,
                 "best_abs_ic": float(rep["ic_mean"].abs().max())},
        note="飞书因子字典候选复现体检",
    )


if __name__ == "__main__":
    main()
