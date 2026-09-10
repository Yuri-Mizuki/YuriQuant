"""论文复现因子族（paperswithbacktest / awesome-systematic-trading）
================================================================

从 awesome-systematic-trading 复现库的 ``static/strategies/`` 中筛选出
可在 A 股数据面（日线 OHLCV + 财务 PIT + 行业 + 公告日历）实现的 21 个
经典已发表策略，翻译为截面因子面板。原始策略多为多空双腿，本库只做多，
因此统一做了如下改造（逐因子细节见 ``scripts/build_paper_factors.py``
的 FACTOR_DEFS 注释）：

- **方向统一**：所有面板值"越大预期收益越高"。原策略的"低好"因子
  （应计/资产增长/β/波动/PB昂贵）已取负号或倒数变换；
- **做空腿 → 多头**：原 L/S 双腿只保留多头腿，多头腿的选股逻辑翻译为
  连续因子值（rank 交互、分位掩码等），由因子库 canonical TopK 回测承接；
- **窗口映射**：1 个月 = 21 交易日，1 年 = 252 交易日；
- **口径差异**：应计因子用现金流量表法/资产负债表法均可，本模块提供
  资产负债表法（原版），折旧以 EBITDA−EBIT 近似（现金流量表无折旧科目）。

无未来函数约定：财务值全部来自 build_pit_panel（公告日对齐），
时序 shift 只向后看；日历效应/调仓日细节（如 1 月不开仓）不进入因子。

用法
----
    data = PaperData(close=..., high=..., volume=..., ...)
    panels = compute_paper_factors(data)   # {name: date×code 面板}
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

M = 21    # 1 个月的交易日数
Y = 252   # 1 年的交易日数


# ===========================================================================
# 数据容器
# ===========================================================================
@dataclass
class PaperData:
    """paper 因子族的输入面板集合（全部 date×code，index=交易日）。"""

    close: pd.DataFrame                  # 后复权收盘价
    high: pd.DataFrame = field(default=None)   # 最高价（ATR/趋势跟踪）
    low: pd.DataFrame = field(default=None)    # 最低价
    volume: pd.DataFrame = field(default=None) # 成交量（公告集中度）
    market_cap: pd.DataFrame = field(default=None)  # 总市值
    industry: pd.Series = field(default=None)  # code → 行业（52周行业新高）
    pit: dict[str, pd.DataFrame] = field(default_factory=dict)  # 财务 PIT 面板
    ann_flag: pd.DataFrame = field(default=None)  # date×code bool，当日有财报公告


# ===========================================================================
# 通用小工具
# ===========================================================================
def _ret(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """逐日截面百分位秩 (0, 1]。"""
    return df.rank(axis=1, pct=True)


def _to_daily(monthly: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """月度面板 → 日频（ffill，月初才可见上月末值，无未来函数）。"""
    return monthly.reindex(daily_index, method="ffill")


def _month_end_panel(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    """日频面板 → 月末面板（last 取月末值 / sum 月内加总 / mean 月均 / max 月内最大）。"""
    g = df.groupby(df.index.to_period("M"))
    if how == "last":
        out = g.last()
    elif how == "sum":
        out = g.sum()
    elif how == "mean":
        out = g.mean()
    elif how == "max":
        out = g.max()
    else:
        raise ValueError(how)
    # period → 月末 Timestamp，向后偏移会让"月末值"在次月初可见，保证无前视
    out.index = out.index.to_timestamp(how="end").normalize()
    return out


# ===========================================================================
# 一、纯量价因子（12 个）
# ===========================================================================
def pap_mom_12_1(d: PaperData) -> pd.DataFrame:
    """经典动量 UMD（Jegadeesh & Titman 1993）：过去 12 个月收益跳过最近 1 月。

    momentum = close[t-21] / close[t-252] - 1。（QC 实现未跳月，此处按论文口径。）
    """
    c = d.close
    return c.shift(M) / c.shift(Y) - 1.0


def pap_mom_consistent(d: PaperData) -> pd.DataFrame:
    """一致动量（Grinblatt & Moskowitz）：两个窗口动量同时强才给高分。

    m1 = close[t-21]/close[t-147] - 1（t-7 月 → t-1 月）
    m2 = close[t]/close[t-126] - 1  （t-6 月 → t）
    factor = min(rank(m1), rank(m2))——双窗口均进入前列才得高分。
    """
    c = d.close
    m1 = c.shift(M) / c.shift(7 * M) - 1.0
    m2 = c / c.shift(6 * M) - 1.0
    return np.minimum(_cs_rank(m1), _cs_rank(m2))


def pap_rev_5d(d: PaperData) -> pd.DataFrame:
    """短期反转：-（近 5 日收益）。上周跌幅越大分越高（A股反转效应）。"""
    return -(d.close / d.close.shift(5) - 1.0)


def pap_momxvol_6m(d: PaperData) -> pd.DataFrame:
    """动量×波动双排序：高波动 × 高动量（6 月动量跳过最近 1 周）交互项。

    factor = rank(perf) × rank(vol)，perf = close[t-5]/close[t-126]-1，
    vol = 日收益 std(窗口 [t-125, t-5])。原策略为"赢家∩高波"做多。
    """
    c = d.close
    r = _ret(c)
    perf = c.shift(5) / c.shift(6 * M) - 1.0
    vol = r.shift(5).rolling(6 * M - 5).std()
    return _cs_rank(perf) * _cs_rank(vol)


def pap_lowvol_1y(d: PaperData) -> pd.DataFrame:
    """低波动因子：-（近 252 日的周收益 std，约 50 个非重叠周）。

    周收益按每 5 个交易日的首尾收盘计算（降低日频自相关），取负号 =
    波动越低分越高。原策略即纯多头。
    """
    c = d.close
    weekly = c.iloc[4::5]                     # 每 5 个交易日取一个收盘
    wret = weekly.pct_change(fill_method=None)
    vol = wret.rolling(50, min_periods=30).std()
    vol_daily = vol.reindex(c.index, method="ffill")
    return -vol_daily


def pap_bab_beta(d: PaperData) -> pd.DataFrame:
    """Betting-Against-Beta：-（252 日个股 β，市场=全A等权）。

    β = Cov(r_i, r_m)/Var(r_m)。原策略多低 β 空 high β + 杠杆重整；
    多头版放弃杠杆，直接做多低 β。用滚动均值积公式避免逐列 rolling.cov。
    """
    r = _ret(d.close)
    mkt = r.mean(axis=1)
    win = Y
    em = r.rolling(win).mean()
    em_m = mkt.rolling(win).mean()
    cov = r.mul(mkt, axis=0).rolling(win).mean() - em.mul(em_m, axis=0)
    var_m = (mkt * mkt).rolling(win).mean() - em_m * em_m
    beta = cov.div(var_m.replace(0.0, np.nan), axis=0)
    return -beta


def pap_high52w(d: PaperData) -> pd.DataFrame:
    """52 周新高临近度（George & Hwang）：close / 252 日滚动最高 - 1。"""
    roll_max = d.close.rolling(Y, min_periods=Y // 2).max()
    return d.close / roll_max - 1.0


def pap_high52w_ind(d: PaperData) -> pd.DataFrame:
    """行业 52 周新高：行业内个股市值加权的 PRILAG，映射回每只成分股。

    PRILAG_i = close/max(close,252d)；行业值 = Σ PRILAG×市值 / Σ 市值。
    原策略做多行业值最高 6 个行业的全部成分（此处因子值即行业值）。
    """
    prilag = d.close.rolling(Y, min_periods=Y // 2).max()
    prilag = d.close / prilag   # 接近 1 = 接近新高（George-Hwang PRILAG）
    cap = d.market_cap
    if d.industry is None or cap is None:
        raise ValueError("pap_high52w_ind 需要 market_cap 与 industry")
    ind = d.industry
    codes = [c for c in d.close.columns if c in ind.index]
    if not codes:
        raise ValueError("industry 与 close 无交集")
    w = cap[codes].fillna(0.0) * prilag[codes].fillna(0.0)
    inds = sorted(set(ind[c] for c in codes))
    num = pd.DataFrame(0.0, index=d.close.index, columns=inds)
    den = pd.DataFrame(0.0, index=d.close.index, columns=inds)
    for k in inds:
        cols = [c for c in codes if ind[c] == k]
        num[k] = w[cols].sum(axis=1)
        den[k] = cap[codes][cols].fillna(0.0).sum(axis=1)
    ind_val = num.div(den.replace(0.0, np.nan), axis=0)   # date×行业
    # 行业值映射回个股（向量化：按列查行业名）
    k_to_col = {k: i for i, k in enumerate(inds)}
    code_ind = ind.reindex(d.close.columns)
    pos = code_ind.map(k_to_col)
    out_vals = np.full((len(d.close.index), len(d.close.columns)), np.nan)
    valid_cols = pos.notna().values
    out_vals[:, valid_cols] = ind_val.values[:, pos[valid_cols].astype(int).values]
    return pd.DataFrame(out_vals, index=d.close.index, columns=d.close.columns)


def pap_breakout_atr(d: PaperData) -> pd.DataFrame:
    """个股趋势跟踪（时间序列信号，非截面）：创历史新高入场，
    10 日 ATR 吊灯止损（只上移不下移）出场。

    因子值 = 在场 1 / 空仓 0（稀疏 0/1 面板；TopK 评估时在场股票即组合候选）。
    """
    c, h = d.close, d.high if d.high is not None else d.close
    low_p = d.low if d.low is not None else d.close
    lo = low_p
    prev_c = c.shift(1)
    tr = np.maximum(h - lo, (h - prev_c).abs())
    tr = np.maximum(tr, (lo - prev_c).abs())
    atr = tr.rolling(10, min_periods=5).mean()

    dates, codes = c.index, c.columns
    c_v = c.values
    h_v = h.values
    atr_v = atr.values
    exp_max = np.maximum.accumulate(np.where(np.isnan(c_v), -np.inf, c_v), axis=0)
    in_pos = np.zeros(c_v.shape, dtype=bool)
    stop = np.full(c_v.shape, np.nan)
    hh = np.full(c_v.shape, np.nan)   # 入场以来最高价
    for t in range(1, len(dates)):
        px, hi, a = c_v[t], h_v[t], atr_v[t]
        held = in_pos[t - 1]
        # 吊灯止损：历史最高（入场以来）− ATR10，且只上移
        hh[t] = np.fmax(np.where(held | False, hh[t - 1], np.nan), hi)
        new_stop = hh[t] - a
        stop[t] = np.fmax(np.where(held, stop[t - 1], np.nan), new_stop)
        exited = held & (px < stop[t])
        # 入场：今日收盘创历史新高（含持平）；exp_max=-inf 表示无历史，不触发
        entry = ((~held | exited) & (px >= exp_max[t - 1])
                 & (exp_max[t - 1] > -np.inf) & ~np.isnan(a))
        in_pos[t] = (~exited & held) | (entry & ~np.isnan(px))
        in_pos[t] = np.where(np.isnan(px), False, in_pos[t])
        hh[t] = np.where(in_pos[t] & ~held, hi, hh[t])
        stop[t] = np.where(in_pos[t] & ~held, hi - a, stop[t])
    return pd.DataFrame(in_pos.astype(float), index=dates, columns=codes)


def pap_smallcap(d: PaperData) -> pd.DataFrame:
    """规模因子（Banz）：-ln(市值)，市值越小分越高。

    原策略多小空大；A 股多头版直接做多小市值组（注意壳价值/退市特性）。
    """
    if d.market_cap is None:
        raise ValueError("pap_smallcap 需要 market_cap")
    return -np.log(d.market_cap.clip(lower=1.0))


def pap_momxag(d: PaperData) -> pd.DataFrame:
    """资产增长×动量：高资产增长组（前 10%）内的高动量股。

    factor = rank(mom_11_1) × 1{rank(AG) > 0.9}，AG=总资产同比。
    原策略 1 月不开仓的日历规则不进入因子。（依赖财务 PIT。）
    """
    ta = d.pit.get("TOTAL_ASSETS")
    if ta is None:
        raise ValueError("pap_momxag 需要 TOTAL_ASSETS PIT 面板")
    ag = ta / ta.shift(Y) - 1.0
    mom = d.close.shift(M) / d.close.shift(11 * M) - 1.0
    ag_hot = (_cs_rank(ag) > 0.9).astype(float)
    ag_hot[ag.isna()] = np.nan
    return _cs_rank(mom) * ag_hot


# ===========================================================================
# 二、财务 PIT 因子（7 个）
# ===========================================================================
def pap_accruals_bs(d: PaperData) -> pd.DataFrame:
    """应计异象·资产负债表法（Sloan 1996，原版公式）：

    ACC = [(ΔCA−ΔCash) − (ΔCL−ΔSTD)] / mean(TA) − Dep/mean(TA)
    其中 Δ 为年报同比变动（PIT 面板 shift 252 近似上年同期），
    Dep 以 EBITDA−EBIT 近似（A 股现金流表无折旧明细科目）。
    取负号 = 应计越低分越高。
    """
    p = d.pit
    need = ("TOTAL_CUR_ASSETS", "CURRENCY_CAP", "TOTAL_CUR_LIAB",
            "ST_BORROWING", "TOTAL_ASSETS", "EBITDA_TTM", "EBIT_TTM")
    missing = [k for k in need if p.get(k) is None]
    if missing:
        raise ValueError(f"pap_accruals_bs 缺少 PIT 面板: {missing}")
    ta = p["TOTAL_ASSETS"]
    ta_avg = 0.5 * (ta + ta.shift(Y))
    dca = p["TOTAL_CUR_ASSETS"] - p["TOTAL_CUR_ASSETS"].shift(Y)
    dcash = p["CURRENCY_CAP"] - p["CURRENCY_CAP"].shift(Y)
    dcl = p["TOTAL_CUR_LIAB"] - p["TOTAL_CUR_LIAB"].shift(Y)
    dstd = p["ST_BORROWING"] - p["ST_BORROWING"].shift(Y)
    dep = (p["EBITDA_TTM"] - p["EBIT_TTM"]).clip(lower=0.0)
    acc = ((dca - dcash) - (dcl - dstd)) / ta_avg.replace(0.0, np.nan) \
        - dep / ta_avg.replace(0.0, np.nan)
    return -acc


def pap_asset_growth(d: PaperData) -> pd.DataFrame:
    """资产增长异象（Cooper et al.）：-（总资产同比增速）。

    低资产增长（保守扩张）公司未来收益更高，取负号 = 增长越低分越高。
    """
    ta = d.pit.get("TOTAL_ASSETS")
    if ta is None:
        raise ValueError("pap_asset_growth 需要 TOTAL_ASSETS PIT 面板")
    return -(ta / ta.shift(Y) - 1.0)


def pap_earn_quality(d: PaperData) -> pd.DataFrame:
    """盈利质量复合因子（四成分等权排名和）：

    ① 应计（CF 法，低好，负向）② CFO/净利（高好）
    ③ 负债率 D/A（低好，负向）④ ROE（高好）。
    factor = Σ 每成分的截面百分位秩。原策略综合分前 1/3 做多。
    """
    p = d.pit
    need = ("NET_PRO_TTM", "CFO_TTM", "TOTAL_ASSETS", "TOTAL_LIAB",
            "EQUITY")
    missing = [k for k in need if p.get(k) is None]
    if missing:
        raise ValueError(f"pap_earn_quality 缺少 PIT 面板: {missing}")
    ta = p["TOTAL_ASSETS"].replace(0.0, np.nan)
    accr = (p["NET_PRO_TTM"] - p["CFO_TTM"]) / ta          # 低好
    cfa = p["CFO_TTM"] / p["NET_PRO_TTM"].replace(0.0, np.nan)  # 高好
    da = p["TOTAL_LIAB"] / ta                              # 低好
    roe = p["NET_PRO_TTM"] / p["EQUITY"].replace(0.0, np.nan)   # 高好
    score = (_cs_rank(-accr) + _cs_rank(cfa) + _cs_rank(-da) + _cs_rank(roe))
    # 四成分任一缺失 → 分数失真，要求至少 3 个成分有效
    n_valid = (_cs_rank(-accr).notna().astype(int) + _cs_rank(cfa).notna().astype(int)
               + _cs_rank(-da).notna().astype(int) + _cs_rank(roe).notna().astype(int))
    return score.where(n_valid >= 3)


def pap_fscore(d: PaperData) -> pd.DataFrame:
    """Piotroski FSCORE（9 项 0-9 打分，TTM 同比口径）：

    盈利性 ①ROA_TTM>0 ②CFO_TTM>0 ③ΔROA>0 ④CFO>净利
    杠杆/流动性 ⑤长债/资产下降 ⑥流动比率上升 ⑦股本未增发
    运营效率 ⑧毛利率上升 ⑨资产周转率上升。
    同比 = PIT 面板 shift(252)（同一公告周期的上年值，PIT 安全）。
    """
    p = d.pit
    need = ("NET_PRO_TTM", "CFO_TTM", "TOTAL_ASSETS", "LT_LOAN",
            "TOTAL_CUR_ASSETS", "TOTAL_CUR_LIAB", "TOT_SHARE",
            "OPERA_REV_TTM", "LESS_OPERA_COST_TTM")
    missing = [k for k in need if p.get(k) is None]
    if missing:
        raise ValueError(f"pap_fscore 缺少 PIT 面板: {missing}")
    ta = p["TOTAL_ASSETS"].replace(0.0, np.nan)
    roa = p["NET_PRO_TTM"] / ta
    lev = p["LT_LOAN"].fillna(0.0) / ta
    curr = p["TOTAL_CUR_ASSETS"] / p["TOTAL_CUR_LIAB"].replace(0.0, np.nan)
    gm = ((p["OPERA_REV_TTM"] - p["LESS_OPERA_COST_TTM"])
          / p["OPERA_REV_TTM"].replace(0.0, np.nan))
    ato = p["OPERA_REV_TTM"] / ta

    def _up(series: pd.DataFrame) -> pd.DataFrame:
        return (series > series.shift(Y)).astype(float).where(series.notna()
                                                              & series.shift(Y).notna())

    score = (
        (roa > 0).astype(float).where(roa.notna())
        + (p["CFO_TTM"] > 0).astype(float).where(p["CFO_TTM"].notna())
        + _up(roa)
        + (p["CFO_TTM"] > p["NET_PRO_TTM"]).astype(float)
        .where(p["CFO_TTM"].notna() & p["NET_PRO_TTM"].notna())
        + _up(-lev)
        + _up(curr)
        + (p["TOT_SHARE"] <= p["TOT_SHARE"].shift(Y)).astype(float)
        .where(p["TOT_SHARE"].notna() & p["TOT_SHARE"].shift(Y).notna())
        + _up(gm)
        + _up(ato)
    )
    return score


def pap_roa_size_adj(d: PaperData) -> pd.DataFrame:
    """ROA 异象·规模分组内排名（原版：大小盘两组内各做多 ROA 前 30%）：

    factor = rank(ROA) − 组内均值(rank(ROA))，组 = 市值中位数上下两半。
    即 ROA 在同规模组内的相对高低（消除大小盘 ROA 水平差异）。
    """
    p = d.pit
    if p.get("NET_PRO_TTM") is None or p.get("TOTAL_ASSETS") is None \
            or d.market_cap is None:
        raise ValueError("pap_roa_size_adj 需要 NET_PRO_TTM/TOTAL_ASSETS/market_cap")
    roa = p["NET_PRO_TTM"] / p["TOTAL_ASSETS"].replace(0.0, np.nan)
    rk = _cs_rank(roa)
    big = d.market_cap.gt(d.market_cap.median(axis=1), axis=0).fillna(False)
    grp_mean = rk.where(big).mean(axis=1)   # 大盘组内均值
    small_mean = rk.where(~big).mean(axis=1)
    mean_by_col = pd.DataFrame(np.where(
        big.values, grp_mean.values[:, None], small_mean.values[:, None]),
        index=rk.index, columns=rk.columns)
    return rk - mean_by_col


def pap_rd_intensity(d: PaperData) -> pd.DataFrame:
    """研发强度（5 年衰减加权）：Σ w_k × RD_{t-k} / 市值，w=[1,.8,.6,.4,.2]。

    RD 用年报研发费用（PIT 到公告日），t-k 以 252 交易日近似。
    研发投入对未来收益有正向预测力（原策略做多 RDS 前 20%）。
    """
    rd = d.pit.get("RD_EXP")
    if rd is None or d.market_cap is None:
        raise ValueError("pap_rd_intensity 需要 RD_EXP PIT 面板与 market_cap")
    cap = d.market_cap.replace(0.0, np.nan)
    rds = sum(w * rd.shift(k * Y) for k, w in
              enumerate((1.0, 0.8, 0.6, 0.4, 0.2)))
    return rds / cap


def pap_value_bp(d: PaperData) -> pd.DataFrame:
    """价值因子 HML 腿（Fama-French）：账面市值比 BP = 股东权益/市值。

    BP 越高越便宜，多头版做多低 PB 组（原策略多低 PB 空高 PB）。
    与库内 bp 因子同构，作为论文复现对照保留（check_dup 会标冗余）。
    """
    p = d.pit
    if p.get("EQUITY") is None or d.market_cap is None:
        raise ValueError("pap_value_bp 需要 EQUITY PIT 面板与 market_cap")
    return p["EQUITY"] / d.market_cap.replace(0.0, np.nan)


# ===========================================================================
# 三、事件类因子（2 个）
# ===========================================================================
def _monthly_volume(d: PaperData) -> pd.DataFrame:
    if d.volume is None:
        raise ValueError("事件类因子需要 volume")
    return _month_end_panel(d.volume.fillna(0.0), how="sum")


def _ann_month_flag(d: PaperData) -> pd.DataFrame:
    """月×code bool：该月内有财报公告。"""
    if d.ann_flag is None:
        raise ValueError("事件类因子需要 ann_flag")
    return _month_end_panel(d.ann_flag.fillna(False).astype(float), how="max") > 0.5


def pap_ann_premium(d: PaperData) -> pd.DataFrame:
    """财报公告溢价（Frazzini & Lamont）：成交量向公告月集中 + 本月预期公告。

    VCR = 近 48 个月中"公告月"的成交量合计 / 48 个月总成交量；
    factor = rank(VCR) × 1{本月预期公告（去年同月有公告）}。
    只在预期公告月有值（稀疏面板）；高集中度+即将公告 → 做多。
    """
    mvol = _monthly_volume(d)
    is_ann = _ann_month_flag(d)
    win = 48
    ann_vol = mvol.where(is_ann, 0.0).rolling(win, min_periods=win // 2).sum()
    tot_vol = mvol.rolling(win, min_periods=win // 2).sum()
    vcr = ann_vol / tot_vol.replace(0.0, np.nan)
    # 本月预期公告：去年同月有公告
    expected = is_ann.shift(12, fill_value=False)
    daily_idx = d.close.index
    vcr_d = _to_daily(vcr, daily_idx)
    exp_d = _to_daily(expected.astype(float), daily_idx) > 0.5
    factor = _cs_rank(vcr_d) * exp_d.astype(float)
    return factor.where(exp_d & vcr_d.notna())


def pap_ann_reversal(d: PaperData) -> pd.DataFrame:
    """公告期反转（公告前 3 日收益，次日有公告才触发）：

    EAR = close[t-2]/close[t-4] - 1（公告前第 4→第 2 交易日）；
    factor = -EAR × 1{t+1 日有公告}。原策略做多"公告前暴跌"股的公告窗口。
    """
    if d.ann_flag is None:
        raise ValueError("pap_ann_reversal 需要 ann_flag")
    ear = d.close.shift(2) / d.close.shift(4) - 1.0
    trigger = d.ann_flag.shift(-1, fill_value=False).astype(bool)
    return (-ear).where(trigger & ear.notna())


# ===========================================================================
# 四、残差动量（独立成段：需要构造 SMB/HML 时间序列）
# ===========================================================================
def pap_mom_residual(d: PaperData) -> pd.DataFrame:
    """残差动量（Blitz et al.）：三因子回归 α 的 12 个月动量。

    1) 用月末收益构造 MKT（全A等权）、SMB（小-大市值 20% 组）、
       HML（低-高 BP 20% 组）月度因子收益；
    2) 对每只股票滚动 36 个月回归 r = α + β1·MKT + β2·SMB + β3·HML + ε；
    3) resid_mom = Σ(近 12 个月 α)/std(近 12 个月 α)。
    剔除风格/市场后的动量更纯。原策略做多 resid_mom 前 10%。
    """
    p = d.pit
    if p.get("EQUITY") is None or d.market_cap is None:
        raise ValueError("pap_mom_residual 需要 EQUITY PIT 面板与 market_cap")
    c = d.close
    m_close = _month_end_panel(c, how="last")
    m_ret = m_close.pct_change(fill_method=None)
    m_cap = _month_end_panel(d.market_cap, how="last")
    bp = (p["EQUITY"] / d.market_cap.replace(0.0, np.nan))
    m_bp = _month_end_panel(bp, how="last")

    def _fac_ret(ret_m: pd.DataFrame, char_m: pd.DataFrame) -> pd.Series:
        """低特征组均值 − 高特征组均值（20%/80% 分位）。"""
        lo = char_m.le(char_m.quantile(0.2, axis=1), axis=0)
        hi = char_m.ge(char_m.quantile(0.8, axis=1), axis=0)
        lo_ret = ret_m.where(lo & ret_m.notna()).mean(axis=1)
        hi_ret = ret_m.where(hi & ret_m.notna()).mean(axis=1)
        return lo_ret - hi_ret

    smb = _fac_ret(m_ret, m_cap)   # 小市值 − 大市值
    hml = _fac_ret(m_ret, m_bp)    # 低 BP − 高 BP
    mkt = m_ret.mean(axis=1)
    F = pd.concat([mkt, smb, hml], axis=1).dropna(how="all")
    F.columns = ["MKT", "SMB", "HML"]

    # 滚动 36 月回归：每月末对窗口 [t-35, t] 做 pinv 批量求解
    R = m_ret.loc[F.index]
    Fv = F.values
    n_month, n_code = R.shape
    alpha = np.full((n_month, n_code), np.nan)
    win = 36
    A_pinv = None
    for t in range(win - 1, n_month):
        Fw = Fv[t - win + 1: t + 1]
        if np.isnan(Fw).any():
            Fw = np.nan_to_num(Fw)
        A = np.hstack([np.ones((win, 1)), Fw])
        A_pinv = np.linalg.pinv(A)
        Rw = R.values[t - win + 1: t + 1]
        valid = ~np.isnan(Rw).any(axis=0)
        coefs = np.full((4, n_code), np.nan)
        if valid.any():
            coefs[:, valid] = A_pinv @ np.nan_to_num(Rw[:, valid])
        alpha[t] = coefs[0]
    alpha_df = pd.DataFrame(alpha, index=F.index, columns=R.columns)
    resid_mom = (alpha_df.rolling(12, min_periods=9).sum()
                 / alpha_df.rolling(12, min_periods=9).std())
    return _to_daily(resid_mom, c.index)


# ===========================================================================
# 批量入口
# ===========================================================================
PRICE_FACTORS = (
    "pap_mom_12_1", "pap_mom_consistent", "pap_rev_5d", "pap_momxvol_6m",
    "pap_lowvol_1y", "pap_bab_beta", "pap_high52w", "pap_high52w_ind",
    "pap_breakout_atr", "pap_smallcap",
)
PIT_FACTORS = (
    "pap_momxag", "pap_accruals_bs", "pap_asset_growth", "pap_earn_quality",
    "pap_fscore", "pap_roa_size_adj", "pap_rd_intensity", "pap_value_bp",
    "pap_mom_residual",
)
EVENT_FACTORS = ("pap_ann_premium", "pap_ann_reversal")

PAPER_FACTORS: dict[str, callable] = {
    name: globals()[name]
    for name in PRICE_FACTORS + PIT_FACTORS + EVENT_FACTORS
}


def compute_paper_factors(d: PaperData, skip: tuple[str, ...] = ()) -> dict[str, pd.DataFrame]:
    """批量计算全部论文复现因子，返回 {name: panel}。

    单个因子失败只告警并跳过（数据缺失时其余因子仍可用）。
    """
    import logging
    log = logging.getLogger(__name__)
    panels: dict[str, pd.DataFrame] = {}
    for name, fn in PAPER_FACTORS.items():
        if name in skip:
            continue
        try:
            panels[name] = fn(d)
        except ValueError as e:
            log.warning("跳过 %s: %s", name, e)
    return panels
