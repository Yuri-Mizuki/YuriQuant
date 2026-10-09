"""全A 另类数据三族因子（altf 族）→ 并入 all_a_2018_2026。
====================================================================

背景（2026-10-07，P-C 立项）
----------------------------
另类数据通道（``data/altdata/``，2026-09-20 P0 落地）刻意留白因子层，
本脚本是第一轮消费：从三张结构化事件表构建 **20 个** date×code float32 面板
（insider 7 / macro 7 / news 6），收尾并入 registry + ic_h{h}。

三族与数据源
------------
======================  ================================  ======================  ==========
族                       表                                 内容                    实测历史
======================  ================================  ======================  ==========
``alt_insider_*``        ``alt_cninfo_holder.parquet``      董监高增减持（巨潮）     2010 起全历史
``alt_macro_*``          ``alt_macro_calendar.parquet``     宏观日历今值/预期/前值   2018-01 起
``alt_news_*``           ``alt_cls/cls_*.parquet``          财联社电报（分片）       回补水位起
======================  ====================               ======================  ==========

PIT 红线（逐表，引用 ``data.altdata.source_akshare.PIT_QUALITY``）
------------------------------------------------------------------
- ``cninfo_holder``：**有真实公告日** ``ann_date``（int YYYYMMDD）⇒ 滞后 0，
  事件从「公告日起第一个交易日」的收盘起可用（公告通常盘前刊登，close→close
  前向收益下无前视）；
- ``macro_calendar`` / ``cls``：公布时点 ``event_time`` / ``pub_time`` 带
  时分秒 ⇒ **15:00 收盘闸门**：事件从「第一个收盘时刻晚于公布时刻的交易日」起
  可用——盘中/凌晨发布当日可用，≥15:00 发布（如美国数据北京时间 20:30）顺延
  到下一交易日。宏观行 ``event_time`` 恰为 00:00:00（占 0.3%，视为"只知日期
  不知时点"）一律保守顺延一日；cls 的 00:00 是真实午夜时间戳，按原规则处理。
- 🚨 增持减持族**严禁**按 ``chg_date``（变动日）对齐——公告滞后变动日数日，
  那是前视。

口径
----
- **无事件记 NaN，不填 0（计数类例外）**：计数/条数/加权计数类在
  「数据源已覆盖的窗口内」0 是真值；数据源窗口之前（如 cls 回补水位之前、
  宏观表 2018 之前）一律 NaN；
- ``alt_insider_*`` 金额口径：``amount`` 缺失率 34%（实测 0.664 非空），
  方向列是权威符号源（原始符号仅 86~99% 与方向一致）⇒
  ``signed_amt = |amount| × (+1 增持 / −1 减持)``；流通市值分母 =
  ``float_share × close_raw.ffill()``（停牌日价格前向填充，市值连续）；
- **宏观族广播口径的先天局限（如实标注）**：广播到全截面 ⇒ 每日截面为常数，
  截面 Rank IC **按构造为 NaN**（时间序列 regime 信息，供面板型 ML 消费，
  非截面选股信号）；为给出可测截面变体，额外提供 1 个交互因子
  ``*_x_vol20``（广播强度 × 个股 20 日波动率百分位中心化——高波动股对宏观
  更敏感的假设，假设本身也标注在 label）；
- 快讯族情绪为**词典代理**（无情绪列）：title（62% 非空，缺失回退 content
  前 80 字）上数涨/跌方向词，逐条净词数截断到 [−1,1]，**不是模型情绪**。

方向表（宏观族，docstring 假设清单的一部分）
--------------------------------------------
「实际值超预期」对 A 股的利好(+1)/利空(−1)方向按关键词规则**自行设定**（数据源
不含方向语义）。规则按序首中即停，未命中 ⇒ 方向 0（不参与 surprise，只进
关注度计数）：

1. ``利率|联邦基金`` → −1（实际偏高 = 更紧，鹰派利空 A 股风险偏好）
2. ``CPI|PPI|PCE|通胀|物价|进口价格|平减`` → −1（通胀超预期 → 收紧担忧）
3. ``失业率|初请|续请`` → −1（就业恶化超预期）
4. ``工资`` → −1（工资通胀 → 联储鹰派）
5. ``非农|ADP|就业人口`` → +1（就业强于预期 → 风险偏好利好）
6. ``GDP|PMI|ISM|工业增加值|工业产出|零售|固定资产投资|出口|进口|贸易帐|
   新增人民币贷款|社会融资|社融|M1|M2|外汇储备|新屋开工|营建许可|耐用品订单|
   工厂订单|领先指标|NFIB|标普全球|信心|景气|利润`` → +1（增长/流动性超预期）

surprise 强度 = ``(actual − forecast) / |forecast|`` 截断 [−3,3]（|forecast|<1e-9
丢弃），按 importance 加权；计数类只保留 importance≥2。

用法
----
    python -m scripts.builders.build_alla_altfactors                     # 三族全量
    python -m scripts.builders.build_alla_altfactors --families insider,macro
    python -m scripts.builders.build_alla_altfactors --only alt_news_cnt_5d --resume

⚠️ cls 历史靠 ``fetch_altdata_daily --cls-backfill`` 游标回补（断点续传），
news 族覆盖率 = 回补水位之后那一段；水位越早 coverage 越高。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_altfactors")
from scripts.builders import common  # noqa: E402
from scripts.builders.common import HORIZONS, IC_CODE_STRIDE  # noqa: E402
# PIT 辅助件复用 holder_dyn builder 的唯一实现（ann_date 三 dtype 归一 /
# 交易日滞后 / 滚动窗口事件→面板），不另抄一份——见该模块 docstring 的实测事故记录。
from scripts.builders.build_alla_holder_dyn_factors import (  # noqa: E402
    _apply_lag, _rolling_sum_panel, ann_to_datetime)

DATASET = "all_a_2018_2026"
FAMILY = "altf"

#: A 股收盘闸门（北京时间）：公布时刻 ≥ 15:00 视为收盘后才可得
CN_CLOSE_CUTOFF = 15

#: 宏观方向表：按序首中即停（(关键词, 方向)），未命中方向 0
_MACRO_DIRECTION_RULES: tuple[tuple[str, int], ...] = (
    ("利率", -1), ("联邦基金", -1),
    ("CPI", -1), ("PPI", -1), ("PCE", -1), ("通胀", -1), ("物价", -1),
    ("进口价格", -1), ("平减", -1),
    ("失业率", -1), ("初请", -1), ("续请", -1),
    ("工资", -1),
    ("非农", 1), ("ADP", 1), ("就业人口", 1),
    ("GDP", 1), ("PMI", 1), ("ISM", 1), ("工业增加值", 1), ("工业产出", 1),
    ("零售", 1), ("固定资产投资", 1), ("出口", 1), ("进口", 1), ("贸易帐", 1),
    ("新增人民币贷款", 1), ("社会融资", 1), ("社融", 1), ("M1", 1), ("M2", 1),
    ("外汇储备", 1), ("新屋开工", 1), ("营建许可", 1), ("耐用品订单", 1),
    ("工厂订单", 1), ("领先指标", 1), ("NFIB", 1), ("标普全球", 1),
    ("信心", 1), ("景气", 1), ("利润", 1),
)

#: 快讯情绪词典（代理，非模型情绪）：title 缺失回退 content 前 80 字
_NEWS_POS = ("涨停", "大涨", "上调", "增持", "回购", "中标", "签订", "签约",
             "合作", "突破", "新高", "获批", "预增", "扭亏", "分红", "涨价",
             "买入评级", "战略合作")
_NEWS_NEG = ("跌停", "大跌", "下调", "减持", "质押", "违规", "处罚", "立案",
             "诉讼", "亏损", "预亏", "退市", "降价", "解禁", "爆雷", "违约",
             "警示", "冻结", "问询", "终止")

#: cls level → 注意力权重（加红等级 A/B 加权）
_LEVEL_WEIGHT = {"A": 3.0, "B": 2.0, "C": 1.0}


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读 (close_adj, close_raw, float_shares, raw_close_mask)。

    返回的四个面板共用同一 date×code 网格（KEEP_FROM 起）。
    """
    from config import Config
    from data.market_cap import build_shares_panel

    close_adj, close_raw = common.load_close_adj(with_raw=True)
    root = Path(str(Config.cache()["root"]))
    es_p = root / "equity_structure.parquet"
    if not es_p.exists():
        raise FileNotFoundError(f"缺少 {es_p} —— 流通市值分母不能缺")
    es = pd.read_parquet(es_p)
    shares = build_shares_panel(es, close_adj.index, close_adj.columns,
                                share_field="float_share")
    return close_adj, close_raw, shares, close_adj.notna()


# ---------------------------------------------------------------------------
# 公共：公布时刻 → 生效交易日（15:00 收盘闸门）
# ---------------------------------------------------------------------------
def pub_time_to_eff(ts: pd.Series, cal_idx: pd.DatetimeIndex,
                    unknown_time_to_next: bool) -> pd.DatetimeIndex:
    """带时分秒的公布时刻 → 「第一个收盘晚于它」的交易日。

    Args:
        ts: 公布时刻（datetime64）。
        cal_idx: 交易日历（normalize 过的日期）。
        unknown_time_to_next: True 时把恰为 00:00:00 的时刻视为
            「只知日期不知时点」，保守顺延一日（宏观表口径）；False 时
            视为真实午夜时间戳（cls 口径）。
    """
    ts = pd.to_datetime(ts)
    d = ts.dt.normalize()
    hour = ts.dt.hour
    # ≥15:00（或时点未知）→ 从次一自然日起找交易日；否则当日即可（若是交易日）
    adv = (hour >= CN_CLOSE_CUTOFF)
    if unknown_time_to_next:
        adv = adv | ((hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0))
    d = d + pd.to_timedelta(adv.astype(int), unit="D")
    pos = cal_idx.searchsorted(pd.DatetimeIndex(d), side="left")
    # 日历末之后的事件必须丢弃，不能钳到最后一天——否则最后一天的面板值
    # 混入「盘后才发布」的未来信息（与 _apply_lag 的越界丢弃同一条红线）。
    ok = ((pd.DatetimeIndex(d) >= cal_idx[0]) & (np.asarray(pos) < len(cal_idx)))
    pos = np.clip(pos, 0, len(cal_idx) - 1)
    eff = cal_idx[np.asarray(pos, dtype=int)]
    return eff, ok


def _daily_to_panel(series: pd.Series, first_valid: pd.Timestamp,
                    cal_idx: pd.DatetimeIndex, codes: pd.Index,
                    mask: pd.DataFrame) -> pd.DataFrame:
    """日级序列 → date×code 广播面板（数据窗口之前 NaN，窗口内广播后按 mask 收边）。"""
    s = series.reindex(cal_idx)
    s[s.index < first_valid] = np.nan
    if s.notna().sum() == 0:
        return pd.DataFrame(np.nan, index=cal_idx, columns=codes, dtype=np.float32)
    panel = pd.DataFrame(np.repeat(s.to_numpy(dtype="float32")[:, None], len(codes), axis=1),
                         index=cal_idx, columns=codes)
    return panel.where(mask).astype(np.float32)


# ---------------------------------------------------------------------------
# 族一：alt_insider_*（董监高增减持，公告日对齐，lag 0）
# ---------------------------------------------------------------------------
def build_insider(cal_idx, codes, shares: pd.DataFrame,
                  close_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from data.altdata.source_akshare import SUGGESTED_LAG_TRADING_DAYS as LAG
    from config import Config

    root = Path(str(Config.cache()["root"]))
    p = root / "alt_cninfo_holder.parquet"
    if not p.exists():
        log.warning("  insider 源缺失（%s），该族不可建", p)
        return {}
    mh = pd.read_parquet(p)
    if mh.empty:
        return {}

    direction = mh["direction"].astype(str)
    sign = np.where(direction == "增持", 1.0, -1.0)
    mh = mh.assign(
        signed_amt=(pd.to_numeric(mh["amount"], errors="coerce").abs() * sign),
        signed_qty=pd.to_numeric(mh["signed_qty"], errors="coerce"),
        evt_sign=sign,
        person=mh["person"].astype(str),
    )
    ev = _apply_lag(mh, LAG["cninfo_holder"], cal_idx)   # 公告日 → 生效交易日
    ev_codes = ev["code"].isin(codes)
    ev = ev[ev_codes]
    if ev.empty:
        return {}
    span = (f"{ann_to_datetime(ev['ann_date']).min().date()}"
            f"~{ann_to_datetime(ev['ann_date']).max().date()}")

    float_mv = (shares * close_raw.ffill()).where(shares > 0)
    # ⚠️ _rolling_sum_panel 内部用 ``mask.notna()`` 收边 ⇒ 必须传**浮点股本面板**
    # （模板口径）——传布尔面板时 notna() 恒真，未上市/退市区会被 0 填满。
    mask = shares
    mask_bool = shares.notna()
    out: dict[str, pd.DataFrame] = {}

    # 金额口径（amount 覆盖 66% 事件；NaN 行自动被 _rolling_sum_panel 丢弃）
    amt20 = _rolling_sum_panel(ev, "signed_amt", 20, cal_idx, codes, mask)
    amt60 = _rolling_sum_panel(ev, "signed_amt", 60, cal_idx, codes, mask)
    out["alt_insider_netamt_floatmv_20d"] = (amt20 / float_mv).astype(np.float32)
    out["alt_insider_netamt_floatmv_60d"] = (amt60 / float_mv).astype(np.float32)
    out["alt_insider_netbuy_mom_20_60"] = (out["alt_insider_netamt_floatmv_20d"]
                                           - out["alt_insider_netamt_floatmv_60d"]
                                           ).astype(np.float32)
    # 股数口径（signed_qty 100% 覆盖，金额缺失的补位视角）
    out["alt_insider_netqty_shares_60d"] = (
        _rolling_sum_panel(ev, "signed_qty", 60, cal_idx, codes, mask)
        / shares).astype(np.float32)
    # 计数类（0 是真值）
    buy = ev[ev["direction"].astype(str) == "增持"]
    sell = ev[ev["direction"].astype(str) == "减持"]
    out["alt_insider_buy_cnt_60d"] = _rolling_sum_panel(
        buy.assign(one=1.0), "one", 60, cal_idx, codes, mask)
    out["alt_insider_sell_cnt_60d"] = _rolling_sum_panel(
        sell.assign(one=1.0), "one", 60, cal_idx, codes, mask)
    # 行为人广度：每日 (eff, code) 独立行为人数 → 20 日滚动求和（人·日口径）
    g = (ev.groupby(["eff", "code"], sort=False)["person"].nunique().unstack()
         .reindex(index=cal_idx, columns=codes).fillna(0.0).astype(np.float32))
    out["alt_insider_person_days_20d"] = (
        g.rolling(20, min_periods=1).sum().where(mask_bool)).astype(np.float32)

    log.info("  insider: %d 事件（公告日 %s，amount 非空 %.1f%%）→ 7 因子",
             len(ev), span, ev["signed_amt"].notna().mean() * 100)
    return out


# ---------------------------------------------------------------------------
# 族二：alt_macro_*（宏观日历 surprise，公布时刻对齐 + 15:00 收盘闸门）
# ---------------------------------------------------------------------------
def _macro_direction(event: pd.Series) -> pd.Series:
    """事件名 → 方向（+1/−1/0），按序首中即停。"""
    ev = event.astype(str)
    direction = pd.Series(0, index=event.index, dtype="float32")
    done = pd.Series(False, index=event.index)
    for pat, d in _MACRO_DIRECTION_RULES:
        hit = ev.str.contains(pat, regex=False) & ~done
        direction[hit] = d
        done |= hit
    return direction


def build_macro(cal_idx, codes, mask: pd.DataFrame,
                close_adj: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from config import Config

    root = Path(str(Config.cache()["root"]))
    p = root / "alt_macro_calendar.parquet"
    if not p.exists():
        log.warning("  macro 源缺失（%s），该族不可建", p)
        return {}
    m = pd.read_parquet(p)
    m = m[m["forecast"].notna() & m["actual"].notna()].copy()
    if m.empty:
        return {}

    m["importance"] = pd.to_numeric(m["importance"], errors="coerce").fillna(0)
    m = m[m["importance"] >= 2].copy()          # 计数/强度只保留 importance≥2
    if m.empty:
        return {}
    actual = pd.to_numeric(m["actual"], errors="coerce")
    forecast = pd.to_numeric(m["forecast"], errors="coerce")
    fc_abs = forecast.abs()
    surprise = ((actual - forecast) / fc_abs.where(fc_abs > 1e-9)).clip(-3.0, 3.0)
    m = m.assign(surprise=surprise, direction=_macro_direction(m["event"]))
    m = m.dropna(subset=["surprise"])
    eff, ok = pub_time_to_eff(m["event_time"], cal_idx, unknown_time_to_next=True)
    m = m.assign(eff=eff)[ok]
    if m.empty:
        return {}
    first_valid = m["eff"].min()

    signed_str = m["direction"] * m["surprise"] * m["importance"]
    df = m.assign(signed_str=signed_str, pos=(signed_str > 0).astype(float),
                  neg=(signed_str < 0).astype(float),
                  one=1.0)
    daily = df.groupby("eff").agg(pos_cnt=("pos", "sum"), neg_cnt=("neg", "sum"),
                                  str_sum=("signed_str", "sum"))
    # imp3 关注度：importance≥3 的全部事件（不限有 forecast / 方向命中）
    m_all = pd.read_parquet(p)
    m_all = m_all[pd.to_numeric(m_all["importance"], errors="coerce") >= 3]
    if not m_all.empty:
        eff_all, ok_all = pub_time_to_eff(m_all["event_time"], cal_idx,
                                          unknown_time_to_next=True)
        daily["imp3"] = pd.Series(1.0, index=m_all[ok_all].index).groupby(
            eff_all[ok_all]).sum()
    daily = daily.reindex(cal_idx)
    daily.loc[daily.index < first_valid, :] = np.nan
    daily_cnt = daily[["pos_cnt", "neg_cnt"]].fillna(0.0)   # 计数类：窗口内 0 是真值
    daily_str = daily[["str_sum", "imp3"]].fillna(0.0)

    out: dict[str, pd.DataFrame] = {}
    out["alt_macro_surprise_pos_cnt_1d"] = _daily_to_panel(
        daily_cnt["pos_cnt"], first_valid, cal_idx, codes, mask)
    out["alt_macro_surprise_neg_cnt_1d"] = _daily_to_panel(
        daily_cnt["neg_cnt"], first_valid, cal_idx, codes, mask)
    str5 = daily_str["str_sum"].rolling(5, min_periods=1).sum()
    net20 = daily_str["str_sum"].rolling(20, min_periods=1).sum()
    imp3_20 = daily_str["imp3"].rolling(20, min_periods=1).sum()
    out["alt_macro_surprise_str_5d"] = _daily_to_panel(str5, first_valid, cal_idx, codes, mask)
    out["alt_macro_surprise_net_20d"] = _daily_to_panel(net20, first_valid, cal_idx, codes, mask)
    chn = df[df["region"].astype(str) == "中国"]
    chn5 = (chn.groupby("eff")["signed_str"].sum().reindex(cal_idx).fillna(0.0)
            .rolling(5, min_periods=1).sum())
    out["alt_macro_chn_surprise_str_5d"] = _daily_to_panel(chn5, first_valid, cal_idx, codes, mask)
    out["alt_macro_imp3_cnt_20d"] = _daily_to_panel(imp3_20, first_valid, cal_idx, codes, mask)

    # 交互变体：广播强度 × 个股 20 日波动率百分位（中心化）——高波动股更宏观敏感（假设）
    ret = close_adj.pct_change(fill_method=None)
    vol20 = ret.rolling(20, min_periods=10).std()
    vol_rank = vol20.rank(axis=1, pct=True) - 0.5
    z5 = (str5 - str5.rolling(250, min_periods=60).mean()) / str5.rolling(
        250, min_periods=60).std()
    interact = vol_rank.mul(z5.reindex(cal_idx), axis=0)
    out["alt_macro_surprise_str5_x_vol20"] = interact.where(mask).astype(np.float32)

    hit_frac = float((df["direction"] != 0).mean())
    log.info("  macro: %d 条 surprise 事件（%s ~ %s，方向表命中率 %.1f%%）→ 7 因子",
             len(df), first_valid.date(), m["eff"].max().date(), hit_frac * 100)
    return out


# ---------------------------------------------------------------------------
# 族三：alt_news_*（财联社电报，pub_time 对齐 + 15:00 收盘闸门）
# ---------------------------------------------------------------------------
def load_cls_slim(cls_dir: Path) -> pd.DataFrame:
    """逐分片读取 cls 存档并**当场瘦身**（丢弃长 content，只留派生量）。

    全量存档百万行 × 长文本，整读会在 concat 处吃掉数 GB 内存；逐分片
    先算好 level 权重 / 文本长度 / 词典情绪再合并。
    """
    files = sorted(cls_dir.glob("cls_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        d = pd.read_parquet(f, columns=["news_id", "pub_time", "level",
                                        "stock_codes", "title", "content"])
        title = d["title"].astype("string")
        content = d["content"].astype("string")
        text = title.fillna(content.str[:80]).fillna("")
        pos_n = pd.Series(0, index=d.index, dtype="int16")
        neg_n = pd.Series(0, index=d.index, dtype="int16")
        for k in _NEWS_POS:
            pos_n += text.str.count(k).fillna(0).astype("int16")
        for k in _NEWS_NEG:
            neg_n += text.str.count(k).fillna(0).astype("int16")
        slim = pd.DataFrame({
            "pub_time": pd.to_datetime(d["pub_time"]),
            "weight": d["level"].map(_LEVEL_WEIGHT).fillna(1.0).astype("float32"),
            "is_red": d["level"].isin(["A", "B"]).astype("float32"),
            "text_len": text.str.len().astype("float32"),
            "sent": (pos_n - neg_n).clip(-1, 1).astype("float32"),
            "stock_codes": d["stock_codes"].astype("string"),
        }, index=d.index)
        slim["news_id"] = d["news_id"].values
        frames.append(slim)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset="news_id")
    return out.drop(columns="news_id")


def build_news(cal_idx, codes, mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from config import Config

    cls_dir = Path(str(Config.cache()["root"])) / "alt_cls"
    if not cls_dir.exists():
        log.warning("  news 源缺失（%s），该族不可建——先跑 "
                    "fetch_altdata_daily --tables cls --cls-backfill", cls_dir)
        return {}
    cls = load_cls_slim(cls_dir)
    if cls.empty:
        return {}
    linked = cls[cls["stock_codes"].notna()].copy()
    # 一条快讯可挂多码（逗号分隔）；逐码展开后过滤到面板股票池
    parts = (linked["stock_codes"].str.split(",").explode().str.strip())
    ex = linked.loc[parts.index].assign(code=parts.values)
    ex = ex[ex["code"].isin(codes)]
    if ex.empty:
        return {}
    eff, ok = pub_time_to_eff(ex["pub_time"], cal_idx, unknown_time_to_next=False)
    ex = ex.assign(eff=eff)[ok]
    first_valid = ex["eff"].min()                 # = cls 回补水位
    log.info("  news: 存档 %d 条，挂个股 %d 条（水位 %s 起）",
             len(cls), len(ex), first_valid.date())

    agg = (ex.groupby(["eff", "code"], sort=False)
           .agg(cnt=("weight", "size"), attention=("weight", "sum"),
                red=("is_red", "sum"), sent=("sent", "sum"),
                len_sum=("text_len", "sum")))
    wide = {c: agg[c].unstack("code").reindex(index=cal_idx, columns=codes)
            for c in ("cnt", "attention", "red", "sent", "len_sum")}
    mask_in = pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    mask_in[mask_in.index >= first_valid] = 1.0   # 数据窗口内才允许 0
    mask_in = mask_in.notna()

    def _roll(col: str, w: int) -> pd.DataFrame:
        x = wide[col].fillna(0.0).astype(np.float32)
        r = x.rolling(w, min_periods=1).sum()
        return r.where(mask & mask_in).astype(np.float32)

    out: dict[str, pd.DataFrame] = {}
    out["alt_news_cnt_5d"] = _roll("cnt", 5)
    out["alt_news_attention_5d"] = _roll("attention", 5)
    out["alt_news_red_cnt_20d"] = _roll("red", 20)
    out["alt_news_lexsent_20d"] = _roll("sent", 20)
    # 条数 zscore（时间序列口径：20 日条数 vs 滚动 250 日分布）
    cnt20 = wide["cnt"].fillna(0.0).astype(np.float32).rolling(20, min_periods=1).sum()
    mu = cnt20.rolling(250, min_periods=60).mean()
    sd = cnt20.rolling(250, min_periods=60).std()
    out["alt_news_cnt_20d_z"] = ((cnt20 - mu) / sd.where(sd > 0)
                                 ).where(mask & mask_in).astype(np.float32)
    # 平均文本长度（信息强度代理；窗口内无快讯 ⇒ NaN，非计数类）
    r_len = wide["len_sum"].fillna(0.0).astype(np.float32).rolling(20, min_periods=1).sum()
    r_cnt = wide["cnt"].fillna(0.0).astype(np.float32).rolling(20, min_periods=1).sum()
    out["alt_news_mean_len_20d"] = (r_len / r_cnt.where(r_cnt > 0)
                                    ).where(mask & mask_in).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# registry 标签
# ---------------------------------------------------------------------------
SUBFAMILY = {"alt_insider": "增减持", "alt_macro": "宏观日历", "alt_news": "快讯情绪"}

LABELS: dict[str, str] = {
    # 增减持族
    "alt_insider_netamt_floatmv_20d": "董监高净增持金额/流通市值·20日",
    "alt_insider_netamt_floatmv_60d": "董监高净增持金额/流通市值·60日",
    "alt_insider_netbuy_mom_20_60": "董监高净增持动量(20日−60日金额比)",
    "alt_insider_netqty_shares_60d": "董监高净增持股数/流通股本·60日",
    "alt_insider_buy_cnt_60d": "董监高增持公告条数·60日",
    "alt_insider_sell_cnt_60d": "董监高减持公告条数·60日",
    "alt_insider_person_days_20d": "董监高行为人·日数·20日",
    # 宏观日历族
    "alt_macro_surprise_pos_cnt_1d": "利好方向宏观超预期事件数·当日",
    "alt_macro_surprise_neg_cnt_1d": "利空方向宏观超预期事件数·当日",
    "alt_macro_surprise_str_5d": "宏观超预期加权强度·5日(广播)",
    "alt_macro_surprise_net_20d": "宏观超预期净强度·20日(广播)",
    "alt_macro_chn_surprise_str_5d": "中国宏观超预期强度·5日(广播)",
    "alt_macro_imp3_cnt_20d": "高重要度(≥3)宏观事件数·20日(广播)",
    "alt_macro_surprise_str5_x_vol20": "宏观超预期强度×20日波动率敏感度(交互)",
    # 快讯情绪族
    "alt_news_cnt_5d": "个股快讯条数·5日",
    "alt_news_attention_5d": "个股快讯加权注意力·5日(加红A/B加权)",
    "alt_news_red_cnt_20d": "个股加红快讯条数·20日",
    "alt_news_cnt_20d_z": "个股快讯条数时序zscore·20/250日",
    "alt_news_lexsent_20d": "个股快讯词典净情绪·20日(代理)",
    "alt_news_mean_len_20d": "个股快讯平均文本长度·20日(信息强度代理)",
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default=None, help="仅构建指定因子")
    ap.add_argument("--families", default="insider,macro,news",
                    help="逗号分隔：insider,macro,news")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / f"factor_stats_{FAMILY}.jsonl"
    done = ({json.loads(line)["name"] for line in
             stats_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            if args.resume and stats_path.exists() else set())

    t0 = time.time()
    close_adj, close_raw, shares, mask = load_inputs()
    cal_idx, codes = close_adj.index, close_adj.columns
    log.info("网格 %d 日 × %d 码（%s ~ %s）", len(cal_idx), len(codes),
             cal_idx[0].date(), cal_idx[-1].date())

    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = codes[::IC_CODE_STRIDE]

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    builders = {
        "insider": lambda: build_insider(cal_idx, codes, shares, close_raw),
        "macro": lambda: build_macro(cal_idx, codes, mask, close_adj),
        "news": lambda: build_news(cal_idx, codes, mask),
    }
    panels_all: dict[str, pd.DataFrame] = {}
    for fam in families:
        if fam not in builders:
            raise SystemExit(f"未知族 {fam!r}；可用：{sorted(builders)}")
        t1 = time.time()
        built = builders[fam]()
        log.info("族 %s 构建完成：%d 因子（%.0fs）", fam, len(built), time.time() - t1)
        panels_all.update(built)

    defs = {k: v for k, v in LABELS.items() if k in panels_all}
    if args.only:
        defs = {k: v for k, v in defs.items() if k == args.only}
        panels_all = {k: v for k, v in panels_all.items() if k == args.only}

    for i, (name, label) in enumerate(sorted(defs.items()), 1):
        if name in done:
            continue
        p = panels_all[name].replace([np.inf, -np.inf], np.nan).clip(
            -1e4, 1e4).astype(np.float32)
        if p.notna().sum().sum() == 0:
            log.warning("[%d/%d] %s 全 NaN，跳过落盘", i, len(defs), name)
            continue
        p.to_parquet(panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        row_std = p.std(axis=1, skipna=True)
        active = row_std > 0
        active_frac = float(active.mean()) if len(active) else 0.0
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        sub = SUBFAMILY.get("_".join(name.split("_")[:2]), "")
        row = {"name": name, "set": FAMILY, "label": label, "subfamily": sub,
               "coverage": cov, "active_day_frac": active_frac,
               "active_first": (str(p.index[active].min().date()) if active.any() else None),
               "active_last": (str(p.index[active].max().date()) if active.any() else None),
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if active_frac < 0.5:
            log.warning("[%d/%d] %s ⚠️ 有效交易日仅 %.1f%%（%s ~ %s）—— "
                        "IC 只覆盖这段，别按全样本解读", i, len(defs), name,
                        active_frac * 100, row["active_first"], row["active_last"])
        log.info("[%d/%d] %s sub=%s cov=%.3f eff=%.3f ic_h1=%s | %.0fs",
                 i, len(defs), name, sub, cov, active_frac,
                 f"{row['ic_mean_h1']:+.4f}" if np.isfinite(row["ic_mean_h1"]) else "nan",
                 time.time() - t0)

    if not args.only:
        common.merge_outputs(ds_dir, FAMILY, family_tag="事件",
                             empty_log="altf stats 为空，无输出可合并")
    log.info("altf 构建完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
