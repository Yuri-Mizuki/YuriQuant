"""事件驱动组合的独立回测（2026-11-15）
==========================================

为什么重做（以及本脚本与 `event_reach_probe` 的区别）
------------------------------------------------------
`event_reach_probe` 测的是「事件能不能改变 K15 的持仓」——那是**横截面框架下的
否决/换股权**，分子 spread 天生被分母 K=15 压死，得到的必然是「事件类弱一个量级」。

但事件驱动的定义不是「改善选股」，而是：

> **事件发生 ⇒ 买入该股 ⇒ 持有 N 天 ⇒ 卖出。** 与选股模型零关系。

两者的经济学完全不同：截面口径赌的是「事件股排序优于全市场」，事件驱动赌的是
「事件股在事件窗内相对市场有异常收益」。后者是个**日历时间组合（calendar-time
portfolio）**，不需要截面广度、不需要 K15、不需要因子合成。

本脚本只做后者，**不碰 K15、不进因子池、不做否决**。

三个口径（都报，缺一不可）
--------------------------
1. **CAR 事件窗**：以可见日为 t=0，看 t−5 ~ t+60 的累积异常收益形态。
   学术口径，用来判断「漂移出现在第几天、持续多久」。
2. **日频等权组合**：每一天都等权持有「当前处于持有窗内的全部事件股」。
   这是**可投资口径**——真实的资金占用、真实的换手、真实的净值。
   收益按 T+1 开盘建仓、H 日后开盘卖出精确计算。
3. **市值匹配基准**：事件股天然偏小盘。若只对全 A 等权比，测到的是小市值 beta
   而不是事件 alpha。故同时给「同市值分位组」基准（默认 5 组）。

判据（先定后算）
----------------
- 扣费后年化超额 ≥ **+3%**（单边 13.5bp，按实际换手计）
- NW t ≥ **2.0**（日频，lag 5）
- 分年同号 ≥ **60%**（8 年中至少 5 年为正）
- 安慰剂（事件日随机平移，同规模、同代码分布）p < **0.10**
四条全过才算「候选」，进 TODO；任一不过即判负收口。

用法
----
    python -m scripts.evaluation.event_driven_probe                  # 全部事件族，H=20
    python -m scripts.evaluation.event_driven_probe --only fc_pos,inc
    python -m scripts.evaluation.event_driven_probe --hold 10,20,60  # 多持有期
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"
OUT = ROOT / "reports" / "event_driven_probe"
PQ = Path("D:/data/parquet")

#: 单边 13.5bp = 往返 27bp（佣金 2bp + 印花 5bp + 滑点 20bp）。
#: 🚨 **固定值、非时变**（2026-11-18 定案，见 ``reports/event_driven_probe/README.md`` §成本口径）：
#: 不按 ``backtest/costs.py::STAMP_DUTY_CHANGES`` 在 2023-08-28 前取千1。已知低估
#: 0.30pp/年（H=20）、0.56pp/年（H=10），不改变任何判定；改成时变会影响全部历史数字，
#: 需要与报告同步改，别单独动这一个常量。
COST_ONE_WAY = 0.00135
NQ = 5                          # 市值分位组数
START, END = "2018-01-01", "2026-11-06"


# ============================================================ 基础数据
def _dt(s: pd.Series) -> pd.Series:
    """项目内事件表日期有三种编码，统一到 Timestamp。"""
    if pd.api.types.is_integer_dtype(s):
        v = s.astype("int64")
        med = float(v.dropna().median()) if v.notna().any() else 0.0
        if 1.9e7 < med < 2.2e7:            # YYYYMMDD
            return pd.to_datetime(v.astype(str), format="%Y%m%d", errors="coerce")
        if med > 1e14:                     # 微秒 epoch
            return pd.to_datetime(v, unit="us", errors="coerce")
        if med > 1e11:                     # 毫秒 epoch
            return pd.to_datetime(v, unit="ms", errors="coerce")
        return pd.to_datetime(v, unit="s", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _sym(code: pd.Series) -> pd.Series:
    """6 位代码 → ``000001.SZ``；已带后缀的原样返回。"""
    c = code.astype(str).str.strip()
    has = c.str.contains(r"\.", regex=True)
    six = c.str.zfill(6)
    p = six.str[0]
    suf = np.select(
        [p.eq("6"), p.isin(["0", "3"]), p.isin(["4", "8", "9"])],
        [".SH", ".SZ", ".BJ"], default="",
    )
    return pd.Series(np.where(has, c, six + suf), index=c.index)


def load_base():
    close = pd.read_parquet(BASE_DIR / "close_adj.parquet").loc[START:END]
    openp = pd.read_parquet(BASE_DIR / "open_adj.parquet").loc[START:END]
    mask = pd.read_parquet(BASE_DIR / "tradable_mask.parquet").loc[START:END]
    bench = pd.read_parquet(BASE_DIR / "bench_eqw.parquet")["ret"].loc[START:END]
    mcap = pd.read_parquet(BASE_DIR / "market_cap.parquet").loc[START:END]
    close = close.reindex(columns=openp.columns)
    mask = mask.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    mcap = mcap.reindex(index=close.index, columns=close.columns)
    return close, openp, mask, bench, mcap


def _avail(cal: pd.DatetimeIndex, d: pd.Series) -> pd.Series:
    """日期 → 日历上「当天或之后的第一个交易日」。

    ⚠️ 早于样本起点的事件必须置 NaT，不能顺延到首日（2026-11-15 修）。
    ``searchsorted`` 对 ``d < cal[0]`` 一律返回 0，若不拦，2010–2017 的历史
    公告会被全部堆到 2018-01-02 建仓（实测 748 个事件 ⇒ 首月一次性 664 只
    持仓），等于用一个「不存在的全量建仓日」污染首年收益。
    """
    dv = pd.DatetimeIndex(pd.to_datetime(d, errors="coerce"))
    pos = cal.searchsorted(dv.values, side="left")
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    ok = (pos < len(cal)) & dv.notna() & (dv.values >= cal[0].to_datetime64())
    out.loc[ok] = cal[pos[ok]]
    return out


# ============================================================ 事件族
# 每个族返回 DataFrame[code, avail, strength]：strength 越大 = 事件越"浓"，
# 用于分层单调性检验；不参与主回测（主回测是「全部事件等权」）。
def _pick(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def fam_fc_pos(cal, close):
    """业绩预增：变动下限 > 0（确定同比增长）。strength = 变动中值 %。"""
    df = pd.read_parquet(PQ / "profit_notice.parquet",
                         columns=["code", "ann_date", "first_ann_date",
                                  "p_change_min", "p_change_max"])
    lo = pd.to_numeric(df["p_change_min"], errors="coerce")
    hi = pd.to_numeric(df["p_change_max"], errors="coerce")
    df = df.assign(strength=(lo + hi) / 2.0)
    df = df[lo > 0]
    df["code"] = _sym(df["code"])
    d = _dt(df["first_ann_date"])
    d = d.where(d.notna(), _dt(df["ann_date"]))
    df = df.assign(avail=_avail(cal, d)).dropna(subset=["avail"])
    return df[["code", "avail", "strength"]]


def fam_fc_neg(cal, close):
    """业绩预减：变动上限 < 0。strength = |变动中值|。"""
    df = pd.read_parquet(PQ / "profit_notice.parquet",
                         columns=["code", "ann_date", "first_ann_date",
                                  "p_change_min", "p_change_max"])
    lo = pd.to_numeric(df["p_change_min"], errors="coerce")
    hi = pd.to_numeric(df["p_change_max"], errors="coerce")
    df = df.assign(strength=-(lo + hi) / 2.0)
    df = df[hi < 0]
    df["code"] = _sym(df["code"])
    d = _dt(df["first_ann_date"])
    d = d.where(d.notna(), _dt(df["ann_date"]))
    df = df.assign(avail=_avail(cal, d)).dropna(subset=["avail"])
    return df[["code", "avail", "strength"]]


def _holder_fam(cal, kw: str, sign: int):
    df = pd.read_parquet(PQ / "alt_cninfo_holder.parquet",
                         columns=["code", "ann_date", "direction", "chg_ratio"])
    df = df[df["direction"].astype(str).str.contains(kw)]
    r = pd.to_numeric(df["chg_ratio"], errors="coerce").abs()
    df = df.assign(strength=r)[r >= 0.1]
    df["code"] = _sym(df["code"])
    df = df.assign(avail=_avail(cal, _dt(df["ann_date"]))).dropna(subset=["avail"])
    return df[["code", "avail", "strength"]]


def fam_inc(cal, close):
    """董监高增持（全部）。注：该表只有董监高，不含大股东/机构增持。"""
    return _holder_fam(cal, "增持", +1)


def _inc_refined(cal, close, reasons=None, rel=None, min_n=1, min_ratio=0.0):
    """精炼版增持：只保留主动买入，并按 (code, 可见日) 聚合。

    为什么要聚合：同一天多名董监高增持同一只票是多条独立记录，聚合后
    「人数」和「合计比例」才是事件的真实强度；不聚合会让「一人买 200 股」
    和「五人合计买 1%」等价。
    """
    df = pd.read_parquet(PQ / "alt_cninfo_holder.parquet",
                         columns=["code", "ann_date", "direction", "subject_type",
                                  "chg_ratio", "reason", "relation"])
    df = df[df["direction"].astype(str).str.contains("增持")]
    if reasons is not None:
        df = df[df["reason"].astype(str).str.contains("|".join(reasons), na=False)]
    if rel is not None:
        df = df[df["relation"].astype(str).isin(rel)]
    df["code"] = _sym(df["code"])
    df["r"] = pd.to_numeric(df["chg_ratio"], errors="coerce").abs()
    df = df.assign(avail=_avail(cal, _dt(df["ann_date"]))).dropna(subset=["avail"])
    df = df[df["r"] >= min_ratio]
    if df.empty:
        return df[["code", "avail", "strength"]]
    g = (df.groupby(["code", "avail"])
           .agg(strength=("r", "sum"), n=("r", "size"))
           .reset_index())
    g = g[g["n"] >= min_n]
    return g[["code", "avail", "strength"]]


#: 主动买入的成交方式（剔除「分红送转」「股权激励实施」等非主动事件）
_ACTIVE = ("竞价交易", "二级市场买卖", "大宗交易")


def fam_inc_pure(cal, close):
    """主动增持 + 增持人本人（剔除亲属代持等间接口径），比例 ≥0.1%。"""
    return _inc_refined(cal, close, reasons=_ACTIVE, rel=("本人",), min_ratio=0.1)


def fam_inc_multi(cal, close):
    """同日 ≥2 名董监高主动增持（集体行动）。"""
    return _inc_refined(cal, close, reasons=_ACTIVE, min_n=2, min_ratio=0.1)


def fam_inc_big(cal, close):
    """同日合计增持 ≥0.5% 总股本（显著比例）。"""
    return _inc_refined(cal, close, reasons=_ACTIVE, min_ratio=0.5)


def fam_red(cal, close):
    """股东/董监高减持 ≥0.1% 总股本。"""
    return _holder_fam(cal, "减持", -1)


def fam_unlock(cal, close):
    """解禁：解禁股占总股本 ≥1%。avail 允许前视到解禁日之前（公告即已知）。"""
    df = pd.read_parquet(PQ / "equity_restricted.parquet",
                         columns=["code", "list_date", "share_ratio"])
    df["code"] = _sym(df["code"])
    sr = pd.to_numeric(df["share_ratio"], errors="coerce")
    df = df.assign(strength=sr)[sr >= 1.0]
    df = df.assign(avail=_avail(cal, _dt(df["list_date"]))).dropna(subset=["avail"])
    return df[["code", "avail", "strength"]]


def _block_fam(cal, close, side: str, cut: float):
    df = pd.read_parquet(PQ / "block_trading.parquet",
                         columns=["trade_date", "code", "b_share_price"])
    df["code"] = _sym(df["code"])
    d = _dt(df["trade_date"])
    idx = close.index.get_indexer(pd.DatetimeIndex(d))
    col = close.columns.get_indexer(df["code"])
    px = np.full(len(df), np.nan)
    ok = (idx >= 0) & (col >= 0)
    px[ok] = close.values[idx[ok], col[ok]]
    disc = df["b_share_price"].to_numpy() / px - 1.0
    df = df.assign(disc=disc, strength=np.abs(disc))
    df = df[disc <= cut] if side == "disc" else df[disc >= cut]
    # 大宗成交价收盘后公告 ⇒ 可见日 = 成交日之后第一个交易日
    d = d.loc[df.index]
    nxt = cal.searchsorted(pd.DatetimeIndex(d).values, side="right")
    keep = nxt < len(cal)
    df = df.loc[keep].assign(avail=cal[nxt[keep]])
    return df[["code", "avail", "strength"]]


def fam_blk_disc(cal, close):
    """大宗折价 ≤ −5%。"""
    return _block_fam(cal, close, "disc", -0.05)


def fam_blk_prem(cal, close):
    """大宗溢价 ≥ +5%（接盘方愿意付溢价）。"""
    return _block_fam(cal, close, "prem", 0.05)


def _holdnum_fam(cal, close, up: bool):
    df = pd.read_parquet(PQ / "holder_num.parquet",
                         columns=["code", "ann_date", "holder_end_date", "holder_num"])
    df["code"] = _sym(df["code"])
    df = df.dropna(subset=["holder_num"]).sort_values(["code", "holder_end_date"])
    chg = df.groupby("code")["holder_num"].pct_change()
    df = df.assign(strength=chg.abs())
    df = df[chg >= 0.10] if up else df[chg <= -0.10]
    df = df.assign(avail=_avail(cal, _dt(df["ann_date"]))).dropna(subset=["avail"])
    return df[["code", "avail", "strength"]]


def fam_hn_up(cal, close):
    """股东户数环比 +10%（筹码分散）。"""
    return _holdnum_fam(cal, close, True)


def fam_hn_dn(cal, close):
    """股东户数环比 −10%（筹码集中）。"""
    return _holdnum_fam(cal, close, False)


def _fin_fam(cal, close, sign: int, cut: float = 30.0):
    """业绩快报：净利润同比增速 ≥ +cut% / ≤ −cut%。strength = |增速|。

    与 `profit_notice`（预告，是区间估计）不同，快报是**审计前的实际数**，
    信息含量更高。日期优先用 ``ACTUAL_ANN_DATE``（实际披露日），缺失才回落
    ``ann_date``（预约披露日），避免用预约日造成前视。
    """
    df = pd.read_parquet(PQ / "profit_express.parquet",
                         columns=["code", "ann_date", "ACTUAL_ANN_DATE",
                                  "YOY_GR_GROSS_PRO"])
    y = pd.to_numeric(df["YOY_GR_GROSS_PRO"], errors="coerce")
    df = df.assign(strength=y.abs())
    df = df[y >= cut] if sign > 0 else df[y <= -cut]
    df["code"] = _sym(df["code"])
    d = _dt(df["ACTUAL_ANN_DATE"])
    d = d.where(d.notna(), _dt(df["ann_date"]))
    df = df.assign(avail=_avail(cal, d)).dropna(subset=["avail"])
    return df[["code", "avail", "strength"]]


def fam_fe_pos(cal, close):
    """业绩快报净利润同比 ≥ +30%。"""
    return _fin_fam(cal, close, +1)


def fam_fe_neg(cal, close):
    """业绩快报净利润同比 ≤ −30%。"""
    return _fin_fam(cal, close, -1)


def _lhb_fam(cal, close, only_inst: bool):
    """龙虎榜聚合。``only_inst=True`` 只留机构专用席位，否则用全部席位。

    两者必须成对跑：龙虎榜上榜本身意味着当日已发生大幅异动（涨跌幅偏离 7%），
    「上榜后均值回归」与「机构买入的信息含量」是两个完全不同的假设，
    只有 ``lhb_all`` 做对照才能把前者剥掉。
    """
    df = pd.read_parquet(PQ / "long_hu_bang.parquet",
                         columns=["code", "trade_date", "TRADER_NAME",
                                  "buy_amount", "sell_amount", "total_amount"])
    if only_inst:
        df = df[df["TRADER_NAME"].astype(str).str.contains("机构专用", na=False)]
    if df.empty:
        return df.iloc[0:0]
    df["code"] = _sym(df["code"])
    df["buy"] = pd.to_numeric(df["buy_amount"], errors="coerce").fillna(0.0)
    df["sell"] = pd.to_numeric(df["sell_amount"], errors="coerce").fillna(0.0)
    df["tot"] = pd.to_numeric(df["total_amount"], errors="coerce")
    d = _dt(df["trade_date"])
    g = (df.assign(d=d)
           .groupby(["code", "d"])
           .agg(buy=("buy", "sum"), sell=("sell", "sum"), tot=("tot", "first"))
           .reset_index())
    g["net"] = g["buy"] - g["sell"]
    g = g[g["tot"] > 0]
    g["strength"] = g["net"] / g["tot"]
    # 全部席位不筛方向（上看榜即全量），机构席位只留净买入
    if only_inst:
        g = g[(g["net"] > 0) & (g["strength"] >= 0.05)]
    nxt = cal.searchsorted(pd.DatetimeIndex(g["d"]).values, side="right")
    keep = nxt < len(cal)
    g = g.loc[keep].assign(avail=cal[nxt[keep]])
    return g[["code", "avail", "strength"]]


def fam_lhb_inst(cal, close):
    """龙虎榜「机构专用」席位**净买入**，且净买入占当日龙虎榜成交 ≥5%。"""
    return _lhb_fam(cal, close, True)


def fam_lhb_all(cal, close):
    """龙虎榜上榜（对照组）：用于剥离「上榜均值回归」这一竞争解释。"""
    return _lhb_fam(cal, close, False)


def _margin_fam(cal, close, up: bool, cut: float = 0.20, win: int = 5):
    """融资余额 5 个交易日累计 ±20%（杠杆资金拥挤 / 撤离）。

    与 `hn_up`（股东户数暴增）同属「资金拥挤度」类，但频率是日频，
    且融资盘是**带强平线的杠杆资金**，理论上的负向冲击应更陡。
    披露口径：融资余额 T 日收盘后公布 ⇒ 可见日 = 下一交易日。
    """
    df = pd.read_parquet(PQ / "margin_detail.parquet",
                         columns=["code", "trade_date", "borrow_money_balance"])
    df["code"] = _sym(df["code"])
    df["bal"] = pd.to_numeric(df["borrow_money_balance"], errors="coerce")
    df = df.dropna(subset=["bal"])
    d = _dt(df["trade_date"])
    df = df.assign(d=d).dropna(subset=["d"]).sort_values(["code", "d"])
    chg = df.groupby("code")["bal"].pct_change(win)
    df = df.assign(strength=chg.abs())
    df = df[chg >= cut] if up else df[chg <= -cut]
    if df.empty:
        return df.iloc[0:0]
    nxt = cal.searchsorted(pd.DatetimeIndex(df["d"]).values, side="right")
    keep = nxt < len(cal)
    df = df.loc[keep].assign(avail=cal[nxt[keep]])
    return df[["code", "avail", "strength"]]


def fam_margin_up(cal, close):
    """融资余额 5 日 +20%（杠杆资金涌入）。"""
    return _margin_fam(cal, close, True)


def fam_margin_dn(cal, close):
    """融资余额 5 日 −20%（杠杆资金撤离 / 去杠杆）。"""
    return _margin_fam(cal, close, False)


FAMILIES = {
    "fc_pos": fam_fc_pos,
    "fc_neg": fam_fc_neg,
    "fe_pos": fam_fe_pos,
    "fe_neg": fam_fe_neg,
    "lhb_inst": fam_lhb_inst,
    "lhb_all": fam_lhb_all,
    "margin_up": fam_margin_up,
    "margin_dn": fam_margin_dn,
    "inc": fam_inc,
    "inc_pure": fam_inc_pure,
    "inc_multi": fam_inc_multi,
    "inc_big": fam_inc_big,
    "red": fam_red,
    "unlock": fam_unlock,
    "blk_disc": fam_blk_disc,
    "blk_prem": fam_blk_prem,
    "hn_up": fam_hn_up,
    "hn_dn": fam_hn_dn,
}


# ============================================================ 建仓区间
def build_positions(ev: pd.DataFrame, cal: pd.DatetimeIndex, hold: int,
                    codes_ok: pd.Index, close: pd.DataFrame | None = None,
                    mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """事件 → 持仓区间 [i0, i1]（日历下标）。

    PIT：``avail`` 是「事件可见的交易日」。若是盘后公告，最早成交在下一交易日
    开盘，故 ``i0 = pos(avail)+1``；若是盘前已知（解禁），仍按 +1 处理，
    这是保守口径（不用任何当日信息）。

    ⚠️ 两道硬闸门（2026-11-15 修）：``tradable_mask`` **不含上市日判断**，
    次新股在上市前的 mask 可能为 True。若不检查价格存在性，会建出「买一只还不存在
    的股票」的仓，其收益为 NaN 被剔除，组合退化成由个别股票主导（实测某日 7 笔
    持仓只剩 1 笔有效，单日收益 ±20%）。故要求：
      1. 建仓日 close 非空；
      2. 整个持有窗 close 非空（缺任一 → 该笔不成立，不是补 0）。

    同股去重：若新事件发生在上一笔持仓的持有窗内，合并为一笔（不叠加仓位）。
    """
    ev = ev[ev["code"].isin(codes_ok)].dropna(subset=["avail"])
    if ev.empty:
        return pd.DataFrame(columns=["code", "i0", "i1"])
    pos = cal.get_indexer(pd.DatetimeIndex(ev["avail"]))
    keep = (pos >= 0) & (pos + 1 + hold < len(cal))
    ev = ev.loc[keep].assign(i0=pos[keep] + 1, i1=pos[keep] + 1 + hold)
    ev = ev.sort_values(["code", "i0"])

    if close is not None:
        cv = close.values
        col = {c: i for i, c in enumerate(close.columns)}
        mv = mask.values if mask is not None else None
        ok_rows = []
        for code, i0, i1 in ev[["code", "i0", "i1"]].itertuples(index=False):
            j = col.get(code)
            if j is None:
                ok_rows.append(False)
                continue
            price_ok = bool(np.isfinite(cv[i0, j]) and np.isfinite(cv[i1, j])
                            and np.isfinite(cv[i0:i1 + 1, j]).all())
            trade_ok = True if mv is None else bool(mv[i0, j])
            ok_rows.append(price_ok and trade_ok)
        ev = ev.loc[np.array(ok_rows)]

    rows = []
    for code, sub in ev.groupby("code", sort=False):
        cur0 = cur1 = None
        for i0, i1 in sub[["i0", "i1"]].itertuples(index=False):
            if cur1 is None or i0 > cur1:      # 不与上一笔重叠 ⇒ 新开一笔
                if cur0 is not None:
                    rows.append((code, int(cur0), int(cur1)))
                cur0, cur1 = i0, i1
            else:                              # 重叠 ⇒ 延长持有（不重复计仓）
                cur1 = max(cur1, i1)
        if cur0 is not None:
            rows.append((code, int(cur0), int(cur1)))
    return pd.DataFrame(rows, columns=["code", "i0", "i1"])


# ============================================================ 日频组合
def run_portfolio(pos: pd.DataFrame, close, openp, mask, cal):
    """事件组合日收益：把每笔持仓摊到它的日区间上，逐日等权。"""
    n = len(cal)
    cv, ov, mv = close.values, openp.values, mask.values
    col = {c: i for i, c in enumerate(close.columns)}

    live = defaultdict(list)      # day -> [(col_idx, phase)]  phase: 0=建仓日, 1=持有, 2=卖出日
    for code, i0, i1 in pos[["code", "i0", "i1"]].itertuples(index=False):
        j = col.get(code)
        if j is None:
            continue
        # 建仓日必须可交易（T+1 口径的 mask），否则这笔根本不成立
        if not bool(mv[i0, j]):
            continue
        live[i0].append((j, 0))
        for i in range(i0 + 1, i1):
            live[i].append((j, 1))
        live[i1].append((j, 2))

    ret = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=int)
    for i in range(1, n):
        arr = live.get(i)
        if not arr:
            continue
        j = np.array([a[0] for a in arr], dtype=int)
        ph = np.array([a[1] for a in arr], dtype=int)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cv[i, j] / cv[i - 1, j] - 1.0                       # 持有
            r0 = cv[i, j] / ov[i, j] - 1.0                          # 建仓日
            r2 = ov[i, j] / cv[i - 1, j] - 1.0                      # 卖出日
        r = np.where(ph == 0, r0, r)
        r = np.where(ph == 2, r2, r)
        # ⚠️ 持有期**不做** mask 过滤：停牌持仓卖不掉，收益应记 0 而不是剔除。
        # 若剔除，等于假设「停牌时可以从组合里消失」，会把组合收益系统性抬高，
        # 且与不过滤的 bench_eqw 不可比（实测口径差 7pp/年）。
        if np.isfinite(r).any():
            ret[i] = float(np.nanmean(r))
            cnt[i] = int(np.isfinite(r).sum())
    out = pd.Series(ret, index=cal).reindex(cal)
    return out, pd.Series(cnt, index=cal)


# ============================================================ 市值匹配基准
def matched_bench(close: pd.DataFrame, mcap: pd.DataFrame, ret: pd.Series | None = None,
                  nq: int = NQ) -> pd.DataFrame:
    """每日按市值分 nq 组，返回各组等权日收益（index=日期, columns=组号）。

    向量化实现：按行做 pct rank → 分箱 → long 表 groupby。
    逐日 qcut 要 3000 次，慢一个量级，且和这里结果等价。
    """
    q = mcap.where(mcap.notna()).rank(axis=1, pct=True)
    g = (q * nq).clip(upper=nq - 1e-9).apply(np.floor)
    r1 = close.pct_change()
    long = pd.DataFrame({"g": g.stack(), "r": r1.stack()}).dropna()
    if long.empty:
        return pd.DataFrame()
    d = long.index.get_level_values(0)
    out = long.groupby([d, "g"])["r"].mean().unstack()
    out.columns = [int(c) for c in out.columns]
    return out.reindex(columns=range(nq)).sort_index()


_GRP_CACHE: dict[int, pd.DataFrame] = {}


def _grp_table(close: pd.DataFrame, mcap: pd.DataFrame, nq: int = NQ) -> pd.DataFrame:
    key = id(close)
    if key not in _GRP_CACHE:
        _GRP_CACHE[key] = matched_bench(close, mcap, None, nq)
    return _GRP_CACHE[key]


def bench_for_holdings(pos: pd.DataFrame, close: pd.DataFrame, mcap: pd.DataFrame,
                       nq: int = NQ) -> pd.Series:
    """事件股所在市值组的等权收益（每日按实际持仓的组别等权）。

    分箱用**前一日**市值，避免用当日信息决定基准分组。
    """
    grp = _grp_table(close, mcap, nq)
    if grp.empty:
        return pd.Series(np.nan, index=close.index)
    n = len(close)
    col = {c: i for i, c in enumerate(close.columns)}

    live = defaultdict(list)
    for code, i0, i1 in pos[["code", "i0", "i1"]].itertuples(index=False):
        j = col.get(code)
        if j is None:
            continue
        for i in range(i0, i1 + 1):
            live[i].append(j)

    out = np.full(n, np.nan)
    mcap_v = mcap.values
    qs = mcap.where(mcap.notna()).rank(axis=1, pct=True).values
    for i in range(1, n):
        js = live.get(i)
        if not js:
            continue
        d = close.index[i]
        if d not in grp.index:
            continue
        j = np.array(js, dtype=int)
        q = qs[i - 1, j]
        ok = np.isfinite(q)
        if not ok.any():
            continue
        g = np.clip(np.floor(q[ok] * nq), 0, nq - 1).astype(int)
        v = grp.loc[d].reindex(g).values
        if np.isfinite(v).any():
            out[i] = float(np.nanmean(v))
    return pd.Series(out, index=close.index)


# ============================================================ 统计
def nw_t(x: np.ndarray, lag: int = 5) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    e = x - x.mean()
    S = (e * e).sum() / n
    for l in range(1, lag + 1):
        S += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).sum() / n
    se = np.sqrt(S / n)
    m = float(x.mean())
    return (m, float(m / se)) if se > 0 else (m, np.nan)


def summarize(ret: pd.Series, pos: pd.DataFrame, cal: pd.DatetimeIndex,
              hold: int, bench: pd.Series, cnt: pd.Series | None = None) -> dict:
    """年化 / 超额 / Sharpe / 分年 / 换手 / 扣费。

    ⚠️ 空仓日必须记 0，不能 dropna。事件驱动是「有事件才持仓」，若只对持仓日
    求均值再 ×250，等于假设「没事件的日子也在赚钱」，会系统性高估（实测差 6pp/年）。
    """
    r = ret.reindex(cal).fillna(0.0)
    b = bench.reindex(cal).fillna(0.0)
    first = ret.first_valid_index()
    if first is None:
        return {}
    r, b = r.loc[first:], b.loc[first:]
    if len(r) < 250:
        return {}
    days = len(r)
    avg_h = float(cnt[cnt > 0].mean()) if cnt is not None and (cnt > 0).any() else 1.0
    # 年化换手（双边）：每日新建仓笔数 / 平均持仓笔数 = 日单边换手率
    ann_turn = float(len(pos) / days / max(avg_h, 1.0) * 2 * 250)
    cost = ann_turn * COST_ONE_WAY

    ex = r - b
    m, t = nw_t(ex.values)
    ann_ex = m * 250
    ann_r = float(r.mean() * 250)
    sharpe = float(r.mean() / r.std() * np.sqrt(250)) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    cover = float((r != 0).mean())

    yr = ex.groupby(ex.index.year).mean() * 250
    pos_yrs = int((yr > 0).sum())
    return {
        "days": days,
        "ann_ret": ann_r,
        "ann_ex": ann_ex,
        "ann_ex_net": ann_ex - cost,
        "t": t,
        "sharpe": sharpe,
        "maxdd": dd,
        "turnover": ann_turn,
        "cost": cost,
        "yr_pos": f"{pos_yrs}/{len(yr)}",
        "avg_hold": avg_h,
        "cover": cover,
        "n_pos": len(pos),
    }


def car_curve(ev: pd.DataFrame, close: pd.DataFrame, openp: pd.DataFrame,
              cal: pd.DatetimeIndex, pre: int = 5, post: int = 60) -> pd.Series:
    """事件窗前 pre ~ 后 post 的等权累积异常收益（基准 = 全A等权）。"""
    col = {c: i for i, c in enumerate(close.columns)}
    cv, ov = close.values, openp.values
    br = close.pct_change().mean(axis=1).values      # 全A等权日收益

    pos = cal.get_indexer(pd.DatetimeIndex(ev["avail"]))
    ok = (pos - pre >= 1) & (pos + post < len(cal))
    pos = pos[ok]
    js = np.array([col.get(c, -1) for c in ev["code"].values[ok]], dtype=int)
    good = js >= 0
    pos, js = pos[good], js[good]
    if len(pos) == 0:
        return pd.Series(dtype=float)

    rel = np.zeros((len(pos), pre + post + 1))
    for a, p in enumerate(pos):
        j = js[a]
        # t=0 从 avail 当日 open 建仓（保守）
        for k in range(-pre, post + 1):
            i = p + k
            if k < 0:
                rr = cv[i, j] / cv[i - 1, j] - 1.0 if i >= 1 else 0.0
            elif k == 0:
                rr = cv[i, j] / ov[i, j] - 1.0
            else:
                rr = cv[i, j] / cv[i - 1, j] - 1.0
            rel[a, k + pre] = (rr if np.isfinite(rr) else 0.0) - (br[i] if np.isfinite(br[i]) else 0.0)
    return pd.Series(np.cumsum(np.nanmean(rel, axis=0)) * 100,
                     index=range(-pre, post + 1))


# ============================================================ 安慰剂
def placebo_dates(ev: pd.DataFrame, cal: pd.DatetimeIndex, rng) -> pd.DataFrame:
    """事件日随机平移：保留 code 与事件密度，把 avail 换成同年内的随机交易日。"""
    yr = pd.DatetimeIndex(ev["avail"]).year
    out = []
    years = sorted(set(yr))
    cal_by_year = {y: cal[cal.year == y] for y in years}
    for y in years:
        sub = ev[yr == y]
        pool = cal_by_year[y]
        if len(pool) == 0:
            continue
        idx = rng.integers(0, len(pool), size=len(sub))
        out.append(sub.assign(avail=pool[idx]))
    return pd.concat(out, ignore_index=True) if out else ev.iloc[0:0]


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的事件族子集")
    ap.add_argument("--hold", default="20", help="持有期（交易日），逗号分隔")
    ap.add_argument("--placebo", type=int, default=0, help="安慰剂重复次数")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    close, openp, mask, bench, mcap = load_base()
    cal = pd.DatetimeIndex(close.index)
    print(f"[base] {cal[0].date()} ~ {cal[-1].date()}  股票={close.shape[1]}")

    holds = [int(x) for x in args.hold.split(",") if x]
    names = [n for n in (args.only.split(",") if args.only else FAMILIES) if n]
    rng = np.random.default_rng(20261115)

    rows, curves = [], {}
    for name in names:
        fn = FAMILIES.get(name)
        if fn is None:
            print(f"[skip] 未知族 {name}")
            continue
        try:
            ev = fn(cal, close)
        except Exception as exc:  # noqa: BLE001
            print(f"[err ] {name}: {type(exc).__name__}: {exc}")
            continue
        if ev.empty:
            print(f"[skip] {name}: 无事件")
            continue
        ev = ev.drop_duplicates(subset=["code", "avail"])

        for hold in holds:
            pos = build_positions(ev, cal, hold, close.columns, close, mask)
            if pos.empty:
                continue
            ret, cnt = run_portfolio(pos, close, openp, mask, cal)
            mbench = bench_for_holdings(pos, close, mcap)
            s = summarize(ret, pos, cal, hold, mbench, cnt)
            if not s:
                continue
            s.update(family=name, hold=hold, n_events=len(ev),
                     avg_names=float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 0.0)

            if args.placebo:
                pl = []
                for _ in range(args.placebo):
                    pev = placebo_dates(ev, cal, rng)
                    ppos = build_positions(pev, cal, hold, close.columns, close, mask)
                    if ppos.empty:
                        continue
                    pret, _ = run_portfolio(ppos, close, openp, mask, cal)
                    pb = bench_for_holdings(ppos, close, mcap)
                    rr = pret.reindex(cal).fillna(0.0)
                    bb = pb.reindex(cal).fillna(0.0)
                    pex = (rr - bb).dropna()
                    if len(pex) > 250:
                        pl.append(float(pex.mean() * 250))
                if pl:
                    s["plc_mean"] = float(np.mean(pl))
                    s["plc_p"] = float(np.mean(np.array(pl) >= s["ann_ex"]))
                    s["edge"] = s["ann_ex"] - s["plc_mean"]
                else:
                    s["plc_mean"] = s["plc_p"] = s["edge"] = np.nan

            rows.append(s)
            # 存逐日序列，供报告里画分年表 / 净值图（不进 summary.csv，太长）
            pd.DataFrame({"ret": ret, "bench": mbench, "n": cnt}).to_parquet(
                OUT / f"daily_{name}_H{hold}.parquet")
            if hold == holds[0]:
                curves[name] = car_curve(ev, close, openp, cal)
            print(f"[ok  ] {name:<9} H={hold:<3} n={len(ev):>6} 持仓={s['avg_names']:>5.0f} "
                  f"年化={s['ann_ret']:+.2%} 超额={s['ann_ex']:+.2%} "
                  f"(扣费{s['ann_ex_net']:+.2%}) t={s['t']:+.2f} "
                  f"Sharpe={s['sharpe']:.2f} 分年={s['yr_pos']} 换手={s['turnover']:.1f}x"
                  + (f" 安慰剂={s['plc_mean']:+.2%}(p={s['plc_p']:.2f}) edge={s['edge']:+.2%}"
                     if args.placebo else ""))

    if not rows:
        print("无有效结果")
        return
    out = pd.DataFrame(rows)
    num = ["ann_ret", "ann_ex", "ann_ex_net", "plc_mean", "edge"]
    for c in num:
        if c in out:
            out[c] = out[c].astype(float)
    out = out.sort_values("ann_ex_net", ascending=False)
    out.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    # CAR 曲线另存（宽表：index=相对日, columns=族）
    if curves:
        cc = pd.DataFrame(curves)
        cc.index.name = "rel_day"
        cc.to_csv(OUT / "car_curves.csv", encoding="utf-8-sig")

    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\n[out] {OUT/'summary.csv'}")


if __name__ == "__main__":
    main()
