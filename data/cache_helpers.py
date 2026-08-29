"""
离线/在线通用的缓存读取工具
============================

把散落在 scripts/ 各入口里的重复"从缓存加载数据"逻辑收敛到这里：

1. ``load_daily``            日历校验 + 成分股 + 日线（4 处 load_data 的公共开头）
2. ``load_backward_factor``  后复权因子（离线读 parquet / 在线走 cache，双分支）
3. ``load_financial_tables`` 财务/股东事件表（同上双分支，7 张表统一）
4. ``returns_from_cache``    从 daily_{pool}.parquet 构造次日收益面板（因子库 IC 口径）
5. ``load_index_returns``    指数日收益（parquet 优先 / real 回源 / 可选日历对齐）

约定：offline 判定依据 cache._ds 是否为 data.offline.OfflineDataSource——
离线时直接读本地 parquet（避免整表覆盖型接口回调数据源抛错），在线时走
cache 接口。2026-08-05 统一（此前 select_stocks/synthesize_library 逐字
重复 returns_from_cache；build_*/intraday_analysis 各自实现 load_data；
walk_forward/backtest_two_periods/build_fundamental_factors 各自实现
_fin 离线读财务表）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import Config
from data.offline import OfflineDataSource

_log = logging.getLogger("cache_helpers")


def _is_offline(cache) -> bool:
    return isinstance(getattr(cache, "_ds", None), OfflineDataSource)


def load_index_returns(
    index_code: str,
    begin: int | None = None,
    end: int | None = None,
    real: bool = False,
    reindex_to: pd.DatetimeIndex | None = None,
) -> pd.Series | None:
    """指数日收益（单一实现；2026-08-29 收敛自 investment_report / run_model_portfolio 两份平行实现）。

    优先读本地缓存 ``index_daily_{code}.parquet``；缺失且 real=True 时回源拉取；
    否则返回 None（调用方自行降级基准，如全池等权）。

    Args:
        index_code: 指数代码（如 ``000300.SH``）。
        begin/end: 8 位日期。begin 前多留 10 个自然日窗口再 pct_change，
            保证 begin 首日是真实收益而非 NaN。
        real: 无缓存时是否回源拉取（需要数据源凭证）。
        reindex_to: 精确对齐的日历（如回测 test_days；缺失日 ffill 前值收益）。
    Returns:
        Series(index=date, name=index_code) 日收益；不可用时 None。
    """
    safe = str(index_code).replace(".", "_")
    p = Path(str(Config.cache()["root"])) / f"index_daily_{safe}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    elif real:
        from data import get_cache  # 懒导入避免与 data.__init__ 循环
        try:
            _log.info("拉取指数数据 %s ...", index_code)
            df = get_cache().get_index_daily(
                index_code, begin or 19900101, end or 99991231)
        except Exception as e:  # noqa: BLE001 - 拉取失败降级为无基准
            _log.warning("指数 %s 拉取失败: %s", index_code, e)
            return None
    else:
        _log.warning("指数 %s 无缓存（mock 模式不可拉取）", index_code)
        return None

    if "code" in getattr(df.index, "names", ()):
        close = df.xs(str(index_code), level="code")["close"].sort_index()
    elif "code" in df.columns:
        close = (df.reset_index()
                 .pivot(index="date", columns="code", values="close")
                 .sort_index().iloc[:, 0])
    else:
        close = df["close"].sort_index()

    if begin is not None:
        close = close[close.index >= pd.Timestamp(str(begin)) - pd.Timedelta(days=10)]
    if end is not None:
        close = close[close.index <= pd.Timestamp(str(end))]

    ret = close.pct_change(fill_method=None)
    if begin is not None:
        ret = ret[ret.index >= pd.Timestamp(str(begin))]
    if reindex_to is not None:
        ret = ret.reindex(reindex_to).ffill()
    ret.name = index_code
    return ret


def load_daily(cache, uni, index_code: str, begin: int, end: int | None,
               pool: str | None = None):
    """日历校验 + 成分股 + 日K线。

    **PIT 口径（2026-08-13 统一）**：股票池取 ``begin~end`` 区间**历史在册
    成分并集**（含期间调出/退市股），并用 ``get_membership_mask`` 按交易日
    将非在册期间的行情置 NaN——等价于「每个交易日只用当时在册成分」，
    消除「当前成分回看」的幸存者偏差（旧口径 target=end 用未来成分选历史池）。

    pool: 股票池（hs300/zz500/zz1000/all_a），默认取 config.universe.default，
        决定读取 daily_{pool}.parquet（2026-08-26 池隔离扩展）。

    Returns:
        (codes, cal, daily): 股票池 / 交易日列表(YYYYMMDD int) / (date, code)
        长表（非在册期间行情已置 NaN）。
    """
    cal = cache.get_calendar(begin, end)
    if not cal:
        raise RuntimeError(f"交易日历为空（{begin}-{end}），请先更新数据")
    target = end or cal[-1]
    if pool == "all_a":
        codes = uni.get_all_a(target)
    else:
        codes = _pit_universe_codes(uni, index_code, begin, target)
    daily = _read_daily_offline(cache, codes, begin, target, pool) if _is_offline(cache) \
        else cache.get_daily_kline(codes, begin, target, pool=pool)
    if pool != "all_a":
        daily = _apply_membership_mask(daily, uni, index_code)
    return codes, cal, daily


def _read_daily_offline(cache, codes, begin: int, end: int,
                        pool: str | None = None):
    """离线读 daily_{pool}.parquet（本地可能只覆盖部分并集池代码，取交集即可）。

    不走 _refresh_long_table：其"新代码触发全量回源"逻辑（cache.py 229 行）
    在 offline 桩下会抛错；离线时本地 parquet 就是数据全集。
    """
    import pandas as pd
    from config import Config

    pool = pool or Config.universe().get("default", "hs300")
    p = Path(cache.root) / f"daily_{pool}.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    if d.empty:
        return d
    has = set(d.index.get_level_values("code").unique())
    codes_avail = [c for c in codes if c in has]
    d = d[d.index.get_level_values("code").isin(codes_avail)]
    ts = d.index.get_level_values(0)
    mask = (ts >= pd.Timestamp(str(begin))) & (ts <= pd.Timestamp(str(end)))
    return d[mask]


def _pit_universe_codes(uni, index_code: str, begin: int, end: int) -> list[str]:
    """区间历史在册成分并集池（in_date <= end 且 (out_date 空 或 out_date > begin)）。"""
    import pandas as pd

    cons = uni.get_all_constituents(index_code).copy()
    if cons.empty:
        return []
    cons["in_date"] = pd.to_datetime(cons["in_date"], errors="coerce")
    cons["out_date"] = pd.to_datetime(cons["out_date"], errors="coerce")
    b0, e0 = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    m = (cons["in_date"] <= e0) & (cons["out_date"].isna() | (cons["out_date"] > b0))
    return cons.loc[m, "con_code"].dropna().unique().tolist()


def _apply_membership_mask(daily: pd.DataFrame, uni, index_code: str) -> pd.DataFrame:
    """把 (date, code) 长表按 PIT 成分归属置 NaN（非在册期间的行情不可用）。

    实现：长表 pivot 成宽表 → 与 mask 对齐按元素 where → 还原为长表。
    """
    if daily.empty or daily.index.nlevels < 2:
        return daily
    codes = daily.index.get_level_values(1).unique().tolist()
    dates = pd.to_datetime(daily.index.get_level_values(0)).unique()
    mask = uni.get_membership_mask(index_code, pd.DatetimeIndex(dates))
    mask = mask.reindex(index=pd.DatetimeIndex(dates),
                        columns=[c for c in codes if c in mask.columns])

    value_cols = daily.columns.tolist()
    wide = daily.reset_index().pivot(index="date", columns="code", values=value_cols)
    # wide 是 MultiIndex columns (col, code)；逐列 where(mask) 后 stack 还原长表
    parts = []
    for col in value_cols:
        w = wide[col].reindex(index=mask.index, columns=mask.columns)
        parts.append(w.where(mask).stack().rename(col))
    masked = pd.concat(parts, axis=1)
    # stack 后 index 为 (date, code) MultiIndex，命名与原始一致
    masked.index = masked.index.set_names(["date", "code"])
    return masked.sort_index()


def load_backward_factor(cache, codes) -> pd.DataFrame:
    """累积后复权因子（date×code 宽表）。

    复权因子是"宽表全量刷新"型接口（_refresh_wide_table 每次回调数据源），
    offline 桩会报错 → 离线时直接读本地 parquet 并按股票池过滤。
    """
    if _is_offline(cache):
        p = Path(cache.root) / "backward_factor.parquet"
        if not p.exists():
            return pd.DataFrame()
        bf = pd.read_parquet(p)
        return bf[[c for c in codes if c in bf.columns]]
    return cache.get_backward_factor(codes)


def build_tradable_mask(close_adj: pd.DataFrame,
                        bwd: pd.DataFrame | None = None,
                        cache_root: str | None = None) -> pd.DataFrame:
    """构建**可交易性掩码**（bool 宽表，True=该日信号可进入组合）。

    信号 T 日收盘产生、T+1 日 VWAP 成交（国君研报执行链口径）。剔除规则
    （对齐华泰单因子测试口径 + 涨跌停一字板可成交性）：

    - T+1 日**停牌**（is_suspended）：无法成交 → 剔除；
    - T+1 日**收盘封涨停/封跌停**（raw_close ≥ high_limited / ≤ low_limited）：
      VWAP 处买不进/平不掉 → 剔除（用收盘封板而非盘中触板，保守适中）;
    - is_st / is_suspended 在 T 日或 T+1 日任一为真：ST 与停牌是持续状态，
      成交日（T+1）处于该状态即不可交易。

    封板判定需要**未复权收盘价**与状态表中的原始涨跌停价比较：
    ``raw_close = close_adj / bwd``（bwd=累积后复权因子，None 时退化为
    close_adj 本身——仅当面板未复权或无除权差异时可用，真实复权数据必须传）。

    状态表缺失的 (date, code) 视为可交易（保守补 True）；返回前 reindex 到
    ``close_adj`` 的行列。状态表不存在时返回全 True 掩码（调用方日志可见）。
    """
    root = cache_root or str(Config.cache()["root"])
    p = Path(root) / "history_stock_status.parquet"
    idx, cols = close_adj.index, close_adj.columns
    if not p.exists():
        return pd.DataFrame(True, index=idx, columns=cols)

    st = pd.read_parquet(p)
    st = st[st.index.get_level_values("code").isin(cols)]
    if st.empty:
        return pd.DataFrame(True, index=idx, columns=cols)

    def _wide(col: str) -> pd.DataFrame:
        if col not in st.columns:
            return pd.DataFrame(False, index=idx, columns=cols)
        return st[col].unstack("code").reindex(index=idx, columns=cols)

    susp = _wide("is_suspended").fillna(False)
    is_st = _wide("is_st").fillna(False)
    hi_lim = _wide("high_limited")
    lo_lim = _wide("low_limited")

    # T+1 状态（封板/停牌用次日信号对齐：T 日信号 → T+1 成交）
    susp_next = susp.shift(-1).fillna(False)
    hi_next = hi_lim.shift(-1)
    lo_next = lo_lim.shift(-1)

    if bwd is not None and len(bwd):
        bwd_al = bwd.reindex(index=idx, columns=cols).ffill()
        raw_next = (close_adj / bwd_al).shift(-1)
    else:
        raw_next = close_adj.shift(-1)

    tol = 1e-6
    sealed_up = (raw_next >= hi_next * (1 - tol)) & hi_next.notna()
    sealed_dn = (raw_next <= lo_next * (1 + tol)) & lo_next.notna()

    st_next = is_st.shift(-1).fillna(False)
    bad = susp_next | sealed_up | sealed_dn | st_next | susp | is_st
    mask = ~bad
    return mask.fillna(True).astype(bool)


# 财务/股东事件表清单（文件名 == cache 接口名）
_FINANCIAL_TABLES = (
    "income", "balance_sheet", "cash_flow", "equity_structure",
    "dividend", "share_holder", "holder_num",
)


def load_financial_tables(cache, codes) -> dict[str, pd.DataFrame]:
    """财务/股东事件表（稀疏长表）：{表名: DataFrame}。

    财务表缓存是"整表覆盖"模式（每次调用都回调数据源），offline 桩会报错；
    离线时直接读本地 parquet 并按股票池过滤。
    """
    if _is_offline(cache):
        out: dict[str, pd.DataFrame] = {}
        for name in _FINANCIAL_TABLES:
            p = Path(cache.root) / f"{name}.parquet"
            df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
            out[name] = df[df["code"].isin(codes)] if "code" in df.columns else df
        return out
    getters = {
        "income": cache.get_income,
        "balance_sheet": cache.get_balance_sheet,
        "cash_flow": cache.get_cash_flow,
        "equity_structure": cache.get_equity_structure,
        "dividend": cache.get_dividend,
        "share_holder": cache.get_share_holder,
        "holder_num": cache.get_holder_num,
    }
    return {k: g(codes) for k, g in getters.items()}


def returns_from_cache(cache, begin: int, end: int,
                       pool: str | None = None) -> pd.DataFrame:
    """从日线缓存构造次日收益面板（与因子库 IC 口径一致）。"""
    from config import Config
    pool = pool or Config.universe().get("default", "hs300")
    cal = cache.get_calendar(begin, end)
    if not cal:
        raise RuntimeError("交易日历为空")
    d = pd.read_parquet(Path(cache.root) / f"daily_{pool}.parquet")
    close_w = d.reset_index().pivot(index="date", columns="code", values="close").sort_index()
    close_w = close_w.loc[str(begin): str(end)]
    return close_w.pct_change().shift(-1)


# 财务字段（PIT 展开）与市值所需字段
_INCOME_FIELDS = ("OPERA_REV", "NET_PRO_INCL_MIN_INT_INC", "BASIC_EPS")
_BALANCE_FIELDS = ("TOTAL_ASSETS", "TOT_SHARE_EQUITY_EXCL_MIN_INT")


def _with_retry(fn, *args, retries: int = 3, wait: int = 20, **kw):
    """数据源调用重试（TGW 连接不稳定 / 并发超限时自动重连等待）。"""
    import time
    for i in range(retries):
        try:
            return fn(*args, **kw)
        except Exception:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  数据拉取失败，{wait}s 后重试 {i + 1}/{retries}", flush=True)
            time.sleep(wait)


def build_panel(
    cfg: dict,
    begin: int,
    end: int,
    *,
    cache=None,
    cache_root: str | None = None,
    sdk_cache: str | None = None,
    offline: bool = False,
    include_market_cap: bool = False,
    retry: bool = False,
):
    """统一真实面板构建（2026-08-17 收敛 mine_factors / run_gflownet_phase1 的重复实现）。

    覆盖两处原有 build_real_panel 的全部能力，口径一致：
    - **PIT 并集池 + membership mask**（历史在册成分，消除幸存者偏差）；
    - 后复权日线（close/high/low/open/volume/amount，mask 后复权）；
    - 财务字段 PIT 展开（公告日对齐，无未来函数）；
    - 可选市值面板（``TOT_SHARE``(PIT) × 后复权 close，研报市值中性化用）。

    Args:
        cfg: 至少含 ``{"universe": {"index_code", "adjust"}}``。
        cache: 可选注入的 DataCache（测试用）；为 None 时按下方选项创建。
        cache_root / sdk_cache / offline: 数据源与缓存选项（对应 gflownet 的参数）。
        include_market_cap: 是否构建市值。为 True 时 panel 额外含
            ``close_m``(masked close) / ``market_cap`` / ``mask``。
        retry: 是否对数据源拉取做重试（TGW 不稳定场景，对应 gflownet）。

    Returns:
        (panel, returns_panel)：panel 为 dict{字段: 宽表}，returns 为
        ``close.pct_change().shift(-1)``（与因子库 IC 口径一致）。
    """
    import pandas as pd

    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.financials import build_pit_panel
    from data.universe import Universe

    index_code = cfg["universe"]["index_code"]
    # 股票池从本函数的 cfg 读取并显式传给 load_daily——load_daily 内部回退
    # 读全局 Config，曾导致 --universe/all_a 覆盖静默失效（2026-08-28 修复）。
    pool = cfg["universe"].get("default", "hs300")

    # 1) 数据源 + 缓存 + 股票池（含离线桩 / SDK 缓存路径）
    if cache is None:
        from config import Config
        if offline:
            from data.offline import OfflineQuietDataSource
            ds = OfflineQuietDataSource()
            print("离线模式：仅用本地 parquet 缓存构建面板")
        else:
            _cfg = Config.datasource()
            if sdk_cache:
                _cfg["amazing_data"]["sdk_local_path"] = sdk_cache
            ds = create_datasource(_cfg)
        cache = DataCache(ds, cache_root=cache_root) if cache_root else DataCache(ds)
    uni = Universe(cache)

    # 2) 区间历史在册并集池 + mask 日线（load_daily 内部已完成 mask）
    cal = cache.get_calendar(begin, end)
    target_date = end if end else cal[-1]
    # load_daily 内部已应用 membership mask，非在册行情为 NaN，无需重复处理。
    if retry:
        codes, cal, daily = _with_retry(load_daily, cache, uni, index_code, begin, end, pool=pool)
    else:
        codes, cal, daily = load_daily(cache, uni, index_code, begin, end, pool=pool)

    # 3) 宽表 + 后复权（mask 已体现在长表，unstack 后非在册自然为 NaN）
    close = daily["close"].unstack("code")
    high = daily["high"].unstack("code")
    low = daily["low"].unstack("code")
    open_ = daily["open"].unstack("code")
    volume = daily["volume"].unstack("code")
    amount = daily["amount"].unstack("code")
    backward = load_backward_factor(cache, codes)
    backward = backward.reindex(index=close.index, columns=close.columns).ffill()
    close = close * backward
    high = high * backward
    low = low * backward
    open_ = open_ * backward

    panel = {
        "close": close, "open": open_, "high": high, "low": low,
        "volume": volume, "amount": amount,
    }

    # 4) 财务字段 PIT 展开（公告日对齐，无未来函数）
    fin = load_financial_tables(cache, codes)
    income = fin.get("income", pd.DataFrame())
    balance = fin.get("balance_sheet", pd.DataFrame())
    if not income.empty:
        for field in _INCOME_FIELDS:
            if field in income.columns:
                panel[field] = build_pit_panel(income, cal, field).reindex(close.index)
    if not balance.empty:
        for field in _BALANCE_FIELDS:
            if field in balance.columns:
                panel[field] = build_pit_panel(balance, cal, field).reindex(close.index)

    # 5) 市值面板（研报市值中性化奖励用）
    if include_market_cap:
        mask = uni.get_membership_mask(index_code, close.index).reindex(columns=codes)
        close_m = close.where(mask)
        if balance.empty or "TOT_SHARE" not in balance.columns:
            raise RuntimeError("资产负债表 TOT_SHARE 缺失，无法构建市值面板")
        tot_share = build_pit_panel(balance, cal, "TOT_SHARE").reindex(
            index=close.index, columns=close.columns)
        panel["close_m"] = close_m
        panel["market_cap"] = (tot_share * close_m).where(mask)
        panel["mask"] = mask

    returns_panel = close.pct_change().shift(-1)
    return panel, returns_panel
