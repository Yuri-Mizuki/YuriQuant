"""中金所股指期货日线通道 + 连续合约/基差构造（纯择时线，2026-11-23）。

只做四件事，供基差探针复用：

1. 从 SDK ``get_hist_future_code_list('ZJ_FUTURE', ...)`` 取历史合约，筛 IF/IH/IC/IM；
2. 日线 parquet 缓存（长表 ``(date, code)``，只增量补缺失合约与尾部日期）；
3. 由合约代码 ``YYMM`` 推到期日（CFFEX 规则：合约月第三个周五）；
4. 产出每品种基差面板：近月 raw/年化基差、主连基差、近远月价差。

口径
----
- 期货收盘与指数收盘同为 15:00，``basis = fut_close / spot_close - 1``，
  t 日收盘即可知（PIT 无前视）。
- 年化基差 = ``basis * 365 / days_to_expiry``；到期日按第三个周五近似，
  真实最后交易日遇节假日顺延，误差 ≤1 天，只进分母、不影响方向研究。
- 主连 = 近月到期前 ``roll_days`` 个自然日切次近月（默认 5），避免到期跳空。
- 只覆盖中金所股指期货（``ZJ_FUTURE``），不含商品期货与国债期货。
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

#: 品种 → 现货指数。IH 标的为上证 50（000016.SH）。
CFFEX_INDEX = {"IF": "000300.SH", "IH": "000016.SH",
               "IC": "000905.SH", "IM": "000852.SH"}

DEFAULT_CACHE = Path(str(Config.cache()["root"]).replace("//", "/")) / "futures_daily.parquet"

_CODE_RE = re.compile(r"^([A-Z]+)(\d{4})\.")


def third_friday(year: int, month: int) -> pd.Timestamp:
    """合约月第三个周五（CFFEX 股指期货最后交易日，节假日顺延不在此修正）。"""
    first = pd.Timestamp(year=year, month=month, day=1)
    first_fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
    return first_fri + pd.Timedelta(days=14)


def parse_contract(code: str) -> tuple[str, pd.Timestamp]:
    """``IF2706.CFE`` → (``IF``, 2027-06-18)。无法解析时抛 ValueError。"""
    m = _CODE_RE.match(code)
    if not m:
        raise ValueError(f"无法解析期货合约代码: {code}")
    prod, yymm = m.group(1), m.group(2)
    return prod, third_friday(2000 + int(yymm[:2]), int(yymm[2:]))


def fetch_contracts(begin: int, end: int, products: tuple[str, ...] = tuple(CFFEX_INDEX),
                    ds=None) -> list[str]:
    """区间内存在过的中金所股指期货合约（含已到期），按代码排序。"""
    if ds is None:
        from data.datasource import create_datasource
        ds = create_datasource()
    codes = ds.get_hist_future_code_list("ZJ_FUTURE", begin, end)
    return sorted(c for c in codes if c[:2] in products)


def load_futures_daily(codes: list[str], begin: int, end: int,
                       cache_path: Path | str = DEFAULT_CACHE, refresh: bool = False,
                       ds=None, batch: int = 20) -> pd.DataFrame:
    """期货日线长表（MultiIndex (date, code)，列 OHLCV/amount），带增量 parquet 缓存。

    只补「缓存里没有的合约」整段 + 「缓存尾部之后的日期」；``refresh=True`` 全量重拉。
    """
    cache_path = Path(cache_path)
    cached = pd.DataFrame()
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        if cached.index.nlevels != 2:
            cached = pd.DataFrame()

    have = set(cached.index.get_level_values("code")) if len(cached) else set()
    last = cached.index.get_level_values("date").max() if len(cached) else None

    todo: list[str] = [c for c in codes if c not in have]
    tail_begin = None
    if last is not None and pd.Timestamp(last) < pd.Timestamp(str(end)) - pd.Timedelta(days=1):
        tail_begin = int((pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y%m%d"))

    if not todo and tail_begin is None:
        return cached.loc[(slice(pd.Timestamp(str(begin)), pd.Timestamp(str(end))), slice(None)), :]

    if ds is None:
        from data.datasource import create_datasource
        ds = create_datasource()

    frames = []
    if todo:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            frames.append(ds.get_daily_kline(chunk, begin, end))
    if tail_begin is not None:
        frames.append(ds.get_daily_kline(list(codes), tail_begin, end))

    fresh = pd.concat([f for f in frames if len(f)], axis=0) if any(len(f) for f in frames) \
        else pd.DataFrame()
    combined = pd.concat([cached, fresh], axis=0) if len(cached) else fresh
    if len(combined):
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path)
    return combined.loc[(slice(pd.Timestamp(str(begin)), pd.Timestamp(str(end))), slice(None)), :]


def build_basis(fut: pd.DataFrame, product: str, spot: pd.Series,
                roll_days: int = 5) -> pd.DataFrame:
    """单品种基差面板（index=交易日，已与 ``spot`` 对齐）。

    列：
      front / next / main          近月、次近月、主连合约代码
      front_close / next_close     收盘价
      days_to_exp                  近月剩余自然日
      basis_front / ann_basis_front 近月 raw / 年化基差
      basis_main / ann_basis_main   主连 raw / 年化基差
      term_spread                  次近月 / 近月 - 1（近远月价差）
    """
    codes = sorted({c for c in fut.index.get_level_values("code") if c.startswith(product)})
    if spot.index.nlevels > 1:  # 兼容 (date, code) 多级索引的共享缓存
        spot = spot.xs(spot.index.get_level_values(1)[0], level=1)
    if not codes:
        return pd.DataFrame(index=spot.index)
    close = fut["close"].unstack("code").reindex(columns=codes)
    exp = pd.Series({c: parse_contract(c)[1] for c in codes})

    rows = []
    for t, px in close.iterrows():
        px = px.dropna()
        if px.empty:
            continue
        elig = exp[exp.index.isin(px.index) & (exp >= t)].sort_values(kind="stable")
        if elig.empty:
            continue
        f = elig.index[0]
        d_f = int((exp[f] - t).days)
        n = elig.index[1] if len(elig) > 1 else None
        m = f if (d_f > roll_days or n is None) else n
        rows.append({
            "date": t, "front": f, "front_close": float(px[f]),
            "days_to_exp": d_f,
            "next": n, "next_close": float(px[n]) if n else np.nan,
            "main": m, "main_close": float(px[m]),
            "main_days": int((exp[m] - t).days),
        })
    panel = pd.DataFrame(rows).set_index("date").reindex(spot.index)
    s = spot.reindex(panel.index)
    panel["basis_front"] = panel["front_close"] / s - 1.0
    panel["ann_basis_front"] = panel["basis_front"] * 365.0 / panel["days_to_exp"].clip(lower=1)
    panel["basis_main"] = panel["main_close"] / s - 1.0
    panel["ann_basis_main"] = panel["basis_main"] * 365.0 / panel["main_days"].clip(lower=1)
    panel["term_spread"] = panel["next_close"] / panel["front_close"] - 1.0
    return panel


def load_spot(code: str, begin: int = 20150101, cache_dir: Path | None = None,
              refresh: bool = False, ds=None) -> pd.DataFrame:
    """指数日 K：共享缓存 → 实验本地缓存 → 回源，均带 5 自然日新鲜度闸门。

    共享缓存是 ``<cache_root>/index_daily_<code>.parquet``（可能只有部分指数）；
    回源走调用方传入的同一个 ``ds``，避免同进程多次登录触发连接上限。
    """
    today = pd.Timestamp(dt.date.today())
    shared = Path(str(Config.cache()["root"]).replace("//", "/")) / \
        f"index_daily_{code.replace('.', '_')}.parquet"
    local = (Path(cache_dir) if cache_dir
             else DEFAULT_CACHE.parent / "spot_cache") / f"{code}.parquet"

    for p in ([shared] if not refresh else []) + [local]:
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.nlevels > 1:
                df = df.xs(code, level="code")
            df = df.sort_index()
            if df.index.max() >= today - pd.Timedelta(days=5):
                return df

    if ds is None:
        from data.datasource import create_datasource
        ds = create_datasource()
    raw = ds.get_daily_kline([code], begin, int(today.strftime("%Y%m%d")))
    df = raw.xs(code, level="code").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    local.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local)
    return df


def coverage(panel: pd.DataFrame) -> dict:
    """基差面板健康检查：样本区间、缺失率、极端值。"""
    n = len(panel)
    return {
        "n": n,
        "begin": str(panel.index.min().date()) if n else "",
        "end": str(panel.index.max().date()) if n else "",
        "na_basis": float(panel["basis_front"].isna().mean()) if n else 1.0,
        "min_basis": float(panel["basis_front"].min()) if n else np.nan,
        "max_basis": float(panel["basis_front"].max()) if n else np.nan,
    }
