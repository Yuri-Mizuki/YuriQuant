"""ETF → 跟踪指数代理：为长样本（2013–2019）提供价格序列（README §4.6 前置）。

为什么需要这一步：ETF 池里绝大多数产品 2019 年后才上市，2013–2019 窗口里它们的
价格不存在。做法是用**跟踪指数的日收益**代理，但代理必须逐只**验证**——映射填错
会让整个长样本回测变成噪声。验证手段 = ETF 后复权日收益 vs 指数日收益的重叠期相关。

判读：
- corr ≥ 0.95 → `ok`，可用
- 0.90 ≤ corr < 0.95 → `warn`，可用但需标注跟踪偏离
- corr < 0.90 → `reject`，映射不可信，该 ETF 在长样本中剔除
- 指数无数据 / 起点晚于 2019 → `nodata`，剔除

⚠️ 数据源约束（2026-10-30 实测）：指数日线最早只到 **2013-01-04**（不是 2010），
且部分指数会中途断供（如中证红利 000922 止于 2019-01-30）。跨境/商品指数
（NDX/SPX/HSI/AU9999）大概率不在国内源里 ⇒ 由验证自动剔除，不要手工补。

复现：`python -m scripts.portfolio.etf_index_proxy --begin 20120101 --end 20261231`
产物：`reports/etf_contest5/etf_index_proxy.csv`
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# 候选映射：ETF → 跟踪指数（代码为最佳知识，**以验证结果为准**）
ETF_INDEX_PROXY: dict[str, tuple[str, str]] = {
    # 宽基
    "510300.SH": ("沪深300ETF", "000300.SH"),
    "510050.SH": ("上证50ETF", "000016.SH"),
    "510500.SH": ("中证500ETF", "000905.SH"),
    "512100.SH": ("中证1000ETF", "000852.SH"),
    "159915.SZ": ("创业板ETF", "399006.SZ"),
    "588000.SH": ("科创50ETF", "000688.SH"),
    "159901.SZ": ("深证100ETF", "399330.SZ"),
    "510880.SH": ("红利ETF", "000015.SH"),
    # 行业
    "512010.SH": ("医药ETF", "000933.SH"),
    "512170.SH": ("医疗ETF", "399989.SZ"),
    "512880.SH": ("证券ETF", "399975.SZ"),
    "512480.SH": ("半导体ETF", "H30184.CSI"),
    "515030.SH": ("新能源车ETF", "399976.SZ"),
    "512400.SH": ("有色金属ETF", "000819.SH"),
    "512660.SH": ("军工ETF", "399967.SZ"),
    "512200.SH": ("房地产ETF", "931775.CSI"),
    "512800.SH": ("银行ETF", "399986.SZ"),
    "512980.SH": ("传媒ETF", "399971.SZ"),
    "512070.SH": ("证券保险ETF", "399394.SZ"),
    "512290.SH": ("生物医药ETF", "930726.CSI"),
    "515220.SH": ("煤炭ETF", "399998.SZ"),
    "159516.SZ": ("半导体设备ETF", "931743.CSI"),
    "159992.SZ": ("创新药ETF", "931152.CSI"),
    # 主题
    "515790.SH": ("光伏ETF", "931151.CSI"),
    "159928.SZ": ("消费ETF", "000932.SH"),
    "159949.SZ": ("创业板50ETF", "399673.SZ"),
    "159869.SZ": ("游戏ETF", "930901.CSI"),
    "516160.SH": ("新能源ETF", "399808.SZ"),
    "512760.SH": ("芯片ETF", "990001.CSI"),
    "159825.SZ": ("农业ETF", "000949.SH"),
    "515880.SH": ("通信ETF", "931160.CSI"),
    "159755.SZ": ("电池ETF", "931719.CSI"),
    "588200.SH": ("科创芯片ETF", "000685.SH"),
    "562500.SH": ("机器人ETF", "931814.CSI"),
    # 债券
    "511010.SH": ("国债ETF", "000012.SH"),
    "511260.SH": ("十年国债ETF", "000140.SH"),
    # 商品
    "518880.SH": ("黄金ETF", "AU9999.SGE"),
    # 跨境
    "513100.SH": ("纳指ETF", "NDX"),
    "513500.SH": ("标普500ETF", "SPX"),
    "513050.SH": ("中概互联ETF", "CSIH11136"),
    "159920.SZ": ("恒生ETF", "HSI"),
    "513180.SH": ("恒生科技ETF", "HSTECH"),
    "513330.SH": ("恒生互联网ETF", "HSIII"),
    "513520.SH": ("日经ETF", "N225"),
}


def fetch_one(cache, index_code: str, begin: int, end: int) -> pd.Series | None:
    try:
        d = cache.get_index_daily(index_code, begin, end)
    except Exception:
        return None
    if d is None or not len(d) or "close" not in d.columns:
        return None
    c = d["close"].unstack("code")
    if c.empty:
        return None
    s = c.iloc[:, 0].dropna()
    return s if len(s) else None


def main(argv=None) -> int:
    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.etf_universe import EtfUniverse

    ap = argparse.ArgumentParser(description="ETF→指数代理映射验证")
    ap.add_argument("--begin", type=int, default=20120101)
    ap.add_argument("--end", type=int, default=20261231)
    ap.add_argument("--csv", default="reports/etf_contest5/etf_index_proxy.csv")
    args = ap.parse_args(argv)

    cache = DataCache(create_datasource())
    uni = EtfUniverse(cache)
    etf_close = uni.load_close(adjust=True)       # date × etf，后复权
    etf_ret = etf_close.pct_change(fill_method=None)

    cache_idx: dict[str, pd.Series | None] = {}
    rows = []
    print(f"拉指数 {args.begin}→{args.end}（{len(ETF_INDEX_PROXY)} 个候选）")
    for i, (code, (name, idx)) in enumerate(ETF_INDEX_PROXY.items(), 1):
        if idx not in cache_idx:
            cache_idx[idx] = fetch_one(cache, idx, args.begin, args.end)
        s = cache_idx[idx]
        row = {"etf": code, "name": name, "index": idx}
        if s is None:
            row.update({"corr": np.nan, "overlap": 0, "idx_start": None,
                        "idx_end": None, "verdict": "nodata"})
        else:
            r = s.pct_change(fill_method=None)
            r.index = pd.to_datetime(r.index)
            common = etf_ret.index.intersection(r.index)
            a = etf_ret.loc[common, code] if code in etf_ret.columns else pd.Series(dtype=float)
            b = r.loc[common]
            m = a.notna() & b.notna()
            n = int(m.sum())
            corr = float(a[m].corr(b[m])) if n >= 60 else np.nan
            verdict = ("ok" if corr >= 0.95 else "warn" if corr >= 0.90
                       else "reject" if n >= 60 else "nodata")
            row.update({"corr": corr, "overlap": n,
                        "idx_start": s.index.min().date(), "idx_end": s.index.max().date(),
                        "verdict": verdict})
        rows.append(row)
        print(f"  [{i:2d}/{len(ETF_INDEX_PROXY)}] {code} {name:<10} → {idx:<12} "
              f"corr={row.get('corr', float('nan')):.3f} n={row.get('overlap', 0):<4} "
              f"{row['verdict']}")

    d = pd.DataFrame(rows)
    d.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print("\n== 汇总 ==")
    print(d["verdict"].value_counts().to_string())
    print(f"\n可用（ok+warn）：{int(d['verdict'].isin(['ok', 'warn']).sum())} 只")
    ok = d[d["verdict"].isin(["ok", "warn"])].sort_values("corr", ascending=False)
    for _, r in ok.iterrows():
        print(f"  {r['etf']} {r['name']:<10} {r['index']:<12} corr={r['corr']:.3f} "
              f"n={r['overlap']:<5} {r['idx_start']}~{r['idx_end']}")
    print(f"\n落盘 → {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
