"""
SUE.txt 事件样本构建（对齐华泰 AI 51 文本PEAD）
================================================

流程（严格对齐研报《人工智能51：文本PEAD选股策略》20220107）：
1. 事件源：业绩预告（巨潮公告，category=业绩预告）。
2. 研报匹配：预告发布后 **5 个自然日** 内该股的研报视为对预告的解读
   （研报假设：难以精确定位与预告相关的研报，宽松匹配）。
3. 标签：预告发布前、后 2 个交易日（T-1~T+1）相对中证500 的两日异常收益
   AR = (1+r_stock)^... 研报定义：前后两个交易日的收盘价，中证500同期收益为基准。
   三等分：前30%上涨(y=1) / 30-70%震荡(y=0) / 后30%下跌(y=-1)。
4. 输出样本 DataFrame：事件级（一条预告 × 匹配研报 = 一行，重复预告多行）。

用法：
    python -m scripts.textmining.build_sue_txt_samples
    python -m scripts.textmining.build_sue_txt_samples --begin 20190101 --end 20261231

产出：
    reports/textmining/sue_txt_samples.parquet（样本）
    reports/textmining/sue_txt_samples_summary.txt（统计）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.cache import DataCache
from data.datasource import create_datasource
from data.textmining.fetch import TextMiningCache

ROOT = Path(__file__).resolve().parents[2]

BENCH = "000905.SH"  # 中证500


def _load_events(cache: TextMiningCache, begin: int, end: int,
                 pool: str = "hs300") -> pd.DataFrame:
    """业绩预告事件（巨潮 category=业绩预告）。"""
    with open(ROOT / f"reports/{pool}_pit_union_2019.json",
              encoding="utf-8") as f:
        codes = json.load(f)
    df = cache.get_cninfo_announcements(
        codes=codes, begin_date=begin, end_date=end, categories=["业绩预告"])
    if df.empty:
        return df
    df = df[df["title"].fillna("").str.contains("预告", na=False)]  # 对齐AI51：仅业绩预告
    df = df.rename(columns={"date": "event_date", "title": "event_title"})
    return df[["code", "event_date", "event_title"]].copy()


def _load_reports(cache: TextMiningCache, codes: list[str]) -> pd.DataFrame:
    """同花顺研报（标题+摘要+评级）。"""
    df = cache.get_ths_reports(codes)
    return df[["code", "date", "title", "summary", "rating", "org"]].copy()


def _match_reports(events: pd.DataFrame, reports: pd.DataFrame,
                   window_days: int = 5) -> pd.DataFrame:
    """预告发布后 window_days 自然日内的研报匹配。

    AI 51：假设发布后 5 个自然日内的所有个股相关研报都是对该预告的评论解读。
    """
    ev = events.copy()
    ev["event_date"] = _to_naive(ev["event_date"])
    rep = reports.copy()
    rep = rep.rename(columns={"date": "report_date"})
    rep["report_date"] = _to_naive(rep["report_date"])

    # 合并后按窗口过滤
    m = ev.merge(rep, on="code", how="inner")
    m = m[(m["report_date"] >= m["event_date"]) &
          (m["report_date"] <= m["event_date"] + pd.Timedelta(days=window_days))]
    return m


def _load_daily(cache: DataCache, codes: list[str], begin: int, end: int) -> pd.DataFrame:
    """直接读 daily_{pool}.parquet 缓存（避免 DataCache.get_daily_kline 触发 calendar
    增量写入 → PermissionError，沙箱对 e:/data 写锁）。数据已由 fetch 阶段拉齐。"""
    from config import Config
    daily = pd.read_parquet(
        Path(str(Config.cache()["root"]).replace("//", "/")) / "daily_hs300.parquet")
    return daily


def _to_naive(s: pd.Series) -> pd.Series:
    """统一为 tz-naive 的 datetime（巨潮带 Asia/Shanghai tz，同花顺/日线为 naive）。"""
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return s


def _abnormal_return(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """计算预告发布日前后 2 个交易日（T-1~T+1）相对中证500 的两日异常收益 AR。

    AR = (P_T+1 / P_T-1 - 1) - (Idx_T+1 / Idx_T-1 - 1)

    事件日 T 归属（对齐研报 AI 51"公告前后 2 个交易日"）：
      - 公告日若为交易日 → T = 公告日当天（公告盘后发布，T 当日收盘价不含
        公告信息，T-1→T+1 两日窗口含 1 个信息交易日，与研报口径一致）；
      - 公告日若非交易日（周末/节假日）→ T = 公告日之前最近交易日。
    前后各取 1 个交易日：c_prev = T-1 收盘价、c_next = T+1 收盘价。
    """
    # 每个 code 的交易日序列（升序）→ 用 searchsorted 定位
    by_code = {}
    for code, sub in daily.reset_index().groupby("code"):
        sub = sub.copy()
        sub["date"] = _to_naive(sub["date"])
        by_code[code] = sub.sort_values("date")[["date", "close"]].reset_index(drop=True)

    def _close_shift(code: str, d: pd.Timestamp, shift: int) -> float | None:
        """code 在日期 d 所在交易日偏移 shift 个交易日的收盘价。"""
        tbl = by_code.get(code)
        if tbl is None or tbl.empty:
            return None
        dates = tbl["date"].values
        # 事件日在交易日序列中的位置：<= d 的最近交易日（即 T 的锚点）
        pos = int(pd.Index(dates).searchsorted(pd.Timestamp(d).to_datetime64(),
                                               side="right")) - 1
        if pos < 0:
            return None  # 事件日早于该股数据起点
        j = pos + shift
        if 0 <= j < len(dates):
            return float(tbl["close"].iloc[j])
        return None

    rows = []
    for _, r in events.iterrows():
        code = r["code"]
        d = pd.Timestamp(r["event_date"]).to_datetime64()
        c_prev = _close_shift(code, d, -1)
        c_next = _close_shift(code, d, 1)
        b_prev = _close_shift(BENCH, d, -1)
        b_next = _close_shift(BENCH, d, 1)
        if None in (c_prev, c_next, b_prev, b_next):
            ar = None
        else:
            ar = (c_next / c_prev - 1) - (b_next / b_prev - 1)
        rows.append({"code": code, "event_date": pd.Timestamp(d), "ar": ar})
    ar_df = pd.DataFrame(rows)
    # 同一事件多条研报会重复计算 AR → 按事件去重（AR 是事件级属性）
    ar_df = ar_df.drop_duplicates(subset=["code", "event_date"], keep="first")
    return events.merge(ar_df, on=["code", "event_date"], how="left")


def build_samples(begin: int = 20190101, end: int = 20261231,
                  pool: str = "hs300",
                  out_dir: str = str(ROOT / "reports" / "textmining")) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = TextMiningCache()
    with open(ROOT / f"reports/{pool}_pit_union_2019.json",
              encoding="utf-8") as f:
        codes = json.load(f)

    print("[1/4] 加载业绩预告事件...")
    events = _load_events(cache, begin, end, pool=pool)
    print(f"      业绩预告事件: {len(events)} 条, 覆盖 {events['code'].nunique()} 只")

    print("[2/4] 加载研报 + 匹配（5 自然日）...")
    reports = _load_reports(cache, codes)
    print(f"      研报: {len(reports)} 条")
    matched = _match_reports(events, reports, window_days=5)
    print(f"      匹配样本: {len(matched)} 行（预告×研报）")

    print("[3/4] 计算两日异常收益 AR...")
    daily = _load_daily(cache, codes + [BENCH], begin, end)
    with_ar = _abnormal_return(matched, daily)
    valid = with_ar.dropna(subset=["ar"])
    print(f"      AR 有效样本: {len(valid)} 行")

    print("[4/4] 三等分标签...")
    q30, q70 = valid["ar"].quantile([0.3, 0.7])
    valid = valid.copy()
    valid["label"] = valid["ar"].apply(
        lambda x: 1 if x >= q70 else (-1 if x <= q30 else 0))
    valid["label"] = valid["label"].astype(int)

    cols = ["code", "event_date", "event_title", "report_date", "title",
            "summary", "rating", "org", "ar", "label"]
    out_df = valid[cols].sort_values(["event_date", "code"]).reset_index(drop=True)
    out_df.to_parquet(out / f"sue_txt_samples_{pool}.parquet", compression="snappy")

    # 统计摘要
    lines = [
        f"业绩预告事件: {len(events)} 条 / {events['code'].nunique()} 只",
        f"匹配样本: {len(matched)} 行（预告×研报）",
        f"AR 有效: {len(valid)} 行",
        f"标签分布: {valid['label'].value_counts().sort_index().to_dict()}",
        f"AR 分位: q30={q30:.4f} q70={q70:.4f}",
        f"平均每事件研报数: {valid.groupby(['code','event_date']).size().mean():.2f}",
    ]
    (out / f"sue_txt_samples_{pool}_summary.txt").write_text("\n".join(lines),
                                                             encoding="utf-8")
    print("\n".join(lines))
    print(f"\n样本已存: {out / f'sue_txt_samples_{pool}.parquet'}")
    return out_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20261231)
    ap.add_argument("--pool", default="hs300", choices=["hs300", "zz1000"])
    args = ap.parse_args()
    build_samples(args.begin, args.end, pool=args.pool)
