"""更新申万一级行业指数日线（31 条，1999-12 起）
=================================================

用途：行业轮动线的**基准层**。此前本地只有自建「流通市值加权行业收益」，
与研报用的申万官方指数口径不同（分红处理、成分权重、调整时点都有差），
导致行业层 IC 无法与研报对照。本脚本把官方点位落盘，供
`scripts/oneoff/etf_industry_factor_p0.py` 直接消费。

产物：`{data_root}/parquet/sw_industry_daily.parquet`
      列 = date / code / close / amount；code 形如 `801010.SI`

⚠️ 覆盖不均（不是 bug，是申万行业分类沿革）：
   - 16 条（801010–801230）   自 1999-12-30
   - 12 条（801710–801950, 801880/801890）自 2014-02-21（申万 2014 版启用）
   - 3 条（801960 石油石化 / 801970 环保 / 801980 美容护理）自 2021-12-13（2021 版拆出）
   ⇒ **截面行业数随时间变化**（16 → 28 → 31），跨期比较 IC 绝对量级不可直接对比。

    python -m scripts.ingest.update_sw_industry            # 全量刷新
    python -m scripts.ingest.update_sw_industry --check    # 只体检不写
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config  # noqa: E402

OUT_NAME = "sw_industry_daily.parquet"


def out_path() -> Path:
    """数据根一律走 Config.paths()，禁硬编码盘符（tests/test_no_hardcoded_drive_paths.py）。"""
    root = Path(Config.paths()["cache_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root / OUT_NAME


def fetch_all() -> pd.DataFrame:
    import akshare as ak

    first = ak.sw_index_first_info()
    print(f"申万一级行业 {len(first)} 个")
    frames = []
    for i, row in first.iterrows():
        code, name = row["行业代码"], row["行业名称"]
        sym = str(code).split(".")[0]
        for attempt in range(3):
            try:
                d = ak.index_hist_sw(symbol=sym, period="day")
                d = d.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount"})
                d["code"] = code
                frames.append(d[["date", "code", "close", "amount"]])
                rng = f"{d['date'].min()} ~ {d['date'].max()}"
                print(f"  [{i + 1:02d}] {sym} {name:6s} {len(d):5d} 行  {rng}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  [{i + 1:02d}] {sym} 失败({attempt + 1}/3): {str(e)[:80]}")
                time.sleep(2)
        time.sleep(0.2)
    if not frames:
        raise RuntimeError("全部行业拉取失败")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    g = df.groupby("code")["date"].agg(["min", "max", "count"])
    print("\n覆盖汇总：")
    print(g.groupby("min").size().to_string())
    print(f"\n总行数 {len(df)}，行业 {df['code'].nunique()} 个，"
          f"日期 {df['date'].min().date()} ~ {df['date'].max().date()}")
    late = g[g["min"] > "2015-01-01"]
    if len(late):
        print(f"⚠️ 起点晚于 2015 的行业 {len(late)} 个：{', '.join(late.index)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只读已有文件做体检，不联网")
    args = ap.parse_args()

    p = out_path()
    if args.check:
        if not p.exists():
            print(f"不存在：{p}")
            return
        report(pd.read_parquet(p))
        return

    df = fetch_all()
    df.to_parquet(p, index=False)
    print(f"\n已写 {p}")
    report(df)


if __name__ == "__main__":
    main()
