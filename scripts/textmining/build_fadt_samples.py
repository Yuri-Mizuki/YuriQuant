"""
FADT 事件样本构建（对齐华泰 AI 57 文本FADT选股）
==================================================

AI 57（《人工智能57：文本FADT选股》20220701）与 AI 51 的本质区别：
**研报本身就是事件样本**，无需"预告+匹配"两步——"分析师盈利预测及评级
调整通常跟随点评报告一起发出，可简化 SUE.txt 的构建流程"。

样本定义（对齐研报 + 个人数据域近似）：
1. 事件 = **摘要/标题含"上调"或"下调"**的研报（盈利预测/目标价/评级调整
   的文本表达；研报用朝阳永续结构化字段判断"本次 vs 上次预测变化"，
   我们以明确上/下调字样近似——实测 2019 以来命中 22.7%，约 5.1 万条）；
2. 剔除 **首次覆盖/首盖** 样本（研报明确要求剔除首盖；首盖无"上次预测"
   可比）；
3. 一条研报 = 一条样本（event_date = 研报发布日期，天然 PIT）。
4. 标签 = 研报发布日 T 前后各 1 个交易日（T-1~T+1）相对中证500 的两日
   异常收益 AR，三等分（30/40/30）在 train 阶段按训练窗口分位生成。

用法：
    python -m scripts.textmining.build_fadt_samples --pool zz1000
    python -m scripts.textmining.build_fadt_samples --pool hs300

产出：
    reports/textmining/fadt_samples_{pool}.parquet
    reports/textmining/fadt_samples_{pool}_summary.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from data.textmining.fetch import TextMiningCache
from scripts.textmining.build_sue_txt_samples import (
    _abnormal_return,
    _load_daily,
    _to_naive,
)

OUT_DIR = Path(r"E:\YuriQuant\reports\textmining")
BENCH = "000905.SH"  # 中证500

# AI 57：调整事件文本线索（近似"本次 vs 上次预测变化"）
UP_PAT = re.compile(r"上调|上调至|上调预测|上调目标|上修")
DOWN_PAT = re.compile(r"下调|下调至|下调预测|下调目标|下修")
# 研报要求剔除首盖样本
COVER_PAT = re.compile(r"首次覆盖|首盖|首次评级")


def load_adjustment_reports(pool: str, begin: int, end: int) -> pd.DataFrame:
    """同花顺研报中识别"盈利预测/评级调整"事件样本。

    近似研报"剔除首盖 + 预测不变"：取明确含上调/下调字样的研报，剔除首盖。
    """
    with open(Path(rf"E:\YuriQuant\reports\{pool}_pit_union_2019.json"),
              encoding="utf-8") as f:
        codes = json.load(f)
    cache = TextMiningCache()
    df = cache.get_ths_reports(codes)
    # get_ths_reports 返回全部缓存（跨池），必须按请求 codes 过滤（防串池）
    from data.textmining.fetch import to_code_std, to_code6  # noqa: PLC0415
    std_codes = {to_code_std(to_code6(c)) for c in codes}
    df = df[df["code"].isin(std_codes)].copy()
    df["date"] = _to_naive(df["date"])
    df = df[(df["date"] >= pd.Timestamp(str(begin))) &
            (df["date"] <= pd.Timestamp(str(end)))].copy()

    txt = (df["title"].fillna("") + " " + df["summary"].fillna(""))
    is_up = txt.str.contains(UP_PAT, na=False)
    is_down = txt.str.contains(DOWN_PAT, na=False)
    is_cover = txt.str.contains(COVER_PAT, na=False)

    ev = df[is_up | is_down].copy()
    ev = ev[~is_cover[is_up | is_down]].copy()  # 剔除首盖
    ev["event_type"] = "both"
    ev.loc[is_up & ~is_down, "event_type"] = "up"
    ev.loc[is_down & ~is_up, "event_type"] = "down"
    ev = ev.rename(columns={"date": "event_date"})
    return ev[["code", "event_date", "event_type", "title", "summary",
               "rating", "org"]].reset_index(drop=True)


def build_samples(pool: str = "zz1000", begin: int = 20190101,
                  end: int = 20261231,
                  out_dir: str = r"E:\YuriQuant\reports\textmining") -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] 识别盈利预测/评级调整研报（含上调|下调，剔除首盖）...")
    ev = load_adjustment_reports(pool, begin, end)
    print(f"      调整事件样本: {len(ev)} 条 / {ev['code'].nunique()} 只")
    print(f"      类型分布: {ev['event_type'].value_counts().to_dict()}")

    print("[2/3] 计算研报发布日 T-1~T+1 两日异常收益 AR（基准中证500）...")
    daily = _load_daily(None, [], begin, end)
    with_ar = _abnormal_return(ev, daily)
    valid = with_ar.dropna(subset=["ar"]).copy()
    print(f"      AR 有效: {len(valid)} 行")

    print("[3/3] 输出样本...")
    cols = ["code", "event_date", "event_type", "title", "summary",
            "rating", "org", "ar"]
    out_df = valid[cols].sort_values(["event_date", "code"]).reset_index(drop=True)
    out_df.to_parquet(out / f"fadt_samples_{pool}.parquet", compression="snappy")

    lines = [
        f"调整事件样本: {len(ev)} 条 / {ev['code'].nunique()} 只",
        f"事件类型: {ev['event_type'].value_counts().to_dict()}",
        f"AR 有效: {len(valid)} 行",
        f"AR 分位: q30={valid['ar'].quantile(0.3):.4f} q70={valid['ar'].quantile(0.7):.4f}",
        f"每股票平均事件数: {valid.groupby('code').size().mean():.1f}",
    ]
    (out / f"fadt_samples_{pool}_summary.txt").write_text("\n".join(lines),
                                                          encoding="utf-8")
    print("\n".join(lines))
    print(f"\n样本已存: {out / f'fadt_samples_{pool}.parquet'}")
    return out_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20261231)
    args = ap.parse_args()
    build_samples(args.pool, args.begin, args.end)
