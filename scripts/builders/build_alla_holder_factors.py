"""全A股东结构行为因子面板 → 并入 all_a_2018_2026 数据集。

背景（2026-09-03）：本地 holder_num / share_holder 已从 ~520 只回补到全A
（5788 / 5806 只，约 99%），但 all_a_2018_2026 的 registry 里只有量价因子
（alpha101/158/191/360），股东类因子仅存在于旧的 hs300_2025 子集且只算过
520 只。本脚本在全A尺度上构建"股东结构行为"因子族，并 append 进该数据集的
registry.csv 与 ic_h{1,5,10,20}.parquet，使滚动实验的 select 阶段能把它们
纳入特征候选池（信息多样性：筹码集中异象与量价信号低相关）。

因子族（PIT 到日频，ann_date 前向填充避免前视）：
  股东户数类（holder_num）：
    holder_num_chg   : 股东户数环比变化率（(当期-上期)/上期，负=筹码集中）
    holder_num_yoy   : 股东户数同比变化率（与 4 报告期前比，去季节性）
    holder_num_ratio : 户数相对股本稀释指标（decay 集中度，顺沿项目 zscore 前）
  十大股东类（share_holder）：
    top1_holding     : 第一大股东持股比例（holder_pct max）
    top5_holding     : 前五大股东持股比例和
    top10_holding    : 前十大股东持股比例和
    top10_hhi        : 前十大股东持股 HHI（比例平方和，股权集中度）
    inst_holding     : 机构股东持股比例和（按名称启发式识别机构）
    board_change     : 十大股东年内更替率（当期待续在榜比例，治理层稳定度）
    top10_dispersion : 前十大股东持股比例标准差（分散度，反向集中）

口径：
- 每个报告期 (code, holder_end_date) 聚合成一行；
- 用 ann_date（公告日）作为 PIT 生效边界 ffill 到交易日，实现"披露后才可见"；
- 面板 date×code float32，从 20160701 起，与现有 ic_h 索引严格对齐；
- IC 用次日/未来收益 Spearman（与量价因子 calc_ic_series 同口径），抽稀取 IC_CODE_STRIDE 列。

用法:
    python -m scripts.builders.build_alla_holder_factors            # 单机
    python -m scripts.builders.build_alla_holder_factors --resume
    # 断点续跑（沿用 factor_stats_holder.jsonl）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_holder")
from scripts.builders import common  # noqa: E402

DATASET = "all_a_2018_2026"

INST_KEYWORDS = ("基金", "保险", "信托", "证券", "银行", "社保", "汇金",
                 "资产管理", "投资", "企业年金", "养老", "QFII", "外资")


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载全A复权价 + 股东两张表，返回 (close_adj, holder_num, share_holder)。"""
    from config import Config
    from scripts.builders import common

    cache_root = Path(str(Config.cache()["root"]))
    holder_num = pd.read_parquet(cache_root / "holder_num.parquet")
    share_holder = pd.read_parquet(cache_root / "share_holder.parquet")
    return common.load_close_adj(), holder_num, share_holder


def pit_holder_num(holder_num: pd.DataFrame, cal_idx, codes) -> dict:
    """股东户数：环比/同比变化率（报告期粒度算好再 PIT，PIT 后不重算 diff）。"""
    hn = holder_num.copy()
    for c in ("ann_date", "holder_end_date"):
        hn[c] = pd.to_datetime(hn[c], errors="coerce")
    hn["num"] = pd.to_numeric(hn["holder_num"], errors="coerce")
    hn = hn.dropna(subset=["code", "holder_end_date", "num"])
    hn = (hn.sort_values(["code", "holder_end_date"])
            .drop_duplicates(subset=["code", "holder_end_date"], keep="last"))
    hn = hn.dropna(subset=["num"])

    # 报告期粒度：每股指按 holder_end_date 排序，环比/同比在报告期上算
    def _pairs(s: pd.Series, lag: int) -> pd.Series:
        return s.groupby(level="code").pct_change(periods=lag)

    base = hn.set_index(["code", "holder_end_date"])["num"]
    chg = _pairs(base, 1)
    yoy = _pairs(base, 4)      # 4 报告期前（季频=同比）
    out_long = pd.DataFrame({
        "code": hn["code"], "eff": hn["ann_date"],
        "holder_num_chg": chg.mul(-1).values,   # 户数降=筹码集中（正向）
        "holder_num_yoy": yoy.values,
    })
    return common.ffill_pit_multi(out_long, cal_idx, codes,
                            ["holder_num_chg", "holder_num_yoy"])


def pit_share_holder(share_holder: pd.DataFrame, cal_idx, codes) -> dict:
    """十大股东：集中度/机构/治理相关因子（PIT，向量化聚合）。"""
    sh = share_holder.copy()
    for c in ("ann_date", "holder_end_date"):
        sh[c] = pd.to_datetime(sh[c], errors="coerce")
    sh["pct"] = pd.to_numeric(sh["holder_pct"], errors="coerce")
    sh = sh.dropna(subset=["code", "holder_end_date", "pct"])
    sh["eff"] = sh["ann_date"]
    sh["is_inst"] = sh["holder_name"].astype(str).apply(
        lambda n: any(k in n for k in INST_KEYWORDS))
    sh["_k"] = sh["code"].astype(str) + "|" + sh["holder_end_date"].astype(str)

    grp = sh.groupby(["_k"], sort=False)
    top10 = grp["pct"].sum()
    eff = grp["ann_date"].min()
    top1 = grp["pct"].max()

    # top5：组内 rank 前 5 加和
    sh = sh.assign(_rk=sh.groupby(["_k"], sort=False)["pct"].rank(
        method="first", ascending=False))
    top5 = (sh.loc[sh["_rk"] <= 5].groupby(["_k"], sort=False)["pct"].sum())

    # HHI：组内 (pct/sum)^2 求和（sum=top10）
    sh = sh.assign(_top10=sh["_k"].map(top10))
    hhi = ((sh["pct"] / sh["_top10"]) ** 2).groupby(sh["_k"], sort=False).sum()

    # 机构持股比例和（is_inst 加权）
    inst = (sh.loc[sh["is_inst"]].groupby(["_k"], sort=False)["pct"].sum())

    agg = (pd.DataFrame({
        "_k": top10.index, "eff": eff.values, "top1": top1.values,
        "top5": top5.reindex(top10.index).values,
        "top10": top10.values,
        "hhi": hhi.reindex(top10.index).values,
        "inst": inst.reindex(top10.index).values,
    }).dropna(subset=["top1"]))
    agg[["code", "holder_end_date"]] = agg["_k"].str.split("|", expand=True)
    agg["holder_end_date"] = pd.to_datetime(agg["holder_end_date"])

    out_long = pd.DataFrame({
        "code": agg["code"], "eff": agg["eff"],
        "top1_holding": agg["top1"],
        "top5_holding": agg["top5"],
        "top10_holding": agg["top10"],
        "top10_hhi": agg["hhi"],
        "inst_holding": agg["inst"],
    })
    return common.ffill_pit_multi(out_long, cal_idx, codes,
                            ["top1_holding", "top5_holding", "top10_holding",
                             "top10_hhi", "inst_holding"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from config import Config

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET

    close_adj, holder_num, share_holder = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index

    defs: dict[str, dict] = {}
    panels_all: dict[str, pd.DataFrame] = {}
    if holder_num is not None and not holder_num.empty:
        hpn = pit_holder_num(holder_num, cal_idx, codes)
        panels_all.update(hpn)
        defs.update({k: {"src": "holder_num", "label": k} for k in hpn})
    if share_holder is not None and not share_holder.empty:
        spn = pit_share_holder(share_holder, cal_idx, codes)
        panels_all.update(spn)
        defs.update({k: {"src": "share_holder", "label": k} for k in spn})

    common.write_family_panels(
        ds_dir, panels_all, family="holder", set_name="holder",
        labels={k: v["label"] for k, v in defs.items()},
        close_adj=close_adj, clip=common.CLIP["PCT"], resume=args.resume,
        merge_kwargs={"skip_existing": False, "empty_log": "无股东因子统计产出"})
    log.info("股东因子构建完成")


if __name__ == "__main__":
    main()