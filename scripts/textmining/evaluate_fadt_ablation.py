"""
AI 63 消融实验汇总评估
======================

对基线（截断 CLS）与各消融变体（分段 / pooler 全连接层 / CLS+词频拼接）
统一产出：覆盖度、月度 RankIC（NW t）、5 层分层（基准全A等权）、变体间
相关性。评估框架复用 evaluate_senti 的按池收益加载器（修复旧脚本硬编码
hs300 日线的错配）。

对应 AI 63 五组扩展测试中的三组（本地复现口径）：
    测试1 截断 vs 分段     → base(trunc) vs seg
    测试3 CLS vs 全连接层  → base(cls) vs pooler（pooler=CLS 线性推导，
                              与 live 前向差 ~1e-6，见 derive_pooler_encoding）
    测试4 CLS+词频 concat  → base(cls) vs clswf

用法：
    python -m scripts.textmining.evaluate_fadt_ablation --pool zz1000 \
        --variants base_xgb,seg,pooler,clswf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402
from scripts.textmining._paths import Out  # noqa: E402
from scripts.textmining.evaluate_senti import (  # noqa: E402
    load_next_ret,
    rank_ic_stats,
    stratified,
)

OUT_DIR = Out("fadt")
log = setup_logging("fadt_ablation_eval")

# 变体标签 → 因子文件名（build_factor_from_pred 产物）
DEFAULT_FACTORS = {
    "base": "fadt_factor_xgb_{pool}.parquet",          # 词频版（AI 57 基线）
    "bert": "fadt_factor_bert_xgb_{pool}.parquet",     # 截断 CLS（AI 63 主线）
    "seg": "fadt_factor_seg_xgb_{pool}.parquet",
    "pooler": "fadt_factor_pooler_xgb_{pool}.parquet",
    "clswf": "fadt_factor_clswf_xgb_{pool}.parquet",
}


def run(pool: str = "zz1000", variants: list[str] | None = None):
    variants = variants or ["bert", "seg", "pooler", "clswf"]
    ret, _ = load_next_ret()
    rows, panels = [], {}
    for v in variants:
        p = OUT_DIR / DEFAULT_FACTORS[v].format(pool=pool)
        if not p.exists():
            log.warning("%s 缺失，跳过（先训练该变体）", p)
            continue
        f = pd.read_parquet(p).reset_index()
        f["date"] = pd.to_datetime(f["date"])
        f["month"] = f["date"].dt.to_period("M")
        panels[v] = f
        cov = f.groupby("month")["code"].size()
        ic = rank_ic_stats(f[["date", "code", "factor"]], ret)
        st = stratified(f[["date", "code", "factor"]], ret,
                        ret.mean(axis=1).rename("bench"))
        rows.append({"variant": v, "cov_mean": round(cov.mean(), 1),
                     "ic_mean": round(ic["ic_mean"], 4),
                     "ic_nw_t": round(ic["ic_nw_t"], 2),
                     "icir": round(ic["icir"], 2),
                     "ic_pos": round(ic["ic_positive"], 2),
                     "L1_ann": round(st.get("L1_annual", float("nan")), 4),
                     "L1_excess": round(st.get("L1_excess", float("nan")), 4),
                     "L1_L5": round(st.get("L1_L5", float("nan")), 4),
                     "n_months": st.get("n_months", 0)})

    sm = pd.DataFrame(rows)
    print("\n== AI 63 消融汇总（池 %s，分层基准 全A等权）==" % pool)
    print(sm.to_string(index=False))

    # 变体间月度截面相关性
    if len(panels) > 1:
        merged = None
        for v, f in panels.items():
            w = f[["month", "code", "factor"]].rename(columns={"factor": v})
            merged = w if merged is None else merged.merge(
                w, on=["month", "code"], how="inner")
        corr = merged.drop(columns=["month", "code"]).corr(method="spearman").round(3)
        print("\n[变体间截面 Spearman 相关]\n", corr.to_string())
    else:
        corr = pd.DataFrame()

    out = OUT_DIR / "fadt_ablation_eval.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# AI 63 消融汇总（pool={pool}，基准 全A等权）\n\n")
        fh.write(sm.to_string(index=False) + "\n\n")
        if not corr.empty:
            fh.write("## 变体间截面 Spearman 相关\n\n" + corr.to_string() + "\n")
    log.info("已存: %s", out)
    return sm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="zz1000", choices=["hs300", "zz1000"])
    ap.add_argument("--variants", default="bert,seg,pooler,clswf")
    args = ap.parse_args()
    run(args.pool, [v.strip() for v in args.variants.split(",")])
