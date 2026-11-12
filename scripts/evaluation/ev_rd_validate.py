"""EV 分母 / 研发加回 四个候选因子的三段式 + 增量检验。

背景（2026-11-12）：`build_alla_ev_rd_factors` 产出 4 个候选面板
（`bp_ev` / `ep_ev` / `sp_ev` / `ep_rd`），本脚本判它们**是否带来母因子之外的
信息**。只有母因子（`bp` / `ep_ttm` / `sp_ttm`）不显著或新因子显著更强，才算增量；
否则只是换了个分母、把同一份信息重新加权。

判据（先定后算，四条全过才算通过）：
- A 全窗 ``|t| > 2``
- B 方向符合先验（价值因子：分母变便宜 ⇒ 预期收益高 ⇒ IC>0）
- C **增量显著**：对母因子横截面正交后，残差 ``|t| > 2``（这才是"新信息"）
- D 三段式同号：train 2018-2021 / valid 2022-2023 / test 2024-2026

三段式切分是**一次性**的，用满 8 年；不做参数搜索，所以 train 段不是"训练"而是
"第一次看见"，valid/test 段不可回头改判据。

两个 IC 口径都要报：
- ``raw``：原始截面（研报口径，只做市值中性化）
- ``neu``：行业 + 市值中性化（我们的生产口径 ``panels_neu``）

⚠️ 重叠样本：h=20 用**不重叠**的 20 日间隔（≈107 个截面），t = mean/std×√n。
h=1 用每日截面（≈2100 个），日频 IC 自相关低，同样公式可接受。

用法:
    python -m scripts.evaluation.ev_rd_validate
    python -m scripts.evaluation.ev_rd_validate --stride 1   # 更密的 h=1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from scripts.builders.common import load_close_adj  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("ev_rd_validate")

NEW_DIR = ROOT / "reports" / "factor_ev_rd" / "panels"
OUT = ROOT / "reports" / "factor_ev_rd"
LIB = Path(str(Config.get()["factor_library"]["root"])) / "all_a_2018_2026"

#: 新因子 → 母因子（对照组）。方向先验全部 +1（越便宜越好）。
PAIRS = {
    "bp_ev": "bp",
    "ep_ev": "ep_ttm",
    "sp_ev": "sp_ttm",
    "ep_rd": "ep_ttm",
}
SEGS = {
    "train": ("2018-01-01", "2021-12-31"),
    "valid": ("2022-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2026-12-31"),
}
MIN_XS = 200          # 截面最少股票数

#: 诊断因子（不参与候选判定）：研发强度。若它自身 IC 已足够，
#: 则 ep_rd 的"增量"只是在代理它，没必要把它塞进价值因子。
EXTRA = ("rd_to_cap",)


# --------------------------------------------------------------------------- 数据
def load_panel(name: str) -> pd.DataFrame | None:
    """新因子优先从本次产出目录读，母因子从因子库读。"""
    p = NEW_DIR / f"{name}.parquet"
    if not p.exists():
        p = LIB / "panels" / f"{name}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).astype(np.float64)


def industry_panel(codes: pd.Index, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """申万一级行业标签（季度采样 + 前向填充）。

    分类表是区间事件（in_date/out_date），日频逐日筛太慢；行业归属变动很稀疏，
    按季末取一次再 ffill 足够。每年 4 次 × 8 年 = 32 次筛选。
    """
    pq = Path(str(Config.paths()["cache_root"]))
    cls = pd.read_parquet(pq / "industry_classification_level1.parquet")
    for c in ("in_date", "out_date"):
        cls[c] = pd.to_datetime(cls[c], errors="coerce")
    cls["out_date"] = cls["out_date"].fillna(pd.Timestamp("2099-12-31"))

    stamps = pd.date_range(cal.min(), cal.max(), freq="QE")
    rows = {}
    for d in stamps:
        sub = cls[(cls["in_date"] <= d) & (cls["out_date"] > d)]
        rows[d] = sub.drop_duplicates("code", keep="last").set_index("code")["industry_code"]
    out = pd.DataFrame(rows).T.reindex(columns=codes)
    return out.reindex(cal, method="ffill")


# --------------------------------------------------------------------------- 工具
def _rank_xs(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(df.notna()).rank(axis=1, method="first")


def _resid_xs(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """逐截面 OLS 取残差（单变量，向量化）。y、x 均已粗对齐。"""
    m = y.notna() & x.notna()
    xm = x.where(m).mean(axis=1)
    ym = y.where(m).mean(axis=1)
    xc = x.where(m).sub(xm, axis=0)
    yc = y.where(m).sub(ym, axis=0)
    den = (xc ** 2).sum(axis=1)
    beta = (xc * yc).sum(axis=1) / den.replace(0.0, np.nan)
    return yc - xc.mul(beta, axis=0)


def _industry_demean(r: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    """行业内去均值（秩空间内，等价于行业内排序后再看全局）。"""
    out = r.copy()
    for d in r.index:
        ri = r.loc[d]
        ii = ind.loc[d] if d in ind.index else None
        if ii is None:
            continue
        g = ri.groupby(ii.reindex(ri.index))
        out.loc[d] = ri - g.transform("mean")
    return out


def ic_series(f: pd.DataFrame, fwd: pd.DataFrame,
              dates: pd.DatetimeIndex) -> pd.Series:
    """逐截面 Spearman IC（f 与 fwd 都已在同一网格）。"""
    a = f.loc[dates].rank(axis=1, method="first")
    b = fwd.loc[dates].rank(axis=1, method="first")
    m = a.notna() & b.notna()
    a = a.where(m)
    b = b.where(m)
    ac = a.sub(a.mean(axis=1), axis=0)
    bc = b.sub(b.mean(axis=1), axis=0)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    s = num / den.replace(0.0, np.nan)
    return s.where(m.sum(axis=1) >= MIN_XS).dropna()


def _stats(s: pd.Series) -> dict:
    if len(s) < 8:
        return {"n": len(s), "ic": np.nan, "t": np.nan, "win": np.nan}
    sd = s.std(ddof=1)
    return {"n": len(s), "ic": float(s.mean()),
            "t": float(s.mean() / sd * np.sqrt(len(s))) if sd > 0 else np.nan,
            "win": float((s > 0).mean())}


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=20, help="预测期限（交易日）")
    ap.add_argument("--stride", type=int, default=None,
                    help="截面抽样步长，缺省=不重叠（h 决定）")
    args = ap.parse_args()
    h = args.h
    stride = args.stride or h

    close_adj, close_raw = load_close_adj(with_raw=True)
    cap = load_panel("ln_mktcap")
    if cap is None:
        log.error("缺 ln_mktcap，无法做市值中性化")
        return 2
    cap = cap.reindex(index=close_adj.index, columns=close_adj.columns)

    # 前瞻收益：用复权价，h 日累计
    fwd = close_adj.shift(-h) / close_adj - 1.0
    fwd = fwd.replace([np.inf, -np.inf], np.nan)

    codes = close_adj.columns
    ind = industry_panel(codes, close_adj.index)

    # 不重叠截面：从末尾往前每 stride 日取一个，保证 fwd 不重叠
    cal = close_adj.index[close_adj.index >= pd.Timestamp("2018-01-01")]
    pick = cal[::-1][::stride][::-1]

    rows, seg_rows, orth_rows = [], [], []

    # 单因子（无母因子）：研发强度本身。用来判断 ep_rd 的增量是否只是它的代理。
    for name in EXTRA:
        f = load_panel(name)
        if f is None:
            continue
        f = f.reindex(index=close_adj.index, columns=codes)
        for cal_name, fx in (("raw", f),
                             ("neu", _industry_demean(_resid_xs(f, cap), ind))):
            s = ic_series(fx, fwd, pick)
            st = _stats(s)
            st.update({"factor": name, "vs": "(无)", "caliber": cal_name})
            rows.append(st)
            for seg, (a, b) in SEGS.items():
                st1 = _stats(s.loc[(s.index >= a) & (s.index <= b)])
                st1.update({"factor": name, "caliber": cal_name, "seg": seg})
                seg_rows.append(st1)

    for new, old in PAIRS.items():
        fn = load_panel(new)
        fo = load_panel(old)
        if fn is None:
            log.warning("跳过 %s：面板缺失", new)
            continue
        fn = fn.reindex(index=close_adj.index, columns=codes)
        if fo is not None:
            fo = fo.reindex(index=close_adj.index, columns=codes)

        for cal_name, f in (("raw", fn),
                            ("neu", _industry_demean(_resid_xs(fn, cap), ind))):
            s = ic_series(f, fwd, pick)
            st = _stats(s)
            st.update({"factor": new, "vs": old, "caliber": cal_name})
            rows.append(st)
            for seg, (a, b) in SEGS.items():
                ss = s.loc[(s.index >= a) & (s.index <= b)]
                st2 = _stats(ss)
                st2.update({"factor": new, "caliber": cal_name, "seg": seg})
                seg_rows.append(st2)

        # 增量检验：对母因子正交（横截面，秩空间），再算 IC
        if fo is not None:
            rk_new = _rank_xs(fn.loc[pick])
            rk_old = _rank_xs(fo.loc[pick])
            resid = _resid_xs(rk_new, rk_old).reindex(columns=codes)
            s_r = ic_series(resid, fwd, pick)
            st = _stats(s_r)
            st.update({"factor": new, "vs": old})
            orth_rows.append(st)

            # 母因子自身的 IC 与三段（没有这个对照，无法判断"更强"是相对谁）
            for cal_name, f0 in (("raw", fo),
                                 ("neu", _industry_demean(_resid_xs(fo, cap), ind))):
                s0 = ic_series(f0, fwd, pick)
                st0 = _stats(s0)
                st0.update({"factor": f"{old}(母)", "vs": "", "caliber": cal_name})
                rows.append(st0)
                for seg, (a, b) in SEGS.items():
                    ss = s0.loc[(s0.index >= a) & (s0.index <= b)]
                    st1 = _stats(ss)
                    st1.update({"factor": f"{old}(母)", "caliber": cal_name,
                                "seg": seg})
                    seg_rows.append(st1)

            # 反向残差：母因子对子因子正交。🚨 残差 IC 是**对称**的——
            # 只报单向会误判成"新因子有增量"，其实可能新因子是母因子的子集。
            # 两边都显著 ⇒ 各有独立信息（可融合）；只有新→母显著 ⇒ 新是母的超集。
            resid_rev = _resid_xs(rk_old, rk_new).reindex(columns=codes)
            s_rev = ic_series(resid_rev, fwd, pick)
            st_rev = _stats(s_rev)
            st_rev.update({"factor": old, "vs": new})
            orth_rows.append(st_rev)

    df = pd.DataFrame(rows)
    dseg = pd.DataFrame(seg_rows)
    dorth = pd.DataFrame(orth_rows)

    pd.set_option("display.width", 220, "display.max_columns", 30,
                  "display.unicode.east_asian_width", True)
    def fmt(v):
        return f"{v:+.4f}" if isinstance(v, float) and abs(v) < 10 else f"{v:.3f}"

    print(f"\n=== 全窗 IC（h={h}，{len(pick)} 个不重叠截面）===")
    print(df[["factor", "vs", "caliber", "n", "ic", "t", "win"]].to_string(
        index=False, float_format=fmt))

    print("\n=== 三段式（raw 口径）===")
    print(dseg[dseg.caliber == "raw"].pivot_table(
        index="factor", columns="seg", values=["ic", "t"]).round(4).to_string())

    print("\n=== 增量：对母因子正交后的残差 IC（h=%d）===" % h)
    print(dorth[["factor", "vs", "n", "ic", "t", "win"]].to_string(
        index=False, float_format=fmt))

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"validate_ic_h{h}.csv", index=False)
    dseg.to_csv(OUT / f"validate_seg_h{h}.csv", index=False)
    dorth.to_csv(OUT / f"validate_orth_h{h}.csv", index=False)
    print(f"\n产物 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
