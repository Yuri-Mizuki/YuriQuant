r"""分组哑变量 A/B：申万行业 vs 相关性簇 vs 随机分组（P1-1）
============================================================

假说（来自 2026-11-09 映翡量化转述的 Synthetic Sector Classification）
---------------------------------------------------------------------
``neutralize(industry_panel=...)`` 喂的分组会改变因子残余，进而改变 IC。
论文报「同一套估值模型、只换分组，5 年累计收益差 20pp+」。我们想问：
把 ``industry_panel`` 从申万一级换成「β 残差相关指派」的簇，因子的可交易
IC 变不变。

四臂（**只有 A/B 是处理组，raw 与 rand 是对照**）
--------------------------------------------------
- ``raw``：只 winsorize + zscore，不中性化 —— 量「中性化一共拿走多少」；
- ``sw`` ：生产口径，``industry_panel`` = ``cov_industry.parquet``（申万一级）；
- ``cc`` ：``industry_panel`` = ``corr_sector_panel.parquet``（β 残差相关簇）；
- ``rand``：把 ``cc`` 每行的标签**随机重排**（组数/组大小完全一致）—— 安慰剂。
  K 次取 IC 的分布。若 ``cc`` 不出这个分布，说明换分组的效果等价于"随便怎么分"。

**判据（跑之前写死，不许事后改）**
--------------------------------
判「``cc`` 优于 ``sw``」必须**三条同时满足**：
  ① \|IC_cc\| > \|IC_sw\|（h1 主判，h5 须同向）；
  ② IC_cc 落在 rand 分布的**更好那一侧尾部之外**（因子为正向时看 P95 上方，
     负向时看 P05 下方——🚨 首次实现写成"两侧任一"是错的：它会把「比随机分组
     还差」也判成"显著"，恰好把结论判反）；
  ③ 前后半段 ΔIC = IC_cc − IC_sw **同号**。
只满足 ①② 记「不可分辨」；①②③ 有反向证据记「判负」。

⚠️ 三条护栏（来自本项目既有教训，不是凑数）
-------------------------------------------
- **不看 R² / 解释力**：论文自己证明 GICS 的 R² 更高而收益更低（`etf_contest5`
  也有同类证据：研报用 −corr(pred,ret) 当损失，IC 有、收不到钱）。
- **必须配随机分组安慰剂**：``reports/industry_rotation/beta_buffer.md`` 的
  教训是中性化目标偏移 δ 能让收益在 4.24%~12.54% 无序跳；没有安慰剂的 A/B
  结果一律不可信。
- **两臂 universe 必须逐格相同**：``cc`` 只覆盖上市股票中的 88%~98%，
  直接对跑会混进"样本不同"这个confound。这里把两臂都限制在 ``cc`` 有标签的
  格点上（``sw`` 用 ``sw.where(cc.notna())``），并报被丢掉的比例。

用法
----
    python -m scripts.portfolio.neutralize_group_ab --factors bp --placebo 2   # 冒烟
    python -m scripts.portfolio.neutralize_group_ab                            # 全量
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from factor.preprocessing import preprocess_factor
from stats.ic import calc_ic_series
from stats.robust_stats import nw_tstat

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "industry_rotation"
BASE = ROOT / "reports" / "alla_rolling" / "_base"
DS = Path(Config.paths()["factor_library_root"]) / "all_a_2018_2026"

EVAL_START = "2018-01-02"
EVAL_END = "2026-11-04"
#: 代表因子：覆盖量价 / 价值 / 基本面 / 红利，且都在 950 唯一名内
DEFAULT_FACTORS = (
    "alpha191_046",    # 量价：反转/低位（记忆里 resfac t=7.7/8.1）
    "alpha158_QTLD20",  # 量价
    "alpha101_004",    # 量价
    "bp",              # 价值（行业层 |t|<2 的典型）
    "sue_q",           # 基本面（行业层唯一过线者；面板含前视修复待重建，两臂同等受影响）
    "div_yield",       # 红利
)
PLACEBO_K = 8


# --------------------------------------------------------------------------- 数据
def load_base() -> dict:
    mc = pd.read_parquet(BASE / "market_cap.parquet")
    sw = pd.read_parquet(BASE / "cov_industry.parquet")
    cc = pd.read_parquet(OUT_DIR / "corr_sector_panel.parquet")
    close = pd.read_parquet(BASE / "close_adj.parquet")
    idx = mc.index.intersection(cc.index)
    cols = mc.columns.intersection(cc.columns)
    return {
        "mc": mc.reindex(index=idx, columns=cols),
        "sw": sw.reindex(index=idx, columns=cols),
        "cc": cc.reindex(index=idx, columns=cols),
        "close": close.reindex(index=idx, columns=cols),
        "idx": idx, "cols": cols,
    }


def permute_rows(lab: pd.DataFrame, seed: int) -> pd.DataFrame:
    """逐行随机重排标签（组数与组大小完全不变）——安慰剂臂。"""
    rng = np.random.default_rng(seed)
    A = lab.to_numpy(dtype=object).copy()
    for i in range(A.shape[0]):
        row = A[i]
        ok = pd.notna(row)
        if ok.sum() < 2:
            continue
        vals = row[ok]
        A[i, ok] = rng.permutation(vals)
    return pd.DataFrame(A, index=lab.index, columns=lab.columns)


def fwd_returns(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "h1": close.pct_change(1, fill_method=None).shift(-1),
        "h5": close.pct_change(5, fill_method=None).shift(-5),
    }


# --------------------------------------------------------------------------- 指标
def rank_autocorr(x: pd.DataFrame, step: int = 10) -> float:
    """相邻交易日**截面秩**相关（每 ``step`` 天抽一次）——分组漂移的换手代理。"""
    dates = x.index[x.index >= pd.Timestamp(EVAL_START)]
    vals = []
    for i in range(1, len(dates), step):
        a = x.loc[dates[i - 1]]
        b = x.loc[dates[i]]
        ok = a.notna() & b.notna()
        if ok.sum() < 100:
            continue
        vals.append(float(a[ok].rank().corr(b[ok].rank())))
    return float(np.mean(vals)) if vals else float("nan")


def ic_stats(x: pd.DataFrame, fwd: pd.DataFrame, h: str) -> dict:
    ic = calc_ic_series(x.loc[EVAL_START:EVAL_END], fwd.loc[EVAL_START:EVAL_END]).dropna()
    if ic.empty:
        return {"ic": np.nan, "ic_t_nw": np.nan, "ic_icir": np.nan,
                "ic_h1": np.nan, "ic_h2": np.nan, "n_ic": 0}
    t_nw = nw_tstat(ic.values)[0]
    mid = len(ic) // 2
    return {
        f"ic_{h}": float(ic.mean()),
        f"t_{h}": float(t_nw),
        f"icir_{h}": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan,
        f"ic_{h}_h1": float(ic.iloc[:mid].mean()),
        f"ic_{h}_h2": float(ic.iloc[mid:].mean()),
        f"n_{h}": int(len(ic)),
    }


# --------------------------------------------------------------------------- 主流程
def run(factors: tuple[str, ...], placebo_k: int) -> None:
    base = load_base()
    mc, sw_raw, cc = base["mc"], base["sw"], base["cc"]
    fwd = fwd_returns(base["close"])

    # universe 对齐：两臂都只用 cc 有标签的格点
    mask_cc = cc.notna()
    sw = sw_raw.where(mask_cc)
    dropped = float(1.0 - mask_cc.to_numpy().sum() / max(sw_raw.notna().to_numpy().sum(), 1))
    print(f"格点对齐：sw 有标签 {int(sw_raw.notna().to_numpy().sum()):,} -> "
          f"保留 {int(sw.notna().to_numpy().sum()):,}（丢弃 {dropped:.1%}）")
    print(f"cc 组数 {int(cc.stack().nunique())}  "
          f"每期平均非空格数 {mask_cc.sum(axis=1).mean():.0f}")

    arms: dict[str, pd.DataFrame] = {"sw": sw, "cc": cc}
    for k in range(placebo_k):
        arms[f"rand{k}"] = permute_rows(cc, seed=1000 + k)

    rows = []
    for name in factors:
        p_raw = pd.read_parquet(DS / "panels" / f"{name}.parquet")
        p_raw = p_raw.reindex(index=mc.index, columns=mc.columns)
        p_raw = p_raw.where(mask_cc)
        print(f"\n### {name}  coverage {float(p_raw.notna().mean().mean()):.3f}")

        arm_list = [("raw", None), ("sw", sw), ("cc", cc)]
        arm_list += [(f"rand{k}", arms[f"rand{k}"]) for k in range(placebo_k)]

        for arm, lab in arm_list:
            t0 = time.time()
            if lab is None:
                x = p_raw.copy()
                x = preprocess_factor(x, market_cap_panel=None, industry_panel=None,
                                      neutralize_size=False)
            else:
                x = preprocess_factor(p_raw, market_cap_panel=mc, industry_panel=lab)
            rec = {"factor": name, "arm": arm,
                   "coverage": float(x.notna().mean().mean()),
                   "rank_autocorr": rank_autocorr(x)}
            for h in ("h1", "h5"):
                rec.update(ic_stats(x, fwd[h], h))
            rows.append(rec)
            print(f"  {arm:6s} IC_h1 {rec['ic_h1']:+.4f} (t {rec['t_h1']:+.2f})  "
                  f"IC_h5 {rec['ic_h5']:+.4f} (t {rec['t_h5']:+.2f})  "
                  f"ac {rec['rank_autocorr']:.3f}  [{time.time()-t0:.0f}s]")
            del x

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "neutralize_group_ab.csv", index=False)

    # ---- 汇总：cc vs sw，并用 rand 分布定位 ----
    summ = []
    for name in factors:
        sub = out[out["factor"] == name]
        sw_r = sub[sub["arm"] == "sw"].iloc[0]
        cc_r = sub[sub["arm"] == "cc"].iloc[0]
        rnd = sub[sub["arm"].str.startswith("rand")]
        rec = {"factor": name}
        for h in ("h1", "h5"):
            rec[f"ic_sw_{h}"] = sw_r[f"ic_{h}"]
            rec[f"ic_cc_{h}"] = cc_r[f"ic_{h}"]
            rec[f"d_ic_{h}"] = cc_r[f"ic_{h}"] - sw_r[f"ic_{h}"]
            rec[f"rand_p05_{h}"] = float(rnd[f"ic_{h}"].quantile(0.05))
            rec[f"rand_p95_{h}"] = float(rnd[f"ic_{h}"].quantile(0.95))
            rec[f"rand_mean_{h}"] = float(rnd[f"ic_{h}"].mean())
            # 「更好」= 正向因子上穿 P95 / 负向因子下穿 P05
            sgn = np.sign(sw_r[f"ic_{h}"])
            rec[f"cc_outside_{h}"] = bool(
                cc_r[f"ic_{h}"] > rec[f"rand_p95_{h}"] if sgn > 0
                else cc_r[f"ic_{h}"] < rec[f"rand_p05_{h}"])
            rec[f"half_sign_{h}"] = bool(
                np.sign(cc_r[f"ic_{h}_h1"] - sw_r[f"ic_{h}_h1"])
                == np.sign(cc_r[f"ic_{h}_h2"] - sw_r[f"ic_{h}_h2"]))
        rec["ac_sw"] = sw_r["rank_autocorr"]
        rec["ac_cc"] = cc_r["rank_autocorr"]
        rec["d_ac"] = cc_r["rank_autocorr"] - sw_r["rank_autocorr"]
        summ.append(rec)
    s = pd.DataFrame(summ)
    s.to_csv(OUT_DIR / "neutralize_group_ab_summary.csv", index=False)

    print("\n== 汇总（ΔIC = cc − sw）==")
    cols = ["factor", "ic_sw_h1", "ic_cc_h1", "d_ic_h1", "rand_p95_h1",
            "cc_outside_h1", "half_sign_h1", "d_ic_h5", "cc_outside_h5",
            "half_sign_h5", "ac_sw", "ac_cc", "d_ac"]
    print(s[cols].round(4).to_string(index=False))

    wins = s[(s["d_ic_h1"].abs() > 0) & s["cc_outside_h1"] & s["half_sign_h1"]
             & (s["ic_cc_h1"].abs() > s["ic_sw_h1"].abs())]
    print(f"\n判据全过（cc 优于 sw）：{len(wins)}/{len(s)} 个因子"
          + (f" -> {list(wins['factor'])}" if len(wins) else ""))
    print("产物：neutralize_group_ab.csv / neutralize_group_ab_summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="*", default=list(DEFAULT_FACTORS))
    ap.add_argument("--placebo", type=int, default=PLACEBO_K)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(tuple(args.factors), args.placebo)


if __name__ == "__main__":
    main()
