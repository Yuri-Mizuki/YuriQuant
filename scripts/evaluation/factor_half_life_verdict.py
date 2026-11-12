"""因子信息半衰期的**判定**（读 `factor_half_life_probe` 的 CSV，不重算 IC）。

为什么单独一个脚本
------------------
`factor_half_life_probe.py` 只负责产出原始 IC/t 值。判定需要额外的统计校正，
放在一起会让「取数」和「下结论」混在一处 —— 而本项目的历史教训是**结论必须
能被单独复算**（改口径只需重跑取数，判定逻辑不动）。

本脚本做三件事，缺一不可
------------------------
① **多重检验校正**：952 个因子各测一次，α=0.05 下纯噪声也会给出约 48 个
   「显著」。必须报 Benjamini-Hochberg FDR 后的显著数，不能只报 raw t>1.96。
② **族内去重**：`alpha191`(180) + `alpha101`(100) 占 280 个，且同族高度相关。
   把「12 个 alpha191 显著」当成 12 个独立证据会严重高估。按族汇总后看**有多少
   个独立族**给出同向结论。
③ **双向对称性检验**：若显著项里「周频更强」与「月频更强」数量相当，说明只是
   噪声在两侧对称溢出，而非存在一批「短周期因子」。这才是判「有没有原料」的关键。

判据（先定后看，写在 README 里）
--------------------------------
🚨 **主判据改成「符号检验 + 效应量」，不再是「显著因子个数」。** 首版用
「BH-FDR 后周频更强的因子数 ≥ 30 且跨 ≥5 族且周/月比 > 2:1」，实测发现两个问题：
1. **族归并 bug 让族数虚高**（`alpha360_VOLUME1..60` 被数成 60 族）⇒ 修正前那版判定
   的 `n_family_pos_sig=77` 是假的；
2. **「个数」不是正确的判据方向** —— 我们要问的是「有没有一批因子在周频显著更强」，
   而「多少个」会被「弱因子 t 值容易显著」这种机制污染（分母小 ⇒ t 大）。

现判据（全部预注册，看结果前写死）：
- **G1 方向性**：`|IC_h5| > |IC_h20|` 的因子占比必须 **显著高于 50%**（二项检验 p<0.01）。
  这是「是否存在短周期原料」的**必要条件**，且不依赖 t 值与重叠结构。
- **G2 移位数**：`d_abs` 的中位数必须 **> 0**（即典型因子在 5 日尺度更强）。
- **G3 可交易性**：FDR 后「周频更强」的因子，**族去重后**需 ≥ 5 族且覆盖 ≥ 30 个因子
  （防止只剩一两个族的变体堆砌）。
- **G4 稳健性**：以上结论在**非重叠采样**口径下方向不变（允许不显著，但符号不能反）。

- **有原料**：G1 **且** G2 **且** G3 通过。
- **无原料**：G1 或 G2 不通过（G4 只作旁证：非重叠口径功效低，不显著不作为否决）。

用法::

    python -m scripts.evaluation.factor_half_life_verdict
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "factor_half_life"

#: 判定门槛（预注册，写在 README 的「判据」节，改这里必须同步改文档）
GATES = {
    "fdr_q": 0.10,               # BH-FDR 水平（仅用于 G3 的候选池，不作主判据）
    "sign_test_p": 0.01,         # G1 二项检验显著性水平
    "min_sig_factors": 30,       # G3 族去重前的最少因子数
    "min_families": 5,           # G3 覆盖的独立族数下限
    "variance_guard_corr": 0.70,  # G0：配对差与 Δstd 的相关上限（防方差假象）
}


def factor_family(name: str) -> str:
    """把因子名映射到**独立的信号族**（用于去重，不做统计）。

    🚨 这不是「分类准确」的问题，而是「避免把同族变体当独立证据」：
    `alpha191_001` 与 `alpha191_002` 可能测同一件事，归一族是**保守**做法。
    若归族过粗（比如把全部 952 个都归成一族）会低估证据量，所以只对
    **明显同源**的做归并。

    族规则（按特异性从高到低匹配）：
    1. `alpha158_VMA5` / `alpha360_VOLUME19` → **按「集 + 算子」归族**
       （`alpha158_vma` / `alpha360_volume`）。Qlib 的 Alpha158/Alpha360 里
       `VOLUME1..60` 是同一算子的 60 个窗口变体，当成 60 份独立证据会严重高估。
       🚨 首版漏了这条：109 个「周频显著更强」因子被数成 77 族，实际远少于此。
    2. `alpha101_083` / `alpha191_171` → 两个公式化大族。
    3. `sue_q` / `sue_notice_20d` → 归到 `sue`（同一基本面的不同披露口径）。
    4. 其余去掉窗口后缀（`_20d` / `_60d`）。

    注意大小写：面板名同时存在 `alpha158_VMA5`（大写算子）与 `alpha360_volume32`
    （小写），两者都要能被规则 1 命中 ⇒ 统一先 `lower()` 再匹配。
    """
    n = name.lower()
    # 规则 1：Qlib 特征集 —— 集名 + 算子名（去掉尾部窗口数字）
    m = re.match(r"^(alpha158|alpha360)[_ ]([a-z]+?)\d+$", n)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # 规则 2：公式化 alpha 大族
    m = re.match(r"^(alpha191|alpha101)[_ ]?\d+$", n)
    if m:
        return m.group(1)
    # 规则 3：同一基本面的不同披露口径
    m = re.match(r"^([a-z]+?)_(?:q|ttm|notice|express|cs|20d|60d|5d|10d|120d)", n)
    if m:
        return m.group(1)
    # 规则 4：去掉窗口后缀
    return re.sub(r"_\d+d$", "", n)


def bh_fdr(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg：返回每个假设在 FDR=q 下是否被拒绝（布尔）。

    🚨 用**双侧** p（`2*(1-Φ(|t|))`）而不是「只看正向」：本探针要判的是
    「周频是否系统性更强」，若只对正向做校正，会把「月频更强」那批漏掉，
    从而看不到双向对称 —— 而那正是「无原料」的主要证据形态。
    """
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    rej = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return rej
    pv = p[idx]
    order = np.argsort(pv)
    m = len(pv)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = pv[order] <= thresh
    if not passed.any():
        return rej
    k = np.max(np.where(passed)[0]) + 1
    rej[idx[order[:k]]] = True
    return rej


def main() -> None:
    src = OUT / "factor_ic_by_horizon.csv"
    if not src.exists():
        raise SystemExit(f"缺 {src} —— 先跑 factor_half_life_probe")
    df = pd.read_csv(src, encoding="utf-8-sig")
    df["family"] = df["factor"].map(factor_family)
    n = len(df)
    g = GATES
    print(f"因子 {n} 个 / 独立族 {df['family'].nunique()} 个", flush=True)

    res: dict = {"n_factor": int(n), "n_family": int(df["family"].nunique())}

    # ===================================================================
    # 前置守卫：**方差假象检查**（本轮最大的坑，必须先过）
    # 若配对差与「两个期限的 IC 日度波动之差」高度线性相关，说明这个差其实是
    # 波动率差异的映射，不是分辨力差异。
    # 实测 v2 口径（逐日 |IC| 之差）corr = **+0.883**、79.2% 的因子 std_h5>std_h20
    # ⇒ 那个口径把结论从 36% 正占比翻到 66%，纯属假象。v3 定向差把它压到
    # 合理水平（见下）。
    # ===================================================================
    dstd = df["ic_std_h5"] - df["ic_std_h20"]
    res["guard_corr_d_vs_dstd"] = float(df["d_h5_h20"].corr(dstd))
    res["guard_share_std_h5_gt_h20"] = float((dstd > 0).mean())
    res["guard_variance_artifact"] = bool(
        abs(res["guard_corr_d_vs_dstd"]) > g["variance_guard_corr"])

    # ===================================================================
    # G1 方向性：定向差 > 0 的占比是否显著高于 50%？
    # 🚨 主判据。用**符号**不用 t 值 —— t 在重叠口径被高估 1.5~5 倍，
    #    在非重叠口径功效不足，两个都不可靠；符号检验不依赖重叠结构。
    # ===================================================================
    dd = df["d_h5_h20"].dropna()
    n_d = len(dd)
    n_pos = int((dd > 0).sum())
    share = n_pos / n_d if n_d else float("nan")
    bt = stats.binomtest(n_pos, n_d, 0.5) if n_d else None
    res["g1_n"] = int(n_d)
    res["g1_n_pos"] = n_pos
    res["g1_share_pos"] = float(share)
    res["g1_binom_p"] = float(bt.pvalue) if bt is not None else float("nan")
    g1_ok = bool(share > 0.5 and res["g1_binom_p"] < g["sign_test_p"])

    # ===================================================================
    # G2 移位数：d_abs 的中位数是否 > 0（典型因子在 5 日尺度更强）
    # 🚨 与 G1 互补：少数因子的大幅增强也能拉住均值，但拉不动中位数。
    # ===================================================================
    res["g2_d_median"] = float(dd.median())
    res["g2_d_mean"] = float(dd.mean())
    # p10/p90 用来看「是不是只有一撮因子在动」
    res["g2_d_p10"] = float(dd.quantile(0.10))
    res["g2_d_p90"] = float(dd.quantile(0.90))
    g2_ok = bool(res["g2_d_median"] > 0)

    # ===================================================================
    # G3 可交易性：FDR 后「周频更强」的因子，族去重后够不够多
    # 🚨 这里只能用**重叠口径**的 t（非重叠功效不足），所以 G3 只作
    #    「若 G1/G2 已通过，这些因子是否分散」的补充，不单独否决。
    # ===================================================================
    t = df["t_h5_h20"].to_numpy(dtype=float)
    tn = df["t_nonoverlap_h5_h20"].to_numpy(dtype=float)
    for tag, tv in (("overlap", t), ("nonoverlap", tn)):
        p = np.where(np.isfinite(tv), 2.0 * stats.norm.sf(np.abs(tv)), np.nan)
        rej = bh_fdr(p, g["fdr_q"])
        df[f"p_{tag}"] = p
        df[f"rej_{tag}"] = rej
        res[f"n_rej_{tag}"] = int(rej.sum())
        res[f"n_rej_pos_{tag}"] = int((rej & (tv > 0)).sum())
        res[f"n_rej_neg_{tag}"] = int((rej & (tv < 0)).sum())

    sig_pos = df[df["rej_overlap"] & (df["t_h5_h20"] > 0)]
    res["g3_n_sig_pos"] = int(len(sig_pos))
    res["g3_n_family_sig_pos"] = int(sig_pos["family"].nunique())
    # 族内代表：每族取 |t| 最大者（避免同族堆砌抬高计数）
    rep = (sig_pos.sort_values("t_h5_h20", ascending=False)
                  .drop_duplicates("family"))
    res["g3_n_rep"] = int(len(rep))
    g3_ok = bool(res["g3_n_sig_pos"] >= g["min_sig_factors"]
                 and res["g3_n_family_sig_pos"] >= g["min_families"])

    # ===================================================================
    # G4 稳健性：非重叠口径符号是否一致
    # 🚨 非重叠口径 n≈93（h=20 每 20 天取一个点），功效很低 ——
    #    实测 |t|>1.96 占比 8.5% 只略高于纯噪声 5.0%，说明它**分不清**。
    #    所以 G4 只验符号（同向即可），**不显著不作为否决**。
    #    首版误把「非重叠 FDR 只剩 1 个」读成「信号全假」，那是把功效不足
    #    当成了无效应。
    # ===================================================================
    res["g4_nonoverlap_t_median"] = float(df["t_nonoverlap_h5_h20"].median())
    res["g4_nonoverlap_mean_d"] = float(dd.mean())
    g4_ok = bool(np.sign(res["g4_nonoverlap_t_median"]) == np.sign(res["g2_d_mean"])
                 or res["g4_nonoverlap_t_median"] == 0)

    res["gate_detail"] = {"G0_no_variance_artifact": not res["guard_variance_artifact"],
                          "G1_sign": g1_ok, "G2_median": g2_ok,
                          "G3_coverage": g3_ok, "G4_robust": g4_ok}
    res["verdict"] = ("有原料（周频专用模型值得做）"
                      if (not res["guard_variance_artifact"] and g1_ok and g2_ok and g3_ok)
                      else "无原料（周频方向从因子层封死）")

    # ---- 附带：三个配对差口径并排（诊断用，证明主口径选得对）----
    for col, tag in (("d_h5_h20", "v3_directional"),
                     ("d_signed_h5_h20", "v1_signed"),
                     ("d_absdiff_h5_h20", "v2_absdiff")):
        if col in df.columns:
            s = df[col].dropna()
            res[f"caliber_{tag}_median"] = float(s.median())
            res[f"caliber_{tag}_share_pos"] = float((s > 0).mean())
            res[f"caliber_{tag}_corr_std"] = float(s.corr(dstd))

    # ---- 附带：全因子水平（共同冲击，非因子特异）----
    for h in (1, 5, 20):
        s = df[f"ic_h{h}"].dropna()
        res[f"ic_h{h}_mean"] = float(s.mean())
        res[f"ic_h{h}_median"] = float(s.median())
        res[f"ic_h{h}_absmedian"] = float(s.abs().median())

    # ---- 误导风险对照：弱因子是否更容易 t 显著（首版踩过的坑）----
    # 🚨 不要先给 df 加列再回头索引 sig_pos —— `sig_pos` 是布尔索引产生的**拷贝**，
    #    建立在加列之前，因此拿不到新列（KeyError: 'ic_abs20'）。直接对切片取 abs。
    if len(sig_pos) >= 10:
        res["weak_vs_strong_ic_abs20_median"] = {
            "sig_pos": float(sig_pos["ic_h20"].abs().median()),
            "others": float(df.loc[~df["rej_overlap"], "ic_h20"].abs().median()),
        }

    df.to_csv(OUT / "factor_ic_by_horizon_annotated.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n=== 判定 ===", flush=True)
    for k, v in res.items():
        if k in ("gate_detail",):
            continue
        print(f"  {k:34s} {v}", flush=True)
    print("\n  预注册四关:", flush=True)
    for k, v in res["gate_detail"].items():
        print(f"    {k:14s} {'PASS' if v else 'FAIL'}", flush=True)
    print(f"\n  ⇒ {res['verdict']}", flush=True)
    print(f"\n产物 → {OUT / 'verdict.json'}", flush=True)


if __name__ == "__main__":
    main()