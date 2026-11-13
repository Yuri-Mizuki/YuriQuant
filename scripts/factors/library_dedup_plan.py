"""因子库去重决策表：把「相关 → 质量 → 生产依赖」三条判据并成一张可拍板的清单。

为什么不直接用 `dup_audit` 的结论
----------------------------------
`dup_audit` 回答的是「两个因子有多像」，它的输出是**几何结构**（簇）。但
「删不删」还取决于两件它不管的事：

1. **质量**：簇内留谁？`dup_audit` 原先按 `sharpe_ls_M` 取最大，而全 A 库该列
   `0/950` 有值（`evals/` 只覆盖 26 个因子）⇒ 代表退化为字母序第一个（2026-11-12 修）。
   本脚本改用 `ic_mean_h20` + **IC 三段稳定性**双判据。
2. **生产依赖**：定版 `yqcore` 族的信号来自 `panels_neu_fundind`，每期经 DPP 从
   候选池选 50 个因子（`reports/alla_rolling_ortho920fundind/selection/y*__h*.json`）。
   一个因子若在 36 期里被选中过，删它就会改历史回测；若从未被选中，删它是
   纯库存清理。这一层 `dup_audit` 完全看不到。

分层判据（tier）——**仅供阅读，不参与删除判定**
------------------------------------------------
- ``T1 硬重复``  ρ ≥ 0.999 —— 截面秩相关接近 1，等价于同一因子的线性变换（含反向）。
- ``T2 近重复``  0.95 ≤ ρ < 0.999 —— 同族近邻（如 `alpha360_LOW20` 与 `LOW21`，
  都是 `low.shift(i)/close` 的滞后快照）。**不能只按阈值删**：残差 IC 与 IC 稳定性
  才是取舍依据。
- ``T3 同族``    0.85 ≤ ρ < 0.95 —— 同样**必须做残差检验**后才能判重。

🚨 2026-11-13 修正：此前 ``T3`` 写「只标注，不建议删」，且实现里确实以 tier 当门禁
（``tier not in ("T1","T2")`` ⇒ 直接 keep）。但叠加上「按簇遍历 + 跳过单元素簇」，
真正的后果是 **|ρ| ∈ [0.85, 0.95) 的因子对从未被检验过**——`alpha101_024` 与
`alpha191_098` 的 ρ 只有 0.8975，秩空间 ``rel_std`` 却是 7.8e-16（严格等价：
后者是前者的覆盖子集，交集上秩相关恒 1.0000000000），照样双双留着。
现在判重只在 ρ < ``CORR_TRIGGER``（0.85）处停手，tier 纯粹是报告标注。

动作（action）
--------------
- ``keep_prod``    定版当期在用 → 不动，要动必须走组合层 A/B
- ``keep_rep``     簇代表 → 保留
- ``keep_quality`` 非代表但质量显著更好（IC 与稳定性双优）→ 保留，代表该换人
- ``safe_drop``    从未进定版 + 非代表 + 质量不占优 → 可删（仍需人工过一遍）
- ``review``       其余（曾进定版、或质量接近）→ 逐个人工判断

用法::

    python -m scripts.factors.library_dedup_plan --dataset all_a_2018_2026
    python -m scripts.factors.library_dedup_plan --thresh 0.95 --hard 0.999
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.factors.dup_criteria import (  # noqa: E402
    CORR_TRIGGER,
    REL_STD_DEGENERATE,
    REL_STD_TINY,
    RESID_IC_NEGLIGIBLE,
    should_check,
)

log = setup_logging("dedup_plan")

SELECTION_DIR = ROOT / "reports" / "alla_rolling_ortho920fundind" / "selection"
OUT_DIR = ROOT / "reports" / "library_governance"
HS = ("h1", "h5", "h10", "h20")


# ---------------------------------------------------------------- 生产依赖
def production_dependence() -> tuple[dict[str, int], set[str], dict]:
    """定版滚动 selection：每个因子被选中的期数 + 最后一期在用集合。

    36 期 = 9 年 × 4 horizon。选中期数直接衡量「这个因子在定版里出现过几次」，
    比任何代理指标都硬——它是 `panels_neu_fundind → DPP → pred` 链条的实际输出。
    """
    sets: dict[tuple[str, str], list[str]] = {}
    for f in sorted(SELECTION_DIR.glob("y20*__h*.json")):
        if f.name.endswith(".fp.json"):
            continue
        y, h = f.stem.split("__")
        sets[(y, h)] = json.loads(f.read_text(encoding="utf-8"))
    if not sets:
        log.warning("读不到 selection 目录：%s，生产依赖列全记 0", SELECTION_DIR)
        return {}, set(), {"n_periods": 0, "years": []}
    years = sorted({y for y, _ in sets})
    cnt: dict[str, int] = {}
    for lst in sets.values():
        for n in set(lst):
            cnt[n] = cnt.get(n, 0) + 1
    cur = set().union(*[set(sets[(years[-1], h)]) for h in HS if (years[-1], h) in sets])
    return cnt, cur, {"n_periods": len(sets), "years": years}


# ---------------------------------------------------------------- IC 稳定性
def ic_stability(names: list[str], h: int = 20, seg: int = 3) -> pd.DataFrame:
    """IC 序列的三段均值 + 全期 NW t。

    为什么不用单期均值：`ic_mean_h20` 是一个数，它可能是「三段都 +0.03」也可能是
    「+0.09 / 0 / −0.03」—— 后者在样本外很容易翻面。去重时留谁，稳定性比均值重要。

    分段按**交易日**等分（不按自然时间），因为不同因子的 IC 有效区间一致（同一份
    `ic_h{h}.parquet`）。
    """
    from stats.robust_stats import nw_tstat

    p = LIB.ic_path(h)
    if not p.exists():
        log.warning("缺 IC 序列 %s，稳定性列记 NaN", p)
        return pd.DataFrame(index=names, columns=["ic_all", "t_nw", "seg_neg", "seg_min"])
    ic = pd.read_parquet(p)
    have = [n for n in names if n in ic.columns]
    sub = ic[have]
    n = len(sub)
    bounds = [int(round(n * i / seg)) for i in range(seg + 1)]
    rows = {}
    for nm in have:
        s = sub[nm].dropna()
        if len(s) < 60:
            continue
        mu = float(s.mean())
        t = float(nw_tstat(s.to_numpy())[0])
        segs = [float(sub[nm].iloc[bounds[i]:bounds[i + 1]].mean()) for i in range(seg)]
        rows[nm] = {
            "ic_all": mu, "t_nw": t,
            # 三段里与全期反号的段数：0 最稳，2 以上基本不可信
            "seg_neg": int(sum(1 for x in segs if np.isfinite(x) and mu * x < 0)),
            "seg_min": float(np.nanmin(segs)) if segs else np.nan,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out = out.reindex(names)
    return out


# ---------------------------------------------------------------- 主流程
class LIB:  # 轻量命名空间，避免把 FactorLibrary 的重依赖拉进来
    @staticmethod
    def _root() -> Path:
        """库根走 ``Config.paths()``，不硬编码盘符（换机器会读旧副本）。"""
        from config import Config

        return Path(str(Config.paths()["factor_library_root"]))

    @classmethod
    def base(cls, ds: str) -> Path:
        return cls._root() / ds

    @classmethod
    def ic_path(cls, h: int) -> Path:
        return cls.base(CFG["dataset"]) / f"ic_h{h}.parquet"


CFG = {"dataset": "all_a_2018_2026"}

#: 延迟构造的 FactorLibrary 句柄（``build_decision`` 里按 dataset 填）。
#: 只用于按名字读面板算秩空间增量——不走它的 registry（退登记后取不到行）。
LIB_RT: dict = {}


def _read_panel(base: Path, name: str) -> pd.DataFrame | None:
    """按文件名读因子面板，**绕过 registry**。

    退登记后 ``registry.csv`` 已无该行 ⇒ ``lib.get_panel(name)`` 返回 None。
    任何核验/决策脚本都必须走这里，否则「取不到面板」会被下游当成
    「无增量」（2026-11-12 实测 8/8 假通过）。
    面板有两种命名：``<name>.parquet``（builder/merge_outputs）与
    ``<slug>.parquet``（register，slug=md5[:12]）。
    """
    from research.factor_library import name_slug

    for p in (base / "panels" / f"{name}.parquet",
              base / "panels" / f"{name_slug(name)}.parquet"):
        if p.exists():
            return pd.read_parquet(p)
    return None


def _rank_increment(name: str, cover: str, lib=None, stride: int = 1) -> dict:
    """对**覆盖者**算秩空间增量；返回 ``has_inc`` / ``rel_std`` / ``ic`` / ``t``。

    ⚠️ **判定逻辑不在这里** —— 阈值与分级全部在 ``scripts.factors.dup_criteria``
    （唯一真源）。本函数只负责读面板、调 ``rank_resid_stats``、把结果搬成扁平 dict。
    改口径请改 ``dup_criteria``，不要在调用点另抄一份阈值。

    ``has_inc=True`` 的语义是「**不能因为这个理由判它可删**」，三种情况都会给 True：
    等价/微小增量以外的真增量、以及**任何测不成的情形**（取不到面板、截面不足）。
    后者绝不写成「无增量」（2026-11-12 的 8/8 假通过事故）。

    ``stride``: 日期抽样步长。精确退化是**严格等式**，抽 1/20 天足以判定
    （2511 天全算要 25 分钟）。非退化情形也只用于 `rel_std`/`ic` 的均值，
    抽样不改变结论方向；判定「有独立增量」时一律保守放行。
    """
    from scripts.factors.dup_audit import rank_resid_stats
    from scripts.factors.dup_criteria import classify, explain

    if lib is None or not name or not cover:
        return {"has_inc": True, "rel_std": float("nan"), "ic": float("nan"),
                "t": float("nan"), "n_days": 0, "n_span": 0, "note": "无法检验"}
    base = Path(lib.base_root) / CFG["dataset"]
    px, pc = _read_panel(base, name), _read_panel(base, cover)
    if px is None or pc is None:
        return {"has_inc": True, "rel_std": float("nan"), "ic": float("nan"),
                "t": float("nan"), "n_days": 0, "n_span": 0,
                "note": "面板缺失：" + ("x" if px is None else "cover")}
    st = rank_resid_stats(px, {cover: pc}, lib.default_returns_panel(),
                          stride=stride)
    # 判定逻辑全部下沉到 ``dup_criteria``（唯一真源）。这里只做字段搬运。
    v = classify(st)
    return {"has_inc": not v.removable, "rel_std": v.rel_std, "ic": v.resid_ic,
            "t": v.resid_t, "n_days": v.n_days, "n_span": v.n_span,
            "note": v.level if v.tested else f"{v.level}: {v.reason}",
            "_verdict": v, "_criteria": explain()}


def load_corr(dataset: str) -> tuple[np.ndarray, list[str]]:
    f = LIB.base(dataset) / "dup_matrix.npz"
    if not f.exists():
        raise SystemExit(f"没有相关矩阵：{f}\n先跑 `python -m scripts.factors.dup_audit "
                         f"--dataset {dataset}`")
    z = np.load(f)
    return z["corr"], [str(n) for n in z["names"]]


def greedy_order(df: pd.DataFrame, n_bucket: int = 10) -> list[int]:
    """贪心遍历顺序：**质量分桶降序 → 桶内覆盖率降序**。

    这只是**处理顺序**，不是「留谁」的判据。等价对里先被访问者会成为保留者，
    所以顺序会影响结果；但「等价时优先留覆盖广的」由 :func:`build_decision` 里的
    **覆盖交换**兜底，不依赖本函数把两者分到同一桶。

    ⚠️ 实测本函数**做不到**把质量相近但跨桶的等价对排对：
    `alpha191_098`（quality 0.0437 / 覆盖 59.4%）落第 5 桶，
    `alpha101_024`（0.0265 / 覆盖 70.6%）落第 3 桶，跨桶后仍按质量排 ⇒
    覆盖更窄的后者优先。这是本函数的能力边界，不是 bug——所以交换逻辑必须存在。

    质量分桶而非纯降序，是为了让「质量可比」的因子在同桶内由覆盖率排序，
    避免覆盖率彻底压过质量（一个 `|IC|≈0` 但覆盖 100% 的因子不该优先于真信号）。
    """
    q = pd.to_numeric(df["quality"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cov = pd.to_numeric(df["cover"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    try:
        bucket = pd.qcut(q, n_bucket, labels=False, duplicates="drop")
        bucket = pd.Series(bucket).fillna(0).to_numpy(dtype=float)
    except (ValueError, TypeError):
        # 极端分布（>90% 同值）时 qcut 无法分桶，退回百分位秩
        bucket = pd.Series(q).rank(pct=True).to_numpy(dtype=float) * n_bucket
    # lexsort 末键为主键：先 -bucket（= 质量降序），再 -cov（= 覆盖降序）
    return [int(i) for i in np.lexsort((-cov, -bucket))]


def _try_cover_swap(df: pd.DataFrame, i: int, cover_name: str,
                    cover_by: np.ndarray, pos: dict[str, int],
                    keep: set[str], *, lib=None,
                    stride: int = 1) -> int | None:
    """判出「i 被 ``cover_name`` 覆盖」之后，决定是否**反过来**留 i 删覆盖者。

    只在「i 的覆盖严格更广 + 覆盖者不是必须保留的角色 + 反向也等价」时交换，
    否则返回 None（维持「删 i」的原判，即保守）。

    为什么需要它
    ------------
    判据只说「这两个等价」，不说「留哪个」。而遍历顺序由质量驱动，
    质量高但覆盖窄的因子会先到、成为保留者，把覆盖更广的信息丢掉。
    实测 `alpha191_098`（覆盖 59.4%）会取代 `alpha101_024`（70.6%），
    而前者是后者的严格覆盖子集。

    三条硬约束
    ----------
    1. 覆盖者不能是 ``keep_prod`` / ``keep_rep``（定版在用 / 簇代表）。
    2. 覆盖者 ``n_sel`` 必须为 0（否则删它会让历史回测不可复现）。
    3. **反向也必须等价**：i 对 ``cover_name`` 回归仍须退化。
       否则是「窄的被宽的包含，但宽的有额外信息」——这时留宽的对，
       但也不能删窄的（窄的确实是宽的单调变换吗？不一定，所以**不做交换**，
       维持原判删 i 才是错的，应改判保留）。第 3 条区分这两种情况。

    Returns:
        可交换则返回覆盖者的行号 ``j``；否则 None。
    """
    from scripts.factors.dup_audit import rank_resid_stats
    from scripts.factors.dup_criteria import classify

    j = pos.get(cover_name)
    if j is None:
        return None
    if keep and cover_name not in keep:      # 覆盖者已不在保留集，无从交换
        return None
    if df.at[j, "in_prod_now"] or df.at[j, "is_rep"]:
        return None
    if int(df.at[j, "n_sel"]) != 0:
        return None
    # 覆盖必须**严格更广**，否则交换无收益
    if not (float(df.at[i, "cover"]) > float(df.at[j, "cover"]) + 1e-9):
        return None
    # 覆盖者不能已在别处被判可删（否则 join 到它身上等于删两个）
    if cover_by[j] != "":
        return None
    # 反向等价检验：留宽的 i，看窄的 cover_name 是否被 i 完全解释
    if lib is None:
        return None
    base = Path(lib.base_root) / CFG["dataset"]
    wide_name = str(df.at[i, "name"])
    px = _read_panel(base, cover_name)
    pc = _read_panel(base, wide_name)
    if px is None or pc is None:
        return None
    st = rank_resid_stats(px, {wide_name: pc}, lib.default_returns_panel(),
                          stride=stride)
    v = classify(st)
    return j if v.is_equivalent else None


def build_decision(dataset: str, thresh: float, hard: float,
                   rank_stride: int = 1, history_is_gate: bool = False) -> pd.DataFrame:
    """出决策表。

    Args:
        history_is_gate: 「曾进定版 selection」是否**当成硬闸**（即阻止删除）。

            ``False``（**默认，2026-11-13 起**）：只保护**当期在用**（``keep_prod``）。
            历史选中期数 ``n_sel`` 降级为报告列，不再阻止删除、也不再充当覆盖源。
            ``True``：旧行为，``n_sel > 0`` 一律判 ``review``。

            为什么默认放开
            --------------
            核实三件事后确认 ``n_sel`` **不保护「已定方案」**：

            1. 出榜链读**静态** selection JSON（`alla_daily_rank.py::load_selection`），
               不读 registry ⇒ 退登记只减少候选，不会改已落盘的清单。
            2. 重建链**明确不重建 selection**（`rebuild_panel_library.py:37-38`
               「不重建」）⇒ 每天 06:00 不会因此改历史。
            3. 当期四 horizon 各 50 个 ∩ 台账 = 0（实测）。

            所以 ``n_sel`` 唯一保护的是「将来主动 ``--stage select`` 重跑旧年份时，
            结果会和当初不同」——那是**重跑问题**，不是「污染已定方案」。
            真正要护的只有当期在用的 116 个。
    """
    from scripts.factors.dup_audit import clusters, neighbors

    corr, names = load_corr(dataset)
    reg = pd.read_csv(LIB.base(dataset) / "registry.csv").set_index("name")
    reg = reg.reindex(names)

    # 秩空间增量要读面板，需要一个 FactorLibrary 句柄（只用于取面板与收益，
    # 不读它的 registry —— 退登记后 registry 里没有那些行）。
    if not LIB_RT.get("lib"):
        from research.factor_library import FactorLibrary
        LIB_RT["lib"] = FactorLibrary(dataset=dataset)

    cnt, cur, prod_meta = production_dependence()
    log.info("定版：%d 期，历史并集 %d，当期在用 %d",
             prod_meta["n_periods"], len(cnt), len(cur))

    # 🚨 `abs_corr=True` 是必需的（2026-11-12 复核）。默认按 ρ 排序会把 ρ=−1 的
    # 镜像对判成「不相关」：`alpha158_SUMN30` vs `SUMP30` 精确 −1.000，但它的
    # 最近邻按正相关是另一个 ρ=+0.93 的因子 ⇒ max_corr=0.93 ⇒ 落进 T3/独立。
    # 实测有 23 个 |ρ|≥0.999 因此未被标成 T1（其中 17 个是纯镜像）。
    nb = neighbors(corr, names, top_k=1, abs_corr=True).set_index("name")
    cl = clusters(corr, names, thresh).set_index("name")

    stab = ic_stability(names, h=20)
    icm = pd.to_numeric(reg.get("ic_mean_h20"), errors="coerce")
    resid_ic = pd.to_numeric(reg.get("resid_ic"), errors="coerce")
    resid_t = pd.to_numeric(reg.get("resid_t_nw"), errors="coerce")

    df = pd.DataFrame({
        "name": names,
        "cluster": cl["cluster"].reindex(names).to_numpy(),
        # 🚨 这一列存的是 **|ρ|**（abs_corr=True）。历史版本存的是最大**正**相关，
        # 导致 ρ=−1 的镜像对显示成 0.4~0.9 并被判成独立/同族。
        "max_corr": nb["abs_corr"].reindex(names).to_numpy(),
        "nearest": nb["top"].reindex(names).to_numpy(),
        "ic_h20": icm.to_numpy(),
        "ic_seg_neg": stab["seg_neg"].reindex(names).to_numpy(),
        "ic_t_nw": stab["t_nw"].reindex(names).to_numpy(),
        "resid_ic": resid_ic.to_numpy(),
        "resid_t": resid_t.to_numpy(),
        "cover": pd.to_numeric(reg.get("coverage"), errors="coerce").to_numpy(),
        "family": reg.get("family").astype(str).to_numpy(),
        "n_sel": [int(cnt.get(n, 0)) for n in names],
        "in_prod_now": [n in cur for n in names],
    })
    df["n_periods"] = prod_meta["n_periods"]
    # 分档阈值与命名统一由 dup_criteria 提供（CLI 的 --hard/--thresh 只能收紧，
    # 不能另起一套命名）。tier **只是标注**，不参与删除判定。
    from scripts.factors.dup_criteria import tier_of

    df["tier"] = [tier_of(float(c)) for c in df["max_corr"].to_numpy()]
    # 秩空间增量的四列（只有进到覆盖判定的行才填；其余留 NaN）
    for c in ("rank_rel_std", "rank_resid_ic", "rank_resid_t", "rank_n_days",
              "rank_note"):
        df[c] = np.nan if c != "rank_note" else ""

    # ---- 簇代表：质量优先（IC 均值 + 稳定性），不看 sharpe（该列空）
    size = df.groupby("cluster")["name"].transform("size")
    df["cluster_size"] = size
    # 质量分：|IC| 打折 —— 三段反号 2 次以上视为不可信，直接砍半
    q = df["ic_h20"].abs().fillna(0.0)
    q = q * np.where(df["ic_seg_neg"].fillna(3) >= 2, 0.5, 1.0)
    q = q * np.where(df["ic_seg_neg"].fillna(3) >= 1, 0.8, 1.0)
    df["quality"] = q
    # 簇内最高质量者 = 代表（只在 size≥2 的簇里有意义）
    rep_idx = df.groupby("cluster")["quality"].idxmax()
    is_rep = pd.Series(False, index=df.index)
    is_rep.loc[rep_idx.to_numpy()] = True
    df["is_rep"] = is_rep.to_numpy() & (df["cluster_size"] >= 2)

    # ---- 动作分层：贪心最大独立集
    # ⚠️ 不能用「与最近邻的 ρ ≥ 阈值」当删除判据——最近邻自己也可能被删。
    # 实测反例：`alpha158_ROC30` 与 `alpha360_CLOSE30` 互为最近邻（ρ=1.000），
    # 两个都被判成 safe_drop ⇒ 该簇这一支被掏空，留下的代表与它们并不等价。
    # 正确判据是**覆盖**：删 x 的前提是 ∃ 一个**确定保留**的 y 使 ρ(x,y) ≥ 触发线。
    #
    # 🚨 2026-11-13：循环**不再按簇分组**。原先 `for _cid, g in groupby("cluster")`
    # + `if len(g) < 2: continue`，而簇是用 **thresh(0.95) 完全链接**切出来的 ⇒
    # 凡是 |ρ| ∈ [CORR_TRIGGER=0.85, 0.95) 的因子对按定义就进不了同一个簇，
    # 单元素簇直接 continue，**从来不做残差检验**。实测 631 个因子 max_corr≥0.85，
    # 只有 280 个真测过；179 个「单元素簇但 max_corr≥0.85」被静默跳过
    # （如 `alpha101_024` / `alpha191_098`，ρ=0.8975 但 rel_std=7.8e-16）。
    # 即 CORR_TRIGGER=0.85 从未生效，判重实际只在 ρ≥0.95 的区域内运行。
    # 现改为**全局**按 `greedy_order` 遍历，覆盖者从整个 keep 集里找。
    pos = {n: i for i, n in enumerate(names)}
    n_row = len(df)
    action = np.full(n_row, "keep", dtype=object)
    cover_by = np.full(n_row, "", dtype=object)
    cover_rho = np.full(n_row, np.nan, dtype=float)

    def _drop_verb(idx: int) -> str:
        """判出「可删」之后的动作名。**两处调用点共用，防口径分叉。**

        若仍把历史选中期数当硬闸（``history_is_gate=True``）：进过 selection 的
        一律降为 ``review``（人工过）。
        否则只有「当期在用」需要保护（那些人已经在 ``keep_prod`` 里，走不到这里）
        ⇒ 直接 ``safe_drop``。
        """
        if history_is_gate and int(df.at[idx, "n_sel"]) != 0:
            return "review"
        return "safe_drop"
    for i, nm in enumerate(names):
        if df.at[i, "in_prod_now"]:
            action[i] = "keep_prod"
        elif df.at[i, "is_rep"]:
            action[i] = "keep_rep"
    # 保留集 = 覆盖源。默认只含「不会被删的角色」：
    #   当期在用（keep_prod）+ 簇代表（keep_rep）
    # 🚨 2026-11-13：默认**不再**把 ``n_sel > 0`` 塞进来。它有两个副作用，都过头了：
    #   ① 自己不能被删（15 个 review 里有 12 个是 rel_std<1e-3 的严格等价）；
    #   ② 充当覆盖源 ⇒ 它保护了「被它覆盖的人」，连锁让删除数远低于应有值。
    # ``history_is_gate=True`` 时恢复旧行为（仅用于对照/回退）。
    keep = set(df.loc[df["is_rep"], "name"])
    if history_is_gate:
        keep |= set(df.loc[df["n_sel"] > 0, "name"])
    for i in greedy_order(df):
        if action[i] in ("keep_prod", "keep_rep"):
            continue
        # 触发线用 ``dup_criteria.CORR_TRIGGER``（唯一真源），**不再**用簇大小或
        # tier 分档当门禁。tier 只是报告标注（见 ``dup_criteria.tier_of``）。
        # T3 档（0.85≤ρ<0.95）在 2026-11-13 后**整档不检验**（触发线 = 0.95），
        # 依据是「补测 156 个只 1 个真等价」的产出比，见 ``dup_criteria`` 注释。
        if not should_check(float(df.at[i, "max_corr"])):
            action[i] = "keep"
            keep.add(names[i])
            df.at[i, "rank_note"] = (
                "T3 同族未达检验线" if float(df.at[i, "max_corr"]) >= 0.85
                else "独立")
            continue
        best_rho, best_nm = -1.0, ""
        for k in keep:
            if k == names[i]:
                continue
            r = abs(float(corr[pos[names[i]], pos[k]]))
            if r > best_rho:
                best_rho, best_nm = r, k
        # ---- 增量判据（2026-11-12 改口径，见 dedup_execution.md §8.1②③）
        #
        # 旧判据用 registry 的 ``resid_ic``/``resid_t_nw``，有两个硬伤：
        #
        # ① **对错了因子**：那两列是对 ``dup_top``（最近邻）算的，不是对这里的
        #    ``best_nm``（覆盖者）。实测 129 个退登记里 72 个两者不同、67 个的
        #    ``dup_top`` 自身已被退登记 ⇒ 判据基本空转。
        # ② **在原始值空间算**：纯单调变换会被判成「有增量」。
        #    ``alpha158_OPEN0`` 是 ``alpha101_033`` 的截面 rank（2511 天秩相关恒
        #    1.000000），raw OLS 却给出 t=−7.40。既然生产链消费的是截面秩/z，
        #    原始值的非线性差异进不了模型，那个 t 是假的。
        #
        # 新判据在**秩空间**、对**覆盖者本身**做：
        #   ``rel_std`` = 残差秩标准差 / 因变量秩标准差，
        #   ``resid_ic`` = 残差秩对覆盖者的未来收益 Rank IC。
        #   · ``rel_std < 1e-3``      ⇒ 秩空间退化（等价/镜像），无增量，可删
        #   · ``rel_std < 0.05`` 且 ``|resid_ic| < 0.02`` ⇒ 微小增量，可删
        #   · 其余                    ⇒ 有独立增量，必须留
        # 控制集只放覆盖者一个（不是整个 keep 集）：控制得越少残差越大，
        # **偏向保留**——宁可漏删，不可误删。
        inc = _rank_increment(name=names[i], cover=best_nm,
                              lib=LIB_RT.get("lib"), stride=rank_stride)
        if best_rho >= CORR_TRIGGER and not inc["has_inc"]:
            # ---- 覆盖交换（2026-11-13 新增）
            #
            # 判出「i 可被 best_nm 覆盖」之后还有一问：**该留哪个？**
            # 判据本身只回答「等价」，不回答方向。若 best_nm 自己的覆盖更窄、
            # 且它**可以反被 i 覆盖**，那么留 best_nm 就是用窄的替掉宽的。
            #
            # 实测 `alpha101_024`（覆盖 70.6%）与 `alpha191_098`（覆盖 59.4%）：
            # 后者是前者的**严格覆盖子集**（交集上秩相关恒 1.0000000000），
            # 却因 `ic_h20` 更高（+0.0437 vs +0.0265）在遍历中先到、成为保留者。
            # 那个 IC 差异是股票池不同造成的，不是信号差异。
            #
            # 边界：只做「双向等价」的交换——i 对 best_nm 回归也必须退化，
            # 否则是**单向子集**（窄的被宽的包含但宽的另有信息），此时留宽的
            # 才对（i 确实覆盖了 best_nm 的全部），但也要求 best_nm 不是簇代表
            # 或定版在用。不满足就维持原判，保守留旧的。
            swap = _try_cover_swap(
                df, i, best_nm, cover_by, pos, keep,
                lib=LIB_RT.get("lib"), stride=rank_stride,
            )
            if swap is not None:
                j = swap
                # 把 best_nm 改判为可删，i 变成保留者
                action[j] = _drop_verb(j)
                cover_by[j], cover_rho[j] = names[i], best_rho
                keep.discard(names[j])
                keep.add(names[i])
                action[i] = "keep"
                df.at[j, "rank_rel_std"] = inc["rel_std"]
                df.at[j, "rank_resid_ic"] = inc["ic"]
                df.at[j, "rank_resid_t"] = inc["t"]
                df.at[j, "rank_n_days"] = inc["n_days"]
                df.at[j, "rank_note"] = "覆盖交换：" + inc.get("note", "")
                df.at[i, "rank_rel_std"] = inc["rel_std"]
                df.at[i, "rank_resid_ic"] = inc["ic"]
                df.at[i, "rank_resid_t"] = inc["t"]
                df.at[i, "rank_n_days"] = inc["n_days"]
                df.at[i, "rank_note"] = (
                    f"覆盖交换后保留（覆盖 {df.at[i, 'cover']:.3f} ≥ "
                    f"{df.at[j, 'cover']:.3f}）")
                continue
            action[i] = _drop_verb(i)
            cover_by[i], cover_rho[i] = best_nm, best_rho
        else:
            # 无覆盖者、或秩空间仍有独立增量 ⇒ 必须留
            action[i] = "keep"
            keep.add(names[i])
        df.at[i, "rank_rel_std"] = inc["rel_std"]
        df.at[i, "rank_resid_ic"] = inc["ic"]
        df.at[i, "rank_resid_t"] = inc["t"]
        df.at[i, "rank_n_days"] = inc["n_days"]
        df.at[i, "rank_note"] = inc.get("note", "")
    df["action"] = action
    df["cover_by"] = cover_by
    df["cover_rho"] = cover_rho
    # 自检：safe_drop 必须都带覆盖者（否则就是「无人覆盖却判可删」的漏网）
    bad = df[(df["action"] == "safe_drop") & (df["cover_by"] == "")]
    if len(bad):
        log.error("自检失败：%d 个 safe_drop 没有覆盖者，改为 keep：%s",
                  len(bad), bad["name"].head(5).tolist())
        df.loc[bad.index, "action"] = "keep"
    return df.sort_values(["tier", "cluster", "quality"], ascending=[True, True, False])


def write_report(df: pd.DataFrame, dataset: str, thresh: float, hard: float) -> None:
    n = len(df)
    vc = df["action"].value_counts()
    tiers = df["tier"].value_counts()
    sel = df[df["n_sel"] > 0]
    lines = [
        "# 因子库去重决策表",
        "",
        f"> `{dataset}` · 因子 {n} 个 · 相关阈值 {hard:.3f}（硬）/ {thresh:.2f}（近）"
        f" · 定版期数 {int(df['n_periods'].iloc[0]) if n else 0}",
        "",
        "## 〇 结论先行",
        "",
        f"1. 定版历史用过 **{len(sel)}** 个（当期在用 **{int(df['in_prod_now'].sum())}** 个），"
        f"从未进过 selection 的 **{n - len(sel)}** 个。",
        f"2. 按相关性分层：T1 硬重复 {int(tiers.get('T1硬重复', 0))} · "
        f"T2 近重复 {int(tiers.get('T2近重复', 0))} · T3 同族 {int(tiers.get('T3同族', 0))} · "
        f"独立 {int(tiers.get('独立', 0))}。",
        f"3. 动作建议：可删（safe_drop）**{int(vc.get('safe_drop', 0))}** 个 · "
        f"待人工（review）**{int(vc.get('review', 0))}** 个 · "
        f"保留 {int(vc.get('keep_prod', 0)) + int(vc.get('keep_rep', 0)) + int(vc.get('keep', 0))} 个。",
        "",
        "> **本表不删任何东西。** 删除清单需人工过一遍后，走快照 + 回收站流程。",
        "",
        "## 一、动作 × 是否进过定版",
        "",
        "| 动作 | 从未进 selection | 曾进 selection | 合计 |",
        "|---|---:|---:|---:|",
    ]
    for a in ["safe_drop", "review", "keep_prod", "keep_rep", "keep"]:
        if a not in vc:
            continue
        sub = df[df["action"] == a]
        lines.append(f"| `{a}` | {int((sub['n_sel'] == 0).sum())} | "
                     f"{int((sub['n_sel'] > 0).sum())} | {len(sub)} |")
    lines += [
        "",
        "## 二、可删清单（safe_drop，按覆盖相关度降序，前 40）",
        "",
        "判据（**贪心最大独立集**）：删 x 的前提是存在一个**确定保留**的因子 y，使 "
        f"|ρ(x,y)| ≥ {CORR_TRIGGER:.2f}（仅为**触发检验**的门槛，不是判定门槛），"
        "**且 x 对 y 的秩空间残差为 0 或微小到没有可用信号**"
        f"（`rel_std < {REL_STD_DEGENERATE:g}`，或 `rel_std < {REL_STD_TINY:g}` 且 "
        f"`|resid_ic| < {RESID_IC_NEGLIGIBLE:g}`）；并且 x 从未进定版 selection、不是簇代表。"
        "「确定保留」= 簇代表 ∪ 定版用过的 ∪ 先被判定要留的。"
        "遍历顺序 = **质量分桶降序 → 桶内覆盖率降序**（等价对里先访问者成为保留者）。",
        "",
        "> 🔑 **判「等价」只看秩空间残差是否为 0，`ρ ≥ 0.95` 什么也不算。**"
        "本库 `|ρ|≥0.95` 有 2554 对、涉及 1142/950 个因子，拿它当等价等于不筛。"
        "实测 `alpha158_OPEN0` 是 `alpha101_033` 的截面 rank（2511 天秩相关恒 1.000000），"
        "原始值 OLS 却给 t=−7.40 —— 那个 t 是伪增量。",
        "",
        "> 🚨 **本表的触发线是 "
        f"{CORR_TRIGGER:.2f}，不是聚类阈值 {thresh:.2f}。**"
        "簇只用于报告分组与「代表」定义，**不参与删除判定**。"
        "2026-11-13 前的版本按簇遍历且跳过单元素簇，而簇是用 "
        f"{thresh:.2f} 切出来的 ⇒ |ρ| ∈ [{CORR_TRIGGER:.2f}, {thresh:.2f}) 的因子对"
        "**永远不进同一簇、永远不做残差检验**，`CORR_TRIGGER` 形同虚设"
        "（实测 631 个因子 max_corr≥0.85，仅 280 个真测过）。现已改为全局遍历。",
        "",
        "> ⚠️ 表中 ρ 为**绝对值**，含反向重复。例如 `alpha158_SUMD10` 与 `SUMP10` 的"
        "真实 ρ = **−1.000**（涨量 vs 跌量，镜像），表中显示 1.000；这类只留一个并"
        "保留其 IC 符号即可，删掉另一个不损失信息。",
        "",
        "| 因子 | 覆盖者 | ρ | 簇大小 | 残差 IC | 残差 t | IC_h20 | 家族 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    sd = df[df["action"] == "safe_drop"].sort_values("cover_rho", ascending=False)
    for _, r in sd.head(40).iterrows():
        lines.append(f"| `{r['name']}` | `{r['cover_by']}` | {r['cover_rho']:.3f} | "
                     f"{int(r['cluster_size'])} | {r['resid_ic']:+.4f} | "
                     f"{r['resid_t']:+.2f} | {r['ic_h20']:+.4f} | {r['family']} |")
    if len(sd) > 40:
        lines.append(f"| … | 其余 {len(sd) - 40} 个见 `dedup_decision.csv` | | | | | |")

    lines += [
        "",
        "## 三、需人工判断（review：高冗余，但定版用过）",
        "",
        "这些因子与某个保留因子 ρ ≥ 阈值，但**在 36 期 selection 里被选中过**（当期未必用）。"
        "删它们会让**对应年份的历史回测不可复现**——不是收益高低的问题，是复现性问题。"
        "要动必须重跑那几年的 `--stage select` 并记录差异。",
        "",
        "| 因子 | 覆盖者 | ρ | 定版选中期数 | IC_h20 |",
        "|---|---|---:|---:|---:|",
    ]
    rv = df[df["action"] == "review"].sort_values("n_sel", ascending=False)
    for _, r in rv.head(30).iterrows():
        lines.append(f"| `{r['name']}` | `{r['cover_by']}` | {r['cover_rho']:.3f} | "
                     f"{int(r['n_sel'])} | {r['ic_h20']:+.4f} |")
    if len(rv) > 30:
        lines.append(f"| … | 其余 {len(rv) - 30} 个见 CSV | | | |")

    lines += [
        "",
        "## 四、口径与边界",
        "",
        f"1. 相关来自 `dup_matrix.npz`：{200} 日 × {3500} 只抽样、逐日截面 Spearman、"
        "NaN 记 0 ⇒ **偏低估**（漏报不误报）。",
        "2. 质量分 = `|ic_mean_h20|` 打折（三段反号 1 次 ×0.8、≥2 次 ×0.5）。"
        "**不用 `sharpe_ls_M`**：该列全库 0/950 有值，用它选代表会退化成字母序。",
        "3. 「定版在用」= `reports/alla_rolling_ortho920fundind/selection/y*__h*.json`，"
        "即 `panel_source: panels_neu_fundind` 那条线（`yqcore` 族）。"
        "**其他档位（solo/focus）同源同信号，一并覆盖**。",
        "4. 相关是**截面**口径；两个因子截面不相关但多空收益时序同步的风险同质，"
        "本表测不到（`evals/` 仅 26 个因子有序列）。",
        "5. `safe_drop` 不等价于「删了没影响」：从未进 selection 只说明**这套 DPP 没选它**，"
        "不排除换配置后被选中。清理前仍要留清单归档。",
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dedup_plan.md").write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(OUT_DIR / "dedup_decision.csv", index=False, encoding="utf-8-sig")
    log.info("报告 → %s（CSV %d 行）", OUT_DIR / "dedup_plan.md", len(df))


def main() -> None:
    ap = argparse.ArgumentParser(description="因子库去重决策表")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--thresh", type=float, default=0.95, help="近重复阈值")
    ap.add_argument("--hard", type=float, default=0.999, help="硬重复（换皮）阈值")
    ap.add_argument("--rank-stride", type=int, default=10,
                    help="秩空间增量的日期抽样步长（默认 10；精确退化不受影响）")
    ap.add_argument("--history-is-gate", action="store_true",
                    help="把「曾进定版 selection」当硬闸（旧行为，默认关闭）。"
                         "默认只保护当期在用——历史 selection 是静态 JSON，"
                         "退登记不改已落盘清单，只影响「主动重跑旧年份」")
    args = ap.parse_args()
    CFG["dataset"] = args.dataset

    df = build_decision(args.dataset, args.thresh, args.hard, args.rank_stride,
                        history_is_gate=args.history_is_gate)
    write_report(df, args.dataset, args.thresh, args.hard)

    vc = df["action"].value_counts()
    print("\n=== 去重决策 ===")
    print(f"因子 {len(df)} 个；定版用过 {(df['n_sel'] > 0).sum()}，"
          f"当期在用 {int(df['in_prod_now'].sum())}")
    for a in ["safe_drop", "review", "keep_prod", "keep_rep", "keep"]:
        if a in vc:
            print(f"  {a:14s} {vc[a]:4d}")
    print(f"\n报告 → {OUT_DIR / 'dedup_plan.md'}")


if __name__ == "__main__":
    main()
