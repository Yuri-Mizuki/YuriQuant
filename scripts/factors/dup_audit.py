"""全库冗余审计：一次性算清 965 个因子的两两截面相关，并把结论写回 registry。

为什么需要它
------------
``register()`` 的 ``check_dup`` 通道是**逐候选 × 全库读面板**：单候选 10.7 s，
全 A 938 个 ≈ 167 分钟（2026-09-12 实测）。贵到没人开，结果是 ``dup_checked``
全库 0 行——``contract_violations()`` 里 "正式因子不许没查过冗余" 这条门禁
永远红着，任何因子都晋升不了 ``active``。

贵的原因是**重复劳动**：每来一个新因子，都把所有旧因子的 58 MB 面板重读一遍。
但"因子 i 与因子 j 有多像"是只与库内容有关的静态事实，算一次能答所有查询。

口径
----
与 ``_run_dup_check`` 一致：**逐日截面 Spearman 相关的均值**。实现上把它写成
矩阵乘法——对采样日 d，把每行（因子）在当日有效股票上的排名做 z-score 得到
``Z_d``（F×C），则当日相关矩阵 ``C_d = Z_d @ Z_d.T / n_d``，对日取加权平均。

样本
----
不抽股票（避免市值/流动性偏差），只在**日期**上抽样：均匀取 K 个交易日
（默认 200，跨 2018-2026）。全 A 5817 只票超内存时按覆盖度+成交额截断到
``--max-codes``（默认 3500），截断情况一定打印。

已知偏差（方向明确）：某日某因子有 NaN 的股票，z 记 0（= 当日均值），
两个因子有效集合不完全相同时相关被**低估**。所以本表可能漏报重复，
不会误报；阈值判断因此偏保守。

产物
----
- ``<库>/dup_matrix.npz``    相关矩阵 + 因子名 + 采样日（3.7 MB，随库走）
- ``<库>/dup_signatures.npy`` 采样签名 (K,F,C) float32（供入库门禁秒级查表）
- ``reports/library_governance/dup_audit.md`` / ``dup_neighbors.csv`` / ``dup_clusters.csv``

用法::

    # 审计（不写 registry）
    python -m scripts.factors.dup_audit --dataset all_a_2018_2026
    # 审计 + 回填 registry
    python -m scripts.factors.dup_audit --apply
    # 只看某几个因子跟谁像
    python -m scripts.factors.dup_audit --names rev_1m,alpha191_144 --query
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402
from research.factor_library import FactorLibrary, _slug  # noqa: E402

log = setup_logging("dup_audit")

DEFAULT_DATES = 200
DEFAULT_MAX_CODES = 3500
DEFAULT_DUP_THRESH = 0.85


# ---------------------------------------------------------------- 样本准备
def _trading_dates(lib: FactorLibrary, k: int) -> pd.DatetimeIndex:
    """从库内面板的实际覆盖范围取交易日，均匀抽 k 个。

    不用日线主表：它从 2015 年起，而本库面板从 2018 年起，抽到 2015-2017
    的日期会让全部因子在该日全空（静默浪费 1/4 的采样预算）。改为读几个
    代表面板 index 的交集——库的**真实**公共日历。
    """
    reg = lib._load_registry()
    picks = reg.head(5)["name"].tolist() + reg.tail(5)["name"].tolist()
    idx: pd.DatetimeIndex | None = None
    for nm in picks:
        p = lib.panel_path(nm) if hasattr(lib, "panel_path") else None
        if p is None:
            p = lib._resolve_panel_path(reg[reg["name"] == nm].iloc[0])
        if p is None or not Path(p).exists():
            continue
        cur = pd.DatetimeIndex(pd.read_parquet(p, columns=[]).index)
        idx = cur if idx is None else idx.intersection(cur)
    if idx is None:
        raise RuntimeError("读不到任何面板的日期索引")
    idx = pd.DatetimeIndex(pd.to_datetime(idx)).normalize().sort_values()
    if len(idx) <= k:
        return idx
    # 等距抽样：保留首尾，中间均匀
    pos = np.linspace(0, len(idx) - 1, k).round().astype(int)
    return idx[np.unique(pos)]


def _code_pool(dates: pd.DatetimeIndex, max_codes: int) -> tuple[list[str], pd.DataFrame]:
    """按"采样日覆盖度 + 中位成交额"选股票池。

    覆盖度优先是为了让每个因子在池内的有效样本尽量一致（NaN→0 的低估越小，
    相关越准）；成交额是次级排序，用来在覆盖度相同时偏向流动性好的票。
    """
    from config import Config

    root = Path(str(Config.cache()["root"]))
    d = pd.read_parquet(root / "daily_all_a.parquet",
                        columns=["amount"]).reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d[d["date"].isin(set(dates))]
    g = d.groupby("code")
    pool = pd.DataFrame({
        "cover": g["date"].nunique() / max(1, len(dates)),
        "amount": g["amount"].median(),
    })
    pool = pool.sort_values(["cover", "amount"], ascending=False)
    if len(pool) > max_codes:
        log.info("股票池截断 %d → %d（按覆盖度+成交额），完整因子相关性以截断池为准",
                 len(pool), max_codes)
        pool = pool.head(max_codes)
    return pool.index.tolist(), pool


# ---------------------------------------------------------------- 签名构建
def _row_rank_zscore(a: np.ndarray) -> np.ndarray:
    """(R, C) 原始值 → 逐行截面 rank 后 z-score，NaN 记 0。

    NaN 记 0 而不是剔除，是因为后面要走矩阵乘法；0 在 z-score 尺度上等于
    "当日该股票处于中位"，对缺失的假设与截面均值填补一致。
    """
    from scipy.stats import rankdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = rankdata(a, axis=1, nan_policy="omit")
        mu = np.nanmean(r, axis=1, keepdims=True)
        sd = np.nanstd(r, axis=1, keepdims=True)
        z = (r - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def build_signatures(lib: FactorLibrary, names: list[str], dates: pd.DatetimeIndex,
                     codes: list[str], *, dtype=np.float32
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """逐因子读面板 → 采样日签名 (K, F, C)。单因子读取失败记全 0（= 无信息）。"""
    import pyarrow.parquet as pq

    K, F, C = len(dates), len(names), len(codes)
    Z = np.zeros((K, F, C), dtype=dtype)
    reg = lib._load_registry()
    rows = {r["name"]: r for _, r in reg.iterrows()}
    cover = np.zeros(F, dtype=np.float32)
    missing: list[str] = []
    t0 = time.time()

    for fi, name in enumerate(names):
        row = rows[name]
        p = lib._resolve_panel_path(row)
        if p is None or not Path(p).exists():
            missing.append(name)
            continue
        try:
            avail = set(pq.ParquetFile(p).schema_arrow.names)
        except Exception:  # noqa: BLE001
            missing.append(name)
            continue
        cols = [c for c in codes if c in avail]
        if len(cols) < 50:
            missing.append(name)
            continue
        df = pd.read_parquet(p, columns=cols).reindex(index=dates, columns=codes)
        a = df.to_numpy(dtype=np.float64)
        cover[fi] = float(np.isfinite(a).sum()) / max(1, a.size)
        Z[:, fi, :] = _row_rank_zscore(a)
        if (fi + 1) % 50 == 0 or fi + 1 == F:
            el = time.time() - t0
            log.info("签名 %d/%d（%.0fs，预计剩余 %.0fs）", fi + 1, F, el,
                     el / (fi + 1) * (F - fi - 1))

    if missing:
        log.warning("%d 个因子无可用面板，签名记 0（不参与相关性判定）：%s",
                    len(missing), missing[:8])
    return Z, cover, missing


# ---------------------------------------------------------------- 相关矩阵
def correlation_matrix(Z: np.ndarray) -> np.ndarray:
    """(K,F,C) 签名 → (F,F) 平均截面相关。

    ``corr_ij = Σ_k z_ik·z_jk / sqrt(Σ_k‖z_ik‖² · Σ_k‖z_jk‖²)``（c=0 的格不
    贡献），即把各期堆起来算的 pooled 相关。

    试过按「逐期共同有效格」归一（pairwise-complete），实测反而更差：共同集
    很小时（30 日分位类因子早期只有几十只票有效）小样本相关噪声大，平均后
    |r| 系统性上偏。两组口径对 4 对已知因子的误差分别为 ≤0.019 / ≤0.028，
    故取单边归一。代价是因子有效集合差异大时相关**偏低**——方向已知，
    只会漏报重复，不会误报。

    每期 z 已按「该因子当日有效股票」标准化，故对角元恒为 1。
    """
    K, F, _C = Z.shape
    num = np.zeros((F, F), dtype=np.float64)
    mass = 0.0
    for k in range(K):
        Zk = Z[k].astype(np.float64, copy=False)
        nz = float((Zk != 0).sum())
        # 空得几乎没观测的交易日直接跳过，避免被单日噪声带偏。阈值按规模缩放：
        # 全库（F=965×C=3500）恒为 100 格，与旧行为逐位一致；小样本库按 10% 算，
        # 否则 60 只票的测试夹具会被整期跳过、mass=0 直接抛错。
        if nz < min(100.0, 0.1 * Zk.size):
            continue
        num += Zk @ Zk.T
        mass += nz                      # Σ_k n_ik（各因子有效数不同，取平均量级）
    if mass <= 0:
        raise RuntimeError("签名全为空，无法计算相关矩阵")
    corr = num / mass
    d = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(np.nan_to_num(corr), -1.0, 1.0).astype(np.float32)


def neighbors(corr: np.ndarray, names: list[str], top_k: int = 3) -> pd.DataFrame:
    """每个因子的最近邻（排除自身）。"""
    c = corr.copy()
    np.fill_diagonal(c, -np.inf)
    rows = []
    for i, nm in enumerate(names):
        order = np.argpartition(-c[i], min(top_k, len(names) - 1))[:top_k]
        order = order[np.argsort(-c[i][order])]
        rec = {"name": nm, "max_corr": float(c[i][order[0]]) if len(order) else np.nan,
               "top": names[order[0]] if len(order) else ""}
        for j, oi in enumerate(order, 1):
            rec[f"nb{j}"] = names[oi]
            rec[f"corr{j}"] = float(c[i][oi])
        rows.append(rec)
    return pd.DataFrame(rows)


def clusters(corr: np.ndarray, names: list[str], thresh: float,
             method: str = "complete") -> pd.DataFrame:
    """重复簇：相关 ≥ thresh 连边后聚类。

    用 **complete** 而不是 single。单链接有链式效应：A~B 0.9、B~C 0.9 但 A~C
    只有 0.1，会被串成一簇——实测全库单链接给出一个 323 个成员、簇内相关
    跨 [0.08, 1.00] 的巨簇，这种簇无法回答"可以合并哪些"。完全链接保证
    簇内**任意两点**都 ≥ thresh，代价是簇更碎（真正能合并的才会在一起）。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = 1.0 - corr.astype(np.float64)
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2, 0, None)
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method=method)
    lab = fcluster(Z, t=1.0 - thresh, criterion="distance")
    return pd.DataFrame({"name": names, "cluster": lab})


# ---------------------------------------------------------------- 回填
def _resid_for(lib: FactorLibrary, name: str, top: str) -> tuple[float, float]:
    """近重复因子的正交残差 IC（对 top 做逐日截面回归取残差）。

    与 ``_run_dup_check`` 内部同口径（带截距、逐日、残差与未来收益都先截面
    rank 再 Pearson），但两处实现差异都是为了能跑全库：

    1. **不走 pandas 行索引**。``panel.loc[d, cc]`` 每期重建两个 Series，
       2487 期下光索引就几秒；改成 ``to_numpy()`` 后按行切片，单因子从
       ~5 s 降到 ~0.5 s（624 个近重复因子，快慢差 45 分钟）。
    2. **残差判零用相对阈值**。float32 面板完全共线时残差 std 是 1e-7 量级
       而不是 0，``_residual_ic`` 原来的绝对阈值 1e-10 永远不会触发，于是
       "换个名字的同一个因子"会算出纯噪声的伪残差 IC（实测给过 +0.0165）。
    """
    import scipy.stats as ss_mod

    from stats.robust_stats import nw_tstat

    panel = lib.get_panel(name)
    old = lib.get_panel(top)
    if panel is None or old is None:
        return float("nan"), 0.0
    rp = lib.default_returns_panel()
    cd = panel.index.intersection(old.index).intersection(rp.index)
    cc = panel.columns.intersection(old.columns).intersection(rp.columns)
    if len(cd) < 20 or len(cc) < 20:
        return float("nan"), 0.0
    Y = panel.reindex(index=cd, columns=cc).to_numpy(dtype=np.float64)
    X = old.reindex(index=cd, columns=cc).to_numpy(dtype=np.float64)
    R = rp.reindex(index=cd, columns=cc).to_numpy(dtype=np.float64)

    vs: list[float] = []
    ds: list = []
    for t in range(len(cd)):
        y, x, r = Y[t], X[t], R[t]
        ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(r)
        if ok.sum() < 20:
            continue
        yv, xv, rv = y[ok], x[ok], r[ok]
        xm, ym = xv.mean(), yv.mean()
        dx = xv - xm
        den = float(dx @ dx)
        if den <= 0:
            continue                                  # x 当日无方差（截面常数）
        b = float(dx @ (yv - ym)) / den
        resid = (yv - ym) - b * dx
        scale = float(np.std(yv))
        if scale <= 0:
            continue
        if float(np.std(resid)) / scale < 1e-6:       # 完全共线 → 无增量信息
            ds.append(cd[t]); vs.append(0.0)
            continue
        if np.std(rv) == 0:
            continue
        ic = float(np.corrcoef(ss_mod.rankdata(resid), ss_mod.rankdata(rv))[0, 1])
        if not np.isnan(ic):
            ds.append(cd[t]); vs.append(ic)
    if len(vs) < 20:
        return float("nan"), 0.0
    t, _se, _lag = nw_tstat(np.asarray(vs))
    return float(np.mean(vs)), float(t)


def _resid_task(task: tuple[str, str, str, str]) -> tuple[str, float, float]:
    """子进程入口：只算数值，registry 由父进程统一写。"""
    root, dataset, name, top = task
    lib = FactorLibrary(root=root, dataset=dataset)
    ic, t = _resid_for(lib, name, top)
    return name, ic, t


def apply_registry(lib: FactorLibrary, nb: pd.DataFrame, corr: np.ndarray,
                   names: list[str], thresh: float, *, limit_resid: int = 0,
                   exclude: set[str] | None = None, workers: int = 4) -> dict:
    """把审计结论写回 registry。

    ``exclude`` 里的因子（面板全空/常数/截面无差异/极稀疏）只写
    ``dup_checked=False``：它们的相关是噪声，写成 True 等于用"查过了"掩盖
    "查不出结果"，正是 2026-09 那次 912 个因子被静默跳过却标 True 的老毛病。
    """
    from concurrent.futures import ProcessPoolExecutor

    exclude = exclude or set()
    payload: dict[str, dict] = {}
    near = nb[(nb["max_corr"] >= thresh) & (~nb["name"].isin(exclude))] \
        .sort_values("max_corr", ascending=False)
    if limit_resid:
        near = near.head(limit_resid)
    for _, r in nb.iterrows():
        if r["name"] in exclude:
            payload[r["name"]] = {"dup_checked": False, "dup_corr_max": "",
                                  "dup_top": "", "resid_ic": "", "resid_t_nw": ""}
            continue
        payload[r["name"]] = {
            "dup_checked": True,
            "dup_corr_max": round(float(r["max_corr"]), 3),
            "dup_top": r["top"],
        }

    # 残差 IC 逐个因子回归，624 个近重复因子要跑十几分钟。中途被打断就全废，
    # 因此算一批落一批盘，续跑时直接跳过已算过的。
    cache_f = Path(str(lib.base_root)) / lib.dataset / "dup_resid.csv"
    resid: dict[str, tuple[float, float]] = {}
    if cache_f.exists():
        c = pd.read_csv(cache_f)
        for _, r in c.iterrows():
            resid[str(r["name"])] = (float(r["resid_ic"]), float(r["resid_t_nw"]))
        log.info("残差缓存命中 %d 个", len(resid))
    if len(near):
        todo = [(str(lib.base_root), lib.dataset, r["name"], r["top"])
                for _, r in near.iterrows() if r["name"] not in resid]
        n_w = max(1, min(workers, len(todo))) if todo else 1
        log.info("残差 IC：待算 %d 个（近重复 %d，缓存 %d），%d 进程",
                 len(todo), len(near), len(resid), n_w)
        t0 = time.time()
        if todo:
            with ProcessPoolExecutor(max_workers=n_w) as ex:
                for i, (nm, ic, t) in enumerate(ex.map(_resid_task, todo), 1):
                    resid[nm] = (ic, t)
                    if i % 50 == 0:
                        pd.DataFrame(
                            [{"name": k, "resid_ic": v[0], "resid_t_nw": v[1]}
                             for k, v in resid.items()]
                        ).to_csv(cache_f, index=False, encoding="utf-8-sig")
                        log.info("残差 IC %d/%d（%.0fs）", i, len(todo),
                                 time.time() - t0)
            pd.DataFrame([{"name": k, "resid_ic": v[0], "resid_t_nw": v[1]}
                          for k, v in resid.items()]
                         ).to_csv(cache_f, index=False, encoding="utf-8-sig")
        fillable = set(near["name"])
        for nm, (ic, t) in resid.items():
            if nm not in fillable:
                continue
            payload[nm]["resid_ic"] = round(ic, 4) if np.isfinite(ic) else ""
            payload[nm]["resid_t_nw"] = round(t, 2) if np.isfinite(ic) else ""

    items = list(payload.items())
    for i in range(0, len(items), 100):
        lib.apply_eval_many(dict(items[i:i + 100]))
    return {"updated": len(payload), "resid": len(resid), "near": len(near),
            "excluded": len([n for n in names if n in exclude])}


# ---------------------------------------------------------------- 落盘与报告
def z_coverage(sig_path: Path, chunk: int = 40) -> np.ndarray:
    """每个因子的「截面可排名率」= 签名非零格占比。

    这一列比「面板非空率」更能说明问题：**常数因子**（整列同一个值）有非空值，
    但截面 rank 后全部并列、z 恒为 0，签名覆盖率是 0。所以
    ``z_cover == 0 而 cover > 0`` 精确指向"有数但无区分度"的死因子。
    """
    sig = np.load(sig_path, mmap_mode="r")
    F = sig.shape[1]
    out = np.zeros(F, dtype=np.float32)
    for i in range(0, F, chunk):
        blk = np.asarray(sig[:, i:i + chunk, :])
        out[i:i + chunk] = (blk != 0).mean(axis=(0, 2))
    return out


def vitality_table(names: list[str], cover: np.ndarray, zcov: np.ndarray) -> pd.DataFrame:
    """把因子按「能否参与相关分析」分类，并给出死因。"""
    kind = np.where(cover <= 0, "面板全空",
                    np.where(zcov <= 0, "常数（截面无差异）",
                             np.where(zcov < 0.05, "极稀疏", "")))
    df = pd.DataFrame({"name": names,
                       "cover": np.round(cover, 4),
                       "z_cover": np.round(zcov, 4),
                       "issue": kind})
    return df.sort_values(["issue", "cover"]).reset_index(drop=True)


def refine_vitality(lib: FactorLibrary, vital: pd.DataFrame) -> pd.DataFrame:
    """对被判可疑的因子读一次面板，把「全空 / 常数 / 极稀疏」分清。

    签名只能说明「截面无可排名信息」，分不清是面板真空还是整列同值——而这两
    个病根完全不同（前者是数据管线断了，后者是公式退化）。只对可疑的十几个
    因子读面板，代价十几秒。
    """
    bad = vital["issue"].fillna("").astype(str).str.strip() != ""
    for i in vital.index[bad]:
        nm = vital.at[i, "name"]
        p = lib.get_panel(nm)
        if p is None:
            vital.at[i, "issue"] = "面板读不到"
            continue
        a = p.to_numpy()
        fin = np.isfinite(a)
        cover = float(fin.mean())
        uniq = int(len(np.unique(a[fin]))) if fin.any() else 0
        # 同日跨股票的唯一值数：宏观/状态类因子每天所有股票同值，进截面库天然失效
        rows = a[:400]
        per_row = np.array([len(np.unique(r[np.isfinite(r)])) for r in rows])
        med_uni = float(np.median(per_row[per_row > 0])) if (per_row > 0).any() else 0.0
        z = float(vital.at[i, "z_cover"] or 0)
        if cover <= 0 or uniq == 0:
            kind = "面板全空"
        elif uniq <= 1:
            kind = "常数（整列同值）"
        elif med_uni <= 1.5:
            kind = f"截面无差异（同日跨股票仅 {med_uni:.0f} 个取值）"
        elif z < 0.05:
            kind = "极稀疏"
        else:
            kind = ""
        vital.at[i, "cover"] = round(cover, 4)
        vital.at[i, "row_card"] = round(med_uni, 1)
        vital.at[i, "issue"] = kind
    return vital


def save_artifacts(lib: FactorLibrary, corr: np.ndarray, names: list[str],
                   dates: pd.DatetimeIndex, codes: list[str], Z: np.ndarray,
                   cover: np.ndarray) -> pd.DataFrame:
    base = Path(lib.base_root) / lib.dataset
    base.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base / "dup_matrix.npz", corr=corr,
                        names=np.array(names, dtype="U64"),
                        dates=dates.values.astype("datetime64[D]").astype(str))
    np.save(base / "dup_signatures.npy", Z)
    zcov = (Z != 0).mean(axis=(0, 2))
    vital = refine_vitality(lib, vitality_table(names, cover, zcov))
    vital.to_csv(base / "dup_vitality.csv", index=False, encoding="utf-8-sig")
    dead = vital[vital["issue"] != ""]
    if len(dead):
        log.warning("%d 个因子无法参与相关分析（面板全空/常数/极稀疏），"
                    "明细见 %s", len(dead), base / "dup_vitality.csv")
    meta = {
        "dataset": lib.dataset,
        "built_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_factors": len(names),
        "n_dates": int(len(dates)),
        "n_codes": int(len(codes)),
        "dates_first": str(dates[0].date()) if len(dates) else "",
        "dates_last": str(dates[-1].date()) if len(dates) else "",
        "max_codes": int(len(codes)),
        "mean_cover": round(float(np.mean(cover)), 4),
        "min_cover": round(float(np.min(cover)), 4),
        "n_low_cover": int((cover < 0.5).sum()),
        "corr_dtype": "float32",
        # 签名对齐用：入库门禁要把新因子重排到这套 dates × codes 上
        "dates": [str(d) for d in dates],
        "codes": list(codes),
        "note": "截面 Spearman 相关，日期抽样；NaN 记 0 故相关偏低估",
    }
    (base / "dup_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("落盘：%s（矩阵 %s / 签名 %s）", base,
             f"{corr.nbytes/1e6:.1f}MB", f"{Z.nbytes/1e9:.2f}GB")
    return vital


def write_report(lib: FactorLibrary, nb: pd.DataFrame, cl: pd.DataFrame,
                 corr: np.ndarray, names: list[str], meta: dict, out_dir: Path,
                 vital: pd.DataFrame | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_out = nb.copy()
    mc = nb_out["max_corr"]
    reg = lib._load_registry().set_index("name")

    def _metric(col: str, idx) -> np.ndarray:
        """取 registry 指标列；列不存在时给全 NaN（不同库的 canonical 列不齐）。"""
        if col not in reg.columns:
            return np.full(len(idx), np.nan)
        return pd.to_numeric(reg.reindex(idx)[col], errors="coerce").to_numpy()

    nb_out["sharpe_ls_M"] = _metric("sharpe_ls_M", nb_out["name"])
    nb_out["ic_mean"] = _metric("ic_mean", nb_out["name"])
    nb_out.sort_values("max_corr", ascending=False).to_csv(
        out_dir / "dup_neighbors.csv", index=False, encoding="utf-8-sig")

    grp = cl.groupby("cluster")["name"].apply(list)
    sizes = grp.map(len)
    big = sizes[sizes >= 2].sort_values(ascending=False)
    rows = []
    for cid, members in grp.items():
        if len(members) < 2:
            continue
        ii = [names.index(m) for m in members]
        sub = corr[np.ix_(ii, ii)].copy()
        np.fill_diagonal(sub, np.nan)
        sh = pd.Series(_metric("sharpe_ls_M", members), index=members)
        best = sh.idxmax() if sh.notna().any() else members[0]
        rows.append({
            "cluster": int(cid), "size": len(members),
            "rep": best, "rep_sharpe": float(sh.get(best, np.nan)),
            "min_corr": float(np.nanmin(sub)), "max_corr": float(np.nanmax(sub)),
            "members": "|".join(members),
        })
    clu = pd.DataFrame(rows).sort_values("size", ascending=False)
    clu.to_csv(out_dir / "dup_clusters.csv", index=False, encoding="utf-8-sig")

    n = len(names)
    buckets = [(0.95, 1.01), (0.9, 0.95), (0.85, 0.9), (0.7, 0.85), (0.5, 0.7), (-1.01, 0.5)]
    lines = [
        "# 因子库冗余审计（全库 pairwise）",
        "",
        f"- 数据集：`{lib.dataset}`，因子 {n} 个",
        f"- 采样：{meta['n_dates']} 个交易日（{meta['dates_first']} ~ {meta['dates_last']}）"
        f" × {meta['n_codes']} 只股票",
        f"- 口径：逐日截面 Spearman 相关，对日加权平均；NaN 记 0 ⇒ 相关**偏低估**",
        f"- 平均有效覆盖 {meta['mean_cover']:.1%}（最低 {meta['min_cover']:.1%}，"
        f"<50% 的有 {meta['n_low_cover']} 个）",
        f"- 构建时间 {meta['built_at']}",
        "",
        "## 一、按「与最相似因子的相关」分档",
        "",
        "| 档位 | 因子数 | 占比 | 含义 |",
        "|---|---:|---:|---|",
    ]
    meaning = {
        (0.95, 1.01): "实质同一因子（可合并）",
        (0.9, 0.95): "高度冗余（择一保留）",
        (0.85, 0.9): "疑似冗余（需残差检验）",
        (0.7, 0.85): "同族变体，仍有差异",
        (0.5, 0.7): "相关但不重复",
        (-1.01, 0.5): "基本独立",
    }
    for lo, hi in buckets:
        m = (mc >= lo) & (mc < hi)
        lines.append(f"| [{lo:.2f}, {hi:.2f}{')' if hi <= 1 else ']'} | {int(m.sum())} | "
                     f"{int(m.sum())/n:.1%} | {meaning[(lo, hi)]} |")

    lines += ["", "## 二、重复簇（完全链接，相关 ≥ 0.85 连边）", ""]
    lines.append(f"共 {len(big)} 个簇、覆盖 {int(sizes[sizes >= 2].sum())} 个因子"
                 f"（占全库 {sizes[sizes >= 2].sum()/n:.0%}）；"
                 f"其中被簇「吸收」的冗余因子 {int(sizes[sizes >= 2].sum()) - len(big)} 个。")
    lines += ["", "| 簇 | 成员数 | 代表（最高 sharpe） | 代表 sharpe | 簇内相关 |",
              "|---:|---:|---|---:|---|"]
    for _, r in clu.head(20).iterrows():
        lines.append(f"| {r['cluster']} | {r['size']} | `{r['rep']}` | "
                     f"{r['rep_sharpe']:.2f} | [{r['min_corr']:.2f}, {r['max_corr']:.2f}] |")
    if len(clu) > 20:
        lines.append(f"| … | 其余 {len(clu)-20} 个簇见 `dup_clusters.csv` | | | |")

    lines += ["", "## 三、最像的 25 对", "", "| 因子 | 最像的 | 相关 | 该因子 sharpe |",
              "|---|---|---:|---:|"]
    # 只列无序对（a→b 与 b→a 是同一对），否则 25 行里有一半是镜像
    seen_pair: set[tuple[str, str]] = set()
    n_pair = 0
    for _, r in nb_out.sort_values("max_corr", ascending=False).iterrows():
        key = tuple(sorted((str(r["name"]), str(r["top"]))))
        if key[0] == key[1] or key in seen_pair:
            continue
        seen_pair.add(key)
        lines.append(f"| `{r['name']}` | `{r['top']}` | {r['max_corr']:.3f} | "
                     f"{r['sharpe_ls_M']:.2f} |")
        n_pair += 1
        if n_pair >= 25:
            break

    # ---- 四、近重复因子的残差增量：这一节回答「相关高，但要不要留」
    if "resid_ic" in reg.columns:
        pick = reg.reindex(nb_out["name"])
        ri = pd.to_numeric(pick["resid_ic"], errors="coerce").to_numpy(dtype=float)
        rt = (pd.to_numeric(pick["resid_t_nw"], errors="coerce").to_numpy(dtype=float)
              if "resid_t_nw" in reg.columns else np.full(len(ri), np.nan))
        pos = {n: i for i, n in enumerate(names)}
        ex = np.array([float(corr[pos[a], pos[b]]) if (a in pos and b in pos) else np.nan
                       for a, b in zip(nb_out["name"], nb_out["top"])])
        ok = np.isfinite(ri)
        if ok.any():
            a = np.abs(ri)

            def _cnt(lo: float, hi: float) -> int:
                return int((ok & (a >= lo) & (a < hi)).sum())

            n_tot = int(ok.sum())
            n_zero = int((ok & (ri == 0)).sum())
            n_useful = _cnt(0.02, 9)
            lines += [
                "", "## 四、近重复因子的残差增量（哪些是真重复）", "",
                f"对前面 {n_tot} 个「与最近邻相关 ≥ 0.85」的因子，逐个对最近邻做"
                "**逐日截面回归（含截距）**，取残差与次日收益的 rank 相关。"
                "残差 IC 越接近 0，说明这个因子被最近邻解释得越彻底——"
                "留两个等于留一个。", "",
                "| 残差 IC 幅度 | 因子数 | 判读 |", "|---|---:|---|",
                f"| = 0（完全共线） | {n_zero} | 换个名字的同一因子，必删其一 |",
                f"| (0, 0.005) | {_cnt(1e-9, 0.005)} | 残差可忽略，实质重复 |",
                f"| [0.005, 0.01) | {_cnt(0.005, 0.01)} | 边际增量，不足以独立持有 |",
                f"| [0.01, 0.02) | {_cnt(0.01, 0.02)} | 有增量，但弱于主因子 |",
                f"| ≥ 0.02 | {n_useful} | **有可用增量**，可考虑与主因子并存 |",
                "",
                f"即：{n_tot} 个近重复里 **{n_tot - n_useful}** 个的增量可以忽略"
                f"（合并或删除即可），**{n_useful}** 个值得单独留。"
                "这正是「相关过 0.9 就删」做不到的区分——两个因子可能同属反转族，"
                "但一个含着另一个没有的波动结构。", "",
                "> **t 值不能当判据**：本样本约 2500 个交易日，|IC| 只要到 0.005 就 "
                "t>2，|t|≥2 的占 65%，几乎不筛人。判读只看上表的 IC 幅度。", "",
            ]
            cand = pd.DataFrame({
                "name": nb_out["name"].to_numpy(), "top": nb_out["top"].to_numpy(),
                "corr": ex, "resid_ic": ri, "sharpe": nb_out["sharpe_ls_M"].to_numpy(),
            })
            cand = cand[cand["resid_ic"].notna() & (cand["resid_ic"].abs() >= 0.02)]
            if len(cand):
                seen_c: set[tuple[str, str]] = set()
                shown = []
                for _, r in cand.sort_values("sharpe", ascending=False).iterrows():
                    k = tuple(sorted((str(r["name"]), str(r["top"]))))
                    if k in seen_c:
                        continue
                    seen_c.add(k)
                    shown.append(r)
                    if len(shown) >= 12:
                        break
                lines += ["| 因子 | 最近邻 | 与最近邻相关 | 残差 IC | 该因子 sharpe |",
                          "|---|---|---:|---:|---:|"]
                for r in shown:
                    lines.append(f"| `{r['name']}` | `{r['top']}` | {r['corr']:.3f} | "
                                 f"{r['resid_ic']:+.4f} | {r['sharpe']:.2f} |")
                lines += ["", "（按 sharpe 取前 12，且残差 |IC| ≥ 0.02；全量见 "
                          "`dup_neighbors.csv` 与 registry 的 `resid_ic` 列。）"]
            lines += ["", "> 口径提醒：这里只对**单个**最近邻正交，未控制风格与行业暴露；"
                      "相关 ≥ 0.999 的残差在数值上只是一层薄片，单独交易没有容量。"
                      "留不保留要看组合层面的边际夏普，不是看残差 IC 本身。", ""]

    limits = ["", "## 六、口径与局限", "",
              "1. **只在日期上抽样**（等距，跨 2016-2026），不抽股票——避免"
              "大盘/小盘偏好造成的系统性偏差。",
              "2. **NaN 记 0**（= 当日截面中位）。两个因子有效股票集合不同时相关被"
              "低估，故本表**可能漏报重复、不会误报**；0.85 阈值偏保守。",
              "3. 与全样本逐日相关的实测误差 ≤0.02（4 对已知因子对照）。",
              f"4. 股票池上限 {meta['max_codes']}（按采样日覆盖度+成交额，覆盖度优先），"
              "超出时按覆盖度截断。",
              "5. 簇用**完全链接**：簇内任意两个成员的相关都必须 ≥ 阈值。单链接会"
              "链式传递（A~B、B~C 但 A 与 C 无关也能并成一簇），全库上串出过 323 个"
              "成员的巨簇，无法回答「哪些能合并」。",
              "6. 本表回答「两个因子有多像」，不回答「是否值得留」——后者要看 "
              "`sharpe_ls_M` 与可交易性，见 `reports/library_governance/README.md`。",
              ""]

    if vital is not None and len(vital):
        dead = vital[vital["issue"].fillna("").astype(str).str.strip() != ""]
        lines += ["", "## 五、无法参与相关分析的因子（顺带查出来的病）", ""]
        if dead.empty:
            lines.append("无。")
        else:
            lines += [
                f"共 **{len(dead)}** 个。这批因子的相关数不可信（签名全 0 或近 0），"
                "本报告不把它们算进重复簇，回填时 ``dup_checked`` 保持 False。",
                "",
                "| 类型 | 个数 | 含义 |",
                "|---|---:|---|",
            ]
            for k, g in dead.groupby("issue"):
                if k == "面板全空":
                    desc = "面板里一个有效值都没有，数据没落地"
                elif k.startswith("常数"):
                    desc = "整列同值（如 CLOSE0 = close/close-1 ≡ 0），公式退化"
                elif k.startswith("截面无差异"):
                    desc = ("有值，但同一天所有股票同值——时序变量（宏观）被广播到面板，"
                            "或占位值（float_mktcap 全市场恒为 10000.0）")
                elif k.startswith("极稀疏"):
                    desc = "可排名格 <5%，相关是噪声"
                else:
                    desc = ""
                lines.append(f"| {k} | {len(g)} | {desc} |")
            lines += ["", "| 因子 | 面板非空率 | 可排名率 | 问题 |", "|---|---:|---:|---|"]
            for _, r in dead.head(40).iterrows():
                cov_s = f"{r['cover']:.1%}" if pd.notna(r["cover"]) else "—"
                lines.append(f"| `{r['name']}` | {cov_s} | {r['z_cover']:.1%} | "
                             f"{r['issue']} |")
            if len(dead) > 40:
                lines.append(f"| … | | | 其余 {len(dead)-40} 个见 `dup_vitality.csv` |")
        lines.append("")

    lines += limits
    (out_dir / "dup_audit.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("报告：%s", out_dir / "dup_audit.md")


# ---------------------------------------------------------------- CLI
def _query(lib: FactorLibrary, names: list[str], top_k: int) -> None:
    base = Path(lib.base_root) / lib.dataset
    f = base / "dup_matrix.npz"
    if not f.exists():
        raise SystemExit(f"没有相关矩阵：{f}，先跑一次审计")
    z = np.load(f)
    corr, all_names = z["corr"], [str(n) for n in z["names"]]
    for q in names:
        if q not in all_names:
            print(f"{q}: 不在矩阵里")
            continue
        i = all_names.index(q)
        order = np.argsort(-corr[i])[:top_k + 1]
        print(f"\n{q}:")
        for oi in order:
            if all_names[oi] == q:
                continue
            print(f"  {corr[i, oi]:+.3f}  {all_names[oi]}")


def _print_summary(nb: pd.DataFrame) -> None:
    mc = nb["max_corr"]
    print(f"\n{int(mc.notna().sum())}/{len(nb)} 个因子有最相似项")
    for lo, hi in [(0.95, 1.01), (0.9, 0.95), (0.85, 0.9), (0.7, 0.85), (-1.01, 0.7)]:
        m = (mc >= lo) & (mc < hi)
        print(f"  [{lo:.2f},{hi:.2f}{')' if hi <= 1 else ']'}  {int(m.sum()):4d}")
    print("\n最像的 10 个：")
    print(nb.sort_values("max_corr", ascending=False)
          .head(10)[["name", "top", "max_corr"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="因子库全库冗余审计")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--root", default=None)
    ap.add_argument("--sample-dates", type=int, default=DEFAULT_DATES)
    ap.add_argument("--max-codes", type=int, default=DEFAULT_MAX_CODES)
    ap.add_argument("--dup-thresh", type=float, default=DEFAULT_DUP_THRESH)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--apply", action="store_true", help="回填 registry 的 dup_* 列")
    ap.add_argument("--no-resid", action="store_true", help="回填时跳过残差 IC")
    ap.add_argument("--resid-workers", type=int, default=4, help="算残差 IC 的进程数")
    ap.add_argument("--limit", type=int, default=0, help="只算前 N 个因子（调试）")
    ap.add_argument("--names", default="", help="逗号分隔，只审计这些因子")
    ap.add_argument("--query", action="store_true", help="查已有矩阵，不重算")
    ap.add_argument("--reuse", action="store_true",
                    help="复用已落盘的相关矩阵，只重算邻居/簇并重写报告（改报告格式时用）")
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "library_governance"))
    args = ap.parse_args()

    lib = FactorLibrary(root=args.root, dataset=args.dataset)

    if args.query:
        if not args.names:
            raise SystemExit("--query 需要 --names")
        _query(lib, [s.strip() for s in args.names.split(",") if s.strip()], args.top_k)
        return

    if args.reuse:
        base = Path(lib.base_root) / lib.dataset
        npz_f = base / "dup_matrix.npz"
        if not npz_f.exists():
            raise SystemExit(f"没有可复用的矩阵：{npz_f}")
        z = np.load(npz_f)
        corr, mnames = z["corr"], [str(x) for x in z["names"]]
        meta = json.loads((base / "dup_meta.json").read_text(encoding="utf-8"))
        if "dates" not in meta:
            # 旧版 meta 没存对齐日历：确定性重算一遍补上（几秒，不碰面板）
            dates = _trading_dates(lib, args.sample_dates)
            codes, _ = _code_pool(dates, args.max_codes)
            meta["dates"] = [str(d) for d in dates]
            meta["codes"] = list(codes)
            (base / "dup_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("补写 meta 的 dates/codes：%d 日 × %d 股票", len(dates), len(codes))
        vital_f = base / "dup_vitality.csv"
        if not vital_f.exists():
            # 复用路径没有原始面板非空率，只能按签名区分「无差异」与「极稀疏」；
            # 想区分"全空"和"常数"要重跑一次完整审计（它读面板时顺手记 cover）
            zcov = z_coverage(base / "dup_signatures.npy")
            vital = pd.DataFrame({
                "name": mnames, "cover": np.nan, "z_cover": np.round(zcov, 4),
                "issue": np.where(zcov <= 0, "无截面差异（全空或常数）",
                                  np.where(zcov < 0.05, "极稀疏", "")),
            })
            vital = refine_vitality(lib, vital)
            vital.to_csv(vital_f, index=False, encoding="utf-8-sig")
            log.info("补算 dup_vitality.csv（%d 个因子）", len(mnames))
        vital = pd.read_csv(vital_f)
        nb = neighbors(corr, mnames, args.top_k)
        cl = clusters(corr, mnames, args.dup_thresh)
        write_report(lib, nb, cl, corr, mnames, meta, Path(args.out_dir), vital=vital)
        if args.apply:
            dead = set(vital.loc[vital["issue"].fillna("") != "", "name"])
            res = apply_registry(lib, nb, corr, mnames, args.dup_thresh, exclude=dead,
                                 workers=args.resid_workers)
            print(f"registry 回填 {res['updated']} 行；近重复 {res['near']}，"
                  f"其中 {res['resid']} 个补了残差 IC")
        _print_summary(nb)
        return

    reg = lib._load_registry()
    names = reg["name"].tolist()
    if args.names:
        want = [s.strip() for s in args.names.split(",") if s.strip()]
        names = [n for n in names if n in set(want)]
    if args.limit:
        names = names[: args.limit]
    log.info("审计 %d 个因子", len(names))

    dates = _trading_dates(lib, args.sample_dates)
    codes, pool = _code_pool(dates, args.max_codes)
    log.info("样本：%d 日 × %d 只股票（%s ~ %s）", len(dates), len(codes),
             dates[0].date(), dates[-1].date())

    Z, cover, missing = build_signatures(lib, names, dates, codes)
    corr = correlation_matrix(Z)
    nb = neighbors(corr, names, top_k=args.top_k)
    cl = clusters(corr, names, args.dup_thresh)

    meta = {
        "n_dates": len(dates), "n_codes": len(codes),
        "dates_first": str(dates[0].date()), "dates_last": str(dates[-1].date()),
        "mean_cover": float(np.mean(cover)), "min_cover": float(np.min(cover)),
        "n_low_cover": int((cover < 0.5).sum()),
        "built_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "max_codes": args.max_codes,
    }
    vital = None
    if len(names) == len(reg) and not args.names and not args.limit:
        vital = save_artifacts(lib, corr, names, dates, codes, Z, cover)
        write_report(lib, nb, cl, corr, names, meta, Path(args.out_dir), vital=vital)
    else:
        log.warning("子集审计（%d/%d），不覆盖库内矩阵与报告", len(names), len(reg))

    mc = nb["max_corr"]
    _print_summary(nb)

    if args.apply:
        dead = set()
        if vital is not None:
            dead = set(vital.loc[vital["issue"].fillna("") != "", "name"])
        res = apply_registry(lib, nb, corr, names, args.dup_thresh,
                             exclude=dead, workers=args.resid_workers,
                             limit_resid=0 if not args.no_resid else 0)
        print(f"\nregistry 已回填 {res['updated']} 行；近重复 {res['near']} 个"
              f"（{res['resid']} 个补了残差 IC）；"
              f"无效因子 {res['excluded']} 个置 dup_checked=False")


if __name__ == "__main__":
    main()
