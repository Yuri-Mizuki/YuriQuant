"""因子库延长与批量入库公共操作（build/extend 家族共用）。

从 scripts 层下沉的因子库维护逻辑：标准化批量入库（``register_panels``）、
历史口径一致性校验（``verify_overlap``）与面板延长（``extend_gp_factors`` /
``extend_alpha_factors``）。scripts 只保留 CLI 入口与数据源解析。

依赖纪律：本模块不依赖 scripts（scripts 是顶层）；``register_panels`` 由
``scripts.common.cli_common`` 转出口保持既有导入面不变。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("factor_extension")

# 重叠区归一化偏差容差（同数据同算子的浮点噪声水平）
_OVERLAP_TOL = 1e-6

#: 同批两两查重的规模上限（对数）。超过就只按库内索引查、并显式告警——
#: 1000 个因子的两两矩阵是 50 万对，不是"稍慢"的量级，静默跳过才是错的。
_BATCH_PAIR_LIMIT = 8000
#: 两两相关扫描的抽样目标格子数（天 × 股票），把单对成本压在毫秒级。
_PAIR_CELLS = 300_000


def _rank_z_and_mask(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """面板 → (逐日截面 rank-z, 有效掩码)。复用库内同一份变换，不自造口径。"""
    from research.factor_library import row_rank_zscore

    a = panel.to_numpy(dtype=np.float64)
    return row_rank_zscore(a), np.isfinite(a)


def _pair_rho(za: np.ndarray, ma: np.ndarray,
              zb: np.ndarray, mb: np.ndarray) -> float:
    """两因子在秩空间的池化截面相关（只算双方都有效的格子）。"""
    m = ma & mb
    x = np.where(m, za, 0.0)
    y = np.where(m, zb, 0.0)
    na = float((x * x).sum())
    nb = float((y * y).sum())
    if na <= 0 or nb <= 0:
        return float("nan")
    return float((x * y).sum() / np.sqrt(na * nb))


def _add_note(notes: dict[str, str], name: str, text: str) -> None:
    """把一句说明追加到该因子的 note（多条用「；」连，不覆盖）。"""
    notes[name] = f"{notes[name]}；{text}" if name in notes else text


def _rank_rel_std(za: np.ndarray, ma: np.ndarray, zb: np.ndarray, mb: np.ndarray,
                  *, min_obs: int = 20) -> float:
    """秩空间残差标准差 / 被解释变量标准差（逐日截面 OLS）。

    与 ``research.dup_criteria.rank_resid_stats`` 的 ``rel_std`` 同义：
    < ``REL_STD_DEGENERATE`` 即「精确等价（单调变换/镜像）」。这里不算 IC，
    因为「是否等价」只取决于两个因子之间的秩空间关系，与收益面板无关。
    """
    m = ma & mb
    rel: list[float] = []
    for t in range(za.shape[0]):
        ok = m[t]
        if int(ok.sum()) < min_obs:
            continue
        y = za[t][ok].astype(np.float64)
        x = zb[t][ok].astype(np.float64)
        sy = float(np.std(y))
        if sy <= 0:
            continue
        M = np.column_stack([np.ones(x.size), x])
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        rel.append(float(np.std(y - M @ beta)) / sy)
    return float(np.mean(rel)) if rel else float("nan")


def _same_batch_dup(names: list[str], std: dict[str, pd.DataFrame],
                    families: dict[str, str] | None) -> tuple[
                        dict[str, tuple[str, float, float]],
                        dict[str, str], dict[str, str]]:
    """同批两两查重 + 自动归族（入库前的最后一道）。

    为什么必须做：库内冗余索引是**快照**，同一批新注册的因子彼此不在索引里，
    于是「一起入库的 4 个同族因子」能全部通过各自的库内预检（实测 2026-11-19：
    4 个日频候选两两 |ρ|=0.81~0.94，全入库）。这里补的就是这一格。

    分档与 ``research.dup_criteria`` **完全一致**（唯一真源，不另抄阈值）：

    - ``|ρ| < TIER_T3``（0.85）：不动；
    - ``TIER_T3 ≤ |ρ| < CORR_TRIGGER``（0.95）：告警 + 归入同一 family，
      让晋升的 ``max_per_family`` 能咬住；
    - ``|ρ| ≥ CORR_TRIGGER``：再做秩空间残差检验，``rel_std <
      REL_STD_DEGENERATE``（等价）⇒ 后入者跳过。

    Returns:
        ``(drop, fam, notes)``：``drop[name] = (覆盖者, ρ, rel_std)``；
        ``fam`` 为归族后的 family 映射（含调用方显式传入的）；
        ``notes`` 为需要追加到 registry ``note`` 的说明。
    """
    from research.dup_criteria import (CORR_TRIGGER, REL_STD_DEGENERATE,
                                       TIER_T3)

    fam = dict(families or {})
    drop: dict[str, tuple[str, float, float]] = {}
    notes: dict[str, str] = {}
    k = len(names)
    n_pairs = k * (k - 1) // 2
    if n_pairs > _BATCH_PAIR_LIMIT:
        log.warning("同批 %d 个因子（%d 对）超过两两查重上限 %d：本批只按库内"
                    "索引查重，同批内部新生成的等价因子不会被拦住",
                    k, n_pairs, _BATCH_PAIR_LIMIT)
        return drop, fam, notes

    # 同批面板的日历/股票池可以不同（日频全A vs 分钟重采样的 hs300 子池）。
    # 但**不能**一刀切用全批交集：那会被口径最窄的因子（hs300、2022 起）
    # 把别人的全 A 长历史截掉，把真实同族关系稀释到阈值以下
    # （2026-11-19 实测：全批交集上只剩 1 对过线，各自交集上有 4 对）。
    zs, ms, idxs, colss = [], [], [], []
    for nm in names:
        p = std[nm]
        z, m = _rank_z_and_mask(p)
        zs.append(z)
        ms.append(m)
        idxs.append(p.index)
        colss.append(p.columns)

    parent = {nm: nm for nm in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(k):
        for j in range(i + 1, k):
            a_name, b_name = names[i], names[j]
            if a_name in drop or b_name in drop:
                continue
            # 只在**这两张面板**的共同日历/股票池上比
            dates_ij = idxs[i].intersection(idxs[j])
            codes_ij = colss[i].intersection(colss[j])
            if len(dates_ij) < 20 or len(codes_ij) < 20:
                continue
            za = zs[i][np.ix_(idxs[i].get_indexer(dates_ij),
                              colss[i].get_indexer(codes_ij))]
            ma = ms[i][np.ix_(idxs[i].get_indexer(dates_ij),
                              colss[i].get_indexer(codes_ij))]
            zb = zs[j][np.ix_(idxs[j].get_indexer(dates_ij),
                              colss[j].get_indexer(codes_ij))]
            mb = ms[j][np.ix_(idxs[j].get_indexer(dates_ij),
                              colss[j].get_indexer(codes_ij))]
            # 抽样再算相关：目标 ~250 日 × 上千股，单对毫秒级
            Tp, Cp = za.shape
            d = min(Tp, 250)
            n = max(10, min(Cp, max(10, _PAIR_CELLS // max(d, 1))))
            ti = np.unique(np.linspace(0, Tp - 1, d).astype(int))
            ci = np.unique(np.linspace(0, Cp - 1, n).astype(int))
            rho = _pair_rho(za[np.ix_(ti, ci)], ma[np.ix_(ti, ci)],
                            zb[np.ix_(ti, ci)], mb[np.ix_(ti, ci)])
            if not np.isfinite(rho):
                continue
            ac = abs(rho)
            if ac < TIER_T3:
                continue
            ra, rb = find(a_name), find(b_name)
            if ra != rb:
                parent[rb] = ra
            if ac < CORR_TRIGGER:
                log.warning("同批查重: %s vs %s |ρ|=%.3f ⇒ 同族（%.2f~%.2f 档，"
                            "%d 日 × %d 股），自动归族；该档按产出比不做残差检验",
                            a_name, b_name, ac, TIER_T3, CORR_TRIGGER,
                            Tp, Cp)
                _add_note(notes, b_name,
                          f"同批查重：与 {a_name} |ρ|={ac:.3f}，同族")
                continue
            rel_std = _rank_rel_std(za, ma, zb, mb)
            if np.isfinite(rel_std) and rel_std < REL_STD_DEGENERATE:
                drop[b_name] = (a_name, rho, rel_std)
                log.warning("同批查重: %s 与 %s 秩空间等价（ρ=%.3f, rel_std=%.2e）"
                            "⇒ 本批内重复，跳过 %s 入库",
                            b_name, a_name, rho, rel_std, b_name)
            else:
                log.info("同批查重: %s vs %s |ρ|=%.3f 但秩空间有增量"
                         "（rel_std=%s），放行", b_name, a_name, ac,
                         "无法检验" if not np.isfinite(rel_std) else f"{rel_std:.3f}")
                _add_note(notes, b_name,
                          f"同批查重：与 {a_name} |ρ|={ac:.3f}，高相关但有增量")

    comps: dict[str, list[str]] = {}
    for nm in names:
        if nm in drop:
            continue
        comps.setdefault(find(nm), []).append(nm)
    for root, members in comps.items():
        cur = next((fam.get(x, "") for x in members if fam.get(x)), "")
        if not cur:
            continue
        for x in members:
            old = fam.get(x, "")
            fam[x] = cur
            if old and old != cur:
                notes[x] = (f"同批查重归族：{old} → {cur}"
                            f"（与 {root} 等同批 |ρ|≥{TIER_T3:g}）")
    return drop, fam, notes


def register_panels(lib, panels, defs, returns_panel=None, source="", *,
                    names=None, kind="raw", on_fail=None,
                    check_dup: bool = True, dup_corr: float = 0.7,
                    write_panel: bool = True,
                    families: dict[str, str] | None = None,
                    batch_dup: bool = True,
                    reject_dup: bool = True):
    """批量入库：对每个定义名做 standardize_zscore + FactorLibrary.register。

    Args:
        defs: {name: formula}。
        names: 注册子集（None=全部）。
        returns_panel: 收益面板；``None``（推荐）表示走库标准收益
            （:meth:`FactorLibrary.default_returns_panel`）。自带面板只有在与
            库内默认值逐值一致时才会被认成 ``adj_close_pct_v1``。
        source: register 的 source 字段。
        on_fail: 可选回调 ``on_fail(name, exc)``，缺省打印告警继续。
        check_dup: 入库时查库内冗余（2026-11-12 起默认 True）。
        dup_corr: 触发"疑似冗余"告警的相关阈值。
        write_panel: ``False`` 表示调用方已按 builder 惯例写好
            ``panels/<name>.parquet``，不再落 ``panels/<slug>.parquet`` 副本。
        families: {name: 因子族}。省略时沿用 registry 里已有的 family；
            本批内 ``|ρ| ≥ 0.85`` 的因子会被**自动归入同一族**并写 note。
        batch_dup: 是否做同批两两查重（默认 True）。库内冗余索引只是快照，
            同批新因子彼此不在索引里，不做这一步同族/等价关系就被漏掉。
        reject_dup: 秩空间等价时是否跳过入库（默认 True，与 ``register`` 同义）。

    Returns:
        注册成功的 ``row`` 列表（等价重复者不在其中）。
    """
    from factor.preprocessing import standardize_zscore

    names = list(defs) if names is None else list(names)
    if returns_panel is not None and hasattr(lib, "_ret_basis_for"):
        if lib._ret_basis_for(returns_panel) == "caller_supplied":
            log.warning("本批自带收益面板与库内默认收益不一致 ⇒ ret_basis 记 "
                        "caller_supplied，晋升第 2 关 stale_caliber 会挡。"
                        "新脚本请传 returns_panel=None 走库标准收益"
                        "（复权收盘 1 日远期 + tradable_t0 掩码）。")
    prepared: dict[str, pd.DataFrame] = {}
    for name in names:
        p = panels.get(name)
        if p is None or p.notna().sum().sum() == 0:
            log.warning("跳过 %s（无有效数据）", name)
            continue
        std = standardize_zscore(p)
        if std.notna().sum().sum() == 0:
            log.warning("跳过 %s（标准化后全 NaN：截面 std=0，可能为恒定值因子）", name)
            continue
        prepared[name] = std

    # 族线索：registry 已有值兜底，调用方显式传入优先
    fam_base: dict[str, str] = {}
    try:
        reg_all = lib.list_all() if hasattr(lib, "list_all") else None
        if reg_all is not None and "family" in reg_all.columns:
            known = dict(zip(reg_all["name"], reg_all["family"]))
            fam_base = {n: str(v) for n, v in known.items()
                        if n in prepared and isinstance(v, str) and v}
    except Exception as exc:  # noqa: BLE001
        log.warning("读取库内 family 失败（%s），本批只按调用方传入归族", exc)
    fam_base.update({k: v for k, v in (families or {}).items() if v})

    drop: dict[str, tuple[str, float, float]] = {}
    fam = fam_base
    notes: dict[str, str] = {}
    if batch_dup and len(prepared) > 1:
        drop, fam, notes = _same_batch_dup(list(prepared), prepared, fam_base)
        if not reject_dup:
            drop = {}   # 只告警不拒收（日志已在上一步打出）

    reg_rows = []
    for name in prepared:
        if name in drop:
            continue
        try:
            row = lib.register(
                name=name, panel=prepared[name], returns_panel=returns_panel,
                kind=kind, formula=defs[name], source=source,
                family=fam.get(name, ""),
                check_dup=check_dup, dup_corr=dup_corr,
                reject_dup=reject_dup, write_panel=write_panel,
            )
        except Exception as e:  # noqa: BLE001
            if on_fail:
                on_fail(name, e)
            else:
                log.warning("注册失败 %s: %s（继续其余因子）", name, e)
            continue
        reg_rows.append(row)
        log.info("  已入库 %s（IC=%.4f, t_nw=%.2f, best_sharpe=%.3f@%s, dup=%s←%s）",
                 name, row["ic_mean"], row["t_stat_nw"],
                 row["best_sharpe"], row["best_config"],
                 row.get("dup_corr_max") if row.get("dup_corr_max") not in ("", None) else "—",
                 row.get("dup_top") or "—")
        # 归族要写回**所有**成员（包括先入库的那个），否则 max_per_family 只看
        # 后入者的族，同族上限形同虚设。
        if fam.get(name) and fam.get(name) != row.get("family", ""):
            lib.set_tag(name, family=fam[name])
        if name in notes:
            lib.set_tag(name, note=notes[name])
    return reg_rows


def warmup_begin(begin: int) -> int:
    """begin 前推约 2 个自然年（覆盖 alpha 公式最大 250 交易日回看）。"""
    y, md = divmod(begin, 10000)
    return max((y - 2) * 10000 + md, 20190102)


def verify_overlap(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """新旧面板重叠区一致性校验。

    Returns:
        {n, max_abs, scale, norm}：n=双方均非 NaN 的对比点数；scale=旧值
        典型量级（重叠区 |值| 中位数）；norm=max_abs/scale（归一化偏差，
        < _OVERLAP_TOL 视为同一口径）。
    """
    idx = new.index.intersection(old.index)
    cols = new.columns.intersection(old.columns)
    if len(idx) == 0 or len(cols) == 0:
        return {"n": 0, "max_abs": float("nan"), "scale": float("nan"),
                "norm": float("nan")}
    a = new.loc[idx, cols]
    b = old.loc[idx, cols]
    mask = (a.notna() & b.notna()).to_numpy()
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "max_abs": float("nan"), "scale": float("nan"),
                "norm": float("nan")}
    diff = (a - b).abs().to_numpy()[mask]
    bvals = b.to_numpy()[mask]
    max_abs = float(diff.max())
    scale = float(np.median(np.abs(bvals)))
    norm = max_abs / scale if scale > 0 else max_abs
    return {"n": n, "max_abs": max_abs, "scale": scale, "norm": norm}


def extend_gp_factors(lib, gp_rows, panel, returns, begin, end,
                      verify_only: bool = False, force: bool = False):
    """按原始口径重算 GP 公式因子并覆盖入库。

    ``returns=None`` 走库标准收益（``default_returns_panel``）；生产调用方
    一律传 ``None``，只有单测需要注入合成收益时才显式给面板。

    Returns:
        (ok, skipped, failed)：成功（或校验通过）名单 / 口径不一致跳过名单 /
        公式重算失败名单。
    """
    from factor.formula import formula_builder

    feats = list(panel.keys())
    b0, e0 = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for _, row in gp_rows.iterrows():
        name = str(row["name"])
        try:
            fp = formula_builder(str(row.get("formula") or name), features=feats)(panel)
        except Exception as e:  # noqa: BLE001
            log.warning("公式重算失败 %s: %s", name, str(e)[:120])
            failed.append(name)
            continue
        fp = fp.loc[b0:e0]
        old = lib.get_panel(name)
        if old is not None and not old.empty:
            v = verify_overlap(fp, old)
            if v["n"] == 0 or (v["norm"] >= _OVERLAP_TOL and not force):
                log.warning("历史口径不一致，跳过 %s（重叠 %d 点，归一化偏差 %.2e）",
                            name, v["n"], v["norm"])
                skipped.append(name)
                continue
            log.info("校验通过 %s（%d 点，归一化偏差 %.2e）", name, v["n"], v["norm"])
        if verify_only:
            ok.append(name)
            continue
        lib.register(
            name, fp, returns,
            kind=str(row.get("kind") or "raw"),
            formula=str(row.get("formula") or name),
            source=str(row.get("source") or ""),
            family=str(row.get("family") or ""),
            frequency=str(row.get("frequency") or ""),
            maturity=str(row.get("maturity") or "experimental"),
            note=str(row.get("note") or ""),
        )
        ok.append(name)
    return ok, skipped, failed


def extend_alpha_factors(lib, cache, uni, index_code: str, sets: list[str],
                         begin: int, end: int, warmup: int,
                         verify_only: bool = False) -> int:
    """重算 alpha101/alpha191/alpha158/alpha360 全集并覆盖入库（与 build_alpha_factors 同路径）。"""
    from factor.alpha101 import compute_alpha101
    from factor.alpha158 import compute_alpha158, compute_alpha360
    from factor.alpha191 import compute_alpha191
    from factor.alpha_base import AlphaData, SET_LABELS, load_alpha_panels

    panels_px, industry, _ = load_alpha_panels(cache, uni, index_code, warmup, end)
    d = AlphaData(panels_px, industry=industry)
    b0, e0 = pd.Timestamp(str(begin)), pd.Timestamp(str(end))

    _compute_fn = {
        "alpha101": compute_alpha101,
        "alpha191": compute_alpha191,
        "alpha158": compute_alpha158,
        "alpha360": compute_alpha360,
    }

    n_total = 0
    for s in sets:
        fn = _compute_fn[s]
        label = SET_LABELS[s]
        log.info("重算 %s（%s）...", s, label)
        # 与 build_alpha_factors 同口径：裁剪后剔除全 NaN 行列（不在册期股票的
        # 死列会虚增覆盖率分母，触发假 coverage_drop 告警）
        computed = {
            k: p.loc[b0:e0].dropna(axis=1, how="all").dropna(axis=0, how="all")
            for k, p in fn(d).items()
        }
        if verify_only:
            log.info("%s 校验模式：可重算 %d 个（不入库）", s, len(computed))
            n_total += len(computed)
            continue
        defs = {k: f"{label} #{k.split('_')[-1]}" for k in computed}
        rows = register_panels(
            lib, computed, defs,
            source=f"{s}:build_alpha_factors:{begin}-{end}")
        n_total += len(rows)
        log.info("%s 完成: %d 个已覆盖入库", s, len(rows))
    return n_total
