"""因子库延长与批量入库公共操作（build/extend 家族共用）。

从 scripts 层下沉的因子库维护逻辑：标准化批量入库（``register_panels``）、
历史口径一致性校验（``verify_overlap``）与面板延长（``extend_gp_factors`` /
``extend_alpha_factors``）。scripts 只保留 CLI 入口与数据源解析。

依赖纪律：本模块不依赖 scripts（scripts 是顶层）；``register_panels`` 由
``scripts.cli_common`` 转出口保持既有导入面不变。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("factor_extension")

# 重叠区归一化偏差容差（同数据同算子的浮点噪声水平）
_OVERLAP_TOL = 1e-6


def warmup_begin(begin: int) -> int:
    """begin 前推约 2 个自然年（覆盖 alpha 公式最大 250 交易日回看）。"""
    y, md = divmod(begin, 10000)
    return max((y - 2) * 10000 + md, 20190102)


def register_panels(lib, panels, defs, returns_panel, source, *,
                    names=None, kind="raw", on_fail=None):
    """批量入库：对每个定义名做 standardize_zscore + FactorLibrary.register。

    Args:
        defs: {name: formula}。
        names: 注册子集（None=全部）。
        source: register 的 source 字段。
        on_fail: 可选回调 ``on_fail(name, exc)``，缺省打印告警继续。

    Returns:
        注册成功的 ``row`` 列表。
    """
    from factor.preprocessing import standardize_zscore

    names = list(defs) if names is None else list(names)
    reg_rows = []
    for name in names:
        p = panels.get(name)
        if p is None or p.notna().sum().sum() == 0:
            log.warning("跳过 %s（无有效数据）", name)
            continue
        std = standardize_zscore(p)
        if std.notna().sum().sum() == 0:
            log.warning("跳过 %s（标准化后全 NaN：截面 std=0，可能为恒定值因子）", name)
            continue
        try:
            row = lib.register(
                name=name, panel=std, returns_panel=returns_panel,
                kind=kind, formula=defs[name], source=source,
            )
        except Exception as e:  # noqa: BLE001
            if on_fail:
                on_fail(name, e)
            else:
                log.warning("注册失败 %s: %s（继续其余因子）", name, e)
            continue
        reg_rows.append(row)
        log.info("  已入库 %s（IC=%.4f, t_nw=%.2f, best_sharpe=%.3f@%s）",
                 name, row["ic_mean"], row["t_stat_nw"],
                 row["best_sharpe"], row["best_config"])
    return reg_rows


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

    panels_px, industry, close_adj = load_alpha_panels(cache, uni, index_code, warmup, end)
    d = AlphaData(panels_px, industry=industry)
    b0, e0 = pd.Timestamp(str(begin)), pd.Timestamp(str(end))
    returns = close_adj.loc[:e0].pct_change().shift(-1).loc[b0:e0]

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
            lib, computed, defs, returns,
            source=f"{s}:build_alpha_factors:{begin}-{end}")
        n_total += len(rows)
        log.info("%s 完成: %d 个已覆盖入库", s, len(rows))
    return n_total
