"""构建因子脚本的公共 CI/CD 骨架。

收敛三个 ``build_*_factors.py`` 脚本中逐字重复的部分（2026-08-17）：
- ``--mock / --offline / --real`` 三态数据源 + 缓存 + 股票池 + 区间/数据集解析
  （2026-08-29 起委托 ``scripts/cli_common.py`` 单一实现）；
- ``returns_from_daily``：``(date, code)`` 长表 → 次日收益面板（与因子库 IC 口径一致）；
- ``register_panels``：批量 ``standardize_zscore + FactorLibrary.register``；
- ``record_experiment_safe``：实验记录（失败不阻断）；
- ``print_no_save``：``--no-save`` 时打印面板概览。

各脚本差异部分（load_data / 指标计算）保留在各脚本内，不强行统一。
"""
from __future__ import annotations

import logging

from scripts.cli_common import (  # noqa: F401  re-export：保持旧 import 路径
    DEFAULT_RANGES,
    add_build_args,
    make_data_context,
)

log = logging.getLogger("build_common")


def returns_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """``(date, code)`` 长表 → 次日收益面板（与因子库 IC 口径一致）。"""
    d = daily.reset_index()
    close_w = d.pivot(index="date", columns="code", values="close").sort_index()
    return close_w.pct_change().shift(-1)


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


def record_experiment_safe(kind, command, params, fingerprint, result_path,
                           metrics, note):
    """写实验记录；失败仅告警不阻断主流程。"""
    from research.experiments import record_experiment
    try:
        record_experiment(kind=kind, command=command, params=params,
                          data_fingerprint=fingerprint, result_path=result_path,
                          metrics=metrics, note=note)
    except Exception as e:  # noqa: BLE001
        log.warning("实验记录写入失败: %s", e)


def print_no_save(names, panels):
    """``--no-save`` 时打印每个面板的形状与非空率。"""
    for name in names:
        p = panels.get(name)
        if p is None:
            log.info("  %-18s 缺失", name)
            continue
        log.info("  %-18s 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                 name, p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
    log.info("未入库（--no-save）。完成")
