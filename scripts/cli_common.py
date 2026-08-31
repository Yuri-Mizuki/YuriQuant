"""
脚本层公共 CLI 骨架
==================

收敛散落在 scripts/ 各入口的 argparse / logging / 三态数据源样板
（2026-08-29 自 ``_build_common`` 推广为全 scripts 通用；build 家族专属助手
``returns_from_daily`` / ``register_panels`` / ``record_experiment_safe`` /
``print_no_save`` 于 2026-08-31 并入本模块，``_build_common`` 已删除）：

- ``setup_logging(name)``      : 统一 ``logging.basicConfig`` 格式并返回 logger
                                  （替代 55 处逐字复制的两行样板）
- ``add_real_mock_args``       : ``--real`` / ``--mock``（可选 ``--offline``）参数对
                                  （替代 21 个脚本各自手写的 add_argument）
- ``add_build_args``           : build 家族完整参数组（--mock/--offline/--index/
                                  --begin/--end/--dataset/--no-save）
- ``make_data_context(args)``  : 三态数据源（mock→MockDataSource 临时缓存 /
                                  offline→OfflineDataSource / 默认 real）+
                                  默认区间/数据集名解析 + Universe
- ``register_panels`` 等 build 助手：批量标准化入库 / 实验记录 / no-save 概览

用法::

    from scripts.cli_common import setup_logging, add_real_mock_args

    log = setup_logging("my_script")

    def main():
        ap = argparse.ArgumentParser(description="...")
        add_real_mock_args(ap)
        ...
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

log = logging.getLogger("cli_common")

# 三态对应的默认区间与数据集名（与三个 build 脚本原默认一致）
DEFAULT_RANGES = {
    "mock": (20230103, 20241231, "mock"),
    "offline": (20250101, 20251231, "hs300_2025"),
    "real": (20250101, 20251231, "hs300_2025"),
}


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """统一日志配置（全 scripts 同一格式），返回以 ``name`` 命名的 logger。"""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    return logging.getLogger(name)


def add_real_mock_args(
    parser: argparse.ArgumentParser,
    offline: bool = False,
    real_help: str = "真实数据（默认 mock）",
    mock_help: str = "用 mock 数据",
) -> argparse.ArgumentParser:
    """追加 ``--real`` / ``--mock``（可选 ``--offline``）参数对。

    仅声明参数与语义，数据分支逻辑由脚本自定（三态数据源请用
    ``make_data_context``）。
    """
    parser.add_argument("--real", action="store_true", help=real_help)
    parser.add_argument("--mock", action="store_true", help=mock_help)
    if offline:
        parser.add_argument("--offline", action="store_true", help="只读本地缓存，不连 SDK")
    return parser


def add_build_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """追加三个 build 脚本共用的命令行参数（含三态数据源开关）。"""
    add_real_mock_args(parser, offline=True,
                       real_help="连 SDK 拉真实数据（默认 offline）")
    parser.add_argument("--index", default="000300.SH")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="因子库数据集名（默认自动推导）")
    parser.add_argument("--no-save", action="store_true", help="只计算不入库")
    return parser


def make_data_context(args):
    """创建数据源/缓存/股票池，并解析区间与数据集名。

    Returns:
        (cache, uni, begin, end, dataset)：cache 为 DataCache，uni 为 Universe。
        调用方需根据自身需求再用 ``cache``/``uni`` 拉取具体数据（如
        ``load_daily`` / ``load_financial_tables`` / 分钟线等）。
    """
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from data.universe import Universe

    if args.mock:
        from tests.conftest import MockDataSource
        ds = MockDataSource()
        cache = DataCache(ds, cache_root=tempfile.mkdtemp(prefix="mock_cache_"))
        mode = "mock"
    elif getattr(args, "offline", False):
        ds = OfflineDataSource()
        cache = DataCache(ds)
        mode = "offline"
    else:
        from data.datasource import create_datasource
        ds = create_datasource()
        cache = DataCache(ds)
        mode = "real"

    d_begin, d_end, d_dataset = DEFAULT_RANGES[mode]
    begin = args.begin or d_begin
    end = args.end or d_end
    dataset = args.dataset or d_dataset
    uni = Universe(cache)
    return cache, uni, begin, end, dataset


# ---------------------------------------------------------------------------
# build 家族专属助手（2026-08-31 自 _build_common 并入）
# ---------------------------------------------------------------------------
def returns_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """``(date, code)`` 长表 → 次日收益面板（与因子库 IC 口径一致）。"""
    d = daily.reset_index()
    close_w = d.pivot(index="date", columns="code", values="close").sort_index()
    return close_w.pct_change().shift(-1)


# register_panels 已下沉 research/factor_extension.py（2026-08-31 D3b），
# 此处转出口保持既有导入面不变
from research.factor_extension import register_panels  # noqa: E402


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
