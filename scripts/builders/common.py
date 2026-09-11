"""全A builder 共用件（2026-09-11 自 11 个 build_alla_* builder 收敛）。

此前 ``merge_outputs`` / ``fuse_horizon_ic`` 各有 11 份拷贝（差异仅为
factor_stats jsonl 文件名与两处小开关）、``ffill_pit_multi`` 4 份、
``KEEP_FROM`` / ``HORIZONS`` / ``IC_CODE_STRIDE`` 常量 11 份，且 status 一份
已经漂移（内联重建 close_adj、硬编码缓存根路径绕过 Config）。本模块是它们的
唯一实现。

口径（与既有拷贝逐位一致）：
- **close_adj 构建**：daily_all_a × backward_factor，交集对齐，float32，
  从 :data:`KEEP_FROM` 起（单一缓存根 = ``Config.cache()["root"]``）；
- **registry 合并**：append + ``drop_duplicates(subset="name", keep="last")``，
  utf-8-sig 落盘；
- **IC 融合**：对本次 stats 里的因子名重算 ``ic_h{h}``，
  ``fwd = close_adj.pct_change(h).shift(-h)``，因子面板按
  ``IC_CODE_STRIDE`` 抽稀取列（与既有 ic_h 同口径）。

调用方式：``from scripts.builders import common`` 后走 ``common.<fn>``，
避免各 builder 再持本地拷贝。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common.cli_common import setup_logging

log = setup_logging("builders.common")

#: 因子面板/IC 的统一起点（11 个 builder 曾各自复制同一字面量）
KEEP_FROM = "2016-07-01"
#: IC 融合的 horizon 族
HORIZONS = (1, 5, 10, 20)
#: IC 抽稀取列步长（全市场 5500+ 股，逐日 Spearman 抽 1/3 列加速）
IC_CODE_STRIDE = 3


def load_close_adj(with_raw: bool = False):
    """全A 复权收盘面板（date×code，float32，KEEP_FROM 起）——builder 唯一构建入口。

    Returns:
        ``close_adj``；``with_raw=True`` 时返回 ``(close_adj, close_raw)``。
    """
    from config import Config

    cache_root = Path(str(Config.cache()["root"]))
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(daily.index.levels[0].normalize(), level=0)
    k0 = pd.Timestamp(KEEP_FROM)
    daily = daily[daily.index.get_level_values(0) >= k0]

    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf.index = bf.index.normalize()
    bf = bf.loc[bf.index >= k0]

    close_raw = daily["close"].unstack()
    # 只保留股票交集：bf 宽表含基金等非股票列，按 bf.columns 反向扩张会把
    # 财务表里的基金一并带进因子面板（2026-09-07 清理 32 只 ETF 列的根源）
    cols = close_raw.columns.intersection(bf.columns)
    close_raw = close_raw.reindex(close_raw.index.intersection(bf.index), columns=cols)
    bf = bf.reindex(index=close_raw.index, columns=close_raw.columns)
    close_adj = (close_raw * bf).astype(np.float32)
    return (close_adj, close_raw) if with_raw else close_adj


def ffill_pit_multi(series_long: pd.DataFrame, cal_idx: pd.DatetimeIndex,
                    codes: pd.Index, value_cols: list[str]) -> dict[str, pd.DataFrame]:
    """长表 (code, eff, c1, c2, ...) → {c: date×code} 宽表（按生效日 PIT ffill）。

    同一长表一次性产出多个因子列，避免重复 groupby/ffill。
    series_long 按 code 分组后 eff 无重复（调用方保证）。
    """
    frames = {c: pd.DataFrame(np.nan, index=cal_idx, columns=codes) for c in value_cols}
    for code, g in series_long.groupby("code"):
        g = g.dropna(subset=["eff"])
        if g.empty:
            continue
        g = (g.sort_values("eff").drop_duplicates(subset="eff", keep="last"))
        if g.empty:
            continue
        idx = g.set_index("eff")
        for c in value_cols:
            s = idx[c].reindex(cal_idx, method="ffill")
            frames[c][code] = s.values
    return frames


def read_stats(ds_dir: Path, family: str) -> list[dict]:
    """读取本次产出的 ``factor_stats_{family}.jsonl``（逐因子统计行）。"""
    p = ds_dir / f"factor_stats_{family}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in
            p.read_text(encoding="utf-8").splitlines() if line.strip()]


def fuse_horizon_ic(ds_dir: Path, h: int, family: str, *,
                    close_adj_loader=None, ic_code_stride: int = IC_CODE_STRIDE,
                    skip_existing: bool = False) -> None:
    """把本次 stats 里因子名的 IC 融入 ``ic_h{h}.parquet``。

    Args:
        close_adj_loader: 惰性返回 close_adj 面板的可调用（缺省
            :func:`load_close_adj`；个别 builder 用自身 load_panels 的头元素）。
        skip_existing: True 时跳过 ic_h 里已存在的因子列（constructed/style
            口径）；False 覆盖重算（其余 builder 口径）。
    """
    from stats.ic import calc_ic_series

    names = [s["name"] for s in read_stats(ds_dir, family)]
    if not names:
        return
    ic = pd.read_parquet(ds_dir / f"ic_h{h}.parquet")
    close_adj = (close_adj_loader or load_close_adj)()
    fwd = close_adj.pct_change(h, fill_method=None).shift(-h)
    ic_codes = close_adj.columns[::ic_code_stride]
    for n in names:
        if skip_existing and n in ic.columns:
            continue
        p = pd.read_parquet(ds_dir / "panels" / f"{n}.parquet")
        ic[n] = calc_ic_series(p[ic_codes], fwd).reindex(ic.index)
    ic = ic.astype(np.float32)
    ic.to_parquet(ds_dir / f"ic_h{h}.parquet")
    log.info("ic_h%d merged: %d 因子", h, ic.shape[1])


def merge_outputs(ds_dir: Path, family: str, *,
                  horizons: tuple[int, ...] = HORIZONS,
                  close_adj_loader=None, ic_code_stride: int = IC_CODE_STRIDE,
                  skip_existing: bool = False, empty_log: str | None = None) -> None:
    """本次产出并入 registry + 融合 ic_h{h}——builder 收尾的唯一入口。

    Args:
        family: 因子族名（决定 ``factor_stats_{family}.jsonl``）。
        empty_log: stats 为空时的 warning 文案；None 则静默返回（与各拷贝
            原行为一一对应）。
    """
    rows = read_stats(ds_dir, family)
    if not rows:
        if empty_log:
            log.warning(empty_log)
        return
    reg = pd.read_csv(ds_dir / "registry.csv")
    new_df = pd.DataFrame(rows)
    before = len(reg)
    reg = (pd.concat([reg, new_df], ignore_index=True)
             .drop_duplicates(subset="name", keep="last"))
    reg.to_csv(ds_dir / "registry.csv", index=False, encoding="utf-8-sig")
    log.info("registry: %d -> %d 因子", before, len(reg))
    for h in horizons:
        fuse_horizon_ic(ds_dir, h, family, close_adj_loader=close_adj_loader,
                        ic_code_stride=ic_code_stride, skip_existing=skip_existing)
