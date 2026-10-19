"""全A builder 共用件（2026-09-11 自 11 个 build_alla_* builder 收敛）。

此前 ``merge_outputs`` / ``fuse_horizon_ic`` 各有 11 份拷贝（差异仅为
factor_stats jsonl 文件名与两处小开关）、``ffill_pit_multi`` 4 份、
``KEEP_FROM`` / ``HORIZONS`` / ``IC_CODE_STRIDE`` 常量 11 份，且 status 一份
已经漂移（内联重建 close_adj、硬编码缓存根路径绕过 Config）。本模块是它们的
唯一实现。

口径（与既有拷贝逐位一致）：
- **close_adj 构建**：daily_all_a × backward_factor，交集对齐，float32，
  从 :data:`KEEP_FROM` 起（单一缓存根 = ``Config.cache()["root"]``）；
- **registry 合并**：统一走 ``FactorLibrary.upsert_rows()``（原子写、列序归一
  到 CANONICAL_COLUMNS、只补空缺不覆盖人工标签）；仅在库根推断失败时才回退
  历史直写（append + ``keep="last"``、utf-8-sig）；
- **IC 融合**：对本次 stats 里的因子名重算 ``ic_h{h}``，
  ``fwd = close_adj.pct_change(h).shift(-h)``，因子面板按
  ``IC_CODE_STRIDE`` 抽稀取列（与既有 ic_h 同口径）。

调用方式：``from scripts.builders import common`` 后走 ``common.<fn>``，
避免各 builder 再持本地拷贝。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
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
#: ic 融合的 coverage 下限——与 panels_neu 中性化的 MIN_COVERAGE 对齐，
#  保证「ic_h 有列 ⟹ panels_neu 有文件」，方案 B 存在性门槛不再误触
MIN_IC_COVERAGE = 0.3


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
    # coverage 对齐（09-25）：coverage < MIN_IC_COVERAGE 的因子不进 ic_h——
    # 否则 ic_h 有列而 panels_neu 无文件（中性化按 coverage 过滤），触发
    # rolling_grid 方案 B 存在性门槛（alpha191_138 等四因子实测）。
    reg_p = ds_dir / "registry.csv"
    if reg_p.exists():
        reg = pd.read_csv(reg_p)
        cov = reg.drop_duplicates(subset="name", keep="last").set_index("name")["coverage"]
        low = [n for n in names if float(cov.get(n, 0.0)) < MIN_IC_COVERAGE]
        if low:
            log.info("ic_h%d 跳过低覆盖因子 %d 个（<%.2f）: %s",
                     h, len(low), MIN_IC_COVERAGE, ", ".join(low[:6]))
            names = [n for n in names if n not in set(low)]
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


def _lib_for(ds_dir: Path):
    """由数据集目录反推 FactorLibrary；推断不一致时返回 None（调用方回退直写）。"""
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(root=ds_dir.parent, dataset=ds_dir.name)
    if Path(lib.root).resolve() != ds_dir.resolve():
        log.warning("库根推断不一致（%s != %s），本批次回退直写 registry",
                    lib.root, ds_dir)
        return None
    return lib


def merge_outputs(ds_dir: Path, family: str, *,
                  horizons: tuple[int, ...] = HORIZONS,
                  close_adj_loader=None, ic_code_stride: int = IC_CODE_STRIDE,
                  skip_existing: bool = False, empty_log: str | None = None,
                  source: str = "", family_tag: str = "", kind: str = "raw",
                  maturity: str = "experimental",
                  frequency: str = "日频") -> None:
    """本次产出并入 registry + 融合 ic_h{h}——builder 收尾的唯一入口。

    Args:
        family: 因子集名（决定 ``factor_stats_{family}.jsonl`` 文件名，同时
            写入 registry 的 ``set`` 列）。注意它**不是**因子族标签——
            族标签是 ``family_tag``。
        empty_log: stats 为空时的 warning 文案；None 则静默返回（与各拷贝
            原行为一一对应）。
        source: 本批来源标注，形如 ``alpha101:builders:all_a_2018_2026``。2026-09-12
            起应显式传入——空来源会让这批因子在
            ``scripts/ingest/extend_factor_library.py`` 的因子集路由
            （``source.split(":")[0]``）里被静默跳过。**前缀必须是因子集名**
            （gp/model/alpha101…），不能写成通用的 ``builders``：路由是按
            前缀判定的，统一前缀会让 gp/model 因子重算被跳过。
        family_tag: 因子族标签（受控词表，见 ``FAMILY_WHITELIST``）。
        kind / maturity / frequency: 轻量登记的默认标签。

    写入统一走 ``FactorLibrary.upsert_rows()``：原子写 + 列序归一 + 只补空缺，
    不再直接 ``to_csv`` 覆盖 registry（那条路径正是 912 个因子
    ``source``/``family``/``kind`` 全空的成因）。
    """
    rows = read_stats(ds_dir, family)
    if not rows:
        if empty_log:
            log.warning(empty_log)
        return
    # 默认来源 = <因子集>:builders:<数据集>，默认族 = SET_TO_FAMILY[因子集]。
    # 故意不在 12 个 builder 里各写一遍：漏改一个就等于那批因子重新落入
    # "有来源无分类"（2026-09-12 那个坑），集中在这里由库统一兜底。
    # 前缀必须是因子集名而非 "builders"——source.split(":")[0] 是路由键，
    # 见 research.factor_library.source_for 的说明。
    if not source:
        from research.factor_library import source_for

        source = source_for(family, scope=ds_dir.name)
    if not family_tag:
        from research.factor_library import family_for_set

        family_tag = family_for_set(family)
        if not family_tag:
            log.warning("因子集 %r 未收录在 SET_TO_FAMILY，本批 family 留空"
                        "（新增集请同步该表）", family)
    lib = _lib_for(ds_dir)
    if lib is not None:
        stat = lib.upsert_rows(rows, source=source, family=family_tag, kind=kind,
                               maturity=maturity, frequency=frequency,
                               set_name=family)
        log.info("registry: %d -> %d 因子（新增 %d, 更新 %d）",
                 stat["before"], stat["after"], stat["added"], stat["updated"])
    else:
        # 兜底：库根推断不出来（非常规目录布局）时保持历史直写行为，
        # 但明确告警——不静默退回"没有来源管理"的老路。
        rp = ds_dir / "registry.csv"
        reg = pd.read_csv(rp) if rp.exists() else pd.DataFrame()
        before = len(reg)
        reg = (pd.concat([reg, pd.DataFrame(rows)], ignore_index=True)
                 .drop_duplicates(subset="name", keep="last"))
        reg.to_csv(rp, index=False, encoding="utf-8-sig")
        log.info("registry(直写回退): %d -> %d 因子", before, len(reg))
    for h in horizons:
        fuse_horizon_ic(ds_dir, h, family, close_adj_loader=close_adj_loader,
                        ic_code_stride=ic_code_stride, skip_existing=skip_existing)


# ---------------------------------------------------------------------------
# 因子层中性化面板（neutralized / fundind 两个 90% 重复脚本的唯一实现）
# ---------------------------------------------------------------------------
#: 进入中性化的 registry coverage 下限（与 MIN_IC_COVERAGE 对齐）
MIN_NEU_COVERAGE = 0.3
_ROOT = Path(__file__).resolve().parents[2]
_NEU_BASE_DIR = _ROOT / "reports" / "alla_rolling" / "_base"


def neutralize_one(task: tuple[str, str, str, bool]) -> tuple[str, float]:
    """处理单个因子：去极值 → 行业(+市值)中性化 → zscore → 原子写盘。

    task = (factor_name, ds_dir, out_subdir, industry_only)。顶层函数便于
    ``ProcessPoolExecutor`` pickle；``industry_only=True`` 用于 E5 fundind
    变体（基本面/股东族只剥行业、保留市值风格）。
    """
    name, ds_str, out_subdir, industry_only = task
    from factor.preprocessing import preprocess_factor

    ds = Path(ds_str)
    base = _NEU_BASE_DIR
    p = pd.read_parquet(ds / "panels" / f"{name}.parquet")
    mc = pd.read_parquet(base / "market_cap.parquet")
    ind = pd.read_parquet(base / "cov_industry.parquet")
    common = mc.index.intersection(p.index)
    p = p.reindex(index=common)
    mc = mc.reindex(index=common, columns=p.columns)
    ind = ind.reindex(index=common, columns=p.columns)
    if industry_only:
        # E5：只剥行业，保市值（价值/红利）风格信息
        x = preprocess_factor(p, industry_panel=ind)
    else:
        x = preprocess_factor(p, market_cap_panel=mc, industry_panel=ind)
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32).clip(-10.0, 10.0)
    cov = float(x.notna().mean().mean())
    # 原子写 + 抗瞬时占用（Windows os.replace 撞杀软/索引器；见 atomic_io）
    from scripts.common.atomic_io import write_parquet_atomic
    write_parquet_atomic(x, ds / out_subdir / f"{name}.parquet")
    return name, cov


def run_neutralization(ds: Path, out_subdir: str, *,
                       workers: int = 6,
                       only: list[str] | None = None,
                       force: bool = False,
                       set_name: str = "neutralized",
                       label: str = "中性化",
                       stats_file: str | None = None,
                       industry_only: frozenset[str] = frozenset()) -> int:
    """按 registry 批量构建中性化面板（两脚本共用；返回处理因子数）。"""
    from scripts.common.atomic_io import sweep_stale_tmp

    reg = pd.read_csv(ds / "registry.csv")
    names = (only if only else
             reg.loc[reg["name"].isin(reg["name"]), "name"].tolist())
    if not only:
        # drop_duplicates：registry 偶发重复名会让 .get 返回 Series
        cov_ok = (reg.drop_duplicates(subset="name", keep="last")
                  .set_index("name")["coverage"])
        names = [n for n in names if float(cov_ok.get(n, 0)) >= MIN_NEU_COVERAGE]
    out_dir = ds / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    _n_tmp = sweep_stale_tmp(out_dir)
    if _n_tmp:
        print(f"清理无主 tmp 残渣: {_n_tmp} 个", flush=True)
    done = set() if force else {f.stem for f in out_dir.glob("*.parquet")}
    todo = [n for n in names if n not in done]
    # registry 行存在但 panels/ 缺失：跳过，不让单点缺失崩整批
    _no_panel = [n for n in todo if not (ds / "panels" / f"{n}.parquet").exists()]
    if _no_panel:
        print(f"跳过 {len(_no_panel)} 个无原始面板的因子: "
              f"{', '.join(_no_panel[:8])}", flush=True)
        todo = [n for n in todo if n not in set(_no_panel)]
    print(f"{label}待处理因子: {len(todo)}；已存在跳过: {len(done)}", flush=True)

    t0 = time.time()
    stats = []
    tasks = [(n, str(ds), out_subdir, n in industry_only) for n in todo]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (name, cov) in enumerate(ex.map(neutralize_one, tasks), 1):
                stats.append({"name": name, "set": set_name, "coverage": cov})
                print(f"[{i}/{len(tasks)}] {name} cov={cov:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            _name, cov = neutralize_one(task)
            stats.append({"name": _name, "set": set_name, "coverage": cov})
            print(f"[{i}/{len(tasks)}] {_name} cov={cov:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    (ds / (stats_file or f"factor_stats_{set_name}.jsonl")).open(
        "w", encoding="utf-8").write(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in stats) + "\n")
    print(f"{label}面板完成 {len(stats)} 个因子 ({time.time()-t0:.0f}s)", flush=True)
    return len(stats)
