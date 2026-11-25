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
  到 CANONICAL_COLUMNS、**统计列刷新 + 标签列只补空缺**，人工标签不覆盖）；
  仅在库根推断失败时才回退历史直写（append + ``keep="last"``、utf-8-sig）；
- **IC 融合**：对本次 stats 里的因子名重算 ``ic_h{h}``，
  ``fwd = close_adj.pct_change(h).shift(-h)``，因子面板按
  ``IC_CODE_STRIDE`` 抽稀取列（与既有 ic_h 同口径）。

调用方式：``from scripts.builders import common`` 后走 ``common.<fn>``，
避免各 builder 再持本地拷贝。
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data.atomic_io import write_parquet_atomic  # noqa: E402
from data.financials import to_datetime_col as to_datetime_col  # noqa: E402,F401
from scripts.common.cli_common import setup_logging

log = setup_logging("builders.common")

#: 因子面板/IC 的统一起点（11 个 builder 曾各自复制同一字面量）
KEEP_FROM = "2016-07-01"
#: IC 融合的 horizon 族
HORIZONS = (1, 5, 10, 20)
#: IC 抽稀取列步长（全市场 5500+ 股，逐日 Spearman 抽 1/3 列加速）
IC_CODE_STRIDE = 3

#: 入库前截断档位。历史上各 builder 各写一个魔法数（100 / 1000 / 1e4 / 1e11），
#: 同一「防异常值」动作有五套口径，raw-value 模型下不可比。这里集中定义，
#: builder 按因子语义选档；**不要**再在脚本里写裸数字。
#: ⚠️ 金额/市值类不能用 ``PCT``（会把整列钉在上限），选 ``MONEY``（≈不截断）。
CLIP = {
    "RATIO": (-10.0, 10.0),      # 标准化 z 值 / 对数比率
    "PCT": (-100.0, 100.0),      # 百分比、占比、增速
    "COUNT": (-1000.0, 1000.0),  # 事件计数、幅度
    "MONEY": (-1e11, 1e11),      # 金额、市值（实际不截断，仅为统一接口）
    "WIDE": (-1e4, 1e4),         # 历史宽档（混合语义，新因子请按语义选档）
}


def write_family_panels(
    ds_dir: Path,
    panels: dict[str, pd.DataFrame],
    *,
    family: str,
    set_name: str | Callable[[str], str],
    labels: dict[str, str],
    close_adj: pd.DataFrame,
    clip: tuple[float, float] | None = None,
    stats_file: str | None = None,
    order: list[str] | None = None,
    resume: bool = False,
    only: str | None = None,
    merge_kwargs: dict | None = None,
) -> None:
    """一个因子族的统一入库链：写面板 → 算 IC → 追加 stats → 合并 registry。

    此前 10 个 builder 各持一份逐行相同的循环（仅 clip 上限与名字不同），
    已漂移出 5 种 clip 口径。这里合并为一个实现。

    Args:
        panels: ``{因子名: date×code 面板}``。
        family: 落盘族名（``merge_outputs`` 的 family）。
        set_name: stats 行里的 ``set`` 字段；可传函数按因子名分派。
        labels: 因子名 → 中文标签（缺失时用因子名）。
        close_adj: 后复权收盘（算前向收益）。
        clip: 截断上下限，见 :data:`CLIP`；``None`` 不截断。
        stats_file: stats 文件名，缺省 ``factor_stats_<family>.jsonl``。
        order: 写盘顺序；缺省按因子名排序。
        resume: 跳过 stats 里已出现的因子。
        only: 只构建指定因子（同时抑制 registry 合并）。
        merge_kwargs: 透传给 :func:`merge_outputs`。
    """
    from stats.ic import calc_ic_series

    ds_dir = Path(ds_dir)
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / (stats_file or f"factor_stats_{family}.jsonl")
    done = ({json.loads(line)["name"] for line in
             stats_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            if resume and stats_path.exists() else set())

    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h)
           for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    names = [n for n in (order or sorted(panels)) if n in panels]
    if only:
        names = [n for n in names if n == only]

    t0 = time.time()
    for i, name in enumerate(names, 1):
        if name in done:
            continue
        p = panels[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan)
        if clip is not None:
            p = p.clip(*clip)
        write_parquet_atomic(p, panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32)
              for h in HORIZONS}
        row = {"name": name,
               "set": set_name(name) if callable(set_name) else set_name,
               "label": labels.get(name, name),
               "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(names),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    if not only:
        merge_outputs(ds_dir, family, **(merge_kwargs or {}))


def resolve_panel_file(ds: Path, name: str) -> Path | None:
    """定位因子面板文件：先 ``panels/<name>.parquet``，再 ``panels/<slug>.parquet``。

    两条入库通道的落盘口径不同：builder 批量通道写**原名**，而
    ``FactorLibrary.register()`` 写 ``name_slug(name)``（md5 前 12 位）。
    本模块的中性化 / IC 融合此前只认原名，于是走库入口登记的因子
    **永远拿不到中性化面板和 ic 列**（2026-11-12 的 12 个 ``pap_*`` 即如此）。
    统一走本函数，两条通道都能命中。
    """
    from research.factor_library import name_slug

    for p in (ds / "panels" / f"{name}.parquet",
              ds / "panels" / f"{name_slug(name)}.parquet"):
        if p.exists():
            return p
    return None
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


# 日期列解析的唯一实现在 data.financials（PIT 展开也走它），这里只转出口；
# 用 `X as X` 的写法让 linter 明白这是刻意重导出而不是未使用的 import。

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
    """读取本次产出的 ``factor_stats_{family}.jsonl``（逐因子统计行，按名去重）。

    🚨 **必须按 ``name`` 去重取最后一行**：这个文件是 builder 用 ``open("a")``
    **追加**写的（重跑保留历史、也支持断点续跑），所以跑过 N 次就有 N 行同名。
    读回去重前，整批同名行会经 ``merge_outputs`` 原样进 registry ——
    ``upsert_rows`` 只排除"库里已有的"，不拦批内重复，于是首次入库即写 2~3 行。
    2026-10-27 实测 ``all_a_2018_2026``：registry 1112 行 / 950 唯一，152 个名字
    重复（altf 40→20、fundamental 178→86、evt 26→11…），全部非 alpha 族 ——
    alpha 四族只因 ``build_alla_alpha_panels._merge_outputs`` 自己
    ``drop_duplicates(keep="last")`` 后全量覆盖才幸免。

    "最后一行 = 最新统计"的依据：追加顺序即运行顺序，实测 152 组里
    ``active_last`` 可比的 26 组 0 例倒退、``coverage`` 144/152 单调不降。
    """
    p = ds_dir / f"factor_stats_{family}.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in
            p.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_name: dict[str, dict] = {}
    for r in rows:
        nm = r.get("name")
        if nm:
            by_name[nm] = r  # 后者覆盖前者 = keep="last"
    if len(by_name) != len(rows):
        log.debug("read_stats(%s): %d 行 → %d 个唯一因子（jsonl 追加写历史）",
                  family, len(rows), len(by_name))
    return list(by_name.values())


def fuse_horizon_ic(ds_dir: Path, h: int, family: str, *,
                    close_adj_loader=None, ic_code_stride: int = IC_CODE_STRIDE,
                    skip_existing: bool = False,
                    names: list[str] | None = None) -> None:
    """把因子名的 IC 融入 ``ic_h{h}.parquet``。

    Args:
        names: 显式指定要融合的因子名（None=读 ``factor_stats_{family}.jsonl``）。
            走 ``FactorLibrary`` 登记的因子没有 stats jsonl 行，补 ic 列时用本参数。
        close_adj_loader: 惰性返回 close_adj 面板的可调用（缺省
            :func:`load_close_adj`；个别 builder 用自身 load_panels 的头元素）。
        skip_existing: True 时跳过 ic_h 里已存在的因子列（constructed/style
            口径）；False 覆盖重算（其余 builder 口径）。
    """
    from stats.ic import calc_ic_series

    names = [s["name"] for s in read_stats(ds_dir, family)] if names is None \
        else list(names)
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
        pf = resolve_panel_file(ds_dir, n)
        if pf is None:
            log.warning("ic_h%d 跳过 %s：面板不存在", h, n)
            continue
        p = pd.read_parquet(pf)
        # 因子面板的列并不保证等于 close_adj 的列（覆盖率 <1 的因子、按池
        # 子集构建的面板都少代码）；直接 p[ic_codes] 会 KeyError。
        codes = [c for c in ic_codes if c in p.columns]
        if not codes:
            log.warning("ic_h%d 跳过 %s：面板与行情无共同代码", h, n)
            continue
        ic[n] = calc_ic_series(p[codes], fwd).reindex(ic.index)
    ic = ic.astype(np.float32)
    from data.atomic_io import write_parquet_atomic
    write_parquet_atomic(ic, ds_dir / f"ic_h{h}.parquet")
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


def gate_stats_rows(rows: list[dict], ds_dir: Path, family: str, *,
                    panels_subdir: str = "panels"
                    ) -> tuple[list[dict], list[dict]]:
    """入库闸：剔除「当日截面恒常数 / 浮点噪声格点」的面板。

    判据与背景见 ``factor.validation``。这里只做 I/O 编排：
    每个因子读 parquet 最后一个 row group 的末 20 行（~0.05s），代价可忽略。

    放行规则：
      - 面板文件不存在 ⇒ 放行（缺面板由 select 阶段的完整性检查负责，不在这里拦）
      - 读盘异常 ⇒ 放行并告警（不能让闸门把整批构建搞崩）
      - ``sparse`` / ``discrete`` ⇒ 放行（稀疏事件因子与布尔因子是合法形态）

    拒收记录追加到 ``factor_stats_{family}_rejected.jsonl``，便于事后审计
    「这批少了一个因子是因为什么」。

    Args:
        rows: ``read_stats`` 读出的统计行（含 ``name``）。
        ds_dir: 数据集根目录（其下 ``panels/`` 是面板）。
        family: 因子集名，仅用于拒收日志文件名。

    Returns:
        ``(kept, rejected)``。
    """
    from factor.validation import check_panel, describe_reason, read_panel_sample

    panel_dir = ds_dir / panels_subdir
    kept: list[dict] = []
    rejected: list[dict] = []
    for r in rows:
        name = r.get("name")
        if not name:
            continue
        p = panel_dir / f"{name}.parquet"
        if not p.exists():
            kept.append(r)
            continue
        try:
            ok, reason, st = check_panel(read_panel_sample(p))
        except Exception as e:                                    # noqa: BLE001
            log.warning("入库闸读面板失败 %s：%s（放行）", name, e)
            kept.append(r)
            continue
        if ok:
            kept.append(r)
            continue
        rejected.append({**r, "reject_reason": reason, **st})
        log.warning("入库闸拒收 %s：%s（uniq=%d ratio=%.2e flat=%.0f%%）",
                    name, describe_reason(reason), st["uniq"],
                    st["uniq_ratio"], st["flat_frac"] * 100)
    if rejected:
        p = ds_dir / f"factor_stats_{family}_rejected.jsonl"
        with p.open("a", encoding="utf-8") as fh:
            for r in rejected:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.warning("入库闸：%d 个因子被拒（明细 %s）", len(rejected), p.name)
    return kept, rejected


def tag_dup_rows(rows: list[dict], ds_dir: Path, lib, *,
                 panels_subdir: str = "panels",
                 only_new: bool = True,
                 topk: int = 1,
                 time_budget: float = 600.0) -> dict:
    """入库前冗余预检：给**新因子**打上 ``dup_checked`` / ``dup_corr_max`` 等标记。

    为什么只查新增
    --------------
    已有因子的 ``dup_*`` 由 ``scripts.factors.dup_audit --apply`` 全库统一回填
    （950 个约 11 分钟）；每次构建都重算既浪费又会和全库口径打架。增量通常
    只有几个到几十个，单个耗时 **0.15~0.41 s**（与全库相关 <0.85 时只做一次
    签名矩阵乘），只有疑似重复（ρ≥0.85）才额外跑残差回归（**8~13 s**）。

    这是"记账 + 告警"，**不是入库闸**
    --------------------------------
    索引缺失、读盘失败、样本不重叠一律**放行并告警**——查重不能阻塞构建，
    更不能让一次索引过期把整批因子挡在库外。要硬拒收，走
    ``FactorLibrary.ingest(reject_dup=True)``（那是单因子通道）。

    Args:
        rows: ``read_stats`` 读出的统计行（含 ``name``）。
        ds_dir: 数据集根目录，其下 ``panels_subdir`` 是面板。
        lib: ``FactorLibrary`` 实例。
        only_new: True 时只查 registry 里没有的因子。
        time_budget: 总耗时上限（秒）；超了停止并告警，剩余留待 ``dup_audit``。

    Returns:
        dict: ``checked`` / ``dups``（判为重复的名） / ``skipped`` / ``secs``。
    """
    import time as _time

    import pandas as pd

    if lib is None or lib.dup_index() is None:
        log.warning("冗余索引不存在，跳过入库前查重（%d 个待查；"
                    "重建索引：python -m scripts.factors.dup_audit --dataset %s）",
                    len(rows), ds_dir.name)
        return {"checked": 0, "dups": [], "skipped": len(rows), "secs": 0.0}

    existing: set[str] = set()
    if only_new:
        try:
            existing = set(lib._load_registry()["name"].astype(str))
        except Exception as e:                                  # noqa: BLE001
            log.warning("读 registry 失败，按全量查重：%s", e)

    todo = [r for r in rows if r.get("name")
            and (not only_new or str(r["name"]) not in existing)]
    if not todo:
        log.info("入库前查重：本批 %d 个因子均已在库，跳过（dup_* 由 dup_audit 统一维护）",
                 len(rows))
        return {"checked": 0, "dups": [], "skipped": len(rows), "secs": 0.0}

    t0 = _time.time()
    checked, dups, skipped = 0, [], 0
    panel_dir = ds_dir / panels_subdir
    for r in todo:
        if _time.time() - t0 > time_budget:
            log.warning("入库前查重超时（%.0fs），剩余 %d 个未查——"
                        "跑 dup_audit 补齐（超时不影响入库）",
                        time_budget, len(todo) - checked - skipped)
            break
        name = str(r["name"])
        p = panel_dir / f"{name}.parquet"
        if not p.exists():
            skipped += 1
            continue
        try:
            res = lib.precheck_dup(pd.read_parquet(p), name=name, topk=topk)
        except Exception as e:                                  # noqa: BLE001
            log.warning("入库前查重失败 %s：%s（放行）", name, e)
            skipped += 1
            continue
        if not res.get("available"):
            skipped += 1
            continue
        ric, rit = res["resid_ic"], res["resid_t"]
        r["dup_checked"] = True
        r["dup_corr_max"] = res["top_corr"]
        r["dup_top"] = res["top_name"]
        r["resid_ic"] = "" if ric is None or pd.isna(ric) else ric
        r["resid_t_nw"] = "" if rit is None or pd.isna(rit) else rit
        checked += 1
        if res["top_corr"] >= 0.95 and (pd.isna(rit) or abs(rit) < 2.0):
            dups.append({"name": name, "top": res["top_name"],
                         "corr": res["top_corr"], "resid_t": rit})
            log.warning("入库前查重：`%s` 与 `%s` 相关 %.3f 且残差无增量 → 疑似重复",
                        name, res["top_name"], res["top_corr"])

    secs = _time.time() - t0
    if checked or dups:
        log.info("入库前查重：查 %d 个（跳过 %d），判重复 %d 个，用 %.1fs",
                 checked, skipped, len(dups), secs)
    return {"checked": checked, "dups": dups, "skipped": skipped, "secs": secs}


def merge_outputs(ds_dir: Path, family: str, *,
                  horizons: tuple[int, ...] = HORIZONS,
                  close_adj_loader=None, ic_code_stride: int = IC_CODE_STRIDE,
                  skip_existing: bool = False, empty_log: str | None = None,
                  source: str = "", family_tag: str = "", kind: str = "raw",
                  maturity: str = "experimental",
                  frequency: str = "日频",
                  dup_check: bool = True,
                  dup_only_new: bool = True) -> None:
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
        dup_check: 入库前对**新因子**跑冗余预检（默认开）。查到的
            ``dup_corr_max`` / ``dup_top`` / ``resid_*`` 写进 registry 行；
            判为重复只告警，**不拒收**——查重是记账，不是入库闸。
        dup_only_new: 只查 registry 里没有的因子（默认）。关掉会把整批重算，
            950 个约十几分钟，一般不需要。

    写入统一走 ``FactorLibrary.upsert_rows()``：原子写 + 列序归一 +
    **统计列刷新 / 标签列只补空缺**（见 ``research.factor_library.STAT_COLUMNS``），
    不再直接 ``to_csv`` 覆盖 registry（那条路径正是 912 个因子
    ``source``/``family``/``kind`` 全空的成因）。
    """
    rows = read_stats(ds_dir, family)
    if not rows:
        if empty_log:
            log.warning(empty_log)
        return
    # 入库闸：截面恒常数的面板一律不进 registry（见 factor.validation）。
    # 放在 upsert 之前是必须的——registry 一旦写入，select 就会把它当候选。
    rows, _rejected = gate_stats_rows(rows, ds_dir, family)
    if not rows:
        log.warning("入库闸拒收全部 %d 个因子，跳过 registry 写入", len(_rejected))
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
        if dup_check:
            tag_dup_rows(rows, ds_dir, lib, only_new=dup_only_new)
        stat = lib.upsert_rows(rows, source=source, family=family_tag, kind=kind,
                               maturity=maturity, frequency=frequency,
                               set_name=family)
        log.info("registry: %d -> %d 因子（新增 %d, 更新 %d）",
                 stat["before"], stat["after"], stat["added"], stat["updated"])
        # 公式回填：只补本批的 formula（alpha 四族从源码 AST 反解析，其余写中文定义）。
        # 不放在 upsert_rows 里是因为那是库层、要处理多库；放这里只碰本批，且
        # 与 tag_dup_rows 并列——查重与公式都是"入库时顺手记的账"。
        try:
            from scripts.factors.formula_extract import backfill_batch

            backfill_batch(ds_dir, [str(r["name"]) for r in rows if r.get("name")])
        except Exception as e:                                  # noqa: BLE001
            log.warning("公式回填失败（不影响入库）：%s", e)
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

#: 快速失败：处理数达到 ``EARLY_N`` 后，若失败率超过 ``EARLY_FAIL_RATE`` 立即中止。
#: 2026-11-05 实测：6 worker 并发推 57MB parquet（≈22MB/s）打穿 Defender 实时扫描
#: 吞吐，失败率 50%；而整批护栏要等 932 个跑完才判，等于白烧数小时算力。
#: 前 20 个足以定性 —— 系统性写盘故障（并发/权限/杀软）vs 偶发占用。
EARLY_N = 20
EARLY_FAIL_RATE = 0.3
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
    panel_file = resolve_panel_file(ds, name)
    if panel_file is None:
        raise FileNotFoundError(f"面板缺失: {name}（panels/ 下原名与 slug 均不存在）")
    p = pd.read_parquet(panel_file)
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
    from data.atomic_io import write_parquet_atomic
    write_parquet_atomic(x, ds / out_subdir / f"{name}.parquet")
    return name, cov


def _safe_neutralize(task: tuple[str, str, str, bool]
                     ) -> tuple[str, float | None, str | None]:
    """``neutralize_one`` 的包装：把单点失败变成可记录的结果而不是异常。

    🚨 2026-11-05：本机 Defender 实时防护会把刚写完的 57MB parquet 占用数十秒
    （实测重试 42s 仍被拒），一次占用就让 932 个因子的批处理整批中止、后续步骤
    全跳过 —— 已经为此重跑三次。单文件被锁不该让另外 900 多个白跑：
    失败的因子单独落盘记录，由调用方汇总并在下次深检中被发现。
    """
    try:
        name, cov = neutralize_one(task)
        return name, cov, None
    except Exception as e:                                        # noqa: BLE001
        return task[0], None, f"{type(e).__name__}: {e}"


def run_neutralization(ds: Path, out_subdir: str, *,
                       # 本机实测（2026-11-05，57MB/因子）：1 worker 15.5s/个、
                       # 3 worker 6.9s/个（失败率 0）、6 worker 2.6s/个但失败率 50%
                       # —— 瓶颈是 Defender 实时扫描吞吐（≈8MB/s 是这台机器的线），
                       # 不是 CPU。默认值取「能跑通」的那一档。
                       workers: int = 3,
                       only: list[str] | None = None,
                       force: bool = False,
                       set_name: str = "neutralized",
                       label: str = "中性化",
                       stats_file: str | None = None,
                       industry_only: frozenset[str] = frozenset()) -> int:
    """按 registry 批量构建中性化面板（两脚本共用；返回处理因子数）。"""
    from data.atomic_io import sweep_stale_tmp

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
    _no_panel = [n for n in todo if resolve_panel_file(ds, n) is None]
    if _no_panel:
        print(f"跳过 {len(_no_panel)} 个无原始面板的因子: "
              f"{', '.join(_no_panel[:8])}", flush=True)
        todo = [n for n in todo if n not in set(_no_panel)]
    print(f"{label}待处理因子: {len(todo)}；已存在跳过: {len(done)}", flush=True)

    t0 = time.time()
    stats = []
    failed: list[dict] = []

    def _abort_if_systemic(i: int, last: str) -> None:
        """前 ``EARLY_N`` 个里失败率过高 ⇒ 立即中止。

        末端的 10% 护栏只能在跑完 932 个之后判——那时代价已经付掉了。写盘故障
        几乎都集中在批处理开头（并发配置不对/权限/杀软），早判能省下全部算力。
        """
        if i >= EARLY_N and len(failed) / i > EARLY_FAIL_RATE:
            raise RuntimeError(
                f"{label}前 {i} 个因子失败 {len(failed)} 个"
                f"（{len(failed) / i:.0%} > {EARLY_FAIL_RATE:.0%}），最后失败：{last}"
                f" ⇒ 判为系统性写盘故障（并发过高 / 杀软占用 / 权限），已中止。"
                f"建议降低 --workers 或把输出目录加入安全软件扫描排除项。")

    tasks = [(n, str(ds), out_subdir, n in industry_only) for n in todo]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (name, cov, err) in enumerate(ex.map(_safe_neutralize, tasks), 1):
                if err is not None:
                    failed.append({"name": name, "error": err})
                    print(f"[{i}/{len(tasks)}] {name} 写盘失败：{err}",
                          file=sys.stderr, flush=True)
                    _abort_if_systemic(i, name)
                    continue
                stats.append({"name": name, "set": set_name, "coverage": cov})
                print(f"[{i}/{len(tasks)}] {name} cov={cov:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                _abort_if_systemic(i, name)
    else:
        for i, task in enumerate(tasks, 1):
            _name, cov, err = _safe_neutralize(task)
            if err is not None:
                failed.append({"name": _name, "error": err})
                print(f"[{i}/{len(tasks)}] {_name} 写盘失败：{err}",
                      file=sys.stderr, flush=True)
                _abort_if_systemic(i, _name)
                continue
            stats.append({"name": _name, "set": set_name, "coverage": cov})
            print(f"[{i}/{len(tasks)}] {_name} cov={cov:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            _abort_if_systemic(i, _name)

    (ds / (stats_file or f"factor_stats_{set_name}.jsonl")).open(
        "w", encoding="utf-8").write(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in stats) + "\n")
    print(f"{label}面板完成 {len(stats)} 个因子 ({time.time()-t0:.0f}s)", flush=True)

    if failed:
        # 失败清单单独落盘：深检只看日期，不会告诉你「这批少了哪个」。
        list_path = ds / f"{label}_failed_{datetime.now():%Y%m%d_%H%M}.txt"
        list_path.write_text(
            "\n".join(f"{f['name']}\t{f['error']}" for f in failed) + "\n",
            encoding="utf-8")
        rate = len(failed) / max(len(tasks), 1)
        print(f"⚠️ {label}写盘失败 {len(failed)}/{len(tasks)}（{rate:.1%}）"
              f"，清单：{list_path}", file=sys.stderr, flush=True)
        # 少量失败是可容忍的文件占用；大面积失败说明是系统性故障（路径/权限/磁盘），
        # 继续跑只会把整库写成半旧半新，必须让调用方中断。
        if rate > 0.1:
            raise RuntimeError(
                f"{label}失败率 {rate:.1%} 超过 10%，判定为系统性故障，"
                f"中止以免产出半旧半新的面板库：{list_path}")
    return len(stats)
