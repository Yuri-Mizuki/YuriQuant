"""因子库面板月度重建（守卫式入口，计划任务用）。

为什么需要它
------------
定版口径（`alla_daily_rank --preset dingban`）的**慢信号** ``s0.5`` 直接读离线
``ds_root/panels_neu`` 的月末截面（`scripts.evaluation.fundamental_blend::build_slow_panel`），
而该面板是 `build_alla_factor_neutralized` 的**批处理产物、不随新数据自动更新** ——
`config/schedule.yaml` 里此前没有任何任务负责重建它。后果：慢信号长期停在最后一次
手工重建的截面（2026-10-10 建于数据止 09-01），定版分数有一半是陈旧信号
（`scripts/pipelines/alla_daily_rank.py:1814-1820` 的注释已如实披露该覆盖边界）。

重建不是"跑一个脚本"，而是一条**有顺序依赖**的链（依据
`reports/prod_pipeline_gap/report.md` §4–5.6）：

    ① 原始数据（`update_data`）                    —— 由出榜任务负责，本脚本只**检查**
    ② `rolling_grid_alla --stage prep`            → `_base/{close_adj,market_cap,cov_industry}`
    ③ 各因子族 builder（16 个）                    → `ds/panels/`（原始面板，全历史重算）
    ④ `build_alla_factor_neutralized`             → `ds/panels_neu`
    ⑤ `build_alla_factor_neutralized_fundind`     → `ds/panels_neu_fundind`

⚠️ **② 必须在 ④⑤ 之前**：`panels_neu` 与 `_base` **版本耦合** —— `_base` 一更新，
旧 `panels_neu` 就无法由当前数据复现（§5.6 实测：`|Δ|>1e-3` 的天数占 22.91%，
最大 11.34，每天涉及全市场 3000–5000 只）。故两者必须同批重建。

守卫（默认行为，零写盘）
------------------------
`panels_neu` 覆盖末日若已到「上一个自然月的最后一个交易日」，说明慢信号已有
新鲜信号月可用 → **直接空转退出（exit 0）**。⇒ 天然月频、缺了会自愈。
（`task_scheduler` 只支持 `daily`/`minute` 触发器，无 monthly，月频只能靠脚本内守卫。）

⚠️ 未纳入本脚本（如实标注边界）：`registry.csv` / `ic_h*.parquet`（因子 IC 表）与
`reports/alla_rolling_ortho920fundind/selection`（当年特征清单）**不重建** —— 前者影响
`--stage select`，后者是出榜链读的静态清单，均与"面板覆盖到哪一天"正交。

用法::

    python -m scripts.pipelines.rebuild_panel_library                # 体检 + 守判（默认，只读）
    python -m scripts.pipelines.rebuild_panel_library --dry-run      # 打印将执行的命令序列
    python -m scripts.pipelines.rebuild_panel_library --run           # 真正执行（预计数小时）
    python -m scripts.pipelines.rebuild_panel_library --run --force   # 忽略守卫强制重建
    python -m scripts.pipelines.rebuild_panel_library --status        # 只看四层覆盖末日
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.common.power import keep_system_awake  # noqa: E402

log = setup_logging("rebuild_panel_library")

DATASET = "all_a_2018_2026"
BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"
#: 覆盖体检的**快速**抽样（取 min：任一样本因子落后即报落后；避免遍历 950 个 parquet）。
#: ⚠️ 仅作快速体检用；守判另有**全量深检** `_stale_panels`，抽样不可单独当判据。
_PROBE_NEU = ("bp", "alpha158_MA60", "alpha101_001")
_PROBE_RAW = ("bp", "alpha158_MA60", "accruals")

#: 中性化 builder 的覆盖率门槛（`build_alla_factor_neutralized::MIN_COVERAGE`）。
#: 深检必须与它对齐，否则低覆盖因子会被永久误判为"落后"。见 `_in_scope_names`。
MIN_COVERAGE = 0.3

#: 出榜链窗口（小时，左闭右开）：17:30 出榜 / 18:00 另类 / 19:00 日报。
#: 重建期间 `panels_neu` 是"半新半旧"的中间态，此时出榜链的慢信号会读到混合面板
#: ⇒ 启动时若落在此窗口内，跳过本次（守卫是月频的，推迟一天无成本；
#: 也顺带挡住 `StartWhenAvailable` 错过补跑恰好撞进窗口的情形）。
_BUSY_WINDOW = (16, 20)


# ---------------------------------------------------------------------------
# 覆盖探测
# ---------------------------------------------------------------------------
def _ds_root() -> Path:
    from config import Config
    return Path(str(Config.get()["factor_library"]["root"])) / DATASET


def _index_dates(path: Path):
    """读 parquet 的索引日期（只读索引，不加载数据）。

    注意 ``daily_all_a.parquet`` 等原始表是 ``MultiIndex(date, code)`` ——
    直接 ``.max()`` 会返回 tuple、被 ``str()`` 成 ``"(Timestamp(...)"`` 这种
    无法参与日期比较的字符串（会静默把"数据已到位"误判为不足），故统一取 level 0。
    """
    idx = pd.read_parquet(path, columns=[]).index
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(0)
    return pd.DatetimeIndex(idx)


def parquet_last_date(path: Path) -> str:
    """单个 parquet 的索引末日（YYYY-MM-DD）。（公开名；跨模块引用走这里）"""
    try:
        idx = _index_dates(path)
        return str(idx.max().date()) if len(idx) else "空表"
    except Exception as exc:  # noqa: BLE001
        return f"ERR({type(exc).__name__})"


#: 兼容别名（模块内历史调用点沿用；勿在新代码用下划线名跨模块 import）
_parquet_last_date = parquet_last_date


def _panel_coverage(subdir: str, probes: tuple[str, ...]) -> str:
    """返回某面板目录的覆盖末日 = 抽样因子中的**最小值**。

    ⚠️ 必须取 min 而非 max：一个目录"已覆盖到最新"等价于**所有**因子都最新，
    取 max 会让任一新写入的因子掩盖其余全部陈旧因子（2026-10-16 实测：`--only`
    手工刷新 3 个探测因子后，`max` 立刻报 10-15、守判"无需重建"，而 950 个里
    944 个仍停在 09-01）。
    """
    root = _ds_root() / subdir
    if not root.exists():
        return "缺失"
    dates = [_parquet_last_date(root / f"{n}.parquet") for n in probes]
    ok = [d for d in dates if not d.startswith("ERR")]
    if not ok:
        return "全部读取失败"
    if len(ok) < len(dates):
        log.warning("%s 抽样中 %d 个读取失败：%s", subdir, len(dates) - len(ok), dates)
    return min(ok)


def _footer_max(path: Path) -> str | None:
    """读 parquet **页脚**的索引列统计取末日（不回读数据）。

    🚨 索引列是**最后一列**（pandas 写盘时 `__index_level_0__` 排在全部数据列之后），
    **不是第 0 列** —— 第 0 列是**第一只股票的因子值**（float）。2026-10-16 实测：
    误取 `column(0)` 会让 `str(st.max)[:10]` 变成 `"0.82766371"` 这种因子值字符串，
    与原始面板比大小毫无意义 ⇒ 深检把 **946/950** 个因子全判成"落后"（真值仅 ~340），
    守卫于是**每天**都判定"需要重建"、触发数小时全量重建。
    """
    try:
        import pyarrow.parquet as pq
        from pyarrow import types as pa_types
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        names = schema.names
        idx_col = names.index("__index_level_0__") if "__index_level_0__" in names else None
        if idx_col is None:  # 索引有名字时列名 = 索引名（如 "date"）⇒ 按类型找
            for i, t in enumerate(schema.types):
                if pa_types.is_timestamp(t) or pa_types.is_date(t):
                    idx_col = i
                    break
        if idx_col is not None:
            st = pf.metadata.row_group(0).column(idx_col).statistics
            if st is not None and st.max is not None:
                return str(st.max)[:10]
    except Exception:  # noqa: BLE001
        pass
    try:
        idx = _index_dates(path)
        return str(idx.max().date()) if len(idx) else None
    except Exception:  # noqa: BLE001
        return None


def _in_scope_names() -> set[str]:
    """中性化 builder 的处理范围 = `registry.csv` 里 coverage ≥ `MIN_COVERAGE` 的因子（去重）。

    `_stale_panels` **必须**用同一口径，否则会把 builder **按设计跳过**的低覆盖因子
    永久判成"落后" ⇒ 守卫每天触发数小时全量重建。2026-10-16 实测：`panels_neu` 全量
    重建后仍有 18 个"落后"，全部是 coverage < 0.3 的（`alpha101_096` cov≈0.0001、
    `alpha191_138` cov=0.008、`alt_news_mean_len_20d` cov=0.216 …）—— builder 的
    `MIN_COVERAGE = 0.3` 明确写着"低于此覆盖率的因子不进入模型候选，跳过中性化省算力"。
    """
    try:
        reg = pd.read_csv(_ds_root() / "registry.csv")
        cov = reg.drop_duplicates(subset="name", keep="last").set_index("name")["coverage"]
        return {n for n in dict.fromkeys(reg["name"].astype(str))
                if float(cov.get(n, 0)) >= MIN_COVERAGE}
    except Exception as exc:  # noqa: BLE001
        log.warning("读取 registry 失败（%s）⇒ 深检退回全量对比（可能误报低覆盖因子）", exc)
        return set()


def _stale_panels(subdir: str, *, limit: int = 6) -> tuple[int, int, list[str]]:
    """全量对比 `panels/` 与 `subdir/`，返回 (落后因子数, 总数, 样例)。

    判据 = 中性化面板的索引末日 < 同因子原始面板的索引末日，**且该因子在 builder 的
    处理范围内**（见 `_in_scope_names` —— 否则 18 个低覆盖因子会让守卫天天误判）。

    🔑 为什么必须全量、且要跟**同因子的原始面板**比：
      ① 抽样+max 会静默漏判（见 `_panel_coverage` docstring）；
      ② 不能简单跟"上月最后交易日"比 —— 库里有天然收得早的因子（数据墙），
         会让守判**每天**都判定需要重建。跟自己的原始面板比则自动免疫：
         全量重建后 neu 末日 == min(raw 末日, _base 末日)，数据墙因子两边一致。
    成本：只读页脚统计，~950 文件 / 约 3.5 分钟，对 06:00 的日频守卫可接受
    （`--status` 走快速抽样，不带这道深检）。
    """
    root = _ds_root()
    raw_dir, neu_dir = root / "panels", root / subdir
    if not neu_dir.exists():
        return 0, 0, [f"{subdir} 目录缺失"]
    scope = _in_scope_names()
    files = sorted(neu_dir.glob("*.parquet"))
    stale: list[str] = []
    checked = 0
    for f in files:
        if scope and f.stem not in scope:   # builder 按设计跳过的低覆盖因子，不算落后
            continue
        raw_p = raw_dir / f.name
        if not raw_p.exists():
            continue
        checked += 1
        neu_d, raw_d = _footer_max(f), _footer_max(raw_p)
        if neu_d is not None and raw_d is not None and neu_d < raw_d:
            stale.append(f"{f.stem}(neu={neu_d} raw={raw_d})")
    return len(stale), checked, stale[:limit]


def coverage_table() -> dict[str, str]:
    """四层覆盖末日快照。"""
    return {
        "base(close_adj)": _parquet_last_date(BASE_DIR / "close_adj.parquet"),
        "base(market_cap)": _parquet_last_date(BASE_DIR / "market_cap.parquet"),
        "panels(原始)": _panel_coverage("panels", _PROBE_RAW),
        "panels_neu": _panel_coverage("panels_neu", _PROBE_NEU),
        "panels_neu_fundind": _panel_coverage("panels_neu_fundind", _PROBE_NEU),
    }


def _prev_month_end_trading_day() -> str | None:
    """上一个自然月的最后一个交易日（取 `_base/close_adj` 的交易日历）。

    `_base` 本身陈旧时它同时是"重建链的第 ② 步也过期"的信号 —— 此时
    ``last`` 会早于真实上月末，守卫仍判定需要重建（正确行为，不掩盖问题）。
    """
    p = BASE_DIR / "close_adj.parquet"
    if not p.exists():
        return None
    try:
        cal = _index_dates(p)
    except Exception as exc:  # noqa: BLE001
        log.warning("读取交易日历失败：%s", exc)
        return None
    if len(cal) == 0:
        return None
    today = pd.Timestamp(datetime.now().date())
    this_month = today.to_period("M")
    before = cal[cal.to_period("M") < this_month]
    return str(before.max().date()) if len(before) else None


# ---------------------------------------------------------------------------
# 执行序列
# ---------------------------------------------------------------------------
def _steps(workers: int) -> list[tuple[str, list[str]]]:
    """重建序列（顺序敏感，勿重排）。`sys.executable` = 调用方解释器。

    ⚠️ `build_alla_alpha_panels` **不吃 ``--workers``**：其多进程实现里**每个 worker
    各自 ``load_alla_panels(warmup)`` 重新加载一遍全量日线**（2861 日 × 5807 股 ≈ 1150
    万行）⇒ 内存 ∝ workers。本机（31GB）`--workers 6` 实测 `_ArrayMemoryError`——连
    11 MiB 都分不到、6 个 worker 全崩（2026-10-16 首次计划任务触发即踩到）。这正是
    `MEMORY-engineering.md` §B4 已记的坑「多进程 worker 读原始日线会 MemoryError」。
    ⇒ 此处**固定单进程**（`--workers 1`，走 builder 的单进程分支、只加载一份数据）。
    """
    py = sys.executable
    b = "scripts.builders."
    return [
        # ② 基础面板 —— 必须早于 ④⑤（panels_neu 与 _base 版本耦合）
        ("base_prep", [py, "-m", "scripts.pipelines.rolling_grid_alla", "--stage", "prep"]),
        # ③ 原始面板（全历史重算；不加 --resume，避免跳过旧文件导致不刷新）
        # ⚠️ 固定 --workers 1：每个 worker 各自加载全量日线 ⇒ 6 实测 OOM（见 docstring）
        ("panels_alpha", [py, "-m", b + "build_alla_alpha_panels", "--workers", "1"]),
        ("panels_fundamental", [py, "-m", b + "build_alla_fundamental_factors"]),
        ("panels_holder", [py, "-m", b + "build_alla_holder_factors"]),
        ("panels_holder_dyn", [py, "-m", b + "build_alla_holder_dyn_factors"]),
        ("panels_pledge", [py, "-m", b + "build_alla_pledge_factors"]),
        ("panels_constructed", [py, "-m", b + "build_alla_constructed_factors"]),
        ("panels_p5", [py, "-m", b + "build_alla_p5_factors"]),
        ("panels_style", [py, "-m", b + "build_alla_style_factors"]),
        ("panels_event", [py, "-m", b + "build_alla_event_factors"]),
        ("panels_sue_pledge", [py, "-m", b + "build_alla_sue_pledge_factors"]),
        ("panels_margin", [py, "-m", b + "build_alla_margin_factors"]),
        ("panels_moneyflow", [py, "-m", b + "build_alla_moneyflow_factors"]),
        ("panels_disc_holder_dyn", [py, "-m", b + "build_alla_disc_holder_dyn"]),
        ("panels_status", [py, "-m", b + "build_alla_status_factors"]),
        ("panels_altfactors", [py, "-m", b + "build_alla_altfactors",
                              "--families", "insider,macro,news"]),
        # ④⑤ 中性化（读 ②③ 的产物）
        # ⚠️ 必须 --force：两个 builder 的跳过判据是「输出文件已存在」，**不看数据是否
        # 已过期**。②③ 全历史重算后旧产物仍在盘上 ⇒ 不加 --force 会 950 个全跳过、
        # panels_neu 覆盖日原地不动（2026-10-16 续跑实测：EXIT=0 但面板毫无变化）。
        ("neu_std", [py, "-m", b + "build_alla_factor_neutralized",
                     "--workers", str(workers), "--force"]),
        ("neu_fundind", [py, "-m", b + "build_alla_factor_neutralized_fundind",
                         "--workers", str(workers), "--force"]),
        # ⑥ alt 族 family-aware 收尾 —— **必须排在 neu_std/neu_fundind 之后**：
        # 标准三步对 alt 族会产出坏面板（零膨胀 ⇒ MAD=0 ⇒ 整截面常数 ⇒ zscore 全 NaN；
        # 市场级广播 ⇒ 截面中性化恒 0 ⇒ 浮点残差被放大成假面板），故由本 builder
        # 按面板形态分派覆盖写回。顺序颠倒 = 把 2026-10-10 修好的 alt 中性化打回原形。
        ("neu_altf", [py, "-m", b + "build_alla_altfactor_neu"]),
    ]


def _run_steps(steps: list[tuple[str, list[str]]], *, dry_run: bool,
               log_path: Path | None = None, start: int = 1) -> int:
    import subprocess

    total = start + len(steps) - 1
    for i, (name, cmd) in enumerate(steps, start):
        line = subprocess.list2cmdline(cmd)
        if dry_run:
            log.info("[dry-run %2d/%d] %s :: %s", i, total, name, line)
            continue
        log.info("[%2d/%d] %s 开始 :: %s", i, total, name, line)
        t0 = time.time()
        # 子进程 stdout/stderr 并入同一日志文件 —— 否则计划任务无 stdout 重定向，
        # 子进程的失败（如 builder 的 traceback）会零痕迹（2026-10-16 踩到的正是这个）。
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as fh:
                proc = subprocess.run(cmd, cwd=str(ROOT), check=False,
                                      stdout=fh, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        dt = time.time() - t0
        if proc.returncode != 0:
            log.error("[%2d/%d] %s 失败（exit=%d，耗时 %.0fs）—— 中止后续步骤，"
                      "已完成的步骤产物保留（下次运行会重跑全链）",
                      i, total, name, proc.returncode, dt)
            return proc.returncode
        log.info("[%2d/%d] %s 完成（%.0fs）", i, total, name, dt)
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="因子库面板月度重建（守卫式）")
    ap.add_argument("--status", action="store_true", help="只打印四层覆盖末日")
    ap.add_argument("--dry-run", action="store_true", help="打印命令序列但不执行")
    ap.add_argument("--run", action="store_true", help="真正执行重建（数小时）")
    ap.add_argument("--force", action="store_true", help="忽略守卫，强制重建")
    ap.add_argument("--ignore-busy-window", action="store_true",
                    help="忽略出榜链时段保护（16:00–20:00），强制在窗口内执行")
    ap.add_argument("--workers", type=int, default=6, help="并行进程数（传给 builder）")
    ap.add_argument("--from-step", type=int, default=1,
                    help="从第 N 步开始（跳过前 N-1 步）——链路较长，中途失败后可续跑，"
                         "不必重算已成功的步骤")
    ap.add_argument("--log-file", default=None,
                    help="日志落盘模板（计划任务用，{date} → YYYYMMDD）")
    args = ap.parse_args()

    log_path: Path | None = None
    if args.log_file:
        # 🚨 不能复用 alla_daily_rank.attach_file_log —— 它把 file handler **硬编码挂到
        # "alla_daily_rank" logger**，而本脚本的 log 是 "rebuild_panel_library"
        # ⇒ 句柄挂空、日志全丢（2026-10-16 首次计划任务触发：exit=1 而日志仅 1 行）。
        log_path = Path(str(args.log_file).format(date=datetime.now().strftime("%Y%m%d")))
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        setup_logging("rebuild_panel_library", file=log_path)
        log.info("日志落盘 → %s", log_path)

    cov = coverage_table()
    target = _prev_month_end_trading_day()
    log.info("面板覆盖体检（%s）", datetime.now().strftime("%Y-%m-%d %H:%M"))
    for k, v in cov.items():
        log.info("  %-20s %s", k, v)
    log.info("  目标（上一个自然月最后交易日）= %s", target)
    if args.status:
        return 0

    # ---- 守卫 ----
    neu_max = cov["panels_neu"]
    need = False
    if target is None:
        log.warning("拿不到目标交易日 ⇒ 无法守判，视为需要重建")
        need = True
    elif neu_max.startswith("ERR") or neu_max in ("缺失", "全部读取失败"):
        log.warning("panels_neu 覆盖末日不可读（%s）⇒ 视为需要重建", neu_max)
        need = True
    elif neu_max < target:
        log.info("守判：panels_neu 止 %s < 目标 %s ⇒ 需要重建", neu_max, target)
        need = True
    else:
        log.info("守判：panels_neu 已覆盖 %s ≥ 目标 %s（抽样 min）", neu_max, target)
    # ---- 深检（全量，逐因子 vs 自己的原始面板）----
    # 抽样只管 3 个因子；这里是"整库有没有落后"的真判据。见 `_stale_panels`。
    # 两套 neutral 都查：fundind 不在出榜链上（快信号是实时重算的），
    # 没有任何下游会因它陈旧而报错 ⇒ 只能靠这里兜住，否则会静默腐烂。
    for sub in ("panels_neu", "panels_neu_fundind"):
        n_stale, n_total, examples = _stale_panels(sub)
        if n_stale:
            log.warning("深检：%s 有 %d/%d 个因子落后于原始面板（例：%s）⇒ 需要重建",
                        sub, n_stale, n_total, "; ".join(examples))
            need = True
        else:
            log.info("深检：%s 全部 %d 个因子均不落后于原始面板", sub, n_total)
    if args.force:
        log.warning("--force：忽略守判，强制重建")
        need = True

    if not args.run and not args.dry_run:
        if not need:
            return 0
        log.info("提示：当前为只读模式。加 --dry-run 看命令序列，加 --run 真正执行。")
        return 0

    if not need and not args.dry_run:
        return 0

    # ---- 时段保护：避开出榜链窗口（见 _BUSY_WINDOW 说明）----
    now_h = datetime.now().hour
    if (args.run and not args.ignore_busy_window
            and _BUSY_WINDOW[0] <= now_h < _BUSY_WINDOW[1]):
        log.info("当前 %02d:xx 落在出榜链窗口 %02d:00–%02d:00 内 ⇒ 本次跳过，"
                 "明日再触发（重建是月频动作，推迟一天无成本）",
                 now_h, _BUSY_WINDOW[0], _BUSY_WINDOW[1])
        return 0

    # ---- 数据前置检查：原始数据是否已推进到目标日 ----
    from config import Config
    daily = Path(str(Config.get()["cache"]["root"])) / "daily_all_a.parquet"
    daily_max = _parquet_last_date(daily)
    log.info("原始数据 daily_all_a 覆盖 = %s（%s）", daily_max, daily)
    if not args.dry_run and target and daily_max < target:
        log.error("原始数据止 %s < 目标 %s ⇒ 先跑 update_data 再重建，本次中止",
                  daily_max, target)
        return 2

    steps = _steps(args.workers)
    if args.from_step > 1:
        if not 1 <= args.from_step <= len(steps):
            log.error("--from-step %d 越界（链路共 %d 步）", args.from_step, len(steps))
            return 2
        log.info("续跑模式：跳过前 %d 步，从第 %d 步「%s」开始（前置步骤产物须已在盘上）",
                 args.from_step - 1, args.from_step, steps[args.from_step - 1][0])
        steps = steps[args.from_step - 1:]
    t0 = time.time()
    with keep_system_awake() as _awake:
        log.info("防待机：keep_system_awake 生效=%s", _awake)
        rc = _run_steps(steps, dry_run=args.dry_run, log_path=log_path,
                        start=args.from_step)
    log.info("重建%s结束（%.0fs）", "演练" if args.dry_run else "执行", time.time() - t0)
    if rc == 0 and not args.dry_run:
        log.info("完成后的覆盖体检：")
        for k, v in coverage_table().items():
            log.info("  %-20s %s", k, v)
    return rc


def _main_guarded() -> int:
    """入口包装：把未捕获异常的 traceback 写进日志文件。

    🚨 为什么必须：计划任务（`YuriQuant RebuildPanels`）**没有重定向 stdout/stderr**
    （schedule.yaml 靠 `--log-file` 留痕）。若 `main()` 直接抛，traceback 只进 stderr
    ⇒ 日志停在 setup 那一行、任务返回 rc=1 —— 2026-10-16 06:00 首跑的真实症状
    （日志 85 字节 = 仅"日志落盘"一行，重启后无从查因）。包一层后任何崩溃都留痕。
    """
    try:
        return main()
    except Exception:  # noqa: BLE001 —— 顶层兜底，任何异常都要落日志
        log.exception("重建入口异常退出（rc=1）；完整 traceback 见下")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main_guarded())
