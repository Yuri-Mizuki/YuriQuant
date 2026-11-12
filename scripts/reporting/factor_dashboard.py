"""聚宽式因子看板：筛选 + 排序表 + 点行看详情（自包含 HTML）。

对齐 joinquant.com/view/factorlib/list 的信息架构：
- 顶部筛选：来源(set) chips / 因子族 / 搜索框
- 主表：因子名 / 来源 / 族 / 覆盖率 / IC(h1,h5,h10,h20) / ICIR / NW-t / IC胜率
  / （有分层回测的库额外：多空年化·月/周切换 / 多头年化 / 超额年化 / 换手率） / 公式
- 点行 → 右侧详情：指标卡 + 月度 IC 柱状图 + （有 evals 的因子）LS/LO 净值曲线
  （含全A等权基准虚线，如数据集有 bench_returns.parquet）

数据来源（全部预计算，前端零后端）：
- registry.csv：元数据 + ic_mean_h*（批量入库行只有多周期 IC 均值，无分层回测指标）
- ic_h{1,5,10,20}.parquet：逐日 IC 矩阵 → 现场算 ICIR / NW-t / IC 胜率 / 月度 IC
  （Newey-West 滞后阶 L = int(4*(T/100)^(2/9))，全库同 T 同 L，向量化一次算完）
- evals/<slug>.parquet：LS/LO 净值（仅部分因子有），降采样到周频内嵌

口径提示（页面 warning 原样展示）：
- 全 A 库批量入库行为样本内统计、无挖-验分离；分层回测指标仅限 registry 里
  有值的行（hs300 旧库 / 部分带 evals 的因子），缺失显示 "—"，不冒充。

用法:
    python scripts/reporting/factor_dashboard.py                              # 默认 all_a_2018_2026
    python scripts/reporting/factor_dashboard.py --dataset hs300_2022_2025
    python scripts/reporting/factor_dashboard.py --out reports/factor_dashboard/board.html
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
    sys.path.insert(0, ROOT)

from config import Config  # noqa: E402
from research.html_report import page  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("factor_dashboard")

#: 来源两级分组：大类（chips 第一层）→ registry 的 set（细分，第二层 chips）。
#: 映射不到的 set 一律落 "其他"——新增 set 值不用改这里也能正确显示，
#: 但建议归入大类后再看（改这张表 = 改展示口径，不影响任何数字）。
SRC_GROUPS: dict[str, set[str]] = {
    "公开量价库": {"alpha101", "alpha158", "alpha191", "alpha360"},
    "基本面": {"fundamental"},
    "另类数据": {"altf", "evt", "holder", "holder_dyn",
                "moneyflow", "margin", "status"},
}

#: 回测区间（对齐聚宽「近三月/近一年/…」）。锚点 = 数据最后一天，向前推自然月。
#: 区间只作用于**展示**：IC 类指标按区间切片现算，回测类从 evals 日收益重算；
#: 不改任何入库/入库判据。
WINDOWS: list[tuple[str, int | None]] = [
    ("3M", 3), ("1Y", 12), ("3Y", 36), ("10Y", 120), ("ALL", None),
]
WIN_LABEL = {"3M": "近三月", "1Y": "近一年", "3Y": "近三年",
             "10Y": "近十年", "ALL": "全部"}


def slice_window(obj, months: int | None):
    """按自然月截取尾部（months=None → 原样返回）。"""
    if months is None or len(obj) == 0:
        return obj
    end = obj.index.max()
    return obj.loc[obj.index > end - pd.DateOffset(months=months)]


def src_group(set_name: str) -> str:
    for g, sets in SRC_GROUPS.items():
        if set_name in sets:
            return g
    return "其他"


# ---------------------------------------------------------------- 数据加载

def load_ic_matrices(lib_root: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    """读 ic_h{1,5,10,20}.parquet，列对齐到 registry 名（缺列补 NaN）。"""
    out = {}
    for h in (1, 5, 10, 20):
        p = lib_root / f"ic_h{h}.parquet"
        if not p.exists():
            log.warning("缺 %s，跳过该 horizon", p.name)
            continue
        df = pd.read_parquet(p)
        missing = [n for n in names if n not in df.columns]
        if missing:
            df = df.reindex(columns=[*df.columns, *missing])
        out[f"h{h}"] = df[list(dict.fromkeys(names))]
        log.info("ic_h%d: %d 日 × %d 因子", h, *df.shape)
    return out


def nw_t_stat(x: np.ndarray) -> np.ndarray:
    """向量化 Newey-West t（逐列）。x: T×N，NaN 按列忽略（用有效样本数）。"""
    t_len, n = x.shape
    lag = max(1, int(4 * (t_len / 100) ** (2 / 9)))
    valid = (~np.isnan(x)).sum(axis=0)                      # 每列有效样本数
    xc = np.where(np.isnan(x), 0.0, x)
    mean = xc.sum(axis=0) / np.maximum(valid, 1)
    xd = xc - mean                                          # NaN 位减均值后仍按 0 处理，
    xd[np.isnan(x)] = 0.0                                   # 协方差项用 valid 缩放近似
    g0 = (xd * xd).sum(axis=0) / np.maximum(valid, 1)
    var = g0.copy()
    for k in range(1, lag + 1):
        gk = (xd[k:] * xd[:-k]).sum(axis=0) / np.maximum(valid - k, 1)
        var += 2 * (1 - k / (lag + 1)) * gk
    se = np.sqrt(np.maximum(var, 1e-18) / np.maximum(valid, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / se
    # 有效样本不足（列全空/过短）→ t 记 NaN，不许 0 冒充
    t[valid < lag + 2] = np.nan
    return t


def ic_metrics(ic: pd.DataFrame) -> pd.DataFrame:
    """逐日 IC 矩阵 → ICIR / NW-t / IC胜率 / 月度 IC（向量化）。"""
    arr = ic.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        icir = mean / std
    win = np.nanmean(arr > 0, axis=0)
    t_nw = nw_t_stat(arr)
    mon = ic.groupby(ic.index.to_period("M")).mean()
    # 只保留真正有数据的因子列（reindex 补出来的全 NaN 列不进 payload，
    # 否则前端判真、画空图而非「无逐日 IC 数据」提示）
    monthly = {c: [None if pd.isna(v) else round(float(v), 3) for v in mon[c]]
               for c in mon.columns if mon[c].notna().any()}
    months = [str(p) for p in mon.index]
    return (pd.DataFrame({"ic_mean": mean, "icir": icir, "win": win, "nw_t": t_nw},
                         index=ic.columns),
            months, monthly)


def window_ic_metrics(mats: dict[str, pd.DataFrame]) -> dict[str, dict[str, dict]]:
    """按区间重算 IC 类指标 → ``{win: {name: {ic1, icir, nw, win, ic5, ic10, ic20}}}``。

    口径与全期一致（同 ``nw_t_stat``、同 NaN 跳过规则），只是切片不同；
    区间内样本不足（<2 日）时返回空 dict，前端显示 —。
    """
    out: dict[str, dict[str, dict]] = {}
    for wkey, m in WINDOWS:
        cols: dict[str, np.ndarray] = {}
        idx_names = None
        if "h1" in mats:
            sub = slice_window(mats["h1"], m)
            idx_names = sub.columns
            if len(sub) >= 2:
                arr = sub.to_numpy(dtype=float)
                mean = np.nanmean(arr, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cols["ic1"] = mean
                    cols["icir"] = mean / np.nanstd(arr, axis=0)
                cols["win"] = np.nanmean(arr > 0, axis=0)
                cols["nw"] = nw_t_stat(arr)
        for h in (5, 10, 20):
            key = f"h{h}"
            if key not in mats:
                continue
            sub = slice_window(mats[key], m)
            if idx_names is None:
                idx_names = sub.columns
            if len(sub) >= 1:
                cols[f"ic{h}"] = np.nanmean(sub.to_numpy(dtype=float), axis=0)
        if idx_names is None or not cols:
            out[wkey] = {}
            continue
        df = pd.DataFrame({k: np.asarray(v, dtype=float).round(4) for k, v in cols.items()},
                          index=idx_names)
        # 显式转 python float + NaN→None（np.float64 / NaN 都不是合法 JSON）
        out[wkey] = {
            nm: {k: (None if not np.isfinite(v) else round(float(v), 4))
                 for k, v in row.items()}
            for nm, row in df.to_dict("index").items()
        }
    return out


def _ann(series: pd.Series | None, ppy: int = 252) -> float | None:
    """净值序列 → 年化收益（几何；按交易日计年）。样本不足返回 None。"""
    if series is None:
        return None
    s = series.dropna()
    if len(s) < 2:
        return None
    a, b = float(s.iloc[0]), float(s.iloc[-1])
    if a <= 0:
        return None
    years = (len(s) - 1) / ppy
    if years <= 0:
        return None
    try:
        return round((b / a) ** (1 / years) - 1, 4)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def load_equity(lib_root: Path, reg: pd.DataFrame):
    """有 evals 的因子 → 月频降采样净值（LS/LO）+ 数据集级基准 + 区间回测指标。

    月频而非周频：回填后全库 ~950 个因子都有 evals，周频内嵌 payload 超 6MB，
    月频（~120 点）足够支撑详情页 200px 高的曲线。

    区间回测指标从 evals 的日净值/日换手重算（年化几何、超额 = 纯多 − 全A等权、
    换手 = 区间内调仓日均值）；缺列则该区间留空，不回落到全期值冒充。
    """
    out, bt = {}, {}
    last_dt = None
    # 基准先读（区间超额要用区间基准年化）
    bench, bench_nav = None, None
    bp = lib_root / "bench_returns.parquet"
    if bp.exists():
        b = pd.read_parquet(bp).iloc[:, 0]
        bench_nav = (1 + b.fillna(0)).cumprod()
        mo = bench_nav.resample("ME").last()
        bench = {"d": [d.strftime("%Y-%m-%d") for d in mo.index],
                 "v": [round(float(v), 4) for v in mo]}
    ev_paths = reg.get("eval_path")
    if ev_paths is None:
        return out, bench, bt, last_dt
    for _, r in reg.iterrows():
        ep = r.get("eval_path")
        if not isinstance(ep, str) or not ep.strip():
            continue
        p = lib_root / ep if not Path(ep).is_absolute() else Path(ep)
        if not p.exists():
            p = lib_root / "evals" / Path(ep).name
        if not p.exists():
            continue
        try:
            ev = pd.read_parquet(p)
        except Exception as exc:
            log.warning("evals 读取失败 %s: %s", p.name, exc)
            continue
        mo = ev.resample("ME").last().dropna(how="all")
        if len(ev.index):
            dmax = ev.index.max()
            last_dt = dmax if last_dt is None or dmax > last_dt else last_dt
        out[r["name"]] = {
            "d": [d.strftime("%Y-%m-%d") for d in mo.index],
            "ls": [round(float(v), 4) if pd.notna(v) else None
                   for v in mo.get("equity_ls_M", pd.Series(dtype=float))],
            "lo": [round(float(v), 4) if pd.notna(v) else None
                   for v in mo.get("equity_lo_M", pd.Series(dtype=float))],
        }
        # 区间回测指标（日净值/日换手重算；月、周调仓各一套）
        rec: dict[str, dict] = {}
        for wk, m in WINDOWS:
            mrec: dict[str, dict] = {}
            for freq in ("M", "W"):
                e_ls = ev.get(f"equity_ls_{freq}")
                e_lo = ev.get("equity_lo_M") if freq == "M" else None
                t_s = ev.get(f"turnover_ls_{freq}")
                d: dict = {}
                if e_ls is not None:
                    d["ls"] = _ann(slice_window(e_ls.dropna(), m))
                    if e_lo is not None:
                        d["lo"] = _ann(slice_window(e_lo.dropna(), m))
                        ba = _ann(slice_window(bench_nav, m)) if bench_nav is not None else None
                        if d["lo"] is not None and ba is not None:
                            d["ex"] = round(d["lo"] - ba, 4)
                if t_s is not None:
                    ts = slice_window(t_s.dropna(), m)
                    if len(ts):
                        d["to"] = round(float(ts.mean()), 4)
                if d:
                    mrec[freq] = d
            if mrec:
                rec[wk] = mrec
        if rec:
            bt[r["name"]] = rec
    log.info("内嵌净值曲线 %d 个因子 | 区间回测指标 %d | 基准: %s",
             len(out), len(bt), "有" if bench else "无")
    return out, bench, bt, last_dt


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="聚宽式因子看板")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--out", default=None, help="输出 HTML 路径")
    args = ap.parse_args()

    lib_root = Path(Config.paths()["factor_library_root"]) / args.dataset
    reg = pd.read_csv(lib_root / "registry.csv")
    reg = reg.drop_duplicates(subset="name", keep="last").reset_index(drop=True)
    names = reg["name"].tolist()
    log.info("因子库 %s: %d 个因子", args.dataset, len(reg))

    mats = load_ic_matrices(lib_root, names)
    months, monthly = ([], {})
    # registry.csv 本身就有（批量入库行全空的）ic_mean 列——merge 前必须改名，
    # 否则撞成 ic_mean_x/y，表格里拿到的是空列。
    mcols = {"ic_mean": "calc_ic1", "icir": "calc_icir",
             "win": "calc_win", "nw_t": "calc_nw"}
    if "h1" in mats:
        metrics, months, monthly = ic_metrics(mats["h1"])
        reg = reg.merge(
            metrics.reset_index().rename(columns={"index": "name", **mcols}),
            on="name", how="left")
        log.info("IC 矩阵覆盖 %d/%d 个因子", metrics["ic_mean"].notna().sum(), len(reg))
    else:
        log.warning("无 ic_h1 矩阵，ICIR/NW-t/胜率不可算")
        for c in mcols.values():
            reg[c] = np.nan

    has_ls = reg["annual_return_ls_M"].notna().any() if "annual_return_ls_M" in reg else False
    has_w = reg["annual_return_ls_W"].notna().any() if "annual_return_ls_W" in reg else False
    has_ex = ("excess_return_lo_M" in reg.columns
              and reg["excess_return_lo_M"].notna().any())
    wm = window_ic_metrics(mats)
    equity, bench, bt_win, data_last = load_equity(lib_root, reg)
    # 区间锚点：有 IC 矩阵用矩阵末日（与 IC 切片一致），否则用 evals 最新日期
    ic_last = mats["h1"].index.max() if "h1" in mats else data_last
    win_start = {wk: (None if m is None or ic_last is None else
                      str((ic_last - pd.DateOffset(months=m)).date()))
                 for wk, m in WINDOWS}

    # ---- 行数据（与前端列一一对应）----
    def f(v, nd=4):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)

    rows = []
    for i, r in reg.iterrows():
        setv = str(r.get("set") or "—")
        # 区间指标：IC 类逐区间现算；回测类优先 evals 重算，全期区间回落到 registry
        # 登记值（registry 的 ALL 值是入库时的权威口径，优先于日序列近似重算）。
        wrow: dict[str, dict] = {}
        name = str(r["name"])
        for wk, _ in WINDOWS:
            d = dict(wm.get(wk, {}).get(name) or {})
            b = bt_win.get(name, {}).get(wk) or {}
            if b:
                d["bt"] = b
            if wk == "ALL" and not wm.get("ALL"):
                # 无 ic_h* 矩阵的库（如 hs300）→ 全期区间回落到 registry 登记值
                for src, dst, nd in (("ic_mean", "ic1", 4), ("ic_ir", "icir", 4),
                                     ("t_stat_nw", "nw", 2), ("ic_win_rate", "win", 3),
                                     ("ic_mean_h5", "ic5", 4), ("ic_mean_h10", "ic10", 4),
                                     ("ic_mean_h20", "ic20", 4)):
                    if dst not in d:
                        v = f(r.get(src), nd)
                        if v is not None:
                            d[dst] = v
            if wk == "ALL" and has_ls:
                mrec = d.setdefault("bt", {}).setdefault("M", {})
                if mrec.get("ls") is None:
                    mrec["ls"] = f(r.get("annual_return_ls_M"))
                if mrec.get("lo") is None:
                    mrec["lo"] = f(r.get("annual_return_lo_M"))
                if mrec.get("to") is None:
                    mrec["to"] = f(r.get("avg_turnover_ls_M"))
                if has_ex and mrec.get("ex") is None:
                    mrec["ex"] = f(r.get("excess_return_lo_M"))
                if has_w:
                    wrec = d["bt"].setdefault("W", {})
                    if wrec.get("ls") is None:
                        wrec["ls"] = f(r.get("annual_return_ls_W"))
                    if wrec.get("to") is None:
                        wrec["to"] = f(r.get("avg_turnover_ls_W"))
            wrow[wk] = d

        rows.append({
            "i": i, "name": name,
            "set": setv, "grp": src_group(setv),
            "fam": str(r.get("family") or "—"),
            "cov": f(r.get("coverage"), 3),
            "w": wrow,
            "sig": bool(r["significant"]) if "significant" in reg and pd.notna(r.get("significant")) else None,
            "has_ic": bool(name in monthly),
            "has_eq": name in equity,
        })

    payload = {
        "rows": rows, "months": months,
        "groups": {g: sorted({r["set"] for r in rows if r["grp"] == g})
                   for g in dict.fromkeys(r["grp"] for r in rows)},
        "monthly": {k: v for k, v in monthly.items()},
        "equity": equity, "bench": bench,
        "formula": {str(r["name"]): str(r.get("formula") or "")[:300]
                    for _, r in reg.iterrows()},
        "meta": {str(r["name"]): {
            "freq": str(r.get("frequency") or "—"), "mat": str(r.get("maturity") or "—"),
            "sub": str(r.get("subfamily") or ""),
            "af": "" if pd.isna(r.get("active_first")) else str(r.get("active_first"))[:10],
            "al": "" if pd.isna(r.get("active_last")) else str(r.get("active_last"))[:10],
        } for _, r in reg.iterrows()},
        "hasLS": bool(has_ls), "hasW": bool(has_w), "hasEx": bool(has_ex),
        "wins": [k for k, _ in WINDOWS],
        "winLabel": WIN_LABEL,
        "winStart": win_start,
    }

    # ---- HTML ----
    out = (Path(args.out) if args.out
           else Path("reports/factor_dashboard") / f"board_{args.dataset}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    html = render(args.dataset, len(reg), reg, payload, months)
    out.write_text(html, encoding="utf-8")
    log.info("看板已生成: %s (%.1f MB)", out, out.stat().st_size / 1e6)

    csv_out = out.with_suffix(".csv")
    cols = ["name", "set", "family", "coverage", "ic_mean_h1", "ic_mean_h5",
            "ic_mean_h10", "ic_mean_h20"]
    extra = [c for c in ("calc_icir", "calc_nw", "calc_win") if c in reg.columns]
    if has_ls:
        extra += ["annual_return_ls_M", "sharpe_ls_M", "annual_return_lo_M",
                  "avg_turnover_ls_M"] + (
                  ["annual_return_ls_W", "avg_turnover_ls_W"] if has_w else []) + (
                  ["excess_return_lo_M", "information_ratio_lo_M"] if has_ex else [])
    reg[[c for c in cols + extra if c in reg.columns]].to_csv(
        csv_out, index=False, encoding="utf-8-sig")
    log.info("全表 CSV: %s", csv_out)


# ---------------------------------------------------------------- 渲染

_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
       background:#f5f7fa; color:#1a1a2e; font-size:13px; }
.topbar { background:linear-gradient(135deg,#1a1a2e,#16213e); color:#fff;
          padding:16px 24px; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
.topbar h1 { font-size:18px; }
.topbar .meta { font-size:12px; opacity:.8; }
.wrap { display:flex; gap:14px; padding:14px 20px; align-items:flex-start; }
.main { flex:1; min-width:0; }
.kpis { display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap; }
.kpi { background:#fff; border-radius:10px; padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,.06); }
.kpi .v { font-size:20px; font-weight:700; color:#16213e; }
.kpi .k { font-size:11px; color:#888; }
.filters { background:#fff; border-radius:10px; padding:12px 14px; margin-bottom:12px;
           box-shadow:0 1px 4px rgba(0,0,0,.06); display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.chip { padding:4px 12px; border-radius:14px; border:1px solid #d5d9e0; cursor:pointer;
        background:#f8f9fa; font-size:12px; user-select:none; }
.chip.on { background:#1a3a6b; color:#fff; border-color:#1a3a6b; }
.chiprow { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.subchip { font-size:11px; padding:3px 10px; color:#555; }
.subchip.on { color:#fff; }
select,.search { padding:6px 10px; border:1px solid #ddd; border-radius:8px; font-size:12px; }
.search { width:220px; }
.note { background:#fff3cd; border:1px solid #ffe082; border-radius:8px; padding:10px 14px;
        font-size:12px; color:#856404; margin-bottom:12px; }
.scroll { max-height:72vh; overflow-y:auto; background:#fff; border-radius:10px;
          box-shadow:0 1px 4px rgba(0,0,0,.06); }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { padding:8px 8px; background:#f0f2f5; border-bottom:2px solid #ddd; cursor:pointer;
     user-select:none; white-space:nowrap; position:sticky; top:0; z-index:2; }
th.num, td.num { text-align:right; }
td { padding:5px 8px; border-bottom:1px solid #f0f0f0; white-space:nowrap; }
tbody tr:hover { background:#f0f4ff; cursor:pointer; }
tbody tr.sel { background:#e8eefc; }
.pos { color:#A32D2D; font-weight:600; }
.neg { color:#3B6D11; font-weight:600; }
.fml { font-family:monospace; font-size:11px; color:#666; max-width:260px;
       overflow:hidden; text-overflow:ellipsis; }
/* 详情抽屉 */
.detail { width:520px; flex-shrink:0; position:sticky; top:14px; background:#fff;
          border-radius:10px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,.06);
          display:none; max-height:88vh; overflow-y:auto; }
.detail.show { display:block; }
.detail h2 { font-size:15px; color:#16213e; margin-bottom:4px; }
.detail .src { font-size:11px; color:#888; margin-bottom:10px; }
.dgrid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; margin-bottom:10px; }
.dcell { background:#f8f9fa; border-radius:8px; padding:8px; text-align:center; }
.dcell .v { font-size:15px; font-weight:700; }
.dcell .k { font-size:10px; color:#888; }
.formula { font-family:monospace; font-size:11px; color:#555; background:#f8f9fa;
           padding:8px 10px; border-radius:6px; margin-bottom:10px; word-break:break-all; }
.dchart { margin-bottom:10px; }
.badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px;
         background:#eef1f5; color:#555; margin-right:6px; }
"""

_JS = r"""
const D = __PAYLOAD__;
const $ = id => document.getElementById(id);
const fmt = (v, nd) => v === null || v === undefined ? '—' : (+v).toFixed(nd === undefined ? 4 : nd);
const cls = v => v === null || v === undefined ? '' : (v > 0 ? 'pos' : 'neg');
const rowByIdx = {}; D.rows.forEach(r => rowByIdx[r.i] = r);

// ---- 筛选状态（来源两级：大类 → 细分 set）----
let activeGrp = '', famFilter = '', q = '';
const activeSubs = new Set();          // 空 = 该大类下全部细分
document.querySelectorAll('#grpChips .chip').forEach(ch => {
  ch.onclick = () => {
    if (ch.dataset.g === '__ALL__') {
      activeGrp = ''; activeSubs.clear();
      document.querySelectorAll('#grpChips .chip')
        .forEach(c => c.classList.toggle('on', c === ch));
    } else {
      activeGrp = ch.classList.contains('on') ? '' : ch.dataset.g;
      activeSubs.clear();
      document.querySelectorAll('#grpChips .chip')
        .forEach(c => c.classList.toggle('on', c === ch && activeGrp !== ''));
    }
    renderSubChips(); renderRows();
  };
});
function renderSubChips() {
  const box = $('subChips');
  box.innerHTML = '';
  if (!activeGrp) { box.style.display = 'none'; return; }
  box.style.display = '';
  (D.groups[activeGrp] || []).forEach(s => {
    const c = document.createElement('span');
    c.className = 'chip subchip'; c.textContent = s;
    c.onclick = () => {
      c.classList.toggle('on');
      c.classList.contains('on') ? activeSubs.add(s) : activeSubs.delete(s);
      renderRows();
    };
    box.appendChild(c);
  });
}
$('famSel').onchange = e => { famFilter = e.target.value; renderRows(); };
$('fsearch').oninput = e => { q = e.target.value.toLowerCase(); renderRows(); };

// ---- 回测区间（近三月/近一年/近三年/近十年/全部）----
let win = 'ALL', curRow = null;
document.querySelectorAll('#winChips .chip').forEach(ch => {
  ch.onclick = () => {
    win = ch.dataset.w;
    document.querySelectorAll('#winChips .chip').forEach(c => c.classList.toggle('on', c === ch));
    renderRows();
    if (curRow) showDetail(curRow);
  };
});

// ---- 调仓频率切换（月/周，仅 hasLS && hasW 时显示）----
let per = 'M';
const perSel = $('perSel');
if (perSel) {
  perSel.style.display = D.hasLS && D.hasW ? '' : 'none';
  perSel.querySelectorAll('.chip').forEach(ch => {
    ch.onclick = () => {
      per = ch.dataset.p;
      perSel.querySelectorAll('.chip').forEach(c => c.classList.toggle('on', c === ch));
      renderRows();
      if (curRow) showDetail(curRow);
    };
  });
}

// 取值器：IC 类 = 当前区间；回测类 = 当前区间 × 当前调仓频率
const curOf = (r, k) => { const d = r.w[win]; return d && d[k] !== undefined ? d[k] : null; };
const btOf = (r, k) => {
  const d = r.w[win]; const b = d && d.bt && d.bt[per];
  return b && b[k] !== undefined ? b[k] : null;
};
const BT_KEYS = new Set(['ls', 'lo', 'ex', 'to']);
const ROOT_KEYS = new Set(['name', 'set', 'fam', 'cov']);
const valOf = (r, k) => ROOT_KEYS.has(k) ? r[k] : (BT_KEYS.has(k) ? btOf(r, k) : curOf(r, k));
const cmp = (a, b, k) => {
  const av = valOf(a, k), bv = valOf(b, k);
  if (av === null && bv === null) return 0;
  if (av === null) return 1; if (bv === null) return -1;
  if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
  return sortAsc ? av - bv : bv - av;
};

// ---- 排序（初始 NW-t 降序，缺值垫底）----
let sortKey = 'nw', sortAsc = false;
D.rows.sort((a, b) => cmp(a, b, 'nw'));
function renderRows() {
  const tb = $('ftable').tBodies[0];
  tb.innerHTML = '';
  let shown = 0;
  for (const r of D.rows) {
    if (activeGrp && r.grp !== activeGrp) continue;
    if (activeSubs.size && !activeSubs.has(r.set)) continue;
    if (famFilter && r.fam !== famFilter) continue;
    if (q && !(r.name.toLowerCase().includes(q) || (D.formula[r.name]||'').toLowerCase().includes(q))) continue;
    shown++;
    const tr = document.createElement('tr');
    tr.dataset.i = r.i;
    tr.innerHTML =
      `<td><b>${r.name}</b></td><td>${r.set}</td><td>${r.fam}</td>` +
      `<td class="num">${fmt(r.cov, 2)}</td>` +
      `<td class="num ${cls(curOf(r,'ic1'))}">${fmt(curOf(r,'ic1'))}</td>` +
      `<td class="num ${cls(curOf(r,'icir'))}">${fmt(curOf(r,'icir'), 2)}</td>` +
      `<td class="num ${cls(curOf(r,'nw'))}">${fmt(curOf(r,'nw'), 2)}</td>` +
      `<td class="num">${fmt(curOf(r,'win'), 2)}</td>` +
      `<td class="num ${cls(curOf(r,'ic5'))}">${fmt(curOf(r,'ic5'))}</td>` +
      `<td class="num ${cls(curOf(r,'ic10'))}">${fmt(curOf(r,'ic10'))}</td>` +
      `<td class="num ${cls(curOf(r,'ic20'))}">${fmt(curOf(r,'ic20'))}</td>` +
      (D.hasLS
        ? `<td class="num ${cls(btOf(r,'ls'))}">${fmt(btOf(r,'ls'))}</td>` +
          `<td class="num ${cls(btOf(r,'lo'))}">${fmt(btOf(r,'lo'))}</td>` +
          (D.hasEx ? `<td class="num ${cls(btOf(r,'ex'))}">${fmt(btOf(r,'ex'))}</td>` : '') +
          `<td class="num">${fmt(btOf(r,'to'), 2)}</td>`
        : '') +
      `<td class="fml" title="${(D.formula[r.name]||'').replace(/"/g,'&quot;')}">${(D.formula[r.name]||'').slice(0,50)}</td>`;
    tr.onclick = () => showDetail(r, tr);
    tb.appendChild(tr);
  }
  $('cnt').textContent = shown;
  document.querySelectorAll('#ftable th[data-k]').forEach(th => {
    th.onclick = () => {
      const k = th.dataset.k;
      if (sortKey === k) sortAsc = !sortAsc; else { sortKey = k; sortAsc = false; }
      D.rows.sort((a, b) => cmp(a, b, k));
      renderRows();
    };
  });
}

// ---- 详情抽屉 ----
let icChart = null, eqChart = null;
function showDetail(r, tr) {
  curRow = r;
  if (tr) {
    document.querySelectorAll('tbody tr.sel').forEach(x => x.classList.remove('sel'));
    tr.classList.add('sel');
  }
  const d = $('detail'); d.classList.add('show');
  const m = D.meta[r.name] || {};
  $('dTitle').textContent = r.name;
  $('dSrc').innerHTML =
    `<span class="badge">来源 ${r.set}</span><span class="badge">族 ${r.fam}</span>` +
    (m.sub && m.sub !== '' ? `<span class="badge">子族 ${m.sub}</span>` : '') +
    `<span class="badge">${m.freq}</span><span class="badge">${m.mat}</span>` +
    (m.af ? `<span class="badge">覆盖 ${m.af} ~ ${m.al}</span>` : '') +
    `<span class="badge">区间 ${D.winLabel[win]}</span>` +
    (D.hasLS ? `<span class="badge">调仓 ${per === 'M' ? '月' : '周'}</span>` : '');
  $('dGrid').innerHTML =
    dcell('IC(h1)', curOf(r,'ic1')) + dcell('ICIR', curOf(r,'icir')) +
    dcell('NW-t', curOf(r,'nw'), 2) + dcell('IC胜率', curOf(r,'win'), 2) +
    dcell('IC(h5)', curOf(r,'ic5')) + dcell('IC(h10)', curOf(r,'ic10')) +
    dcell('IC(h20)', curOf(r,'ic20')) + dcell('覆盖率', r.cov, 2) +
    (D.hasLS ? dcell('多空年化', btOf(r,'ls')) + dcell('多头年化', btOf(r,'lo')) +
      dcell('超额年化', btOf(r,'ex')) + dcell('换手率', btOf(r,'to'), 2) : '');
  $('dFml').textContent = D.formula[r.name] || '（无公式记录）';

  // 月度 IC（按当前区间截取尾部 N 个月）
  const ctx1 = $('dIcChart');
  if (icChart) icChart.destroy();
  const start = D.winStart[win];
  const from = start ? start.slice(0, 7) : '';
  let idxFrom = 0;
  if (from) {
    const k = D.months.findIndex(x => x >= from);
    idxFrom = k < 0 ? D.months.length : k;
  }
  const hasIc = D.monthly[r.name] && D.months.length - idxFrom >= 1;
  $('dIcWrap').style.display = hasIc ? '' : 'none';
  $('dIcEmpty').style.display = hasIc ? 'none' : '';
  if (hasIc) {
    const labels = D.months.slice(idxFrom);
    const vals = D.monthly[r.name].slice(idxFrom);
    icChart = new Chart(ctx1, {
      type: 'bar',
      data: { labels: labels,
        datasets: [{ data: vals,
          backgroundColor: vals.map(v => v === null ? '#ccc' : (v > 0 ? '#A32D2D' : '#3B6D11')) }]},
      options: { responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: false },
          tooltip: { callbacks: { label: c => 'IC ' + (+c.raw).toFixed(4) } } },
        scales: { x: { ticks: { maxRotation: 90, minRotation: 60, font: { size: 9 } } },
                  y: { grid: { color: '#eee' } } } }
    });
  }
  // LS/LO 净值（按当前区间截取）
  const ctx2 = $('dEqChart');
  if (eqChart) eqChart.destroy();
  const eq = D.equity[r.name];
  const keep = eq ? eq.d.map((x, i) => (!start || x >= start) ? i : -1).filter(i => i >= 0) : [];
  const hasEq = eq && keep.length >= 2;
  $('dEqWrap').style.display = hasEq ? '' : 'none';
  $('dEqEmpty').style.display = hasEq ? 'none' : '';
  if (hasEq) {
    const labels = keep.map(i => eq.d[i]);
    const benchMap = {};
    if (D.bench) D.bench.d.forEach((x, i) => benchMap[x] = D.bench.v[i]);
    eqChart = new Chart(ctx2, {
      type: 'line',
      data: { labels: labels, datasets: [
        { label: '多空LS(月)', data: keep.map(i => eq.ls[i]), borderColor: '#A32D2D',
          backgroundColor: 'rgba(163,45,45,.08)', borderWidth: 1.4, pointRadius: 0, fill: true, tension: 0 },
        { label: '多头LO(月)', data: keep.map(i => eq.lo[i]), borderColor: '#185FA5',
          borderWidth: 1.4, pointRadius: 0, tension: 0 },
        ...(D.bench ? [{ label: '全A等权', data: labels.map(x => benchMap[x] ?? null),
          borderColor: '#999', borderDash: [4, 3], borderWidth: 1, pointRadius: 0, tension: 0 }] : [])]},
      options: { responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } },
        scales: { x: { ticks: { maxTicksLimit: 8, font: { size: 9 } } },
                  y: { grid: { color: '#eee' } } } }
    });
  }
}
function dcell(k, v, nd) {
  return `<div class="dcell"><div class="v ${cls(v)}">${fmt(v, nd)}</div><div class="k">${k}</div></div>`;
}
$('dClose').onclick = () => { $('detail').classList.remove('show'); };

renderRows();
"""


def render(dataset: str, n: int, reg: pd.DataFrame, payload: dict, months: list) -> str:
    sets = sorted(reg["set"].dropna().astype(str).unique())
    fams = sorted(reg["family"].dropna().astype(str).unique())
    has_ls = payload["hasLS"]
    has_w = payload["hasW"]
    has_ex = payload["hasEx"]
    date_min = min(months) if months else "—"
    date_max = max(months) if months else "—"
    n_ic = sum(1 for r in payload["rows"] if r["has_ic"])
    n_eq = len(payload["equity"])
    js = _JS.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False,
                                               separators=(",", ":")))

    # 大类 chips（按数量降序，带计数）；细分 chips 由前端按所选大类动态渲染
    from collections import Counter
    grp_cnt = Counter(r["grp"] for r in payload["rows"])
    chips = '<span class="chip on" data-g="__ALL__">全部</span>' + "".join(
        f'<span class="chip" data-g="{g}">{g} <span style="opacity:.65">{c}</span></span>'
        for g, c in sorted(grp_cnt.items(), key=lambda kv: -kv[1]))
    fam_opts = '<option value="">全部族</option>' + "".join(
        f'<option value="{f}">{f}</option>' for f in fams)
    ls_head = ("<th data-k='ls'>多空年化</th><th data-k='lo'>多头年化</th>"
               + ("<th data-k='ex'>超额年化</th>" if has_ex else "")
               + "<th data-k='to'>换手率</th>") if has_ls else ""
    per_sel = ("""<span class="chiprow" id="perSel" style="display:none">
<span class="chip on" data-p="M">月调仓</span><span class="chip" data-p="W">周调仓</span></span>"""
               if has_ls and has_w else "")
    win_chips = '<span style="font-size:12px;color:#888">区间</span>' + "".join(
        f'<span class="chip{" on" if k == "ALL" else ""}" data-w="{k}">{WIN_LABEL[k]}</span>'
        for k, _ in WINDOWS)

    body = f"""<div class="topbar">
<h1>YuriQuant 因子看板 — {dataset}</h1>
<div class="meta">{n} 个因子 | 逐日 IC {date_min} ~ {date_max} | 有 IC 序列 {n_ic} | 有分层净值 {n_eq}</div>
</div>
<div class="wrap">
<div class="main">
<div class="kpis">
<div class="kpi"><div class="v">{n}</div><div class="k">因子总数</div></div>
<div class="kpi"><div class="v">{len(grp_cnt)}</div><div class="k">来源大类（细分 {len(sets)}）</div></div>
<div class="kpi"><div class="v">{len(fams)}</div><div class="k">因子族</div></div>
<div class="kpi"><div class="v">{date_min}~{date_max}</div><div class="k">IC 覆盖区间</div></div>
</div>
<div class="filters">
<div class="chiprow" id="winChips">{win_chips}</div>
<div class="chiprow" id="grpChips">{chips}</div>
<div class="chiprow" id="subChips" style="display:none"></div>
<select id="famSel">{fam_opts}</select>
{per_sel}
<input class="search" id="fsearch" placeholder="搜索因子名 / 公式…">
<span style="font-size:12px;color:#888">显示 <b id="cnt">0</b> 个 | 点击表头排序，点行看详情</span>
</div>
<div class="note"><b>口径提示：</b>顶部「区间」控制<b>全部指标</b>的时间窗口（近三月/一年/三年/十年/全部，
锚点 = 数据最后一天）：IC 类按区间切片现算；回测类（年化/超额/换手）从 evals 日序列重算，
「全部」用 registry 登记值。Newey-West 稳健 t，滞后阶 L=int(4·(T/100)^(2/9))；
批量入库行为<b>全样本统计、无挖-验分离</b>。分层回测 = canonical TopK（k=30）多空/纯多，
月/周两档，成本走引擎默认费率；超额年化 = 纯多 − 全A等权。缺失显示 —。
口径详见 reports/README.md §一。红=正、绿=负（A 股惯例）。</div>
<div class="scroll"><table id="ftable">
<thead><tr>
<th data-k="name">因子名</th><th data-k="set">细分来源</th><th data-k="fam">族</th>
<th data-k="cov" class="num">覆盖率</th>
<th data-k="ic1" class="num">IC(h1)</th><th data-k="icir" class="num">ICIR</th>
<th data-k="nw" class="num">NW-t</th><th data-k="win" class="num">IC胜率</th>
<th data-k="ic5" class="num">IC(h5)</th><th data-k="ic10" class="num">IC(h10)</th>
<th data-k="ic20" class="num">IC(h20)</th>{ls_head}
<th>公式</th>
</tr></thead><tbody></tbody></table></div>
</div>
<div class="detail" id="detail">
<button id="dClose" style="float:right;border:none;background:none;font-size:16px;cursor:pointer">✕</button>
<h2 id="dTitle"></h2><div class="src" id="dSrc"></div>
<div class="dgrid" id="dGrid"></div>
<div class="formula" id="dFml"></div>
<div class="dchart" id="dIcWrap"><div style="font-size:12px;color:#555;margin-bottom:4px">月度 IC（红=正 / 绿=负）</div>
<div style="height:180px"><canvas id="dIcChart"></canvas></div></div>
<div style="display:none;font-size:12px;color:#999" id="dIcEmpty">该因子无逐日 IC 数据</div>
<div class="dchart" id="dEqWrap"><div style="font-size:12px;color:#555;margin-bottom:4px">分层净值（TopK k=30 · 月调仓 · 成本=引擎默认）</div>
<div style="height:200px"><canvas id="dEqChart"></canvas></div></div>
<div style="display:none;font-size:12px;color:#999" id="dEqEmpty">该区间内无分层净值数据（需先跑指标回填）</div>
</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
{js}
</script>"""
    # 外壳统一走 research.html_report.page（守卫 test_report_shell 要求全仓只有一份）
    return page(title=f"YuriQuant 因子看板 — {dataset}",
                body=body, css=_CSS, header="")


if __name__ == "__main__":
    main()
