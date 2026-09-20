"""从出榜产物生成每日邮件正文（markdown + 配套 HTML）。

用法（须在项目根目录执行，因为脚本用相对路径读产物）：
    # 常规：产物在 reports/alla_daily，对比基线取同口径目录
    python scripts/reporting/build_daily_email_body.py --ds 20260917 \
        --dir reports/alla_daily --prev-dir reports/alla_daily_ortho

    # 同目录内对比（--dir 与 --prev-dir 相同，可省略 --prev-dir）
    python scripts/reporting/build_daily_email_body.py --ds 20260916

输出两个文件：`email_body_<ds>.md`（存档/可读）与 `email_body_<ds>.html`（**发信用它**）。
发信务必用 `.html` + `--body-file`：`agently-cli` 对 `.md` 按 Markdown 发送，而国内邮箱
客户端不渲染 Markdown，收件人只会看到 `#`、`|`、`**` 原始符号（2026-09-17 首封实测）。

口径要点（**下游别改坏**）：
    - 「选股 Top 10」取 **picks_<ds>.csv**（已剔除 ST / 停牌 / 涨跌停封板的可交易候选），
      **不是** ranking_<ds>.csv 的全池打分前 10 —— 后者会排出买不进的标的（实测 09-17 全池前 10
      里 6 只是 ST，全部 tradable=False）。
    - 「与上一交易日的变化」也基于 picks，保持一致口径。
    - 跨口径对比时必须用 --prev-dir 指向**同口径**产物目录，否则会把口径差异误报成榜单变化
      （例如 reports/alla_daily 内 09-16 那份是 zscore 老口径，而 09-17 起是 ortho 新口径）。

产物列（与口径无关，新老口径同结构）：
    ranking_<ds>.csv        code/rank/name/industry_l2/industry_l1/score/pct_rank/
                            top_frac/tradable/flag
    picks_<ds>.csv          同上 + weight
    industry_rank_<ds>.csv  industry/rank/n_stocks/mean_score/median_score/mean_pct_rank/
                            n_top_frac/industry_l1/top_frac_share/top_stock
    industry_rank_l1_<ds>.csv
                            申万一级上卷（同上口径）+ n_picks/pick_share/
                            top_stock_tradable（可交易口径）；2026-09-18 起产出，
                            缺失时第三节自动降级为一行说明
    leaders_<ds>.csv        code/rank/cap_rank/name/industry_l2/industry_l1/mktcap/score/tradable
    history.csv             每轮摘要（新口径含 preproc/ensemble/excluded_features/horizons）
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from md_to_email_html import convert as md_to_html  # noqa: E402

CALENDAR = Path(r"E:\data\parquet\calendar.parquet")
STATUS = Path(r"E:\data\parquet\history_stock_status.parquet")


def next_trading_day(ds: str) -> tuple[str | None, bool]:
    """给定 YYYYMMDD，返回 (其后第一个交易日, 是否外推)。

    交易日历通常只覆盖到「当日」（数据源不提供未来日期），所以日历查不到时
    按「跳过周末的下一个工作日」外推，并把 extrapolated 置 True ——
    法定节假日以交易所公告为准，正文里必须如实标注是外推值。
    """
    try:
        cal = pd.read_parquet(CALENDAR)
        days = sorted({str(x)[:10].replace("-", "") for x in cal.iloc[:, 0].astype(str)})
        later = [d for d in days if d > ds]
        if later:
            return later[0], False
    except Exception:
        return None, False
    try:
        d = datetime.strptime(ds, "%Y%m%d") + timedelta(days=1)
    except Exception:
        return None, False
    while d.weekday() >= 5:  # 周六=5 周日=6
        d += timedelta(days=1)
    return d.strftime("%Y%m%d"), True


def fmt_day(ds: str) -> str:
    return f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"


def prev_file(d: Path, ds: str, prefix: str) -> Path | None:
    """目录里小于 ds 的最近一份 <prefix>_YYYYMMDD.csv（严格匹配，排除带 _后缀 的文件）。"""
    cands = []
    for p in glob.glob(str(d / f"{prefix}_*.csv")):
        m = re.search(rf"{prefix}_(\d{{8}})\.csv$", os.path.basename(p))
        if m and m.group(1) < ds:
            cands.append((m.group(1), Path(p)))
    return sorted(cands)[-1][1] if cands else None


def prev_ranking_file(d: Path, ds: str) -> Path | None:
    return prev_file(d, ds, "ranking")


def prev_industry_file(d: Path, pds: str, level: int = 2) -> Path | None:
    """同口径目录内的行业表（level=2 → 二级表；level=1 → 一级上卷表）。"""
    tag = "" if level == 2 else f"_l{level}"
    p = d / f"industry_rank{tag}_{pds}.csv"
    return p if p.exists() else None


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 组合风格暴露（读 reports/monitoring/production_ic_daily.csv）
# ---------------------------------------------------------------------------
#: 风格键 → 中文标签（顺序即展示顺序）
STYLE_LABELS = (("size", "市值"), ("mom", "20日动量"),
                ("vol", "20日波动率"), ("turn", "20日换手率"))
#: z 是「横截面 rank 百分位」标准化后的值（rank~U(0,1)，均值 .5、标准差 1/√12），
#: 因此可无损还原为分位：pct = 0.5 + z / √12。
_Z_TO_PCT = 1.0 / (12 ** 0.5)


def load_style_exposure(ds: str,
                        path: Path) -> dict | None:
    """读逐日 IC/暴露台账中 ``ds`` 那一行；缺失/落后时返回 None 或降级标记。

    返回 ``{z, ic_full, ic_recent, ic_neu_recent, n_days, stale}``。
    ``z`` 为组合持仓在各风格上的横截面分位（0~1）。
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "predict_date" not in df.columns:
        return None
    df = df.copy()
    df["ds"] = pd.to_datetime(df["predict_date"]).dt.strftime("%Y%m%d")
    cur = df[df["ds"] == ds]
    row = cur.iloc[-1] if len(cur) else df.iloc[-1]
    z = {}
    for k, _ in STYLE_LABELS:
        v = row.get(f"z_{k}")
        if v is not None and v == v:
            z[k] = min(max(0.5 + float(v) * _Z_TO_PCT, 0.0), 1.0)
    if not z:
        return None
    ic = df["ic_raw"].dropna() if "ic_raw" in df.columns else pd.Series(dtype=float)
    neu = df["ic_neutral"].dropna() if "ic_neutral" in df.columns else pd.Series(dtype=float)
    n_live = int((df.get("source") == "live").sum()) if "source" in df.columns else 0
    return {
        "z": z,
        "ds_row": str(row["ds"]),
        "stale": str(row["ds"]) != ds,
        "ic_full": float(ic.mean()) if len(ic) else None,
        "ic_recent": float(ic.tail(60).mean()) if len(ic) else None,
        "ic_n": int(len(ic)),
        "n_live": n_live,
        "ic_neu_recent": float(neu.tail(60).mean()) if len(neu) else None,
    }


def render_exposure_block(e: dict, ds: str) -> list[str]:
    """组合风格暴露 + 近期 IC 的一小段（压成 2 行，不新增编号小节）。"""
    parts = [f"{lab} **{e['z'][k]:.0%}**" for k, lab in STYLE_LABELS if k in e["z"]]
    low = [lab for k, lab in STYLE_LABELS
           if k in e["z"] and e["z"][k] < 0.4]
    high = [lab for k, lab in STYLE_LABELS
            if k in e["z"] and e["z"][k] > 0.6]
    if len(low) > len(high):
        tilt = "偏「" + " · ".join(f"低{t}" for t in low) + "」"
    elif high:
        tilt = "偏「" + " · ".join(f"高{t}" for t in high) + "」"
    else:
        tilt = "接近市场中位，无明显风格倾斜"

    out = [f"**组合风格暴露**（模型 Top10% 等权组合持仓在全市场的横截面分位，"
           f"50% = 中位）：{' · '.join(parts)} —— {tilt}。"]
    if e["ic_recent"] is not None:
        extra = ""
        if e.get("n_live"):
            extra = f"，其中 {e['n_live']} 日为实盘每日记录、其余为 2018 年起回测重放"
        ic_note = (f"**近期表现**：已有 {e['ic_n']} 个交易日的次日 IC 记录{extra}，"
                   f"近 60 日均值 {e['ic_recent']:+.3f}"
                   f"（全期 {e['ic_full']:+.3f}）")
        if e["ic_neu_recent"] is not None:
            ic_note += f"，剥离风格后 {e['ic_neu_recent']:+.3f}"
        ic_note += "；单日 IC 方差极大（±0.3 常见），判断失效要看多日累积而非单日。"
        out.append(ic_note)
    if e["stale"]:
        out.append(f"> ⚠️ 风格暴露取的是 **{fmt_day(e['ds_row'])}** 的截面"
                   f"（当日 {fmt_day(ds)} 尚无记录），仅供参考。")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="预测日 YYYYMMDD")
    ap.add_argument("--dir", default="reports/alla_daily", help="产物目录")
    ap.add_argument("--out", default=None, help="输出 md 路径（默认 <dir>/email_body_<ds>.md）")
    ap.add_argument("--prev-dir", default=None,
                    help="对比基线目录（默认同 --dir）。跨口径时指向同口径产物目录，例如"
                         " --dir reports/alla_daily --prev-dir reports/alla_daily_ortho")
    ap.add_argument("--html-out", default=None,
                    help="HTML 版输出路径（默认与 --out 同名 .html）。发邮件应发 HTML 版，"
                         "见 md_to_email_html 的说明：.md 会被 CLI 当 Markdown 发，客户端不渲染")
    ap.add_argument("--no-html", action="store_true", help="不生成 HTML 版")
    ap.add_argument("--exposure-csv", default="reports/monitoring/production_ic_daily.csv",
                    help="逐日 IC/风格暴露台账（由 scripts/reporting/monitor_production_ic.py"
                         " 产出）；文件缺失时第一节不展示风格暴露，不阻断")
    args = ap.parse_args()

    ds = args.ds
    d = Path(args.dir)
    out = Path(args.out) if args.out else d / f"email_body_{ds}.md"

    rank = pd.read_csv(d / f"ranking_{ds}.csv")
    picks = pd.read_csv(d / f"picks_{ds}.csv")
    ind = pd.read_csv(d / f"industry_rank_{ds}.csv")
    # 一级行业表为可选产物（2026-09-18 起由出榜链落盘）；缺失时第三节降级为说明行
    ind_l1_p = d / f"industry_rank_l1_{ds}.csv"
    ind_l1 = pd.read_csv(ind_l1_p) if ind_l1_p.exists() else None
    lead = pd.read_csv(d / f"leaders_{ds}.csv")
    hist = pd.read_csv(d / "history.csv")
    h = hist[hist["predict_date"].astype(str) == ds]
    h = h.iloc[-1] if len(h) else hist.iloc[-1]

    nxt, nxt_extrap = next_trading_day(ds)
    # ---- 元信息 ----
    preproc = str(h.get("preproc", "") if "preproc" in h else "")
    ens = str(h.get("ensemble", "") if "ensemble" in h else "")
    excl = h.get("excluded_features", "") if "excluded_features" in h else ""
    hor = h.get("horizons", "") if "horizons" in h else ""
    n_scored = int(h["n_scored"])
    n_picks = int(h["n_picks"])
    n_ind = int(h["n_industries"])
    per_h = h.get("per_horizon", "")
    frac = float(h.get("frac", 0.1))

    train_note = "500 日滚动训练窗"
    if isinstance(per_h, str) and per_h:
        pat = (r"'train_begin': '([\d-]+)'.*?'train_end': '([\d-]+)'"
               r".*?'horizon': (\d+)")
        m = re.findall(pat, per_h)
        if m:
            parts = [f"h{hz}: {b} ~ {e}" for b, e, hz in m]
            train_note = "500 日滚动训练窗（" + "；".join(parts) + "）"

    lines: list[str] = []
    lines.append(f"# YuriQuant 每日选股预测 · {fmt_day(ds)} 收盘")
    lines.append("")
    if nxt:
        _note = "；该日期为日历外推" if nxt_extrap else ""
        lines.append(f"**预测日**：{fmt_day(ds)} 收盘截面（代表对 **{fmt_day(nxt)}** "
                     f"的选股观点{_note}）")
    else:
        lines.append(f"**预测日**：{fmt_day(ds)} 收盘截面（代表对**下一个交易日**的选股观点）")
    lines.append(f"**模型**：GBDT（LightGBM）· {train_note} · 全 A 池 {n_scored} 只（有效打分）")
    lines.append(f"**口径**：Top{frac:.0%} ＝ {int(h['n_top_frac'])} 只，"
                 f"剔除不可交易后 ＝ **{n_picks} 只候选**（等权参考权重）")
    lines.append(f"**细分行业**：申万二级 {n_ind} 个")
    if preproc or ens:
        bits = []
        if preproc:
            bits.append(f"预处理 {preproc}")
        if hor:
            bits.append(f"horizons {hor}")
        if ens:
            bits.append(f"集成 {ens}")
        if excl and str(excl) not in ("[]", "nan", ""):
            bits.append(f"剔除特征 {excl}")
        lines.append("**特征/集成**：" + " · ".join(bits))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 一、Top10（可交易口径）----
    # 榜单必须展示「买得进」的股票：ranking 是全池打分（含 ST/停牌/涨跌停封板等不可交易标的），
    # picks 才是剔除不可交易后的候选集。历史邮件同样没有直接取全池 Top10。
    raw_top10 = rank.nsmallest(10, "rank")
    top10 = picks.nsmallest(10, "rank")
    lines.append("## 一、选股榜 Top 10（可交易口径）")
    lines.append("")
    rows = [[int(r["rank"]), r["code"], r["name"], r["industry_l2"], r["industry_l1"],
             f"{r['score']:.4f}"] for _, r in top10.iterrows()]
    lines.append(md_table(["全池名次", "代码", "名称", "二级行业", "一级行业", "分数"], rows))
    lines.append("")
    dropped = raw_top10[~raw_top10["tradable"].astype(bool)]
    if len(dropped):
        dn = "、".join(f"{r['code']} {r['name']}" for _, r in dropped.iterrows())
        n_st = int(dropped["name"].astype(str).str.upper().str.contains("ST").sum())
        lines.append(f"> **口径说明**：全池打分前 10 中有 **{len(dropped)} 只不可交易**"
                     f"（ST {n_st} 只，其余为停牌 / 涨跌停封板；判定见 `data/tradability.py`），"
                     f"已按可交易口径剔除：{dn}。上表为剔除后的可交易 Top10，"
                     f"「全池名次」列为该股在全池中的原始名次。")
        lines.append("")
    l1 = top10["industry_l1"].value_counts()
    l2 = top10["industry_l2"].value_counts()
    seg = "、".join(f"{k}（{v} 席）" for k, v in l2.items() if v >= 2)
    dom = "、".join(f"{k}（{v} 席）" for k, v in l1.items() if v >= 2)
    desc = "Top10 集中于"
    if seg and dom:
        desc += f"二级行业 {seg}；一级行业 {dom}"
    elif seg:
        desc += f"二级行业 {seg}"
    elif dom:
        desc += f"一级行业 {dom}"
    else:
        desc += "分散，无行业出现 2 席及以上"
    lines.append(f"**主线特征**：{desc}。")
    lines.append("")
    # ---- 一之二、组合风格暴露（事前可算，不需未来收益）----
    expo = load_style_exposure(ds, Path(args.exposure_csv))
    if expo:
        lines.extend(render_exposure_block(expo, ds))
    lines.append("---")
    lines.append("")

    # ---- 二、行业榜 ----
    lines.append("## 二、细分行业排名 Top 10（申万二级）")
    lines.append("")
    ind10 = ind.nsmallest(10, "rank")
    rows = [[int(r["rank"]), r["industry"], r["industry_l1"], int(r["n_stocks"]),
             f"{r['mean_score']:+.4f}", f"{r['top_frac_share']:.1%}", r["top_stock"]]
            for _, r in ind10.iterrows()]
    lines.append(md_table(["排名", "二级行业", "一级归属", "股数", "均分",
                           "Top10%占比", "榜首"], rows))
    lines.append("")
    tail = ind.nlargest(5, "rank")
    tail_s = "、".join(f"{r['industry']}（{r['mean_score']:+.4f}）" for _, r in tail.iterrows())
    lines.append(f"**尾部 5**：{tail_s}。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 三、一级行业榜（申万一级上卷，2026-09-18 起）----
    if ind_l1 is not None and len(ind_l1):
        lines.append("## 三、一级行业排名 Top 5（申万一级）")
        lines.append("")
        l1_top = ind_l1.nsmallest(5, "rank")
        rows = [[int(r["rank"]), r["industry"], int(r["n_stocks"]),
                 f"{r['mean_score']:+.4f}", f"{r['top_frac_share']:.1%}",
                 int(r.get("n_picks", 0)), r.get("top_stock_tradable", "")]
                for _, r in l1_top.iterrows()]
        lines.append(md_table(["排名", "一级行业", "股数", "均分", "Top10%占比",
                               "候选", "可交易龙头"], rows))
        lines.append("")
        zero = ind_l1[ind_l1["top_frac_share"] <= 0]
        if len(zero):
            zs = "、".join(str(x) for x in zero["industry"])
            lines.append(f"**零超配行业**（组内无一只进全A Top{frac:.0%}）：{zs}。")
            lines.append("")
        l1_tail = ind_l1.nlargest(3, "rank")
        l1_tail_s = "、".join(f"{r['industry']}（{r['mean_score']:+.4f}）"
                              for _, r in l1_tail.iterrows())
        lines.append(f"**尾部 3**：{l1_tail_s}。")
        lines.append("")
        lines.append(f"> **口径**：本表是把**同一批个股分数**按申万一级聚合（上卷），"
                     f"不是另一次打分，与上表二级同源。"
                     f"「候选」= 组内可交易候选数（`top_frac & tradable`），"
                     f"「可交易龙头」= 组内可交易成员中的最高分 —— "
                     f"全池第一名可能落在 ST／停牌上（判定见 `data/tradability.py`），"
                     f"故另给可交易口径。完整 {len(ind_l1)} 行见附件 "
                     f"`industry_rank_l1_{ds}.csv`。")
    else:
        lines.append("## 三、一级行业排名（申万一级）")
        lines.append("")
        lines.append(f"本日产物缺 `industry_rank_l1_{ds}.csv`（2026-09-18 起由出榜链"
                     f"落盘），本节跳过。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 四、龙头股 ----
    lines.append("## 四、龙头股榜 Top 5（大市值口径）")
    lines.append("")
    lead5 = lead.nsmallest(5, "rank")
    rows = [[int(r["rank"]), r["code"], r["name"], r["industry_l2"], r["industry_l1"],
             f"{r['mktcap']:.0f}", f"{r['score']:.4f}"] for _, r in lead5.iterrows()]
    lines.append(md_table(["排名", "代码", "名称", "二级行业", "一级行业",
                           "市值(亿)", "分数"], rows))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 四、与上一日对比 ----
    prev_dir = Path(args.prev_dir) if args.prev_dir else d
    prev = prev_file(prev_dir, ds, "picks")
    lines.append("## 五、与上一交易日的变化")
    lines.append("")
    if prev:
        same_caliber = prev_dir.resolve() == d.resolve()
        if not same_caliber:
            lines.append(f"> 对比基线取自**同口径**产物目录 `{prev_dir}`；本目录 `{d}` 内的同日产物"
                         f"若属其他口径则不参与对比，避免把口径差异误报成榜单变化。")
            lines.append("")
        pds = re.search(r"picks_(\d{8})\.csv$", prev.name).group(1)
        pr = pd.read_csv(prev)
        # 口径一致性核对：读基线目录 history.csv 的 preproc 字段（老口径产物无该字段）
        cur_cal = str(preproc) if preproc else "(无 preproc 字段)"
        prev_cal = None
        _hp = prev_dir / "history.csv"
        if _hp.exists():
            try:
                _h = pd.read_csv(_hp)
                _row = _h[_h["predict_date"].astype(str) == pds]
                if len(_row):
                    _r = _row.iloc[-1]
                    prev_cal = (str(_r["preproc"]) if "preproc" in _h.columns
                                else "(无 preproc 字段)")
            except Exception:
                prev_cal = None
        if prev_cal is not None and prev_cal != cur_cal:
            lines.append(f"> ⚠️ **口径不一致警告**：当前口径 `{cur_cal}`，"
                         f"基线 {fmt_day(pds)} 口径为 `{prev_cal}` —— "
                         f"下列「变化」含口径差异，不能全部归因于榜单真实变动。")
            lines.append("")
        cur_codes = list(picks.nsmallest(10, "rank")["code"])
        prv_codes = list(pr.nsmallest(10, "rank")["code"])
        keep = [c for c in cur_codes if c in set(prv_codes)]
        new_in = [c for c in cur_codes if c not in set(prv_codes)]
        drop = [c for c in prv_codes if c not in set(cur_codes)]
        nm = dict(zip(picks["code"], picks["name"]))
        nm_p = dict(zip(pr["code"], pr["name"]))

        def label(codes, m):
            return "、".join(f"{c} {m.get(c, '')}" for c in codes) if codes else "无"

        lines.append(f"**① 选股 Top10（可交易口径）**："
                     f"替换 {len(new_in)} 只，留存 {len(keep)} 只。")
        lines.append("")
        lines.append(f"- 新进：{label(new_in, nm)}")
        lines.append(f"- 掉出（{fmt_day(pds)}）：{label(drop, nm_p)}")
        lines.append("")

        pi_path = prev_industry_file(prev_dir, pds)
        pi_l1_path = prev_industry_file(prev_dir, pds, level=1)
        if pi_path:
            pi = pd.read_csv(pi_path)
            p3 = list(pi.nsmallest(3, "rank")["industry"])
            c3 = list(ind.nsmallest(3, "rank")["industry"])
            # 一级列：基线或当日缺 L1 表时整列 "-"（不臆造、不静默省略）
            if pi_l1_path and ind_l1 is not None:
                pi_l1 = pd.read_csv(pi_l1_path)
                p3_l1 = list(pi_l1.nsmallest(3, "rank")["industry"])
                c3_l1 = list(ind_l1.nsmallest(3, "rank")["industry"])
            else:
                p3_l1, c3_l1 = [], []
            lines.append("**② 行业榜前三重排**（二级 ＋ 一级）：")
            lines.append("")
            hdr = ["位次", f"{fmt_day(pds)} 二级", f"{fmt_day(ds)} 二级",
                   f"{fmt_day(pds)} 一级", f"{fmt_day(ds)} 一级"]
            lines.append(md_table(hdr,
                                  [[i + 1, p3[i] if i < len(p3) else "-",
                                    c3[i] if i < len(c3) else "-",
                                    p3_l1[i] if i < len(p3_l1) else "-",
                                    c3_l1[i] if i < len(c3_l1) else "-"]
                                   for i in range(3)]))
            lines.append("")
            ind_note = f"（上一份 {fmt_day(pds)}：候选 {len(pr)} 只 / {len(pi)} 行业）"
        else:
            ind_note = f"（上一份 {fmt_day(pds)}：候选 {len(pr)} 只）"
        lines.append(f"**③ 池子规模**：有效打分 {n_scored} 只，可交易候选 {n_picks} 只，"
                     f"行业数 {n_ind} 个{ind_note}。")
    else:
        lines.append(f"`{prev_dir}` 内无更早的 ranking 产物，无法自动对比。")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 五、Caveat ----
    lines.append("## 六、Caveat（必读）")
    lines.append("")
    cav = []
    if nxt:
        _note2 = "，该日期由交易日历外推（法定节假日以交易所公告为准）" if nxt_extrap else ""
        cav.append(f"**预测日 = {fmt_day(ds)} 收盘截面**，即对 **{fmt_day(nxt)}** 的选股观点；"
                   f"不是对当天{_note2}。")
    else:
        cav.append(f"**预测日 = {fmt_day(ds)} 收盘截面**，"
                   f"即对**下一个交易日**的选股观点；不是对当天。")
    if preproc == "ortho":
        cav.append("**口径**：本榜为实时正交化（ortho）+ h1/h5 秩平均集成口径，"
                   "已剔除 `limit_pos` 等状态族特征（状态族是可交易性掩码，非收益预测因子）。")
    cav.append("**榜单范围**：正文 Top10 与「与上一日变化」均基于**可交易候选集**"
               "（已剔除 ST / 停牌 / 涨跌停封板，判定见 `data/tradability.py`），非全池原始打分；"
               "被剔除标的见第一节口径说明。")
    cav.append("**行业表口径**：二级表按 `industry_l2` 聚合，一级表（第三节）是把同一批"
               "个股分数按 `industry_l1` **上卷** —— 不是另一次打分；两表的 `top_stock` 均为"
               "组内**全池**第一名（可能落在 ST/停牌上），一级表另给可交易口径列。")
    cav.append("**风格暴露**：「组合风格暴露」是模型的**固有属性**，不是当日信号 —— "
               "组合结构性偏小市值 / 低波动 / 低流动性。风格逆向滚动时组合会跑输，"
               "而同期选股的**相对**排序（中性化 IC）可能仍然有效；"
               "把风格逆风误判成模型失效会导致错误的调参。口径见 "
               "`reports/monitoring/production_ic_daily.csv`。")
    # 状态表降级检测
    try:
        st = pd.read_parquet(STATUS)
        col = next((c for c in st.columns
                    if str(c).lower() in ("date", "trade_date", "datetime")), None)
        if col is not None:
            days = st[col]
        else:
            ix = st.index
            days = ix.get_level_values(0) if isinstance(ix, pd.MultiIndex) else ix
        smax = max(str(x)[:10].replace("-", "") for x in days)
        if smax < ds:
            cav.append(f"**状态表降级**：`history_stock_status` 最新数据停在 **{fmt_day(smax)}**，"
                       f"预测日 {fmt_day(ds)} 的涨跌停/停牌/ST 由日线行情推断替代，"
                       f"`suspend_*`/`st_days` 等状态族特征在预测日可能为 NaN"
                       f"（静默的轻微信号损失，不影响出榜正确性）。")
        else:
            cav.append(f"**状态表**：`history_stock_status` 已更新至 "
                       f"{fmt_day(smax)}，与预测日一致。")
    except Exception as e:
        cav.append(f"**状态表检查失败**：{e}")
    cav.append("**分数含义**：新口径 score 为 h1/h5 两个 horizon 的"
               "截面百分位秩均值（0~1，越大越靠前），"
               "与旧口径的原始回归分不可直接跨期比较。")
    cav.append("**无整手约束**：weight 列为等权占位，实际下单需按 `floor(w·V/(P·100))·100` 取整，"
               "每只目标金额建议 ≥2 万元。")
    for i, c in enumerate(cav, 1):
        lines.append(f"{i}. {c}")
    lines.append("")
    lines.append("---")
    lines.append("")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    prod = f"`ranking_{ds}.csv` / `picks_{ds}.csv` / `industry_rank_{ds}.csv`"
    if ind_l1 is not None:
        prod += f" / `industry_rank_l1_{ds}.csv`"
    prod += f" / `leaders_{ds}.csv`"
    lines.append(f"*生成时间：{now} · 数据源 AmazingData (PIT) · 产物：{prod}*")
    lines.append("")

    md_text = "\n".join(lines)
    out.write_text(md_text, encoding="utf-8")
    print(f"written: {out} ({len(lines)} lines)")

    if not args.no_html:
        hp = Path(args.html_out) if args.html_out else out.with_suffix(".html")
        html = md_to_html(md_text, title=f"YuriQuant 每日选股预测 · {fmt_day(ds)}")
        hp.write_text(html, encoding="utf-8")
        print(f"written: {hp} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
