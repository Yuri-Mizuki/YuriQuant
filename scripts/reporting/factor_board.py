"""因子看板前端（仿聚宽因子库：列表页 + 因子详情页）。

页面结构
--------
``reports/factor_board/index.html``
    筛选区（分类 / 因子集 / 回溯周期 / 组合构建 / 手续费）+ 可排序表格：
    因子名称 · 最小分位/最大分位 超额年化 · 换手率 · IC均值 · IR值。
``reports/factor_board/f/<name>.html``
    因子定义（类别/公式/更新/数据处理）+ 控制条（区间/成本，**可切换**）+
    IC 三格 + 分位业绩表 + 净值曲线（普通/对数轴）+ IC 时序（含 22 日均线）+
    行业 IC 横条 + 换手率曲线。

口径（页面底部与 README 同文）
------------------------------
组合为**月频、T+1 开盘成交、组内等权**；成本档「27bp」= 生产口径
（佣金万1双边 + 时变印花税 + 20bp 滑点）；基准 = 全A等权；
超额 = (1+分位年化)/(1+基准年化)−1。**这是因子横向比较用的展示口径，
不是定版改进结论**——判改进仍须走 ``evaluation/caliber.py`` + 21 相位。

用法::

    python -m scripts.reporting.factor_board            # 生成 index + 全部详情页
    python -m scripts.reporting.factor_board --top 80   # 只生成前 80 个的详情页
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from stats.robust_stats import nw_tstat

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "reports" / "factor_board" / "cache"
OUT = ROOT / "reports" / "factor_board"
ECHARTS = ROOT / "reports" / "docs" / "design" / "_shared" / "js" / "echarts.min.js"

# 数据盘 / 库根走 config 单一真源（原先硬编码 D:/data 与 E:/data，换机或
# 数据根切换后会静默退化：`D:/data/parquet` 2026-11-18 已删除，硬编码会让
# 行业名 resolution 悄悄回落到内置 30 条表，页面看着正常但行业 IC 是错的）。
_PATHS = Config.paths()
DATASET = "all_a_2018_2026"
REGISTRY = Path(_PATHS["factor_library_root"]) / DATASET / "registry.csv"
IND_SRC = Path(_PATHS["cache_root"]) / "industry_classification_level1.parquet"


def _industry_names() -> dict:
    """申万一级 代码→中文名（来自行业分类缓存；缺文件时用内置表兜底）。"""
    fallback = {
        "801010.SI": "农林牧渔", "801030.SI": "基础化工", "801040.SI": "钢铁",
        "801050.SI": "有色金属", "801080.SI": "电子", "801110.SI": "家用电器",
        "801120.SI": "食品饮料", "801130.SI": "纺织服饰", "801140.SI": "轻工制造",
        "801150.SI": "医药生物", "801160.SI": "公用事业", "801170.SI": "交通运输",
        "801180.SI": "房地产", "801200.SI": "商贸零售", "801210.SI": "社会服务",
        "801230.SI": "综合", "801710.SI": "建筑材料", "801720.SI": "建筑装饰",
        "801730.SI": "电力设备", "801740.SI": "国防军工", "801750.SI": "计算机",
        "801760.SI": "传媒", "801770.SI": "通信", "801780.SI": "银行",
        "801790.SI": "非银金融", "801880.SI": "汽车", "801890.SI": "机械设备",
        "801950.SI": "煤炭", "801960.SI": "石油石化", "801970.SI": "环保",
        "801980.SI": "美容护理",
    }
    if not IND_SRC.exists():
        return fallback
    try:
        d = pd.read_parquet(IND_SRC, columns=["industry_code", "industry_name"]).dropna()
        return {**fallback, **dict(d.drop_duplicates("industry_code").values)}
    except Exception:                                     # noqa: BLE001
        return fallback

# ---------------------------------------------------------------------------
# 分类与因子集（2026-11-18 重建，旧口径见 git 历史）
#
# 旧口径按 `registry.family` 原值 + `source` 一刀切：739/888（83%）落在「量价」，
# 勾掉它只剩 149 个 ⇒ 筛选形同虚设；且 family 与 source 一一对应，等于第二份来源表。
# 三个硬伤（详见 reports/factor_board/README.md §一）：
#   1. alpha360 的 352 个是「原始 OHLCV 滞后」（CLOSE_k = close.shift(k)/close），
#      本质是不同回看期的反转，不是 352 个独立因子，却混在「量价」里；
#   2. alpha158 的「134 个」实为 35 个基名 × 5 窗口，同族参数变体被当独立因子；
#   3. style 的 17 个（研发强度/资产负债率/流动比率…）本是财务比率，却被标成「风格」。
# 现在按「因子在算什么」分 13 个逻辑族，只改展示层映射，**不动 registry.csv**。
# ---------------------------------------------------------------------------
LOGIC_FAMILIES = [
    "原始序列", "均线通道", "趋势强度", "连续涨跌", "极值形态", "动量波动",
    "价量衍生", "基本面质量", "盈余与成长", "股东治理", "资金与情绪",
    "事件与宏观", "论文策略",
]

# Alpha158：35 个基名 × 5 窗口 → 5 个逻辑族
A158_FAMILY = {
    **dict.fromkeys(["MA", "MAX", "MIN", "QTLU", "QTLD", "RANK", "VMA", "VWAP",
                     "HIGH", "LOW"], "均线通道"),
    **dict.fromkeys(["BETA", "CORR", "CORD", "RSQR", "RESI"], "趋势强度"),
    **dict.fromkeys(["CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "VSUMP", "VSUMD"],
                    "连续涨跌"),
    **dict.fromkeys(["IMAX", "IMIN", "IMXD", "RSV", "KSFT", "KMID", "KLEN",
                     "KUP", "KLOW"], "极值形态"),
    **dict.fromkeys(["ROC", "STD", "VSTD", "WVMA"], "动量波动"),
}

# 因子集归并：19 个 source → 7 组（组只是快捷全选，筛选仍按子集判定）
SRC_GROUP = {
    "alpha101": "公开 Alpha 库", "alpha158": "公开 Alpha 库", "alpha191": "公开 Alpha 库",
    "alpha360": "原始序列",
    "fundamental": "基本面", "constructed": "基本面", "p5": "基本面", "style": "基本面",
    # sue_pledge 4/7 是 SUE 盈余惊喜（其余 3 个质押冻结），按主体归「基本面」；
    # 组是「一个 source 只能属于一个组」的粗维度，逻辑族仍按 label 逐条拆
    "sue_pledge": "基本面",
    "holder": "股东治理", "holder_dyn": "股东治理", "disc_holder_dyn": "股东治理",
    "pledge": "股东治理",
    "event": "事件与宏观", "status": "事件与宏观", "altf": "事件与宏观",
    "moneyflow": "资金与两融", "margin": "资金与两融",
    "paper": "论文策略",
}
SRC_SUB = {
    "alpha101": "Alpha101", "alpha158": "Alpha158", "alpha191": "Alpha191",
    "alpha360": "Alpha360",
    "fundamental": "财务指标", "constructed": "自建合成", "p5": "P5 挖掘",
    "style": "财务比率",
    "holder": "股东户数", "holder_dyn": "实控人变动", "disc_holder_dyn": "大宗折价",
    "pledge": "质押冻结", "sue_pledge": "SUE/质押",
    "event": "财报事件", "status": "交易状态", "altf": "宏观另类",
    "moneyflow": "大宗资金", "margin": "两融", "paper": "21 式",
}
GROUP_ORDER = ["公开 Alpha 库", "原始序列", "基本面", "股东治理",
               "事件与宏观", "资金与两融", "论文策略"]


def _logic_family(src: str, label: str) -> str:
    """source + 因子名 → 逻辑族（判不出来的落「基本面质量」并保持可追）。"""
    if src == "alpha158":
        base = re.sub(r"\d+$", "", str(label).split("#")[-1])
        return A158_FAMILY.get(base, "动量波动")
    if src in ("alpha101", "alpha191"):
        return "价量衍生"
    if src == "alpha360":
        return "原始序列"
    if src == "paper":
        return "论文策略"
    if src in ("altf", "event", "status"):
        return "事件与宏观"
    if src in ("margin", "moneyflow"):
        return "资金与情绪"
    if src in ("holder", "holder_dyn", "disc_holder_dyn", "pledge"):
        return "股东治理"
    if src == "sue_pledge":                       # 一半质押冻结、一半 SUE
        return "股东治理" if re.search("质押|冻结", str(label)) else "盈余与成长"
    if src == "p5":
        return "盈余与成长"
    return "基本面质量"                            # fundamental / constructed / style


WINDOWS = [("1y", "近 1 年"), ("3y", "近 3 年"), ("5y", "近 5 年"), ("all", "全样本")]

CSS = """
:root{--navy:#16304f;--blue:#2f6fdb;--bg:#f4f5f7;--card:#fff;--line:#e5e8ee;
      --ink:#22262e;--muted:#8a92a3;--pos:#d9363e;--neg:#12a05c;--q1:#2f6fdb;--q5:#e07b39}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:13px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
a{color:var(--blue);text-decoration:none}
.topbar{background:var(--navy);color:#fff;height:50px;display:flex;align-items:center;
        padding:0 24px;gap:18px;position:sticky;top:0;z-index:30}
.topbar b{font-size:16px;letter-spacing:.5px}
.topbar .sp{flex:1}
.topbar .meta{font-size:12px;color:#a9bdd6}
.topbar a{color:#cfe0f5;font-size:12.5px;padding:4px 10px;border-radius:4px}
.topbar a:hover{background:rgba(255,255,255,.12)}
.topbar a.on{background:rgba(255,255,255,.18);color:#fff}
.wrap{max-width:1240px;margin:0 auto;padding:18px 24px 60px}
/* 列表页 8 列：1240 会挤到横向溢出，单独放宽 */
.wrap.wide{max-width:1480px}
.card{background:var(--card);border:1px solid var(--line);border-radius:4px;
      margin-bottom:16px}
.card-h{display:flex;align-items:center;gap:12px;padding:13px 18px;
        border-bottom:1px solid var(--line);font-weight:600;font-size:14px}
.card-h .sp{flex:1}
.card-h .hint{font-weight:400;font-size:12px;color:var(--muted)}
.card-bd{padding:14px 18px}
.card-bd.tight{padding:0}
h1.fname{font-size:21px;font-weight:500;margin:6px 0 14px}
h1.fname code{font-size:15px;color:var(--muted);font-weight:400;margin-left:8px}

/* 筛选 */
.frow{display:flex;gap:10px;align-items:flex-start;padding:7px 0;font-size:13px}
.frow>.lb{width:82px;color:var(--muted);flex:none;text-align:right;padding-top:1px}
.frow>.opts{display:flex;flex-wrap:wrap;gap:6px 18px}
label.chk{display:inline-flex;align-items:center;gap:5px;cursor:pointer;
          color:#3d4557;user-select:none}
label.chk input{accent-color:var(--blue);margin:0}
label.chk.dis{color:#b6bcc9;cursor:not-allowed}
/* 因子集两级：组 → 子集（默认收起，避免 19 个复选框平铺） */
.gsrc{display:inline-flex;align-items:center;gap:8px}
.gsrc .exp{color:var(--blue);font-size:11.5px;cursor:pointer;border-bottom:1px dotted #b8cdf0}
.gsrc .exp:hover{color:#1b52b8}
.gsrc .subs{display:none;gap:6px 14px;flex-wrap:wrap;
            width:100%;padding:2px 0 2px 18px;border-left:2px solid #e8eef8;margin-left:2px}
.gsrc .subs.on{display:flex}
.gsrc .subs .sub{font-size:12px;color:#5d6474}
.gsrc .subs .sub input{width:12px;height:12px}
.searchbox{float:right;font-size:13px}
.searchbox input{border:1px solid var(--line);border-radius:3px;padding:5px 9px;width:210px;outline:none}
.searchbox input:focus{border-color:var(--blue)}

/* 表格 */
.tbl{width:100%;border-collapse:collapse;font-size:12.5px}
.tbl th{background:#fafbfc;color:#5d6474;font-weight:500;text-align:right;
        padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;
        position:sticky;top:50px;z-index:5;cursor:pointer;user-select:none}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl th.sorted{color:var(--blue)}
.tbl th .ar{font-size:9px;margin-left:3px;opacity:.55}
.tbl td{padding:8px 12px;border-bottom:1px solid #f0f2f5;text-align:right;
        font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl tbody tr:hover{background:#f6f9ff;cursor:pointer}
.tbl td.nm{color:#3a5f9e;max-width:290px}
.tbl td.nm .nm1{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tbl .deg{background:#fdeaea;color:#c0392b;font-size:10.5px;padding:0 5px;border-radius:2px;
          vertical-align:1px}
.tbl td.nm .nm2{color:#9aa2b2;font-size:11.5px;overflow:hidden;text-overflow:ellipsis;
                white-space:nowrap}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:#b6bcc9}
.tblwrap{max-height:none;overflow:auto}
.foot{padding:10px 18px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);
      display:flex;gap:16px}

/* 详情 */
.defs{display:grid;grid-template-columns:84px 1fr;gap:5px 10px;font-size:12.5px;color:#3d4557}
.defs .k{color:var(--muted)}
.defs code{background:#f6f7f9;padding:1px 5px;border-radius:3px;font-size:12px}
.ctrl{display:flex;flex-wrap:wrap;gap:14px 22px;align-items:center;
      border:1px solid var(--line);border-radius:4px;padding:14px 18px;background:#fcfcfd}
.ctrl .it{display:flex;align-items:center;gap:8px;font-size:12.5px;color:#5d6474}
.ctrl select{border:1px solid var(--line);border-radius:3px;padding:5px 8px;background:#fff;
             font-size:12.5px;color:var(--ink);outline:none;min-width:110px}
.ctrl select:disabled{background:#f4f5f7;color:#a8aebb}
.btn{background:var(--blue);color:#fff;border:0;border-radius:3px;padding:7px 26px;
     font-size:13px;cursor:pointer}
.btn:hover{background:#2560c6}
.btn.ghost{background:#fff;color:var(--blue);border:1px solid #c9d8f5}
.ic3{display:flex;gap:0;padding:6px 0}
.ic3 div{flex:1;padding:4px 18px}
.ic3 .k{color:var(--muted);font-size:12px}
.ic3 .v{font-size:26px;font-weight:400;letter-spacing:.5px}
.mtbl th{text-align:right}.mtbl th:first-child,.mtbl td:first-child{text-align:left}
.mtbl th{background:transparent;position:static;cursor:default;color:var(--muted);
         font-size:12px;padding:6px 12px;border-bottom:0}
.mtbl td{padding:7px 12px;font-size:13px}
.mtbl td:first-child{color:#3d4557}
.chart{height:360px}
.chart-sm{height:360px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.toggles{display:flex;gap:18px;align-items:center;font-size:12.5px;color:#5d6474}
.toggles label{display:inline-flex;gap:5px;align-items:center;cursor:pointer}
.note{color:var(--muted);font-size:12px;line-height:1.8;padding:4px 18px 14px}
.note code{background:#f6f7f9;padding:1px 4px;border-radius:3px}
.tag{display:inline-block;padding:1px 7px;border-radius:3px;background:#eef3fc;
     color:#3a5f9e;font-size:11.5px;margin-left:6px}
@media(max-width:980px){.grid2{grid-template-columns:1fr}}
"""


def _fmt(v, nd=2, pct=False, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if pct:
        return f"{v * 100:+.{nd}f}%" if sign else f"{v * 100:.{nd}f}%"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _cls(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "mut"
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _shell(title: str, body: str, scripts: str, root: str = ".") -> str:
    """页面外壳：自包含（echarts 走同目录 assets，避免 888 份 1MB 内联）。"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="{root}/assets/echarts.min.js"></script>
<style>{CSS}</style>
</head>
<body>
{body}
{scripts}
</body></html>"""


def _topbar(n_factor: int, generated: str, active: str, root: str = ".",
            up: str = "..") -> str:
    def tab(href, label, key):
        on = " class='on'" if key == active else ""
        return f"<a href='{href}'{on}>{label}</a>"
    return (
        "<div class='topbar'><b>YuriQuant · 因子库</b>"
        + tab(f"{root}/index.html", "因子看板", "board")
        + tab(f"{up}/yq_core/backtest.html", "策略回测", "bt")
        + tab(f"{up}/yq_core/live.html", "模拟交易", "live")
        + "<span class='sp'></span>"
        + f"<span class='meta'>{n_factor} 个因子 · 生成 {generated}</span></div>"
    )


# ===========================================================================
# 列表页
# ===========================================================================
def _row_payload(reg: pd.DataFrame, summ: pd.DataFrame) -> list[dict]:
    rows = []
    for _, s in summ.iterrows():
        name = s["name"]
        src = str(s.get("source") or "")
        label = str(s.get("label") or name)
        if "#" in label:                     # Alpha101/158/191 的 label 带长前缀
            head = label.split("(")[0].strip()
            label = f"{head} #{label.split('#')[-1]}"
        if len(label) > 30:
            label = label[:29] + "…"
        fam = _logic_family(src, s.get("label") or name)
        rec = {"n": name, "cn": label, "fam": fam,
               "grp": SRC_GROUP.get(src, "其他"),
               "deg": bool(s.get("degenerate")) if isinstance(s.get("degenerate"), (bool, np.bool_)) else None,
               "uq": _num(s.get("unique_levels")),
               "srcn": SRC_SUB.get(src, src or "其他"),
               "ic1": s.get("ic_1y_ic_mean"), "ir1": s.get("ic_1y_ic_ir")}
        for w, _ in WINDOWS:
            for tag in ("q1_net", "q5_net", "q1_gross", "q5_gross"):
                pre = f"{tag}_{w}_"
                rec[f"{tag}_{w}"] = [
                    _num(s.get(pre + "excess_ann")), _num(s.get(pre + "turnover")),
                    _num(s.get(pre + "sharpe")), _num(s.get(pre + "cum")),
                ]
            rec[f"ic_{w}"] = [_num(s.get(f"ic_{w}_ic_mean")),
                              _num(s.get(f"ic_{w}_ic_ir")),
                              _num(s.get(f"ic_{w}_ic_gt"))]
        rows.append(rec)
    return rows


def _num(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 5) if np.isfinite(f) else None


def build_index(reg: pd.DataFrame, summ: pd.DataFrame, generated: str) -> Path:
    rows = _row_payload(reg, summ)
    fam_counts = pd.Series([r["fam"] for r in rows]).value_counts()

    fam_html = "".join(
        f"<label class='chk'><input type='checkbox' class='fam' value='{f}' checked>"
        f"{f}<span class='mut'> ({fam_counts.get(f, 0)})</span></label>" for f in LOGIC_FAMILIES)

    # 因子集：7 组 + 可展开子集。组勾选 = 全选该组子集；筛选仍按子集判定
    grp_cnt = pd.Series([r["grp"] for r in rows]).value_counts()
    src_counts = pd.Series([r["srcn"] for r in rows]).value_counts()
    src_html = ""
    for g in GROUP_ORDER:
        subs = [(k, v) for k, v in SRC_SUB.items() if SRC_GROUP.get(k) == g
                and src_counts.get(v, 0)]
        if not subs:
            continue
        sub_in = "".join(
            f"<label class='chk sub'><input type='checkbox' class='src' value='{v}' checked>"
            f"{v}<span class='mut'> ({src_counts.get(v, 0)})</span></label>" for _, v in subs)
        src_html += (
            f"<span class='gsrc'><label class='chk'><input type='checkbox' class='g'"
            f" data-g='{g}' checked>{g}<span class='mut'> ({grp_cnt.get(g, 0)})</span></label>"
            f"<a class='exp' data-g='{g}'>{len(subs)} 个子集</a>"
            f"<span class='subs' data-g='{g}'>{sub_in}</span></span>")

    win_html = "".join(
        f"<label class='chk'><input type='radio' name='win' value='{k}'"
        f"{' checked' if k == '3y' else ''}>{lab}</label>" for k, lab in WINDOWS)
    cost_html = ("<label class='chk'><input type='radio' name='cost' value='net' checked>"
                 "27bp（生产口径）</label>"
                 "<label class='chk'><input type='radio' name='cost' value='gross'>"
                 "无成本</label>")

    body = f"""
{_topbar(len(rows), generated, "board")}
<div class='wrap wide'>
  <div class='card'>
    <div class='card-h'>因子看板<span class='sp'></span>
      <span class='searchbox'><input id='q' placeholder='搜索因子名 / 中文名'></span></div>
    <div class='card-bd'>
      <div class='frow'><div class='lb'>分 类</div><div class='opts'>
        <label class='chk'><input type='checkbox' id='allfam' checked>全选</label>{fam_html}
      </div></div>
      <div class='frow'><div class='lb'>因子集</div><div class='opts'>{src_html}</div></div>
      <div class='frow'><div class='lb'>股票池</div><div class='opts'>
        <label class='chk'><input type='radio' checked>全 A（含退市回补）</label>
        <label class='chk dis'><input type='radio' disabled>沪深300</label>
        <label class='chk dis'><input type='radio' disabled>中证500</label>
        <label class='chk dis'><input type='radio' disabled>中证1000</label>
        <span class='mut' style='font-size:12px'>其余池未在本库覆盖，不做假选项</span>
      </div></div>
      <div class='frow'><div class='lb'>回溯周期</div><div class='opts'>{win_html}</div></div>
      <div class='frow'><div class='lb'>组合构建</div><div class='opts'>
        <label class='chk'><input type='radio' checked>纯多头组合（等权 · 月频）</label>
        <label class='chk dis'><input type='radio' disabled>多空组合 I</label>
        <label class='chk dis'><input type='radio' disabled>多空组合 II</label>
        <span class='mut' style='font-size:12px'>空头腿需借券费与保证金口径，本页不出</span>
      </div></div>
      <div class='frow'><div class='lb'>手续费<br>及滑点</div>
        <div class='opts'>{cost_html}</div></div>
      <div class='frow'><div class='lb'>过滤涨停<br>及停牌股</div><div class='opts'>
        <label class='chk dis'><input type='radio' checked>是</label>
        <span class='mut' style='font-size:12px'>已用 tradable_mask 固定过滤（不可关）</span>
      </div></div>
    </div>
  </div>

  <div class='card'>
    <div class='card-bd tight'><div class='tblwrap'>
      <table class='tbl' id='t'>
        <thead><tr>
          <th data-k='cn' style='width:27%'>因子名称</th>
          <th data-k='q1x'>最小分位<br>超额年化收益<span class='ar'>⇅</span></th>
          <th data-k='q5x'>最大分位<br>超额年化收益<span class='ar'>⇅</span></th>
          <th data-k='q1t'>最小分位<br>换手率(月)<span class='ar'>⇅</span></th>
          <th data-k='q5t'>最大分位<br>换手率(月)<span class='ar'>⇅</span></th>
          <th data-k='ic'>IC均值<span class='ar'>⇅</span></th>
          <th data-k='ir'>IR值<span class='ar'>⇅</span></th>
          <th data-k='q5s'>最大分位<br>夏普<span class='ar'>⇅</span></th>
        </tr></thead>
        <tbody id='tb'></tbody>
      </table>
    </div>
    <div class='foot'><span id='cnt'></span>
      <span>点击行进入因子详情 · 表头可排序</span><span class='sp' style='flex:1'></span>
      <span>口径：月频 / T+1 开盘 / 组内等权 / 基准=全A等权</span></div>
    </div>
  </div>

  <div class='card'><div class='card-h'>口径与边界</div>
    <div class='note'>
      1. <b>超额年化</b> = (1+分位年化)/(1+基准年化)−1，基准为<b>全A等权</b>（与因子同池）；换手率为<b>月均单边换手</b>（每次调仓单边换手均值；详情页给出年化 = ×12）。<br>
      2. 分位组 = 调仓日截面秩的 20% 切分：<b>最小分位数</b> = 最低 20%，<b>最大分位数</b> = 最高 20%，组内等权，月频调仓、<b>T+1 开盘成交</b>，已扣 27bp 生产口径成本（含 2023-08-28 印花税下调）。<br>
      3. <b>IC</b> = 截面 spearman(因子<sub>t</sub>, 收益<sub>t→t+1</sub>) 的时序均值；<b>IR</b> = IC均值 / IC标准差（未年化）。<br>
      4. 这是<b>因子横向比较</b>用的统一口径，不是定版改进结论——判改进仍须 <code>evaluation/caliber.py</code> + 21 相位 + 组合层检验。
      5. 全样本自 2019-01 起（面板与掩码对齐后的生产窗口），更早年份未纳入。<br>
      6. <b>分类按「因子在算什么」划分</b>（动量波动 / 均线通道 / 极值形态 / 盈余与成长…），
      不是按数据来源；`alpha360` 的 352 个是<b>原始 OHLCV 滞后序列</b>
      （`CLOSE_k = close.shift(k)/close`），本质是不同回看期的反转，单列「原始序列」
      以免与真正的 alpha 因子混计；`alpha158` 的 134 个实为 <b>35 个基名 × 5 窗口</b>。
    </div></div>
</div>"""

    js = """
const ROWS = __ROWS__;
const WIN = '3y';
let COST = 'net', SORT = {k: 'q5x', desc: true};

function val(r, key) {
  const w = window.__WIN || WIN;
  const pre = COST === 'net' ? '_net' : '_gross';
  switch (key) {
    case 'cn': return r.cn;
    case 'q1x': return (r['q1' + pre + '_' + w] || [])[0];
    case 'q5x': return (r['q5' + pre + '_' + w] || [])[0];
    case 'q1t': return (r['q1' + pre + '_' + w] || [])[1];
    case 'q5t': return (r['q5' + pre + '_' + w] || [])[1];
    case 'q5s': return (r['q5' + pre + '_' + w] || [])[2];
    case 'ic': return (r['ic_' + w] || [])[0];
    case 'ir': return (r['ic_' + w] || [])[1];
  }
  return null;
}
function pct(v) { return v == null ? '—' : (v > 0 ? '+' : '') + (v * 100).toFixed(2) + '%'; }
function cls(v) { return v == null ? 'mut' : (v > 0 ? 'pos' : (v < 0 ? 'neg' : '')); }
function num(v, nd) { return v == null ? '—' : v.toFixed(nd); }

function activeFams() {
  return [...document.querySelectorAll('.fam')].filter(e => e.checked).map(e => e.value);
}
function activeSrcs() {
  return [...document.querySelectorAll('.src')].filter(e => e.checked).map(e => e.value);
}

function render() {
  const fams = new Set(activeFams()), srcs = new Set(activeSrcs());
  const q = (document.getElementById('q').value || '').trim().toLowerCase();
  let rows = ROWS.filter(r => fams.has(r.fam) && srcs.has(r.srcn)
    && (!q || r.n.toLowerCase().includes(q) || r.cn.toLowerCase().includes(q)));
  const k = SORT.k, dir = SORT.desc ? -1 : 1;
  rows.sort((a, b) => {
    const va = val(a, k), vb = val(b, k);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === 'string') return va.localeCompare(vb) * dir;
    return (va - vb) * dir;
  });
  const tb = document.getElementById('tb');
  tb.innerHTML = rows.map(r => {
    const q1x = val(r, 'q1x'), q5x = val(r, 'q5x');
    const q1t = val(r, 'q1t'), q5t = val(r, 'q5t'), q5s = val(r, 'q5s');
    const ic = val(r, 'ic'), ir = val(r, 'ir');
    return `<tr data-n="${r.n}">
      <td class='nm'><div class='nm1'>${r.cn}${r.deg ? ' <span class="deg">分位不可分</span>' : ''}</div><div class='nm2'>${r.n}</div></td>
      <td class='${cls(q1x)}'>${pct(q1x)}</td>
      <td class='${cls(q5x)}'>${pct(q5x)}</td>
      <td>${q1t == null ? '—' : (q1t / 12 * 100).toFixed(2) + '%'}</td>
      <td>${q5t == null ? '—' : (q5t / 12 * 100).toFixed(2) + '%'}</td>
      <td class='${cls(ic)}'>${num(ic, 4)}</td>
      <td class='${cls(ir)}'>${num(ir, 3)}</td>
      <td class='${cls(q5s)}'>${num(q5s, 2)}</td>
    </tr>`;
  }).join('');
  document.getElementById('cnt').textContent = `共 ${rows.length} 个因子`;
  tb.querySelectorAll('tr').forEach(tr => {
    tr.onclick = () => { location.href = './f/' + encodeURIComponent(tr.dataset.n) + '.html'; };
  });
}

document.querySelectorAll('.fam').forEach(e => e.onchange = () => {
  document.getElementById('allfam').checked =
    [...document.querySelectorAll('.fam')].every(x => x.checked);
  render();
});
document.getElementById('allfam').onchange = e => {
  document.querySelectorAll('.fam').forEach(x => x.checked = e.target.checked); render();
};
function subsOf(g) {
  return [...document.querySelectorAll('.subs[data-g="' + g + '"] .src')];
}
function syncGroups() {
  document.querySelectorAll('.g').forEach(g => {
    const subs = subsOf(g.dataset.g), on = subs.filter(x => x.checked).length;
    g.checked = on > 0;
    g.indeterminate = on > 0 && on < subs.length;
  });
}
document.querySelectorAll('.g').forEach(g => g.onchange = () => {
  subsOf(g.dataset.g).forEach(x => x.checked = g.checked); render();
});
document.querySelectorAll('.src').forEach(e => e.onchange = () => { syncGroups(); render(); });
document.querySelectorAll('.exp').forEach(a => a.onclick = ev => {
  ev.preventDefault();
  const box = document.querySelector('.subs[data-g="' + a.dataset.g + '"]');
  box.classList.toggle('on');
  a.textContent = box.classList.contains('on')
    ? '收起' : box.querySelectorAll('.src').length + ' 个子集';
});
syncGroups();
document.querySelectorAll("input[name=win]").forEach(e => e.onchange = () => {
  window.__WIN = e.value; render();
});
document.querySelectorAll("input[name=cost]").forEach(e => e.onchange = () => {
  COST = e.value; render();
});
document.getElementById('q').oninput = render;
document.querySelectorAll('#t th').forEach(th => th.onclick = () => {
  const k = th.dataset.k;
  SORT = {k, desc: (SORT.k === k) ? !SORT.desc : true};
  document.querySelectorAll('#t th').forEach(x => x.classList.remove('sorted'));
  th.classList.add('sorted');
  render();
});
window.__WIN = WIN;
render();
"""
    js = js.replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
    path = OUT / "index.html"
    path.write_text(_shell("因子看板 · YuriQuant", body, f"<script>{js}</script>"),
                    encoding="utf-8")
    return path


# ===========================================================================
# 详情页
# ===========================================================================
def _deg_note(meta: dict) -> str:
    """退化因子（截面取值太少）的提示条；正常因子返回空串。"""
    if not meta.get("degenerate"):
        return ""
    uq = meta.get("unique_levels")
    return ("<div class='card'><div class='card-bd' style='color:#b03a2e'>"
            f"⚠️ 该因子面板每日截面仅 <b>{uq}</b> 种有效取值，无法切 20% 分位 ⇒ "
            "<b>只出 IC，不出分位业绩</b>（硬切会造出空仓组合，业绩数字无意义）。"
            "</div></div>")


def _weekly(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    g = df.resample("W-FRI")
    return g.last() if how == "last" else g.mean()


def build_detail(name: str, reg_row: pd.Series, meta: dict, nav: pd.DataFrame,
                 ind_names: dict) -> Path:
    q_cols = [c for c in ("nav_q1_net", "nav_q5_net", "nav_q1_gross", "nav_q5_gross")
              if c in nav.columns]
    if q_cols:
        weekly = _weekly(nav[q_cols]).reindex(columns=q_cols)
    else:                       # 退化因子只有 IC 列
        weekly = pd.DataFrame(index=_weekly(nav[["ic"]]).index, columns=q_cols)
    bench = None
    bench_nav = _bench_nav()
    if bench_nav is not None:
        bench = bench_nav.reindex(weekly.index.union(bench_nav.index)).ffill().reindex(weekly.index)

    ic = nav["ic"].dropna()
    # IC 的显著性一律用 NW-t（IC 序列有自相关，naive mean/std·√n 会高估 2~4 倍）
    for _w, _yrs in (("1y", 1), ("3y", 3), ("5y", 5), ("all", None)):
        _blk = (meta.get("ic") or {}).get(_w)
        if not _blk:
            continue
        _cut = ic.index[-1] - pd.DateOffset(years=_yrs) if _yrs else None
        _seg = (ic if _cut is None else ic.loc[ic.index >= _cut]).dropna()
        if len(_seg) > 30:
            try:
                _blk["ic_t_nw"], _, _ = nw_tstat(_seg.values)
            except Exception:                              # noqa: BLE001
                pass
    ic_w = ic.resample("W-FRI").mean()
    ma = ic.rolling(22, min_periods=10).mean().resample("W-FRI").last()

    col = lambda k: weekly[k] if k in weekly.columns else None   # noqa: E731
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
        "q1": _arr(col("nav_q1_net")), "q5": _arr(col("nav_q5_net")),
        "q1g": _arr(col("nav_q1_gross")), "q5g": _arr(col("nav_q5_gross")),
        "bench": _arr(bench) if bench is not None else None,
        "ic": {"dates": [d.strftime("%Y-%m-%d") for d in ic_w.index],
               "v": _arr(ic_w), "ma": _arr(ma)},
        "ind": {ind_names.get(k, k): round(v, 4)
                for k, v in (meta.get("industry_ic") or {}).items()},
        "meta": meta,
        "reg": {"family": reg_row.get("family"), "formula": reg_row.get("formula"),
                "source": str(reg_row.get("source", "")),
                "created": str(reg_row.get("created_at", ""))[:19],
                "coverage": _num(reg_row.get("coverage")),
                "label": reg_row.get("label"), "name": name,
                "note": reg_row.get("note")},
    }

    _src = str(reg_row.get("source", "")).split(":")[0]
    fam = _logic_family(_src, reg_row.get("label") or name)
    # 详情页 tag 只挂子集名（组=粗维度，且 sue_pledge 这类会跨逻辑族，挂组会自相矛盾）
    src_tag = SRC_SUB.get(_src, _src or "自定义")
    formula = str(reg_row.get("formula") or "—")
    note = str(reg_row.get("note") or "")
    cov = payload["reg"]["coverage"]
    created = payload["reg"]["created"] or "—"
    if created in ("nan", "NaT", "None"):
        created = "—"
    if note in ("nan", "NaT", "None"):
        note = ""

    deg_note = _deg_note(meta)
    win_html = "".join(
        f"<option value='{k}'{' selected' if k == '3y' else ''}>{lab}</option>"
        for k, lab in WINDOWS)

    body = f"""
{_topbar(0, "", "board", root="..", up="../..").replace("0 个因子", "").replace(" · 生成 ", "")}
<div class='wrap'>
  <h1 class='fname'>{html.escape(str(reg_row.get('label') or name))}
    <code>{html.escape(name)}</code>
    <a class='btn ghost' style='float:right;font-size:12.5px;padding:6px 16px'
       href='../index.html'>← 返回因子列表</a></h1>

  <div class='card'><div class='card-bd'>
    <div class='defs'>
      <div class='k'>因子类别</div><div>{html.escape(str(fam or '—'))}
        <span class='tag'>{html.escape(src_tag)}</span></div>
      <div class='k'>计算公式</div><div><code>{html.escape(formula)}</code></div>
      <div class='k'>更新时间</div><div>{html.escape(created)}（面板首末：{meta['start']} ~ {meta['end']}）</div>
      <div class='k'>数据处理</div><div>截面中位数去极值 → 行业+市值中性化 → zscore 标准化（因子库统一管线）</div>
      <div class='k'>默认参数</div><div>分位组 = 5（20% 切分）· 组内等权 · 月频调仓 · T+1 开盘成交 · 面板覆盖 {_fmt(cov, 1, True)}（registry 原值，分母含上市前空档）</div>
      <div class='k'>备注</div><div>{html.escape(note) if note else '—'}</div>
    </div>
  </div></div>

  <div class='card'><div class='card-bd'>
    <div class='ctrl'>
      <div class='it'>组合构建
        <select disabled><option>纯多头组合（等权）</option></select></div>
      <div class='it'>股票池
        <select disabled><option>全 A（含退市回补）</option></select></div>
      <div class='it'>回溯区间<select id='win'>{win_html}</select></div>
      <div class='it'>过滤涨停及停牌股
        <select disabled><option>是</option></select></div>
      <div class='it'>调仓周期
        <select disabled><option>1 个月</option></select></div>
      <div class='it'>手续费及滑点
        <select id='cost'>
          <option value='net'>27bp（生产口径）</option>
          <option value='gross'>无</option>
        </select></div>
      <button class='btn' id='go'>确定</button>
    </div>
  </div></div>

  {deg_note}
  <div class='card'><div class='card-bd'>
    <div class='ic3'>
      <div><div class='k'>IC 均值</div><div class='v' id='icm'>—</div></div>
      <div><div class='k'>|IC| &gt; 0.02 的比率</div><div class='v' id='icg'>—</div></div>
      <div><div class='k'>IR 值</div><div class='v' id='icr'>—</div></div>
      <div style='flex:1.4'><div class='k'>IC t 值 / 胜率</div>
        <div class='v' style='font-size:17px;line-height:2.1' id='ict'>—</div></div>
    </div>
  </div></div>

  <div class='card' id='perfCard'>
    <div class='card-h'>因子详情<span class='sp'></span>
      <span class='hint' id='winhint'></span></div>
    <div class='card-bd'>
      <table class='tbl mtbl'><thead><tr>
        <th></th><th class='num'>累计收益</th><th class='num'>年化收益</th>
        <th class='num'>超额年化收益</th><th class='num'>基准年化收益</th>
        <th class='num'>最大回撤</th><th class='num'>夏普比率</th><th class='num'>换手率(年化)</th>
      </tr></thead><tbody id='mt'></tbody></table>
      <div class='toggles' style='margin:14px 0 4px'>
        <span>坐标</span>
        <label><input type='radio' name='ax' value='v' checked>普通轴</label>
        <label><input type='radio' name='ax' value='log'>对数轴</label>
        <span class='mut' style='font-size:12px'>全分位视图未出（本页只算 20% 两端分位）</span>
      </div>
      <div id='navChart' class='chart'></div>
    </div>
  </div>

  <div class='grid2'>
    <div class='card'><div class='card-h'>IC 时序图<span class='sp'></span>
      <span class='hint'>周度均值 · 22 日均线</span></div>
      <div class='card-bd'><div id='icChart' class='chart-sm'></div></div></div>
    <div class='card'><div class='card-h'>行业 IC<span class='sp'></span>
      <span class='hint'>行业内 spearman IC 均值（月抽样 · 按 |IC| 取前 18）</span></div>
      <div class='card-bd'><div id='indChart' class='chart-sm'></div></div></div>
  </div>

  <div class='card' id='toCard'><div class='card-h'>换手率（每次调仓单边）<span class='sp'></span>
    <span class='hint'>两端分位 · 月频</span></div>
    <div class='card-bd'><div id='toChart' class='chart-sm'></div></div></div>

  <div class='card'><div class='note'>
    <b>口径</b>：月频调仓 · T+1 开盘成交 · 组内等权 · 基准=全A等权 ·
    「27bp」= 佣金万1双边 + 时变印花税（2023-08-28 起万5）+ 20bp 滑点。<br>
    <b>边界</b>：分位组合是<b>因子横向比较</b>的展示口径，未做相位平均、未做容量/冲击压力，
    <b>不构成改进结论</b>；判改进请走 <code>evaluation/caliber.py</code>。
    曲线为周频抽样绘制（指标用日频算）。
  </div></div>
</div>"""

    js = """
const F = __PAY__;
const WIN = {'1y': 1, '3y': 3, '5y': 5, 'all': 0};
const charts = [];
function mk(id, opt) {
  const el = document.getElementById(id);
  if (!el) return null;
  const c = echarts.init(el, null, {renderer: 'canvas'});
  c.setOption(opt); charts.push(c); return c;
}
function cut(arr, nYears, dates) {
  if (!nYears) return arr;
  const end = new Date(dates[dates.length - 1]);
  const cutd = new Date(end); cutd.setFullYear(end.getFullYear() - nYears);
  return dates.map((d, i) => new Date(d) >= cutd ? arr[i] : null);
}
function firstIdx(dates, nYears) {
  if (!nYears) return 0;
  const end = new Date(dates[dates.length - 1]);
  const cutd = new Date(end); cutd.setFullYear(end.getFullYear() - nYears);
  for (let i = 0; i < dates.length; i++) if (new Date(dates[i]) >= cutd) return i;
  return 0;
}
function pct(v, nd, sign) {
  if (v == null) return '—';
  return (sign && v > 0 ? '+' : '') + (v * 100).toFixed(nd == null ? 2 : nd) + '%';
}
function cls(v) { return v == null ? 'mut' : (v > 0 ? 'pos' : (v < 0 ? 'neg' : '')); }

let curWin = '3y', curCost = 'net', curAxis = 'v';
let navChart = null, icChart = null, indChart = null, toChart = null;

function metrics(which) {
  const key = which + '_' + curCost;
  return (F.meta.metrics[key] || {})[curWin] || null;
}

function drawTable() {
  const q1 = metrics('q1'), q5 = metrics('q5');
  const row = (nm, m) => m ? `<tr><td>${nm}</td>
    <td class='${cls(m.cum)}'>${pct(m.cum)}</td>
    <td class='${cls(m.ann)}'>${pct(m.ann)}</td>
    <td class='${cls(m.excess_ann)}'>${pct(m.excess_ann)}</td>
    <td class='${cls(m.bench_ann)}'>${pct(m.bench_ann)}</td>
    <td class='neg'>${pct(m.mdd)}</td>
    <td class='${cls(m.sharpe)}'>${m.sharpe == null ? '—' : m.sharpe.toFixed(2)}</td>
    <td>${m.turnover == null ? '—' : (m.turnover * 100).toFixed(2) + '%'}</td></tr>`
    : `<tr><td>${nm}</td><td colspan='7' class='mut'>样本不足</td></tr>`;
  document.getElementById('mt').innerHTML = row('最小分位数', q1) + row('最大分位数', q5);
  const lab = {'1y': '近 1 年', '3y': '近 3 年', '5y': '近 5 年', 'all': '全样本'}[curWin];
  document.getElementById('winhint').textContent = lab + ' · ' + (curCost === 'net' ? '扣 27bp' : '无成本');
}

// IC 三格单独成函数：分位不可分（degenerate）的因子不画业绩表，
// 但 IC 仍要出数，所以不能藏在 drawTable 里（2026-11-17 踩过：整块显示「—」）
function drawIc3() {
  const ic = (F.meta.ic || {})[curWin];
  if (!ic) return;
  const setv = (id, txt) => {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  };
  setv('icm', ic.ic_mean == null ? '—' : ic.ic_mean.toFixed(4));
  setv('icg', ic.ic_gt == null ? '—' : (ic.ic_gt * 100).toFixed(1) + '%');
  setv('icr', ic.ic_ir == null ? '—' : ic.ic_ir.toFixed(3));
  setv('ict', (ic.ic_t_nw == null ? '—' : ic.ic_t_nw.toFixed(2)) + ' ／ '
              + (ic.ic_win == null ? '—' : (ic.ic_win * 100).toFixed(1) + '%'));
}

function drawNav() {
  const D0 = F.dates, i0 = firstIdx(D0, WIN[curWin]), D = D0.slice(i0);
  const base = a => { const v = a[i0] || 1;
    return a.slice(i0).map(x => x == null ? null : x / v); };
  const q1 = base(curCost === 'net' ? F.q1 : F.q1g);
  const q5 = base(curCost === 'net' ? F.q5 : F.q5g);
  const bm = F.bench ? base(F.bench) : null;
  const series = [
    {name: '全A等权', type: 'line', data: bm, showSymbol: false, sampling: 'lttb',
      lineStyle: {width: 1.4, color: '#9aa3b5', type: 'dashed'}, itemStyle: {color: '#9aa3b5'}},
    {name: '最小分位数', type: 'line', data: q1, showSymbol: false, sampling: 'lttb',
      lineStyle: {width: 1.8, color: '#2f6fdb'}, itemStyle: {color: '#2f6fdb'}},
    {name: '最大分位数', type: 'line', data: q5, showSymbol: false, sampling: 'lttb',
      lineStyle: {width: 1.8, color: '#e07b39'}, itemStyle: {color: '#e07b39'}},
  ];
  const opt = {
    animation: false, grid: {left: 60, right: 24, top: 34, bottom: 48},
    legend: {top: 2, left: 0, itemWidth: 13, itemHeight: 9,
             textStyle: {fontSize: 12, color: '#5d6474'}},
    tooltip: {trigger: 'axis', valueFormatter: v => v == null ? '—' : v.toFixed(3) + '×'},
    xAxis: {type: 'category', data: D, axisLabel: {fontSize: 11, hideOverlap: true},
            axisLine: {lineStyle: {color: '#e4e7ec'}}},
    yAxis: {type: curAxis === 'log' ? 'log' : 'value', scale: true, min: curAxis === 'log' ? 0.5 : null,
            axisLabel: {fontSize: 11, formatter: v => curAxis === 'log' ? v.toFixed(2) : v.toFixed(2)},
            splitLine: {lineStyle: {color: '#f2f4f7'}}},
    series,
  };
  if (!navChart) navChart = mk('navChart', opt);
  else navChart.setOption(opt, true);
}

function drawIc() {
  const N0 = F.ic, i0 = firstIdx(N0.dates, WIN[curWin]);
  const N = {dates: N0.dates.slice(i0), v: N0.v.slice(i0), ma: N0.ma.slice(i0)};
  const v = N.v, ma = N.ma;
  const opt = {
    animation: false, grid: {left: 48, right: 18, top: 28, bottom: 40},
    legend: {top: 0, left: 0, itemWidth: 10, itemHeight: 8, textStyle: {fontSize: 11.5}},
    tooltip: {trigger: 'axis', valueFormatter: x => x == null ? '—' : x.toFixed(4)},
    xAxis: {type: 'category', data: N.dates, axisLabel: {fontSize: 10.5, hideOverlap: true}},
    yAxis: {type: 'value', scale: true, axisLabel: {fontSize: 10.5},
            splitLine: {lineStyle: {color: '#f2f4f7'}}},
    series: [
      {name: 'IC', type: 'bar', data: v, itemStyle: {color: '#7fb0e8'}, barMaxWidth: 4},
      {name: '22 日均线', type: 'line', data: ma, showSymbol: false,
        lineStyle: {width: 1.5, color: '#d9363e'}, itemStyle: {color: '#d9363e'}},
    ],
  };
  if (!icChart) icChart = mk('icChart', opt); else icChart.setOption(opt, true);
}

function drawInd() {
  const all = Object.entries(F.ind || {});
  // 31 个申万一级行业塞进 360px 会行高 11px、相邻数值标签压在一起：
  // 只画 |IC| 最大的 18 个（卡片头 hint 已注明），不足以判行业轮动，够看暴露方向
  const top = all.slice().sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 18);
  const entries = top.sort((a, b) => a[1] - b[1]);
  const m = Math.max(0.02, ...entries.map(e => Math.abs(e[1])));
  const opt = {
    animation: false, grid: {left: 100, right: 46, top: 10, bottom: 22},
    tooltip: {trigger: 'axis', axisPointer: {type: 'shadow'},
               valueFormatter: v => v == null ? '—' : v.toFixed(4)},
    xAxis: {type: 'value', min: -m, max: m, axisLabel: {fontSize: 10.5},
            splitLine: {lineStyle: {color: '#f2f4f7'}}},
    yAxis: {type: 'category', data: entries.map(e => e[0]),
            axisLabel: {fontSize: 10.5, margin: 8, color: '#4b5563'},
            axisTick: {show: false}, axisLine: {show: false}},
    series: [{type: 'bar', data: entries.map(e => ({
      value: e[1],
      itemStyle: {color: e[1] >= 0 ? '#d9363e' : '#12a05c'},
      // 负值标签放左侧，否则全挤在 0 轴附近互相重叠
      label: {position: e[1] >= 0 ? 'right' : 'left',
              formatter: () => e[1].toFixed(3)}})), barMaxWidth: 11,
      label: {show: true, fontSize: 9.5, color: '#6b7280', distance: 4}}],
  };
  indChart = mk('indChart', opt);
}

function drawTo() {
  const T = F.meta.turnover || {};
  const opt = {
    animation: false, grid: {left: 56, right: 20, top: 28, bottom: 40},
    legend: {top: 0, left: 0, itemWidth: 12, itemHeight: 8, textStyle: {fontSize: 11.5}},
    tooltip: {trigger: 'axis', valueFormatter: v => v == null ? '—' : (v * 100).toFixed(1) + '%'},
    xAxis: {type: 'category', data: (T.q5 || {}).dates || [],
            axisLabel: {fontSize: 10.5, hideOverlap: true}},
    yAxis: {type: 'value', axisLabel: {fontSize: 10.5, formatter: v => (v * 100).toFixed(0) + '%'},
            splitLine: {lineStyle: {color: '#f2f4f7'}}},
    series: [
      {name: '最小分位数', type: 'line', data: (T.q1 || {}).values, showSymbol: false,
        lineStyle: {width: 1.2, color: '#2f6fdb'}, itemStyle: {color: '#2f6fdb'},
        areaStyle: {color: 'rgba(47,111,219,.10)'}},
      {name: '最大分位数', type: 'line', data: (T.q5 || {}).values, showSymbol: false,
        lineStyle: {width: 1.2, color: '#e07b39'}, itemStyle: {color: '#e07b39'}},
    ],
  };
  toChart = mk('toChart', opt);
}

function refresh() { drawTable(); drawIc3(); drawNav(); drawIc(); }
document.getElementById('go').onclick = () => {
  curWin = document.getElementById('win').value;
  curCost = document.getElementById('cost').value;
  refresh();
};
document.getElementById('win').onchange = () => {
  curWin = document.getElementById('win').value; refresh();
};
document.getElementById('cost').onchange = () => {
  curCost = document.getElementById('cost').value; refresh();
};
document.querySelectorAll("input[name=ax]").forEach(e => e.onchange = () => {
  curAxis = e.value; drawNav();
});
if (F.meta.degenerate) {
  // 分位不可分：藏掉业绩表 / 净值 / 换手，只留 IC 与行业 IC
  ['perfCard', 'toCard'].forEach(id => {
    const el = document.getElementById(id); if (el) el.style.display = 'none';
  });
} else {
  drawTable(); drawNav(); drawTo();
}
drawIc3(); drawIc(); drawInd();
window.addEventListener('resize', () => charts.forEach(c => c.resize()));
"""
    js = js.replace("__PAY__", json.dumps(payload, ensure_ascii=False))

    path = OUT / "f" / f"{name}.html"
    path.write_text(_shell(f"{reg_row.get('label') or name} · 因子详情", body,
                           f"<script>{js}</script>", root=".."), encoding="utf-8")
    return path


def _arr(s: pd.Series | None):
    if s is None:
        return None
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 5) for v in s.values]


_BENCH: pd.Series | None = None


def _bench_nav() -> pd.Series | None:
    global _BENCH
    if _BENCH is None:
        p = ROOT / "reports" / "alla_rolling" / "_base" / "bench_eqw.parquet"
        if not p.exists():
            return None
        r = pd.read_parquet(p)["ret"].replace([np.inf, -np.inf], np.nan).dropna()
        r = r.loc[r.index >= pd.Timestamp("2019-01-01")]
        _BENCH = (1 + r).cumprod()
    return _BENCH


def _row_from_meta(meta: dict, reg_row: pd.Series) -> dict:
    """与 factor_board_data._summary_row 同构（这里独立实现，避免渲染端反向依赖计算端）。"""
    row = {"name": meta["name"], "family": reg_row.get("family"),
           "degenerate": meta.get("degenerate"),
           "unique_levels": meta.get("unique_levels"),
           "source": str(reg_row.get("source", "")).split(":")[0],
           "label": reg_row.get("label"), "formula": reg_row.get("formula"),
           "coverage": reg_row.get("coverage"), "n_days": meta.get("n_days"),
           "start": meta.get("start"), "end": meta.get("end")}
    for key, block in (meta.get("metrics") or {}).items():
        for w, m in block.items():
            for k, v in m.items():
                row[f"{key}_{w}_{k}"] = v
    for w, m in (meta.get("ic") or {}).items():
        for k, v in m.items():
            row[f"ic_{w}_{k}"] = v
    vals = list((meta.get("industry_ic") or {}).values())
    if vals:
        row["ind_ic_max"] = float(np.max(vals))
        row["ind_ic_min"] = float(np.min(vals))
    return row


def _rebuild_summary(cache: Path, reg: pd.DataFrame) -> pd.DataFrame:
    """从 cache/*.json 重建 _summary.csv（批跑中断/分段跑时用）。"""
    rows = []
    for j in sorted(cache.glob("*.json")):
        name = j.stem
        if name not in reg.index:
            continue
        try:
            meta = json.loads(j.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(_row_from_meta(meta, reg.loc[name]))
    df = pd.DataFrame(rows)
    df.to_csv(cache / "_summary.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="因子看板：列表页 + 因子详情页")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--top", type=int, default=None, help="只为前 N 个因子生成详情页")
    ap.add_argument("--index-only", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)
    (out / "f").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    if ECHARTS.exists():
        shutil.copy2(ECHARTS, out / "assets" / "echarts.min.js")

    reg = pd.read_csv(REGISTRY).set_index("name", drop=False)
    n_json = len(list(cache.glob("*.json")))
    summ = (pd.read_csv(cache / "_summary.csv") if (cache / "_summary.csv").exists()
            else pd.DataFrame())
    # 一律以 cache/*.json 为准重建横向总表：csv 可能是上一版脚本写的，
    # 缺列会让列表页静默少显示（2026-11-17 踩过：degenerate 列没进表）
    if n_json:
        summ = _rebuild_summary(cache, reg)
        print(f"  汇总由缓存重建：{len(summ)} 行（json {n_json} 个）")
    ind_names = _industry_names()

    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    idx = build_index(reg, summ, generated)
    print(f"[列表] {idx}  {idx.stat().st_size / 1e6:.2f} MB · {len(summ)} 个因子")
    if args.index_only:
        return

    names = list(summ.sort_values("q5_net_all_sharpe", ascending=False)["name"])
    if args.top:
        names = names[: args.top]
    wrote = 0
    for n in names:
        p, j = cache / f"{n}.parquet", cache / f"{n}.json"
        if not p.exists() or not j.exists() or n not in reg.index:
            continue
        build_detail(n, reg.loc[n], json.loads(j.read_text(encoding="utf-8")),
                     pd.read_parquet(p), ind_names)
        wrote += 1
    n_html = len(list((out / "f").glob("*.html")))
    print(f"[详情] 本次写入 {wrote} 页 → {out / 'f'}（共 {n_html} 个文件）")


if __name__ == "__main__":
    main()
