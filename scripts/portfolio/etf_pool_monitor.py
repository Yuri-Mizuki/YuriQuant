"""池内环境监测：策略「机制条件」是否还成立的唯一事前指标（2026-10-30）。

为什么需要它
------------
ETF 轮动三锋的收益结构已被量化（`reports/etf_contest5/README.md` §4.5–§4.6）：

    HRP 收益 = 单押（约 80% 权重）× 池内右尾

它不是「稳定 alpha」，是**条件性凸性**：只有当池里存在 1–2 只「当期动量最强、且与
其它腿相关最低、随后大幅上涨」的标的时，放大器才存在。同一套代码同一信号：

| 池 / 窗口          | 平均两两相关 | 动量离散度 | 放大器 HRP−等权 |
|-------------------|------------|-----------|----------------|
| 24 只 2014–2019   | **0.452**  | **0.151** | **−0.97 ~ −3.89pp** |
| 24 只 2020–2026   | 0.346      | 0.165     | +3.40 ~ +3.77pp |
| 45 只 2020–2026   | 0.355      | **0.204** | **+20.08pp**   |

⇒ 两个条件同时恶化时放大器归零。这两个数就是**失效前兆**，而它们可以每天算出来。

阈值依据（不是拍脑袋）
----------------------
WARN / ALERT 都取自长样本实测失效区（0.452 / 0.151），再留缓冲：

- `avg_corr`：基线 0.355 → WARN 0.40 → **ALERT 0.45（已进入长样本失效区）**
- `mom_disp`：基线 0.204 → WARN 0.17 → **ALERT 0.15（同上）**
- `max_weight`：基线中位 0.807 → **ALERT < 0.50**（HRP 不再押重 ⇒ 池里没有右尾可押）

三者是「或」关系告警、但**结论按同时恶化的个数读**：1 项 = 观察，≥2 项 = 机制受损。

用法::

    python -m scripts.portfolio.etf_pool_monitor                  # 最近 60 日截面
    python -m scripts.portfolio.etf_pool_monitor --asof 2026-10-29

输出：控制台表 + `reports/etf_contest5/pool_monitor.csv`（**默认保留历史**、只更新当日行；
`--overwrite` 才清空重建。监测的价值在趋势，误覆盖一次序列就毁了）。
只读，不改任何参数、不产生交易信号。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.cache import DataCache
from data.datasource import create_datasource
from data.etf_universe import ETF_CANDIDATES

#: 长样本实测「失效区」值（README §4.6.6）——阈值的唯一来源，改动须同步文档。
FAIL_ZONE_CORR = 0.452
FAIL_ZONE_DISP = 0.151
#: 当前有效区基线（45 只 2020–2026 生产池的**时间中位**，README §4.6.6）。
#: 🚨 基线是 7 年时间中位、当前值是最近 60 日截面 ⇒ 两者可比但不同口径，
#: 差 ±0.05 属正常波动，不要因为「对不上 0.355」就认为算错了。
BASE_CORR = 0.355
BASE_DISP = 0.204
BASE_MAXW = 0.807

#: 比赛屏蔽的 5 只：**从 spec 导入，不在此重列**（曾凭记忆重列写错 5 个代码）
from scripts.portfolio.etf_contest_spec import EXCLUDED_5  # noqa: E402

THRESH = {
    "avg_corr": {"warn": 0.40, "alert": FAIL_ZONE_CORR, "higher_is_worse": True,
                 "base": BASE_CORR, "fmt": "{:.3f}"},
    "mom_disp": {"warn": 0.17, "alert": FAIL_ZONE_DISP, "higher_is_worse": False,
                 "base": BASE_DISP, "fmt": "{:.3f}"},
    "max_weight": {"warn": 0.70, "alert": 0.50, "higher_is_worse": False,
                   "base": BASE_MAXW, "fmt": "{:.1%}"},
}

OUT_CSV = Path("reports/etf_contest5/pool_monitor.csv")
SIGNAL_DIR = Path("reports/etf_champ")


def load_pool_close(cache: DataCache, upto: str | None) -> pd.DataFrame:
    """生产池后复权收盘价（与 `etf_long_sample`/回测同源：close × backward_factor）。

    剔除 `EXCLUDED_5`（比赛屏蔽）⇒ 45 只，与 `FINAL` 的宇宙一致。
    """
    raw = cache.read_daily("etf")
    cl = raw["close"].unstack("code").sort_index()
    keep = [c for c in ETF_CANDIDATES if c in cl.columns and c not in EXCLUDED_5]
    cl = cl[keep]
    if upto:
        cl = cl.loc[cl.index <= pd.Timestamp(upto)]
    bf = cache.get_backward_factor(list(cl.columns))
    return cl * bf.reindex(index=cl.index, columns=cl.columns)


def pool_stats(close: pd.DataFrame, lookback: int = 60) -> dict:
    """截面环境：两两相关、有效只数、动量离散度、波动中位。口径同 §4.6.6。"""
    sub = close.tail(lookback + 1)
    ret = sub.pct_change(fill_method=None).dropna(how="all")
    # 只用完整覆盖的列，否则 corr 会把 NaN 当 0 摊薄
    ret = ret.loc[:, ret.notna().sum() >= lookback * 0.8]
    if ret.shape[1] < 3:
        raise SystemExit(f"有效标的不足（{ret.shape[1]} 只），无法计算池内指标")
    c = ret.corr()
    n = c.shape[0]
    avg_corr = float((c.to_numpy().sum() - n) / (n * (n - 1)))
    vol = ret.std() * np.sqrt(252.0)
    # 12-1 动量截面离散度（与策略信号同定义：21 日前/252 日前 - 1）
    px = close[ret.columns]
    mom = (px.shift(21) / px.shift(252) - 1.0).iloc[-1].dropna()
    return {
        "n_codes": int(n),
        "avg_corr": avg_corr,
        "n_eff": float(n / (1 + (n - 1) * avg_corr)),
        "mom_disp": float(mom.std()),
        "vol_med": float(vol.median()),
    }


def read_live_max_weight(profile: str = "contest3") -> tuple[float, str]:
    """从最近一次信号产物读单只最大权重；档案不符或缺失返回 (nan, 原因)。

    🚨 必须校验 `meta.json` 的 profile：`reports/etf_champ/` 是多档案共用目录，
    最后一次跑的是 `champ`（5 只 inverse_vol，最大权重 22%）时，直接读会给出
    「单押消失」的**假告警**——实测踩过一次。只认 profile 匹配的那份。
    🚨 只读 `signals.csv` 的 weight 列，不重算（重算会与生产信号板不一致）。
    """
    sig, meta_p = SIGNAL_DIR / "signals.csv", SIGNAL_DIR / "meta.json"
    if not sig.exists():
        return float("nan"), "无 signals.csv（还没跑过信号板）"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 坏 meta 不应中断监测
            meta = {}
        got = str(meta.get("profile", "")).strip()
        if got and got != profile:
            return float("nan"), f"signals.csv 属 `{got}` 档案（非 `{profile}`）⇒ 跳过"
    try:
        df = pd.read_csv(sig, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001 - 读失败不应中断监测
        return float("nan"), f"读失败 {type(exc).__name__}"
    if "weight" not in df.columns or not len(df):
        return float("nan"), "无 weight 列"
    w = pd.to_numeric(df["weight"].astype(str).str.rstrip("%"), errors="coerce") / 100.0
    if w.isna().all():
        return float("nan"), "weight 全 NaN"
    return float(w.max()), f"signals.csv ({len(df)} 行, profile={profile})"


def judge(key: str, value: float) -> str:
    """返回 OK / WARN / ALERT / n/a。"""
    if value is None or not np.isfinite(value):
        return "n/a"
    t = THRESH[key]
    if t["higher_is_worse"]:
        if value >= t["alert"]:
            return "ALERT"
        return "WARN" if value >= t["warn"] else "OK"
    if value <= t["alert"]:
        return "ALERT"
    return "WARN" if value <= t["warn"] else "OK"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="池内环境监测（机制条件是否还成立）")
    ap.add_argument("--asof", default=None, help="截止日 YYYY-MM-DD（复盘用）；默认数据末日")
    ap.add_argument("--lookback", type=int, default=60, help="滚动窗口交易日数")
    ap.add_argument("--profile", default="contest3",
                    help="读哪份信号产物的权重（须与 meta.json 的 profile 一致）")
    ap.add_argument("--overwrite", action="store_true",
                    help="清空 pool_monitor.csv 重建（默认保留历史、只更新当日行）")
    ap.add_argument("--csv", default=str(OUT_CSV))
    args = ap.parse_args(argv)

    cache = DataCache(create_datasource())
    close = load_pool_close(cache, args.asof)
    st = pool_stats(close, args.lookback)
    maxw, maxw_src = read_live_max_weight(args.profile)
    st["max_weight"] = maxw

    asof = close.index[-1].date()
    rows = []
    for key in ("avg_corr", "mom_disp", "max_weight"):
        t = THRESH[key]
        v = st.get(key)
        rows.append({
            "指标": key,
            "当前": "n/a" if not np.isfinite(v) else t["fmt"].format(v),
            "有效基线": t["fmt"].format(t["base"]),
            "WARN": t["fmt"].format(t["warn"]),
            "ALERT": t["fmt"].format(t["alert"]),
            "状态": judge(key, v),
        })
    tbl = pd.DataFrame(rows).set_index("指标")

    n_bad = sum(1 for r in rows if r["状态"] in ("WARN", "ALERT"))
    n_alert = sum(1 for r in rows if r["状态"] == "ALERT")
    if n_alert >= 2 or (n_alert >= 1 and n_bad >= 2):
        verdict = "🚨 机制受损（≥2 项进入失效区）⇒ 按手册降级处置"
    elif n_bad >= 2:
        verdict = "⚠️ 机制转弱（≥2 项预警）⇒ 观察，不加仓"
    elif n_bad == 1:
        verdict = "· 单项预警 ⇒ 仅记录"
    else:
        verdict = "✅ 机制条件正常"

    print(f"== 池内环境监测 · asof {asof} · 窗口 {args.lookback} 日 ==")
    print(f"池 {st['n_codes']} 只 · 有效独立只数 {st['n_eff']:.2f} · "
          f"波动中位 {st['vol_med']:.1%}")
    print(f"权重来源：{maxw_src}")
    print()
    print(tbl.to_string())
    print()
    print("阈值来源：长样本 2014–2019 实测失效区（avg_corr 0.452 / mom_disp 0.151，"
          "该区放大器 −0.97~−3.89pp），见 README §4.6.6。")
    print(f"判定：{verdict}")

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = pd.DataFrame([{
        "date": str(asof), "n_codes": st["n_codes"], "avg_corr": round(st["avg_corr"], 4),
        "n_eff": round(st["n_eff"], 3), "mom_disp": round(st["mom_disp"], 4),
        "vol_med": round(st["vol_med"], 4),
        "max_weight": (round(maxw, 4) if np.isfinite(maxw) else np.nan),
        "n_warn": n_bad, "n_alert": n_alert, "verdict": verdict.split("（")[0],
    }])
    # 🚨 默认就合并历史：监测的价值在趋势，误覆盖一次就把序列毁了。
    # 同日重跑只替换当日行，不产生重复。`--overwrite` 才清空重来。
    if out.exists() and not args.overwrite:
        try:
            old = pd.read_csv(out, encoding="utf-8-sig")
            rec = pd.concat(
                [old[old["date"].astype(str) != str(asof)], rec], ignore_index=True)
        except Exception as exc:  # noqa: BLE001 - 坏历史不应丢当日结果
            print(f"⚠️ 历史 {out} 读取失败（{type(exc).__name__}），保留原文件并写入新文件")
            out = out.with_name(out.stem + f"_{asof}.csv")
    rec.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n→ {out}（{len(rec)} 行历史）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
