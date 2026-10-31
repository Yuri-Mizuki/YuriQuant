"""三锋 HRP「单押 80%」溯源诊断（2026-10-31）
================================================

由莉酱的质疑：

> HRP 的核心逻辑是按风险（方差）反比分配，理论上三个资产中某一个拿到 80% 权重，
> 需要它与其他两个资产高度不相关甚至负相关，但我们的三个资产挺相关的呀。

## 这个直觉对「逆方差」成立，对 HRP 在 n=3 时**正好反过来**

n=3 的 HRP 不是全局逆方差，而是**先聚成 2+1 再二分**。设孤立腿 a 与配对腿 (b,c)：

    w_a = v_bc / (v_a + v_bc),  v_bc = σ_b² + σ_c² + 2ρ·σ_b·σ_c

等波动（σ_a=σ_b=σ_c=σ）时化简为

    w_a = (2 + 2ρ) / (3 + 2ρ)

ρ=0 → 66.7%，ρ=0.5 → 75%，ρ=0.9 → **79.2%**，ρ→1 → 80%。

⇒ **相关越高，孤立腿拿得越多**（配对簇的方差被 2ρσσ 抬高，按逆方差它就该少拿）。
`w_a=80%` 恰是「三者高度相关」的推论，不是「不相关」的推论。80% 也是 n=3 HRP 的
**理论上限**（ρ→1）。真正需要低相关才能出 80% 的是 `inv_var`（w ∝ 1/σ²）。

## 三个模式

1. ``--mode theory``：合成协方差上扫 ρ 与波动比，把 `hrp` / `inv_var` / `erc`
   三条曲线并排 —— 证明上面的方向。
2. ``--mode real``：跑定版 contest3（T+1 open / PIT 修复后），逐调仓日录下
   实际选中的三只的 60 日协方差、HRP 权重、风险贡献，回答「我们的 80% 里
   有多少来自波动差、多少来自相关」。
3. ``--mode begin``：45 只池逐只最早可用行情日 —— 回答「回测从 2020 开始
   是不是因为 ETF 最早只有这个时间」。

用法::

    python -m scripts.portfolio.etf_hrp_concentration_diag --mode theory,real,begin

产物：``reports/etf_contest5/hrp_concentration_diag.md``（stdout 同内容）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.portfolio.etf_contest_spec import FINAL, POOL  # noqa: E402
from scripts.portfolio.etf_weight_probe import load_env  # noqa: E402
from strategy.etf_rotation import (  # noqa: E402
    EtfRotationAdvanced,
    _erc_weights,
    _hrp_weights,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = ROOT / "reports/etf_contest5/hrp_concentration_diag.md"


# ---------------------------------------------------------------- 合成协方差
def cov3(sd: tuple[float, float, float], rho: float = 0.9,
         rho_bc: float | None = None) -> np.ndarray:
    """三资产协方差：a 与 (b,c) 相关 ``rho``，b 与 c 相关 ``rho_bc``（默认同 rho）。"""
    r = np.full((3, 3), float(rho))
    r[1, 2] = r[2, 1] = float(rho if rho_bc is None else rho_bc)
    np.fill_diagonal(r, 1.0)
    s = np.asarray(sd, float)
    return r * np.outer(s, s)


def _w3(cov: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "hrp": _hrp_weights(cov),
        "inv_var": (lambda v: v / v.sum())(1.0 / np.diag(cov)),
        "erc": _erc_weights(cov),
    }


def mode_theory(lines: list[str]) -> pd.DataFrame:
    rows = []
    # (A) 对称：三者两两相关 ρ。等波动 ⇒ 选谁当孤立腿都一样，权重只由 ρ 决定。
    for rho in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        w = _w3(cov3((1.0, 1.0, 1.0), rho))
        rows.append({"block": "A 对称·扫 ρ", "axis": f"ρ={rho:.2f}", "vol_a": 1.0,
                     **{k: float(v.max()) for k, v in w.items()}})
    # (B) 扫波动比：必须给 a 一个**真正的孤立腿**身份（ρ_a·=0.5 < ρ_bc=0.9），
    # 否则对称结构下"谁是孤立腿"由 linkage 的平局决定，扫波动比没有意义。
    for ratio in (1.0, 1.5, 2.0, 2.83, 4.0, 8.0):
        w = _w3(cov3((1.0 / ratio, 1.0, 1.0), 0.5, rho_bc=0.90))
        rows.append({"block": "B 孤立腿·扫 σ 比", "axis": f"σ_a/σ_others={1 / ratio:.2f}",
                     "vol_a": 1.0 / ratio,
                     **{k: float(v.max()) for k, v in w.items()}})
    # (C) 固定三者波动相同，只扫「配对腿之间的相关」——这才是 HRP 的开关。
    for rho_bc in (0.0, 0.3, 0.6, 0.8, 0.9, 0.95):
        w = _w3(cov3((1.0, 1.0, 1.0), 0.3, rho_bc=rho_bc))
        rows.append({"block": "C 扫配对相关 ρ_bc", "axis": f"ρ_bc={rho_bc:.2f}",
                     "vol_a": 1.0, **{k: float(v.max()) for k, v in w.items()}})
    df = pd.DataFrame(rows)

    lines += ["## 一、理论：n=3 HRP 的 80% 从哪来", "",
              "n=3 的 HRP 是「先聚成 2+1，再二分」。孤立腿 a 与配对腿 (b,c)：",
              "",
              "    w_a = v_bc / (v_a + v_bc),   v_bc = σ_b² + σ_c² + 2ρ_bc·σ_b·σ_c",
              "",
              "等波动时 `w_a = (2+2ρ)/(3+2ρ)`。**注意 ρ 越大 w_a 越大**：配对簇的方差被",
              "`2ρ_bc·σ_b·σ_c` 抬高，按逆方差它就该让出份额。", "",
              "| 自变量 | 孤立腿 σ | HRP max | 逆方差 max | ERC max |",
              "|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.block} · {r.axis} | {r.vol_a:.2f} | {r.hrp:.3f} | "
                     f"{r.inv_var:.3f} | {r.erc:.3f} |")
    lines += ["",
              "- **A 组**：三者相关 0→0.99，HRP 的最大份额 66.7%→79.9% 单调**上升**；",
              "  逆方差与 ERC 恒 33.3%（等波动时它们看不见相关，无法集中）。",
              "- **C 组**（σ 全相同、只动另外两只之间的 ρ_bc）：最大份额 72.2%→79.6%。",
              "  ⇒ **配对腿越抱团，被孤立那只拿越多**。",
              "- **B 组**（a 已是被孤立那只，只降它的 σ）：份额 79.2%→99.6% 继续叠加。",
              "- 关键：`w_a` 的顶层二分里**只有 v_a 与 v_bc 两块**，`ρ_a·`（孤立腿与别人的",
              "  相关）根本不进公式 —— 它只决定「谁是那只」。所以「我们的三只挺相关的」",
              "  这句话在 HRP 里正好是**集中度来源**，不是集中度的反证。", "",
              "⇒ 由莉酱的直觉（80% 需要低相关）是 `inv_var` 的性质：等波动下要 80% 需",
              "σ_a/σ = 1/2.83。**n=3 HRP 的 80% 恰是「三者高度相关」的推论**，",
              "而且 79.9% 是 n=3 的理论上限（ρ→1）。", ""]
    return df


# ---------------------------------------------------------------- 实盘录数
class _RecordHrp(EtfRotationAdvanced):
    """录下每次 `_risk_parity` 的输入协方差与输出权重。"""

    records: list[dict] = []

    def _risk_parity(self, sel, date, fallback):  # noqa: D102
        out = super()._risk_parity(sel, date, fallback)
        try:
            sub = self.ret_panel.loc[:date].tail(self.cov_window).reindex(columns=sel)
            sub = sub.dropna(how="all").dropna(axis=1, how="all")
            if len(sub) >= 20 and sub.shape[1] == len(sel):
                cov = sub.cov().to_numpy() * 252.0
                type(self).records.append({
                    "date": pd.Timestamp(date), "codes": list(sel), "cov": cov,
                    "w": out.to_numpy(float).copy(),
                })
        except Exception:  # 录数失败不影响回测
            pass
        return out


def _risk_share(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = w * (cov @ w)
    tot = float(w @ cov @ w)
    return m / tot if tot > 0 else np.full_like(w, np.nan)


def mode_real(lines: list[str]) -> pd.DataFrame:
    from backtest.engine import build_execution_split
    from scripts.portfolio.etf_contest5_eval import BEGIN
    from scripts.portfolio.etf_rotation_probe import run_variant

    env = load_env()
    spec = dict(FINAL)
    r0 = run_variant(env["bundle"], env["panels"], spec, BEGIN, None)
    spl = build_execution_split(env["open_adj"], env["close"],
                                set(pd.DatetimeIndex(r0["rebalance_dates"])))
    _RecordHrp.records = []          # 只留 open 口径那一次（PIT 已 shift，看到的是 ≤T）
    run_variant(env["bundle"], env["panels"], spec, BEGIN, None,
                execution_split=spl, execution_shift=True,
                strategy_cls=_RecordHrp)
    recs = _RecordHrp.records
    if not recs:
        lines += ["## 二、实测", "", "⚠️ 未录到 `_risk_parity` 调用（可能全走 fallback）。", ""]
        return pd.DataFrame()

    rows = []
    for r in recs:
        cov, w = r["cov"], r["w"]
        j = int(np.argmax(w))
        sd = np.sqrt(np.diag(cov))
        others = [i for i in range(3) if i != j]
        corr = cov / np.outer(sd, sd)

        def _sd2cov(c: np.ndarray, s: np.ndarray) -> np.ndarray:
            return c * np.outer(s, s)

        # 三个单点反事实（每次只动一个旋钮，看最大权重怎么变）
        c_pair0 = corr.copy()                     # ① 配对腿之间相关置 0
        c_pair0[others[0], others[1]] = 0.0
        c_pair0[others[1], others[0]] = 0.0
        w_pair0 = _hrp_weights(_sd2cov(c_pair0, sd))[j]

        c_top0 = corr.copy()                      # ② 最大腿与另两只相关置 0
        for i in others:
            c_top0[j, i] = c_top0[i, j] = 0.0
        w_top0 = _hrp_weights(_sd2cov(c_top0, sd))[j]

        w_flat = _hrp_weights(_sd2cov(corr, np.full(3, float(np.exp(np.log(sd).mean())))))[j]

        rs = _risk_share(cov, w)
        rows.append({
            "date": r["date"], "top": r["codes"][j], "maxw": float(w[j]),
            "maxw_pair0": float(w_pair0), "d_pair": float(w[j] - w_pair0),
            "maxw_top0": float(w_top0), "d_top": float(w[j] - w_top0),
            "maxw_flatsd": float(w_flat), "d_vol": float(w[j] - w_flat),
            "sd_top": float(sd[j]), "sd_others": float(np.mean(sd[others])),
            "sd_ratio": float(sd[j] / np.mean(sd[others])),
            "corr_top": float(np.mean([corr[j, i] for i in others])),
            "corr_pair": float(corr[others[0], others[1]]),
            "risk_top": float(rs[j]),
            "is_minvol": bool(j == int(np.argmin(sd))),
            "is_leastcorr": bool(j == int(np.argmin(corr.sum(axis=1)))),
        })
    df = pd.DataFrame(rows)

    lines += ["## 二、实测：定版 contest3 的 80% 是怎么拼出来的", "",
              f"样本 {len(df)} 个调仓日（录到三腿协方差的那些，T+1 open、PIT 修复后）。", "",
              "| 量 | p10 | 中位 | p90 |", "|---|---|---|---|"]
    for col, label in (("maxw", "最大单只权重（实际）"),
                       ("maxw_pair0", "① 配对腿相关置 0 后"),
                       ("d_pair", "① 的贡献（实际 − 置 0）"),
                       ("maxw_top0", "② 最大腿与别人相关置 0 后"),
                       ("d_top", "② 的贡献"),
                       ("maxw_flatsd", "③ 三腿波动拉平后"),
                       ("d_vol", "③ 的贡献"),
                       ("sd_ratio", "最大腿波动 ÷ 另两只均值"),
                       ("corr_top", "最大腿与另两只平均相关"),
                       ("corr_pair", "另两只之间相关"),
                       ("risk_top", "最大腿风险贡献份额")):
        qq = df[col].quantile([0.1, 0.5, 0.9])
        lines.append(f"| {label} | {qq.loc[0.1]:.3f} | {qq.loc[0.5]:.3f} | {qq.loc[0.9]:.3f} |")
    lines += ["",
              f"- 最大权重落在**波动最小**那只：{df.is_minvol.mean():.1%}；"
              f"落在**平均相关最低**那只：{df.is_leastcorr.mean():.1%}",
              f"- 最大腿与另两只平均相关 {df.corr_top.mean():.3f}；"
              f"另两只之间 {df.corr_pair.mean():.3f}（去重阈值 0.90 是它的上界）",
              f"- 归因（中位）：配对相关 **{df.d_pair.median():+.3f}** / "
              f"最大腿自身相关 **{df.d_top.median():+.3f}** / "
              f"波动差 **{df.d_vol.median():+.3f}**",
              ""]
    return df


# ---------------------------------------------------------------- 起点
def mode_begin(lines: list[str]) -> pd.DataFrame:
    from data.cache import DataCache
    from data.datasource import create_datasource

    cache = DataCache(create_datasource())
    raw = cache.read_daily("etf")
    px = raw["close"].unstack("code")
    px = px.reindex(columns=[c for c in POOL if c in px.columns])
    first = px.apply(lambda s: s.first_valid_index())
    n_bars = px.apply(lambda s: int(s.loc[:"2020-01-01"].notna().sum()))

    from data.etf_universe import ETF_CANDIDATES

    df = pd.DataFrame({
        "code": first.index,
        "first_day": first.to_numpy(),
        "bars_to_2020": n_bars.to_numpy(),
    })
    df["name"] = [ETF_CANDIDATES[c][0] for c in df.code]
    df = df.sort_values("first_day").reset_index(drop=True)

    alive = px.notna().sum(axis=1)
    per_year = alive.groupby(alive.index.year).agg(["min", "median", "max"])

    lines += ["## 三、45 只池的最早可用行情", "",
              f"池内最早一只：{df.first_day.min().date()}；最晚一只：{df.first_day.max().date()}。",
              f"2020 前已有 ≥250 日：{(df.bars_to_2020 >= 250).sum()} / {len(df)} 只。",
              "", "| 代码 | 名称 | 首根有效 K 线 | 2020 前交易日数 |", "|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.code} | {r.name} | {r.first_day.date()} | {r.bars_to_2020} |")
    lines += ["", "逐年「当天有行情的池内只数」：", "",
              "| 年份 | 最少 | 中位 | 最多 |", "|---|---|---|---|"]
    for y, r in per_year.iterrows():
        lines.append(f"| {y} | {int(r['min'])} | {int(r['median'])} | {int(r['max'])} |")
    lines.append("")
    return df


def _force_offline() -> None:
    """把 `create_datasource` 换成离线桩（本地 parquet 已完整时用，避开 TGW 连接数限制）。"""
    import data.datasource as dsmod
    from data.offline import OfflineQuietDataSource

    dsmod.create_datasource = lambda *a, **k: OfflineQuietDataSource()  # noqa: ARG005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="theory,real,begin")
    ap.add_argument("--offline", action="store_true", help="走本地 parquet（不连 TGW）")
    args = ap.parse_args()
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    if args.offline:
        _force_offline()

    lines = ["# 三锋 HRP 单押 80% 溯源（2026-10-31）", ""]
    frames: dict[str, pd.DataFrame] = {}
    if "theory" in modes:
        frames["theory"] = mode_theory(lines)
    if "real" in modes:
        frames["real"] = mode_real(lines)
    if "begin" in modes:
        frames["begin"] = mode_begin(lines)

    text = "\n".join(lines)
    print(text)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(text, encoding="utf-8")
    print(f"\n[saved] {OUT_MD}")
    for k, v in frames.items():
        if len(v):
            v.to_csv(OUT_MD.with_name(f"hrp_diag_{k}.csv"), index=False)
            print(f"\n--- {k} ---\n{v.to_string(max_rows=30)}")


if __name__ == "__main__":
    main()
