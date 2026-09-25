"""批次 8：表格基础模型全量滚动对照（TabPFN-3 / TabICL V2 vs LightGBM）。

GOOD_MACHINE_TASKS.md 批次 8 的好机器全量 runner（v1 结论只作方向假设来源，
全部臂用最新版：tabicl 2.2.0（V2 权重 tabicl-regressor-v2-20260212）+
tabpfn 8.4.0（v3 权重 Prior-Labs/tabpfn_3）。v1 遗留局限直接在本 runner 修复：
test_step=1 全量测试日（v1 每步 3 天抽样高估 IR）、滚动 W 月窗口扫描、
可交易掩码 IC（防纸面三判据）、组合层对照口径 = 批次 1 正名 open（T+1 开盘）。

滚动协议：每个月初边界 b 重训一次，训练窗 = b 前 W 个自然月（2m/3m 扫描），
预测该月**全部**交易日（test_step=1）；OOS 逐日 RankIC + Newey-West t（lag 5）。
组合层：月末调仓 Top10% 等权，信号取边界日、T+1 开盘成交、入场掩可交易
（data/tradability T+1 口径），成本用 default_costs()。

臂：gbdt / tabpfn3 / tabicl（PREDICTORS 注册）；另输出 tabpfn3+gbdt 秩平均
参考臂（先打印两模型秩相关，>0.3 则标注"无增量空间"）。

用法（先 hs300 全量外推，再决定全A 是否铺满）::

    python -m scripts.evaluation.tabfm_rolling --dataset hs300_2022_2025 \
        --windows 2,3 --methods gbdt,tabpfn3,tabicl --top 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# tabpfn_3 权重已由 hf-mirror 预取进 HF 缓存（门控绕行）；离线加载防 etag 检查
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TABPFN_TOKEN", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("tabfm_rolling")

NW_LAG = 5


def _nw_t(ic: pd.Series) -> float:
    """逐日 IC 的 Newey-West t（lag=5，自动化长VAR；均值先取再去心）。"""
    x = ic.dropna().to_numpy(dtype=float)
    if len(x) < 10:
        return float("nan")
    mu = float(x.mean())
    x = x - mu
    n = len(x)
    g = np.zeros(NW_LAG + 1)
    g[0] = float(x @ x) / n
    for l in range(1, NW_LAG + 1):
        g[l] = float(x[l:] @ x[:-l]) / n
    lrv = g[0] + 2.0 * sum((1.0 - l / (NW_LAG + 1.0)) * g[l]
                           for l in range(1, NW_LAG + 1))
    if lrv <= 0:
        return float("nan")
    return float(np.sqrt(n) * mu / np.sqrt(lrv))


def _month_boundaries(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """测试段内每月首个交易日。"""
    s = pd.Series(days, index=days)
    first = s.groupby([days.year, days.month]).first()
    return [pd.Timestamp(d) for d in first]


def _ns(idx) -> pd.DatetimeIndex:
    """parquet 读出的面板索引是 us 精度、内存构建的是 ns——统一到 ns，
    否则 pandas 3.0 下跨精度 reindex/loc 全部落空（None of ... in index）。"""
    return pd.DatetimeIndex(idx).as_unit("ns")


def load_setup(dataset: str, top: int, test_begin: str):
    """特征（定型期固定选择，无前视）+ 复权 close + 可交易掩码 + 基准。"""
    from config import Config
    from data.cache_helpers import build_panel
    from data.tradability import build_tradable_mask
    from factor.preprocessing import standardize_zscore
    from research.factor_library import FactorLibrary
    from scripts.common.e2e_common import select_features

    tb = pd.Timestamp(test_begin)
    panel, _ = build_panel(Config.get(), int(Config.discipline()["begin"]),
                           20261231, offline=True, include_market_cap=True)
    close = panel["close"]
    close.index = _ns(close.index)
    days = close.index
    dev_days = days[days < tb]
    valid_days = dev_days[dev_days > tb - pd.Timedelta(days=365)]

    feats = FactorLibrary(dataset=dataset).load_library_features()
    # 行列都对齐到 close 网格：各 builder 逐因子丢过全 NaN 列，列集不齐时
    # select_features 的严格交集会把网格坍缩到个位数股票
    feats = {k: standardize_zscore(v.reindex(_ns(v.index))
                                   .reindex(index=days, columns=close.columns))
             for k, v in feats.items()}
    fwd_v = close.pct_change(1, fill_method=None).shift(-1).loc[valid_days]
    sel, _q = select_features(feats, fwd_v, quality_days=valid_days,
                              panel_days=dev_days, max_features=top)
    log.info("特征 %d 个（定型期固定选择，quality=%s~%s）",
             len(sel), valid_days[0].date(), valid_days[-1].date())

    bwd_path = Path(str(Config.cache()["root"])) / "backward_factor.parquet"
    bwd = (pd.read_parquet(bwd_path)
           if bwd_path.exists() else None)
    if bwd is not None:
        bwd.index = _ns(bwd.index)
        bwd = bwd.reindex(index=close.index, columns=close.columns).ffill()
    trad = build_tradable_mask(close, bwd=bwd)
    if trad.columns.duplicated().any():
        trad = trad.loc[:, ~trad.columns.duplicated()]
    return sel, close, trad, panel


def rolling_predict(method: str, features: dict, labels: pd.DataFrame,
                    close: pd.DataFrame, test_begin: str, window_months: int,
                    device: str) -> tuple[pd.DataFrame, dict]:
    """按月边界滚动：W 月训练窗 → 预测次月全部交易日。返回 (pred, per-window 计时)。"""
    from model.predictor import PREDICTORS
    days = close.index
    tb = pd.Timestamp(test_begin)
    bounds = [b for b in _month_boundaries(days) if b >= tb]
    preds = []
    timing = []
    cls = PREDICTORS[method]
    for j, b in enumerate(bounds):
        w_start = b - pd.DateOffset(months=window_months)
        tr = days[(days >= w_start) & (days < b)]
        # 预测段到下一个边界（月末 3-5 天不留给下个月，无缝铺满测试段）
        nxt = bounds[j + 1] if j + 1 < len(bounds) else b + pd.DateOffset(months=1)
        te = days[(days >= b) & (days < nxt)]
        # W2m 窗口在春节月只有 ~34-39 个交易日，阈值取 30 防大面积误跳
        if len(tr) < 30 or len(te) == 0:
            log.warning("窗口 %s 训练日不足（%d），跳过", b.date(), len(tr))
            continue
        # ICL 模型不容忍全 NaN 特征列（会清零逐行有效掩码）；窗内覆盖
        # <50% 的特征剔除，各窗特征集动态收窄。门控看训练窗切片，
        # 但保留全时段面板——predict 还要查测试日。
        feats_ok = {
            k: v for k, v in features.items()
            if float(v.loc[tr].notna().mean().mean()) >= 0.5
        }
        if len(feats_ok) < 5:
            log.warning("窗口 %s 有效特征不足（%d），跳过", b.date(), len(feats_ok))
            continue
        t0 = time.time()
        # tabicl 显式默认 cpu（与 09-22 冒烟口径一致）；tabpfn3 None=自动（cuda 优先）
        device_eff = device if device else ("cpu" if method == "tabicl" else None)
        model = cls(device=device_eff) if "device" in cls.__init__.__code__.co_varnames \
            else cls()
        model.fit({k: v.loc[tr] for k, v in feats_ok.items()}, labels.loc[tr])
        t_fit = time.time() - t0
        t0 = time.time()
        p = model.predict({k: v.loc[te] for k, v in feats_ok.items()})
        t_pred = time.time() - t0
        preds.append(p)
        timing.append({"bound": str(b.date()), "n_train": len(tr), "n_test": len(te),
                       "fit_s": round(t_fit, 1), "pred_s": round(t_pred, 1),
                       "n_context": getattr(model, "n_samples_", None)})
        log.info("[%s W%dm] fit %.0fs pred %.0fs（context %s）", b.date(),
                 window_months, t_fit, t_pred, getattr(model, "n_samples_", "-"))
    pred = pd.concat(preds) if preds else pd.DataFrame()
    if not pred.empty:
        pred = pred[~pred.index.duplicated(keep="last")]
    return pred, {"windows": timing}


def eval_ic(pred: pd.DataFrame, fwd: pd.DataFrame, trad: pd.DataFrame) -> dict:
    """全量测试日 RankIC：原始 + 可交易掩码两套（pred 缺失日自动收窄）。"""
    from stats.ic import calc_ic_series
    common = pred.index.intersection(fwd.index)
    pm = pred.loc[common]
    ic_all = calc_ic_series(pm, fwd.loc[common])
    mask = trad.reindex(index=pm.index, columns=pm.columns).fillna(True)
    ic_trad = calc_ic_series(pm.where(mask), fwd.loc[common])
    return {"ic_mean": float(ic_all.mean()), "ic_ir": _icir(ic_all), "t_nw": _nw_t(ic_all),
            "ic_trad_mean": float(ic_trad.mean()), "ic_trad_ir": _icir(ic_trad),
            "t_nw_trad": _nw_t(ic_trad), "n_days": int(ic_all.dropna().shape[0])}


def _icir(ic: pd.Series) -> float:
    s = float(ic.dropna().std(ddof=1))
    return float(ic.dropna().mean() / s) if s > 0 else float("nan")


def portfolio_arm(sig: pd.DataFrame, panel: dict, trad: pd.DataFrame,
                  frac: float = 0.10) -> dict:
    """Top10% 等权月度组合：信号=边界日截面、T+1 开盘成交（批次 1 正名 open 口径）。"""
    from backtest.costs import default_costs
    c = default_costs()
    # 单次进出近似：双边佣金 + 卖出印花 + 双边滑点（一次性换手成本）
    round_trip = 2 * c.commission_rate + c.stamp_duty + 2 * c.slippage_bp / 1e4
    openp = panel["open"]
    days = openp.index
    bounds = [b for b in _month_boundaries(days) if b >= pd.Timestamp("2025-01-01")]
    costs = default_costs()
    to_cost = costs.buy + costs.sell if hasattr(costs, "buy") else 0.003
    nav, turns, rets = 1.0, [], []
    for i in range(len(bounds) - 1):
        b, nb = bounds[i], bounds[i + 1]
        if b not in sig.index:
            continue
        # 信号日 b 的次日开盘入场 → 持有到 nb 次日开盘
        try:
            entry = openp.loc[days[days > b][0]]
            exit_d = days[days > nb][0]
        except IndexError:
            continue
        exitp = openp.loc[exit_d]
        s = sig.loc[b].reindex(entry.index)
        tm = trad.loc[b].reindex(entry.index).fillna(True)
        s = s.where(tm)
        pick = s.dropna().nlargest(max(1, int(len(s.dropna()) * frac))).index
        if len(pick) == 0:
            continue
        r = (exitp.reindex(pick) / entry.reindex(pick) - 1.0).fillna(0.0)
        gross = float(r.mean())
        nav *= (1.0 + gross - round_trip)
        rets.append(gross - round_trip)
        turns.append(1.0)
    years = max(len(rets) / 12.0, 1e-9)
    ann = nav ** (1 / years) - 1.0
    vol = float(np.std(rets, ddof=1)) * np.sqrt(12) if len(rets) > 1 else float("nan")
    return {"p_annual": ann, "p_sharpe": ann / vol if vol and vol > 0 else float("nan"),
            "n_months": len(rets), "turnover_per_month": 1.0}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="hs300_2022_2025")
    ap.add_argument("--windows", default="2,3", help="训练窗（自然月），逗号分隔")
    ap.add_argument("--methods", default="gbdt,tabpfn3,tabicl")
    ap.add_argument("--device", default=None, help="ICL 模型设备（默认自动）")
    ap.add_argument("--top", type=int, default=50, help="定型期固定选择的特征数")
    ap.add_argument("--test-begin", default="2025-01-01")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--out", default="reports/tabfm_rolling")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from model.labels import build_labels
    out = Path(args.out) / f"{args.dataset}_tb{args.test_begin[:7]}"
    out.mkdir(parents=True, exist_ok=True)

    sel, close, trad, panel = load_setup(args.dataset, args.top, args.test_begin)
    labels, _emb = build_labels(close, horizon=1, mode="rank")
    # IC 口径：pred(d) vs d→d+1 前瞻收益（build_labels 同向；组合臂用开盘价独立算）
    fwd = (close.shift(-1) / close - 1.0)
    test_days = close.index[close.index >= pd.Timestamp(args.test_begin)]

    rows, timing_all = [], {}
    preds_by = {}
    for w in [int(x) for x in args.windows.split(",")]:
        for method in args.methods.split(","):
            key = f"{method}_w{w}m"
            log.info("===== %s =====", key)
            try:
                pred, info = rolling_predict(method, sel, labels, close,
                                             args.test_begin, w, args.device)
            except Exception as exc:  # noqa: BLE001
                log.error("%s 失败: %s", key, exc)
                rows.append({"arm": key, "error": str(exc)[:200]})
                continue
            if pred.empty:
                continue
            preds_by[key] = pred
            td = test_days.intersection(pred.index)
            m = eval_ic(pred.loc[td], fwd.loc[td], trad)
            m.update(portfolio_arm(pred, panel, trad, args.frac))
            m["arm"] = key
            rows.append(m)
            timing_all[key] = info
            log.info("%s: IC=%.4f t=%.2f（可交易 IC=%.4f t=%.2f）年化=%.2f%%",
                     key, m["ic_mean"], m["t_nw"], m["ic_trad_mean"],
                     m["t_nw_trad"], m.get("p_annual", float("nan")) * 100)

    # 参考臂：tabpfn3+gbdt 秩平均（先看两模型族信号秩相关是否 <0.3）
    for w in [int(x) for x in args.windows.split(",")]:
        ka, kb = f"tabpfn3_w{w}m", f"gbdt_w{w}m"
        if ka in preds_by and kb in preds_by:
            a, b = preds_by[ka].align(preds_by[kb], join="inner")
            ra = a.rank(axis=1, pct=True)
            rb = b.rank(axis=1, pct=True)
            corr = float(np.nanmean([ra.loc[d].corr(rb.loc[d])
                                     for d in ra.index[:20]]))
            ens = (ra + rb) / 2.0
            td = test_days.intersection(ens.index)
            m = eval_ic(ens.loc[td], fwd.loc[td], trad)
            m.update(portfolio_arm(ens, panel, trad, args.frac))
            m["arm"] = f"tabpfn3+gbdt_rankavg_w{w}m"
            m["rank_corr_vs_gbdt"] = corr
            m["note"] = "秩相关>=0.3 → 集成无增量空间" if corr >= 0.3 else "秩相关<0.3"
            rows.append(m)
            log.info("秩平均 w%dm: 秩相关=%.3f IC=%.4f", w, corr, m["ic_mean"])

    pd.DataFrame(rows).set_index("arm").to_csv(
        out / "summary.csv", encoding="utf-8-sig")
    (out / "timing.json").write_text(
        json.dumps(timing_all, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("已写出 %s", out)


if __name__ == "__main__":
    main()
