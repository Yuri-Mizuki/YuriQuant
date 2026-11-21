"""申万 0922 择时研报门槛探针（纯择时研究线首篇，reports/sws_timing_probe/）。

研报：申万宏源 2025-09-22《金工量化择时策略研究系列之三：「趋势」「震荡」环境的
划分与择时策略——以上证指数为例》（PDF 仓库外 D:\\project\\研报\\择时\\）。

探针范围（判据先于数据定义，过 gate 才进策略回测）
--------------------------------------------------
G1 可预测性：平滑后模型测试集准确率 − 持续性基线（y_hat=y_{t-1}）≥ +3pp 且 ≥ 60%。
G2 经济相关性：平滑信号（T+1 可用）对 5 日前瞻收益的周频（周三采样）Spearman IC，
   测试窗 ρ ≥ 0.3 且 t ≥ 2。
G3 上界可实现：完美标签 T+1 延迟 + 研报仓位规则 + 10bp/边成本，年化超额 > 0
   且 Sharpe > 半仓持有基准。
任一 FAIL ⇒ 探针收口，不进策略回测。

复现口径（尽量 1:1，差异显式标注）
----------------------------------
- 标注：Zig-Zag（转折 >10%，|年化对数收益| ≥20% 且 ≥63 交易日 ⇒ 趋势段）+ 分层 Binseg
  单断点修正（后段 |β2|<|β1| 且单侧 z>1.645 ⇒ 尾段改标震荡；**余下头部复检趋势条件，
  站不住 ⇒ 整段塌震荡**——对应研报「分层诊断」措辞，校准靶驱动的解读，见 §四 边界）。
- 特征 6 个（全部 {0,1}，1=趋势票）：MA_1030(40d) / MA_0510(20d) / price_120(q80) /
  Volume(q70，用昨日量) / Volatility_past(布林带宽 20d 均值 q80) / Squeeze_Breakout
  (压缩 + 30d 内突破 >1 次)。参数取研报表 2-7 选定档。
- 模型：等权（6 票均值 ≥0.5）/ LogisticRegression（sklearn 默认）/ DecisionTree
  （max_depth=4，对照研报图 6 形态），训练 2020/1/2–2023/12/18、测试至 2025/8/29
  （研报表 1 口径），20 日信号平滑。
- 校准靶：研报表 8（单特征训练准确度）/ 表 9（逻辑回归系数）/ 表 11（模型准确度）/
  混淆矩阵行和（训练震荡 456 / 趋势 505，测试 197 / 215）。

用法::

    python -m scripts.timing.sws_regime_probe               # 全量跑
    python -m scripts.timing.sws_regime_probe --self-check  # 因果性抽查 + 不变量
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from backtest.metrics import annual_return, max_drawdown, sharpe_ratio
from stats import PERIODS_PER_YEAR

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "sws_timing_probe"
DATA_CACHE = OUT / "data" / "000001.SH.parquet"

INDEX_CODE = "000001.SH"
FETCH_BEGIN = 20140101  # 特征预热（252 日分位窗）+ 2015 起标注
LABEL_BEGIN = "2015-01-01"  # 研报：上证指数 2015 年以来
TRAIN_BEGIN, TRAIN_END = "2020-01-02", "2023-12-18"  # 研报表 1
TEST_BEGIN, TEST_END = "2023-12-19", "2025-08-29"

ZZ_THRESH = 0.10  # 转折阈值
ZZ_MIN_ANN = 0.20  # 最小年化收益（对数年化）
ZZ_MIN_LEN = 63  # 最短趋势段（交易日）
BINSEG_MIN_SEG = 10  # 断点两侧最短子段
BINSEG_Z = 1.645  # 单侧 5%
FEAT_WARMUP = 252  # 最长回看窗（年分位）
SMOOTH = 20  # 信号平滑窗

REBALANCE_WEEKDAY = 2  # 周三
INIT_POS = 0.5
STEP = 0.1  # 每 1% 回看收益动 0.1 仓
COST_GRID_BPS = (0.0, 10.0, 30.0)  # 每边成本敏感性

FEATURES = [
    "f_ma_1030", "f_ma_0510", "f_price_120", "f_volume",
    "f_volatility_past", "f_squeeze_breakout",
]

# 研报校准靶（表 8 / 表 9 / 表 11 / 混淆矩阵行和）
REPORT_FEAT_ACC = {  # 训练集单特征准确度
    "f_ma_1030": 0.5858, "f_ma_0510": 0.5380, "f_price_120": 0.6025,
    "f_volume": 0.6348, "f_volatility_past": 0.5578, "f_squeeze_breakout": 0.5754,
}
REPORT_LOGIT_COEF = {
    "f_ma_1030": 1.70, "f_price_120": 1.46, "f_ma_0510": 1.26,
    "f_volume": 1.03, "f_squeeze_breakout": 0.58, "f_volatility_past": -0.04,
}
REPORT_MODEL_ACC = {  # 表 11：(是否平滑) -> (模型 -> (训练, 测试))
    False: {"equal": (0.6472, 0.8204), "logit": (0.7024, 0.8325), "tree4": (0.6982, 0.8010)},
    True: {"equal": (0.6067, 0.8859), "logit": (0.7097, 0.8350), "tree4": (0.7076, 0.8398)},
}
REPORT_LABEL_RATIO = {"train_osc": 456, "train_trend": 505, "test_osc": 197, "test_trend": 215}


# ---------------------------------------------------------------- 数据
def load_index_daily(refresh: bool = False) -> pd.DataFrame:
    """上证指数日K（000001.SH），带 parquet 缓存（>5 天自动重拉）。"""
    if DATA_CACHE.exists() and not refresh:
        df = pd.read_parquet(DATA_CACHE)
        if df.index.max() >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=5):
            return df
    from data.datasource import create_datasource

    ds = create_datasource()
    raw = ds.get_daily_kline([INDEX_CODE], FETCH_BEGIN, int(dt.date.today().strftime("%Y%m%d")))
    df = raw.xs(INDEX_CODE, level="code").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_CACHE)
    return df


# ---------------------------------------------------------------- 标注
def _slope_ols(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """返回 (斜率, 斜率标准误, 段内 SSE)。"""
    tm = t - t.mean()
    sxx = float((tm * tm).sum())
    beta = float((tm * (y - y.mean())).sum() / sxx)
    alpha = float(y.mean() - beta * t.mean())
    resid = y - alpha - beta * t
    sse = float((resid * resid).sum())
    se = float(np.sqrt(sse / (len(t) - 2) / sxx))
    return beta, se, sse


def zigzag_legs(close: pd.Series) -> list[tuple[int, int]]:
    """Zig-Zag 枢纽点 → 相邻枢纽间波段 [(i0, i1), ...]，末段延伸到样本尾（未确认）。"""
    px = close.to_numpy()
    n = len(px)
    piv = [0]
    direction = 0
    hi = lo = 0
    for i in range(1, n):
        if direction == 0:
            if px[i] > px[hi]:
                hi = i
            if px[i] < px[lo]:
                lo = i
            if px[i] > px[lo] * (1 + ZZ_THRESH):
                piv.append(lo)
                direction = 1
                hi = i
            elif px[i] < px[hi] * (1 - ZZ_THRESH):
                piv.append(hi)
                direction = -1
                lo = i
        elif direction == 1:
            if px[i] > px[hi]:
                hi = i
            elif px[i] < px[hi] * (1 - ZZ_THRESH):
                piv.append(hi)
                direction = -1
                lo = i
        else:
            if px[i] < px[lo]:
                lo = i
            elif px[i] > px[lo] * (1 + ZZ_THRESH):
                piv.append(lo)
                direction = 1
                hi = i
    piv.append(n - 1)
    return [(piv[k], piv[k + 1]) for k in range(len(piv) - 1)]


def binseg_flatten(px: np.ndarray, i: int, j: int) -> int | None:
    """趋势段 [i, j] 内搜单断点（最小化分段线性 SSE）；后段显著走平 ⇒ 返回断点下标。"""
    y = np.log(px[i : j + 1])
    t = np.arange(len(y), dtype=float)
    best_sse = np.inf
    best: tuple | None = None
    for k in range(BINSEG_MIN_SEG, len(y) - BINSEG_MIN_SEG + 1):
        b1, se1, s1 = _slope_ols(t[:k], y[:k])
        b2, se2, s2 = _slope_ols(t[k:], y[k:])
        if s1 + s2 < best_sse:
            best_sse = s1 + s2
            best = (k, b1, se1, b2, se2)
    if best is None:
        return None
    k, b1, se1, b2, se2 = best
    z = (b1 - b2) / float(np.hypot(se1, se2))
    if abs(b2) < abs(b1) and np.sign(z) == np.sign(b1) and abs(z) > BINSEG_Z:
        return i + k
    return None


def _leg_qualifies(px: np.ndarray, i: int, j: int) -> bool:
    """波段 [i, j) 是否满足趋势条件：≥63 交易日且 |年化对数收益| ≥ 20%。"""
    n = j - i
    if n < ZZ_MIN_LEN:
        return False
    ann = abs(float(np.log(px[j] / px[i]))) * PERIODS_PER_YEAR / n
    return ann >= ZZ_MIN_ANN


def _trend_span(px: np.ndarray, i: int, j: int) -> tuple[int, int] | None:
    """分层 Binseg 修正：断点后尾段改标震荡；**余下头部波段复检趋势条件**，
    站不住 ⇒ 整段塌成震荡（递归到头部站得住为止）。返回最终趋势区间或 None。"""
    brk = binseg_flatten(px, i, j)
    if brk is None:
        return (i, j)
    if _leg_qualifies(px, i, brk):
        return _trend_span(px, i, brk)
    return None


def build_labels(close: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """全样本 regime 标签（训练基准；**含未来信息**——研报口径如此，非因果，见 G3 处理）。

    返回 (label ∈ {0,1}，legs 表：起止 / 年化 / 时长 / 是否趋势 / 最终趋势区间)。
    """
    px = close.to_numpy()
    idx = close.index
    rows = []
    for (i, j) in zigzag_legs(close):
        if i >= j:
            continue
        n = j - i
        ann = abs(float(np.log(px[j] / px[i]))) * PERIODS_PER_YEAR / n
        is_trend = bool(ann >= ZZ_MIN_ANN and n >= ZZ_MIN_LEN)
        span = _trend_span(px, i, j) if is_trend else None
        rows.append({"i": i, "j": j, "begin": idx[i], "end": idx[j], "days": n,
                     "ann_ret": ann, "is_trend": is_trend,
                     "trend_i": span[0] if span else None,
                     "trend_j": span[1] if span else None})
    legs_df = pd.DataFrame(rows)
    label = pd.Series(0, index=idx, dtype=int)
    for r in rows:
        if r["trend_i"] is not None:
            label.iloc[r["trend_i"]: r["trend_j"]] = 1  # 其余（尾段/塌掉整段）保持 0
    label[label.index < LABEL_BEGIN] = 0  # 标注域 2015 起（与研报一致）
    return label, legs_df


# ---------------------------------------------------------------- 特征（全部因果：仅用 ≤t 数据）
def _run_ge(sign: pd.Series, win: int) -> pd.Series:
    """同号连续游程 ≥ win ⇒ 1；sign=0 视为断开。"""
    grp = (sign != sign.shift()).cumsum()
    runlen = sign.groupby(grp).cumcount() + 1
    return ((runlen >= win) & (sign != 0)).astype(int)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """六个 {0,1} 特征（1=趋势票）。逐个仅依赖 ≤t 的 close/volume。"""
    close, vol = df["close"], df["volume"]
    f = pd.DataFrame(index=df.index)
    # 1/2) 均线排列持续性（研报 §2.2.1，参数取表 2/3 选定档）
    f["f_ma_1030"] = _run_ge(np.sign(close.rolling(10).mean() - close.rolling(30).mean()), 40)
    f["f_ma_0510"] = _run_ge(np.sign(close.rolling(5).mean() - close.rolling(10).mean()), 20)
    # 3) 120 日价格斜率 |绝对值| > 过去一年 q80 ⇒ 0（长期斜率平缓特征的反向：剧烈=末期）
    slope = close.rolling(120).apply(
        lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0]), raw=True
    )
    q = abs(slope).rolling(FEAT_WARMUP).quantile(0.80)
    f["f_price_120"] = (abs(slope) <= q).astype(int)
    # 4) 昨日成交量 > 过去一年 q70 ⇒ 0（§2.2.2 用「昨天」的量）
    vq = vol.shift(1).rolling(FEAT_WARMUP).quantile(0.70)
    f["f_volume"] = (vol.shift(1) <= vq).astype(int)
    # 5) 布林带宽（20,2σ）20 日均值 > 过去一年 q80 ⇒ 0（高波动震荡）
    bbw = 4.0 * close.rolling(20).std(ddof=1)
    bbw20 = bbw.rolling(20).mean()
    f["f_volatility_past"] = (bbw20 <= bbw20.rolling(FEAT_WARMUP).quantile(0.80)).astype(int)
    # 6) 压缩（当前带宽 < 年中位数）且 30 日内突破布林带 >1 次 ⇒ 1
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std(ddof=1)
    breakout = (close > ma20 + 2 * sd20) | (close < ma20 - 2 * sd20)
    cnt = breakout.rolling(30).sum()
    f["f_squeeze_breakout"] = ((bbw < bbw.rolling(FEAT_WARMUP).median()) & (cnt > 1)).astype(int)
    return f


# ---------------------------------------------------------------- 模型
def smooth(sig: pd.Series) -> pd.Series:
    """研报 §2.3：回看 20 天信号取均值，≥0.5 判趋势。"""
    return (sig.rolling(SMOOTH).mean() >= 0.5).astype(int)


def fit_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> dict[str, pd.Series]:
    """三模型 ×（训练/测试预测）。返回 {名: 全窗 0/1 预测}（训练段+测试段拼接）。"""
    pred = {}
    pred["equal"] = pd.concat(
        [(X_tr.mean(axis=1) >= 0.5).astype(int), (X_te.mean(axis=1) >= 0.5).astype(int)]
    )
    logit = LogisticRegression(max_iter=1000, random_state=0)
    logit.fit(X_tr.to_numpy(), y_tr.to_numpy())
    pred["logit"] = pd.Series(
        logit.predict(X_tr.to_numpy()).tolist() + logit.predict(X_te.to_numpy()).tolist(),
        index=pred["equal"].index,
    )
    pred["logit_coef"] = pd.Series(logit.coef_[0], index=list(X_tr.columns))
    tree = DecisionTreeClassifier(max_depth=4, random_state=0)  # 研报图 6 为 depth-4 形态
    tree.fit(X_tr.to_numpy(), y_tr.to_numpy())
    pred["tree4"] = pd.Series(
        tree.predict(X_tr.to_numpy()).tolist() + tree.predict(X_te.to_numpy()).tolist(),
        index=pred["equal"].index,
    )
    return pred


def accuracy(pred: pd.Series, label: pd.Series, mask: pd.Series) -> float:
    idx = mask[mask].index
    p, y = pred.loc[idx], label.loc[idx]
    ok = p == y
    return float(ok.mean())


# ---------------------------------------------------------------- 策略（研报 §3 规则）
def run_strategy(close: pd.Series, sig: pd.Series, cost_bps: float) -> dict:
    """研报动态仓位：周三调仓，回看期收益 = 上次调仓日收盘 → 本次调仓前一交易日收盘；
    趋势追涨杀跌 / 震荡均值回归，每满 1% 动 0.1 仓，钳 [0,1]。
    ``sig`` 传入**调用方已延迟到 T+1 可用**的信号；调仓以收盘价执行、次日起生效。
    首日以 0.5 基础仓位进入（建仓成本不计，见 §四 边界）。
    """
    ret = close.pct_change().fillna(0.0)
    dates = close.index
    pos_eff = pd.Series(INIT_POS, index=dates)  # 每日生效仓位（调仓次日起更新）
    cost = pd.Series(0.0, index=dates)
    cur = INIT_POS
    prev_r: pd.Timestamp | None = None
    for r in [d for d in dates if d.weekday() == REBALANCE_WEEKDAY]:
        loc = dates.get_loc(r)
        if loc == 0:
            continue
        if prev_r is not None:
            lb = float(close.iloc[loc - 1] / close.loc[prev_r] - 1.0)
        else:
            lb = 0.0  # 首个调仓日无回看期，仓位不动
        s = float(sig.loc[r])
        pct = float(np.floor(abs(lb) * 100.0) * STEP) * np.sign(lb)
        new = float(np.clip(cur + (pct if s >= 0.5 else -pct), 0.0, 1.0))
        cost.iloc[loc] = abs(new - cur) * cost_bps * 1e-4
        cur = new
        pos_eff.iloc[loc + 1:] = cur
        prev_r = r
    sr = pos_eff * ret - cost
    return {
        "ret": sr,
        "pos": pos_eff,
        "avg_pos": float(pos_eff.mean()),
        "turnover": float(cost.sum() / max(cost_bps * 1e-4, 1e-12)) if cost_bps > 0 else float("nan"),
    }


def metrics_row(ret: pd.Series) -> dict:
    return {
        "total": float((1 + ret).prod() - 1),
        "annual": annual_return(ret),
        "vol": float(ret.std() * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe": sharpe_ratio(ret),
        "mdd": max_drawdown(ret),
    }


# ---------------------------------------------------------------- gate 指标
def weekly_ic(close: pd.Series, sig: pd.Series, mask: pd.DatetimeIndex) -> tuple[float, float, int]:
    """周三采样：已延迟信号 vs 5 日前瞻收益的 Spearman ρ 与 t。"""
    fwd = close.shift(-5) / close - 1.0
    a = pd.Series([sig.loc[d] for d in mask], dtype=float)
    b = pd.Series([fwd.loc[d] for d in mask], dtype=float)
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if a.nunique() < 2 or len(a) < 10:
        return float("nan"), float("nan"), len(a)
    rho = float(a.corr(b, method="spearman"))
    t = rho * np.sqrt((len(a) - 2) / (1 - rho * rho)) if abs(rho) < 1 else float("nan")
    return rho, float(t), len(a)


# ---------------------------------------------------------------- self-check
def self_check(raw: pd.DataFrame, feat: pd.DataFrame, label: pd.Series,
               legs_df: pd.DataFrame) -> list[str]:
    """因果性抽查（截断重算逐值比对）+ 标注/策略不变量。返回违例列表（空 = 全过）。"""
    bad: list[str] = []
    rng = np.random.default_rng(0)
    valid = feat.dropna().index
    valid = valid[valid >= pd.Timestamp(LABEL_BEGIN)]
    # 1) 因果性：任意 5 个日期，用 ≤d 的截断表重算特征，必须与全样本逐值一致
    for d in pd.DatetimeIndex(rng.choice(valid.to_numpy(), size=5, replace=False)):
        f_trunc = build_features(raw.loc[:d]).iloc[-1]
        for col in FEATURES:
            if int(f_trunc[col]) != int(feat.loc[d, col]):
                bad.append(f"因果泄漏: {col}@{d.date()} 截断={f_trunc[col]} 全样本={feat.loc[d, col]}")
    # 2) 标注不变量
    if not label.isin([0, 1]).all():
        bad.append("标签非 {0,1}")
    lw = label.loc[LABEL_BEGIN:TEST_END]
    share = float(lw.mean())
    if not (0.2 <= share <= 0.8):
        bad.append(f"趋势占比超出常识带 [0.2,0.8]: {share:.3f}")
    if len(legs_df) < 10:
        bad.append(f"Zig-Zag 波段过少: {len(legs_df)}")
    # 3) 策略不变量：仓位 ∈ [0,1] 且有限
    win = raw["close"].loc[TRAIN_BEGIN:TEST_END]
    sig = label.shift(1).loc[win.index]  # T+1 可用
    r = run_strategy(win, sig, 10.0)
    p = r["pos"]
    if not ((p >= 0).all() and (p <= 1).all()) or not np.isfinite(p).all():
        bad.append("仓位越界 [0,1] 或含非有限值")
    return bad


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-check", action="store_true", help="只跑因果抽查 + 不变量")
    ap.add_argument("--refresh-data", action="store_true", help="强制重拉指数日K")
    args = ap.parse_args()

    raw = load_index_daily(refresh=args.refresh_data)
    print(f"指数日K {INDEX_CODE}: {raw.index.min().date()} ~ {raw.index.max().date()}（{len(raw)} 行）")

    label, legs_df = build_labels(raw["close"])
    feat = build_features(raw)
    feat_valid = feat.dropna().index  # 252 日预热完成之后

    if args.self_check:
        bad = self_check(raw, feat, label, legs_df)
        if bad:
            print("SELF-CHECK FAIL:")
            for b in bad:
                print("  -", b)
            return 1
        print("SELF-CHECK PASS（因果抽查 5 点逐值一致；标注/仓位不变量全过）")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    legs_df.to_csv(OUT / "zigzag_legs.csv", index=False, encoding="utf-8-sig")

    # ---------- 校准：标签比例 + 单特征准确度（研报表 8 / 混淆矩阵行和） ----------
    tr_mask = pd.Series(
        (feat_valid >= pd.Timestamp(TRAIN_BEGIN)) & (feat_valid <= pd.Timestamp(TRAIN_END)),
        index=feat_valid,
    )
    te_mask = pd.Series(
        (feat_valid >= pd.Timestamp(TEST_BEGIN)) & (feat_valid <= pd.Timestamp(TEST_END)),
        index=feat_valid,
    )
    y_tr, y_te = label.loc[tr_mask], label.loc[te_mask]
    X_tr, X_te = feat.loc[tr_mask], feat.loc[te_mask]

    calib = pd.DataFrame({
        " ours": [int((y_tr == 0).sum()), int((y_tr == 1).sum()),
                  int((y_te == 0).sum()), int((y_te == 1).sum())],
        "研报": [REPORT_LABEL_RATIO["train_osc"], REPORT_LABEL_RATIO["train_trend"],
                 REPORT_LABEL_RATIO["test_osc"], REPORT_LABEL_RATIO["test_trend"]],
    }, index=["train_osc", "train_trend", "test_osc", "test_trend"])
    print("\n[校准] 标签段数（本实现 vs 研报混淆矩阵行和）")
    print(calib.to_string())

    feat_acc = pd.DataFrame({
        "本实现": {c: accuracy(feat[c], label, tr_mask) for c in FEATURES},
        "研报表8": REPORT_FEAT_ACC,
    })
    feat_acc["Δ"] = feat_acc["本实现"] - feat_acc["研报表8"]
    print("\n[校准] 单特征训练集准确度 vs 研报表 8")
    print(feat_acc.map(lambda v: f"{v:.4f}").to_string())
    calib_out = pd.concat([
        calib.assign(指标="标签段数"),
        feat_acc.map(lambda v: round(float(v), 4)).assign(指标="单特征训练准确度"),
    ])
    calib_out.to_csv(OUT / "calibration.csv", encoding="utf-8-sig")

    # ---------- 模型准确度 + 持续性基线（G1，研报表 11） ----------
    pred = fit_predict(X_tr, y_tr, X_te)
    full_mask = tr_mask | te_mask
    rows = []
    for name in ("equal", "logit", "tree4"):
        raw_p = pred[name]
        sm_p = smooth(pred[name])
        rows.append({"model": name, "smoothed": False,
                     "train": accuracy(raw_p, label, tr_mask), "test": accuracy(raw_p, label, te_mask)})
        rows.append({"model": name, "smoothed": True,
                     "train": accuracy(sm_p, label, tr_mask), "test": accuracy(sm_p, label, te_mask)})
    persist = label.shift(1)  # 持续性基线：昨天是趋势 ⇒ 今天也是
    rows.append({"model": "persistence", "smoothed": False,
                 "train": accuracy(persist, label, tr_mask), "test": accuracy(persist, label, te_mask)})
    acc_df = pd.DataFrame(rows)
    print("\n[G1] 模型准确度 vs 持续性基线（研报表 11 为校准靶）")
    print(acc_df.round(4).to_string(index=False))
    best_model = acc_df[(acc_df["model"] != "persistence") & acc_df["smoothed"]].set_index("model")["test"]
    persist_test = float(acc_df[(acc_df["model"] == "persistence")]["test"].iloc[0])
    g1 = {m: float(v) - persist_test for m, v in best_model.items()}
    print(f"[G1] 平滑后测试集准确率 − 持续性基线：{ {k: round(v, 4) for k, v in g1.items()} }"
          f"（基线 {persist_test:.4f}；判据 ≥ +3pp 且 ≥ 60%）")
    acc_df.to_csv(OUT / "model_accuracy.csv", index=False, encoding="utf-8-sig")

    coef = pred["logit_coef"]
    coef_cmp = pd.DataFrame({"本实现": coef, "研报表9": pd.Series(REPORT_LOGIT_COEF)}).round(3)
    print("\n[校准] 逻辑回归系数 vs 研报表 9")
    print(coef_cmp.to_string())
    coef_cmp.to_csv(OUT / "logit_coef.csv", encoding="utf-8-sig")

    # ---------- G2：周频 IC（周三采样，信号 T+1 可用） ----------
    wed_tr = [d for d in label.index[tr_mask] if d.weekday() == REBALANCE_WEEKDAY]
    wed_te = [d for d in label.index[te_mask] if d.weekday() == REBALANCE_WEEKDAY]
    ic_rows = []
    for name in ("equal", "logit", "tree4"):
        sig_d1 = smooth(pred[name]).shift(1)  # T+1 可用
        rho_tr, t_tr, n_tr = weekly_ic(raw["close"], sig_d1, pd.DatetimeIndex(wed_tr))
        rho_te, t_te, n_te = weekly_ic(raw["close"], sig_d1, pd.DatetimeIndex(wed_te))
        ic_rows.append({"model": name, "ic_train": rho_tr, "t_train": t_tr, "n_train": n_tr,
                        "ic_test": rho_te, "t_test": t_te, "n_test": n_te})
    ic_df = pd.DataFrame(ic_rows)
    print("\n[G2] 平滑信号 × 5 日前瞻收益 周频 Spearman IC（判据：测试窗 ρ ≥ 0.3 且 t ≥ 2）")
    print(ic_df.round(4).to_string(index=False))
    ic_df.to_csv(OUT / "weekly_ic.csv", index=False, encoding="utf-8-sig")

    # ---------- G3：T+1 完美标签上界 + 模型信号参考策略 ----------
    win_close = raw["close"].loc[TRAIN_BEGIN:TEST_END]
    win_ret = win_close.pct_change().fillna(0.0)
    ub_rows = []
    for cname, cbps in (("bp0", 0.0), ("bp10", 10.0), ("bp30", 30.0)):
        sig = label.shift(1).loc[win_close.index]  # 完美标签 T+1（乐观上界）
        r = run_strategy(win_close, sig, cbps)
        m = metrics_row(r["ret"])
        ub_rows.append({"variant": f"perfect_T1_{cname}", **m,
                        "avg_pos": r["avg_pos"], "turnover": r["turnover"]})
    r_same = run_strategy(  # 同日口径 = 研报「人工标注信号」口径（标签当日可用 ⇒ 前视）
        win_close, label.shift(0).loc[win_close.index], 0.0
    )
    ub_rows.append({"variant": "perfect_same_day_bp0", **metrics_row(r_same["ret"]),
                    "avg_pos": r_same["avg_pos"], "turnover": float("nan")})
    for name in ("equal", "logit", "tree4"):
        sig = smooth(pred[name]).shift(1).loc[win_close.index]
        r = run_strategy(win_close, sig, 10.0)
        m = metrics_row(r["ret"])
        ub_rows.append({"variant": f"{name}_bp10", **m, "avg_pos": r["avg_pos"], "turnover": r["turnover"]})
    # 研报 §3.1 两只不依赖标签的对照：永远趋势逻辑 / 永远震荡逻辑
    for vname, sval in (("pure_momentum_bp0", 1.0), ("pure_reversal_bp0", 0.0)):
        r = run_strategy(win_close, pd.Series(sval, index=win_close.index), 0.0)
        ub_rows.append({"variant": vname, **metrics_row(r["ret"]),
                        "avg_pos": r["avg_pos"], "turnover": float("nan")})
    # 真基准：0.5 / 1.0 仓买入持有（无调仓）
    for bname, w in (("bench_half_bh", 0.5), ("bench_full_bh", 1.0)):
        m = metrics_row(win_ret * w)
        ub_rows.append({"variant": bname, **m, "avg_pos": w, "turnover": 0.0})
    ub_df = pd.DataFrame(ub_rows)
    print("\n[G3] 策略层（2020/1/2–2025/8/29，研报仓位规则）")
    print(ub_df.round(4).to_string(index=False))
    ub_df.to_csv(OUT / "upper_bound.csv", index=False, encoding="utf-8-sig")

    # ---------- 判定 ----------
    pl = ub_df[ub_df["variant"] == "perfect_T1_bp10"].iloc[0]
    bh = ub_df[ub_df["variant"] == "bench_half_bh"].iloc[0]
    g3_pass = bool(pl["annual"] > bh["annual"] and pl["sharpe"] > bh["sharpe"])
    g1_pass = any(v >= 0.03 and best_model[m] >= 0.60 for m, v in g1.items())
    g2_pass = bool(((ic_df["ic_test"] >= 0.3) & (ic_df["t_test"] >= 2.0)).any())
    print("\n===== 门槛判定（判据先于数据预注册） =====")
    print(f"G1 可预测性  : {'PASS' if g1_pass else 'FAIL'}  （平滑后测试准确率 − 持续性基线 ≥ +3pp 且 ≥ 60%）")
    print(f"G2 经济相关性: {'PASS' if g2_pass else 'FAIL'}  （测试窗周频 IC ≥ 0.3 且 t ≥ 2）")
    print(f"G3 上界可实现: {'PASS' if g3_pass else 'FAIL'}  （完美标签 T+1 + 10bp：年化 > 半仓基准且 Sharpe 更高）")
    verdict = pd.DataFrame({
        "gate": ["G1", "G2", "G3"],
        "pass": [g1_pass, g2_pass, g3_pass],
        "判据": ["平滑后测试准确率−持续性基线≥+3pp且≥60%", "测试窗周频IC≥0.3且t≥2",
                 "完美标签T+1+10bp 年化>半仓基准且Sharpe更高"],
    })
    verdict.to_csv(OUT / "gates.csv", index=False, encoding="utf-8-sig")
    all_pass = g1_pass and g2_pass and g3_pass
    print("\n结论：", "三门全过 → 进策略回测（三模型 + 敏感性 + 多窗口）" if all_pass
          else "至少一门 FAIL → 探针收口，不进策略回测")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
