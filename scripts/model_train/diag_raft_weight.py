"""RAFT 式样本加权在全A 主模型上的裁决探针。

**问题**：行业轮动线上验证过的 RAFT 思路（检索历史相似片段决定训练样本），
改成全A 选股主模型可接受的形态——**不改窗口、不碎片化，只给训练样本加权**——
能否提升折外 IC。

**为什么是加权而不是选窗**：选窗会减少并碎片化样本（全A 每折已 200 万行，
覆盖面是它的优势）；加权保留全部样本、零推理成本、不增加自由度。

**三族臂**（同一特征集 / 同一 fold / 同一标签，只换 ``sample_weight``）：

- ``equal``：等权（现状，500 日窗）
- ``time`` ：时间指数衰减 ``0.5**((T-1-i)/H)``——**关键对照**。因为「用当期
  状态检索」天然偏向近期样本（状态缓慢演化时 d_t 单调随距今上升），
  若 sim 的效果 ≈ time，则「相似度」没有独立于「时间」的信息，
  这条线就退化成「调时间衰减率」，不是 RAFT。
- ``sim*`` ：状态相似度 ``exp(-λ·d/median(d))``，d = 训练窗内各日状态到
  **折起始状态**（= 训练窗最后一天，PIT 干净）的标准化欧氏距离。

**判据（预注册，写进报告前不得改）**：
配对日 IC 差（vs ``equal``）NW t ≥ 2.0，且分年方向一致 ≥ 2/3 年份，
且增益不能仅由「平均样本量变化」解释（权重均值恒归一到 1）。

用法：
    python -m scripts.model_train.diag_raft_weight --years 2024 --quick
    python -m scripts.model_train.diag_raft_weight --years 2022,2023,2024,2025
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor.cv import forward_roll_folds
from model.labels import build_labels
from model.predictor import _long_matrix
from scripts.common.alla_common import BASE_DIR, FeatureStore, ds_root, load_base

log = logging.getLogger("raft_weight")

SEL_DIR = Path("reports/alla_rolling_ortho920fundind4/selection")
OUT_DIR = Path("reports/raft_weight")

STATE_COLS = ["mom20", "vol20", "disp20", "sizesp", "breadth"]


# ---------------------------------------------------------------------------
# 状态向量（严格 PIT：rolling 标准化，不用全样本均值/方差）
# ---------------------------------------------------------------------------
def build_state(close: pd.DataFrame, mcap: pd.DataFrame,
                win_z: int = 500) -> pd.DataFrame:
    """五个市场状态量，均在 t 日收盘可知。返回滚动 zscore 后的面板。"""
    r1 = close.pct_change(fill_method=None)
    ew = r1.mean(axis=1)
    mom20 = np.log1p(ew).rolling(20).sum()
    vol20 = ew.rolling(20).std() * np.sqrt(252)
    disp20 = r1.std(axis=1).rolling(20).mean()

    rk = mcap.reindex_like(close).rank(axis=1, pct=True)
    small = r1.where(rk <= 0.3).mean(axis=1)
    big = r1.where(rk >= 0.7).mean(axis=1)
    sizesp = (small - big).rolling(20).sum()
    breadth = (r1 > 0).mean(axis=1).rolling(20).mean()

    s = pd.DataFrame({"mom20": mom20, "vol20": vol20, "disp20": disp20,
                      "sizesp": sizesp, "breadth": breadth})
    # PIT 标准化：只用过去 win_z 日
    mu = s.rolling(win_z, min_periods=120).mean()
    sd = s.rolling(win_z, min_periods=120).std()
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------
def day_weights(kind: str, state_tr: pd.DataFrame, halflife: float) -> np.ndarray:
    """训练窗**逐日**权重（长度 = 训练日数），均值归一到 1。

    ``state_tr`` 末行即折起始前最后一天（= 当期状态，PIT 干净）。
    """
    n = len(state_tr)
    if kind == "equal":
        w = np.ones(n)
    elif kind == "time":
        w = 0.5 ** ((n - 1 - np.arange(n)) / float(halflife))
    elif kind.startswith("sim"):
        lam = float(kind[3:])
        ref = state_tr.iloc[-1].values.astype(float)
        cur = state_tr.values.astype(float)
        d = np.sqrt(np.nansum((cur - ref[None, :]) ** 2, axis=1))
        d = np.where(np.isfinite(d), d, np.nan)
        med = np.nanmedian(d)
        med = med if med and med > 0 else 1.0
        d = np.where(np.isfinite(d), d, med)
        w = np.exp(-lam * d / med)
    else:
        raise ValueError(f"未知臂 {kind!r}")
    return w / w.mean()


# ---------------------------------------------------------------------------
# 评价
# ---------------------------------------------------------------------------
def daily_ic(pred: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    v = pred.notna() & fwd.notna() & mask.reindex_like(pred).fillna(False).astype(bool)
    pr = pred.where(v).rank(axis=1)
    rr = fwd.where(v).rank(axis=1)
    s = pr.corrwith(rr, axis=1)
    return s[v.sum(axis=1) >= 100].dropna()


def nw_t(x: np.ndarray, lag: int = 10) -> float:
    """Newey-West t（均值是否为 0）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = float(e @ e) / n
    for l in range(1, lag + 1):
        if l >= n:
            break
        g = float(e[l:] @ e[:-l]) / n
        s += 2.0 * (1.0 - l / (lag + 1.0)) * g
    s = max(s, 1e-18)
    return float(x.mean() / np.sqrt(s / n))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(years: list[int], n_folds: int, max_train: int | None,
        arms: list[str], halflife: float, n_feat: int | None,
        out_dir: Path) -> pd.DataFrame:
    from lightgbm import LGBMRegressor
    from model.params import DEFAULT_MODEL_PARAMS

    base = load_base()
    close, mask = base["close"], base["mask"]
    all_days = close.index

    labels, _ = build_labels(close, horizon=1, mode="rank", tradable_mask=mask)
    fwd = close.pct_change(1, fill_method=None).shift(-1)
    state = build_state(close, base["market_cap"])
    cols = labels.columns

    store = FeatureStore(ds_root() / "panels")
    params = dict(DEFAULT_MODEL_PARAMS["gbdt"])

    ic_all: dict[str, list[pd.Series]] = {a: [] for a in arms}
    fold_log = []

    for year in years:
        sel = SEL_DIR / f"y{year}__h1.json"
        feats = json.loads(sel.read_text(encoding="utf-8"))
        if n_feat:
            feats = feats[:n_feat]
        names = sorted(feats)
        F = store.get_many(feats)
        test_days = all_days[(all_days >= pd.Timestamp(f"{year}-01-01"))
                             & (all_days <= pd.Timestamp(f"{year}-12-31"))]
        folds = forward_roll_folds(all_days, test_days, n_folds, embargo_days=1)
        log.info("y%d: %d 特征, %d 折", year, len(names), len(folds))

        for fi, fold in enumerate(folds):
            tr = fold.train_days
            if max_train:
                tr = tr[-max_train:]
            if len(tr) < 120:
                log.warning("  fold%d 训练段不足 %d，跳过", fi, len(tr))
                continue

            t0 = time.time()
            X = _long_matrix(F, names, tr, cols)
            y = labels.loc[tr].values.ravel()
            ok = np.isfinite(y)
            X, y = X[ok], y[ok]
            day_of_row = np.repeat(np.arange(len(tr)), len(cols))[ok]

            st = state.reindex(tr)
            st = st.ffill().bfill()
            Xte = _long_matrix(F, names, fold.test_days, cols)

            preds = {}
            for arm in arms:
                wd = day_weights(arm, st, halflife)
                wr = wd[day_of_row]
                m = LGBMRegressor(**params, verbose=-1)
                m.fit(X, y, sample_weight=wr)
                p = m.predict(Xte).reshape(len(fold.test_days), len(cols))
                preds[arm] = pd.DataFrame(p, index=fold.test_days, columns=cols)

            del X, y
            for arm in arms:
                ic_all[arm].append(daily_ic(preds[arm], fwd, mask))

            icf = {a: ic_all[a][-1].mean() for a in arms}
            fold_log.append({"year": year, "fold": fi,
                             "train_days": len(tr), "n_rows": int(ok.sum()),
                             "test_days": len(fold.test_days),
                             **{f"ic_{a}": icf[a] for a in arms},
                             "sec": round(time.time() - t0, 1)})
            log.info("  fold%d train %d 日 / %.0f 万行 / %d 测试日 %.0fs  %s",
                     fi, len(tr), ok.sum() / 1e4, len(fold.test_days),
                     time.time() - t0,
                     "  ".join(f"{a}:{icf[a]:+.4f}" for a in arms))
            del Xte, preds

    # ---- 汇总 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    fl = pd.DataFrame(fold_log)
    fl.to_csv(out_dir / "folds.csv", index=False, encoding="utf-8-sig")

    icdf = pd.DataFrame({a: pd.concat(ic_all[a]).sort_index() for a in arms})
    icdf.to_parquet(out_dir / "daily_ic.parquet")

    rows = []
    base_ic = icdf["equal"]
    for a in arms:
        s = icdf[a]
        d = s - base_ic
        rows.append({"arm": a, "ic_mean": s.mean(),
                     "ic_ir": s.mean() / s.std(),
                     "ic_win": float((s > 0).mean()),
                     "n_days": int(s.notna().sum()),
                     "d_ic_vs_equal": d.mean(), "t_nw": nw_t(d.values),
                     "d_win": float((d > 0).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    log.info("\n%s", res.round(4).to_string(index=False))
    yr = icdf.groupby(icdf.index.year).mean()
    log.info("\n逐年 IC：\n%s", yr.round(4).to_string())
    log.info("\n逐年 ΔIC (vs equal)：\n%s",
             yr.sub(yr["equal"], axis=0).round(4).to_string())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="RAFT 式样本加权裁决探针（全A 主模型）")
    ap.add_argument("--years", default="2024",
                    help="测试年，逗号分隔")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=500)
    ap.add_argument("--arms", default="equal,time,sim0.5,sim1,sim2")
    ap.add_argument("--halflife", type=float, default=125.0,
                    help="time 臂半衰期（交易日）")
    ap.add_argument("--n-feat", type=int, default=None,
                    help="只用前 n 个特征（冒烟用）")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    years = [int(x) for x in args.years.split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    res = run(years, args.n_folds, args.max_train, arms, args.halflife,
              args.n_feat, Path(args.out))
    return 0 if len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
