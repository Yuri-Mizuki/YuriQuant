"""日内微观结构因子：增量信息体检（2026-11-15）。

背景
----
全A 主模型的预测器侧已系统测空（七类改动 + 混合权重 + 新 horizon），
增量只能来自**新信息源**，门槛是「自身可交易 IR ≥ 0.3」（TODO §四条提升路径）。

本地唯一「有数据但从没做过因子」的异源是 **5 分钟频**：`min5_hs300.parquet`
（2022-01 ~ 2026-09，421 只曾经在沪深300 的股票、每日 48 根 bar），
项目此前只用它查过涨停封板覆盖（3.02%）。

本脚本回答一个问题：**日内微观结构里还有没有、相对现有 920 因子模型仍然正交的信息？**

三层输出
--------
1. 每个日内因子的**日频 rank IC** 与 IC_IR（对 T+1 开盘→T+1 收盘的收益）；
2. **残差 IC**——把因子对当日模型预测做截面秩回归后，残差与未来收益的相关
   （这是「新信息」的严格判据：模型已经知道的部分不算）；
3. **十分位多空**（D10−D1 等权、日频）的**年化 Sharpe**，直接对照 IR 0.3 门槛。

⚠️ 样本是 HS300（每日约 417 只），不是全A。截面窄 ⇒ 因子强度会被低估，
结论只用于「值不值得为全A 分钟数据付采集/存储成本」这个二元判断。

用法::

    python -m scripts.evaluation.intraday_factor_probe
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN5 = "D:/data/parquet/min5_hs300.parquet"
PRED = "reports/alla_rolling_ortho920fundind/pred/gbdt__h1.parquet"
BASE = "reports/alla_rolling/_base"
#: 因子名 → 说明（全部在 t 日收盘时点可知）
FACTORS = {
    "iskew": "5 分钟收益偏度",
    "iac1": "5 分钟收益一阶自相关（反转强度）",
    "iclose30": "尾盘 30 分钟成交额占比",
    "iopen30": "开盘 30 分钟成交额占比",
    "ivwap": "收盘 / 日内 VWAP − 1",
    "ibig": "成交额最大 10% 的 bar 的成交额占比",
    "irvol": "日内已实现波动（5 分钟收益 std × √48）",
    "ilate": "尾盘 30 分钟收益",
    "iamih": "日内 Amihud：mean(|r| / 成交额)",
}


def build_intraday() -> dict[str, pd.DataFrame]:
    """从 5 分钟 bar 造日频日内因子，返回 {因子名: (date × code) 宽表}。"""
    d = pd.read_parquet(MIN5).reset_index()
    d["date"] = d["kline_time"].dt.normalize()
    d["bar"] = d["kline_time"].dt.strftime("%H:%M")

    dates = np.sort(d["date"].unique())
    codes = np.sort(d["code"].unique())
    bars = np.sort(d["bar"].unique())
    di = pd.Index(dates).get_indexer(d["date"])
    ci = pd.Index(codes).get_indexer(d["code"])
    bi = pd.Index(bars).get_indexer(d["bar"])

    shape = (len(dates), len(bars), len(codes))
    close = np.full(shape, np.nan, dtype=np.float32)
    amount = np.full(shape, np.nan, dtype=np.float32)
    vol = np.full(shape, np.nan, dtype=np.float32)
    close[di, bi, ci] = d["close"].to_numpy(np.float32)
    amount[di, bi, ci] = d["amount"].to_numpy(np.float32)
    vol[di, bi, ci] = d["volume"].to_numpy(np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        ret = np.log(close) - np.log(np.roll(close, 1, axis=1))
        ret[:, 0, :] = np.nan                       # 首根无前收
        ret = ret[:, 1:, :]                          # 只用可算的 47 根
        amt = amount[:, 1:, :]

        out: dict[str, np.ndarray] = {}
        # 偏度 / 标准差 / 一阶自相关
        mu = np.nanmean(ret, axis=1)
        sd = np.nanstd(ret, axis=1)
        z = (ret - mu[:, None, :]) / sd[:, None, :]
        out["iskew"] = np.nanmean(z**3, axis=1)
        out["irvol"] = sd * np.sqrt(ret.shape[1])
        a, b = ret[:, :-1, :], ret[:, 1:, :]
        am, bm = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
        acov = np.nanmean((a - am[:, None, :]) * (b - bm[:, None, :]), axis=1)
        out["iac1"] = acov / (np.nanstd(a, axis=1) * np.nanstd(b, axis=1))

        # 成交额结构（6 根 = 30 分钟）
        tot = np.nansum(amt, axis=1)
        out["iclose30"] = np.nansum(amt[:, -6:, :], axis=1) / tot
        out["iopen30"] = np.nansum(amt[:, :6, :], axis=1) / tot
        # 成交额前 10% 的 bar 占比（大单近似）
        thr = np.nanpercentile(amt, 90, axis=1)
        out["ibig"] = np.nansum(np.where(amt >= thr[:, None, :], amt, 0), axis=1) / tot
        # VWAP 偏离
        vwap = np.nansum(amount, axis=1) / np.nansum(vol, axis=1)
        out["ivwap"] = close[:, -1, :] / vwap - 1
        # 尾盘动量
        out["ilate"] = close[:, -1, :] / close[:, -7, :] - 1
        # 日内 Amihud
        out["iamih"] = np.nanmean(np.abs(ret) / amt, axis=1)

    frames = {}
    for name, arr in out.items():
        arr = np.where(np.isfinite(arr), arr, np.nan)
        frames[name] = pd.DataFrame(arr, index=dates, columns=codes, dtype="float32")
    return frames


def ic_stats(f: pd.DataFrame, r: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """逐日截面 rank IC。"""
    ok = f.notna() & r.notna() & mask
    return f.where(ok).rank(axis=1).corrwith(r.where(ok).rank(axis=1), axis=1).dropna()


def residual_ic(f: pd.DataFrame, p: pd.DataFrame, r: pd.DataFrame,
                mask: pd.DataFrame) -> pd.Series:
    """对模型预测做截面秩回归后的残差 IC。"""
    out = {}
    for dt in f.index:
        ok = f.loc[dt].notna() & p.loc[dt].notna() & r.loc[dt].notna() & mask.loc[dt]
        if ok.sum() < 100:
            continue
        x = p.loc[dt][ok].rank()
        y = f.loc[dt][ok].rank()
        xc, yc = x - x.mean(), y - y.mean()
        beta = (xc * yc).sum() / (xc * xc).sum()
        resid = yc - beta * xc
        out[dt] = resid.corr(r.loc[dt][ok].rank())
    return pd.Series(out).dropna()


def main() -> None:
    print("构建日内因子 …")
    frames = build_intraday()
    any_f = next(iter(frames.values()))
    print(f"  面板 {any_f.shape}  日期 {len(any_f)}  股票 {any_f.shape[1]}")

    close = pd.read_parquet(f"{BASE}/close_adj.parquet")
    openp = pd.read_parquet(f"{BASE}/open_adj.parquet")
    tradable = pd.read_parquet(f"{BASE}/tradable_mask.parquet")
    pred = pd.read_parquet(PRED)

    dates = any_f.index
    codes = any_f.columns
    # T+1 开盘 → T+1 收盘（与 S2 执行同相位）
    fwd = (close.shift(-1) / openp.shift(-1) - 1).reindex(index=dates, columns=codes)
    mask = tradable.reindex(index=dates, columns=codes).fillna(False).astype(bool)
    p = pred.reindex(index=dates, columns=codes)

    print(f"\n{'因子':10s} {'IC':>8s} {'IC_IR':>7s} {'t':>6s} {'残差IC':>8s} {'残差IR':>7s} {'LS Sharpe':>10s}")
    rows = []
    for name in FACTORS:
        f = frames[name]
        s = ic_stats(f, fwd, mask)
        rs = residual_ic(f, p, fwd, mask)
        # 十分位多空（D10 − D1，日频等权）
        ls = []
        for dt in f.index:
            ok = f.loc[dt].notna() & fwd.loc[dt].notna() & mask.loc[dt]
            if ok.sum() < 100:
                continue
            q = pd.qcut(f.loc[dt][ok].rank(method="first"), 10, labels=False)
            v = fwd.loc[dt][ok]
            ls.append(v[q == 9].mean() - v[q == 0].mean())
        ls = pd.Series(ls).dropna()
        sharpe = ls.mean() / ls.std() * np.sqrt(252) if len(ls) > 30 else np.nan
        rows.append((name, s.mean(), s.mean() / s.std(), s.mean() / s.std() * np.sqrt(len(s)),
                     rs.mean(), rs.mean() / rs.std(), sharpe))
        print(f"{name:10s} {s.mean():+8.4f} {s.mean()/s.std():+7.3f} "
              f"{s.mean()/s.std()*np.sqrt(len(s)):+6.2f} {rs.mean():+8.4f} "
              f"{rs.mean()/rs.std():+7.3f} {sharpe:+10.2f}")

    print("\n门槛参照：自建 IR ≥ 0.30 才准入；残差 IC 是「模型不知道的部分」。")
    print(f"有效日数约 {len(s)}，HS300 截面约 {int(mask.sum(axis=1).median())} 只。")


if __name__ == "__main__":
    main()
