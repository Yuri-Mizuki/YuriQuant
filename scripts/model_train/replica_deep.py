"""按研报 §2.1.1 训练 GRU / TCN 行业轮动深度学习因子（个股加权到行业）。

复现对象：国泰海通《七维共振》表 1 的 12 个配置，本脚本生成其中 **日频 8 个**
（`GRU_2_day60/day30` × `5d/10d`、`TCN_4_day60` / `TCN_3_day30` × `5d/10d`）。
`week30` 四个需要把日线重采样为周线，另由 `--freq week` 生成（见 RESEARCH 备注）。

严格照抄的六条（研报原文 → 本脚本）：
1. 输入 = 30/60 日行情，**6 个特征**：高、开、低、收、均价、换手率，去量纲+标准化
   → 实现为 `open_r, high_r, low_r, close_ret, vwap_r, turn`，再逐日截面 z-score。
   🚨 **去量纲的基准价研报没写**，2026-11-01 起四口径并列（`--descale`，见模块常量
   `DESCALES`）：`close`（÷当日收盘，**只剩日内形状、60 日窗口内没有跨日趋势信息**）、
   `prev`（÷前收，保留累计涨幅）、`avg`（÷均价）、`zscore`（个股自身时序标准化）。
   默认仍是 `close`，只为和历史产物同名；判断哪个口径对，看 `--descale` 对照臂。
2. 标签 = t+1 起的 5 日 / 10 日**均价**涨跌幅，标准化（逐日截面 z-score）
   → 分母取**当日收盘价**，不用 `avg[t]`（理由见 `forward_avg_ret` docstring）
3. 滚动训练 = 每年一个模型供下一年使用；训练集末尾按标签长度 purge
4. **损失 = 截面预测值与标签相关系数取负**
5. **batch = 一个交易日的股票截面**
6. GRU 2 层；TCN kernel=3，60 日 4 层 `[8,16,32,64]`、30 日 3 层 `[8,16,32]`

与研报的已知差异（引用必须带，见 README §四）：
- 数据起点 2015-01（`daily_all_a` 起点；研报 2008 起）⇒ 首个可训模型落在 2018 年，
  短 3 年以上
- 研报「前 9 年训练 + 第 10 年验证」；我们是「截至上年底全部历史训练 + 上一年验证」
  （数据不足 9 年，改成最小 2 年训练窗）
- 研报未公开 hidden_size / lr / epochs，本脚本取 hidden=64、lr=1e-3、早停
- 研报未写明**训练集是否按标签长度降采样**；本脚本默认 `--train-step h`（重叠标签去冗余，
  等于砍掉 90% 样本），`--train-step 1` 是判别这一点的对照臂
- 研报用中信一级，我们用申万一级（PIT）

用法：
    python -m scripts.model_train.replica_deep --smoke        # 单配置单年，测通与计时
    python -m scripts.model_train.replica_deep --configs all
    python -m scripts.model_train.replica_deep --configs TCN_4_day60_10d \
        --descale prev --tag dprev                            # 去量纲对照臂
    python -m scripts.model_train.replica_deep --configs TCN_4_day60_10d \
        --train-step 1 --tag step1                            # 降采样对照臂
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config  # noqa: E402
from model.seq_deep import DailyCrossSectionTrainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "report_replica"
DL_DIR = OUT_DIR / "dl"
PQ = Path(Config.paths()["cache_root"])

FEATS = ["open_r", "high_r", "low_r", "close_ret", "vwap_r", "turn"]
DATA_START = pd.Timestamp("2015-01-01")
MIN_TRAIN_DAYS = 400          # 最少训练交易日（≈1.6 年），不足则该年跳过
PURGE_EXTRA = 2               # 标签长度之外再间隔 2 天

# 去量纲口径（研报只写「去量纲+标准化」，未给基准价；2026-11-01 起四口径并列做实验）
#   close : p / 当日收盘 —— 只剩日内形状，60 日窗口内**没有跨日趋势信息**
#   prev  : p / 前收     —— 保留隔夜跳空与累计涨幅，是唯一「像行情」的口径（本次新增）
#   avg   : p / 当日均价 —— 相对 VWAP 的形状，第 5 个特征取倒数（close/avg）
#   zscore: 先按 prev 去量纲，再对每只股票做**自身**过去 60 日时序标准化
DESCALES = ("close", "prev", "avg", "zscore")
TS_WIN = 60                   # zscore 口径的时序窗口
TS_MIN_PERIODS = 20


# --------------------------------------------------------------------- 数据
def _cs_zscore(a: np.ndarray) -> np.ndarray:
    """逐日截面 z-score，NaN 保持 NaN。`a` 形状 (T, N)。"""
    m = np.isfinite(a)
    cnt = m.sum(axis=1)
    x = np.where(m, a, 0.0)
    mu = x.sum(axis=1) / np.maximum(cnt, 1)
    var = (((x - mu[:, None]) ** 2) * m).sum(axis=1) / np.maximum(cnt - 1, 1)
    sd = np.sqrt(var)
    z = (a - mu[:, None]) / np.where(sd > 1e-12, sd, np.nan)[:, None]
    return z.astype(np.float32)


def _close_panel(idx: pd.DatetimeIndex, codes: pd.Index) -> np.ndarray:
    """收盘价面板 (T,N)。只读一列，比重建 `arrays.npz` 便宜得多。"""
    d = pd.read_parquet(PQ / "daily_all_a.parquet", columns=["close"])
    w = d["close"].unstack("code").reindex(index=idx, columns=codes)
    return w.to_numpy(np.float32)


def build_arrays(cache: bool = True, descale: str = "close") -> dict:
    """构造特征 (T,N,6) 与均价面板；缓存到 `reports/report_replica/dl/arrays_<起始年>.npz`。

    Args:
        cache: 有缓存则直接读。
        descale: 去量纲口径，见 `DESCALES`。缓存里存的是**原始比值** `raw`
            （= `close` 口径、未截面 z-score），其余口径由它现场变换，不必重读行情。

    Returns:
        dict：`feat`（已按 `descale` 变换 + 逐日截面 z-score + 停牌掩码）、
        `raw`、`avg`、`close`、`index`、`codes`、`tradable`。
    """
    if descale not in DESCALES:
        raise ValueError(f"未知 descale={descale!r}，只支持 {DESCALES}")
    DL_DIR.mkdir(parents=True, exist_ok=True)
    fp = DL_DIR / f"arrays_{DATA_START.year}.npz"
    if cache and fp.exists():
        z = np.load(fp, allow_pickle=True)
        idx = pd.DatetimeIndex(z["index"])
        codes = pd.Index(z["codes"])
        # 旧缓存没有 close 键（2026-10-31 前生成），补读一列，避免整块重建
        close = z["close"] if "close" in z.files else _close_panel(idx, codes)
        if "raw" not in z.files:
            print("  缓存缺 raw（2026-11-01 前生成）⇒ 重建，否则无法切换去量纲口径")
            cache = False
        else:
            base = {"feat": z["feat"] if (descale == "close" and "feat" in z.files)
                    else None,
                    "raw": z["raw"], "avg": z["avg"], "index": idx, "codes": codes,
                    "tradable": z["tradable"], "close": close}
            if base["feat"] is None:
                base["feat"] = _features_from_raw(base["raw"], base["tradable"], descale)
            return base

    from scripts.builders.industry_context import float_share_panel

    t0 = time.time()
    d = pd.read_parquet(PQ / "daily_all_a.parquet",
                        columns=["open", "high", "low", "close", "volume", "amount"])
    idx = pd.DatetimeIndex(d.index.get_level_values(0).unique()).sort_values()
    idx = idx[idx >= DATA_START]
    codes = pd.Index(d.index.get_level_values(1).unique()).sort_values()

    def wide(col: str) -> np.ndarray:
        w = (d[col].unstack("code").reindex(index=idx, columns=codes))
        return w.to_numpy(np.float32)

    close = wide("close")
    close_raw = close.copy()                  # 标签分母用未填充的原始收盘（停牌即 NaN）
    opn, high, low = wide("open"), wide("high"), wide("low")
    vol, amt = wide("volume"), wide("amount")

    # 停牌：价格前向填充（停牌期价格不变），成交量置 0；当日无成交视为不可交易
    def ff(a: np.ndarray, limit: int = 20) -> np.ndarray:
        s = pd.DataFrame(a).ffill(limit=limit)
        return s.to_numpy(np.float32)

    close, opn, high, low = ff(close), ff(opn), ff(high), ff(low)
    tradable = np.isfinite(close) & (close > 0) & np.isfinite(vol) & (vol > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(vol > 0, amt / np.where(vol > 0, vol, 1.0), np.nan).astype(np.float32)
    avg = ff(avg)

    fs = float_share_panel(idx, codes).to_numpy(np.float32)     # 股数
    with np.errstate(divide="ignore", invalid="ignore"):
        turn = np.where(fs > 0, vol * 100.0 / fs, np.nan).astype(np.float32)

    prev_close = np.vstack([np.full((1, len(codes)), np.nan, np.float32), close[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.stack([
            opn / close, high / close, low / close,
            close / prev_close - 1.0,
            avg / close,
            turn,
        ], axis=-1).astype(np.float32)                            # (T,N,6)

    feat = _features_from_raw(raw, tradable, descale)
    feat_close = feat if descale == "close" else _features_from_raw(raw, tradable, "close")
    print(f"特征构造完成 feat={feat.shape} 口径={descale} 用时 {time.time() - t0:.1f}s "
          f"（{idx.min().date()}~{idx.max().date()}, {len(codes)} 只）")

    # 存 raw（未 z-score 的原始比值）+ close 口径的 feat；其余口径现场变换。
    # 用 savez 而非 savez_compressed：400MB 级 float32 压缩收益有限，读写慢 10 倍。
    np.savez(fp, raw=raw, feat=feat_close, avg=avg, close=close_raw, index=idx.values,
             codes=np.asarray(codes), tradable=tradable)
    return {"feat": feat, "raw": raw, "avg": avg, "index": idx, "codes": codes,
            "tradable": tradable, "close": close_raw}


def descale_features(raw: np.ndarray, descale: str) -> np.ndarray:
    """把「÷当日收盘」的 6 个原始比值变换到另一种去量纲口径（仍未 z-score）。

    列序：0 `open/close`、1 `high/close`、2 `low/close`、3 `close_ret`、
    4 `vwap/close`、5 `turn`。`close_ret` 与 `turn` 本身就是无量纲比值，四口径下
    一律不动 —— 变的只有 4 个价格特征（0/1/2/4）。

    变换用到的恒等式（全部可从缓存里的 raw 反解，不必重读行情）：
        close[t]/close[t-1] = 1 + raw[...,3]
        p/prev = (p/close) × (1 + raw[...,3])
        p/avg  = (p/close) ÷ raw[...,4]
    """
    if descale == "close":
        return raw
    out = np.array(raw, dtype=np.float32, copy=True)
    prev = 1.0 + raw[..., 3]                       # close[t] / close[t-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        if descale == "prev":
            for k in (0, 1, 2, 4):
                out[..., k] = raw[..., k] * prev
        elif descale == "avg":
            vw = raw[..., 4]                       # avg/close
            for k in (0, 1, 2):
                out[..., k] = raw[..., k] / vw
            out[..., 4] = 1.0 / vw                 # 第 5 个特征改为 close/avg
        elif descale == "zscore":
            for k in (0, 1, 2, 4):
                out[..., k] = raw[..., k] * prev
            out = _ts_zscore(out, TS_WIN, TS_MIN_PERIODS, cols=(0, 1, 2, 4))
        else:
            raise ValueError(f"未知 descale={descale!r}，只支持 {DESCALES}")
    return out.astype(np.float32)


def _ts_zscore(a: np.ndarray, win: int, min_periods: int,
               cols: tuple[int, ...]) -> np.ndarray:
    """逐股票、沿时间轴的滚动 z-score（只用过去 `win` 日，不含未来）。"""
    out = np.array(a, dtype=np.float32, copy=True)
    for k in cols:
        s = pd.DataFrame(a[:, :, k])
        mu = s.rolling(win, min_periods=min_periods).mean()
        sd = s.rolling(win, min_periods=min_periods).std(ddof=1)
        z = (s - mu) / sd.where(sd > 1e-12)
        out[:, :, k] = z.to_numpy(np.float32)
    return out


def _features_from_raw(raw: np.ndarray, tradable: np.ndarray, descale: str) -> np.ndarray:
    """原始比值 → 指定去量纲口径 → 逐日截面 z-score（研报「标准化」）→ 停牌掩码。"""
    x = descale_features(raw, descale)
    feat = np.stack([_cs_zscore(x[:, :, k]) for k in range(x.shape[-1])], axis=-1)
    feat[~tradable] = np.nan
    return feat


def forward_avg_ret(avg: np.ndarray, idx: pd.DatetimeIndex, h: int,
                    close: np.ndarray, *, zscore: bool = True) -> np.ndarray:
    """`mean(avg[t+1..t+h]) / close[t] - 1`，再逐日截面 z-score（研报「标准化」）。

    🚨 **分母必须是可成交价 `close[t]`，绝不能用 `avg[t]`**。研报原文只写「均价涨跌幅」
    未给分母，若照字面用 `avg[t]` 当分母，会与特征 `vwap/close = avg[t]/close[t]`
    共用同一个 `avg[t]`，产生纯机械相关：

        label ≈ (未来均价/close) - 1 - log(avg[t]/close[t]) = r - d
        ⇒ corr(-d, label) ≈ σ(d)/σ(r)

    实测 σ(d)=0.95%（日频截面）、σ(10 日收益)=5.33% ⇒ 机械相关上限 **0.178**。
    一个「只知道今日均价偏离收盘多少」的假预测就能拿到 IC 0.17，与 `close` 分母下
    的 −0.013 判若两个因子。换分母后 TCN 个股层 IC 从 0.198 降到 0.073（见
    `reports/report_replica/dl/README.md` §四）。
    """
    s = pd.DataFrame(avg, index=idx)
    fut = s[::-1].rolling(h, min_periods=h).mean()[::-1].shift(-1).to_numpy(np.float32)
    b = np.asarray(close, dtype=np.float32)
    b = np.where(b > 0, b, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = fut / b - 1.0
    return _cs_zscore(r) if zscore else r


def make_seq(feat: np.ndarray, seq_len: int) -> np.ndarray:
    """滑窗视图：`(T-seq+1, N, seq, F)`，第 t 行对应「窗口结束于原始第 t+seq-1 天」。"""
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.where(np.isfinite(feat), feat, 0.0).astype(np.float32)
    w = sliding_window_view(x, seq_len, axis=0)      # (T-seq+1, N, F, seq)
    return np.moveaxis(w, -1, 2)                      # (T-seq+1, N, seq, F) 仍是视图


# --------------------------------------------------------------------- 训练
CONFIGS = {
    # 名称按研报表 1 命名；层数/数据集/标签一一对应
    "GRU_2_day60_10d": dict(arch="gru", seq=60, h=10),
    "GRU_2_day60_5d": dict(arch="gru", seq=60, h=5),
    "GRU_2_day30_10d": dict(arch="gru", seq=30, h=10),
    "GRU_2_day30_5d": dict(arch="gru", seq=30, h=5),
    "TCN_4_day60_10d": dict(arch="tcn", seq=60, h=10),
    "TCN_4_day60_5d": dict(arch="tcn", seq=60, h=5),
    "TCN_3_day30_10d": dict(arch="tcn", seq=30, h=10),
    "TCN_3_day30_5d": dict(arch="tcn", seq=30, h=5),
}


def out_name(cfg_name: str, descale: str = "close", tag: str = "") -> str:
    """产物键：配置名 + `@去量纲口径`（非 close 才带）+ `__tag`。

    这样 `--descale prev` / `--tag step1` 的对照臂不会覆盖基线产物，
    `eval_dl_industry --configs all` 也会把它们当成独立配置各自评估。
    """
    key = cfg_name if descale == "close" else f"{cfg_name}@{descale}"
    return f"{key}__{tag}" if tag else key


def rolling_predict(name: str, cfg: dict, arr: dict, *, years: list[int],
                    epochs: int, patience: int, verbose: bool,
                    args_threads: int = 8, train_step: int = 0,
                    val_step: int = 5, device: str = "cpu",
                    seed: int = 3, ckpt: Path | None = None,
                    resume: bool = False) -> pd.DataFrame:
    """年度滚动：Y 年模型用 `<Y-1` 年训练、`Y-1` 年做验证集，预测 Y 年。

    Args:
        ckpt: 逐年落盘的检查点路径。长跑（多年 × 全样本训练，单年 20 分钟起）必须给，
            否则进程中途被杀 = 全部年份白跑（结果只在最后一次性写出）。
        resume: 从 `ckpt` 续跑：已有预测的年份直接跳过。判据是「该年测试窗内
            过半交易日有预测值」（单年可能有个别停牌/样本不足的空日）。
    """
    feat, idx = arr["feat"], arr["index"]
    seq_len, h = cfg["seq"], cfg["h"]

    lab = forward_avg_ret(arr["avg"], idx, h, arr["close"])       # (T,N) z-score
    seq = make_seq(feat, seq_len)                                # (T-seq+1,N,seq,F)
    end_days = idx[seq_len - 1:]                                 # 每个窗口的结束日
    years_end = end_days.year.to_numpy()

    out = np.full(feat.shape[:2], np.nan, dtype=np.float32)
    if resume and ckpt is not None and ckpt.exists():
        prev = pd.read_parquet(ckpt)
        keep_idx = idx.isin(prev.index)
        out[np.ix_(keep_idx, np.arange(len(arr["codes"])))] = prev.reindex(
            index=idx[keep_idx], columns=arr["codes"]).to_numpy(np.float32)
        print(f"  续跑：已载入 {ckpt.name}（覆盖 {int(np.isfinite(prev).any(axis=1).sum())} 天）")
    log_rows = []
    for yr in years:
        tr_m = years_end <= (yr - 2)
        va_m = years_end == (yr - 1)
        te_m = years_end == yr
        # 训练窗末尾 purge：标签要看到未来 h 天，去掉末尾 h+PURGE_EXTRA 个窗口
        cut = np.where(tr_m)[0]
        if len(cut) == 0:
            continue
        if te_m.sum():
            done = np.isfinite(out[seq_len - 1 + np.where(te_m)[0]]).any(axis=1)
            if resume and done.mean() > 0.5:
                print(f"  {yr}: 已有预测（{int(done.sum())}/{int(te_m.sum())} 天），跳过")
                continue
        keep = cut[cut <= cut[-1] - (h + PURGE_EXTRA)]
        n_raw = len(keep)
        if n_raw < MIN_TRAIN_DAYS or va_m.sum() == 0 or te_m.sum() == 0:
            print(f"  {yr}: 训练窗不足（{n_raw} 天），跳过")
            continue
        # 标签为 h 日重叠收益，相邻日样本高度冗余 ⇒ 每 train_step 天取一天。
        # 这是研报未写明、但对重叠标签必需的一步（不降采样等价于把同一条样本
        # 重复 h 次，早停会在验证集上提前触发）。train_step=0 表示取 h。
        step = int(train_step) if int(train_step) > 0 else h
        keep = keep[::max(step, 1)]
        tr_m = np.zeros_like(tr_m)
        tr_m[keep] = True

        n_tr = int(tr_m.sum())

        t0 = time.time()
        lab_s = lab[seq_len - 1:]                     # 与窗口位置一一对应
        tr = DailyCrossSectionTrainer(
            cfg["arch"], seq_len, feat.shape[-1], epochs=epochs, patience=patience,
            seed=int(seed), threads=int(args_threads), device=device,
            verbose=verbose)
        va_pos = np.where(va_m)[0]
        # 验证集的标签同样要看到未来 h 天；末 h+PURGE_EXTRA 天会越界到测试年，
        # 用于早停等于偷看测试期尾部 ⇒ 一并剔除。
        if len(va_pos) > h + PURGE_EXTRA:
            va_pos = va_pos[:-(h + PURGE_EXTRA)]
        va_pos = va_pos[::max(int(val_step), 1)]      # 再降采样，只为早停判据
        tr.fit(seq, lab_s, np.where(tr_m)[0], va_pos)
        te_pos, pred = tr.predict(seq, np.where(te_m)[0], lab_s)
        for k, p in enumerate(te_pos):
            out[seq_len - 1 + p] = pred[k]
        log_rows.append({"year": yr, "n_train": n_tr, "n_val": int(va_m.sum()),
                         "n_test": int(te_m.sum()), "best_epoch": tr.best_epoch,
                         "ic_val": tr.best_score, "sec": round(time.time() - t0, 1)})
        print(f"  {yr}: 训 {n_tr} 天 / 验 {int(va_m.sum())} 天 / 测 {int(te_m.sum())} 天 "
              f"| ver IC {tr.best_score:+.4f} @ep{tr.best_epoch} | {time.time() - t0:.0f}s")
        del tr
        if ckpt is not None:
            # 逐年落盘 + 同步日志：长跑被杀也能续（`--resume`）
            pd.DataFrame(out, index=idx, columns=arr["codes"]).to_parquet(ckpt)
            pd.DataFrame(log_rows).to_csv(DL_DIR / f"train_log_{name}.csv", index=False,
                                          encoding="utf-8-sig")

    if log_rows:      # 续跑时可能整轮都被跳过，别把已有日志覆盖成空文件
        pd.DataFrame(log_rows).to_csv(DL_DIR / f"train_log_{name}.csv", index=False,
                                      encoding="utf-8-sig")
    print(f"  [{name}] 训练完成，验证集最佳 IC 见 train_log_{name}.csv")
    return pd.DataFrame(out, index=idx, columns=arr["codes"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all",
                    help="逗号分隔的配置名，或 all")
    ap.add_argument("--years", default="auto",
                    help="逗号分隔的预测年份，默认按数据起点自动推断")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--start", default="2018")
    ap.add_argument("--smoke", action="store_true", help="只跑 1 个配置 1 年，测通+计时")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seeds", default="3", help="逗号分隔随机种子，默认研报的 seed 3")
    ap.add_argument("--train-step", type=int, default=0,
                    help="训练集每 N 天取 1 天（重叠标签去冗余）；0 = 取标签长度 h")
    ap.add_argument("--val-step", type=int, default=5,
                    help="验证集（仅早停用）每 N 天取 1 天")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--descale", default="close", choices=list(DESCALES),
                    help="去量纲口径；非 close 时产物名带 @口径 后缀，不会覆盖基线")
    ap.add_argument("--tag", default="",
                    help="额外产物后缀（同一配置跑多个对照臂时用，如 --tag step1）")
    ap.add_argument("--resume", action="store_true",
                    help="多年跑时从已有产物续跑（跳过已出预测的年份）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    arr = build_arrays(descale=args.descale)
    idx = arr["index"]
    names = list(CONFIGS) if args.configs == "all" else args.configs.split(",")

    if args.smoke:
        names = ["GRU_2_day60_10d"]
        years = [int(args.start)]
        args.epochs = min(args.epochs, 3)
    elif args.years == "auto":
        first = max(int(args.start), idx[-1].year - 1)
        years = list(range(int(args.start), idx[-1].year + 1))
        del first
    else:
        years = [int(y) for y in args.years.split(",")]

    DL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"配置 {len(names)} 个 × 年份 {years[0]}~{years[-1]} "
          f"(epochs={args.epochs}, patience={args.patience})")
    seeds = [int(x) for x in args.seeds.split(",")]
    for name in names:
        cfg = CONFIGS[name]
        key = out_name(name, args.descale, args.tag)
        for seed in seeds:
            print(f"\n[{key}] arch={cfg['arch']} seq={cfg['seq']} "
                  f"label={cfg['h']}d descale={args.descale} "
                  f"device={args.device} seed={seed} train_step={args.train_step or cfg['h']}")
            pred = rolling_predict(key, cfg, arr, years=years, epochs=args.epochs,
                                   patience=args.patience, verbose=args.verbose,
                                   args_threads=args.threads,
                                   train_step=args.train_step,
                                   val_step=args.val_step, device=args.device,
                                   seed=seed,
                                   ckpt=(DL_DIR / (f"pred_{key}.parquet" if len(seeds) == 1
                                                   else f"pred_{key}_s{seed}.parquet")
                                         if len(years) > 1 else None),
                                   resume=args.resume)
            fp = DL_DIR / (f"pred_{key}.parquet" if len(seeds) == 1
                           else f"pred_{key}_s{seed}.parquet")
            pred.to_parquet(fp)
            cov = pred.notna().any(axis=1)
            rng = (f"{pred.index[cov][0].date()}~{pred.index[cov][-1].date()}"
                   if cov.any() else "空")
            print(f"  → {fp.name} 覆盖 {int(cov.sum())} 天 ({rng})")
            (DL_DIR / f"meta_{key}.json").write_text(
                json.dumps({**cfg, "years": years, "epochs": args.epochs,
                            "patience": args.patience, "seed": seed,
                            "device": args.device, "descale": args.descale,
                            "train_step": args.train_step or cfg["h"],
                            "val_step": args.val_step,
                            "tag": args.tag}, ensure_ascii=False, indent=2),
                encoding="utf-8")


if __name__ == "__main__":
    main()
