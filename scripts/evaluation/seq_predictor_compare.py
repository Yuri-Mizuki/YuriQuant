"""P1 —— 时序预测器对照：序列归纳偏置 vs 截面/树归纳偏置。

科学问题
--------
项目全部预测模型（gbdt / LambdaRank / ridge / tabicl / tabpfn）都是**截面拟合器**：
输入是「某一天的因子取值」，对时间维零归纳偏置。而 LSTM/Transformer 的代码虽然存在，
却只用于**因子搜索的策略网络**（``factor/rl/alphapool_nets.py``、``factor/gflownet/net.py``），
**从未做过收益预测**。``factor/alpha158.py`` 自述「Alpha360 零加工序列适合
LSTM/Transformer 输入」，该实现从未落地。

本脚本做**唯一没验证过的归纳偏置对照**：给所有臂**完全相同的输入张量**，
只让模型架构不同 —— 差异只来自归纳偏置。

臂
--
- ``gbdt_raw``      : LightGBM on 展平的 alpha360（360 维）—— 截面/树归纳偏置（基线）
- ``gru``           : GRU on (T, 6) 序列 —— 时序归纳偏置
- ``transformer``   : TransformerEncoder on (T, 6) —— 时序归纳偏置（注意力）
- ``gbdt_a158``     : LightGBM on alpha158（113 工程化因子）—— 参考：工程化特征有多强
- ``equal_weight``  : 等权合成 —— 无模型基线
- 每族跑两个 seed（``*_s0`` / ``*_s1``）→ **自带噪声地板**，判据不靠单点。

口径
----
- 池：**HS300 PIT 成分并集**（``index_constituent_000300_SH``，逐日 mask，无幸存者偏差）。
- 输入：``daily_all_a`` + ``backward_factor``（后复权）→ AlphaData → alpha360/alpha158。
- 标签：``build_labels(close_adj, horizon, mode="rank")``，embargo = horizon。
- 切分：``forward_roll_folds``（生产唯一合法前推纪律），折前全部历史 − embargo。
- 指标：``evaluate_model`` 的 rank IC 均值 / ICIR / NW t 值。

用法
----
    python scripts/evaluation/seq_predictor_compare.py --smoke
    python scripts/evaluation/seq_predictor_compare.py --begin 20180101 --end 20260831 \
        --horizon 5 --n-folds 4 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("seq_predictor_compare")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")

DATA = Path("E:/data/parquet")
OUT_DIR = ROOT / "reports" / "p1_sequence_predictor"
FIELDS = ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME"]
FIELDS_LOWER = ["open", "high", "low", "close", "volume", "amount"]


# ---------------------------------------------------------------------------
# 1. 数据：直连 parquet 自建后复权 PIT 面板
# ---------------------------------------------------------------------------
def _build_px_panels(index_code: str, warmup: int, end: int):
    """从本地 parquet 直建后复权 OHLCV/PIT 面板（对齐 load_alpha_panels 口径）。"""
    cons = pd.read_parquet(DATA / f"index_constituent_{index_code.replace('.', '_')}.parquet")
    codes = sorted(cons["con_code"].unique().tolist())

    d0, d1 = pd.Timestamp(str(warmup)), pd.Timestamp(str(end))
    daily = pd.read_parquet(DATA / "daily_all_a.parquet")
    idx_dt = daily.index.get_level_values(0)
    daily = daily[(idx_dt >= d0) & (idx_dt <= d1)]
    daily = daily[daily.index.get_level_values(1).isin(codes)]
    dates = daily.index.get_level_values(0).unique().sort_values()

    # PIT 成分掩码：in_date <= t <= out_date（out_date 空 = 仍在册）
    mask = pd.DataFrame(False, index=dates, columns=codes)
    ind = pd.to_datetime(cons["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    outd = pd.to_datetime(cons["out_date"].astype(str), format="%Y%m%d", errors="coerce")
    for code, s, e in zip(cons["con_code"], ind, outd):
        if code not in mask.columns:
            continue
        ss = s if pd.notna(s) else dates[0]
        ee = e if pd.notna(e) else dates[-1]
        mask.loc[(dates >= ss) & (dates <= ee), code] = True

    panels: dict[str, pd.DataFrame] = {}
    for col in FIELDS_LOWER:
        p = daily[col].unstack("code").reindex(index=dates, columns=codes)
        panels[col] = p.where(mask)

    bf = pd.read_parquet(DATA / "backward_factor.parquet")
    b = bf.reindex(index=dates, columns=codes).ffill()
    for col in ("open", "high", "low", "close"):
        panels[col] = panels[col] * b
    vol = panels["volume"].replace(0, np.nan)
    panels["vwap"] = (panels["amount"] / vol) * b

    log.info("[px] codes=%d dates=%d (%s~%s) 每日在册中位=%.0f",
             len(codes), len(dates), dates[0].date(), dates[-1].date(),
             float(mask.sum(axis=1).median()))
    return panels, panels["close"]


def _cs_z(p: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 z-score（显式实现，口径 = 项目 standardize_zscore）。"""
    m = p.mean(axis=1)
    s = p.std(axis=1)
    s = s.where(s > 0, np.nan)
    return p.sub(m, axis=0).div(s, axis=0)


def build_arrays(index_code: str, warmup: int, end: int, n_lags: int):
    """构造 (arr360, flat158, close_adj, dates, codes)。"""
    from factor.alpha158 import compute_alpha158, compute_alpha360
    from factor.alpha_base import AlphaData

    panels, close_adj = _build_px_panels(index_code, warmup, end)
    d = AlphaData(panels)          # d 持有 panels 引用，末尾一并释放
    t0 = time.time()
    p360 = compute_alpha360(d)
    log.info("[a360] %d 面板 (%.0fs)", len(p360), time.time() - t0)
    t0 = time.time()
    p158 = compute_alpha158(d)
    log.info("[a158] %d 面板 (%.0fs)", len(p158), time.time() - t0)

    dates = close_adj.index
    codes = close_adj.columns
    n_d, n_c = len(dates), len(codes)
    arr = np.full((n_d, n_c, n_lags, len(FIELDS)), np.nan, dtype=np.float32)
    for fi, f in enumerate(FIELDS):
        for t in range(n_lags):
            name = f"alpha360_{f}{t}"
            if name not in p360:
                raise KeyError(f"缺少面板 {name}")
            v = _cs_z(p360.pop(name)).reindex(index=dates, columns=codes).values
            arr[:, :, n_lags - 1 - t, fi] = v.astype(np.float32)   # 0=最早，T-1=最新
    del p360

    names158 = sorted(p158)
    flat158 = np.full((n_d, n_c, len(names158)), np.nan, dtype=np.float32)
    for j, name in enumerate(names158):
        v = _cs_z(p158.pop(name)).reindex(index=dates, columns=codes).values
        flat158[:, :, j] = v.astype(np.float32)
    del p158, d, panels
    log.info("[tensor] arr=%s (%.1f GB) flat158=%s", arr.shape,
             arr.nbytes / 1e9, flat158.shape)
    return arr, flat158, close_adj, dates, codes


# ---------------------------------------------------------------------------
# 2. 时序模型
# ---------------------------------------------------------------------------
class SeqArm:
    """GRU / Transformer 序列预测器（同输入张量，仅架构不同）。"""

    def __init__(self, arch: str = "gru", hidden: int = 64, epochs: int = 20,
                 lr: float = 1e-3, batch: int = 8192, seed: int = 0,
                 n_lags: int = 60, threads: int = 8, time_perm: bool = False):
        self.arch = arch
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed
        self.n_lags = n_lags
        self.threads = threads
        # 决定性控制：把时间轴按固定置换打乱。若打乱后仍与未打乱同强，
        # 说明增益来自「模型容量」而非「时序结构」。
        self.perm = (np.random.RandomState(seed + 12345).permutation(n_lags)
                     if time_perm else None)
        self.net = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = self._permute(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        if self.arch == "mlp":
            X = X.reshape(len(X), -1)          # 无时序：直接展平
        return X

    def _permute(self, X: np.ndarray) -> np.ndarray:
        return X if self.perm is None else X[:, self.perm, :]

    def _make_net(self, f_dim: int):
        import torch.nn as nn

        arch, hidden = self.arch, self.hidden

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.arch = arch
                if arch == "mlp":
                    self.mlp = nn.Sequential(
                        nn.Linear(f_dim, hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, 1))
                elif arch == "gru":
                    self.rnn = nn.GRU(f_dim, hidden, batch_first=True, num_layers=1)
                elif arch == "transformer":
                    self.inp = nn.Linear(f_dim, hidden)
                    layer = nn.TransformerEncoderLayer(
                        d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                        dropout=0.1, batch_first=True, activation="gelu",
                        norm_first=False)          # False 才启用 nested-tensor 快路径
                    self.enc = nn.TransformerEncoder(layer, num_layers=1)
                else:
                    raise ValueError(f"未知 arch={arch!r}")
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):                       # x: (B, T, F) 或 (B, T*F)
                if self.arch == "mlp":
                    return self.mlp(x).squeeze(-1)
                if self.arch == "gru":
                    out, _ = self.rnn(x)
                    h = out[:, -1, :]
                else:
                    out = self.enc(self.inp(x))
                    h = out[:, -1, :]
                return self.head(h).squeeze(-1)

        return _Net()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeqArm":
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        X = self._prep(X)
        self.net = self._make_net(X.shape[-1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss()
        n = len(X)
        xt = torch.from_numpy(X)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.net.train()
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for s in range(0, n, self.batch):
                b = perm[s:s + self.batch]
                opt.zero_grad()
                out = self.net(xt[b])
                loss = lossf(out, yt[b])
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(b)
            if (ep + 1) % max(1, self.epochs // 4) == 0:
                log.info("    [%s_s%d] epoch %d/%d loss=%.5f",
                         self.arch, self.seed, ep + 1, self.epochs, tot / n)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = self._prep(X)
        self.net.eval()
        out = np.full(len(X), np.nan, dtype=np.float64)
        with torch.no_grad():
            xt = torch.from_numpy(X)
            for s in range(0, len(X), self.batch * 4):
                e = min(s + self.batch * 4, len(X))
                out[s:e] = self.net(xt[s:e]).numpy()
        return out


class GbdtArm:
    """LightGBM 截面回归（展平输入）。"""

    def __init__(self, seed: int = 0, n_estimators: int = 300, learning_rate: float = 0.03,
                 num_leaves: int = 15, min_child_samples: int = 50, threads: int = 8):
        self.seed = seed
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           n_jobs=threads, random_state=seed, verbose=-1)
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GbdtArm":
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# 3. 滚动评估
# ---------------------------------------------------------------------------
def _row_idx(day_positions: np.ndarray, n_codes: int) -> np.ndarray:
    """(date, code) 在 reshape(-1, ...) 后的行号 = date_pos * n_codes + code_pos。"""
    return (day_positions[:, None] * n_codes + np.arange(n_codes)[None, :]).ravel()


def _panel_from_vec(vec: np.ndarray, days: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(vec.reshape(len(days), len(codes)), index=days, columns=codes)


def run(args):
    from factor.cv import forward_roll_folds
    from model.evaluation import evaluate_model
    from model.labels import build_labels, forward_returns

    arr, flat158, close_adj, dates, codes = build_arrays(
        args.index, args.warmup, args.end, args.n_lags)

    b0, e0 = pd.Timestamp(str(args.begin)), pd.Timestamp(str(args.end))
    keep = (dates >= b0) & (dates <= e0)
    arr, flat158, close_adj = arr[keep], flat158[keep], close_adj.loc[dates[keep]]
    dates = dates[keep]
    log.info("[slice] %s~%s %d 日 × %d 股", dates[0].date(), dates[-1].date(),
             len(dates), len(codes))

    labels, embargo = build_labels(close_adj, horizon=args.horizon, mode="rank")
    fwd = forward_returns(close_adj, horizon=args.horizon)
    log.info("[label] horizon=%dd embargo=%dd", args.horizon, embargo)

    n = len(dates)
    n_test = max(120, int(n * args.test_frac))
    test_days = dates[-n_test:]
    folds = forward_roll_folds(dates, test_days, args.n_folds, embargo)
    log.info("[cv] test %s~%s，%d 折", test_days[0].date(), test_days[-1].date(), len(folds))

    pos = {d: i for i, d in enumerate(dates)}
    # (day, code) 是否在册（非在册 = 全 NaN 输入，训练必须剔除，否则 65% 算力浪费在空行）
    valid_2d = close_adj.notna().values
    arms = _build_arms(args)
    preds: dict[str, pd.DataFrame] = {k: pd.DataFrame(np.nan, index=test_days, columns=codes)
                                      for k in arms}
    arr_flat = arr.reshape(-1, args.n_lags, len(FIELDS))     # (n_d*n_c, T, F) 视图，零拷贝
    f158_flat = flat158.reshape(-1, flat158.shape[-1])
    n_codes = len(codes)

    for fi, fold in enumerate(folds, 1):
        tr_days, te_days = fold.train_days, fold.test_days
        if len(tr_days) < args.min_train_days or len(te_days) == 0:
            continue
        tri = np.array([pos[d] for d in tr_days])
        tei = np.array([pos[d] for d in te_days])
        y_tr = labels.loc[tr_days].values.ravel().astype(np.float32)
        ok = valid_2d[tri].ravel() & ~np.isnan(y_tr)
        rows_tr = _row_idx(tri, n_codes)[ok]
        rows_te = _row_idx(tei, n_codes)
        log.info("[fold %d/%d] train %s~%s (%d 日, 有效样本 %d/%d) -> test %s~%s",
                 fi, len(folds), tr_days[0].date(), tr_days[-1].date(), len(tr_days),
                 len(rows_tr), len(valid_2d[tri].ravel()), te_days[0].date(), te_days[-1].date())

        Xseq_tr, Xseq_te = arr_flat[rows_tr], arr_flat[rows_te]
        X158_tr, X158_te = f158_flat[rows_tr], f158_flat[rows_te]
        yv = y_tr[ok]
        for name, spec in arms.items():
            kind, kw = spec
            if kind == "gbdt360":
                m = GbdtArm(**kw).fit(Xseq_tr.reshape(len(rows_tr), -1), yv)
                p = m.predict(Xseq_te.reshape(len(rows_te), -1))
            elif kind == "gbdt158":
                m = GbdtArm(**kw).fit(X158_tr, yv)
                p = m.predict(X158_te)
            elif kind == "seq":
                m = SeqArm(n_lags=args.n_lags, **kw).fit(Xseq_tr, yv)
                p = m.predict(Xseq_te)
            elif kind == "eq":
                with np.errstate(all="ignore"):
                    p = np.nanmean(Xseq_te, axis=(1, 2))
                p = np.where(np.isfinite(p), p, np.nan)
            else:
                raise ValueError(kind)
            preds[name].loc[te_days] = _panel_from_vec(p, te_days, codes)
        del Xseq_tr, Xseq_te, X158_tr, X158_te

    rows = []
    for name, pred in preds.items():
        pz = _cs_z(pred)
        ev = evaluate_model(pz, fwd.loc[pred.index])
        rows.append({"arm": name, "ic_mean": ev["ic_mean"], "ic_ir": ev["ic_ir"],
                     "ic_t_nw": ev["ic_t_nw"], "n_days": int(ev["ic_series"].notna().sum())})
    table = pd.DataFrame(rows).sort_values("ic_mean", ascending=False)
    return table, preds, fwd, test_days, dates, codes


def _build_arms(args) -> dict:
    arms: dict = {}
    n_est = args.n_estimators
    # LightGBM 在本参数下确定性（无 bagging/feature_fraction）→ 只跑一个 seed
    arms["gbdt_raw"] = ("gbdt360", dict(seed=0, n_estimators=n_est, threads=args.threads))
    arms["gbdt_a158"] = ("gbdt158", dict(seed=0, n_estimators=n_est, threads=args.threads))
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    for arch in archs:
        perm = arch.endswith("_shuf")
        base = arch[:-5] if perm else arch
        seeds = (0, 1) if base in ("gru", "mlp") and not perm else (0,)
        for s in seeds:
            kw = dict(arch=base, seed=s, threads=args.threads, time_perm=perm)
            if base == "transformer":
                kw.update(hidden=args.t_hidden, epochs=args.t_epochs)
            else:
                kw.update(hidden=args.hidden, epochs=args.epochs)
            arms[arch if perm else f"{arch}_s{s}"] = ("seq", kw)
    if args.capacity_check:
        arms["gbdt_raw_big"] = ("gbdt360", dict(seed=0, n_estimators=1000,
                                                num_leaves=63, min_child_samples=20,
                                                threads=args.threads))
    arms["equal_weight"] = ("eq", {})
    return arms


def main():
    ap = argparse.ArgumentParser(description="P1 时序预测器对照（序列 vs 截面归纳偏置）")
    ap.add_argument("--index", default="000300.SH")
    ap.add_argument("--begin", type=int, default=20180101)
    ap.add_argument("--end", type=int, default=20260831)
    ap.add_argument("--warmup", type=int, default=20160601)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-lags", type=int, default=60)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--min-train-days", type=int, default=240)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--archs", default="gru,transformer")
    ap.add_argument("--capacity-check", action="store_true",
                    help="追加 gbdt_raw_big（1000 树 / 63 叶）验证「基线是否欠调」")
    ap.add_argument("--hidden", type=int, default=48, help="GRU 隐层")
    ap.add_argument("--epochs", type=int, default=30, help="GRU 训练轮数")
    ap.add_argument("--t-hidden", type=int, default=32, help="Transformer 隐层")
    ap.add_argument("--t-epochs", type=int, default=15, help="Transformer 训练轮数")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-tag", default="hs300_h5")
    ap.add_argument("--smoke", action="store_true", help="快速冒烟：短区间 + 少 epoch")
    args = ap.parse_args()

    if args.smoke:
        args.begin, args.end, args.warmup = 20230601, 20260831, 20220101
        args.n_folds, args.epochs, args.t_epochs, args.n_estimators = 1, 2, 2, 60
        args.out_tag = "smoke"

    t0 = time.time()
    table, preds, fwd, test_days, dates, codes = run(args)
    print(f"\n===== P1 对照表（test {test_days[0].date()}~{test_days[-1].date()}，"
          f"h={args.horizon}）=====")
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        print(table.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"arms_{args.out_tag}.csv", index=False)
    for name, pred in preds.items():
        _cs_z(pred).to_parquet(OUT_DIR / f"pred_{name}_{args.out_tag}.parquet")
    meta = {k: v for k, v in vars(args).items()}
    meta["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / f"meta_{args.out_tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("产物 -> %s（耗时 %.0fs）", OUT_DIR, time.time() - t0)


if __name__ == "__main__":
    main()
