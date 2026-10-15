"""多 horizon「前融合 vs 后融合」最小验证探针（P0，2026-10-14 立项）。

问题
----
现行生产定版信号 ``h1020`` = **四个独立训练的 gbdt**（各配自己的 horizon 标签）
预测的**截面秩平均** —— 即「**后融合**」（back fusion）。全项目从未试过
「**前融合**」（一个模型在**训练时**就见到全部 horizon）。
本探针做**受控最小验证**：同一训练窗、同一数据根、同一评分口径，只换融合方式。

臂（同一训练窗 / 同一 horizon 集合 / 同一模型超参）
---------------------------------------------------
===============  =========  ============================================
tag              融合        说明
===============  =========  ============================================
``indep_h1``     无         单周期 h1 gbdt（最弱单周期参照）
``indep_h20``    无         单周期 h20 gbdt（最强单周期参照）
``back_prod``    后融合      每周期用**自己的 50 特征** → 4 模型秩平均
                            = **现行生产形态**（零改动）
``back_union``   后融合      4 模型共用**并集特征** → 秩平均
                            （与 ``front_*`` 同输入，隔离「融合方式」的对照臂）
``front_pool``   前融合      1 模型，样本堆叠 + horizon token（同树预算 150）
``front_pool_x4``前融合      同 ``front_pool`` 但树数 ×4（容量对齐 4×150）
``front_labelmix``前融合     1 模型，标签 = 4 周期 rank 标签的**截面均值**
                            （标签层融合，训练侧看到多周期共识）
===============  =========  ============================================

为什么必须同时有 ``back_union``
-------------------------------
``back_prod`` 与 ``front_*`` 的**输入特征集不同**（50 vs 108）⇒ 直接比会把
「特征集差异」误读成「融合方式差异」。``back_union`` 把后者单独隔离出来。

口径（与 ``scripts/evaluation/caliber.py`` 同源）
-------------------------------------------------
- 训练：单一折，训练窗 = 测试起点前 500 交易日，去掉 embargo=20 日隔离带
  （> 最大 horizon，防 h20 标签前视进测试段）；
- 特征：``panels_neu``（882 ortho 线，``FeatureStore`` 同源加载：截面 zscore + ±10σ）；
- 标签：``model.labels.build_labels(mode="rank")``，评价口径 = ``pct_change(h).shift(-h)``；
- 组合：production profile（固定 10bp + T+1 开盘 + 100 万），Top10% 等权 / buffer(15/40）；
- 幽灵股守卫：``existence_mask``，**全部臂共用同一份掩码**（用并集特征算，防阈值口径漂移）。

判据
----
主：OOS 组合层年化（生产口径，``caliber.evaluate_strategy``）。
副：OOS rank IC（vs h1 前瞻收益 = 月频调仓的直接预测目标；另报 h20）。
⚠️ 最小验证的边界：单折（不做年内季度重训）、OOS 跨 2 年（第 2 年模型已陈旧，
对全部臂同等不利）。**只用于判「融合方式」的相对优劣，绝对数字不可引为定版。**

自建 ``rank_average`` 而非复用 ``rolling_grid_alla._rank_average``：跨模块
import 私有名被 ``tests/test_layering.py`` 守卫禁止（同口径两份实现是既有先例）。

用法::

    D:/python/Python312/python.exe scripts/evaluation/multihorizon_fusion_probe.py
    ... --test-begin 2021 --test-end 2022 --train-window 500
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

from config import Config  # noqa: E402
from model.labels import build_labels  # noqa: E402
from model.params import DEFAULT_MODEL_PARAMS  # noqa: E402
from model.predictor import LGBMPredictor  # noqa: E402
from scripts.evaluation import caliber  # noqa: E402
from scripts.pipelines.rolling_grid_alla import (  # noqa: E402
    DATASET,
    FeatureStore,
    existence_mask,
    load_base,
)
from stats.ic import calc_ic_series  # noqa: E402

log = logging.getLogger("mh_fusion")
HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
OUT_DEFAULT = "reports/multihorizon_fusion"


# ---------------------------------------------------------------------------
# 原语（与生产同口径；建模矩阵用 float32 + 位置切片省内存）
# ---------------------------------------------------------------------------
def lgb_kwargs(n_estimators_mult: int = 1) -> dict:
    """由 ``model.params.DEFAULT_MODEL_PARAMS['gbdt']`` **派生** LightGBM 原生参数。

    走 ``LGBMPredictor`` 的实例属性读取（而非再抄一份字面量），保证与生产
    预测器**同一真源**：``max_depth`` / ``n_jobs`` 等默认值也一并继承。
    """
    p = LGBMPredictor(**dict(DEFAULT_MODEL_PARAMS["gbdt"]))
    return dict(n_estimators=p.n_estimators * n_estimators_mult,
                learning_rate=p.learning_rate, num_leaves=p.num_leaves,
                max_depth=p.max_depth, min_child_samples=p.min_child_samples,
                n_jobs=p.n_jobs, random_state=p.seed, verbose=-1)


def stack_long(arrays: dict[str, np.ndarray], names: list[str],
               lo: int, hi: int) -> np.ndarray:
    """把 108 张 (n_days, n_codes) float32 数组按 [lo:hi) 行拼成 (n, n_feat) 长矩阵。

    行序 = 日期优先（与 ``model.predictor._long_matrix`` 的 ``from_product`` 约定一致）。
    ⚠️ 不能用 ``reindex``：那会为每张面板复制一次整个面板（108 × 1.3GB）。
    """
    n_rows = (hi - lo) * arrays[names[0]].shape[1]
    X = np.empty((n_rows, len(names)), dtype=np.float32)
    for j, n in enumerate(names):
        X[:, j] = arrays[n][lo:hi].ravel()
    return X


def rank_average(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """逐日截面秩平均（**全部面板都必须有值**才输出，同生产集成语义）。

    与 ``rolling_grid_alla._rank_average`` 同口径：有效代码取各面板交集、
    秩为 ``rank(pct=True)``。独立实现（跨模块 import 私有名被守卫禁止）。
    """
    out = pd.DataFrame(np.nan, index=panels[0].index, columns=panels[0].columns)
    vals = np.stack([p.rank(axis=1, pct=True).to_numpy(dtype=np.float32)
                     for p in panels])
    cnt = np.sum(~np.isnan(vals), axis=0)
    mean = np.nanmean(vals, axis=0)
    ok = cnt == len(panels)
    out.iloc[:] = np.where(ok, mean, np.nan)
    return out


def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                 n_estimators_mult: int = 1) -> np.ndarray:
    from lightgbm import LGBMRegressor
    ok = ~np.isnan(ytr)
    # ok 全 True 时**不做布尔索引**：Xtr[ok] 会整块复制（堆叠臂上是 3.4GB 副本）
    if ok.all():
        Xf, yf = Xtr, ytr
    else:
        Xf, yf = Xtr[ok], ytr[ok]
    m = LGBMRegressor(**lgb_kwargs(n_estimators_mult))
    m.fit(Xf, yf)
    pred = m.predict(Xte)
    del m
    return pred.astype(np.float32)


def _nanmean_stack(mats: list[np.ndarray]) -> np.ndarray:
    """按元素 nanmean（各 horizon 的 rank 标签求共识）。"""
    acc = np.zeros_like(mats[0], dtype=np.float32)
    cnt = np.zeros_like(mats[0], dtype=np.float32)
    for m in mats:
        good = ~np.isnan(m)
        acc += np.where(good, m, 0.0)
        cnt += good
    return np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan).astype(np.float32)


def _to_panel(vec: np.ndarray, idx: pd.Index, cols: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(vec.reshape(len(idx), len(cols)), index=idx, columns=cols)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selection-dir", default="reports/alla_rolling_ortho/selection")
    ap.add_argument("--panels", default="", help="面板目录（默认 ds_root()/panels_neu）")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--select-year", type=int, default=2021,
                    help="用哪一年的 selection（须 <= 测试起点年）")
    ap.add_argument("--test-begin", type=int, default=2021)
    ap.add_argument("--test-end", type=int, default=2022)
    ap.add_argument("--train-window", type=int, default=500)
    ap.add_argument("--buffer", default="0.15:0.40")
    ap.add_argument("--arms", default="",
                    help="只跑指定臂（逗号分隔，默认全跑）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    dest = ROOT / args.out
    (dest / "pred").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = load_base()
    close = base["close"]
    panels_dir = (Path(args.panels) if args.panels
                  else Path(str(Config.get()["factor_library"]["root"])) / DATASET
                  / "panels_neu")

    # ---- 1) 特征选择清单（PIT：只读测试起点**之前**选出的清单） ----
    sel: dict[int, list[str]] = {}
    for h in HORIZONS:
        p = ROOT / args.selection_dir / f"y{args.select_year}__h{h}.json"
        if not p.exists():
            raise FileNotFoundError(f"缺 selection：{p}")
        sel[h] = json.loads(p.read_text(encoding="utf-8"))
    union_names = sorted(set().union(*sel.values()))
    log.info("特征集：每周期 50 / 并集 %d（select_year=%d，PIT 合规）",
             len(union_names), args.select_year)

    # ---- 2) 训练/测试日历与列（一次性对齐） ----
    test_lo = pd.Timestamp(f"{args.test_begin}-01-01")
    test_hi = pd.Timestamp(f"{args.test_end}-12-31")
    test_days = close.index[(close.index >= test_lo) & (close.index <= test_hi)]
    # 信号面板可能比行情面板短（如 882 pred 止 2026-09-07 而行情到 10-08）：
    # 在构建任何特征/臂面板**之前**裁剪到交集，保证全链路日历一致。
    _pred_path = ROOT / "reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet"
    _pred_idx = pd.DatetimeIndex(pd.read_parquet(_pred_path).index)
    _n0 = len(test_days)
    test_days = test_days[test_days.isin(_pred_idx)]
    if len(test_days) != _n0:
        log.warning("测试日历裁剪到信号面板交集：%d → %d 日（止 %s）",
                    _n0, len(test_days), test_days[-1].date())
    tr_all = close.index[close.index < test_lo]
    tr_days = tr_all[-(args.train_window + max(HORIZONS)):][:-max(HORIZONS)]
    if len(tr_days) < args.train_window:
        log.warning("训练窗仅 %d 日（< 请求 %d）", len(tr_days), args.train_window)
    codes = close.loc[test_days].notna().any(axis=0)
    codes = close.columns[codes.to_numpy()]
    days = pd.DatetimeIndex(list(tr_days) + list(test_days))
    log.info("训练 %s ~ %s（%d 日）| 测试 %s ~ %s（%d 日）| 股票 %d",
             tr_days[0].date(), tr_days[-1].date(), len(tr_days),
             test_days[0].date(), test_days[-1].date(), len(test_days), len(codes))

    # ---- 3) 特征面板（FeatureStore 同源变换；cap=1 防常驻 108 张全历史面板） ----
    store = FeatureStore(panels_dir, cap=1)
    feats: dict[str, np.ndarray] = {}
    for n in union_names:
        feats[n] = store.get(n).reindex(index=days, columns=codes)\
            .to_numpy(dtype=np.float32)
    log.info("特征就位：%d 张 × %s（%.1fs）", len(feats), feats[union_names[0]].shape,
             time.time() - t0)

    # ---- 4) 标签 ----
    labels: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        lab, _ = build_labels(close, horizon=h, mode="rank")
        labels[h] = lab.reindex(index=days, columns=codes).to_numpy(dtype=np.float32)

    n_tr = len(tr_days)
    n_days_all = len(days)

    def te_panel(vec: np.ndarray) -> pd.DataFrame:
        return _to_panel(vec, test_days, codes)

    # ---- 5) 各臂 ----
    arms: dict[str, pd.DataFrame] = {}

    def need(tag: str) -> bool:
        want = [x.strip() for x in args.arms.split(",") if x.strip()]
        return (not want) or tag in want

    def indep(name_list: list[str], h: int) -> np.ndarray:
        X = stack_long(feats, name_list, 0, n_tr)
        Xte = stack_long(feats, name_list, n_tr, n_days_all)
        p = _fit_predict(X, labels[h][:n_tr].ravel(), Xte)
        del X, Xte
        return p

    if need("indep_h1"):
        arms["indep_h1"] = te_panel(indep(sel[1], 1))
        log.info("[indep_h1] 完成 %.0fs", time.time() - t0)
    if need("indep_h20"):
        arms["indep_h20"] = te_panel(indep(sel[20], 20))
        log.info("[indep_h20] 完成 %.0fs", time.time() - t0)

    if need("back_prod") or need("back_union"):
        per_h: dict[tuple[str, int], np.ndarray] = {}
        for h in HORIZONS:
            names = sel[h] if need("back_prod") else None
            if names is not None:
                per_h[("prod", h)] = indep(names, h)
            if need("back_union"):
                per_h[("union", h)] = indep(union_names, h)
            log.info("[back h%d] 完成 %.0fs", h, time.time() - t0)
        if need("back_prod"):
            arms["back_prod"] = rank_average(
                [te_panel(per_h[("prod", h)]) for h in HORIZONS])
        if need("back_union"):
            arms["back_union"] = rank_average(
                [te_panel(per_h[("union", h)]) for h in HORIZONS])
        del per_h
        log.info("[back] 完成 %.0fs", time.time() - t0)

    if need("front_pool") or need("front_pool_x4"):
        # 样本堆叠：4 × n_tr 行，最后一列 = horizon token
        sizes = [int((~np.isnan(labels[h][:n_tr])).sum()) for h in HORIZONS]
        X = np.empty((sum(sizes), len(union_names) + 1), dtype=np.float32)
        y = np.empty(sum(sizes), dtype=np.float32)
        pos = 0
        for h in HORIZONS:
            lab = labels[h][:n_tr].ravel()
            Xx = stack_long(feats, union_names, 0, n_tr)
            keep = ~np.isnan(lab) & ~np.isnan(Xx).all(axis=1)
            k = int(keep.sum())
            X[pos:pos + k, :len(union_names)] = Xx[keep]
            X[pos:pos + k, len(union_names)] = float(h)
            y[pos:pos + k] = lab[keep]
            pos += k
            del Xx, keep
            log.info("[pool] 堆叠 h%d：%d 行（累计 %d）%.0fs", h, k, pos,
                     time.time() - t0)
        X, y = X[:pos], y[:pos]
        Xte_base = stack_long(feats, union_names, n_tr, n_days_all)
        for tag, mult in (("front_pool", 1), ("front_pool_x4", 4)):
            if not need(tag):
                continue
            panels = []
            for h in HORIZONS:
                Xte = np.empty((Xte_base.shape[0], Xte_base.shape[1] + 1),
                               dtype=np.float32)
                Xte[:, :-1] = Xte_base
                Xte[:, -1] = float(h)
                panels.append(te_panel(_fit_predict(X, y, Xte, mult)))
                del Xte
            arms[tag] = rank_average(panels)
            log.info("[%s] 完成 %.0fs", tag, time.time() - t0)
        del X, y, Xte_base

    if need("front_labelmix"):
        mix = _nanmean_stack([labels[h][:n_tr] for h in HORIZONS])
        X = stack_long(feats, union_names, 0, n_tr)
        Xte = stack_long(feats, union_names, n_tr, n_days_all)
        arms["front_labelmix"] = te_panel(_fit_predict(X, mix.ravel(), Xte))
        del X, Xte, mix
        log.info("[front_labelmix] 完成 %.0fs", time.time() - t0)

    # ---- 6) 评测 ----
    base_ctx = caliber.load_context(signal="ens_h1h5", window="prod",
                                    begin=int(f"{args.test_begin}0101"),
                                    end=int(f"{args.test_end}1231"),
                                    execution="open")
    if not base_ctx.pred.index.equals(test_days):
        if set(base_ctx.pred.index) <= set(test_days):
            # 信号面板比行情面板短（如 882 pred 止 2026-09-07 而行情到 10-08）：
            # 裁剪到交集并披露 —— 掩码/收益/成交价全部同源自 caliber，无前视。
            n0 = len(test_days)
            test_days = base_ctx.pred.index
            log.warning("测试日历裁剪到信号面板交集：%d → %d 日（止 %s）",
                        n0, len(test_days), test_days[-1].date())
        else:
            raise AssertionError(
                "测试日历与 caliber 上下文不一致："
                f"{test_days[0].date()}~{test_days[-1].date()} ({len(test_days)}日) vs "
                f"{base_ctx.pred.index[0].date()}~{base_ctx.pred.index[-1].date()} "
                f"({len(base_ctx.pred.index)}日)")
    log.info("%s", caliber.describe(base_ctx, "production", 1e6, "production"))
    cost = caliber.cost_profile("production")
    cap = 1e6
    prof = caliber.get_profile("production")

    # 幽灵股守卫：**全臂共用同一份**，且用**并集特征**计数 —— 若各臂按自己的
    # 特征数算（50 vs 108），阈值 ``max(1, n//4)`` 会不同（12 vs 27），掩码差异会被
    # 误读成融合方式的差异。此处复刻 ``rolling_grid_alla.existence_mask`` 的语义
    # （非 NaN 特征计数，bool 须显式整型累加），但以 numpy 累加避免为 108 张面板
    # 各建一个 DataFrame；下方自检查证明复刻与原函数逐位一致。
    def _avail(names: list[str]) -> np.ndarray:
        acc = np.zeros((len(test_days), len(codes)), dtype=np.int16)
        for nm in names:
            acc += ~np.isnan(feats[nm][n_tr:])
        return acc

    close_c = close.reindex(columns=codes)
    _chk = union_names[:2]
    _ref = existence_mask(
        {nm: pd.DataFrame(feats[nm][n_tr:], index=test_days, columns=codes)
         for nm in _chk}, close_c, test_days)
    _mine = (close_c.reindex(index=test_days).notna()
             & (_avail(_chk) >= max(1, len(_chk) // 4)))
    if not _ref.equals(_mine):
        raise AssertionError(
            f"自刻掩码与 existence_mask 语义不一致（口径守卫）："
            f"差异格 {int((_ref.fillna(False) != _mine.fillna(False)).to_numpy().sum())}")
    log.info("掩码自检通过：复刻 == existence_mask（2 张面板对拍，逐位一致）")

    valid = (close_c.reindex(index=test_days).notna()
             & (_avail(union_names) >= max(1, len(union_names) // 4)))
    log.info("共用掩码：可用股票/日中位 %d", int(valid.sum(axis=1).median()))
    fwd1 = close.pct_change(1, fill_method=None).shift(-1)\
        .reindex(index=test_days, columns=base_ctx.pred.columns)
    fwd20 = close.pct_change(20, fill_method=None).shift(-20)\
        .reindex(index=test_days, columns=base_ctx.pred.columns)

    e, x = args.buffer.split(":")
    from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly
    rows = []
    for tag, panel in arms.items():
        panel = panel.reindex(index=test_days, columns=base_ctx.pred.columns)
        panel = panel.where(valid.reindex(columns=panel.columns))
        panel.to_parquet(dest / "pred" / f"{tag}.parquet")
        ctx = caliber.CaliberContext(
            window=base_ctx.window, signal=tag, begin=base_ctx.begin,
            pred=panel, close=base_ctx.close, fwd=base_ctx.fwd, mask=base_ctx.mask,
            amt=base_ctx.amt, vol60=base_ctx.vol60, ret_hist=base_ctx.ret_hist,
            fingerprint=caliber.fingerprint_panel(panel),
            execution=base_ctx.execution, fill=base_ctx.fill)
        row = {"arm": tag, "fingerprint": ctx.fingerprint}
        for nm, fw in (("ic_h1", fwd1), ("ic_h20", fwd20)):
            ic = calc_ic_series(panel, fw)
            row[nm] = float(ic.mean())
            row[f"{nm}_ir"] = float(ic.mean() / ic.std() * np.sqrt(252)) if ic.std() else np.nan
            row[f"{nm}_t"] = (float(ic.mean() / ic.std() * np.sqrt(len(ic.dropna())))
                              if ic.std() else np.nan)
        for aname, strat in (("eqw_top10pct", TopFracLongOnly(frac=0.10)),
                             ("eqw_buf", BufferedTopFracLongOnly(float(e), float(x)))):
            m = caliber.evaluate_strategy(strat, ctx, cost, cap,
                                          execution=prof.execution)
            for k, v in m.items():
                row[f"{aname}__{k}"] = v
        rows.append(row)
        log.info("[%s] IC(h1)=%+.4f IC(h20)=%+.4f | Top10%% 年化=%.2f%% Sharpe=%.3f | "
                 "buf 年化=%.2f%% Sharpe=%.3f 换手=%.3f", tag, row["ic_h1"],
                 row["ic_h20"], row["eqw_top10pct__ann"] * 100,
                 row["eqw_top10pct__sharpe"], row["eqw_buf__ann"] * 100,
                 row["eqw_buf__sharpe"], row["eqw_buf__turnover"])

    df = pd.DataFrame(rows).set_index("arm")
    df.to_csv(dest / "results.csv", encoding="utf-8-sig")
    meta = {"select_year": args.select_year, "train_window": len(tr_days),
            "n_features_union": len(union_names),
            "test": [str(test_days[0].date()), str(test_days[-1].date())],
            "test_days": len(test_days), "n_codes": len(codes),
            "profile": "production", "cost": "production",
            "execution": prof.execution, "capital": cap,
            "lgb": lgb_kwargs(), "elapsed_s": round(time.time() - t0, 1)}
    (dest / "results.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log.info("完成 %.0fs -> %s", time.time() - t0, dest / "results.csv")
    show = df[["ic_h1", "ic_h20", "eqw_top10pct__ann", "eqw_top10pct__sharpe",
               "eqw_buf__ann", "eqw_buf__sharpe", "eqw_buf__turnover"]].copy()
    show["eqw_top10pct__ann"] = (show["eqw_top10pct__ann"] * 100).round(2)
    show["eqw_buf__ann"] = (show["eqw_buf__ann"] * 100).round(2)
    show = show.round(4)
    log.info("汇总：\n%s", show.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
