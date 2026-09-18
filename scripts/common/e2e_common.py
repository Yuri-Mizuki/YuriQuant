"""
端到端工作流共享模块
====================

e2e_stock_picks.py（今日选股）、e2e_backtest.py（walk-forward 回测）与
investment_report / optimize_e2e 等 e2e 家族脚本共用的编排逻辑：
数据加载（``load_daily_data``）、特征选择漏斗（``select_features``）、
面板新鲜度守卫（``drop_stale_factors``）、风格中性化
（``build_neutral_covariates`` / ``neutralize_predictions``）。

另含 **im_* 分钟特征装载**（2026-09-16，国金24 × GFlowNet 交叉）：
``load_im_panels`` 是 `im_*` 面板的**单一口径真源**，``attach_im_features``
把分钟特征按 close 网格对齐＋mask 后并入终端特征集，``aligned_eval_masks``
保证启用 im 后两类公式落在同一评估段（否则奖励/OOS IC 不可比）。

约定（e2e 家族必须一致）：
- 股票池：因子库 significant 面板的列并集（HS300 PIT 历史成员，~420 股）
- 因子：经典量价 12（``factor.classic.compute_classic_features``）+ 因子库
  significant（``FactorLibrary.load_significant_features``，默认排除 model:*）
- 特征选择：build_feature_set 三级漏斗（覆盖率>=0.5 → |corr|<0.7 去冗余
  → valid 段 |IC| 质量分降序截断），只在调用方指定的选择窗口上做（防前视）

已下沉（2026-08-31，见 packages 单一实现）：
- 经典特征 / mock 数据 / 标签构建 / 因子库加载 → factor.classic / data.mock /
  model.labels / research.factor_library，scripts 只保留编排入口。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("e2e_common")

HORIZON = 5
GBDT_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_child_samples": 100,
    "n_estimators": 200,
    "seed": 42,
}
RIDGE_ALPHA = 1.0
DATASET = "hs300_2022_2025"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_daily_data(begin: int = 20220101) -> tuple[dict, dict]:
    """读日线缓存并对齐到因子库股票池。

    Returns:
        (px, lib_feats): px = {open/high/low/close/volume/amount: date×code}，
        lib_feats = {name: date×code} 因子库 significant 面板（默认排除 model:*）。
    """
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from research.factor_library import FactorLibrary

    cache = DataCache(OfflineDataSource())
    d = pd.read_parquet(cache.root / "daily_hs300.parquet")

    # 股票池 = significant 因子面板列并集（排除 model:*）
    lib_feats = FactorLibrary(dataset=DATASET).load_significant_features(exclude_model=True)
    pool = None
    for p in lib_feats.values():
        pool = set(p.columns) if pool is None else pool | set(p.columns)
    codes = sorted(pool)
    dates_all = sorted(d.index.get_level_values("date").unique())
    log.info("日线缓存: %s ~ %s | 股票池 %d 股", dates_all[0].date(), dates_all[-1].date(), len(codes))

    px = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        w = (d.reset_index()
             .pivot(index="date", columns="code", values=col).sort_index())
        w = w.reindex(columns=codes)
        w = w[w.index >= pd.Timestamp(str(begin))]
        px[col] = w
    return px, lib_feats


def load_library_grid_panels(dataset: str = DATASET) -> dict[str, pd.DataFrame]:
    """读日线缓存，对齐到因子库全面板网格的 OHLCV 宽表（date×code）。

    2026-08-31 自 ml_synthesis_experiment._px_panels 下沉（cpcv_h1_eval 等共用）。
    """
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from research.factor_library import FactorLibrary

    cache = DataCache(OfflineDataSource())
    d = pd.read_parquet(cache.root / "daily_hs300.parquet")

    grid = next(iter(FactorLibrary(dataset=dataset).load_library_features().values()))
    codes, dates = grid.columns, grid.index

    out = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        w = (d.reset_index()
             .pivot(index="date", columns="code", values=col).sort_index()
             .reindex(index=dates, columns=codes))
        out[col] = w
    return out


# ---------------------------------------------------------------------------
# im_* 分钟特征装载（国金24 × GFlowNet 交叉：日内统计特征接进挖掘框架）
# ---------------------------------------------------------------------------
#: 原始量价字段（= GFlowNet 既有 FEATURES，顺序不可变——动作 id 前缀依赖它）
RAW_FEATURES: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")

#: ``--feat-source`` 取值 -> (是否含原始 OHLCV, im 子集名)
FEAT_SOURCES: dict[str, tuple[bool, str]] = {
    "raw": (True, ""),
    "im_independent": (False, "im_independent"),
    "im_all": (False, "im_all"),
    "raw+im_independent": (True, "im_independent"),
    "raw+im_all": (True, "im_all"),
}

#: 研报口径的 im 独立显著筛选条件（国金24 / mine_im_combos 同源）
IM_DUP_CORR_MAX = 0.5
IM_T_NW_MIN = 2.0


def resolve_feat_source(feat_source: str) -> tuple[bool, str]:
    """``--feat-source`` -> ``(use_raw, im_source)``（非法取值直接抛错，不静默退化）。"""
    if feat_source not in FEAT_SOURCES:
        raise ValueError(
            f"未知 feat_source={feat_source!r}，可选: {', '.join(FEAT_SOURCES)}")
    return FEAT_SOURCES[feat_source]


def load_im_panels(source: str = "im_independent", dataset: str = DATASET
                   ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """从因子库取 ``im_*`` 分钟特征面板（**单一口径真源**）。

    2026-09-16 自 ``scripts/factors/mine_im_combos.load_im_panels`` 上移，
    供 GP 第二层挖掘与 GFlowNet 共用，避免两套筛选口径漂移。

    Args:
        source: ``im_independent`` = 独立且显著（``dup_corr1<0.5`` 且
            ``|t_nw|>2``，读 ``reports/intraday_stat_checkup.csv``）；
            ``im_all`` = 库内全部 ``im_*``（42 个）。
        dataset: 因子库数据集名。

    Returns:
        ``(panels, docs)``：panels = {name: date×code 面板}（已剔除全 NaN 面板），
        docs = {name: formula 说明}。
    """
    from research.factor_library import FactorLibrary

    if source not in ("im_independent", "im_all"):
        raise ValueError(f"未知 im 子集: {source!r}（可选 im_independent / im_all）")

    lib = FactorLibrary(dataset=dataset)
    reg = lib.list_all()
    reg = reg[reg["name"].str.startswith("im_")]
    n_all = len(reg)
    if source == "im_independent":
        chk = pd.read_csv("reports/intraday_stat_checkup.csv")
        ok = set(chk[(chk["dup_corr1"] < IM_DUP_CORR_MAX)
                     & (chk["t_stat_nw"].abs() > IM_T_NW_MIN)]["name"])
        reg = reg[reg["name"].isin(ok)]
        log.info("独立显著筛选: %d/%d 个 im_* 入选终端集", len(reg), n_all)

    panels = {n: lib.get_panel(n) for n in reg["name"]}
    panels = {n: p for n, p in panels.items()
              if p is not None and p.notna().any().any()}
    docs = dict(zip(reg["name"], reg["formula"]))
    return panels, docs


def attach_im_features(panel: dict[str, pd.DataFrame], feat_source: str = "raw",
                       dataset: str = DATASET, loader=None
                       ) -> tuple[dict[str, pd.DataFrame], list[str],
                                  pd.DatetimeIndex | None]:
    """把 ``im_*`` 分钟特征并入 ``panel``，返回 ``(panel, features, im_index)``。

    设计约束（改动前先读）：

    - **特征名顺序**：原始 6 字段在前、``im_*`` 追加在后。动作 id 由
      ``FactorMDP`` 按 ``n_op + n_win + feature 下标`` 编码，保持前缀不变才能让
      ``raw`` 臂与混合臂的动作语义对齐（否则同一条轨迹的 token 含义会漂移）。
    - **对齐口径**：im 面板 reindex 到 ``panel['close']`` 的 index×columns，
      再按 ``panel['mask']`` 掩码 —— 与 ``close_m`` 同口径，防止未在册行混入 IC。
    - **覆盖区间**：im 特征只在有分钟数据的日子有效（当前 2022-01 起）。
      归一化后的 ``im_index`` 交回调用方，由调用方裁**评估段**而不是裁面板行
      （裁面板行会连带丢掉 rolling 算子的 warm-up 历史）。
    - 全 NaN 面板剔除而非静默保留；全部失败时抛错。

    Args:
        panel: 原始面板 dict（须含 ``close``）。
        feat_source: 见 :data:`FEAT_SOURCES`。
        dataset: 因子库数据集名。
        loader: 可注入的装载器，签名 ``(source, dataset) -> (panels, docs)``
            （单测用；None 时走 :func:`load_im_panels`）。

    Returns:
        ``(panel, features, im_index)``。未启用 im 时 ``im_index`` 为 None。
    """
    use_raw, im_source = resolve_feat_source(feat_source)
    features = list(RAW_FEATURES) if use_raw else []
    if not im_source:
        return panel, features, None

    load = loader if loader is not None else load_im_panels
    im_panels, _docs = load(im_source, dataset)
    if not im_panels:
        raise RuntimeError(f"im 面板为空（source={im_source}, dataset={dataset}）")

    close = panel["close"]
    mask = panel.get("mask")
    kept: dict[str, pd.DataFrame] = {}
    dropped: list[str] = []
    for name, p in im_panels.items():
        q = p.reindex(index=close.index, columns=close.columns)
        if mask is not None:
            q = q.where(mask)
        if q.notna().any().any():
            kept[name] = q
        else:
            dropped.append(name)
    if dropped:
        log.warning("im 面板在目标网格上全 NaN，已剔除 %d 个: %s",
                    len(dropped), ", ".join(sorted(dropped)))
    if not kept:
        raise RuntimeError(f"im 特征全部对齐失败（feat_source={feat_source}）")

    names = sorted(kept)
    out = dict(panel)
    out.update(kept)

    covered: pd.DatetimeIndex | None = None
    for q in kept.values():
        ix = q.index[q.notna().any(axis=1)]
        covered = ix if covered is None else covered.union(ix)
    assert covered is not None and len(covered) > 0

    log.info("im 特征接入: %d 个（source=%s），有值日 %s ~ %s 共 %d 日，"
             "特征总数 %d（raw=%s）",
             len(names), im_source, covered[0].date(), covered[-1].date(),
             len(covered), len(features) + len(names), use_raw)
    return out, features + names, covered


def aligned_eval_masks(index: pd.DatetimeIndex, test_begin: int | str,
                       im_index: pd.DatetimeIndex | None = None,
                       keep_window: bool = False,
                       eval_begin: int | str | None = None,
                       eval_end: int | str | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """训练/测试日期掩码（启用 im 时裁到 im 覆盖区间）。

    为什么需要：``im_*`` 特征只在有分钟数据的日子有值（当前 2022-01 起）。若把
    含 im 的公式和纯 raw 公式放在同一段 IC 上比，前者的样本期天然更短 —— 奖励与
    OOS IC **都不可比**。所以这里让**两类公式用同一评估段**。

    注意裁的是**评估段**而非面板行：面板仍保留更早的行情，rolling 算子才有
    warm-up 历史（裁行会让 ``ts_mean_20`` 在窗口首日全 NaN）。

    ``eval_begin``/``eval_end`` 存在的理由（2026-09-16）：光靠 ``--train-begin``
    裁剪会**同时改变面板构建区间**，进而改变列并集（实测 520 → 421 列）与 Barra
    风格参照系，等于给对照臂引入第二个差异。把「评估窗口」与「面板构建区间」
    解耦后，raw 对照臂可用同一面板对齐窗口：
    ``--feat-source raw --eval-begin <im首日> --eval-end <im末日>``。

    Args:
        index: 面板日期索引。
        test_begin: 训练/测试切分日（int YYYYMMDD 或日期串）。
        im_index: im 特征有值日（None = 未启用 im，等价于全段可用）。
        keep_window: 为 True 时即使启用 im 也不裁（调用方须自行告警不可比）。
        eval_begin: 评估段起点；与 im 覆盖起点取**较晚者**。
        eval_end: 评估段终点；与 im 覆盖终点取**较早者**。

    Returns:
        (train_mask, test_mask)：``train = index < test_begin``、
        ``test = index >= test_begin``，各自再与上述约束取交。
    """
    tb = pd.Timestamp(str(test_begin))
    train_mask = np.asarray(index < tb)
    test_mask = np.asarray(index >= tb)

    lo: pd.Timestamp | None = None
    hi: pd.Timestamp | None = None
    if eval_begin is not None:
        lo = pd.Timestamp(str(eval_begin))
    if eval_end is not None:
        hi = pd.Timestamp(str(eval_end))
    if im_index is not None and not keep_window:
        im_lo, im_hi = im_index[0], im_index[-1]
        lo = im_lo if lo is None else max(lo, im_lo)
        hi = im_hi if hi is None else min(hi, im_hi)
    if lo is not None:
        train_mask = train_mask & np.asarray(index >= lo)
    if hi is not None:
        # 上界同时作用于训练段（防 --eval-end 早于 test_begin 的错配场景）
        train_mask = train_mask & np.asarray(index <= hi)
        test_mask = test_mask & np.asarray(index <= hi)
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# 特征选择（防前视：质量分与漏斗都只在调用方指定的定型窗口上做）
# ---------------------------------------------------------------------------
def select_features(
    all_feats: dict,
    fwd: pd.DataFrame,
    quality_days: pd.DatetimeIndex,
    panel_days: pd.DatetimeIndex | None = None,
    max_features: int = 30,
    min_coverage: float = 0.5,
    dedup_corr: float = 0.7,
) -> tuple[dict, pd.Series | None]:
    """build_feature_set 三级漏斗（**单一实现**，2026-08-29 收敛 7 份脚本内联副本）。

    质量分在 ``quality_days``（通常 valid 段）上计算 |IC|，漏斗在
    ``panel_days``（通常 dev 段 = train+valid）上做覆盖率 -> 去冗余 -> 截断，
    两窗口分离以匹配"定型期选择、test 不参与"的防前视纪律。

    Args:
        all_feats: {name: 全时段面板}（建议已截面标准化）。
        fwd: horizon 前瞻收益面板（与质量分窗口对齐用）。
        quality_days: 质量分计算窗口。
        panel_days: 漏斗窗口；None = 与 quality_days 相同（旧调用行为）。
        max_features / min_coverage / dedup_corr: 漏斗三参数。
    Returns:
        (入选因子的【全时段】面板 {name: panel}, 质量分 Series | None)。
        质量分为空时返回 None（build_feature_set 回退按独立性去冗余）。
    """
    from model.features import build_feature_set
    from stats.ic import calc_ic_series

    q = {}
    for nm, p in all_feats.items():
        try:
            ic = calc_ic_series(p.reindex(index=quality_days),
                                fwd.reindex(index=quality_days)).dropna()
            if len(ic) >= 10:
                q[nm] = abs(float(ic.mean()))
        except Exception:
            pass
    quality = pd.Series(q).sort_values(ascending=False) if q else None
    days = quality_days if panel_days is None else panel_days

    # reindex 而非 loc：经典因子与因子库面板的日期网格可能不同（reindex 是超集）
    feats_sel = build_feature_set(
        {k: v.reindex(index=days) for k, v in all_feats.items()},
        min_coverage=min_coverage, dedup_corr=dedup_corr,
        max_features=max_features, quality=quality)
    selected = sorted(feats_sel)
    log.info("特征漏斗: %d -> %d（覆盖率>=%.2f, |corr|<%.2f, 上限 %d）",
             len(all_feats), len(selected), min_coverage, dedup_corr, max_features)
    return {k: all_feats[k] for k in selected}, quality


def drop_stale_factors(
    feats: dict,
    as_of_date: pd.Timestamp,
    buffer_days: int = 5,
) -> dict:
    """面板新鲜度守卫：剔除末端早于 as_of_date - buffer 的失效面板。

    单个滞后面板（如数据源中断未回补）会把整个公共网格的末端拖回，
    导致预测日/回测区间人为缩短——剔除并告警，而不是拖垮全流程。
    """
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=buffer_days)
    stale = []
    for k, v in feats.items():
        last = v.dropna(how="all").index
        if len(last) == 0 or last[-1] < cutoff:
            stale.append(k)
    if stale:
        log.warning("剔除 %d 个滞后面板（末端 < %s）: %s ...",
                    len(stale), cutoff.date(), sorted(stale)[:5])
        feats = {k: v for k, v in feats.items() if k not in set(stale)}
    return feats


# ---------------------------------------------------------------------------
# 风格中性化（华泰五因子：市值/行业/动量/波动/换手）
# ---------------------------------------------------------------------------
def build_neutral_covariates(px: dict, close: pd.DataFrame, real: bool = True):
    """构建市值 + 行业 + mom/vol/turn 协变量面板（华泰五因子口径）。

    Returns:
        (market_cap_panel | None, industry_panel | None, extra_covariates dict)
    """
    from factor.preprocessing import build_style_covariates

    mc_panel, ind_panel = None, None
    if real:
        try:
            from data.cache import DataCache
            from data.industry import IndustryClassification
            from data.market_cap import build_market_cap_panel
            from data.offline import OfflineQuietDataSource

            # 注意：必须用 OfflineQuietDataSource（数据源方法返回空 -> 走缓存
            # fallback）。普通 OfflineDataSource 直接抛异常，get_equity_structure
            # 的缓存分支在 try 之外，永远到不了 -> 市值/行业恒为 None。
            cache = DataCache(OfflineQuietDataSource())
            codes = list(close.columns)
            es = cache.get_equity_structure(codes)
            mc_panel = build_market_cap_panel(es, close)
            ind_panel = IndustryClassification(cache).get_industry_panel(codes, close.index)
            log.info("市值面板: %d 日 × %d 股 | 行业: %d 类",
                     mc_panel.shape[0], mc_panel.shape[1],
                     ind_panel.dropna(how="all").shape[1] if len(ind_panel) else 0)
        except Exception as e:
            log.warning("市值/行业加载失败，仅用价量风格: %s", e)
    extra = build_style_covariates(px, mc_panel, ind_panel)
    return mc_panel, ind_panel, extra


def neutralize_predictions(pred: pd.DataFrame, mc, ind, extra) -> pd.DataFrame:
    """预测分数五因子中性化（逐日截面回归取残差）。

    协变量面板列集（全股票池）需对齐到预测面板列集（模型覆盖子集），
    否则 neutralize 内布尔索引 index union 后 Unalignable。
    """
    from factor.preprocessing import neutralize

    cols = pred.columns
    mc_a = mc.reindex(columns=cols) if mc is not None else None
    ind_a = ind.reindex(columns=cols) if ind is not None else None
    # extra 里 industry/size 与 industry_panel/market_cap_panel 重复（且 industry
    # 是字符串，直接当连续协变量 astype(float) 会崩）。剔除后只留 mom/vol/turn。
    extra_a = {k: v.reindex(columns=cols) for k, v in extra.items()
               if k not in ("industry", "size")}
    neu = neutralize(pred, market_cap_panel=mc_a, industry_panel=ind_a,
                     extra_covariates=extra_a)
    log.info("中性化: 有效值 %d (原始 %d)", neu.notna().sum().sum(),
             pred.notna().sum().sum())
    return neu
