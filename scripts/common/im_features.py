"""``im_*`` 分钟特征装载与评估窗口对齐（GFlowNet / GP 第二层共用口径）。

2026-11-18 从 ``scripts/archive/e2e_common.py`` 提取。原模块拆成三份后：

- 本模块：``im_*`` 接入（数据集参数化，不绑具体库）
- :mod:`scripts.common.feature_funnel`：``select_features``（口径中性）
- 删除：``load_daily_data`` / ``load_library_grid_panels`` / ``DATASET`` 等
  HS300 绑定项（读死 ``daily_hs300.parquet`` + ``hs300_2022_2025`` 库）

⚠️ 现状（2026-11-18 实测）：``im_*`` 面板在活跃库 ``all_a_2018_2026`` 里
**0 个**；归档的 ``hs300_2022_2025`` registry 有 95 条但 ``panels/`` 下 0 个文件。
即：本模块的**代码口径是真的，数据是缺的**。启用 ``--feat-source im_*`` 前必须
先把 im 面板重建进活跃库，否则 ``load_im_panels`` 会返回空集并抛错（这是刻意
的显式失败，不静默退化成纯 raw）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("im_features")

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


def load_im_panels(source: str = "im_independent", dataset: str | None = None
                   ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """从因子库取 ``im_*`` 分钟特征面板（**单一口径真源**）。

    2026-09-16 自 ``scripts/factors/mine_im_combos.load_im_panels`` 上移，
    供 GP 第二层挖掘与 GFlowNet 共用，避免两套筛选口径漂移。

    Args:
        source: ``im_independent`` = 独立且显著（``dup_corr1<0.5`` 且
            ``|t_nw|>2``，读 ``reports/intraday_stat_checkup.csv``）；
            ``im_all`` = 库内全部 ``im_*``。
        dataset: 因子库数据集名（**必填**——不再默认某个库，避免静默读错数据集）。

    Returns:
        ``(panels, docs)``：panels = {name: date×code 面板}（已剔除全 NaN 面板），
        docs = {name: formula 说明}。
    """
    from research.factor_library import FactorLibrary

    if not dataset:
        raise ValueError("load_im_panels 需要显式 dataset（不设默认库，防静默跨库）")
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
                       dataset: str | None = None, loader=None
                       ) -> tuple[dict[str, pd.DataFrame], list[str],
                                  pd.DatetimeIndex | None]:
    """把 ``im_*`` 分钟特征并入 ``panel``，返回 ``(panel, features, im_index)``。

    设计约束（改动前先读）：

    - **特征名顺序**：原始 6 字段在前、``im_*`` 追加在后。动作 id 由
      ``FactorMDP`` 按 ``n_op + n_win + feature 下标`` 编码，保持前缀不变才能让
      ``raw`` 臂与混合臂的动作语义对齐（否则同一条轨迹的 token 含义会漂移）。
    - **对齐口径**：im 面板 reindex 到 ``panel['close']`` 的 index×columns，
      再按 ``panel['mask']`` 掩码 —— 与 ``close_m`` 同口径，防止未在册行混入 IC。
    - **覆盖区间**：im 特征只在有分钟数据的日子有效。归一化后的 ``im_index``
      交回调用方，由调用方裁**评估段**而不是裁面板行
      （裁面板行会连带丢掉 rolling 算子的 warm-up 历史）。
    - 全 NaN 面板剔除而非静默保留；全部失败时抛错。

    Args:
        panel: 原始面板 dict（须含 ``close``）。
        feat_source: 见 :data:`FEAT_SOURCES`。
        dataset: 因子库数据集名；``feat_source`` 不含 im 时可不传。
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

    为什么需要：``im_*`` 特征只在有分钟数据的日子有值。若把含 im 的公式和纯
    raw 公式放在同一段 IC 上比，前者的样本期天然更短 —— 奖励与 OOS IC
    **都不可比**。所以这里让**两类公式用同一评估段**。

    注意裁的是**评估段**而非面板行：面板仍保留更早的行情，rolling 算子才有
    warm-up 历史（裁行会让 ``ts_mean_20`` 在窗口首日全 NaN）。

    ``eval_begin``/``eval_end`` 存在的理由（2026-09-16）：光靠 ``--train-begin``
    裁剪会**同时改变面板构建区间**，进而改变列并集与 Barra 风格参照系，等于给
    对照臂引入第二个差异。把「评估窗口」与「面板构建区间」解耦后，raw 对照臂
    可用同一面板对齐窗口。

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
