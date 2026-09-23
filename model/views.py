"""预测层观点注入 —— 华泰 AI43 的**等价实现**（非源码复现，2026-09-23）。

AI43 原文改 sklearn 随机森林源码（Cython），新增 ``speci_features``（优先分裂
因子）+ ``max_speci_depth``（树顶部用优先因子分裂的层数），人为放大观点因子
在模型中的话语权。本项目主栈是 LightGBM/ridge，改树分裂源码不可行也不必要
——注入观点的本质是「在训练前把先验偏好写进模型结构」，有两条与项目基建
兼容的等价路径（研读笔记 AI43_AI29 §一.2）：

- :func:`inject_by_feature_duplication` —— **特征复制 ×k**：观点因子在特征集
  中重复 k 份。等价机制：树模型每次分裂抽特征时观点因子被抽中的概率 ×k
  （LightGBM ``feature_fraction`` 语义下），其可占用的分裂次数上限也放大；
  线性模型（ridge）中观点因子系数的先验收缩相对减弱。k 是连续可调的
  "观点强度"旋钮（对应 AI43 的 max_speci_depth 递增）。
- :class:`TwoStagePredictor` —— **两段模型**：先用观点因子打分把截面分层，
  段内再训全因子模型，预测 = 层内模型输出（AI43「树顶部按观点因子分裂」
  的离散版：树顶部 k 层本质就是把样本按观点因子切分进子区域）。

**与优化层注入的关系**：项目已有观点注入优化器（``bl_views_from_factor``
→ :func:`optimize.solver.solve_portfolio` method="bl"）。预测层注入（本模块）
与 BL 注入是两个不同注入点，互补而非重复；对照实验形态 = 同一观点下
BL 注入 vs 预测层注入 vs 不注入三臂（RESEARCH_TODO §一）。

**评价口径**：注入后的模型**不再是"无观点"模型**——对比实验必须三臂同框
（观点因子相同、训练窗口/折切分/标签全一致），只看注入臂与基线臂的
OOS IC/组合指标差异，且须披露观点因子自身的单因子 IC（观点可能是错的）。
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from model.predictor import BasePredictor, grid_of

__all__ = ["inject_by_feature_duplication", "TwoStagePredictor"]


def inject_by_feature_duplication(
    features: Mapping[str, pd.DataFrame],
    view_features: list[str],
    k: int = 2,
) -> dict[str, pd.DataFrame]:
    """特征复制注入：把观点因子在特征集中重复 k 份。

    Args:
        features: 原特征集 {name: date×code 面板}。
        view_features: 观点因子名（须 ∈ features）。
        k: 复制份数。k=1 即原特征集（无注入）；k=2 表示每个观点因子额外
            多 1 份副本（共出现 2 次）。
    Returns:
        新特征 dict（原面板对象不复制——副本引用同一 DataFrame，内存开销
        只是引用；下游 `_long_matrix` 按列拼接，副本列独立占位）。
        副本命名 ``<name>__dup<i>``（i = 2..k）。
    """
    if k < 1:
        raise ValueError(f"k 须 ≥ 1（k=1 即无注入），收到 {k}")
    missing = [n for n in view_features if n not in features]
    if missing:
        raise KeyError(f"观点因子不在特征集中: {missing}")
    out = dict(features)
    for name in view_features:
        for i in range(2, k + 1):
            out[f"{name}__dup{i}"] = features[name]
    return out


class TwoStagePredictor(BasePredictor):
    """两段模型：观点因子先分层，层内再训全因子模型（AI43 等价实现②）。

    第一段：训练段内按观点因子逐日截面分数的 ``(1 - top_frac)`` 分位点把
    股票日分为"观点内"（分位以上）与"观点外"两组（逐日判定，**只用当日
    截面信息**，无未来函数；阈值取训练段中位数以保证两组都有足够样本）。
    第二段：每组各训一个底层预测器（同构，``predictor_cls`` 参数化），
    预测时按当日观点分位归组，输出对应组的层内预测。

    与特征复制的差异：复制是"软"注入（改变分裂概率/收缩结构），两段是
    "硬"注入（模型结构上保证观点因子决定样本分区）。AI43 树顶部 k 层
    分裂的极端形态即"每个叶子路径都由观点因子开头"= 硬注入。

    ⚠️ 观点因子必须是**有经济含义的连续因子**（如估值/质量分组）；若观点
    因子接近常数，分层退化为单组（行为 = 不注入，不报错——注入强度自然
    为零比静默报错更符合"观点权重可调"的设计）。
    """

    name = "two_stage"

    def __init__(self, predictor_cls: type[BasePredictor], view_feature: str,
                 top_frac: float = 0.5, **predictor_params):
        if not (0.0 < top_frac < 1.0):
            raise ValueError(f"top_frac 须 ∈ (0,1)，收到 {top_frac}")
        self.predictor_cls = predictor_cls
        self.view_feature = view_feature
        self.top_frac = top_frac
        self.predictor_params = predictor_params

    def _daily_mask(self, view_panel: pd.DataFrame, idx: pd.Index,
                    cols: pd.Index) -> pd.DataFrame:
        """逐日截面：观点分位 >= 阈值 → True（观点内组）。

        阈值 = 该截面有效值的中位数（稳健；观点值 NaN 的股票归观点外组
        ——保守默认，观点缺失不享受观点溢价）。
        """
        med = view_panel.median(axis=1)
        mask = view_panel.ge(med, axis=0)
        return mask.reindex(index=idx, columns=cols).fillna(False).astype(bool)

    def fit(self, features: Mapping[str, pd.DataFrame],
            labels: pd.DataFrame) -> "TwoStagePredictor":
        if self.view_feature not in features:
            raise KeyError(f"观点因子 {self.view_feature!r} 不在特征集中")
        idx, cols = grid_of(features, labels)
        self.feature_names_ = sorted(features.keys())
        vp = features[self.view_feature].reindex(index=idx, columns=cols)
        m = self._daily_mask(vp, idx, cols)

        # 组内子样本：把 (date, code) 网格按 mask 拆两份，各自 fit 一个底层模型
        top_features = {k: v.where(m) for k, v in features.items()}
        rest_features = {k: v.where(~m) for k, v in features.items()}
        self.model_top_ = self.predictor_cls(**self.predictor_params)
        self.model_rest_ = self.predictor_cls(**self.predictor_params)
        self.model_top_.fit(top_features, labels.where(m))
        self.model_rest_.fit(rest_features, labels.where(~m))
        self._fit_idx, self._fit_cols = idx, cols
        return self

    def predict(self, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        self._check_features(features)
        # 分组判定在**输入面板自身**的网格上做（OOS 折的日期与训练段无交集，
        # 不能与 _fit_idx 求交集——否则折内 predict 全部报"无交集"，09-23 修）。
        vp = features[self.view_feature]
        m = self._daily_mask(vp, vp.index, vp.columns)

        top_features = {k: v.where(m) for k, v in features.items()}
        rest_features = {k: v.where(~m) for k, v in features.items()}
        pred_top = self.model_top_.predict(top_features)
        pred_rest = self.model_rest_.predict(rest_features)
        out = pred_top.where(m, pred_rest)
        return out.reindex(index=vp.index, columns=vp.columns)
