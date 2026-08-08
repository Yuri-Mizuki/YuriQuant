"""
model —— 02 模型层（对齐聚宽 AI 投研流程）。

模型设计（registry.ModelRegistry 规格）→ 模型训练（training.train_and_register）
→ 模型评价（evaluation.evaluate_model）→ 模型迭代（同名再注册 = 新版本，
experiments.py 记录实验）。

当前 ML 能力集中在因子合成（ML stacking，factor/synthesis.py），本包是
模型层的流程容器：注册表 + 训练/评价标准入口。新增独立收益预测模型时
在同一接口下扩展。
"""
from model.evaluation import evaluate_model
from model.registry import ModelRegistry, default_model_root
from model.training import train_and_register, train_stacking_model

__all__ = [
    "ModelRegistry", "default_model_root",
    "train_stacking_model", "train_and_register",
    "evaluate_model",
]
