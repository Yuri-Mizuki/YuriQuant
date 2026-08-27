"""
model —— 02 模型层（对齐聚宽 AI 投研流程）。

模型设计（registry.ModelRegistry 规格）→ 模型训练（training.train_and_register）
→ 模型评价（evaluation.evaluate_model）→ 模型迭代（同名再注册 = 新版本，
experiments.py 记录实验）。

五组件（2026-08-19 模型层建设，详见 reports/yuriquant_model_layer_design）：
- ① FeatureStore  ``features.build_feature_set``   因子面板 → 对齐特征集
- ② LabelBuilder  ``labels.build_labels``          horizon × mode 标签 + embargo
- ③ Predictor     ``predictor.PREDICTORS``         ridge / gbdt 截面预测器
- ④ Trainer       ``training.train_and_register(kind="predictor")``
- ⑤ 消费出口      ``serving.register_model_as_factor``（回写因子库）；
                  策略/优化器直接吃预测面板（date×code 即接口，零适配）
"""
from model.evaluation import evaluate_model
from model.features import build_feature_set
from model.labels import build_labels, forward_returns
from model.predictor import (
    PREDICTORS,
    BasePredictor,
    LGBMPredictor,
    RidgePredictor,
    fit_predict_oos,
)
from model.registry import ModelRegistry, default_model_root
from model.serving import register_model_as_factor
from model.training import (
    train_and_register,
    train_predictor_model,
    train_stacking_model,
)

__all__ = [
    "ModelRegistry", "default_model_root",
    "train_stacking_model", "train_predictor_model", "train_and_register",
    "evaluate_model",
    "build_feature_set", "build_labels", "forward_returns",
    "BasePredictor", "RidgePredictor", "LGBMPredictor", "PREDICTORS",
    "fit_predict_oos",
    "register_model_as_factor",
]
