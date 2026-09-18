"""
AlphaPool 策略网络的时序特征提取器（华泰 AI97 图表15 口径）
===========================================================

研报把 ``features_extractor_class`` 设为 ``LSTMSharedNet`` / ``TransformerSharedNet``，
并给出三个结构超参：``n_layers=1``、``d_model=128``、``dropout=0.1``。

本模块提供两个 :class:`~stable_baselines3.common.torch_layers.BaseFeaturesExtractor`：

- :class:`LSTMSharedNet` —— 单层 LSTM，取末位时间步；
- :class:`TransformerSharedNet` —— 单层 TransformerEncoder，取末位时间步。

观测布局（由 :class:`factor.rl.alphapool_gym.AlphaPoolGymEnv` 决定）::

    obs = [ token_id × max_len , 手工特征 × 3 ]
           ↑ pad 用 id == n_actions 填充

两个网络共用的编码方式：token 走 ``nn.Embedding``，3 个手工特征经线性层投影成
**一个额外的"状态 token"追加到序列末尾**。这样做的两个好处：

1. 序列末位**永远是有效位置**，不需要额外的 pooling 逻辑或掩码；
2. LSTM / Transformer 都在同一份输入上比较，差异只来自时序结构本身
   —— 避免"两个网络因为读手工特征的方式不同而不可比"。

``vocab = n_actions + 1``（多出的一个 id 留给 pad）。``n_actions`` 与 ``seq_len``
默认从 ``observation_space`` 推断（``high.max()`` 与 ``shape[0] - 3``），也可显式传入。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

__all__ = ["LSTMSharedNet", "TransformerSharedNet", "make_extractor_kwargs"]

#: 研报图表15 的结构超参
REPORT_N_LAYERS = 1
REPORT_D_MODEL = 128
REPORT_DROPOUT = 0.1
_HAND_DIM = 3


class _TokenSeqExtractor(BaseFeaturesExtractor):
    """公共部分：token 嵌入 + 手工特征 token 拼接。"""

    def __init__(self, observation_space, features_dim: int = REPORT_D_MODEL,
                 d_model: int = REPORT_D_MODEL, dropout: float = REPORT_DROPOUT,
                 n_actions: int | None = None, seq_len: int | None = None):
        super().__init__(observation_space, features_dim)
        total = int(observation_space.shape[0])
        self.seq_len = int(seq_len) if seq_len is not None else total - _HAND_DIM
        if self.seq_len <= 0:
            raise ValueError(f"观测维度 {total} 不足以容纳 3 个手工特征")
        self.n_actions = int(n_actions) if n_actions is not None else \
            int(observation_space.high.max())
        self.pad_id = self.n_actions
        self.d_model = int(d_model)
        self.dropout_p = float(dropout)

        self.emb = nn.Embedding(self.n_actions + 1, self.d_model)
        self.hand_proj = nn.Linear(total - self.seq_len, self.d_model)
        self.head = nn.Sequential(nn.Linear(self.d_model, features_dim), nn.ReLU())

    def _embed(self, obs: torch.Tensor):
        """obs (B, L+3) → (B, L+1, d_model) 与 pad 掩码 (B, L+1)。

        掩码长度必须是 **L+1**（与拼上手工特征 token 后的序列等长）—— 手工特征 token
        恒为有效位，故末列为 ``False``。少一位会让 ``nn.TransformerEncoder`` 抛
        ``mask shape should be (B, L)``。
        """
        ids = obs[:, :self.seq_len].long()
        hand = obs[:, self.seq_len:]
        tokens = self.emb(ids)                                # (B, L, d_model)
        hand_tok = self.hand_proj(hand).unsqueeze(1)          # (B, 1, d_model)
        x = torch.cat([tokens, hand_tok], dim=1)              # (B, L+1, d_model)
        mask = torch.cat(
            [ids.eq(self.pad_id),
             torch.zeros(ids.shape[0], 1, dtype=torch.bool, device=ids.device)],
            dim=1)
        return x, mask


class LSTMSharedNet(_TokenSeqExtractor):
    """单层 LSTM 共享网络（研报 ``LSTMSharedNet``）。"""

    def __init__(self, observation_space, features_dim: int = REPORT_D_MODEL,
                 d_model: int = REPORT_D_MODEL, n_layers: int = REPORT_N_LAYERS,
                 dropout: float = REPORT_DROPOUT, **kw):
        super().__init__(observation_space, features_dim=features_dim,
                         d_model=d_model, dropout=dropout, **kw)
        # nn.LSTM 在 num_layers=1 时会忽略 dropout 参数 → 失活单独放在层后
        self.rnn = nn.LSTM(self.d_model, self.d_model,
                           num_layers=int(n_layers), batch_first=True)
        self.drop = nn.Dropout(self.dropout_p)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x, _pad = self._embed(obs)
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1, :]))            # 末位 = 手工特征 token


class TransformerSharedNet(_TokenSeqExtractor):
    """单层 TransformerEncoder 共享网络（研报 ``TransformerSharedNet``）。

    ``nhead`` 默认 4（``d_model=128`` 可整除）。pad 位置用 ``src_key_padding_mask``
    屏蔽；``_embed`` 给出的掩码末列恒为 ``False``（手工特征 token 永远有效），
    因此即便某条轨迹尚未生成任何 token（整行 pad），也不会出现全屏蔽导致注意力
    softmax 全 ``-inf`` → NaN 的情形。
    """

    def __init__(self, observation_space, features_dim: int = REPORT_D_MODEL,
                 d_model: int = REPORT_D_MODEL, n_layers: int = REPORT_N_LAYERS,
                 dropout: float = REPORT_DROPOUT, nhead: int = 4,
                 dim_feedforward: int | None = None, **kw):
        super().__init__(observation_space, features_dim=features_dim,
                         d_model=d_model, dropout=dropout, **kw)
        if self.d_model % nhead != 0:
            raise ValueError(f"d_model={self.d_model} 不能被 nhead={nhead} 整除")
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=int(nhead),
            dim_feedforward=int(dim_feedforward or 4 * self.d_model),
            dropout=self.dropout_p, batch_first=True, norm_first=True,
        )
        # norm_first=True 与 nested tensor 优化不兼容，torch 会打 UserWarning；
        # 本场景序列短（L+1 约 10~20）且带 padding mask，nested tensor 用不上，
        # 显式关闭以保持日志干净（不影响数值）。
        self.enc = nn.TransformerEncoder(layer, num_layers=int(n_layers),
                                        enable_nested_tensor=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x, mask = self._embed(obs)
        out = self.enc(x, src_key_padding_mask=mask)
        return self.head(out[:, -1, :])                       # 末位 = 手工特征 token


def make_extractor_kwargs(policy: str, *, n_layers: int = REPORT_N_LAYERS,
                          d_model: int = REPORT_D_MODEL,
                          dropout: float = REPORT_DROPOUT) -> dict:
    """把 ``--policy`` 名字转成 sb3 的 ``policy_kwargs``（研报三个结构超参透传）。"""
    p = (policy or "mlp").lower()
    if p in ("mlp", "none", ""):
        return {}
    cls = {"lstm": LSTMSharedNet, "transformer": TransformerSharedNet}.get(p)
    if cls is None:
        raise ValueError(f"未知策略网络 {policy!r}，可选：mlp / lstm / transformer")
    return {
        "features_extractor_class": cls,
        "features_extractor_kwargs": {
            "d_model": int(d_model), "n_layers": int(n_layers),
            "dropout": float(dropout), "features_dim": int(d_model),
        },
    }
