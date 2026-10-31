"""序列深度学习模型（GRU / TCN）与截面相关系数损失的**唯一真源**。

对应国泰海通《七维共振》§2.1.1（`reports/report_replica/README.md` §4.1）：
本模块只提供"网 + 损失 + 一次训练"，数据构造与滚动编排在
`scripts/model_train/replica_deep.py`，聚合与评测沿用 `scripts/portfolio/report_replica.py`。

三个与生产 GBDT 的本质差异（也是研报能到 IC 6.85% 的全部原因）：

1. **损失函数 = 截面内预测值与标签的相关系数取负**，不是 MSE。
   MSE 逼模型拟合绝对收益（受大盘 beta 主导，截面排序湮没）；
   −corr 直接优化横截面排序，与「选行业」这件事目标一致。
2. **batch = 一个交易日的截面**，不是随机凑 8192 行。
   −corr 必须在一个完整截面上算，跨日拼接会把 beta 混进来。
3. **滚动训练**：每年一个模型，只用该年之前的数据；训练集末尾按标签长度
   再做一次 purge，杜绝标签跨入预测期。

网络超参照研报原文：
- GRU：2 层，`sequence_length` 由数据集决定（day60→60，day30/week30→30）
- TCN：kernel_size=3；时间长度 60 → 4 层 `[8,16,32,64]`；长度 30 → 3 层 `[8,16,32]`
  因果卷积 + 膨胀（dilation 1/2/4/8）+ 残差连接
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def corr_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """截面 Pearson 相关系数取负（研报 §2.1.1.3 第 4 条）。

    两个向量都已去均值即 Pearson；这里显式中心化后归一，eps 防常数截面。
    """
    p = pred - pred.mean()
    q = y - y.mean()
    denom = p.norm() * q.norm()
    return -(p @ q) / (denom + 1e-8)


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class GruNet(nn.Module):
    """GRU：`num_layers=2`，取最后时间步隐状态接线性头。"""

    def __init__(self, f_dim: int, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(f_dim, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (B, T, F)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _CausalBlock(nn.Module):
    """TCN 残差块：两层因果膨胀卷积（左侧 padding = (k-1)*d）。"""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad1, self.pad2 = pad, pad
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, C, T)
        h = self.conv1(nn.functional.pad(x, (self.pad1, 0)))
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(nn.functional.pad(h, (self.pad2, 0)))
        h = self.act(h)
        h = self.drop(h)
        return h + self.down(x)


class TcnNet(nn.Module):
    """TCN：`num_channels=[8,16,32(,64)]`，kernel=3，dilation 逐层翻倍。"""

    def __init__(self, f_dim: int, channels: tuple[int, ...] = (8, 16, 32),
                 kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        blocks, c_prev = [], f_dim
        for i, c in enumerate(channels):
            blocks.append(_CausalBlock(c_prev, c, kernel, dilation=2 ** i,
                                       dropout=dropout))
            c_prev = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(c_prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, T, F)
        h = x.transpose(1, 2)                                # -> (B, F, T)
        h = self.blocks(h)
        return self.head(h[:, :, -1]).squeeze(-1)


def build_net(arch: str, f_dim: int, seq_len: int, hidden: int = 64,
              seed: int | None = None) -> nn.Module:
    """按研报的「时长 → 层数/通道」规则建网。

    `arch` 取 `gru` / `tcn`；`seq_len` 只影响 TCN 层数（60→4 层，其余→3 层）。
    """
    torch.manual_seed(0 if seed is None else int(seed))
    if arch == "gru":
        return GruNet(f_dim, hidden=hidden, num_layers=2)
    if arch == "tcn":
        ch = (8, 16, 32, 64) if seq_len >= 60 else (8, 16, 32)
        return TcnNet(f_dim, channels=ch, kernel=3)
    raise ValueError(f"未知 arch={arch!r}（只支持 gru / tcn）")


# ---------------------------------------------------------------------------
# 训练（按交易日截面 batch + −corr 损失 + 验证集早停）
# ---------------------------------------------------------------------------
class DailyCrossSectionTrainer:
    """研报式训练器：一个 epoch 走一遍所有交易日，每交易日一个 batch。

    参数
    ----
    seq      : `(n_days, n_codes, seq_len, f_dim)` 的 float32 序列数组（通常是
               `sliding_window_view` 出来的视图，不额外占内存）
    y        : `(n_days, n_codes)` 标签，已截面标准化；NaN 表示该股当日不参与
    epochs   : 最多轮数（早停实际会更少）
    patience : 验证集 IC 连续多少轮不升就停
    min_n    : 一个截面至少多少只有效股票才计入损失
    """

    def __init__(self, arch: str, seq_len: int, f_dim: int, *, hidden: int = 64,
                 lr: float = 1e-3, weight_decay: float = 0.0, epochs: int = 30,
                 patience: int = 5, min_n: int = 100, seed: int = 3,
                 threads: int = 8, device: str = "cpu", verbose: bool = False):
        self.arch, self.seq_len, self.f_dim = arch, seq_len, f_dim
        self.hidden, self.lr, self.wd = hidden, lr, weight_decay
        self.epochs, self.patience, self.min_n = epochs, patience, min_n
        self.seed, self.threads, self.verbose = seed, threads, verbose
        self.device = device
        self.net: nn.Module | None = None
        self.best_epoch = -1
        self.best_score = float("nan")
        self.history: list[dict] = []

    # -- 内部工具 ---------------------------------------------------------
    def _batch_ok(self, y: np.ndarray) -> np.ndarray:
        """当日可训练股票的布尔掩码（标签有限且股票数够）。"""
        return np.isfinite(y)

    @staticmethod
    def _ic(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3:
            return np.nan
        a = a - a.mean()
        b = b - b.mean()
        d = math.sqrt(float(a @ a) * float(b @ b))
        return float(a @ b) / d if d > 0 else np.nan

    def _eval(self, seq: np.ndarray, y: np.ndarray,
              pos: np.ndarray | None = None) -> tuple[float, float]:
        """返回 (按天截面 IC 均值, 按天 −corr 损失均值)。"""
        self.net.eval()
        ics, losses = [], []
        pos = np.arange(len(seq)) if pos is None else np.asarray(pos)
        with torch.no_grad():
            for t in pos:
                m = self._batch_ok(y[t])
                if m.sum() < self.min_n:
                    continue
                xb = torch.from_numpy(
                    np.ascontiguousarray(seq[t][m])).to(self.device)
                yb = torch.from_numpy(y[t][m].astype(np.float32)).to(self.device)
                p = self.net(xb)
                ics.append(self._ic(p.cpu().numpy(), yb.cpu().numpy()))
                losses.append(float(corr_loss(p, yb)))
        self.net.train()
        ics = [v for v in ics if np.isfinite(v)]
        return (float(np.mean(ics)) if ics else np.nan,
                float(np.mean(losses)) if losses else np.nan)

    # -- 主入口 -----------------------------------------------------------
    def fit(self, seq: np.ndarray, y: np.ndarray, tr_pos: np.ndarray,
            va_pos: np.ndarray) -> "DailyCrossSectionTrainer":
        """`seq`/`y` 是全量滑窗视图，`tr_pos`/`va_pos` 是窗口位置数组。

        传位置而不是布尔掩码/切片，避免 numpy 复制整个训练集（(T,N,seq,F)
        布尔索引会实打实分配 5GB+）。
        """
        torch.set_num_threads(int(self.threads))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = build_net(self.arch, self.f_dim, self.seq_len,
                             hidden=self.hidden, seed=self.seed).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.wd)
        best_state, bad = None, 0
        self.net.train()

        tr_pos = np.asarray(tr_pos)
        for ep in range(self.epochs):
            order = np.random.permutation(tr_pos)
            tot, nb = 0.0, 0
            for t in order:
                m = self._batch_ok(y[t])
                if m.sum() < self.min_n:
                    continue
                xb = torch.from_numpy(
                    np.ascontiguousarray(seq[t][m])).to(self.device)
                yb = torch.from_numpy(y[t][m].astype(np.float32)).to(self.device)
                opt.zero_grad()
                loss = corr_loss(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach())
                nb += 1
            ic_va, l_va = self._eval(seq, y, va_pos)
            self.history.append({"epoch": ep + 1, "loss_tr": tot / max(nb, 1),
                                 "ic_va": ic_va, "loss_va": l_va})
            if self.verbose:
                print(f"    ep{ep + 1:02d} loss_tr={tot / max(nb, 1):+.4f} "
                      f"ic_va={ic_va:+.4f}")
            if np.isfinite(ic_va) and (not np.isfinite(self.best_score)
                                       or ic_va > self.best_score + 1e-5):
                self.best_score, self.best_epoch, bad = ic_va, ep + 1, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, seq: np.ndarray, pos: np.ndarray,
                y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """逐窗口输出原始分数；返回 `(pos, scores[n_pos, n_codes])`。"""
        assert self.net is not None, "先 fit"
        pos = np.asarray(pos)
        out = np.full((len(pos), seq.shape[1]), np.nan, dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            for k, t in enumerate(pos):
                m = (self._batch_ok(y[t]) if y is not None
                     else np.isfinite(seq[t][:, 0, 0]))
                if m.sum() == 0:
                    continue
                xb = torch.from_numpy(
                    np.ascontiguousarray(seq[t][m])).to(self.device)
                out[k, m] = self.net(xb).cpu().numpy()
        return pos, out
