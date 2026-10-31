"""标签口径回归测试：`forward_avg_ret` 的分母必须是可成交价，不能是 `avg[t]`。

背景（2026-10-31 事故，详见 `reports/report_replica/dl/README.md` §四）：
研报只写「均价涨跌幅」没给分母。若照字面用当日均价 `avg[t]` 做分母，而特征里又有
`vwap/close = avg[t]/close[t]`，两边共用同一个 `avg[t]`，就会产生**纯机械相关**：

    label = mean(avg[t+1..t+h]) / avg[t] - 1 ≈ r - d ,   d ≡ log(avg[t]/close[t])
    ⇒ corr(-d, label) ≈ σ(d) / σ(r)

后果是「什么都学不到的假预测」也能拿到个股 IC 0.17 —— 曾把 TCN 的个股 IC 从真实
0.072 虚报成 0.198。本测试用**构造数据**（`d` 与未来收益完全独立）把这个机制钉死：
只要分母被换回 `avg[t]`，第一个断言也会失败。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.model_train.replica_deep import forward_avg_ret


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _synthetic(n_days: int = 260, n_stocks: int = 400, h: int = 10,
               seed: int = 0) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """构造「均价相对收盘的偏离独立于未来收益」的随机市场。

    `close` 走个股独立随机游走，`avg = close * exp(d)`、`d` 为独立同分布噪声。
    这样 `d` 对**任何**未来收益都没有预测力，只剩分母共线能造出 IC。
    """
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 0.02, size=(n_days, n_stocks)).astype(np.float32)
    close = (10.0 * np.exp(np.cumsum(ret, axis=0))).astype(np.float32)
    d = rng.normal(0.0, 0.01, size=(n_days, n_stocks)).astype(np.float32)
    avg = (close * np.exp(d)).astype(np.float32)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return idx, avg, close


def _mean_daily_corr(x: np.ndarray, y: np.ndarray) -> float:
    """逐日截面相关后取均值（与 eval 脚本口径一致）。"""
    out = []
    for t in range(x.shape[0]):
        m = np.isfinite(x[t]) & np.isfinite(y[t])
        if m.sum() >= 20:
            out.append(_corr(x[t][m], y[t][m]))
    return float(np.nanmean(out))


@pytest.mark.parametrize("h", [5, 10])
def test_label_denominator_must_be_close(h: int) -> None:
    idx, avg, close = _synthetic(h=h)
    # 裸特征：−(avg/close - 1) 的截面 z-score 等价形式，直接用对数偏离
    d = np.log(avg / close)

    lab_ok = forward_avg_ret(avg, idx, h, close, zscore=False)   # 正确：分母 = 收盘
    lab_bad = forward_avg_ret(avg, idx, h, avg, zscore=False)     # 事故口径：分母 = 均价

    ic_ok = _mean_daily_corr(-d, lab_ok)
    ic_bad = _mean_daily_corr(-d, lab_bad)

    # 正确口径下 d 对未来收益毫无预测力（数据构造保证），IC 必须在 0 附近
    assert abs(ic_ok) < 0.02, f"分母=close 时不该有信号，实测 {ic_ok:+.4f}"
    # 事故口径下会被机械抬到 σ(d)/σ(r) 量级，这是本测试要锁住的「假信号」
    assert ic_bad > 0.08, f"分母=avg 时应出现伪相关（>0.08），实测 {ic_bad:+.4f}"
    # 两者差距就是这次虚报的量级
    assert ic_bad - ic_ok > 0.08


def test_forward_avg_ret_requires_close_positionally() -> None:
    """`close` 必须是必填位置参数，防止再退回「默认用 avg 做分母」。"""
    idx, avg, close = _synthetic(n_days=80, n_stocks=60)
    with pytest.raises(TypeError):
        forward_avg_ret(avg, idx, 10)  # noqa: F841  —— 缺 close


def test_label_alignment_no_lookahead() -> None:
    """标签必须严格来自 t+1..t+h：把最后 h 天改成 NaN，倒数第 h+1 天的标签应仍有效。"""
    idx, avg, close = _synthetic(n_days=60, n_stocks=40, h=5)
    lab = forward_avg_ret(avg, idx, 5, close, zscore=False)
    assert np.isfinite(lab[-6]).any(), "t = T-h-1 的标签应可用"
    assert np.isnan(lab[-5:]).all(), "最后 h 天没有完整前瞻窗，标签必须为 NaN"
