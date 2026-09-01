"""
技术指标算子测试（2026-08-12 新增，参考华泰报告26）：
KAMA/AROONOSC/HT_DCPHASE/BOLL/OBV/RSI/ADX 面板级实现 + GP 原语注册冒烟。
mock 数据，不依赖 SDK。
"""
import numpy as np
import pandas as pd

from factor.operators import (
    TECH_OPS, adx, aroonosc, boll_pctb, ht_dcphase, kama, obv, rsi,
)
from factor.genetic_mining import build_primitive_set, eval_tree


def _mock_ohlcv(n_days=120, n_codes=10, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_codes)), axis=0)),
                         idx, codes)
    high = close * (1 + rng.uniform(0, 0.01, close.shape))
    low = close * (1 - rng.uniform(0, 0.01, close.shape))
    volume = pd.DataFrame(rng.integers(1e5, 1e6, close.shape).astype(float), idx, codes)
    return {"close": close, "high": high, "low": low, "volume": volume}


def _shape_ok(out, ref):
    assert isinstance(out, pd.DataFrame)
    assert out.shape == ref.shape
    return out


def test_rsi_bounded_and_directional():
    p = _mock_ohlcv()
    r = _shape_ok(rsi(p["close"], 14), p["close"])
    assert r.notna().to_numpy().any()
    # RSI 应在 (0, 100) 内（数值安全）
    valid = r.stack().dropna()
    assert ((valid > 0) & (valid < 100)).all()
    # 单调上涨序列 → RSI 应接近 100（>50）
    up = pd.DataFrame(np.arange(100, 200, dtype=float).reshape(-1, 1),
                      index=pd.date_range("2023-01-01", periods=100, freq="B"),
                      columns=["A"])
    r_up = rsi(up, 14).iloc[-1, 0]
    assert r_up > 50


def test_obv_accumulates():
    p = _mock_ohlcv()
    _shape_ok(obv(p["close"], p["volume"]), p["close"])
    # OBV 是累积量：全涨序列 OBV 单调不减
    up = pd.DataFrame(np.arange(100, 200, dtype=float).reshape(-1, 1),
                      index=pd.date_range("2023-01-01", periods=100, freq="B"),
                      columns=["A"])
    vol = pd.DataFrame(np.full(100, 1e5), index=up.index, columns=["A"])
    o_up = obv(up, vol)
    assert (o_up["A"].diff().dropna() >= 0).all()


def test_kama_follows_trend():
    p = _mock_ohlcv()
    _shape_ok(kama(p["close"], 10), p["close"])
    # 单调上涨序列：KAMA 应单调跟随（不完全相等但单调不减）
    up = pd.DataFrame(np.arange(100, 200, dtype=float).reshape(-1, 1),
                      index=pd.date_range("2023-01-01", periods=100, freq="B"),
                      columns=["A"])
    k_up = kama(up, 10)
    assert (k_up["A"].diff().dropna() >= -1e-9).all()


def test_boll_pctb_bounded():
    p = _mock_ohlcv()
    b = _shape_ok(boll_pctb(p["close"], 20), p["close"])
    valid = b.stack().dropna()
    # %B 大致在带内（-2~2 区间外为超带，极端值允许少量）
    assert (valid.abs() < 10).all()


def test_aroonosc_bounded():
    p = _mock_ohlcv()
    a = _shape_ok(aroonosc(p["high"], p["low"], 20), p["close"])
    valid = a.stack().dropna()
    assert (valid.abs() <= 100 + 1e-9).all()


def test_adx_positive_bounded():
    p = _mock_ohlcv()
    a = _shape_ok(adx(p["high"], p["low"], p["close"], 14), p["close"])
    valid = a.stack().dropna()
    assert (valid >= 0).all()
    assert (valid <= 100 + 1e-9).all()
    # 强单边趋势 → ADX 应明显大于 0
    n = 120
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.DataFrame(np.linspace(100, 200, n), index=idx, columns=["A"])
    high = close + 1.0
    low = close - 1.0
    a_trend = adx(high, low, close, 14).iloc[-1, 0]
    assert a_trend > 10


def test_ht_dcphase_range():
    p = _mock_ohlcv()
    h = _shape_ok(ht_dcphase(p["close"]), p["close"])
    valid = h.stack().dropna()
    assert (valid.abs() <= 180).all()


def test_ht_dcphase_causal():
    """因果性（2026-08-28 前视偏差修复）：改变 t 之后的序列不得影响 t 的相位。

    旧实现用 scipy.signal.hilbert 作用在整段序列上（全局卷积，非因果），
    在 GP 挖掘中构成"时间机器"——样本内外都能偷看未来刷出虚假夏普。
    """
    rng = np.random.default_rng(0)
    t = np.arange(200)
    s = pd.DataFrame({"a": np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.01, 200)},
                     index=pd.date_range("2023-01-01", periods=200, freq="B"))
    h1 = ht_dcphase(s)
    s2 = s.copy()
    s2.iloc[150:] = 0.0                      # 只改未来段
    h2 = ht_dcphase(s2)
    both = h1.dropna().index.intersection(h2.dropna().index)
    common = [d for d in both if d <= s.index[140]]   # 只比较被修改点之前的相位
    assert len(common) > 50
    assert np.allclose(h1.loc[common, "a"], h2.loc[common, "a"]), \
        "ht_dcphase 在 t 日的取值依赖了 t 之后的数据（前视偏差）"


def test_gp_primitive_registration():
    """7 个指标都注册进 GP 原语集，且 eval_tree 可求值。"""
    feats = ["close", "high", "low", "volume"]
    pset, prim_map = build_primitive_set(feats, windows=(5, 10, 20))
    names = set(prim_map.keys())
    # 窗口编名 + 无窗口
    for expected in ["kama_5", "rsi_10", "boll_pctb_20", "aroonosc_5",
                     "adx_10", "ht_dcphase", "obv"]:
        assert expected in names, f"缺少原语 {expected}"
    # eval_tree 对含技术指标公式求值
    p = _mock_ohlcv()
    panel = {f: p[f] for f in feats}
    from deap import gp as deap_gp
    for expr_str in ["kama_5(close)", "rsi_10(close)", "obv(close, volume)",
                     "adx_5(high, low, close)", "aroonosc_10(high, low)",
                     "ht_dcphase(close)", "boll_pctb_20(close)"]:
        tree = deap_gp.PrimitiveTree.from_string(expr_str, pset)
        out = eval_tree(tree, panel, prim_map)
        assert out is not None and out.shape == p["close"].shape, expr_str


def test_tech_ops_registered_in_operators():
    """TECH_OPS 注册表齐全（exhaustive 自动跳过 arity≠1）。"""
    names = {op.name for op in TECH_OPS}
    assert names == {"aroonosc", "adx", "ht_dcphase", "obv"}
    # arity 信息正确
    by_name = {op.name: op for op in TECH_OPS}
    assert by_name["adx"].arity == 3
    assert by_name["ht_dcphase"].n_window == 0
