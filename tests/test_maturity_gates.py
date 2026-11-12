"""成熟度晋升门槛测试（2026-11-05）。

锁四类事：

1. **门槛是绝对量**：夏普不够就是不够，不因为"库里排名靠前"而晋级。
2. **口径即证据**：旧口径（未复权/无掩码）的数字不许晋级——这条如果失守，
   2026-11-04 那次 `alpha191_015` 2.86→−0.71 的教训就白吃了。
3. **队列去重是双闸**：相关（`crowded`）+ 命名族（`family_cap`）。只有相关那一道
   时实测放进了 43 个 active，40 个是同一公式的窗口网格。
4. **写回不破坏契约**：`apply_maturity` 之后 `contract_violations()` 必须为空。

夹具不走 ``ingest``/``evaluate``（那两条路要真实面板与回测，且指标不可控），
直接写 registry 行 + eval 序列——本文件测的是"给定证据如何定档"，不是回测本身。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import sharpe_ratio
from research.factor_library import (MATURITY_GATES, FactorLibrary, _family_key,
                                     _window_stats, name_slug)

N = 800          # 约 3.2 年；后段 30% = 240 天，够"近 250 日"窗口取不满时退化


def _series(*, mu: float = 0.0025, sigma: float = 0.012, seed: int = 1,
            flip_at: int | None = None, flip_mu: float = -0.001,
            n: int = N) -> pd.Series:
    """造一条日收益序列；``flip_at`` 之后换成另一个均值（用来做"后段失效"）。

    先把标准正态**去均值除标准差**再加 ``mu``：不这么做的话，800 个样本的抽样
    误差会和 ``mu`` 同量级，同一份夹具换个 numpy 版本就换结论。

    默认 ``mu=0.0025`` 而非 0.001：后 30% 段只有 240 个样本，局部均值的抽样误差
    约 ``sigma/√240 ≈ 0.0008``；``mu`` 若也是 0.001，一个"全程正收益"的因子会被
    随机判成 ``unstable``。这是夹具的噪声预算，不是标准放松。
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    z = (z - z.mean()) / z.std(ddof=0)
    r = mu + sigma * z
    if flip_at is not None:
        z2 = rng.normal(0, 1, n - flip_at)
        z2 = (z2 - z2.mean()) / z2.std(ddof=0)
        r[flip_at:] = flip_mu + sigma * z2
    return pd.Series(r, index=pd.bdate_range("2018-01-02", periods=n))


def _build(root, specs: dict, *, matrix: tuple | None = None) -> FactorLibrary:
    """按 ``{name: 参数}`` 写一个最小库（registry + eval），返回库实例。

    ``specs[name]`` 支持的键：``sharpe``（用 mu 调）、``flip_at``、``maturity``、
    ``ret_basis``、``mask_basis``、``dup_checked``、``dup_corr_max``、
    ``resid_ic``、``eval_status``、``no_eval_file``。
    """
    lib = FactorLibrary(root=root, dataset="t")
    rows = []
    for i, (name, sp) in enumerate(specs.items()):
        dret = _series(mu=sp.get("mu", 0.0025), seed=i + 1,
                       flip_at=sp.get("flip_at"), n=sp.get("n", N))
        ic = _series(mu=sp.get("ic_mu", 0.03), sigma=0.1, seed=i + 101,
                     flip_at=sp.get("flip_at"), flip_mu=sp.get("ic_flip_mu", -0.03),
                     n=sp.get("n", N)) * 0.1
        if not sp.get("no_eval_file"):
            pd.DataFrame({"ic": ic, "dret_ls_M": dret}).to_parquet(
                lib.evals_dir / f"{name_slug(name)}.parquet")
        rows.append({
            "name": name,
            "eval_status": sp.get("eval_status", "evaluated"),
            "evaluated_at": sp.get("evaluated_at", "2026-11-01 10:00:00"),
            "maturity": sp.get("maturity", ""),
            "sharpe_ls_M": float(sharpe_ratio(dret)),
            "ret_basis": sp.get("ret_basis", "adj_close_pct_v1"),
            "mask_basis": sp.get("mask_basis", "tradable_t0"),
            "dup_checked": sp.get("dup_checked", True),
            "dup_corr_max": sp.get("dup_corr_max", 0.3),
            "resid_ic": sp.get("resid_ic", np.nan),
            "t_stat_nw": 3.0,
            "note": "",
        })
    pd.DataFrame(rows).to_csv(lib.root / "registry.csv", index=False)
    if matrix is not None:
        corr, names = matrix
        np.savez(lib.root / "dup_matrix.npz", corr=corr,
                 names=np.array(names, dtype="U64"))
    return lib


def _diag(names: list[str], hi: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """对角 1 的相关矩阵；``hi={(a,b): r}`` 指定高相关对。"""
    n = len(names)
    c = np.eye(n, dtype=np.float64)
    for (a, b), r in (hi or {}).items():
        i, j = names.index(a), names.index(b)
        c[i, j] = c[j, i] = r
    return c, names


# ---------------------------------------------------------------------------
# 1. 族键
# ---------------------------------------------------------------------------
def test_family_key_strips_window_but_keeps_numbering():
    # 窗口型：数字紧跟字母，剥掉后是同族
    assert _family_key("alpha360_VOLUME36") == "alpha360_VOLUME"
    assert _family_key("alpha158_VSUMN60") == "alpha158_VSUMN"
    # 编号型：数字紧跟下划线，剥了会把 180 个不同公式并成一族 → 不剥
    assert _family_key("alpha191_166") == "alpha191_166"
    assert _family_key("alpha191_001") == "alpha191_001"
    # 无数字：原样
    assert _family_key("float_mktcap") == "float_mktcap"


# ---------------------------------------------------------------------------
# 2. 分段统计
# ---------------------------------------------------------------------------
def test_window_stats_sharpe_matches_library_convention():
    """分段夏普必须与 registry 的 ``sharpe_ls_M`` 同口径，否则两者不可比。"""
    r = _series(mu=0.001, seed=7)
    st = _window_stats(r, r * 0.01)
    whole = float(sharpe_ratio(r))
    # 前段 + 后段拼接 ≈ 全段（只在两端都够样本时成立）
    assert st["sharpe_back"] == pytest.approx(float(sharpe_ratio(r.iloc[:560])), rel=1e-9)
    assert st["sharpe_fwd"] == pytest.approx(float(sharpe_ratio(r.iloc[560:])), rel=1e-9)
    assert st["sharpe_recent"] == pytest.approx(float(sharpe_ratio(r.iloc[-250:])), rel=1e-9)
    assert np.isfinite(whole)


def test_window_stats_returns_nan_when_too_short():
    r = _series(n=40)
    st = _window_stats(r, r, min_days=60)
    assert all(np.isnan(st[k]) for k in st)


# ---------------------------------------------------------------------------
# 3. 定档
# ---------------------------------------------------------------------------
def test_strong_factor_reaches_active_and_weak_falls_back(tmp_path):
    lib = _build(tmp_path, {"strong": {}, "weak": {"mu": 0.0}},
                 matrix=_diag(["strong", "weak"]))
    df = _assess(lib)
    assert df.at["strong", "verdict"] == "active"
    assert "low_sharpe" in df.at["weak", "fails"]
    assert df.at["weak", "verdict"] == "experimental"


def test_no_corr_matrix_means_no_active(tmp_path):
    """没有相关矩阵时上限是 ``oos_verified``——证不了独立就不给白名单。"""
    lib = _build(tmp_path, {"a": {}, "b": {}})
    df = lib.assess_maturity()
    assert set(df["verdict"]) == {"oos_verified"}
    assert (df["queue"] == -1).all()


def test_crowded_rejects_second_of_a_pair(tmp_path):
    lib = _build(tmp_path, {"a": {}, "b": {"mu": 0.0023}},
                 matrix=_diag(["a", "b"], {("a", "b"): 0.99}))
    df = _assess(lib)
    assert df.at["a", "verdict"] == "active"
    assert df.at["b", "verdict"] == "oos_verified"
    assert df.at["b", "fails"] == ("crowded",)


def test_family_cap_limits_window_grid(tmp_path):
    """同族三个都达标，只留 ``max_per_family`` 个，其余降为 oos_verified。"""
    names = ["alpha360_VOLUME10", "alpha360_VOLUME20", "alpha360_VOLUME30"]
    lib = _build(tmp_path, {n: {} for n in names}, matrix=_diag(names))
    df = _assess(lib)
    assert int((df["verdict"] == "active").sum()) == int(MATURITY_GATES["max_per_family"])
    capped = df[df["fails"] == ("family_cap",)]
    assert len(capped) == 1
    assert capped["verdict"].iloc[0] == "oos_verified"


def test_unstable_back_half_flips_to_negative(tmp_path):
    lib = _build(tmp_path, {"flip": {"flip_at": 400, "flip_mu": -0.002}},
                 matrix=_diag(["flip"]))
    df = _assess(lib)
    assert "unstable" in df.at["flip", "fails"]
    assert df.at["flip", "verdict"] == "experimental"


def test_old_caliber_blocks_promotion(tmp_path):
    lib = _build(tmp_path, {"legacy": {"ret_basis": "close_pct_v1"}},
                 matrix=_diag(["legacy"]))
    df = _assess(lib)
    assert "stale_caliber" in df.at["legacy", "fails"]


def test_high_dup_needs_residual_increment(tmp_path):
    """相关 ≥0.95 且残差无信息 = 别人的影子，不晋级；残差够大则放行。"""
    lib = _build(tmp_path, {
        "shadow": {"dup_corr_max": 0.99, "resid_ic": 0.002},
        "novel": {"dup_corr_max": 0.96, "resid_ic": 0.03},
    }, matrix=_diag(["shadow", "novel"]))
    df = _assess(lib)
    assert "redundant" in df.at["shadow", "fails"]
    assert df.at["novel", "fails"] == ()


def test_invalid_is_never_rejudged(tmp_path):
    lib = _build(tmp_path, {"dead": {"maturity": "invalid"}},
                 matrix=_diag(["dead"]))
    df = _assess(lib)
    assert df.at["dead", "verdict"] == "invalid"


# ---------------------------------------------------------------------------
# 4. 写回
# ---------------------------------------------------------------------------
def test_apply_fills_blank_and_keeps_note_history(tmp_path):
    lib = _build(tmp_path, {"a": {}, "b": {"mu": 0.0}},
                 matrix=_diag(["a", "b"]))
    df = _assess(lib)
    n = lib.apply_maturity(df)
    assert n == 2
    reg = lib.list_all().set_index("name")
    assert reg.at["a", "maturity"] == "active"
    assert reg.at["b", "maturity"] == "experimental"
    assert f"[{df['stamp'].iloc[0]} experimental]" in str(reg.at["b", "note"])
    # 达标的行也要在表里自证（写证据摘要，不是留空）
    assert "夏普" in str(reg.at["a", "note"])
    # 幂等：同样的判定再写一次不改任何行
    assert lib.apply_maturity(df) == 0


def test_promoted_rows_pass_contract(tmp_path):
    lib = _build(tmp_path, {"a": {}, "b": {}}, matrix=_diag(["a", "b"]))
    df = _assess(lib)
    lib.apply_maturity(df)
    assert lib.contract_violations() == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _assess(lib: FactorLibrary) -> pd.DataFrame:
    """读回相关矩阵后评定，模拟脚本 ``load_corr`` + ``assess_maturity`` 的调用。"""
    z = np.load(lib.root / "dup_matrix.npz", allow_pickle=True)
    return lib.assess_maturity(corr=z["corr"],
                               corr_names=[str(x) for x in z["names"]])
