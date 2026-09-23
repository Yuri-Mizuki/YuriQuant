"""rolling_grid_alla 实验脚本的核心纯函数单测（不依赖全A数据缓存）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pd.Timestamp("2018-01-01")


def _make_base(tmp_path):
    """最小 _base 目录：close/mask/基准（float32 小面板）。"""
    base = tmp_path / "_base"
    base.mkdir(parents=True)
    idx = pd.bdate_range("2018-01-01", "2019-12-31")
    cols = [f"c{i}" for i in range(6)]
    close = pd.DataFrame(
        100 + np.arange(len(idx))[:, None] * 0.1 + np.zeros((1, len(cols))),
        index=idx, columns=cols, dtype=np.float32)
    close.iloc[:3, 0] = np.nan          # 首日 pct_change NaN（引擎口径）
    close.to_parquet(base / "close_adj.parquet")
    pd.DataFrame(True, index=idx, columns=cols).to_parquet(base / "tradable_mask.parquet")
    ret = close.pct_change(fill_method=None).mean(axis=1).fillna(0)
    ret.to_frame("ret").to_parquet(base / "bench_index.parquet")
    ret.to_frame("ret").to_parquet(base / "bench_eqw.parquet")
    pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32).to_parquet(
        base / "market_cap.parquet")
    return base


def test_yearly_metrics_basic():
    from scripts.pipelines.rolling_grid_alla import _yearly_metrics
    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    dr = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    bench = pd.Series(0.0, index=idx)
    ym = _yearly_metrics(dr, bench)
    assert set(ym) == {2019, 2020}
    for y, m in ym.items():
        sub = dr[dr.index.year == y]
        assert m["excess"] == pytest.approx(m["annual"] - 0.0, abs=1e-9)
        assert m["max_dd"] >= 0      # 正幅度（同 backtest.metrics 约定）
        # 年化 = 复利年化
        expect = (1 + sub).prod() ** (252 / len(sub)) - 1
        assert m["annual"] == pytest.approx(expect, rel=1e-9)


def test_rebalance_days_validated():
    from scripts.pipelines.rolling_grid_alla import _rebalance_days_validated
    idx = pd.bdate_range("2018-01-01", "2018-06-30")
    # 2M：每 2 个月首个交易日；末日 6/1 的区间（6/1->末尾）跨度 21 >= 1 保留
    rbd = _rebalance_days_validated(idx, 1, "2M")
    months = sorted({pd.Timestamp(d).month for d in rbd})
    assert months == [1, 3, 5]
    for d in rbd:
        same_month = idx[(idx.month == d.month) & (idx.year == d.year)]
        assert d == same_month[0]
    # 跨度守卫：horizon 超过末段跨度时，末段调仓日被剔除
    # 2018-06-01 到序列末尾共 21 个交易日位置（跨度 20），h=20 保留、h=25 剔除
    rbd20 = _rebalance_days_validated(idx, 20, "M")
    rbd25 = _rebalance_days_validated(idx, 25, "M")
    assert pd.Timestamp("2018-06-01") not in rbd25
    assert pd.Timestamp("2018-06-01") in rbd20


def test_select_features_dedup_and_coverage(tmp_path, monkeypatch):
    from scripts.pipelines import rolling_grid_alla as R

    days = pd.bdate_range("2016-07-01", "2018-12-31")
    codes = [f"c{i}" for i in range(8)]
    rng = np.random.default_rng(1)
    # f_good 有真实 IC，f_dup 与其完全相关，f_lowcov 覆盖不足
    signal = pd.DataFrame(rng.normal(size=(len(days), len(codes))), index=days,
                          columns=codes)
    fwd = -0.3 * signal + rng.normal(scale=0.1, size=(len(days), len(codes)))
    ic_series = {
        "f_good": (fwd * signal).mean(axis=1) / 100,
        "f_mid": (fwd * (0.5 * signal)).mean(axis=1) / 100,
        "f_lowcov": (fwd * signal).mean(axis=1) / 100,
        "f_weak": pd.Series(rng.normal(scale=1e-5, size=len(days)), index=days),
    }
    ic_cache = pd.DataFrame(ic_series)

    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    # f_weak 与 f_good 面板独立（否则会被相关去冗余正确剔除，测不到质量排序）
    weak_panel = pd.DataFrame(rng.normal(size=(len(days), len(codes))),
                              index=days, columns=codes)
    for n in ("f_good", "f_mid", "f_lowcov", "f_weak"):
        if n == "f_lowcov":
            p = signal.where(
                pd.DataFrame(rng.random((len(days), len(codes))) > 0.7,
                             index=days, columns=codes))
        elif n == "f_weak":
            p = weak_panel
        else:
            p = signal.copy()
        p.astype(np.float32).to_parquet(panels_dir / f"{n}.parquet")
    registry = pd.DataFrame({
        "name": ["f_good", "f_mid", "f_lowcov", "f_weak"],
        "coverage": [1.0, 1.0, 0.2, 1.0]})

    store = R.FeatureStore(panels_dir)
    feats = R.select_features_for_year(2018, 1, ic_cache, registry, store, days)
    assert "f_good" in feats
    assert "f_lowcov" not in feats        # 覆盖率过滤
    assert feats.index("f_good") < feats.index("f_weak")  # 质量降序优先


def test_res_metrics_matches_manual():
    from scripts.pipelines.rolling_grid_alla import res_metrics
    idx = pd.bdate_range("2019-01-01", periods=252)
    rng = np.random.default_rng(2)
    dr = pd.Series(rng.normal(0.001, 0.012, len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    to = pd.Series([0.3] * 12, index=idx[::21])
    m = res_metrics(dr, bench, to)
    assert m["turnover"] == pytest.approx(0.3)
    expect_ann = (1 + dr).prod() ** (252 / len(dr)) - 1
    assert m["annual"] == pytest.approx(expect_ann, rel=1e-9)
    assert m["max_dd"] > 0


def test_load_base_guard(tmp_path, monkeypatch):
    from scripts.pipelines import rolling_grid_alla as R
    monkeypatch.setattr(R, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError, match="先跑 --stage prep"):
        R.load_base()
    _make_base(tmp_path)
    base = R.load_base()
    assert set(base) >= {"close", "mask", "bench_index", "bench_eqw"}
    assert base["close"].index[0] == pd.Timestamp("2018-01-01")


def test_warn_duplicate_base_detects_identical_arms(tmp_path, monkeypatch):
    """各臂 _base 字节相同 → 只警告、不阻断、不删除（2026-09-16 新增守卫）。"""
    from scripts.pipelines import rolling_grid_alla as R

    reports = tmp_path / "reports"
    arm_a = reports / "alla_rolling"
    arm_b = reports / "alla_rolling_ortho"
    for arm in (arm_a, arm_b):
        _make_base(arm)          # 两份内容完全相同（同函数生成）
    assert (arm_a / "_base" / "tradable_mask.parquet").exists()

    monkeypatch.chdir(tmp_path)  # `_warn_duplicate_base` 用相对路径 Path("reports")
    monkeypatch.setattr(R, "OUT", arm_b)

    records = []

    class _Sink:
        def warning(self, msg, *a):
            records.append(msg % a if a else msg)

    monkeypatch.setattr(R, "log", _Sink())
    R._warn_duplicate_base(arm_b / "_base")

    assert records, "相同内容应产生警告"
    assert "alla_rolling" in records[0]
    # 关键：只警告，两臂文件都还在（本函数永不删除）
    assert (arm_a / "_base" / "tradable_mask.parquet").exists()
    assert (arm_b / "_base" / "tradable_mask.parquet").exists()


def test_warn_duplicate_base_silent_when_no_sibling(tmp_path, monkeypatch):
    """无同内容兄弟臂 / 本臂首建 → 静默返回，不误报。"""
    from scripts.pipelines import rolling_grid_alla as R

    reports = tmp_path / "reports"
    arm = reports / "alla_rolling"
    arm.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(R, "OUT", arm)

    records = []

    class _Sink:
        def warning(self, msg, *a):
            records.append(msg % a if a else msg)

    monkeypatch.setattr(R, "log", _Sink())
    # 本臂 _base 还不存在（首建）
    R._warn_duplicate_base(arm / "_base")
    assert records == []

    # 建好本臂但无兄弟臂
    _make_base(arm)
    R._warn_duplicate_base(arm / "_base")
    assert records == []


def test_reserve_alt_subfamily_cov_and_dedup():
    """另类族保留席位：子族轮转 + 低覆盖率门槛 + 同值去重。"""
    from scripts.pipelines import rolling_grid_alla as R

    items = {
        # 事件族：两个 |IC| 完全相同（模拟同源公式 notice_profit_ttm/forecast_hit）
        "notice_forecast_hit": 0.0100,
        "notice_profit_ttm": 0.0100,
        "notice_sue": 0.0060,
        # 事件族 |IC| 最高但覆盖仅 1.8% → 应被 ALT_MIN_COVERAGE 挡掉
        "sue_express_20d": 0.0310,
        # 资金流族
        "lhb_count_20d": 0.0419,
        "margin_bal_chg_20d": 0.0266,
        "lhb_net_buy": 0.0003,
        # 状态族
        "limit_pos": 0.0261,
        "st_days": 0.0144,
        # 非另类族：不应入席
        "alpha101_001": 0.9000,
    }
    quality = pd.Series(items).sort_values(ascending=False)
    cov = pd.Series({
        "sue_express_20d": 0.018,        # < ALT_MIN_COVERAGE
        "notice_forecast_hit": 0.732, "notice_profit_ttm": 0.732,
        "notice_sue": 0.768, "lhb_count_20d": 0.071,
        "margin_bal_chg_20d": 0.403, "lhb_net_buy": 0.603,
        "limit_pos": 0.734, "st_days": 0.023, "alpha101_001": 1.0,
    })

    alt = R._reserve_alt(quality, cov)
    assert "alpha101_001" not in alt              # 非另类族不入席
    assert "sue_express_20d" not in alt           # 覆盖率门槛生效
    # 同值去重：notice_profit_ttm / notice_forecast_hit 只留一个
    assert not ({"notice_forecast_hit", "notice_profit_ttm"} <= set(alt))
    # 子族轮转：三族都有代表
    for sub in R.ALT_SUBFAMILIES:
        assert any(n in sub for n in alt)
    # 各族内按 |IC| 择优
    assert "lhb_count_20d" in alt and "limit_pos" in alt
    assert len(alt) <= R.RESERVED_ALT_SLOTS


def test_select_features_alt_slots_enter_selection(tmp_path):
    """另类族保留席位确实进入 selection；关开关则退回「量价+基本面」旧口径。"""
    from scripts.pipelines import rolling_grid_alla as R

    days = pd.bdate_range("2016-07-01", "2018-12-31")
    codes = [f"c{i}" for i in range(6)]
    rng = np.random.default_rng(7)

    ics = {"m0": 0.090, "m1": 0.080, "m2": 0.070,
           "lhb_count_20d": 0.042, "limit_pos": 0.026, "notice_sue": 0.007}
    ic_cache = pd.DataFrame(
        {n: np.full(len(days), v) for n, v in ics.items()}, index=days)

    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    for n in ics:
        pd.DataFrame(rng.normal(size=(len(days), len(codes))),
                     index=days, columns=codes).astype(np.float32).to_parquet(
            panels_dir / f"{n}.parquet")
    registry = pd.DataFrame({
        "name": list(ics),
        "coverage": [1.0, 1.0, 1.0, 0.071, 0.734, 0.768]})

    store = R.FeatureStore(panels_dir)
    feats = R.select_features_for_year(2018, 1, ic_cache, registry, store, days)
    # lhb_count_20d 覆盖率仅 7%：通用门槛 0.5 会砍掉，靠 ALT 席位进来
    assert "lhb_count_20d" in feats
    assert "limit_pos" in feats

    off = R.select_features_for_year(2018, 1, ic_cache, registry, store, days,
                                     include_alt=False)
    # 关开关后，低覆盖率因子失去席位保护 → 进不来
    assert "lhb_count_20d" not in off
    # 覆盖率达标者仍能走通用候选池（本就合理，不属席位保护范围）
    assert set(off) <= {"m0", "m1", "m2", "limit_pos", "notice_sue"}


def test_tradable_labels_switch_default_off():
    """训练标签掩码开关默认关闭（同 exclude 的零回归纪律：不静默改主实验口径）。"""
    from scripts.pipelines import rolling_grid_alla as R
    assert R.USE_TRADABLE_LABELS is False


def test_select_features_exclude_hard_drops_all_paths(tmp_path):
    """exclude 硬剔除覆盖候选池 + 保留席位；默认 None 时逐字零改动。

    消融臂正确性的前提：剔除一个**本来未入选**的特征必须结果逐字不变 ——
    否则「净贡献 = 0」这类结论会被选择噪声污染（ortho 臂正是这种极限情形）。
    """
    from scripts.pipelines import rolling_grid_alla as R

    days = pd.bdate_range("2016-07-01", "2018-12-31")
    codes = [f"c{i}" for i in range(6)]
    rng = np.random.default_rng(11)
    ics = {"m0": 0.090, "m1": 0.080, "limit_pos": 0.026, "notice_sue": 0.007}
    ic_cache = pd.DataFrame(
        {n: np.full(len(days), v) for n, v in ics.items()}, index=days)

    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    for n in ics:
        pd.DataFrame(rng.normal(size=(len(days), len(codes))),
                     index=days, columns=codes).astype(np.float32).to_parquet(
            panels_dir / f"{n}.parquet")
    registry = pd.DataFrame({"name": list(ics),
                             "coverage": [1.0, 1.0, 0.734, 0.768]})
    store = R.FeatureStore(panels_dir)

    base = R.select_features_for_year(2018, 1, ic_cache, registry, store, days)
    assert "limit_pos" in base                     # 另类族保留席位的入选者

    exc = R.select_features_for_year(2018, 1, ic_cache, registry, store, days,
                                     exclude={"limit_pos"})
    assert "limit_pos" not in exc                  # 候选池与席位两条路径都堵住
    assert set(exc) == set(base) - {"limit_pos"}   # 小池无替补，其余逐字保持

    # 剔除本来未入选的名字 = 空操作（ortho 臂的极限情形，净贡献须为精确 0）
    assert R.select_features_for_year(
        2018, 1, ic_cache, registry, store, days,
        exclude={"never_in_pool_x"}) == base
    # 显式空集 / None 与基线一致
    assert R.select_features_for_year(
        2018, 1, ic_cache, registry, store, days, exclude=None) == base
    assert R.select_features_for_year(
        2018, 1, ic_cache, registry, store, days, exclude=set()) == base


def test_alla_daily_dispatch_keys_disjoint():
    """分派白名单互斥，且另类键集合与 ALT_FAMILY_SETS 逐字对齐。"""
    from scripts.pipelines import alla_daily_rank as DR
    from scripts.pipelines import rolling_grid_alla as RG

    groups = {
        "holder": DR._HOLDER_NUM_KEYS | DR._HOLDER_TOP_KEYS,
        "pledge": DR._PLEDGE_KEYS,
        "constructed": DR._CONSTRUCTED_KEYS,
        "style": DR._STYLE_KEYS,
        "event": DR._EVENT_KEYS,
        "sue_pledge": DR._SUE_PLEDGE_KEYS,
        "margin": DR._MARGIN_KEYS,
        "moneyflow": DR._MONEYFLOW_KEYS,
        "disc_dyn": DR._DISC_HOLDER_DYN_KEYS,
        "status": DR._STATUS_KEYS,
    }
    seen: dict[str, str] = {}
    for g, keys in groups.items():
        for n in keys:
            assert n not in seen, f"{n} 同时登记在 {seen[n]} 与 {g}"
            seen[n] = g

    # 另类路径的键，剔除归基本面/股东族保管的 5 个后，应与 ALT_FAMILY_SETS 一致
    non_alt = {"pledge_chg_20d", "pledge_frn_density", "pledge_holder_density",
               "top10_hold_chg", "holder_stability"}
    alt_daily = (DR._EVENT_KEYS | DR._SUE_PLEDGE_KEYS | DR._MARGIN_KEYS
                 | DR._MONEYFLOW_KEYS | DR._DISC_HOLDER_DYN_KEYS
                 | DR._STATUS_KEYS) - non_alt
    assert alt_daily == RG.ALT_FAMILY_SETS


def test_yearly_rows_recompute_every_metric():
    """逐年行不得继承整体口径的 excess_idx / ir / turnover。

    回归 2026-09-14 修的 bug：此前三处调用都写 ``{**row, "year": year, **ym}``，
    而 row 里的 excess_idx/excess_eqw/ir/turnover 是**整体**值、_yearly_metrics
    又不产这几个键 → 逐年表这几列"每年完全相同"。这里用两年差异极大的
    换手/超额，断言逐年值真的不同且等于手算结果。
    """
    from scripts.pipelines.rolling_grid_alla import _yearly_rows

    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    rng = np.random.default_rng(7)
    dr = pd.Series(rng.normal(0.0002, 0.008, len(idx)), index=idx)
    bench = pd.Series(0.0, index=idx)
    # 2019 换手 1%，2020 换手 50% —— 若继承整体均值则两年会相同
    to = pd.Series(0.01, index=idx)
    to[to.index.year == 2020] = 0.50

    row = {"run_id": "x", "horizon": 1, "freq": "M", "neut": "raw",
           "frac": 0.2, "weight": "equal",
           "excess_idx": 0.1234, "ir": 9.99, "turnover": 0.9999}
    rows = _yearly_rows(row, dr, bench, to)
    assert [r["year"] for r in rows] == [2019, 2020]

    # 换手必须逐年重算（不是 0.9999，也不是两年同值）
    assert rows[0]["turnover"] == pytest.approx(0.01)
    assert rows[1]["turnover"] == pytest.approx(0.50)
    assert rows[0]["turnover"] != rows[1]["turnover"]

    # ir 必须是逐年算的，不能等于整体 9.99，且两年不同
    assert rows[0]["ir"] != pytest.approx(9.99)
    assert rows[0]["ir"] != pytest.approx(rows[1]["ir"])

    # excess_idx 与 _yearly_metrics 的 excess 一致、且等于手算（基准为 0）
    for r in rows:
        sub = dr[dr.index.year == r["year"]]
        expect = (1 + sub).prod() ** (252 / len(sub)) - 1
        assert r["excess_idx"] == pytest.approx(r["excess"], abs=1e-12)
        assert r["annual"] == pytest.approx(expect, rel=1e-9)

    # 只保留标识列，不把整体行整个塞进逐年行
    assert rows[0]["run_id"] == "x"
    assert "arm" not in rows[0]        # row 里没有的键不得凭空出现


def test_yearly_rows_tolerates_none_turnover():
    """换手序列缺失（部分回测路径）→ turnover 记 NaN，不抛异常。"""
    from scripts.pipelines.rolling_grid_alla import _yearly_rows

    idx = pd.bdate_range("2019-01-01", "2019-12-31")
    rng = np.random.default_rng(1)
    dr = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    rows = _yearly_rows({"run_id": "y"}, dr, pd.Series(0.0, index=idx), None)
    assert len(rows) == 1
    assert np.isnan(rows[0]["turnover"])


# ---------------------------------------------------------------------------
# 产物指纹（2026-09-23，P0 治本：exists-skip 静默陈旧）
# ---------------------------------------------------------------------------
def test_fp_sidecar_lifecycle(tmp_path):
    """指纹 sidecar：写入 → 匹配 → 内容不符/缺失/损坏 → 一律视为过期。"""
    from scripts.pipelines.rolling_grid_alla import _fp, _fp_match, _fp_write

    artifact = tmp_path / "eq__x.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    fp = _fp({"execution": "open", "costs": "[36, 0]"})
    _fp_write(artifact, fp, {"run_id": "x"})

    assert _fp_match(artifact, fp)                      # 命中
    assert not _fp_match(artifact, _fp({"execution": "close"}))  # 口径变了
    (tmp_path / "eq__x.csv.fp.json").unlink()
    assert not _fp_match(artifact, fp)                  # sidecar 缺失
    (tmp_path / "eq__x.csv.fp.json").write_text("{broken", encoding="utf-8")
    assert not _fp_match(artifact, fp)                  # sidecar 损坏


def test_fp_select_changes_on_config_and_neutralizes(tmp_path, monkeypatch):
    """selection 指纹对口径敏感：exclude/开关/常量/panels 源任一变化即失配。"""
    from scripts.pipelines import rolling_grid_alla as R

    monkeypatch.setattr(R, "OUT", tmp_path)
    monkeypatch.setattr(R, "PANELS_DIR", None)
    monkeypatch.setattr(R, "NAME_DIR", None)
    monkeypatch.setattr(R, "EXCLUDE_FEATURES", set())
    monkeypatch.setattr(R, "INCLUDE_FUNDAMENTAL", True)
    monkeypatch.setattr(R, "INCLUDE_ALT", True)

    fp0 = R._selection_fp()
    assert fp0 == R._selection_fp()                     # 稳定可复现

    for attr, val in [("EXCLUDE_FEATURES", {"limit_pos"}),
                      ("INCLUDE_ALT", False),
                      ("PANELS_DIR", tmp_path / "panels_neu"),
                      ("NAME_DIR", {"ln_mktcap": tmp_path / "p"})]:
        monkeypatch.setattr(R, attr, val)
        assert R._selection_fp() != fp0, f"{attr} 变化必须改变指纹"
        monkeypatch.setattr(R, attr, {"EXCLUDE_FEATURES": set(),
                                      "INCLUDE_ALT": True,
                                      "PANELS_DIR": None,
                                      "NAME_DIR": None}[attr])
    assert R._selection_fp() == fp0                     # 复原后回到基线


def test_fp_predict_changes_on_years_and_sel_files(tmp_path, monkeypatch):
    """pred 指纹：quick(2019) 与全量(2018-2026) 必须不同（防冒烟产物被
    全量复用）；selection 目录内容变化 → 指纹变。"""
    from scripts.pipelines import rolling_grid_alla as R

    monkeypatch.setattr(R, "OUT", tmp_path)
    monkeypatch.setattr(R, "EXCLUDE_FEATURES", set())
    monkeypatch.setattr(R, "PANELS_DIR", None)
    monkeypatch.setattr(R, "NAME_DIR", None)
    monkeypatch.setattr(R, "INCLUDE_ALT", True)
    monkeypatch.setattr(R, "USE_TRADABLE_LABELS", False)

    fp_quick = R._predict_fp([2019])
    fp_full = R._predict_fp(list(range(2018, 2027)))
    assert fp_quick != fp_full

    # selection 内容入指纹：同目录放一个 json 后指纹变；内容不变则稳定
    sel_dir = tmp_path / "selection"
    sel_dir.mkdir()
    (sel_dir / "y2019__h1.json").write_text('["m0", "m1"]', encoding="utf-8")
    fp_sel = R._predict_fp([2019])
    assert fp_sel != fp_quick
    assert R._predict_fp([2019]) == fp_sel

    # selection 内容变了 → pred 指纹变 → pred 重训
    (sel_dir / "y2019__h1.json").write_text('["m0", "m2"]', encoding="utf-8")
    assert R._predict_fp([2019]) != fp_sel


def test_fp_backtest_senses_pred_file_change(tmp_path):
    """eq 指纹绑定 pred 文件内容（size+mtime_ns）：pred 重训覆盖 → eq 失效。"""
    import os

    from scripts.pipelines.rolling_grid_alla import _backtest_fp

    pf = tmp_path / "gbdt__h1.parquet"
    pf.write_bytes(b"pred-v1")
    fp1 = _backtest_fp("open", pf)
    assert fp1 == _backtest_fp("open", pf)              # 未动 → 稳定（断点续跑命中）

    assert fp1 != _backtest_fp("close", pf)             # 执行价变了

    # pred 重训覆盖（模拟）：内容+时间戳变 → 指纹变 → eq 重算
    old = os.stat(pf)
    os.utime(pf, ns=(old.st_atime_ns, old.st_mtime_ns - 1_000_000))
    pf.write_bytes(b"pred-v2-longer")
    assert fp1 != _backtest_fp("open", pf)
