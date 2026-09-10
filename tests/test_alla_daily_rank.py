"""alla_daily_rank 每日排名脚本的核心纯函数单测（不依赖全A数据缓存）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_tail_n_days():
    from scripts.alla_daily_rank import HORIZON, TAIL_BUFFER, WARMUP, tail_n_days
    assert tail_n_days(500) == 500 + WARMUP + HORIZON + TAIL_BUFFER
    assert tail_n_days(750) - tail_n_days(500) == 250


def test_preprocess_panel():
    from scripts.alla_daily_rank import preprocess_panel
    idx = pd.bdate_range("2024-01-01", periods=3)
    p = pd.DataFrame({"a": [1.0, 1.0, np.inf], "b": [2.0, 4.0, -np.inf],
                      "c": [3.0, 3.0, 3.0]}, index=idx)
    out = preprocess_panel(p)
    assert out.dtypes.unique()[0] == np.float32
    # inf → NaN；第3行剩余唯一有效值 c 的截面 std 退化 → 整行 NaN（截面=行向）
    assert np.isnan(out.loc[idx[2], "a"]) and np.isnan(out.loc[idx[2], "b"])
    assert np.isnan(out.loc[idx[2], "c"])
    # 第1行 [1,2,3] 截面 zscore：b = (2-2)/1 = 0
    assert out.loc[idx[0], "b"] == pytest.approx(0.0, abs=1e-6)


def test_load_selection_exact_and_fallback(tmp_path):
    from scripts.alla_daily_rank import load_selection
    names = ["alpha158_KLEN", "bp"]
    (tmp_path / "y2026__h1.json").write_text(
        json.dumps(names), encoding="utf-8")
    got, year = load_selection(pd.Timestamp("2026-09-04"), tmp_path)
    assert got == names and year == 2026
    # 跨年缺失 → 回退最近年份
    got, year = load_selection(pd.Timestamp("2027-03-02"), tmp_path)
    assert got == names and year == 2026
    # 无任何文件 → 报错
    with pytest.raises(FileNotFoundError):
        load_selection(pd.Timestamp("2027-03-02"), tmp_path / "empty")


def _make_status(parquet_path: Path, predict_date: pd.Timestamp):
    codes = ["c0", "c1", "c2", "c3", "c4", "c5"]
    raw_close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], index=codes)
    rows = [
        # c0 正常；c1 停牌；c2 ST；c3 收盘=涨停价（封板）；c4 收盘<涨停价（未封）；
        # c5 收盘=跌停价（封板）
        dict(date=predict_date, code="c0", is_suspended=False, is_st=False,
             high_limited=None, low_limited=None),
        dict(date=predict_date, code="c1", is_suspended=True, is_st=False,
             high_limited=None, low_limited=None),
        dict(date=predict_date, code="c2", is_suspended=False, is_st=True,
             high_limited=None, low_limited=None),
        dict(date=predict_date, code="c3", is_suspended=False, is_st=False,
             high_limited=10.0, low_limited=8.0),
        dict(date=predict_date, code="c4", is_suspended=False, is_st=False,
             high_limited=15.0, low_limited=12.0),
        dict(date=predict_date, code="c5", is_suspended=False, is_st=False,
             high_limited=16.0, low_limited=14.0),
    ]
    st = pd.DataFrame(rows)
    st["date"] = pd.to_datetime(st["date"])
    st = st.set_index(["date", "code"])
    st.to_parquet(parquet_path)
    close_raw = pd.DataFrame([raw_close], index=[predict_date])
    return close_raw, codes


def test_signal_day_tradable(tmp_path):
    from scripts.alla_daily_rank import signal_day_tradable
    d = pd.Timestamp("2026-09-04")
    cache = tmp_path / "cache"
    cache.mkdir()
    close_raw, codes = _make_status(cache / "history_stock_status.parquet", d)

    mask = signal_day_tradable(close_raw, d, cache_root=str(cache))
    assert mask["c0"] and mask["c4"]              # 正常 / 未触涨停价
    assert not mask["c1"] and not mask["c2"]      # 停牌 / ST
    assert not mask["c3"]                         # 收盘 == 涨停价（封板）
    assert mask["c5"]                             # raw=15 在 lo=14 之上，未封跌停

    # 状态表缺失 → 全可交易
    mask2 = signal_day_tradable(close_raw, d, cache_root=str(tmp_path / "none"))
    assert mask2.all()

    # 状态表存在但预测日无行 → 降级日线推断（行情有该日则推断，见 fallback 测试）；
    # 行情也无该日（降级不可用）→ 全放行
    cache2 = tmp_path / "cache2"
    cache2.mkdir()
    _make_status(cache2 / "history_stock_status.parquet", d)
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-09-03", "2026-09-04"]), ["c0"]],
        names=["date", "code"])
    pd.DataFrame({"close": [10.0, 10.0], "volume": [1e6, 1e6]},
                 index=idx).to_parquet(cache2 / "daily_all_a.parquet")
    mask3 = signal_day_tradable(close_raw, d + pd.Timedelta(days=1),
                                cache_root=str(cache2))
    assert mask3.all()


def test_build_ranking():
    from scripts.alla_daily_rank import build_ranking
    codes = [f"c{i}" for i in range(10)]
    scores = pd.Series(np.arange(10.0), index=codes)        # c9 最高
    tradable = pd.Series([i % 3 != 0 for i in range(10)], index=codes)
    ranking, picks = build_ranking(scores, tradable, frac=0.2)

    assert list(ranking.index[:3]) == ["c9", "c8", "c7"]
    assert ranking["rank"].iloc[0] == 1
    assert ranking["score"].is_monotonic_decreasing
    k = max(1, int(round(10 * 0.2)))
    assert ranking["top_frac"].sum() == k == 2
    assert ranking.loc["c9", "pct_rank"] == pytest.approx(1.0)
    # picks = top_frac ∧ tradable；等权
    assert set(picks.index) <= set(ranking.index[ranking["top_frac"]])
    assert all(tradable[c] for c in picks.index)
    if len(picks):
        assert np.allclose(picks["weight"], 1.0 / len(picks))
    # NaN 分数剔除
    scores2 = scores.copy()
    scores2["c9"] = np.nan
    ranking2, _ = build_ranking(scores2, tradable, frac=0.2)
    assert "c9" not in ranking2.index and len(ranking2) == 9


def test_build_ranking_annotations():
    from scripts.alla_daily_rank import build_ranking
    codes = ["c0", "c1", "c2", "c3"]
    scores = pd.Series([4.0, 3.0, 2.0, 1.0], index=codes)
    tradable = pd.Series(True, index=codes)
    names = pd.Series({"c0": "贵州茅台", "c1": "宁德时代", "c3": "中国平安"})
    industry = pd.Series({"c0": "食品饮料", "c1": "电力设备", "c2": "电子"})

    ranking, picks = build_ranking(scores, tradable, frac=0.5,
                                   names=names, industry=industry)
    # 名称列：缺失的 c2 填空串
    assert ranking.loc["c0", "name"] == "贵州茅台"
    assert ranking.loc["c2", "name"] == ""
    # 行业列：缺失/空值 → "未知"
    assert ranking.loc["c3", "industry"] == "未知"
    assert ranking.loc["c2", "industry"] == "电子"
    # 列顺序：rank/name/industry 在最前，picks 继承标注列
    assert list(ranking.columns[:3]) == ["rank", "name", "industry"]
    assert {"name", "industry"} <= set(picks.columns)
    # 不传标注 → 无对应列
    ranking2, _ = build_ranking(scores, tradable, frac=0.5)
    assert "name" not in ranking2.columns and "industry" not in ranking2.columns


def test_industry_series():
    from scripts.alla_daily_rank import industry_series
    d = pd.Timestamp("2026-09-04")
    panel = pd.DataFrame(
        {"c0": ["801120"], "c1": ["801760"], "c2": [np.nan]},
        index=pd.DatetimeIndex([d]))
    name_map = pd.Series({"801120": "食品饮料", "801760": "电力设备"})
    got = industry_series(panel, d, name_map)
    assert got["c0"] == "食品饮料" and got["c1"] == "电力设备"
    # 无行业归属的股票不出现在返回中（下游 reindex 后落为 NaN → "未知"）
    assert "c2" not in got.index
    # 缺名表 → 回退行业代码
    got2 = industry_series(panel, d, pd.Series(dtype=object))
    assert got2["c0"] == "801120"
    # 空面板 / 非预测日 → 空
    assert industry_series(None, d, name_map).empty
    assert industry_series(panel, d + pd.Timedelta(days=1), name_map).empty


def test_build_industry_table():
    from scripts.alla_daily_rank import build_industry_table
    codes = ["c0", "c1", "c2", "c3", "c4"]
    ranking = pd.DataFrame({
        "rank": range(1, 6),
        "name": ["龙头A", "龙头B", "", "", ""],
        "industry": ["食品饮料", "食品饮料", "电子", "电子", "电子"],
        "score": [3.0, 1.0, 0.5, 0.0, -0.5],
        "pct_rank": [1.0, 0.8, 0.6, 0.4, 0.2],
        "top_frac": [True, False, True, False, False],
    }, index=codes)

    out = build_industry_table(ranking)
    # 按 mean_score 降序：食品饮料 mean=2.0 > 电子 mean=0.0
    assert list(out.index) == ["食品饮料", "电子"]
    assert list(out["rank"]) == [1, 2]
    assert out.loc["食品饮料", "n_stocks"] == 2
    assert out.loc["食品饮料", "mean_score"] == pytest.approx(2.0)
    assert out.loc["电子", "top_frac_share"] == pytest.approx(1 / 3, abs=5e-5)
    # top_stock = 行业内全A排名最高的成员（带名称时 "code 名称"）
    assert out.loc["食品饮料", "top_stock"] == "c0 龙头A"
    assert out.loc["电子", "top_stock"] == "c2"
    # 无行业列 → 全部并入"未知"一行
    out2 = build_industry_table(ranking.drop(columns=["industry"]))
    assert list(out2.index) == ["未知"] and out2.loc["未知", "n_stocks"] == 5


def _mk_daily(tmp_path) -> Path:
    """构造最小 daily_all_a.parquet（主板/创业板/北交所 × 2 日）。"""
    root = tmp_path
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-09-04", "2026-09-07"]),
         ["600000.SH", "300001.SZ", "830001.BJ"]],
        names=["date", "code"])
    daily = pd.DataFrame({
        "open":   [10.00, 20.00, 5.00, 11.00, 21.00, 5.00],
        "high":   [10.50, 20.50, 5.10, 11.00, 21.50, 5.00],
        "low":    [9.90, 19.80, 4.95, 11.00, 20.90, 5.00],
        "close":  [10.00, 20.00, 5.00, 11.00, 21.00, 5.00],
        "volume": [1e6, 2e6, 3e6, 5e5, 2e6, 0.0],
    }, index=idx)
    p = root / "daily_all_a.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(p)
    return root


def test_limit_from_daily(tmp_path):
    from scripts.alla_daily_rank import _limit_from_daily
    root = _mk_daily(tmp_path)
    close_raw = pd.DataFrame({
        "600000.SH": [10.0, 11.0], "300001.SZ": [20.0, 21.0],
        "830001.BJ": [5.0, 5.0]},
        index=pd.to_datetime(["2026-09-04", "2026-09-07"]))
    hi, lo, susp = _limit_from_daily(
        close_raw, pd.Timestamp("2026-09-07"), cache_root=str(root))
    # 主板 ±10% / 创业板 ±20% / 北交所 ±30%
    assert hi["600000.SH"] == 11.00 and hi["300001.SZ"] == 24.00
    assert hi["830001.BJ"] == 6.50 and lo["830001.BJ"] == 3.50
    # 停牌：成交量 0
    assert susp["830001.BJ"] and not susp["600000.SH"]
    # 行情无预测日 → None
    assert _limit_from_daily(close_raw, pd.Timestamp("2026-09-30"),
                             cache_root=str(root)) is None


def test_signal_day_tradable_fallback(tmp_path):
    """状态表缺预测日 → 降级日线推断：一字板/停牌不可交易，正常股可交易。"""
    from scripts.alla_daily_rank import signal_day_tradable
    root = _mk_daily(tmp_path)
    # 状态表只到 09-02（预测日 09-07 缺失 → 触发降级）
    st_idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-09-02"]), ["600000.SH", "300001.SZ"]],
        names=["date", "code"])
    st = pd.DataFrame({
        "pre_close": [10.0, 20.0], "high_limited": [11.0, 24.0],
        "low_limited": [9.0, 16.0], "is_st": [False, False],
        "is_suspended": [False, False]}, index=st_idx)
    st.to_parquet(root / "history_stock_status.parquet")

    close_raw = pd.DataFrame({
        "600000.SH": [10.0, 11.0], "300001.SZ": [20.0, 21.0],
        "830001.BJ": [5.0, 5.0]},
        index=pd.to_datetime(["2026-09-04", "2026-09-07"]))
    tradable = signal_day_tradable(close_raw, pd.Timestamp("2026-09-07"),
                                   cache_root=str(root))
    # 600000 收盘 11.00 = 涨停价 11.00 → 一字封板不可交易
    assert not tradable["600000.SH"]
    # 300001 收盘 21.00 < 24.00 → 可交易
    assert tradable["300001.SZ"]
    # 830001 停牌（volume=0）→ 不可交易
    assert not tradable["830001.BJ"]


def test_write_outputs(tmp_path):
    from scripts.alla_daily_rank import write_outputs
    d = pd.Timestamp("2026-09-04")
    ranking = pd.DataFrame({
        "rank": [1, 2], "name": ["贵州茅台", "宁德时代"],
        "industry": ["食品饮料", "电力设备"],
        "score": [3.0, 1.0], "pct_rank": [1.0, 0.9],
        "top_frac": [True, False], "tradable": [True, True],
    }, index=["c0", "c1"])
    picks = ranking[ranking["top_frac"]].copy()
    ind = pd.DataFrame({"rank": [1], "n_stocks": [2], "mean_score": [2.0]},
                       index=["食品饮料"])
    paths = write_outputs(ranking, picks, d, {"n_scored": 2},
                          industry_table=ind, out_dir=tmp_path)

    assert "industry" in paths
    assert (tmp_path / "industry_rank_20260904.csv").exists()
    assert (tmp_path / "latest_industry_rank.csv").exists()
    # 排名/持仓 CSV 含中文名列（utf-8-sig 回读）
    rk = pd.read_csv(tmp_path / "ranking_20260904.csv", index_col=0)
    assert rk.loc["c0", "name"] == "贵州茅台"
    # 无行业表 → 不产出 industry 键
    paths2 = write_outputs(ranking, picks, d, {}, out_dir=tmp_path)
    assert "industry" not in paths2


def test_write_latest_locked(tmp_path, monkeypatch):
    """latest 副本被外部进程占用时：告警并继续，不中断主输出。"""
    import scripts.alla_daily_rank as mod
    d = pd.Timestamp("2026-09-04")
    ranking = pd.DataFrame({
        "rank": [1], "name": ["贵州茅台"], "score": [3.0],
        "pct_rank": [1.0], "top_frac": [True], "tradable": [True],
    }, index=["c0"])
    picks = ranking.copy()

    def _boom(src, dst):
        raise PermissionError(13, "used by another process", str(dst))

    # 单个副本：被占用 → 返回 False 不抛异常（仅本块内劫持 os.replace）
    with monkeypatch.context() as m:
        m.setattr(mod.os, "replace", _boom)
        src = tmp_path / "x.csv"
        src.write_text("a", encoding="utf-8-sig")
        assert mod._write_latest(tmp_path / "latest_ranking.csv", src) is False

    # 整体输出：latest 副本失败（已内部吞掉）不中断，主文件照常产出
    monkeypatch.setattr(mod, "_write_latest", lambda p, src: False)
    paths = mod.write_outputs(ranking, picks, d, {}, out_dir=tmp_path)
    assert (tmp_path / "ranking_20260904.csv").exists()
    assert (tmp_path / "history.csv").exists()
    assert paths["ranking"].endswith("ranking_20260904.csv")


def test_write_main_locked_raises(tmp_path, monkeypatch):
    """主输出被占用（用户打开着 CSV）→ 明确报错而非静默/半写。"""
    import scripts.alla_daily_rank as mod
    d = pd.Timestamp("2026-09-04")
    ranking = pd.DataFrame({
        "rank": [1], "name": ["贵州茅台"], "score": [3.0],
        "pct_rank": [1.0], "top_frac": [True], "tradable": [True],
    }, index=["c0"])
    monkeypatch.setattr(mod, "_write_table", lambda p, df: False)
    with pytest.raises(PermissionError, match="被其他进程占用"):
        mod.write_outputs(ranking, ranking.copy(), d, {}, out_dir=tmp_path)


def test_append_history_idempotent(tmp_path):
    from scripts.alla_daily_rank import append_history
    row1 = {"predict_date": "20260904", "n_scored": 5000, "runtime_sec": 100.0}
    row2 = {"predict_date": "20260905", "n_scored": 5010, "runtime_sec": 90.0}
    append_history(row1, out_dir=tmp_path)
    append_history(row2, out_dir=tmp_path)
    df = pd.read_csv(tmp_path / "history.csv", dtype={"predict_date": str})
    assert list(df["predict_date"]) == ["20260904", "20260905"]

    # 同日重跑 → 覆盖不追加
    row1b = {"predict_date": "20260904", "n_scored": 5020, "runtime_sec": 80.0}
    append_history(row1b, out_dir=tmp_path)
    df = pd.read_csv(tmp_path / "history.csv", dtype={"predict_date": str})
    assert list(df["predict_date"]) == ["20260904", "20260905"]
    assert df.loc[df["predict_date"] == "20260904", "n_scored"].iloc[0] == 5020


def test_feature_dispatch_sets():
    """选择文件里的名字必须能被五条路径之一接住（含 2026-09-08 重选后的实际组合）。"""
    from scripts.alla_daily_rank import (
        _ALPHA_PREFIXES,
        _CONSTRUCTED_KEYS,
        _HOLDER_NUM_KEYS,
        _HOLDER_TOP_KEYS,
        _PLEDGE_KEYS,
        compute_features,
    )
    names = ["bp", "sp_ttm", "div_yield", "ep_ttm", "holder_num_yoy",
             "ln_mktcap", "holder_num_chg", "pcf_ttm", "alpha158_KLEN",
             "alpha360_VOLUME48", "altman_zscore", "pledge_ratio"]
    alpha = [n for n in names if n.startswith(_ALPHA_PREFIXES)]
    holder = [n for n in names if n in (_HOLDER_NUM_KEYS | _HOLDER_TOP_KEYS)]
    pledge = [n for n in names if n in _PLEDGE_KEYS]
    constructed = [n for n in names if n in _CONSTRUCTED_KEYS]
    fund = [n for n in names
            if n not in alpha and n not in holder and n not in pledge
            and n not in constructed]
    assert len(alpha) == 2 and len(holder) == 2 and len(fund) == 6
    assert constructed == ["altman_zscore"] and pledge == ["pledge_ratio"]
    # compute_features 的分派逻辑是纯集合划分，这里静态校验划分无遗漏
    assert set(alpha) | set(holder) | set(fund) | set(pledge) | set(constructed) \
        == set(names)
    # 防呆：修改分派集合时保持函数签名可导入
    assert callable(compute_features)


def test_glossary_covers_selection_2026():
    """当年选择文件的全部 50 因子都有中文释义（防漏表）。"""
    import json
    from pathlib import Path

    from scripts.alla_daily_rank import lookup_glossary
    p = (Path(__file__).resolve().parents[1] / "reports" / "alla_rolling"
         / "selection" / "y2026__h1.json")
    if not p.exists():
        # reports/ 不入库（实验产物），CI 上没有该文件——守卫的是本机漏表，
        # 缺文件时跳过而非失败（与 test_e2e_pipeline / test_investment_report 同惯例）
        pytest.skip("本机无 alla_rolling 当年选择文件")
    names = json.loads(p.read_text(encoding="utf-8"))
    assert len(names) == 50
    missing = [n for n in names if lookup_glossary(n)[1] == "未收录释义"]
    assert not missing, f"缺释义的因子: {missing}"


def test_explain_features():
    from scripts.alla_daily_rank import explain_features, factor_family
    imp = pd.Series({"alpha360_VOLUME48": 8.0, "bp": 1.0, "ln_mktcap": 1.0})
    out = explain_features(imp)
    assert list(out["feature"]) == ["alpha360_VOLUME48", "bp", "ln_mktcap"]
    assert list(out["rank"]) == [1, 2, 3]
    # share 归一化到 1，cum_share 递增
    assert out["share"].sum() == pytest.approx(1.0)
    assert list(out["cum_share"]) == [0.8, 0.9, 1.0]
    # 中文名与释义（VOLUME 通配 + 静态表 + 族）
    assert out.loc[0, "name_cn"] == "48日量能回溯"
    assert "成交量" in out.loc[0, "description"]
    assert out.loc[1, "name_cn"] == "账面市值比"
    assert factor_family("bp") == "估值因子" and factor_family("ln_mktcap") == "规模因子"
    assert factor_family("alpha360_VOLUME48") == "量价因子"


def test_explain_stocks():
    from scripts.alla_daily_rank import explain_stocks
    codes = ["c0", "c1"]
    contrib = pd.DataFrame({
        "bp": [0.8, -0.2], "alpha158_KLEN": [-0.2, 0.5],
        "alpha360_VOLUME48": [-0.3, 0.1], "bias": [0.0, 0.0]},
        index=codes)
    z_scores = {"bp": pd.Series({"c0": 1.5, "c1": -0.5}),
                "alpha158_KLEN": pd.Series({"c0": 1.0, "c1": -2.0}),
                "alpha360_VOLUME48": pd.Series({"c0": 0.5, "c1": 0.3})}
    ranking = pd.DataFrame({"rank": [1, 2], "name": ["股A", "股B"],
                            "industry": ["食品饮料", "电子"],
                            "score": [2.0, 1.0]}, index=codes)
    out = explain_stocks(contrib, z_scores, ranking, n_factors=2)
    assert list(out["code"]) == codes
    # c0 主驱动 = |贡献| 最大的 bp(+0.8)，次驱动 VOLUME48(-0.3) 为拖累
    assert out.loc[0, "drv1_feature"] == "bp" and out.loc[0, "drv1_z"] == 1.5
    assert out.loc[0, "drv2_feature"] == "alpha360_VOLUME48"
    assert "主要驱动" in out.loc[0, "summary"] and "拖累" in out.loc[0, "summary"]
    assert "z=+1.50" in out.loc[0, "summary"]
    # c1 主驱动 KLEN(+0.5) 但 z=-2.0 → 低值看多标注
    assert out.loc[1, "drv1_feature"] == "alpha158_KLEN"
    assert out.loc[1, "drv1_name"] == "K线长度"
    assert "低值看多" in out.loc[1, "summary"]
    # 列集合固定：code + 标注 + 2 组驱动 + summary
    assert list(out.columns) == ["code", "name", "industry", "rank", "score",
                                 "drv1_feature", "drv1_name", "drv1_contrib", "drv1_z",
                                 "drv2_feature", "drv2_name", "drv2_contrib", "drv2_z",
                                 "summary"]


def test_build_leaders():
    from scripts.alla_daily_rank import build_leaders

    codes = [f"c{i}" for i in range(6)]
    ranking = pd.DataFrame({
        "rank": list(range(1, 7)),   # 全A名次（build_ranking 产物），应被层内名次替换
        "name": [f"股{i}" for i in range(6)],
        "industry": ["电子"] * 6,
        "score": [2.0, 1.8, 1.5, 1.2, 0.9, 0.5],
        "tradable": [True] * 6,
    }, index=codes)
    mktcap = pd.Series({"c0": 50.0, "c1": 900.0, "c2": 800.0, "c3": 700.0,
                        "c4": 600.0, "c5": 100.0})  # 亿元；c5/c0 为小票
    out = build_leaders(ranking, mktcap, n_leaders=4, top_k=2)
    # 市值前 4 = c1,c2,c3,c4；其中模型分前 2 = c1(1.8), c2(1.5)
    assert list(out.index) == ["c1", "c2"]
    assert list(out["rank"]) == [1, 2]
    # cap_rank 为全市场市值名次（c1=1, c2=2, c3=3, c4=4）
    assert list(out["cap_rank"]) == [1, 2]
    assert list(out["mktcap"]) == [900.0, 800.0]
    # 小票 c0/c5 被市值分层排除
    assert "c5" not in out.index and "c0" not in out.index
    # 列集合固定
    assert list(out.columns) == ["rank", "cap_rank", "name", "industry",
                                 "mktcap", "score", "tradable"]


def test_compute_market_cap_scale(tmp_path):
    from scripts.alla_daily_rank import compute_market_cap

    idx = pd.date_range("2026-09-04", periods=2)
    codes = ["600001.SH", "600002.SH"]
    close_raw = pd.DataFrame([[10.0, 20.0], [10.0, 20.0]], idx, codes)
    # TOT_SHARE=10 亿股（balance 表，ann_date 前向填充）→ 市值 = 10 × 10 = 100 亿
    balance = pd.DataFrame({
        "code": ["600001.SH", "600002.SH"],
        "ann_date": [pd.Timestamp("2026-03-31")] * 2,
        "report_period": [pd.Timestamp("2025-12-31")] * 2,
        "TOT_SHARE": [1e9, 1e9],
    })
    tables = {"income": pd.DataFrame(), "balance": balance,
              "cashflow": pd.DataFrame(), "equity": pd.DataFrame(),
              "dividend": pd.DataFrame()}
    cap = compute_market_cap(close_raw, tables)
    assert list(cap.index) == list(idx)
    assert list(cap.columns) == codes
    assert cap.iloc[0, 0] == pytest.approx(100.0)   # 10元 × 10亿股 / 1e8
    assert cap.iloc[0, 1] == pytest.approx(200.0)   # 20元 × 10亿股 / 1e8
