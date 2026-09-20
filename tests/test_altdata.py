"""另类数据层（data/altdata）单元测试。

**全部离线** —— 不触网、不依赖 akshare 已安装（akshare 只在抓取函数内部惰性导入）。
覆盖 2026-09-20 首轮真数据冒烟暴露的三类坑：

1. **代码标准化**：雪球返回 ``SH600328`` 前缀式；``920xxx`` 是北交所新代码段
   （不修正会让 join 项目面板的成功率掉到 0.2%）；
2. **去重键含 NaN 失效**：``NaN != NaN`` ⇒ 一次重抓就把表撑大（实测 5593→5870）；
3. **符号双重取反**：巨潮「变动数量」已带符号，再乘 direction 符号会翻方向。

前两类是纯函数/纯逻辑，必须由本测试长期锁住 —— 它们都不会抛异常，
只会静默产出错数据（本项目「静默降级」同型事故）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.altdata import AltDataCache
from data.altdata import source_news as news_src
from data.altdata.source_akshare import PIT_QUALITY, SUGGESTED_LAG_TRADING_DAYS, _code


# ---------------------------------------------------------------------------
# ① 代码标准化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("SH600328", "600328.SH"),      # 雪球前缀式（实测 inner_trade 的主力写法）
    ("SZ301171", "301171.SZ"),
    ("BJ430047", "430047.BJ"),
    ("600328", "600328.SH"),        # 纯 6 位
    ("688981", "688981.SH"),
    ("000001", "000001.SZ"),
    ("600328.SH", "600328.SH"),     # 已标准
    ("920242", "920242.BJ"),        # 北交所新段 —— to_code_std 会误判为 .SH
    ("920061", "920061.BJ"),
    ("430047", "430047.BJ"),
    ("830874", "830874.BJ"),
    ("870818", "870818.BJ"),
])
def test_code_normalization(raw, expected):
    assert _code(pd.Series([raw])).iloc[0] == expected


def test_code_handles_missing():
    # pandas 会把 map 出来的 None 归一成 NaN（dtype 转换）⇒ 用 pd.isna 判空，
    # 不能用 ``is None``。空串也必须为空（否则 to_code6("") 会产出 "000000.SZ"）。
    out = _code(pd.Series([None, np.nan, "", "600000.SH"]))
    assert pd.isna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2]), "空串应归为缺失，而不是造出 000000.SZ"
    assert out.iloc[3] == "600000.SH"


@pytest.mark.parametrize("raw", ["SH920242", "SH920061"])
def test_bj_stock_with_wrong_exchange_prefix(raw):
    """雪球把北交所股标成 SH920xxx —— 判市场须以**代码段**为准（项目口径）。"""
    assert _code(pd.Series([raw])).iloc[0].endswith(".BJ")


def test_bj_segment_matches_project_universe():
    """项目 code_info 里 920xxx / 43x / 83x / 87x 一律记为 .BJ（本项目口径的真源）。"""
    assert _code(pd.Series(["920375"])).iloc[0] == "920375.BJ"
    assert _code(pd.Series(["835228"])).iloc[0] == "835228.BJ"


# ---------------------------------------------------------------------------
# ② 去重（NaN 键不得让表膨胀）
# ---------------------------------------------------------------------------
def _hc_frame():
    return pd.DataFrame({
        "code": ["600000.SH", "000001.SZ", None],
        "chg_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "ann_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "controller": ["A", None, "C"],          # 含 NaN 键 —— 旧实现会失效
        "hold_ratio": [1.0, 2.0, 3.0],
    })


def test_merge_is_idempotent_with_nan_keys(tmp_path):
    """同一份数据重抓一次，行数必须不变（NaN 键也要正确判重）。"""
    cache = AltDataCache(tmp_path)
    local = _hc_frame()
    merged = cache._merge(local, local.copy(), "holder_control")
    assert len(merged) == len(local) == 3, (
        "含 NaN 的去重键失效 —— 会把同一行反复累积（实测 5593 -> 5870）")


def test_merge_keeps_new_and_adds_new_rows(tmp_path):
    cache = AltDataCache(tmp_path)
    local = _hc_frame()
    new = pd.concat([local.iloc[[0]], pd.DataFrame({
        "code": ["300750.SZ"], "chg_date": pd.to_datetime(["2025-06-01"]),
        "ann_date": pd.to_datetime(["2025-06-01"]), "controller": ["D"],
        "hold_ratio": [9.0]})], ignore_index=True)
    merged = cache._merge(local, new, "holder_control")
    assert len(merged) == 4


def test_merge_empty_new_returns_local(tmp_path):
    cache = AltDataCache(tmp_path)
    local = _hc_frame()
    assert len(cache._merge(local, pd.DataFrame(), "holder_control")) == 3


# ---------------------------------------------------------------------------
# ③ 新闻幂等键
# ---------------------------------------------------------------------------
def test_news_dedup_key_collapses_multi_source_reposts():
    df = pd.DataFrame({
        "pub_time": pd.to_datetime(["2026-09-20 10:00:00"] * 3),
        "title": ["同一标题"] * 3,
        "content": ["正文摘要" * 30] * 3,          # 前 80 字相同 ⇒ 视为同一条
        "url": [None, "http://a", "http://b"],
        "source": ["ths", "sina", "em"],
    })
    assert news_src.dedup_key(df).nunique() == 1


def test_news_dedup_key_distinguishes_different_times():
    df = pd.DataFrame({
        "pub_time": pd.to_datetime(["2026-09-20 10:00:00", "2026-09-20 10:00:01"]),
        "title": ["t", "t"], "content": ["c" * 100, "c" * 100],
    })
    assert news_src.dedup_key(df).nunique() == 2


# ---------------------------------------------------------------------------
# ④ 缓存路径与 meta
# ---------------------------------------------------------------------------
def test_news_path_is_per_day(tmp_path):
    cache = AltDataCache(tmp_path)
    p = cache.news_path("20260920")
    assert p.name == "news_20260920.parquet"
    assert p.parent.name == "alt_news"
    # 带横线也要归一
    assert cache.news_path("2026-09-20").name == "news_20260920.parquet"


def test_load_news_range_skips_missing_days(tmp_path):
    cache = AltDataCache(tmp_path)
    assert cache.load_news_range("20260901", "20260903").empty
    cache.append_news(date="20260902") if False else None   # 不触网：直接写盘
    df = pd.DataFrame({"pub_time": pd.to_datetime(["2026-09-02 10:00:00"]),
                       "title": ["t"], "content": ["c"], "url": [None], "source": ["ths"]})
    df.to_parquet(cache.news_path("20260902"), index=False)
    out = cache.load_news_range("20260901", "20260903")
    assert len(out) == 1


def test_macro_missing_forecast_days_flags_empty_forecast(tmp_path):
    cache = AltDataCache(tmp_path)
    cache._meta["macro_daily_stats"] = {
        "20260914": {"rows": 31, "forecast_notna": 0},    # 实测漏洞日
        "20260915": {"rows": 46, "forecast_notna": 1},    # 实测只有 1 条
        "20260916": {"rows": 47, "forecast_notna": 29},   # 正常
        "20260917": {"rows": 0, "forecast_notna": 0},     # 无事件，不算漏洞
    }
    miss = cache.macro_missing_forecast_days()
    assert list(miss["date"]) == ["20260914"]


def test_as_date_avoids_pandas3_format_kwarg_and_1970_trap():
    """``_as_date`` 必须同时绕开两个坑（pandas 3.x 已实测踩中过第一个）。"""
    from data.altdata.fetch import _as_date

    assert _as_date(20180101) == pd.Timestamp("2018-01-01")      # int 不当 ns 时间戳
    assert _as_date("20180101") == pd.Timestamp("2018-01-01")
    assert _as_date("2018-01-01") == pd.Timestamp("2018-01-01")
    ts = pd.Timestamp("2020-05-05")
    assert _as_date(ts) is ts                                    # Timestamp 原样返回


def test_meta_roundtrip(tmp_path):
    c1 = AltDataCache(tmp_path)
    c1._meta["macro_done_dates"] = ["20260920", "20260919"]
    c1._save_meta()
    c2 = AltDataCache(tmp_path)
    assert sorted(c2._meta["macro_done_dates"]) == ["20260919", "20260920"]


def test_save_meta_merges_disk_fields_and_never_clobbers(tmp_path):
    """复现 2026-09-20 事故：并发跑两个 --tables 时后写者抹掉先写者的字段。

    真实时序：B（cls 进程）**先启动**、加载当时的 ``_meta``（不含 macro 字段）；
    随后 A（macro 进程）写完 ``macro_done_dates``；最后 B 落盘。旧实现的
    「全量覆写」会因 B 的内存副本没有该字段而把它整段抹掉 ⇒ 水位丢失。
    修后 ``_save_meta`` 先重读磁盘合并 ⇒ A 的字段必须存活。
    """
    b = AltDataCache(tmp_path)              # B 先启动（此刻 meta 为空）
    a = AltDataCache(tmp_path)
    a._meta["macro_done_dates"] = ["20260901", "20260902"]
    a._save_meta()                          # A 写水位

    b._meta["cls_items"] = 42               # B 的内存副本无 macro 字段
    b._save_meta()                          # B 落盘

    c = AltDataCache(tmp_path)
    assert c._meta["macro_done_dates"] == ["20260901", "20260902"], "A 的字段被 B 抹掉"
    assert c._meta["cls_items"] == 42


def test_save_meta_memory_wins_on_same_key(tmp_path):
    """合并写时**内存为准**：本进程刚算出的值必须覆盖磁盘上的旧值。"""
    a = AltDataCache(tmp_path)
    a._meta["cls_items"] = 1
    a._save_meta()
    a._meta["cls_items"] = 99
    a._save_meta()
    b = AltDataCache(tmp_path)
    assert b._meta["cls_items"] == 99


# ---------------------------------------------------------------------------
# ⑤ PIT 口径常量（因子层依赖它们决定滞后，改动须显式）
# ---------------------------------------------------------------------------
def test_pit_quality_declares_no_ann_date_tables():
    assert PIT_QUALITY["mgmt_hold"] == "has_ann_date"        # 唯一有公告日的
    assert PIT_QUALITY["inner_trade"] == "no_ann_date"
    assert PIT_QUALITY["holder_control"] == "no_ann_date"
    # 无公告日的表必须有非零滞后，否则就是前视
    for t, q in PIT_QUALITY.items():
        if q == "no_ann_date":
            assert SUGGESTED_LAG_TRADING_DAYS[t] > 0, f"{t} 无公告日却零滞后 = 前视"


def test_every_pit_table_declares_lag():
    """每个登记的表都必须**同时**有 PIT 质量与滞后声明 —— 漏一个就是静默前视。

    这是「新增数据表」的检查清单：加表时忘登记 ``SUGGESTED_LAG_TRADING_DAYS``
    必须直接红掉，而不是靠人记得。
    """
    for t in PIT_QUALITY:
        assert t in SUGGESTED_LAG_TRADING_DAYS, f"{t} 缺滞后声明"
        assert SUGGESTED_LAG_TRADING_DAYS[t] >= 0, f"{t} 滞后不能为负"
    # 自建全历史表给的是**真实公告日** ⇒ 必须 has_ann_date 且零滞后
    assert PIT_QUALITY["cninfo_holder"] == "has_ann_date"
    assert SUGGESTED_LAG_TRADING_DAYS["cninfo_holder"] == 0


# ---------------------------------------------------------------------------
# ⑥ 因子构建的 PIT 滞后（builder 侧）
# ---------------------------------------------------------------------------
def test_apply_lag_shifts_by_trading_days():
    from scripts.builders.build_alla_holder_dyn_factors import _apply_lag

    cal = pd.date_range("2025-01-01", periods=30, freq="B")
    ev = pd.DataFrame({"code": ["600000.SH"], "ann_date": pd.to_datetime(["2025-01-06"]),
                       "signed_qty": [1000.0]})
    pos0 = int(cal.searchsorted(pd.Timestamp("2025-01-06"), side="left"))

    for lag in (0, 2, 60):
        out = _apply_lag(ev, lag, cal)
        if lag >= 28:                       # 60 交易日超出 30 日日历 ⇒ 应被丢弃
            assert out.empty, "滞后后超出日历的事件应被丢弃，而不是留在最后一天"
        else:
            assert len(out) == 1
            assert out["eff"].iloc[0] == cal[pos0 + lag]


# ---------------------------------------------------------------------------
# ⑨ `ann_date` dtype 契约（事件表 → 时间轴）
#
# 2026-09-20 实测事故：akshare 系表给 ``datetime64``，自建 cninfo_holder 按口径
# 要求给 **int YYYYMMDD**。直接 ``pd.DatetimeIndex(int_series)`` 会把整数当
# **纳秒**（→1970），于是 26 万条事件的 searchsorted 全返回 0，
# **全部被塞进日历第一天**，产出一个人造尖峰面板且不抛异常。
# ---------------------------------------------------------------------------
def test_ann_to_datetime_handles_all_three_dtypes():
    from scripts.builders.build_alla_holder_dyn_factors import ann_to_datetime

    want = pd.Timestamp("2024-03-15")
    for s in (pd.Series([20240315]),                      # int YYYYMMDD（自建表）
              pd.Series(["20240315"]),                    # str 紧凑
              pd.Series(["2024-03-15"]),                  # str 带横线
              pd.Series(pd.to_datetime(["2024-03-15"]))):  # datetime64（akshare 表）
        got = ann_to_datetime(s)
        assert pd.Timestamp(got.iloc[0]) == want, f"dtype={s.dtype} 解析错误：{got.iloc[0]}"


def test_apply_lag_accepts_int_ann_date():
    """int ``ann_date`` 必须与 datetime64 等价 —— 这是 dtype 事故的回归锁。"""
    from scripts.builders.build_alla_holder_dyn_factors import _apply_lag

    cal = pd.date_range("2024-01-01", periods=60, freq="B")
    base = {"code": ["600000.SH"], "signed_qty": [1000.0]}
    as_dt = _apply_lag(pd.DataFrame({**base, "ann_date": pd.to_datetime(["2024-03-15"])}), 0, cal)
    as_int = _apply_lag(pd.DataFrame({**base, "ann_date": [20240315]}), 0, cal)
    assert len(as_int) == 1 and len(as_dt) == 1
    assert as_int["eff"].iloc[0] == as_dt["eff"].iloc[0] == pd.Timestamp("2024-03-15")
    assert as_int["eff"].iloc[0] != cal[0], "int 被当成纳秒解析 ⇒ 落到日历第一天"


def test_apply_lag_drops_events_before_calendar_start():
    """日历起点之前的事件必须**丢弃**，不能钳到第一天。

    ``searchsorted`` 对「早于首元素」和「等于首元素」都返回 0，
    老写法只判 ``pos < len`` 就会把样本外事件全部堆到首日，
    制造跨截面假信号（实测 5 万条堆在 2016-07-01，面板 max 47.8）。
    """
    from scripts.builders.build_alla_holder_dyn_factors import _apply_lag

    cal = pd.date_range("2024-01-01", periods=60, freq="B")
    ev = pd.DataFrame({"code": ["600000.SH"] * 3, "signed_qty": [1.0] * 3,
                       "ann_date": [20150101, 20240315, 20180601]})   # 两条样本外
    out = _apply_lag(ev, 0, cal)
    assert len(out) == 1, f"样本外事件未被丢弃：{len(out)} 条"
    assert out["eff"].iloc[0] == pd.Timestamp("2024-03-15")
    assert (out["eff"] == cal[0]).sum() == 0, "有事件被钳到日历第一天"


def test_rolling_panel_is_not_a_single_day_spike():
    """面板必须**跨日有效**，而不是只有第一天有值（钳位事故的外观特征）。"""
    from scripts.builders.build_alla_holder_dyn_factors import (
        _apply_lag,
        _rolling_sum_panel,
    )

    cal = pd.date_range("2024-01-01", periods=40, freq="B")
    codes = ["600000.SH", "000001.SZ"]
    mask = pd.DataFrame(1.0, index=cal, columns=codes)
    ev = pd.DataFrame({"code": ["600000.SH"] * 4, "signed_qty": [100.0] * 4,
                       "ann_date": [20240105, 20240118, 20240201, 20240215]})
    ev = _apply_lag(ev, 0, cal)
    panel = _rolling_sum_panel(ev, "signed_qty", 20, cal, codes, mask)

    nz_rows = (panel.notna() & (panel != 0)).any(axis=1)
    assert nz_rows.sum() > 20, f"只有 {nz_rows.sum()} 个交易日有效，疑似被钳到首日"
    # 最后一次事件（2024-02-15）之后的 20 个交易日内都应有值
    assert nz_rows.iloc[-1], "滚动窗口应延续到日历末段"


# ---------------------------------------------------------------------------
# ⑦ 财联社电报（可回补历史源）
#
# 这一节最重要的是 **时区不变量**：2026-09-20 实测踩中「游标每页凭空前进 8 小时
# → 回补死循环」的静默 bug（naive Timestamp 的 .timestamp() 被 pandas 按 UTC 解释）。
# 下面两个测试是它的回归锁，任何人把 _ts_cst 简化回 .timestamp() 都会红。
# ---------------------------------------------------------------------------
_CST_MID_20260915 = 1789401600        # 2026-09-15 00:00:00+08:00 的 unix 秒


def _fake_roll(rows):
    """把 ``[(news_id, ctime, stock_list)]`` 造成 ``parse_cls_items`` 的输入。"""
    return [
        {"id": nid, "ctime": ct, "title": f"T{nid}", "content": f"【T{nid}】正文段",
         "level": "B", "stock_list": stocks or [],
         "subjects": [{"subject_name": "测试主题"}],
         "reading_num": 1, "share_num": 2, "comment_num": 3, "sort_score": ct}
        for nid, ct, stocks in rows
    ]


def test_ts_cst_is_cst_midnight_not_naive_utc():
    """``_ts_cst`` 必须是「北京零点」；naive ``.timestamp()`` 会少 8 小时。"""
    from data.altdata.fetch import _as_date, _ts_cst

    assert _ts_cst(20260915) == _CST_MID_20260915
    assert _ts_cst(20260915) == int(pd.Timestamp("2026-09-15", tz="Asia/Shanghai").timestamp())
    # 明确锁住「两个写法差 28800 秒」这件事 —— 差值一旦不是 28800，说明约定被改了
    assert int(_as_date(20260915).timestamp()) - _ts_cst(20260915) == 28800


def test_cls_parse_ctime_to_pub_time_is_absolute():
    """``ctime``（unix 秒）→ ``pub_time``（北京钟面）必须是绝对时刻，与机器时区无关。"""
    from data.altdata import source_cls

    df = source_cls.parse_cls_items(_fake_roll([
        (1, _CST_MID_20260915, [{"StockID": "sh600975", "name": "新五丰", "RiseRange": 0.19}]),
        (2, _CST_MID_20260915 + 3600, None),
    ]))
    assert list(df["ctime"]) == [_CST_MID_20260915, _CST_MID_20260915 + 3600]
    assert list(df["pub_time"]) == [pd.Timestamp("2026-09-15 00:00:00"),
                                   pd.Timestamp("2026-09-15 01:00:00")]
    assert df["stock_codes"].iloc[0] == "600975.SH"
    assert df["stock_names"].iloc[0] == "新五丰"
    assert pd.isna(df["stock_codes"].iloc[1])        # 无关联个股 = NaN（alt 惯例）


@pytest.mark.parametrize("raw,expected", [
    ("sh600975", "600975.SH"),
    ("sz300793", "300793.SZ"),
    ("bj430047", "430047.BJ"),
    ("SH600975", "600975.SH"),      # 大写前缀也认
    ("hk00700", "hk00700"),         # 非 A 股：保留原值，不硬塞市场后缀
    ("", None),
    (None, None),
])
def test_cls_normalize_stock_id(raw, expected):
    from data.altdata.source_cls import normalize_stock_id

    assert normalize_stock_id(raw) == expected


def test_cls_append_is_idempotent_and_splits_by_day(tmp_path):
    from data.altdata import source_cls

    df = source_cls.parse_cls_items(_fake_roll([
        (1, _CST_MID_20260915, None),
        (2, _CST_MID_20260915 + 3600, None),
        (3, _CST_MID_20260915 + 86400, None),        # 次日
    ]))
    c = AltDataCache(tmp_path)
    assert c.append_cls(df)["added"] == 3
    assert c.append_cls(df)["added"] == 0           # 幂等：news_id 去重
    assert sorted(p.name for p in c.cls_dir.glob("*.parquet")) == \
        ["cls_20260915.parquet", "cls_20260916.parquet"]

    loaded = c.load_cls()
    assert len(loaded) == 3
    assert loaded["news_id"].is_unique
    assert list(loaded["news_id"]) == [1, 2, 3]     # 按 ctime 升序


def test_cls_backfill_terminates_and_resumes(tmp_path, monkeypatch):
    """回补必须**收敛**（游标严格递减）且可续传。

    这是那个死循环的回归锁：用老的 ``pub_time.timestamp()`` 算游标时，
    游标每页 +8h 偏移 ⇒ 永远越不过 ``begin_ts`` ⇒ 本测试会撞上 ``max_pages``
    安全阀并因 ``reached_begin is False`` 失败（而不是把 pytest 挂死）。
    """
    from data.altdata import source_cls

    hour = 3600

    def fake_fetch(last_time, rn=50, **kw):
        cts = [int(last_time) - (i + 1) * hour for i in range(rn)]
        return source_cls.parse_cls_items(_fake_roll([(c, c, None) for c in cts]))

    monkeypatch.setattr(source_cls, "fetch_cls_page", fake_fetch)
    c = AltDataCache(tmp_path)

    st = c.backfill_cls(begin=20260915, end=20260916, flush_every=1,
                        sleep=0, max_pages=500, progress=False)
    assert st["reached_begin"] is True, f"回补未收敛（疑似游标偏移）：{st}"
    assert st["cursor"] < _CST_MID_20260915
    assert st["added"] > 0

    # 幂等重跑同一区间：不应新增任何行（news_id 去重）
    st2 = c.backfill_cls(begin=20260915, end=20260916, flush_every=1,
                         sleep=0, max_pages=500, progress=False, resume=False)
    assert st2["added"] == 0, f"重跑不该新增：{st2}"

    # 续传：水位已到 begin 之前，再跑应「无事可做」（cursor 起点在 begin 之前）
    st3 = c.backfill_cls(begin=20260915, end=20260916, flush_every=1,
                         sleep=0, max_pages=500, progress=False, resume=True)
    assert st3["pages"] == 0, f"续传应直接判定已到 begin：{st3}"


def test_cls_load_range_filters_by_day(tmp_path):
    from data.altdata import source_cls

    df = source_cls.parse_cls_items(_fake_roll([
        (1, _CST_MID_20260915, None),
        (2, _CST_MID_20260915 + 86400, None),
        (3, _CST_MID_20260915 + 2 * 86400, None),
    ]))
    c = AltDataCache(tmp_path)
    c.append_cls(df)

    assert list(c.load_cls(begin=20260916, end=20260916)["news_id"]) == [2]
    assert list(c.load_cls(begin=20260916)["news_id"]) == [2, 3]
    assert list(c.load_cls(end=20260915)["news_id"]) == [1]
    assert c.cls_coverage()["days"] == 3


def test_cls_shard_skips_schema_version_check(tmp_path):
    """新闻/电报存档是**追加型原始文本** —— 口径变更不该让历史失效。"""
    from data.altdata import source_cls

    c = AltDataCache(tmp_path)
    c._meta["schema_versions"] = {"cls": -1}        # 故意写个错的版本
    c._save_meta()
    c.append_cls(source_cls.parse_cls_items(_fake_roll([(1, _CST_MID_20260915, None)])))
    assert len(c.load_cls()) == 1                   # schema 校验不适用于 cls


# ---------------------------------------------------------------------------
# ⑧ 巨潮高管增减持全历史（P0 Step 1b）
#
# 两个坑各有一把锁：
#   · 响应是 list[dict]（缩码键），akshare 用**位置**赋列名 ⇒ 键序一变就静默错列；
#   · 撞 20000 行上限后的二分若按 YYYYMMDD **整数中点**切，会产生
#     ``20170666`` 这类非法日期，服务端不报错但整年数据消失。
# ---------------------------------------------------------------------------
#: 一条真实的 p_sysapi1030 record（2019-04-03 平安银行，键序即服务端返回序）
_CNINFO_REC = {
    "SECNAME": "平安银行", "DECLAREDATE": "2019-04-03", "HUMANNAME": "王晓洁",
    "F009N": None, "F008N": 13.28, "SECCODE": "000001", "F007N": 0.0001,
    "F006N": 1000, "ENDDATE": "2019-04-01", "F005N": None, "F004N": None,
    "F003V": "兄弟姐妹", "F002V": "独立董事", "F001V": "王松奇",
    "F011V": "临时公告", "F010V": "竞价交易",
}


def test_cninfo_normalize_maps_by_key_not_position():
    """列名必须按 **key** 取；键序打乱后结果必须逐位一致。"""
    from data.altdata import cninfo_holder as ch

    a = ch._normalize([_CNINFO_REC], "增持")
    shuffled = dict(reversed(list(_CNINFO_REC.items())))
    b = ch._normalize([shuffled], "增持")

    assert a.equals(b), "键序改变导致输出变化 ⇒ 说明仍在按位置映射"
    r = a.iloc[0]
    assert r["code"] == "000001.SZ"
    assert r["sec_name"] == "平安银行"
    assert r["ann_date"] == 20190403          # 公告日（PIT 对齐轴）
    assert r["chg_date"] == 20190401          # 截止日（变动日）
    assert r["person"] == "王晓洁"             # HUMANNAME = 变动人
    assert r["related_person"] == "王松奇"      # F001V = 关联董监高
    assert r["relation"] == "兄弟姐妹"
    assert r["duty"] == "独立董事"
    assert r["reason"] == "竞价交易"
    assert r["data_source"] == "临时公告"
    assert r["signed_qty"] == 1000
    assert r["amount"] == pytest.approx(1000 * 13.28)
    assert r["subject_type"] == "董监高"


def test_cninfo_direction_sign_and_bj_code():
    """减持必须为负；北交所代码要走 ``_std_code6`` 的 .BJ 修正。"""
    from data.altdata import cninfo_holder as ch

    rec = dict(_CNINFO_REC, F006N=-5000, SECCODE="920375", SECNAME="测试北交")
    out = ch._normalize([rec], "减持")
    assert out["signed_qty"].iloc[0] == -5000      # |−5000| × (−1)
    assert out["chg_qty"].iloc[0] == -5000         # 接口原值保留
    assert out["code"].iloc[0] == "920375.BJ"

    pos = ch._normalize([dict(rec, F006N=5000)], "增持")
    assert pos["signed_qty"].iloc[0] == 5000       # 增持为正


def test_cninfo_drops_rows_without_ann_date():
    """无公告日的行不能 PIT 对齐 ⇒ 必须丢弃（而不是留下当日生效）。"""
    from data.altdata import cninfo_holder as ch

    out = ch._normalize([_CNINFO_REC, dict(_CNINFO_REC, DECLAREDATE=None)], "增持")
    assert len(out) == 1


@pytest.mark.parametrize("raw,expected", [
    ("2019-04-03", 20190403),
    ("20190403", 20190403),
    ("2019/04/03", 20190403),
    (None, None),
    ("", None),
    ("2019-04", None),          # 不完整日期宁可丢，也不能猜
])
def test_cninfo_int_date_parsing(raw, expected):
    from data.altdata.cninfo_holder import _to_int_date

    assert _to_int_date(raw) == expected


def test_cninfo_recursion_split_lands_on_real_dates(monkeypatch):
    """**非法日期二分的回归锁**：所有被请求的区间端点都必须是真实日历日。

    老实现按 ``YYYYMMDD`` 整数中点切，会产生 ``20170666`` 这种端点：
    服务端照常返回、不抛异常，但过滤语义被破坏（实测整个 2010/2011/2013 消失）。
    这里拦截每次请求的端点做格式校验，任何非法日期都会让它红。
    """
    from data.altdata import cninfo_holder as ch

    seen: list[tuple[str, str]] = []

    def fake_post(sdate, edate, direction, **kw):
        # 注意：_post 收到的是 _fmt_date() 的 **int**，不是 str ——
        # 且 pd.Timestamp(int) 会当纳秒（1970 陷阱），必须 str() 后再解析。
        s, e = str(sdate), str(edate)
        seen.append((s, e))
        if (pd.Timestamp(e) - pd.Timestamp(s)).days > 200:
            return [], 999999              # total 远大于 records ⇒ 判为截断
        iso = f"{e[:4]}-{e[4:6]}-{e[6:]}"
        return [dict(_CNINFO_REC, DECLAREDATE=iso, ENDDATE=iso)], 1

    monkeypatch.setattr(ch, "_post", fake_post)
    ch.fetch_holder_trades(20100101, 20121231, "增持", sleep=0)

    assert len(seen) >= 4, f"宽区间应触发递归切分，实际只请求了 {seen}"
    for s, e in seen:
        for v in (s, e):
            assert len(v) == 8 and v.isdigit(), f"端点不是 YYYYMMDD：{v!r}"
            month, day = int(v[4:6]), int(v[6:8])
            assert 1 <= month <= 12, f"非法月份（整数中点二分的典型症状）：{v!r}"
            assert 1 <= day <= 31, f"非法日（整数中点二分的典型症状）：{v!r}"
        assert pd.Timestamp(s) <= pd.Timestamp(e), f"区间上下界颠倒：{s}~{e}"


def test_cninfo_refuses_incomplete_instead_of_silently_truncating(monkeypatch):
    """递归到自设深度仍被截断时必须**抛异常**，不能返回疑似不完整数据。"""
    from data.altdata import cninfo_holder as ch

    monkeypatch.setattr(ch, "_post",
                        lambda sdate, edate, direction, **kw: ([_CNINFO_REC], 999999))
    with pytest.raises(RuntimeError, match="拒绝返回不完整数据"):
        ch.fetch_holder_trades(20100101, 20121231, "增持", sleep=0, _max_depth=3)


def test_cninfo_year_coverage_gate():
    from data.altdata import cninfo_holder as ch

    df = pd.concat([
        ch._normalize([dict(_CNINFO_REC, DECLAREDATE="2019-04-03",
                            ENDDATE="2019-04-01")], "增持"),
        ch._normalize([dict(_CNINFO_REC, DECLAREDATE="2020-04-03",
                            ENDDATE="2020-04-01")], "增持"),
    ], ignore_index=True)
    cov = ch.year_coverage(df)
    assert dict(zip(cov["year"], cov["chg_rows"])) == {2019: 1, 2020: 1}

    ch.assert_no_empty_years(df, [2019, 2020])          # 齐全 ⇒ 通过
    with pytest.raises(RuntimeError, match="整年缺失"):
        ch.assert_no_empty_years(df, [2019, 2020, 2021])  # 2021 空洞 ⇒ 必须炸


def test_cninfo_overlap_windows_cover_each_year_twice():
    """stride-1 的宽窗口保证中间年份被覆盖两次（切窄会丢行，见模块头坑 3）。"""
    from data.altdata.cninfo_holder import iter_overlap_windows

    ws = list(iter_overlap_windows([2019, 2020, 2021], span=2))
    assert ws == [(20190101, 20201231), (20200101, 20211231)]
    for y in (2019, 2021):                       # 端点年只被覆盖一次
        assert sum(1 for b, e in ws if b // 10000 <= y <= e // 10000) == 1
    assert sum(1 for b, e in ws if b // 10000 <= 2020 <= e // 10000) == 2


def test_cninfo_holder_dedup_keys_are_registered():
    """新表必须同时登记到 _FILES 与 _DEDUP_KEYS，否则 _merge 会退化成全列去重。"""
    from data.altdata.fetch import _DEDUP_KEYS, _FILES

    assert "cninfo_holder" in _FILES
    assert _DEDUP_KEYS["cninfo_holder"]
    assert "ann_date" in _DEDUP_KEYS["cninfo_holder"]


