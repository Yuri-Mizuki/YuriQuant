"""分钟面板存储 + 统计特征层测试。

覆盖三块：
1. MinutePanelStore：长表 → 稠密面板 → 长表 往返一致性（含缺失 bar/停牌日）；
   跨年分区、区间切片、流式分块、覆盖统计。
2. intraday_features：向量化实现 vs 独立手写慢参照（逐日逐码循环），
   用 float64 面板对拍（避免 float32 精度噪声干扰正确性验证）。
3. 半拉天守卫：有效 bar 不足 min_bar_frac 的当日特征为 NaN。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.intraday import MinutePanelStore
from factor.intraday_features import extract_from_block

BARS_30MIN = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]

# 标准 A 股 5 分钟 bar 时刻表（bar 开始时刻语义，与真实 min5 面板一致）
_b5 = ([pd.Timestamp(f"2025-01-01 09:{m:02d}") for m in range(30, 60, 5)]
       + [pd.Timestamp(f"2025-01-01 10:{m:02d}") for m in range(0, 60, 5)]
       + [pd.Timestamp(f"2025-01-01 11:{m:02d}") for m in range(0, 30, 5)]
       + [pd.Timestamp(f"2025-01-01 13:{m:02d}") for m in range(0, 60, 5)]
       + [pd.Timestamp(f"2025-01-01 14:{m:02d}") for m in range(0, 60, 5)])
BARS_5MIN = [t.strftime("%H:%M") for t in _b5]
assert len(BARS_5MIN) == 48


def make_long(n_days: int = 8, n_codes: int = 3, seed: int = 3,
              drop: list[tuple[str, str]] | None = None,
              bars: list[str] | None = None) -> pd.DataFrame:
    """合成分钟长表；drop 指定剔除的 (date 'YYYYMMDD', code) 的 bar 序号。

    默认 8 个工作日从 2025-12-24 起 → 跨年（25 年 6 天 + 26 年 2 天）。
    bars 缺省 30 分钟表；传 BARS_5MIN 可构造 5 分钟面板（时段切片测试用）。
    """
    bars = BARS_30MIN if bars is None else bars
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-12-24", periods=n_days)  # 跨年：25 年 + 26 年
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    frames = []
    for c in codes:
        closes = []
        for d in dates:
            walk = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.004, len(bars))))
            closes.append(walk)
        for di, d in enumerate(dates):
            walk = closes[di]
            open_ = np.roll(walk, 1)
            open_[0] = walk[0] * (1 + rng.normal(0, 0.001))
            high = np.maximum(open_, walk) * (1 + np.abs(rng.normal(0, 0.001, len(walk))))
            low = np.minimum(open_, walk) * (1 - np.abs(rng.normal(0, 0.001, len(walk))))
            vol = rng.lognormal(9, 0.3, len(walk))
            rows = {
                "kline_time": [d.replace(hour=int(t[:2]), minute=int(t[3:])) for t in bars],
                "code": c, "open": open_, "high": high, "low": low,
                "close": walk, "volume": vol, "amount": vol * walk,
            }
            df = pd.DataFrame(rows)
            if drop:
                for dd, cc in drop:
                    if dd == d.strftime("%Y%m%d") and cc == c:
                        df = df.sample(frac=0.6, random_state=1)  # 缺 bar 的半拉天
            frames.append(df)
    out = pd.concat(frames, ignore_index=True).set_index(["kline_time", "code"])
    return out.sort_index()


@pytest.fixture()
def store(tmp_path):
    long_df = make_long(drop=[("20251224", "600001.SH")])
    s = MinutePanelStore.build(long_df, 30, "test", cache_root=tmp_path)
    return s


class TestMinutePanelStore:
    def test_roundtrip(self, store):
        """面板 → to_long 还原后与原长表（数值部分）一致。"""
        rebuilt = store.to_long()
        src = make_long(drop=[("20251224", "600001.SH")])
        joined = src.join(rebuilt, rsuffix="_r")
        for col in ("open", "high", "low", "close", "volume", "amount"):
            assert np.allclose(joined[col], joined[f"{col}_r"], rtol=1e-6)
        assert set(rebuilt.index) == set(src.index)

    def test_layout_and_nan(self, store):
        """shape / memmap / 停牌与缺 bar → NaN。"""
        assert store.years == [2025, 2026]
        assert store.n_bars == len(BARS_30MIN)
        assert store.bar_times[0] == "1000"
        arr = store.field("close", 2025)  # 默认 mmap
        assert isinstance(arr, np.memmap)
        n_days_2025 = len(store.days(2025))
        assert arr.shape == (n_days_2025, len(BARS_30MIN), 3)
        # 完整缺失的 (date, code)（drop 的 600001 首日只留 60% bar）与停牌格 → NaN
        close25 = np.asarray(arr)
        day0 = store.days(2025)[0]
        i0 = store.days(2025).index(day0)
        assert np.isnan(close25[i0]).any()  # 半拉天有缺失 bar

    def test_load_range(self, store):
        days_all = store.all_days()
        mid = days_all[len(days_all) // 2]
        got = store.load(("close",), begin_date=mid, end_date=mid)
        assert got["close"].shape[0] == 1
        got2 = store.load(("close",))
        assert got2["close"].shape[0] == len(days_all)
        assert store.load(("close",), 20270101, 20271231)["close"].shape[0] == 0

    def test_iter_blocks(self, store):
        seen: list[int] = []
        for chunk, block in store.iter_blocks(("close", "volume"), block_days=1):
            assert set(block) == {"close", "volume"}
            assert block["close"].shape[0] == len(chunk)
            seen.extend(chunk)
        assert seen == store.all_days()

    def test_coverage_stats(self, store):
        cov = store.coverage_stats()
        assert set(cov["year"]) == {2025, 2026}
        assert (cov["cell_coverage"] <= 1.0).all()
        assert cov["days"].sum() == len(store.all_days())

    def test_open_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MinutePanelStore.open(30, "nope", cache_root=tmp_path)


# ----------------------------------------------------------------------
# 特征层：向量化 vs 独立慢参照
# ----------------------------------------------------------------------
def _slow_reference(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """逐日逐码循环的独立实现（仅覆盖被对拍的特征子集）。"""
    out: dict[str, dict] = {k: {} for k in (
        "mom_intraday_ret", "mom_am_ret", "mom_trend_drift", "ret_mean", "ret_std",
        "ret_skew", "ret_kurt", "vol_rv", "vol_rsj", "shape_autocorr_l1",
        "shape_longest_pos_run", "pv_vwap_dev", "pv_amihud", "mom_pos_in_range",
        "shape_crossing_cnt", "candle_close_pos")}
    long_df = long_df.reset_index()
    long_df["date"] = long_df["kline_time"].dt.normalize()
    for (date, code), g in long_df.groupby(["date", "code"], sort=True):
        g = g.sort_values("kline_time")
        c, o = g["close"].to_numpy(), g["open"].to_numpy()
        h, lo_arr = g["high"].to_numpy(), g["low"].to_numpy()
        v, a = g["volume"].to_numpy(), g["amount"].to_numpy()
        r = np.empty(len(c))
        r[0] = c[0] / o[0] - 1
        r[1:] = c[1:] / c[:-1] - 1
        B = len(c)

        def put(name, val):
            out[name][(date, code)] = val

        put("mom_intraday_ret", c[-1] / c[0] - 1)
        am = np.array([t.strftime("%H%M") <= "1130" for t in g["kline_time"]])
        put("mom_am_ret", r[am].sum())
        slope = np.polyfit(np.arange(B), np.log(c), 1)[0]
        put("mom_trend_drift", slope * B)
        m = r.mean()
        put("ret_mean", m)
        put("ret_std", r.std(ddof=0))
        put("ret_skew", ((r - m) ** 3).mean() / ((r - m) ** 2).mean() ** 1.5)
        put("ret_kurt", ((r - m) ** 4).mean() / ((r - m) ** 2).mean() ** 2 - 3)
        put("vol_rv", (r * r).sum())
        up, dn = (r[r > 0] ** 2).sum(), (r[r < 0] ** 2).sum()
        put("vol_rsj", (up - dn) / (up + dn))
        d = r - m
        put("shape_autocorr_l1", (d[1:] * d[:-1]).sum() / (d * d).sum())
        best = cur = 0
        for x in r > 0:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        put("shape_longest_pos_run", best)
        vwap = ((h + lo_arr + c) / 3 * v).sum() / v.sum()
        put("pv_vwap_dev", c[-1] / vwap - 1)
        put("pv_amihud", (np.abs(r) / a).mean())
        put("mom_pos_in_range", (c[-1] - lo_arr.min()) / (h.max() - lo_arr.min()))
        put("shape_crossing_cnt", ((r[:-1] < m) & (r[1:] >= m)).sum())
        rng = h - lo_arr
        put("candle_close_pos", ((c - lo_arr) / rng).mean())

    panels = {name: pd.Series(vals) for name, vals in out.items()}
    panels = {name: s.unstack() for name, s in panels.items()}
    return panels


class TestIntradayFeatures:
    def test_against_slow_reference(self, tmp_path):
        long_df = make_long(n_days=8, n_codes=3, seed=11)
        # float64 构建：排除 float32 精度噪声，对拍纯逻辑正确性
        store = MinutePanelStore.build(long_df, 30, "ref", cache_root=tmp_path,
                                       float_dtype=np.float64)
        chunk_days = store.all_days()
        fields = store.load(("open", "high", "low", "close", "volume", "amount"))
        feats = extract_from_block(fields, chunk_days, store.codes,
                                   store.bar_times, store.period)
        ref = _slow_reference(long_df)

        for name, ref_panel in ref.items():
            got = feats[name]
            joined = ref_panel.stack(future_stack=True).align(got.stack(future_stack=True))
            a, b = joined
            mask = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
            assert mask.sum() > 0, name
            np.testing.assert_allclose(a.to_numpy()[mask], b.to_numpy()[mask],
                                       rtol=1e-8, atol=1e-12, err_msg=name)

    def test_min_bars_guard(self, tmp_path):
        """有效 bar < min_bar_frac×B 的当日特征为 NaN。"""
        long_df = make_long(n_days=4, n_codes=2, seed=5,
                            drop=[("20251224", "600001.SH"), ("20251225", "600001.SH")])
        store = MinutePanelStore.build(long_df, 30, "half", cache_root=tmp_path)
        fields = store.load(("open", "high", "low", "close", "volume", "amount"))
        feats = extract_from_block(fields, store.all_days(), store.codes,
                                   store.bar_times, store.period, min_bar_frac=0.9)
        for name, panel in feats.items():
            col = panel["600001.SH"]
            assert col.iloc[0:2].isna().all(), f"{name} 半拉天应 NaN"

    def test_feature_docs_cover_keys(self):
        """FEATURE_DOCS 与产出的特征 key 一致（入库元数据不漂移）。

        时段切片族（*_segNN_*）依赖 period（20 分钟窗须整除 bar 周期），
        在 30 分钟 mock 下不产出——故对非条件特征做全等、对 seg 特征做
        「产出 ⊆ docs」+ 独立的 5 分钟全产出测试（见 TestIntradaySegments）。
        """
        from factor.intraday_features import FEATURE_DOCS
        produced = {
            "mom_intraday_ret", "mom_open30_ret", "mom_close30_ret", "mom_am_ret",
            "mom_pm_ret", "mom_am_pm_diff", "mom_first_bar", "mom_last_bar",
            "mom_trend_drift", "mom_trend_r2", "mom_pos_in_range",
            "ret_mean", "ret_std", "ret_skew", "ret_kurt", "ret_min", "ret_max",
            "ret_median", "ret_abs_sum", "ret_rms", "ret_up_ratio",
            "vol_rv", "vol_rsj", "vol_downside_rms", "vol_range",
            "vol_extreme_bar_share",
            "shape_autocorr_l1", "shape_autocorr_l2", "shape_crossing_cnt",
            "shape_longest_pos_run", "shape_longest_neg_run",
            "pv_vol_trend", "pv_ret_vol_corr", "pv_vol_tail_ratio",
            "pv_big_move_vol_share", "pv_vol_skew", "pv_amihud", "pv_vwap_dev",
            "candle_high_in_am", "candle_upper_shadow", "candle_lower_shadow",
            "candle_close_pos",
        }
        assert produced <= set(FEATURE_DOCS)
        non_seg = {k for k in FEATURE_DOCS if "_seg" not in k}
        assert produced == non_seg


SEG_KEYS = {f"{p}_seg{k:02d}_{s}" for k in range(1, 13)
            for p, s in (("mom", "ret"), ("vol", "rvshare"), ("pv", "volshare"))}


class TestIntradaySegments:
    """时段切片矩阵族（20 分钟窗 × 动量/波动/量，36 个）。"""

    @pytest.fixture()
    def seg_feats(self, tmp_path):
        long_df = make_long(n_days=6, n_codes=3, seed=17, bars=BARS_5MIN)
        store = MinutePanelStore.build(long_df, 5, "seg", cache_root=tmp_path,
                                       float_dtype=np.float64)
        fields = store.load(("open", "high", "low", "close", "volume", "amount"))
        feats = extract_from_block(fields, store.all_days(), store.codes,
                                   store.bar_times, store.period)
        return store, fields, feats

    def test_all_seg_features_produced(self, seg_feats):
        _, _, feats = seg_feats
        assert SEG_KEYS <= set(feats)

    def test_window_boundaries(self, seg_feats):
        """窗边界对拍：seg02 = 位置 4..7 的 bar 收益和（抓 off-by-one 错位）。"""
        store, fields, feats = seg_feats
        r = fields["close"][:, 1:, :] / fields["close"][:, :-1, :] - 1.0
        first = fields["close"][:, 0:1, :] / fields["open"][:, 0:1, :] - 1.0
        r = np.concatenate([first, r], axis=1)
        manual = np.nansum(r[:, 4:8, :], axis=1)
        got = feats["mom_seg02_ret"].to_numpy()
        mask = np.isfinite(got)
        assert mask.sum() > 0
        np.testing.assert_allclose(got[mask], manual[mask], rtol=1e-10, atol=1e-12)

    def test_momentum_conservation(self, seg_feats):
        """12 窗动量之和 = 全日 48 根 bar 收益之和（切片无遗漏、无重复）。

        注意：窗动量是加法量（nansum），与 mom_intraday_ret（close末/首根
        close-1，含开盘缺口锚）之间没有精确恒等式——乘积守恒只对逐 bar
        收益链成立，对「窗内求和」带二阶交叉项。
        """
        store, fields, feats = seg_feats
        r = fields["close"][:, 1:, :] / fields["close"][:, :-1, :] - 1.0
        first = fields["close"][:, 0:1, :] / fields["open"][:, 0:1, :] - 1.0
        r = np.concatenate([first, r], axis=1)
        want = np.nansum(r, axis=1)
        total = sum(feats[f"mom_seg{k:02d}_ret"].to_numpy() for k in range(1, 13))
        mask = np.isfinite(total) & np.isfinite(want)
        assert mask.sum() > 0
        np.testing.assert_allclose(total[mask], want[mask],
                                   rtol=1e-9, atol=1e-12)

    def test_shares_sum_to_one(self, seg_feats):
        """RV 份额与量份额各自沿 12 窗求和 = 1（完整日）。"""
        _, _, feats = seg_feats
        for prefix in ("vol_seg", "pv_seg"):
            total = sum(feats[f"{prefix}{k:02d}_{'rvshare' if prefix == 'vol_seg' else 'volshare'}"]
                        .to_numpy() for k in range(1, 13))
            mask = np.isfinite(total)
            assert mask.sum() > 0
            np.testing.assert_allclose(total[mask], 1.0, rtol=1e-8, atol=1e-10)

    def test_skipped_for_30min(self, tmp_path):
        """20 分钟窗与 30 分钟 bar 不对齐 → 不产出时段特征（不硬凑半窗）。"""
        long_df = make_long(n_days=3, n_codes=2, seed=7)
        store = MinutePanelStore.build(long_df, 30, "segskip", cache_root=tmp_path)
        fields = store.load(("open", "high", "low", "close", "volume", "amount"))
        feats = extract_from_block(fields, store.all_days(), store.codes,
                                   store.bar_times, store.period)
        assert not (SEG_KEYS & set(feats))


class TestBatchDupReport:
    def test_daily_rank_corr_mean_vs_corrwith(self):
        """向量化逐日秩相关均值 vs pandas corrwith 慢参照（含 NaN/列不齐）。"""
        from scripts.factors.build_intraday_stat_factors import daily_rank_corr_mean

        rng = np.random.default_rng(21)
        T, C = 40, 25
        dates = pd.bdate_range("2025-01-01", periods=T)
        codes = [f"{i:06d}.SZ" for i in range(C)]

        def rand_panel(na_frac=0.1):
            a = rng.normal(size=(T, C))
            a[rng.random((T, C)) < na_frac] = np.nan
            return pd.DataFrame(a, index=dates, columns=codes)

        new_panels = {"f1": rand_panel(), "f2": rand_panel(0.3)}
        # f3 与 old 同源，相关应显著为正
        old = rand_panel(0.05)
        src = old.to_numpy()
        new_panels["f3"] = pd.DataFrame(
            np.where(np.isfinite(src), src + rng.normal(0, 0.5, (T, C)), np.nan),
            index=dates, columns=codes)

        new_stack = {k: v.rank(axis=1).to_numpy() for k, v in new_panels.items()}
        valid_union = {k: np.isfinite(v.to_numpy()) for k, v in new_panels.items()}
        got = daily_rank_corr_mean(new_stack, valid_union, old, dates, codes)

        for i, (name, p) in enumerate(new_panels.items()):
            # 慢参照：与库内 _run_dup_check 同口径（截面 rank 的逐日 pearson 均值）
            a = old.rank(axis=1)
            b = p.rank(axis=1)
            ref = a.corrwith(b, axis=1, method="pearson").mean()
            np.testing.assert_allclose(got[i], ref, rtol=1e-10, atol=1e-12, err_msg=name)
        assert got[2] > 0.3  # 同源因子相关应显著
