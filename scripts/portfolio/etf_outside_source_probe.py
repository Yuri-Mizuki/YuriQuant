"""池外信息源廉价近似探针（2026-10-30）
=====================================

回答一个问题：**「池外新信息源」值不值得进组合实测（L3）**。只做预筛，不出采纳证据。

两个源（``--source``）：

- ``cls``  ：财联社电报主题注意力（本地 ``alt_cls/``，2014-12 起逐日，4320 个日文件）
- ``rates``：美债 10Y 实际利率（FRED ``DFII10``）+ 美元指数（``DTWIDX``/``DTWEXBGS``）
  → 逐 ETF 的「宏观顺逆风」评分（滚动 β × 近 20 日变动）

三层判据（**事前注册，跑之前写死**）

- **L1 增量性**：对生产信号 ``mom12_1_crowdsoft_pre10`` 做**截面正交**后的 Rank IC，
  Newey-West t。原始 IC 不算数——新闻热度≈近期涨幅、宏观顺风≈近期动量，两者都可能
  被共线伪装成增量。
- **L2 决策相关性**：本策略单只权重中位 **80.7%** ⇒ 全池 IC 提升 0.01 对收益几乎无意义，
  真正决定结果的是「第一名是谁」。故只统计**该源把 Top1 换掉的那些周**，换人之后的实现
  超额（新 Top1 − 原 Top1）。
- **安慰剂**：逐周随机置换评分 200 次，取同一指标的 95 分位做零假设分布。**没过安慰剂的
  一律按「任何随机评分也能做到」处理。**

🚨 必读边界（写进产物，不许只放在代码里）

1. **基线不是自造**：信号面板由 ``build_extra_panels`` 生成，与生产同源（含 crowdsoft/pre10）；
   但**排序只用纯信号 argmax**，未复现 dedup(0.9) 与 HRP 权重层 ⇒ 这是「选标层」探针，
   不是组合层实测。
2. **收益口径 = close→close 未来 5 日**，仅用于探针排序诊断；组合实测（L3）必须换
   T+1 open（生产口径）。两者不可混引。
3. **样本 = 2020-01 起**（ETF 行情本地起 2018-01、生产 warmup 后 2020 才干净）；
   ``rates`` 的长样本能力（2013+）本探针**没用上**——只在被评价的窗口内取值。
4. ``cls`` 的 PIT 截断 = 信号日 T 当日 24:00（下周 T+1 开盘执行，故不构成未来信息）；
   主题词典是**人工前置**的，未做词频学习 ⇒ 词典本身也算一次试验，别当无偏。
5. 通过 L1+L2+安慰剂 ⇔「值得做 L3 组合实测」，**不等于**「有增量」。L3 未跑之前不得引用。

用法::

    python -m scripts.portfolio.etf_outside_source_probe --source cls
    python -m scripts.portfolio.etf_outside_source_probe --source rates
    python -m scripts.portfolio.etf_outside_source_probe --source cls,rates --placebo 200
"""
from __future__ import annotations

import argparse
import glob
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_rotation_panels import build_extra_panels
from scripts.portfolio.etf_rotation_probe import build_panels, load_bundle
from stats.ic import calc_ic_series
from stats.significance import mean_inference

OUT = Path("reports/etf_contest5")
CACHE = OUT / "_cache"
TXT = OUT / "outside_source_probe_2026-10-30.txt"
SIGNAL = "mom12_1_crowdsoft_pre10"
EVAL_BEGIN = 20200101
FWD_DAYS = 5
CLS_BEGIN = 20180601          # 暖机：2020 起点前留 60 日注意力基线
PLACEBO_N = 200

#: 主题词典：ETF 代码 → (主题名, 关键词正则)。**人工前置**，改动算新试验。
#: 只收录「有明确主题词」的腿；宽基（300/50/500/1000/深100）刻意不收——
#: 「A股/大盘」这类词几乎每条电报都出现，属于常数而非信号。
THEMES: dict[str, tuple[str, str]] = {
    # ── 池内 A 股腿 ──
    "512480.SH": ("半导体", r"半导体|芯片|晶圆|集成电路|光刻|封测|存储芯片|制程"),
    "512760.SH": ("半导体", r"半导体|芯片|晶圆|集成电路|光刻|封测|存储芯片|制程"),
    "159516.SZ": ("半导体设备", r"半导体设备|光刻机|刻蚀|薄膜沉积|量测设备"),
    "588200.SH": ("科创芯片", r"科创芯片|芯片|半导体"),
    "515880.SH": ("通信", r"光模块|CPO|5G|通信设备|运营商|交换机|基站|光通信"),
    "512660.SH": ("军工", r"军工|国防|航天|航空发动机|导弹|战机|军费|军贸"),
    "512880.SH": ("证券", r"券商|证券公司|两融|IPO|注册制|投行"),
    "512070.SH": ("证券保险", r"券商|保险|险企|保费|两融"),
    "512800.SH": ("银行", r"银行|息差|存款|信贷|降准|LPR|净息差"),
    "512010.SH": ("医药", r"医药|创新药|集采|医保|药企|临床|CXO"),
    "512170.SH": ("医疗", r"医疗器械|医疗服务|医院|集采|医保"),
    "512290.SH": ("生物医药", r"生物医药|疫苗|生物制品|抗体"),
    "159992.SZ": ("创新药", r"创新药|临床|获批|License|BD交易"),
    "159928.SZ": ("消费", r"消费|社零|白酒|食品饮料|免税|零售"),
    "515790.SH": ("光伏", r"光伏|硅料|硅片|组件|逆变器|装机|多晶硅"),
    "515030.SH": ("新能源车", r"新能源车|电动车|整车|充电桩|汽车销量"),
    "159755.SZ": ("电池", r"锂电|动力电池|碳酸锂|固态电池|电芯"),
    "516160.SH": ("新能源", r"新能源|风电|储能|绿电|装机"),
    "512400.SH": ("有色", r"铜价|铝价|稀土|锂矿|镍|有色|钨|钴"),
    "515220.SH": ("煤炭", r"煤炭|动力煤|焦煤|煤价|火电|秦皇岛"),
    "512200.SH": ("地产", r"房地产|楼市|地产|保交楼|房贷|土拍"),
    "512980.SH": ("传媒", r"传媒|影视|短剧|广告|出版|院线"),
    "159869.SZ": ("游戏", r"游戏|版号|手游|电竞"),
    "562500.SH": ("机器人", r"机器人|人形机器人|减速器|灵巧手|具身智能"),
    "588000.SH": ("科创", r"科创板|科创50"),
    "159915.SZ": ("创业板", r"创业板"),
    "159949.SZ": ("创业板", r"创业板"),
    "510880.SH": ("红利", r"红利|高股息|分红|股息率"),
    # ── 池外定价腿（锚在外部市场，正是本探针的靶心）──
    "518880.SH": ("黄金", r"黄金|金价|现货金|伦敦金|避险|金饰"),
    "159980.SZ": ("有色期货", r"铜价|铝价|期货|有色金属"),
    "513100.SH": ("纳指", r"纳斯达克|纳指|美股|科技股|英伟达|苹果|特斯拉"),
    "513500.SH": ("标普", r"标普|美股|道指|华尔街"),
    "513050.SH": ("中概", r"中概|中概股|美股中概"),
    "159920.SZ": ("港股", r"港股|恒生指数|南向资金|恒指"),
    "513180.SH": ("恒生科技", r"恒生科技|港股科技|南向资金"),
    "513330.SH": ("恒生互联网", r"港股|互联网|恒生科技"),
    "513520.SH": ("日经", r"日经|日本股市|日央行|日元"),
    "511010.SH": ("国债", r"国债|债市|收益率|逆回购|MLF|央行"),
    "511260.SH": ("国债", r"国债|债市|收益率|逆回购|MLF|央行"),
}

FRED = {"DFII10": "real10y", "DTWEXBGS": "dxy"}


# ─────────────────────────── 面板与日历 ───────────────────────────
def load_frame() -> dict:
    """加载价格面板 + 生产信号 + 流动性掩码（与生产同源）。"""
    cache = DataCache(create_datasource())
    uni, close, amount, _bench = load_bundle(cache)
    panels = build_panels(close, amount)
    panels = build_extra_panels(close, panels)
    return {
        "cache": cache,
        "close": close,
        "signal": panels[SIGNAL],
        "mask": panels["mask"],
    }


def weekly_dates(idx: pd.DatetimeIndex, begin: int = EVAL_BEGIN) -> pd.DatetimeIndex:
    """每周首个交易日（与生产 ``freq="W"`` 同口径）。"""
    s = pd.Series(idx, index=idx)
    iso = idx.isocalendar()
    first = s.groupby([iso["year"].values, iso["week"].values]).first()
    d = pd.DatetimeIndex(first.values)
    return d[d >= pd.Timestamp(str(begin))]


def fwd_returns(close: pd.DataFrame) -> pd.DataFrame:
    """探针用未来 5 日收益（close→close；组合实测须换 T+1 open）。"""
    return close.shift(-FWD_DAYS) / close - 1.0


# ─────────────────────────── 源 1：财联社 ───────────────────────────
def load_cls_daily(begin: int = CLS_BEGIN) -> pd.DataFrame:
    """逐日主题命中数（带 CSV 缓存）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / "cls_theme_daily.csv"
    if cf.exists():
        df = pd.read_csv(cf, index_col=0, parse_dates=True)
        print(f"  [cache] {cf.name} {df.shape}")
        return df

    files = sorted(glob.glob(str(Path("E:/data/parquet/alt_cls") / "cls_*.parquet")))
    files = [f for f in files if re.search(r"(\d{8})", Path(f).stem).group(1) >= str(begin)]
    # 同主题多 ETF 合并关键词，避免重复扫描
    sig: dict[str, str] = {}
    for _code, (theme, kw) in THEMES.items():
        sig[theme] = kw if theme not in sig else f"{sig[theme]}|{kw}"
    pats = {t: re.compile(k) for t, k in sig.items()}

    rows = []
    for i, f in enumerate(files, 1):
        day = re.search(r"(\d{8})", Path(f).stem).group(1)
        try:
            d = pd.read_parquet(f, columns=["title", "subjects", "content"])
        except Exception as exc:                     # 单日坏文件不整批中断
            print(f"  ⚠️ {Path(f).name} 读取失败：{exc}")
            continue
        txt = (d["title"].fillna("") + " " + d["subjects"].fillna("")
               + " " + d["content"].fillna(""))
        rows.append({"date": pd.Timestamp(day),
                     **{t: int(txt.str.contains(p).sum()) for t, p in pats.items()}})
        if i % 500 == 0:
            print(f"  … {i}/{len(files)}")
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out.to_csv(cf, encoding="utf-8-sig")
    print(f"  [build] {cf.name} {out.shape}")
    return out


def cls_scores(frame: dict, daily: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """主题注意力「惊喜度」→ ETF 评分面板。

    score = log1p(近 5 日命中) / log1p(过去 60 日日均命中) - 1，再做截面 z。
    取比值而非差分：不同主题的基频差一个量级，比值可比。
    """
    cum5 = daily.rolling(5, min_periods=3).sum()
    base = daily.rolling(60, min_periods=30).mean()
    ratio = np.log1p(cum5) / np.log1p(base * 5).replace(0.0, np.nan) - 1.0
    ratio = ratio.loc[: dates.max()]

    close = frame["close"]
    cols = close.columns
    score = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for code, (theme, _kw) in THEMES.items():
        if code in cols and theme in ratio.columns:
            score[code] = ratio[theme].reindex(dates, method="ffill")
    return score


# ─────────────────────────── 源 2：实际利率 / 美元 ───────────────────────────
def fetch_fred(series: str) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"fred_{series}.csv"
    if not cf.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        with urllib.request.urlopen(url, timeout=60) as r:   # noqa: S310 - 固定 FRED 域名
            raw = r.read()
        cf.write_bytes(raw)
        print(f"  [fetch] {series} {len(raw)} bytes")
    df = pd.read_csv(cf)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df.set_index("date")["value"], errors="coerce").dropna()
    return s.sort_index()


def _roll_beta(ret: pd.DataFrame, x: pd.Series, win: int = 120, minp: int = 60) -> pd.DataFrame:
    """逐列滚动 β = cov(r, x) / var(x)，并 shift(1) 保证只用 t−1 及以前的信息。"""
    cov = ret.rolling(win, min_periods=minp).cov(x)
    var = x.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0).shift(1)


def rate_scores(frame: dict, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """宏观顺逆风评分：β_real·Δreal20 + β_dxy·Δdxy20（越高=越顺风）。"""
    idx = frame["close"].index
    real = fetch_fred("DFII10").reindex(idx, method="ffill").shift(1)
    dxy = fetch_fred("DTWEXBGS").reindex(idx, method="ffill").shift(1)
    d_real = real.diff()
    d_dxy = dxy.pct_change(fill_method=None)
    ret = frame["close"].pct_change(fill_method=None)

    beta_r = _roll_beta(ret, d_real)
    beta_d = _roll_beta(ret, d_dxy)
    d_real20 = real - real.shift(20)
    d_dxy20 = dxy / dxy.shift(20) - 1.0
    score = beta_r.mul(d_real20, axis=0) + beta_d.mul(d_dxy20, axis=0)
    meta = {
        "real10y_last": float(real.dropna().iloc[-1]),
        "dxy_last": float(dxy.dropna().iloc[-1]),
        "real_cov": f"{real.dropna().index.min().date()}~{real.dropna().index.max().date()}",
        "dxy_cov": f"{dxy.dropna().index.min().date()}~{dxy.dropna().index.max().date()}",
    }
    return score.reindex(dates), meta


#: 分腿宇宙（**探索性子组，非预注册**）：池外定价腿 = 跨境 7 + 商品 2 + 债券 2。
CROSS = ["513100.SH", "513500.SH", "513050.SH", "159920.SZ",
         "513180.SH", "513330.SH", "513520.SH"]
COMM = ["518880.SH", "159980.SZ"]
BOND = ["511010.SH", "511260.SH"]
OUTSIDE = tuple(CROSS + COMM + BOND)


def zrow(df: pd.DataFrame) -> pd.DataFrame:
    """逐行（截面）z 标准化。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0.0, np.nan), axis=0)


def rate_variants(frame: dict, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """利率类**构造阶梯**（跑之前未定，故全部列出、门槛提到 99 分位补偿多重检验）。

    V1 = β_real(d,120)·Δreal20 + β_dxy(d,120)·Δdxy20  ← **预注册构造**
    V2 = 两通道先截面 z 再相加（修 V1 的量纲失衡：β_dxy≈−2 vs β_real≈−0.1，
         实测 V1 的排名几乎全由美元通道决定，"实利率"名义上在、实际被淹没）
    V3 = 实利率通道改用周频 β(52w)、美元仍用日频 β(120d)
    V4 = 只实利率通道（周 β）——分离检验
    V5 = 只美元通道（日 β）——分离检验
    """
    idx = frame["close"].index
    close = frame["close"]
    real = fetch_fred("DFII10").reindex(idx, method="ffill").shift(1)
    dxy = fetch_fred("DTWEXBGS").reindex(idx, method="ffill").shift(1)
    d_real = real.diff()
    d_dxy = dxy.pct_change(fill_method=None)
    ret = close.pct_change(fill_method=None)
    d_real20 = real - real.shift(20)
    d_dxy20 = dxy / dxy.shift(20) - 1.0

    b_r = _roll_beta(ret, d_real)
    b_d = _roll_beta(ret, d_dxy)
    ch_r = b_r.mul(d_real20, axis=0)
    ch_d = b_d.mul(d_dxy20, axis=0)

    # 周频 β(52w)：日频 β 对日频宏观序列噪声太大（实测全池 β_real 挤在 −0.07~−0.12）
    wk_ret = close.resample("W-FRI").last().pct_change(fill_method=None)
    wk_dreal = real.resample("W-FRI").last().diff()
    cov = wk_ret.rolling(52, min_periods=30).cov(wk_dreal)
    var = wk_dreal.rolling(52, min_periods=30).var()
    b_w = cov.div(var, axis=0).shift(1).reindex(idx, method="ffill")
    ch_rw = b_w.mul(d_real20, axis=0)

    return {
        "V1 β_real120+β_dxy120（预注册）": (ch_r + ch_d).reindex(dates),
        "V2 两通道截面z后相加": (zrow(ch_r) + zrow(ch_d)).reindex(dates),
        "V3 实利率周β52w+美元日β": (zrow(ch_rw) + zrow(ch_d)).reindex(dates),
        "V4 只实利率（周β52w）": zrow(ch_rw).reindex(dates),
        "V5 只美元（日β120d）": zrow(ch_d).reindex(dates),
    }


# ─────────────────────────── 评测（L1 / L2 / 安慰剂）───────────────────────────
def _orth_rows(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """逐行截面正交（numpy 向量化）：x 的秩对 y 的秩回归后的残差，保留 NaN 模式。"""
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    out = np.full_like(xv, np.nan)

    def _rank(a: np.ndarray) -> np.ndarray:
        s = pd.Series(a)
        return s.rank().to_numpy()

    for i in range(xv.shape[0]):
        m = ~np.isnan(xv[i]) & ~np.isnan(yv[i])
        if int(m.sum()) < 5:
            continue
        ra, rb = _rank(xv[i, m]), _rank(yv[i, m])
        ac, bc = ra - ra.mean(), rb - rb.mean()
        den = float((bc ** 2).sum())
        if den <= 0:
            continue
        beta = float((ac * bc).sum()) / den
        out[i, m] = ac - beta * bc
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def evaluate(name: str, score: pd.DataFrame, frame: dict, dates: pd.DatetimeIndex,
             cols: list[str] | None = None) -> dict:
    """L1 + L2。``cols`` 给定时只在该子宇宙里评（分腿检查用，探索性）。"""
    close, signal, mask = frame["close"], frame["signal"], frame["mask"]
    fwd = fwd_returns(close)
    cols = close.columns if cols is None else [c for c in cols if c in close.columns]

    s = score.reindex(index=dates, columns=cols)
    sg = signal.reindex(index=dates, columns=cols)
    fw = fwd.reindex(index=dates, columns=cols)
    mk = mask.reindex(index=dates, columns=cols)

    orth = _orth_rows(s, sg)

    ic_raw = calc_ic_series(s, fw, tradable=mk)
    ic_orth = calc_ic_series(orth, fw, tradable=mk)
    r_raw, r_orth = mean_inference(ic_raw), mean_inference(ic_orth)

    # L2：Top1 换人周的实现超额
    swaps = []
    for t in dates:
        a = s.loc[t].where(mk.loc[t]).dropna()
        b = sg.loc[t].where(mk.loc[t]).dropna()
        f = fw.loc[t]
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        a, b = a[common], b[common]
        top_src, top_mom = a.idxmax(), b.idxmax()
        if top_src == top_mom:
            continue
        if pd.isna(f.get(top_src)) or pd.isna(f.get(top_mom)):
            continue
        swaps.append({"date": t, "src": top_src, "mom": top_mom,
                      "delta": float(f[top_src] - f[top_mom])})
    sw = pd.DataFrame(swaps)
    d_inf = mean_inference(sw["delta"]) if len(sw) else {"n": 0, "mean": np.nan,
                                                        "t_stat_nw": np.nan}

    return {
        "name": name, "n_week": int(ic_raw.notna().sum()),
        "ic_raw": r_raw["mean"], "ic_raw_t": r_raw["t_stat_nw"],
        "ic_orth": r_orth["mean"], "ic_orth_t": r_orth["t_stat_nw"],
        "swap_n": len(sw), "swap_rate": len(sw) / max(len(dates), 1),
        "swap_delta": d_inf.get("mean", np.nan),
        "swap_t": d_inf.get("t_stat_nw", np.nan),
        "swap_hit": float((sw["delta"] > 0).mean()) if len(sw) else np.nan,
        "swaps": sw,
    }


def placebo(score: pd.DataFrame, frame: dict, dates: pd.DatetimeIndex,
            n: int = PLACEBO_N, seed: int = 20261030,
            cols: list[str] | None = None) -> dict:
    """逐周随机置换评分，取 L1/L2 指标的零假设分布（置换保持边际分布、毁掉截面信息）。

    ``cols`` 给定时在子宇宙内置换 ⇒ 零分布的**截面宽度随之变化**（11 只比 50 只噪声大），
    所以子组的 IC 必须与本组的安慰剂分位比，不能跨组比。
    """
    rng = np.random.default_rng(seed)
    close, signal, mask = frame["close"], frame["signal"], frame["mask"]
    cols = list(close.columns) if cols is None else [c for c in cols if c in close.columns]
    fw = fwd_returns(close).reindex(index=dates, columns=cols)
    sg = signal.reindex(index=dates, columns=cols)
    mk = mask.reindex(index=dates, columns=cols)
    s0 = score.reindex(index=dates, columns=cols).to_numpy()
    mkv, sgv, fwdv = mk.to_numpy(), sg.to_numpy(), fw.to_numpy()

    ic_list, dl_list = [], []
    for _ in range(n):
        perm = s0.copy()
        for i in range(perm.shape[0]):
            m = ~np.isnan(perm[i]) & mkv[i]
            if m.sum() > 1:
                perm[i, m] = rng.permutation(perm[i, m])
        sp = pd.DataFrame(perm, index=dates, columns=cols)
        ic_list.append(float(calc_ic_series(_orth_rows(sp, sg), fw, tradable=mk).mean()))

        d = []
        for i in range(perm.shape[0]):
            m = mkv[i] & ~np.isnan(sgv[i])
            if m.sum() < 2:
                continue
            a = np.where(m, perm[i], np.nan)
            b = np.where(m, sgv[i], np.nan)
            if np.all(np.isnan(a)) or np.all(np.isnan(b)):
                continue          # 暖机期整行 NaN（如两通道 z 相加在 β 未就绪时）
            ta, tb = int(np.nanargmax(a)), int(np.nanargmax(b))
            if ta == tb or np.isnan(fwdv[i, ta]) or np.isnan(fwdv[i, tb]):
                continue
            d.append(float(fwdv[i, ta] - fwdv[i, tb]))
        dl_list.append(float(np.mean(d)) if d else np.nan)

    ic_arr, dl_arr = np.array(ic_list), np.array(dl_list)
    return {
        "ic_mean": float(np.nanmean(ic_arr)),
        "ic_q95": float(np.nanpercentile(ic_arr, 95)),
        "ic_q99": float(np.nanpercentile(ic_arr, 99)),
        "delta_mean": float(np.nanmean(dl_arr)),
        "delta_q95": float(np.nanpercentile(dl_arr, 95)),
        "delta_q99": float(np.nanpercentile(dl_arr, 99)),
    }


def legs(close: pd.DataFrame) -> list[tuple[str, list[str] | None]]:
    """按**定价锚在哪**分组。``None`` = 全池。

    ⚠️ 分腿是**探索性**的（跑之前没预注册），只用来看「源性方向是否被 A 股腿稀释」，
    不作为准入判据——准入只看全池 L1/L2。
    """
    cols = list(close.columns)
    out = [c for c in OUTSIDE if c in cols]
    return [("全池", None), ("池外11只", out), ("A股腿", [c for c in cols if c not in set(out)])]


def verdict(m: dict, pl: dict | None) -> tuple[bool, bool, list[str]]:
    """事前注册判读：L1 = 正交 IC>0 且 NW-t>2；L2 = 换人周超额>0 且 NW-t>2 且胜率>55%。"""
    l1 = (m["ic_orth"] > 0) and (m["ic_orth_t"] > 2.0)
    l2 = (m["swap_delta"] > 0) and (m["swap_t"] > 2.0) and (m["swap_hit"] > 0.55)
    notes = []
    if pl is not None:
        if m["ic_orth"] <= pl["ic_q95"]:
            notes.append(f"L1 未超安慰剂 95 分位（{m['ic_orth']:.4f} ≤ {pl['ic_q95']:.4f}）")
            l1 = False
        if not (m["swap_delta"] > pl["delta_q95"]):
            notes.append(f"L2 未超安慰剂 95 分位（{m['swap_delta']:.4f} ≤ {pl['delta_q95']:.4f}）")
            l2 = False
    return l1, l2, notes


def diagnose_beta(frame: dict, say) -> pd.DataFrame:
    """利率通道的**构造可辨识性**诊断。

    判否之前必须回答一个问题：是「利率无效」还是「我的构造没测到利率」？
    判据是 β 的**截面离散度**——若全池 β 挤在一个点上，该通道在横截面上没有排序能力，
    无论回测结果如何，它对「选哪只」这件事天然无能为力。
    """
    idx = frame["close"].index
    close = frame["close"]
    real = fetch_fred("DFII10").reindex(idx, method="ffill").shift(1)
    dxy = fetch_fred("DTWEXBGS").reindex(idx, method="ffill").shift(1)
    ret = close.pct_change(fill_method=None)
    rows = []

    for nm, s in (("DFII10 实际利率", real), ("DTWEXBGS 美元", dxy)):
        d = s.diff().dropna()
        rows.append({"item": f"{nm} 日变动", "stat": "零值占比",
                     "value": float((d.abs() < 1e-12).mean())})
        rows.append({"item": f"{nm} 日变动", "stat": "std", "value": float(d.std())})

    for nm, x in (("β_real(120d)", real.diff()), ("β_dxy(120d)", dxy.pct_change(fill_method=None))):
        b = _roll_beta(ret, x)
        last = b.iloc[-1].dropna()
        rows += [
            {"item": nm, "stat": "截面std", "value": float(last.std())},
            {"item": nm, "stat": "截面极差", "value": float(last.max() - last.min())},
            {"item": nm, "stat": "负值占比", "value": float((last < 0).mean())},
        ]
    df = pd.DataFrame(rows)

    br = _roll_beta(ret, real.diff())
    bd = _roll_beta(ret, dxy.pct_change(fill_method=None))
    ratio = float(br.iloc[-1].std() / bd.iloc[-1].std())

    say("")
    say("  ── 构造可辨识性诊断（判否前必读：是「无效」还是「没测到」）──")
    say(f"  {'项':<22}{'统计量':<10}{'值':>12}")
    for r in rows:
        say(f"  {r['item']:<22}{r['stat']:<10}{r['value']:>12.4f}")
    say(f"  β_real / β_dxy 截面离散度比 = **{ratio:.2f}**")
    if ratio < 0.2:
        say("  ⇒ 🚨 **实利率通道构造退化**：全池 β_real 几乎同号同值，该通道在横截面上")
        say("     **没有排序能力**（不是信号弱，是根本没有差异可排）。因此本探针检验的是")
        say("     「美元敏感度 × 美元变动」，**实利率这一路实际未被有效检验**；")
        say("     而要对它做有效检验，必须先有截面离散——这与构造方式无关，是这条源")
        say("     对该池子的结构性限制。它最多只能做择时，而择时已收口。")
    return df


def report_split(say, frame: dict, dates: pd.DatetimeIndex,
                 variants: dict[str, pd.DataFrame], pl: dict | None,
                 split_n: int = 100) -> list[dict]:
    """构造阶梯 × 分腿的体检表。

    目的只有一个：**在接受「判否」之前，先排除构造退化**——β 全是 0、被美元通道淹没、
    或方向正确但被 A 股腿稀释。分腿是探索性的，不进准入判据。

    🚨 每个分组**各自算安慰剂**（``split_n`` 次）：11 只截面的零分布更宽，
    拿全池 q95/q99 去比子组 IC 会**系统性地高估**子组。
    """
    rows = []
    groups = legs(frame["close"])
    for vname, sc in variants.items():
        say("")
        say(f"  ── {vname} ──")
        say(f"  {'分组':<10}{'只数':>5}{'原始IC':>10}{'NW-t':>7}{'正交IC':>10}{'NW-t':>7}"
            f"{'组内q95':>9}{'组内q99':>9}{'换人周':>7}{'换人超额':>10}{'NW-t':>7}{'胜率':>8}")
        for gname, gcols in groups:
            m = evaluate(vname, sc, frame, dates, cols=gcols)
            n = len(gcols) if gcols is not None else frame["close"].shape[1]
            gpl = None
            if split_n > 0:
                gpl = placebo(sc, frame, dates, n=split_n, cols=gcols)
            q95 = f"{gpl['ic_q95']:>+9.4f}" if gpl else f"{'—':>9}"
            q99 = f"{gpl['ic_q99']:>+9.4f}" if gpl else f"{'—':>9}"
            say(f"  {gname:<10}{n:>5}{m['ic_raw']:>+10.4f}{m['ic_raw_t']:>7.2f}"
                f"{m['ic_orth']:>+10.4f}{m['ic_orth_t']:>7.2f}{q95}{q99}"
                f"{m['swap_n']:>7}"
                f"{m['swap_delta']:>+10.4f}{m['swap_t']:>7.2f}{m['swap_hit']:>8.1%}")
            rows.append({"variant": vname, "group": gname, "n_cols": n,
                         "ic_raw": m["ic_raw"], "ic_raw_t": m["ic_raw_t"],
                         "ic_orth": m["ic_orth"], "ic_orth_t": m["ic_orth_t"],
                         "grp_ic_q95": gpl["ic_q95"] if gpl else np.nan,
                         "grp_ic_q99": gpl["ic_q99"] if gpl else np.nan,
                         "swap_n": m["swap_n"], "swap_delta": m["swap_delta"],
                         "swap_t": m["swap_t"], "swap_hit": m["swap_hit"]})
    if pl is not None:
        say("")
        say("  全池安慰剂（同参数置换，50 只）："
            f"正交 IC 95/99 分位 {pl['ic_q95']:+.4f}/{pl['ic_q99']:+.4f}；"
            f"换人超额 95/99 分位 {pl['delta_q95']:+.4f}/{pl['delta_q99']:+.4f}")
        say(f"  🚨 本表 {len(rows)} 个组合（{len(variants)} 构造 × {len(groups)} 分组）"
            f"**全是同一批样本上跑出来的** ⇒ 门槛读 **99 分位**，且只与**组内**分位比。")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="池外信息源廉价近似探针")
    ap.add_argument("--source", default="cls,rates", help="cls / rates / 逗号组合")
    ap.add_argument("--placebo", type=int, default=PLACEBO_N, help="置换次数，0=跳过")
    ap.add_argument("--split", action="store_true", default=True,
                    help="跑构造阶梯 × 分腿体检（默认开）")
    ap.add_argument("--no-split", dest="split", action="store_false")
    ap.add_argument("--split-placebo", type=int, default=100,
                    help="分腿表每组各自的置换次数（0=跳过，则无组内分位）")
    ap.add_argument("--out", default=str(TXT))
    args = ap.parse_args(argv)

    want = [s.strip() for s in args.source.split(",") if s.strip()]
    CACHE.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    say("=" * 100)
    say("ETF 池外信息源探针（L1 正交增量 IC / L2 Top1 换人 / 置换安慰剂）  2026-10-30")
    say("=" * 100)

    frame = load_frame()
    close = frame["close"]
    dates = weekly_dates(close.index)
    say(f"面板 {close.shape} {close.index.min().date()}~{close.index.max().date()}")
    say(f"周频调仓日 {len(dates)} 个（{dates.min().date()}~{dates.max().date()}）；"
        f"基线信号 `{SIGNAL}`；收益口径 close→close 未来 {FWD_DAYS} 日（**探针用，非对外口径**）")

    results: dict[str, dict] = {}
    for name in want:
        say("")
        if name == "cls":
            daily = load_cls_daily()
            score = cls_scores(frame, daily, dates)
            cover = int(score.notna().any(axis=0).sum())
            say(f"== 源 1：财联社主题注意力（{daily.shape[1]} 个主题 → "
                f"覆盖 {cover}/{close.shape[1]} 只）==")
            say(f"   电报覆盖 {daily.index.min().date()}~{daily.index.max().date()}，"
                f"主题命中均值 " + ", ".join(f"{c}:{daily[c].mean():.1f}"
                                            for c in list(daily.columns)[:5]) + " …")
        elif name == "rates":
            score, meta = rate_scores(frame, dates)
            say("== 源 2：美债实际利率 + 美元指数（逐 ETF 滚动 β × 20 日变动）==")
            say(f"   DFII10 {meta['real_cov']}（末值 {meta['real10y_last']:.2f}%）· "
                f"DTWEXBGS {meta['dxy_cov']}（末值 {meta['dxy_last']:.1f}）")
        else:
            say(f"⚠️ 未知源 {name}，跳过")
            continue

        m = evaluate(name, score, frame, dates)
        pl = placebo(score, frame, dates, n=args.placebo) if args.placebo > 0 else None
        l1, l2, notes = verdict(m, pl)
        m.update({"l1": l1, "l2": l2, "notes": notes})
        results[name] = m

        say("")
        say(f"  L1 原始 IC {m['ic_raw']:+.4f}（NW-t {m['ic_raw_t']:+.2f}）"
            f" → **正交后 IC {m['ic_orth']:+.4f}（NW-t {m['ic_orth_t']:+.2f}）**"
            f"  [{m['n_week']} 周]")
        say(f"  L2 Top1 换人 {m['swap_n']} 周（{m['swap_rate']:.1%}）："
            f"换人后实现超额 {m['swap_delta']:+.4f}/周（NW-t {m['swap_t']:+.2f}，"
            f"胜率 {m['swap_hit']:.1%}）")
        if pl:
            say(f"  安慰剂（{args.placebo} 次置换）：正交 IC 均值 {pl['ic_mean']:+.4f}、"
                f"95 分位 {pl['ic_q95']:+.4f}；换人超额均值 {pl['delta_mean']:+.4f}、"
                f"95 分位 {pl['delta_q95']:+.4f}")
        for n_ in notes:
            say(f"    ⚠️ {n_}")
        say(f"  ⇒ L1 {'✅' if l1 else '❌'} · L2 {'✅' if l2 else '❌'} · "
            f"{'进入 L3 组合实测' if (l1 and l2) else '**不进 L3**'}")

        if args.split:
            say("")
            say("=" * 100)
            say(f"构造体检 × 分腿（{name}）—— 目的：判否之前先排除「构造退化 / 被稀释」")
            say("=" * 100)
            if name == "rates":
                diag = diagnose_beta(frame, say)
                diag.to_csv(OUT / "diag_beta.csv", index=False, encoding="utf-8-sig")
                say("")
            vs = rate_variants(frame, dates) if name == "rates" else {name: score}
            rows = report_split(say, frame, dates, vs, pl, split_n=args.split_placebo)
            pd.DataFrame(rows).to_csv(OUT / f"split_{name}.csv", index=False,
                                      encoding="utf-8-sig")

    say("")
    say("=" * 100)
    say("判读汇总（事前注册：L1 = 正交 IC>0 且 NW-t>2；L2 = 换人周超额>0 且 NW-t>2 且胜率>55%）")
    say("=" * 100)
    say(f"{'源':<8}{'正交IC':>10}{'NW-t':>8}{'换人周':>7}{'换人超额/周':>12}"
        f"{'NW-t':>7}{'胜率':>8}{'L1':>4}{'L2':>4}")
    for name, m in results.items():
        say(f"{name:<8}{m['ic_orth']:>+10.4f}{m['ic_orth_t']:>8.2f}{m['swap_n']:>7}"
            f"{m['swap_delta']:>+12.4f}{m['swap_t']:>7.2f}{m['swap_hit']:>8.1%}"
            f"{'✅' if m['l1'] else '❌':>4}{'✅' if m['l2'] else '❌':>4}")

    say("")
    say("== 必读边界 ==")
    for txt in [
        "① 基线 = 生产信号面板 `mom12_1_crowdsoft_pre10`（与生产同源），"
        "   但**排序只用信号 argmax**，",
        "   未复现 dedup(0.9) 与 HRP 权重层 ⇒ 这是「选标层」预筛，不是组合层实测。",
        "② 收益口径 close→close 未来 5 日；对外数字必须是 T+1 open，两者不可混引。",
        "③ 通不过安慰剂的源一律按「随机评分也能做到」处理，不得进 L3。",
        "④ 主题词典是人工前置（算一次试验）；`rates` 只用 2018+ 日频面板，2013+ 长样本未用上。",
        "⑤ L1+L2 通过 ⇔「值得做 L3 组合实测」，**≠ 有增量**；L3 未跑前不得引用任何增益。",
        "⑥ 分腿（池外11只 / A股腿）是**探索性**的，不进准入判据；准入只看全池 L1/L2。",
        "⑦ 阶梯表 15 个组合 + cls 3 个 = 18 个，同一批样本 ⇒ 门槛读 **99 分位**。",
    ]:
        say(txt)

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, m in results.items():
        if m["swap_n"]:
            m["swaps"].to_csv(OUT / f"swaps_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\n报告 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
