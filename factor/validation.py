"""因子面板的**截面有效性**校验 —— 入库闸（2026-11-05 建立）。

背景
----
`panels_neu*` 里曾出现 46 个「sd=1 但无信息」的假面板，其中 35 个是
`winsorize_mad` 的 MAD=0 分支把整列压成常数造成的（已修），剩下 11 个
在**原始面板层**就没有截面差异。完整链路是：

    无截面差异（广播/常数/零膨胀）
      → winsorize 把整列压成同一常数
      → 中性化残差 ~1e-7（float32 尾数）
      → standardize_zscore 把这些尾数放大成 sd=1 的面板

这条链能通过所有常规检查（覆盖率正常、无重复行、无极端值、sd 正常），
唯一认得出的指纹是**当日截面唯一值数**。所以入库前必须问一句：

    这个面板在「同一天的不同股票之间」到底有没有差异？

没有差异的因子对截面选股模型贡献恒为 0，只会占掉候选池名额。

判据
----
``panel_variation_stats`` 逐日统计截面唯一值数与有效样本数，取中位数：

    uniq_ratio = median(当日唯一值数) / median(当日有效样本数)

* **截面恒常数**：把样本按时间等分 5 段，**每段**当日 `uniq <= 1` 的占比
  都 > 50% ⇒ 拒收（市场级广播/定义常数）。只要「最近一段正常」或「多数段正常」
  就放行——因子会演化，历史零膨胀但近期有差异的因子仍然可用
* **浮点噪声格点**：`uniq < 50` 且 `uniq_ratio < 1%` **且相对跨度 < 1e-3** ⇒ 拒收
* **合法离散因子**：`uniq <= 5` 但每天都真有多档取值 ⇒ 放行
  （`is_st` 修复后是 0/1 两值，按 1% 会被误杀，故为它开豁免）
* **近全空**：有效样本占比 <= 2% ⇒ 返回 `sparse`，**不在这里拒收**
  （稀疏事件因子是另一类问题，剔除它会改变现有 31 个因子的行为）

🚨 为什么唯一值少 **不能** 直接判假
--------------------------------
2026-11-05 首次全量回归把 40+ 个真因子判成了 noise：`alpha158_CNTD10`
（10 日下跌计数，唯一值 19）、`alpha158_IMAX20`（argmax 位置/20，20 档位）、
`alpha191_103`（5 的整数倍窗口值）全是**离散但合法**的因子。它们的共同点是
**相对跨度大**（ptp / |median| = 1~10），而浮点噪声只在 1e-7 量级内抖动
（相对跨度 < 1e-3）。所以判据必须带 `rel_span` 护栏，否则「唯一值少」这个
信号会把整个离散因子族一起误杀。

阈值经过现存 950 个面板回归：真面板 `uniq_ratio` 实测 0.5~0.99，
离散因子 0.0004~0.001，假面板 <= 1e-5，1% 这条线离两侧都远。

用法::

    from factor.validation import check_panel, judge
    ok, reason, stats = check_panel(df)          # df = date×code 面板
    ok, reason = judge(panel_variation_stats(df))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: 尾部取多少个交易日做判定（避开单日异常，与扫描脚本同口径）
TAIL_DAYS = 20
#: 浮点噪声判据：唯一值数 / 有效样本数 的下限
MIN_UNIQ_RATIO = 0.01
#: 唯一值数绝对下限（噪声放大只停在这个量级）
NOISE_MAX_UNIQ = 50
#: 合法离散因子豁免：唯一值数不超过此值且不是常数
DISCRETE_MAX_UNIQ = 5
#: 当日截面恒常数的天数占比上限
MAX_FLAT_FRAC = 0.5
#: 有效样本占比下限（低于此判 sparse，另行处理而非拒收）
MIN_VALID_FRAC = 0.02
#: 浮点噪声的相对跨度上限：ptp / |median| 小于此才算「在同一量级内抖动」
MIN_REL_SPAN = 1e-3
#: 判定「截面是否恒常数」所需的最小当日有效样本数
#: 样本只有 1~2 只时 uniq=1 是必然的，不能算「全市场同值」
MIN_ROW_SAMPLES = 50


def panel_variation_stats(panel: pd.DataFrame, tail: int | None = None,
                          segments: int = 5) -> dict:
    """截面变异度统计（按段判定）。

    Args:
        panel: date×code 面板（index = 交易日，columns = 股票代码）。
        tail: 只取末尾多少行；``None``（默认）= 用传入的全部行。
            🚨 调用方若已用 ``read_panel_sample`` 抽过样，**必须**保持 None：
            再截一次尾部会把抽样结果削成「只剩最后一个段」，等价于回到
            「只看末尾」的误判（2026-11-05 实测 pledge_chg_20d 因此被误杀）。
        segments: 把行等分成几段分别判（配合 ``read_panel_sample`` 的抽样段数）。

    Returns:
        dict，字段：
          - ``uniq``：可判日截面唯一值数的中位数
          - ``n_valid``：可判日的有效样本数中位数
          - ``uniq_ratio``：``uniq / n_valid``（n_valid=0 时 nan）
          - ``flat_frac``：可判日中 ``uniq <= 1`` 的占比（全局）
          - ``seg_flat``：逐段的 ``flat_frac``（nan = 该段无足够样本）
          - ``valid_frac``：整段的有效样本占比
          - ``rel_span``：``ptp / max(|median|, eps)``，区分「离散档位」与
            「浮点抖动」的关键护栏（见模块文档的误杀记录）
          - ``sd``：全局标准差（仅供参考，**不能**用于判定：zscore 后面板的
            sd 恒为 1，这正是假面板藏了半年的原因）
    """
    if panel is None or panel.size == 0:
        return dict(uniq=0, n_valid=0, uniq_ratio=float("nan"),
                    flat_frac=1.0, seg_flat=[], valid_frac=0.0,
                    rel_span=float("nan"), sd=float("nan"))
    t = panel if not tail else (panel.iloc[-tail:] if len(panel) > tail else panel)
    vals = t.to_numpy(dtype="float32")
    finite = np.isfinite(vals)
    uniqs = np.array([np.unique(row[f]).size for row, f in zip(vals, finite, strict=False)],
                     dtype="float64")
    nvalids = finite.sum(axis=1).astype("float64")
    # 🚨 flat_frac 只在「当天有足够截面样本」的日子里算：
    # 全 NaN 的日子谈不上「截面常数」；只有 1~2 个样本的日子 uniq 必然等于 1，
    # 两者算进分母都会误判（实测 pledge_chg_20d 前 4 段 uniq 175~316 正常，
    # 旧算法把稀疏日与全 NaN 日都计入，flat=75% 差点误杀）。
    judgeable = nvalids >= MIN_ROW_SAMPLES
    uniq = float(np.median(uniqs[judgeable])) if judgeable.any() else 0.0
    n_valid = float(np.median(nvalids[judgeable])) if judgeable.any() else 0.0
    flat_frac = float(np.mean(uniqs[judgeable] <= 1)) if judgeable.any() else 1.0

    seg_flat: list[float] = []
    for idx in np.array_split(np.arange(len(t)), max(1, segments)):
        if idx.size == 0:
            continue
        m = judgeable[idx]
        seg_flat.append(float(np.mean(uniqs[idx][m] <= 1)) if m.any() else float("nan"))

    allv = vals[finite]
    if allv.size:
        span = float(np.ptp(allv))
        med = float(np.median(np.abs(allv)))
        rel_span = span / max(med, 1e-12)
    else:
        rel_span = float("nan")
    return dict(
        uniq=int(uniq),
        n_valid=int(n_valid),
        uniq_ratio=float(uniq / n_valid) if n_valid > 0 else float("nan"),
        flat_frac=flat_frac,
        seg_flat=seg_flat,
        valid_frac=float(finite.mean()),
        rel_span=rel_span,
        sd=float(np.nanstd(vals)) if finite.any() else float("nan"),
    )


def judge(stats: dict, *,
          min_uniq_ratio: float = MIN_UNIQ_RATIO,
          max_flat_frac: float = MAX_FLAT_FRAC,
          min_valid_frac: float = MIN_VALID_FRAC,
          min_rel_span: float = MIN_REL_SPAN) -> tuple[bool, str]:
    """按 ``panel_variation_stats`` 的输出判定面板能否作为截面因子。

    Returns:
        ``(ok, reason)``。``reason`` 取值：``ok``（正常连续因子）/ ``discrete``
        （放行，唯一值少但相对跨度为真）/ ``sparse``（放行，覆盖不足）/
        ``flat``（拒收，截面恒常数）/ ``noise``（拒收，浮点抖动）。
    """
    # 先判「有没有截面差异」，再判「是不是太稀疏」——顺序不能反：
    # 覆盖 1.5% 但每天 90 只样本全同值（is_st 旧版）是「无差异」，
    # 不能因为覆盖低就归到 sparse 放过。
    # 段级判定（比全局 flat_frac 稳健）：
    #   近段正常  ⇒ 放行（选股用的就是近期 IC 窗）
    #   多数段正常 ⇒ 放行（因子近期退化但历史有效，如 pledge_chg_20d 最近一个月源数据全 0）
    #   全段恒常数 ⇒ 拒（alt_macro_* / alpha360 基准项）
    segs = [f for f in stats.get("seg_flat") or [] if np.isfinite(f)]
    if segs:
        last_ok = segs[-1] <= max_flat_frac
        n_ok = sum(f <= max_flat_frac for f in segs)
        if not (last_ok or n_ok * 2 > len(segs)):
            return False, "flat"
    elif stats["flat_frac"] > max_flat_frac and stats["valid_frac"] >= min_valid_frac:
        # 没有可判日（样本太少）时不要用全局 flat_frac 判死：
        # 全 NaN 面板的 flat_frac 恒为 1.0，但它是「近全空」而非「无差异」。
        return False, "flat"
    if stats["valid_frac"] < min_valid_frac:
        return True, "sparse"
    if stats["uniq"] <= DISCRETE_MAX_UNIQ:
        return True, "discrete"
    ratio = stats["uniq_ratio"]
    if not np.isfinite(ratio):
        return True, "sparse"
    if stats["uniq"] < NOISE_MAX_UNIQ and ratio < min_uniq_ratio:
        # 护栏：唯一值少不等于假。rank 归一化档位 / 整数计数 / argmax 位置都是
        # 真因子（实测 alpha158_CNTD10 uniq=19、alpha191_103 uniq=20~40）；
        # 只有相对跨度也在同一量级内（浮点尾数抖动）才判噪声。
        rs = stats.get("rel_span", float("nan"))
        if np.isfinite(rs) and rs < min_rel_span:
            return False, "noise"
        return True, "discrete"
    return True, "ok"


def check_panel(panel: pd.DataFrame, *, tail: int | None = None,
                segments: int = 5,
                **kwargs) -> tuple[bool, str, dict]:
    """``panel_variation_stats`` + ``judge`` 的组合便捷入口（默认用全部行）。"""
    st = panel_variation_stats(panel, tail=tail, segments=segments)
    ok, reason = judge(st, **kwargs)
    return ok, reason, st


def read_panel_sample(path: str | Path, *, windows: int = 5,
                      rows: int = TAIL_DAYS) -> pd.DataFrame:
    """从面板**全期均匀抽 ``windows`` 段**、每段 ``rows`` 个交易日，用于判定。

    🚨 为什么不能只取尾部（2026-11-05 实测）
    --------------------------------------
    `pledge_chg_20d` 末 20 日截面唯一值 = 1（覆盖 29%，最近一批质押变动恰好同值），
    但它在中性化面板里唯一值 1634、历史各段完全正常 —— 只看尾部会误杀一个真因子。
    低更新频率的因子（质押/股东/分红）本来就有「最近一段时间没新数据」的窗口。

    抽 5 段后，单段退化只让 ``flat_frac`` 升到 0.2（仍低于 0.5 的阈值），
    只有**多数窗口都退化**才判死，正好符合「这个面板整体没有截面信息」的语义。

    Args:
        path: parquet 路径。
        windows: 抽多少段。
        rows: 每段多少交易日。

    Returns:
        拼接后的面板（索引保持原交易日，段间不重叠）。
    """
    import pyarrow.parquet as pq

    df = pq.ParquetFile(Path(path)).read().to_pandas()
    n = len(df)
    if n <= windows * rows:
        return df
    starts = np.linspace(0, n - rows, windows).astype(int)
    return pd.concat([df.iloc[s:s + rows] for s in starts])


def read_panel_tail(path: str | Path, rows: int = TAIL_DAYS) -> pd.DataFrame:
    """只读 parquet 最后一个 row group 的末尾 ``rows`` 行（快速体检用）。

    入库闸请用 ``read_panel_sample``（全期抽样），本函数只适合「扫一眼现在
    长什么样」。两者混用会导致低更新频率因子被误判。

    🚨 parquet 里索引列排在**数据列之后**，所以这里用 ``to_pandas()`` 的默认
    列语义而非按位置取列——历史上 `column(0)` 取到的是第一只股票的值。
    """
    import pyarrow.parquet as pq

    p = Path(path)
    pf = pq.ParquetFile(p)
    tb = pf.read_row_group(pf.num_row_groups - 1)
    df = tb.to_pandas()
    return df.iloc[-rows:]


_REASON_TEXT = {
    "flat": "截面恒常数（当日全市场同值 ≥50% 天数）",
    "noise": "浮点噪声格点（唯一值数停在几十）",
}


def describe_reason(reason: str) -> str:
    """把判定码翻成人话，供日志/报告使用。"""
    return _REASON_TEXT.get(reason, reason)
