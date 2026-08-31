"""
技术面迭代/累积类指标 → 日频因子构建与入库
============================================

从星耀数智 ad-technical-analysis 技能的 56 个指标中挑选「现有算子空间
无法表达」的 9 个迭代/累积/EMA 类指标，作为独立信号源补入因子库：

    macd_hist   MACD 柱 = 2*(DIF-DEA)，DIF=EMA12-EMA26, DEA=EMA(DIF,9)
    rsi_12      RSI(12) = SMA(MAX(C-LC,0),12,1)/SMA(ABS(C-LC),12,1)*100
    kdj_j       KDJ 的 J = 3K-2D（K/D 用通达信 SMA 递归）
    trix_12     TRIX = (MTR-REF(MTR,1))/REF(MTR,1)*100, MTR=EMA³(C,12)
    obv_dev     OBV 偏离 30 日均线 = OBV/MA(OBV,30)-1（累积量取乖离）
    wad_dev     WAD 偏离 30 日均线（WAD 为多空累积力度线）
    asi_26      ASI = 26 日 SI 滚动和（振动升降指标）
    cho         CHO = (MA(MID,10)-MA(MID,20))/100, MID=ΣV*(2C-H-L)/(H+L)
    sar_dev     (close-SAR)/close，SAR 抛物线转向（Wilder 迭代）

口径说明
--------
- 价格一律用「后复权价」：daily 缓存为原始价，adj = raw × backward_factor。
  避免技能脚本"前复权用最新因子归一化"导致的样本末端漂移（未来信息）。
- 除权除息日由后复权价格自然衔接，无需剔除样本。
- 因子在 t 日收盘后可得，预测 t+1 收益（与库内 IC 口径一致）。
- 每因子面板 zscore 截面标准化后入库。

用法
----
    python -m scripts.build_technical_factors --offline            # 读缓存（推荐）
    python -m scripts.build_technical_factors --mock               # mock 验证
    python -m scripts.build_technical_factors --offline --no-save  # 只算不入库
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache_helpers import load_backward_factor, load_daily  # noqa: E402
from factor.technical import calc_indicators  # noqa: E402
from factor.technical_indicators import TechnicalIndicators as _TI  # noqa: E402
from research.factor_library import FactorLibrary  # noqa: E402
from scripts.cli_common import (  # noqa: E402
    add_build_args, make_data_context, print_no_save, record_experiment_safe,
    register_panels, returns_from_daily,
    setup_logging,
)

log = setup_logging("build_technical_factors")

FACTOR_DEFS: dict[str, str] = {
    "macd_hist": "MACD柱 = 2*(DIF-DEA)，DIF=EMA12-EMA26, DEA=EMA(DIF,9)",
    "rsi_12": "RSI(12) = SMA(MAX(C-LC,0),12,1)/SMA(ABS(C-LC),12,1)*100",
    "kdj_j": "KDJ-J = 3K-2D（RSV→SMA(3,1) 递归）",
    "trix_12": "TRIX = 三重EMA(C,12)变化率",
    "obv_dev": "OBV/MA(OBV,30)-1（能量潮乖离）",
    "wad_dev": "WAD/MA(WAD,30)-1（威廉多空力度线乖离）",
    "asi_26": "ASI = 26日SI滚动和（振动升降指标）",
    "cho": "CHO = (MA(MID,10)-MA(MID,20))/100，MID=ΣV*(2C-H-L)/(H+L)",
    "sar_dev": "(close-SAR)/close，Wilder抛物线SAR",
}

# 新增技术面因子（2026-08-05）：复用星耀 ad-technical-analysis skill 的
# TechnicalIndicators 类（通达信口径，移植自 factor/technical_indicators.py）。
# 结构：{因子名: (TI方法, 输出字段, 输入参数名列表, 公式说明)}
EXTRA_TECH: dict[str, tuple] = {
    "wr_14":    ("WR",    "WR10",  ("close", "high", "low"),                  "威廉%R(10) 超买超卖"),
    "cci_14":   ("CCI",   "CCI",   ("close", "high", "low"),                  "顺势指标CCI(14)"),
    "roc_12":   ("ROC",   "MAROC", ("close",),                                "变动率ROC(12,6)平滑"),
    "mtm_12":   ("MTM",   "MAMTM", ("close",),                                "动量MTM(12,6)平滑"),
    "skdj_d":   ("SKDJ",  "D",     ("close", "high", "low"),                  "慢速随机SKDJ-D(9,3)"),
    "mfi_14":   ("MFI",   "MFI",   ("close", "high", "low", "volume"),        "资金流量MFI(14)"),
    "osc_20":   ("OSC",   "MAOSC", ("close",),                                "震荡指标OSC(20,6)平滑"),
    "accer_8":  ("ACCER", "ACCER", ("close",),                                "加速度ACCER(8)"),
    "dmi_adx":  ("DMI",   "ADX",   ("close", "high", "low"),                  "趋向指标ADX(14,6)"),
    "dma_dif":  ("DMA",   "DIF",   ("close",),                                "平行线差DMA-DIF(10,50)"),
    "arbr":     ("ARBR",  "AR",    ("close", "open_", "high", "low"),         "人气指标AR(26)"),
    "emv_14":   ("EMV",   "MAEMV", ("close", "high", "low", "volume"),        "简易波动EMV(14,9)平滑"),
    "dpo_20":   ("DPO",   "MADPO", ("close",),                                "区间震荡DPO(20,6)平滑"),
    "vhf_28":   ("VHF",   "VHF",   ("close",),                                "趋势效率VHF(28)"),
    "cr_26":    ("CR",    "CR",    ("close", "high", "low"),                  "能量指标CR(26)"),
    "psy_12":   ("PSY",   "PSY",   ("close",),                                "心理线PSY(12)"),
    "vr_26":    ("VR",    "VR",    ("close", "volume"),                       "成交量比率VR(26)"),
    "wvad":     ("WVAD",  "WVAD",  ("close", "open", "high", "low", "volume"),"威廉变异离散WVAD(24)"),
    "bbi":      ("BBI",   "BBI",   ("close",),                                "多空均线BBI(3,6,12,24)"),
    "atr_14":   ("ATR",   "ATR",   ("close", "high", "low"),                  "真实波幅ATR(14)"),
}

# ===========================================================================
# 面板级基础算子（作用于 date×code DataFrame，时间轴在 index）
# ===========================================================================

def _ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """EMA：Y = (X*2 + Y'*(N-1))/(N+1)，与 ts_ema 一致。"""
    return x.ewm(span=n, adjust=False, min_periods=n).mean()

def _sma_tdx(x: pd.DataFrame, n: int, m: int = 1) -> pd.DataFrame:
    """通达信 SMA(X,N,M) 递归：Y = (X*M + Y'*(N-M))/N。

    等价于 ewm(alpha=M/N, adjust=False)。KDJ/RSI 的平滑依赖此式。
    """
    alpha = m / n
    return x.ewm(alpha=alpha, adjust=False, min_periods=n).mean()

def _sma_ma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """简单移动平均（右闭窗口，前 n-1 行为 NaN）。"""
    return x.rolling(n, min_periods=n).mean()

def _sma_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).sum()

# ===========================================================================
# 主流程
# ===========================================================================

def load_data(cache, uni, index_code, begin, end):
    codes, cal, daily = load_daily(cache, uni, index_code, begin, end)
    bf = load_backward_factor(cache, codes)
    log.info("日线 %d 行 / 复权因子 %d 列", len(daily), bf.shape[1] if not bf.empty else 0)
    return codes, cal, daily, bf

def build_factor_panels(daily, bf) -> dict[str, pd.DataFrame]:
    """返回 {因子名: 原始面板(date×code)}，价格均为后复权。"""
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    codes = sorted(d["code"].unique())

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, h, l, c, v = _panel("open"), _panel("high"), _panel("low"), _panel("close"), _panel("volume")

    # 后复权：adj = raw × backward_factor（bf 可能缺部分股票/日期 → NaN 保持）
    if not bf.empty:
        dates = c.index
        for pnl, name in ((o, "open"), (h, "high"), (l, "low"), (c, "close")):
            f = bf.reindex(index=dates).reindex(columns=pnl.columns)
            pnl[:] = pnl.values * f.values
    else:
        log.warning("无复权因子，使用原始价（除权日指标会有跳变）")

    panels: dict[str, pd.DataFrame] = {name: pd.DataFrame(index=c.index, columns=c.columns)
                                       for name in FACTOR_DEFS}
    for code in codes:
        res = calc_indicators(c[code], h[code], l[code], o[code], v[code])
        for name, s in res.items():
            panels[name][code] = s
    return panels

def calc_extra_panels(daily, codes) -> dict[str, pd.DataFrame]:
    """用星耀 TechnicalIndicators 类批量计算 date×code 技术指标面板（EXTRA_TECH）。

    daily: (date, code) 长表（已后复权）。逐股票调用静态方法，
    输出 Series 按位置对齐后拼成宽表（不依赖 Series index）。
    """
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    dates = pd.Index(sorted(d["date"].unique()))
    panels: dict[str, pd.DataFrame] = {
        name: pd.DataFrame(index=dates, columns=list(codes), dtype=float)
        for name in EXTRA_TECH
    }
    for code in codes:
        sub = d[d["code"] == code].sort_values("date")
        if sub.empty:
            continue
        s_map = {
            "close": sub["close"], "open": sub["open"], "open_": sub["open"],
            "high": sub["high"], "low": sub["low"], "volume": sub["volume"],
        }
        for name, (method, field, args, _desc) in EXTRA_TECH.items():
            try:
                kw = {a: s_map[a].reset_index(drop=True) for a in args}
                out = getattr(_TI, method)(**kw)
                s = out[field]
                vals = s.values if hasattr(s, "values") else np.asarray(s)
                n = min(len(vals), len(dates))
                panels[name].iloc[:n, panels[name].columns.get_loc(code)] = vals[:n]
            except Exception:
                continue
    return panels

def main():
    parser = argparse.ArgumentParser(description="技术面迭代/累积类指标构建入库")
    add_build_args(parser)
    parser.add_argument("--only-extra", action="store_true",
                        help="只注册新增星耀指标（EXTRA_TECH 20 个），跳过已有 9 个"
                             "（已有因子内容正确时刷新无意义，且可避开个别文件被外部锁定的情况）")
    args = parser.parse_args()

    cache, uni, begin, end, dataset = make_data_context(args)

    codes, cal, daily, bf = load_data(cache, uni, args.index, begin, end)
    if daily.empty:
        log.error("数据为空")
        sys.exit(1)

    all_defs = {**FACTOR_DEFS, **{k: v[3] for k, v in EXTRA_TECH.items()}}
    log.info("计算技术面指标（%d 个）...", len(all_defs))
    panels = build_factor_panels(daily, bf)
    log.info("计算星耀 TechnicalIndicators 指标（%d 个）...", len(EXTRA_TECH))
    extra = calc_extra_panels(daily, codes)
    panels.update(extra)

    if args.no_save:
        print_no_save(all_defs, panels)
        return

    lib = FactorLibrary(dataset=dataset)
    returns_panel = returns_from_daily(daily)

    log.info("入库到数据集: %s", dataset)
    register_names = list(EXTRA_TECH) if args.only_extra else list(all_defs)
    log.info("本次注册 %d 个因子（%s）", len(register_names),
             "仅新增星耀指标" if args.only_extra else "全部技术面")

    failed: list[tuple[str, str]] = []
    def _on_fail(name, e):
        failed.append((name, str(e)[:120]))
    register_panels(
        lib, panels, all_defs, returns_panel,
        source=f"technical:build_technical_factors:{begin}-{end}",
        names=register_names, on_fail=_on_fail,
    )

    record_experiment_safe(
        kind="technical_factors",
        command=" ".join(sys.argv),
        params={"index": args.index, "begin": begin, "end": end, "dataset": dataset},
        fingerprint=cache.get_fingerprint(),
        result_path=str(lib.root),
        metrics={"n_factors": len(all_defs)},
        note="技术面指标入库（含星耀 TechnicalIndicators 扩展）",
    )

    if failed:
        log.warning("有 %d 个因子注册失败（环境文件锁，可稍后重跑补齐）: %s",
                    len(failed), [n for n, _ in failed])
    log.info("完成。数据集 %s 现有 %d 个因子", dataset, len(lib.list_all()))

if __name__ == "__main__":
    main()