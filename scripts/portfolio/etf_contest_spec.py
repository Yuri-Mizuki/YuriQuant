"""正式赛定版参数（单一真源，2026-10-27 锁定）
=================================================

回测探针（``etf_contest_probe``）与实盘信号板（``etf_champ_signals``）必须共用
**同一个** 字典，避免"回测用一个参数、下单用另一个"的漂移。改参数只改这里，
两边自动同步；``tests/test_etf_contest_spec.py`` 会断言两边确实一致。

实验分组与证据索引（改哪一组就跑哪条命令）：

| 代码里的分组 | 用途 | 证据目录 |
|---|---|---|
| ``SPECS`` | 早期 51 池信号筛选（20/60/12-1、双周、拥挤硬过滤）
  （2026-10-27 方案作废，产物目录已删） | 复跑 ``--specs base`` |
| ``SPECS[k5_*]`` | Top5 档：去重、fill、等权/逆波/ERC/HRP、上限
  （同上；最新 Top5 口径见 ``reports/etf_contest5_opt/``） | 复跑 ``--specs k5`` |
| ``CONC_SPECS`` | 集中档 Top1/2/3/5 + 拥挤软惩罚 + 6-1 + 周五 + 扩池
  （证据见 ``reports/etf_contest5_opt/``） | 同上 |
| ``LT_SPECS`` | 长线月频 champ 对照 | ``reports/etf_contest5/`` §九 |
| ``FINAL`` | **定版**：Top3 + 拥挤软惩罚 + 120 日去重 + HRP 无上限
  （证据见 ``reports/etf_contest5/`` §十二） | 同上 |

定版依据（真实白名单 80 个滚动 2 月窗口）：均值 +4.44% / 中位 +3.99% /
p10 −8.16% / 最差 −23.82% / 胜率 66.3% / >10% 27.5% / <−10% 3.75%；
同族 Top5 HRP 为 +3.85% / +2.41% / −13.8%。
"""
from __future__ import annotations

# 比赛指定池不含、必须屏蔽的 5 只（否则选标会落到不可交易品种）。
EXCLUDED_5 = ["512690.SH", "511090.SH", "511380.SH", "159985.SZ", "159981.SZ"]

# 定版参数（不含 name；具体 profile 名由调用方补）。
FINAL: dict = {
    "signal": "mom12_1_crowdsoft",
    "freq": "W",
    "top_k": 3,
    "absolute": True,
    "weighting": "hrp",
    "commission_bp": 1.0,
    "slippage_bp": 5.0,
    "dedup_corr": 0.90,
    "dedup_fill": True,
}

# 执行日历：周频 = 每周首个交易日。收盘出信号；实盘次日开盘下单，
# 回测按调仓日收盘价成交（略乐观，未计隔夜跳空）。
EXEC_NOTE = "每周首个交易日：收盘出信号，实盘次日开盘下单"
