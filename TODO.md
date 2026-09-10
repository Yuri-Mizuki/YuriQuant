# YuriQuant 待办清单

> 2026-08-29 六轮工程整改（P0–P5）完成后的基线盘点。整改内容见提交
> `c6019f7` / `2c0a159` / `3b92fd3`（测试兼容修复 → stats 公共层 → scripts 收敛）。
> 现状：核心包达"清晰"档（零循环依赖、统计工具单一真源、521 测试全绿），
> scripts 层仍有存量样板债。按「对结论可信度的影响」排序。

---

## 一、研究验证欠账（最高优先级——影响结论可信度）

口径变更后，以下历史实验结论需重跑才能继续引用：

- [x] **重跑 `scripts/multiyear_oos.py`**（2026-09-01 完成，620s）：README 已标注
  "h>1 结论基于回测引擎 bug 下的伪结果"。修复后重跑结论：**h1×M 仍是唯一
  三年一致稳健的频率解**（gbdt +5.5%±1.7%、ranker +5.4%±4.6%，均 3/3 年正超额；
  h1×W/D 与 h5×M 全负，ridge 全负）。注意待查：脚本 `oos_ic` 为原始预测 vs
  收益的 IC（h1 全模型为负 −0.1~−0.36），而策略交易中性化后信号——两口径
  方向相反，需核对 IC 口径或将其并入"信号强度"研究问题。
- [x] **重跑 `scripts/freq_tune.py`**（2026-09-01 完成）：修复后引擎下重新生成。
  h=1 排名不变（M +6.1% > W −8.9% > D −40.6%），h5×M 从伪结果（Sharpe −3.1）
  修正为超额 −16.6%/Sharpe 0.95——h1×M 仍最优但差幅可信。
  修复前旧结果存档于 `reports/freq_tune/freq_tune_prefix_engine_fix_20260825.csv`。
- [ ] **重跑 e2e 族报告对齐年化口径**：`perf_stats` 年化 244→252 后，
  `reports/e2e_backtest/`、`reports/investment_report/` 中 e2e 链路历史数字
  与现行口径存在 ~3% 系统性偏移。
- [x] **攻"跑输全池基准"的研究问题——全A滚动训练实验**（2026-09-01 完成，
  `scripts/rolling_grid_alla.py` → `reports/alla_rolling/report.html`）：
  全A 5549 股 × 798 公因子 × 2018~2026 分年 walk-forward 网格
  （horizon×频率×模型×超参×中性化×集中度，120 组合）。**选股宽度是关键答案**：
  全A Top10%~20% 等权多头显著跑赢等权基准（gbdt h1×M raw Top10% 年化 14.0%，
  超额上证 +11.8%/年、超额全A等权 +5.2%/年；gbdt_deep h1×W raw Top10% 正超额
  8/9 年）——HS300 top-50 集中持仓跑输的旧问题在"更宽股票池 + 更分散持仓"下
  方向反转。h1 短视野 + 月/周频 + gbdt 系模型是一致稳健解；日频换手成本全灭；
  h>1 受长 horizon 特征去冗余后特征数不足拖累（详见报告披露⑥）。
- [x] **全A实验数据卫生修复**（2026-09-07 完成）：① 退市股回补——按 SDK 历史清单
  （沪深A 2015 至今含退市 ∪ 当前全A）补齐 238 只退市类代码的 K 线/股本/复权因子，
  全链路重建后幸存者偏差消除，头部配置超额回落 ~2pp（量级合理、排序不变）；
  ② 状态表全量重拉（修复 2019-2021 覆盖锯齿）；③ 基本面面板 32 只 ETF 列剔除 +
  IC 重算；④ 2026-09-02 半拉日全链路裁剪（水位回退自愈）。
- [x] **最优方案生产化每日推理**（2026-09-08 完成，`scripts/alla_daily_rank.py`）：
  全A滚动实验最优方案（gbdt h1 + 当年入选 50 因子 + 500 日窗 + raw Top10%）的
  每日盘后推理——增量数据更新 → 尾部重算因子（同一复权基准自洽，不改写冻结
  数据集）→ 重训预测最新截面 → 全A排名 + Top10% 可交易候选 → `reports/alla_daily/`；
  与实验 OOS 面板末日截面 Spearman 0.92 验证一致性；`--install-task` 注册每日
  计划任务（`YuriQuant AllaDailyRank`）。已知边界：训练段用发布时点已知标签
  （实时无未来可窥）；可交易性为信号日状态估计；跨年回退最近年份特征选择并告警。
- [ ] **全A实验的后续深挖**：① h5/h10 特征不足问题已由 DPP 去冗余解决
  （每期选满 50 特征），余下：验证 ortho 正交化口径（panels_neu 已建，predict 未跑）；
  ② 全A数据集纳入正式因子库监控（all_a_2018_2026 目前为轻量 registry）；
  ③ ~~2026-09-02 盘中数据整日重拉~~（2026-09-08 由 alla_daily_rank 全流程运行
  顺带完成，daily_all_a 缓存已到 20260907）；
  ④ 数据层 bug 已修（2026-09-08）：`DataCache.get_calendar` 在 end=None 时
  永不回源，日历被历史显式 end 封顶后"更新到最新"永远看不到新交易日
  （实测卡 20260902）——改为 end=None 语义=覆盖到今天，附回归测试。

- [x] **全A超额收益来源复核（α/β 分解 + Brinson 行业归因）**（2026-09-08 完成，
  `scripts/alla_excess_attribution.py` → `reports/alla_attribution/`）：复跑主策略
  （gbdt h1×M×raw Top10%）取每日权重后归因。结论：① 对上证的超额 +10.2%/年中
  约 一半来自风格敞口——对上证 β=1.10、R²=0.61（小盘暴露），剔除后 α=+11.2%/年
  但 t=1.89 未过显著线；对全A等权（自然基准）β≈0.97、R²=0.93，**纯选股 α=+5.1%/年、
  t=2.14 显著**（p=0.033）；② Brinson（vs 全A等权，申万一级）：累计主动收益 +35.6%
  = 选择 +64.1% + 配置 −15.5% + 交互 −12.3%——**超额几乎全部来自行业内选股，
  行业配置是拖累**。
- [x] **超额与夏普的统计显著性检验进回测指标**（2026-09-08 完成）：`calc_all_metrics`
  新增 `sharpe_t_stat`（日收益均值 Newey-West t）/ `years_to_prove` = (1.96/|SR|)²
  （paperswithbacktest 复现口径：证明策略所需年限）/ `excess_t_stat`（超额均值 t 检验），
  HTML 明细/对比表与 `metrics_overall.csv`、`rolling_grid` 报告同步。主策略实测：
  8.35 年样本 Sharpe 0.515 → t=1.69（未过 1.96，证明需 14.5 年）、超额 t=1.947
  （恰好压线）—— headline 年化的统计证据比直觉弱，引用时须带 t 值。
- [x] **论文复现因子族入库（awesome-systematic-trading 21 式）**（2026-09-08 完成，
  `factor/paper_factors.py` + `scripts/build_paper_factors.py`）：61 个已复现策略中
  筛出可在 A 股数据面实现的 21 个，翻译为截面因子入 all_a_2018_2026（879→900），
  全部走 PIT + IC/NW-t + canonical 回测标准流程。A 股实测速览：短期反转
  （IC=0.038, t=11.3）、公告期反转（t=3.2）、低波（t=8.2）、价值 BP（t=8.5）、
  研发强度（t=5.6）、盈利质量/FSCORE（t≈4.4）显著为正；**月频动量族为负**
  （一致动量 t=−3.0、动量×波动 t=−8.0，A 股动量反转经典结论复现）；BAB/应计
  （资产负债表法）/资产增长不显著。待办：`dup_checked=False`（冗余预检未自动跑，
  pap_value_bp 与库内 bp 同构，待库报告确认相关性）。

## 二、研究功能缺口（新能力）

- [ ] **分钟频因子挖掘 pipeline**（数据层已建，2026-09-09）：
  - [x] 数据层：`data/intraday.py` MinutePanelStore——parquet 分钟长表 → 按年分区
    MemMap 稠密面板 `[日,bar,码]`（全市场 GB 级不进内存，参照 Alpha掘金 24 的
    MemMap+年份切片方案）；`DataCache.read_minute_kline` 只读入口；
    `scripts/build_minute_panel.py`（--features/--demo-ic）。hs300 2022–2026
    已物化（1085 日 × 48 bar × 339 码，覆盖 88%）。
  - [x] 特征层：`factor/intraday_features.py`——tsfresh 风格向量化"分钟→日频"
    统计特征 42 个（动量/分布/波动/形态自相关/量价/蜡烛，Alpha掘金 22 的降维
    路线）。初测 IC 方向符合预期：尾盘动量反转（mom_last_bar IC=-0.058,
    t=-18.6）、低波效应（vol_rv IC=-0.035）、深跌反转（ret_min IC=+0.030）。
  - [ ] 挖掘层：把 42 特征接入 GP/GFlowNet 日频挖掘框架（Alpha掘金 22 的
    "降维后复用日频框架"）；涨跌停/ST 可执行性处理对齐
    build_intraday_factors 口径。
  - [x] 第一层单因子体检（2026-09-09 完成）：`scripts/build_intraday_stat_factors.py`
    ——除权除息日掩码（1754 格）→ zscore 入库 hs300_2022_2025（42 个 im_*，
    821→863）；批量冗余体检（vs 库内 821 因子逐日秩相关 + top1 正交残差 IC，
    报告 `reports/intraday_stat_checkup.csv`；`daily_rank_corr_mean` 有 corrwith
    对拍测试）。结论：37/42 NW-t 显著；**最大增量=尾盘分钟动量反转**
    （im_mom_last_bar IC=-0.061 t_nw=-23.4，全库最高相关仅 0.19；
    im_mom_close30_ret t_nw=-14.2 相关 0.30）——现有日频库完全缺失该信息。
    冗余 3 个完全同构（vol_range≈KLEN、pos_in_range≈KSFT2、ret_mean≈KMID，
    corr>0.95）、波动率族与 KLEN 0.78-0.80 边缘；高相关但残差显著含增量：
    pv_vwap_dev(resid t=-7.4)、vol_rsj(3.6)、pv_ret_vol_corr(3.5)、
    am/pm 时段差(6.1)。独立且显著 17 个——第二层挖掘的原料优先选这批。
  - [ ] 扩展：all_a 池物化（~5500 码 × 48 bar 需验证 MemMap 分块性能）；
    1 分钟档位（存储×5，需评估磁盘）。
- [ ] **生产级执行**：实盘下单对接、实时行情驱动（当前为日频 + 盘后信号；
  "信号→次日执行"的滑点假设未经真实成交验证）。
- [ ] **文本挖掘合规化**：当前依赖同花顺/巨潮爬虫（页面变更即断），
  生产化需对接 iFinD/Wind/Choice 等商业源，或明确接受"研究性补充"定位。
- [ ] **在线 dashboard**：报告均为静态 HTML，无增量刷新的监控页面。

## 三、工程层剩余债务

### 3.1 结构性（应先做）

- [x] **实验脚本私有函数倒挂残余**：`cpcv_eval.py` / `cpcv_h1_eval.py` /
  `ml_algorithm_compare.py` 仍 import `ml_synthesis_experiment` 的
  `_eval_row` / `_px_panels` / `_fit_predict_valid` / `_monthly_ic` 等私有函数
  （P3 仅解掉了 `_classic_features`）。公共函数应迁至 `e2e_common` 或独立模块。
- [x] **最小 CI**：无 `.github/workflows`。加最简 GitHub Actions
  （pytest + ruff check + tests/test_layering.py 门禁），
  把口径守卫和分层守卫变成强制约束（测试漂移到无法收集才被发现，
  根因就是无 CI）。

### 3.2 机械性（可批量清理）

- [x] **cli_common 推广**（骨架已建、采用率 <15%）：
  - 53 处 `logging.basicConfig` 手写样板 → `setup_logging()`（仅 scripts 层）
  - 22 处手写 `--real`/`--mock` add_argument → `add_real_mock_args()`
- [x] **HTML 报告模板收编**：8 套各自内嵌的模板（`monitoring/runner.py` 的
  317 行 `generate_html_report` 与 `research/html_report.py` 同名异构、
  `investment_report` / `factor_explorer_report` / `factor_library_full_report` /
  `risk_decomposition_report` / `run_etf_rotation` / `report_pipeline`）。
  以 `research/html_report.py` 为基座统一，顺带拆掉超长函数。
- [x] **超长文件/函数拆分**：
  - `factor/genetic_mining.py` 1692 行（`run_gp_mining` 单函数 ~301 行）
  - `factor/alpha191.py` 1273 行（公式库，可辩护）
  - `scripts/mine_factors.py` 800 行、`scripts/factor_explorer_report.py` 的
    `build_factor_data` ~493 行
- [x] **核心包卫生**：25 处 `print(` → logging（factor 15、data 8）；
  10 处 `except: pass` 静默吞异常逐个审查。
- [x] **依赖锁文件**：当前仅 `>=` 下界，加 lock（pip-tools / uv）保证可复现。

## 四、低优先级（知情即可）

- [ ] `factor/technical.py` 与 `factor/technical_indicators.py` 双口径指标
  （有意保留；SAR 有两份实现，`calc_sar` / SDK 版）
- [ ] GFlowNet 自写 PPO（`factor/gflownet/ppo.py`）与 AlphaPool MaskablePPO
  （`factor/rl/`）双轨；`factor/rl` 目前仅测试消费、无 scripts 入口
- [ ] optimize 标注的 P3 待建：完整多期最优执行、风险预算非等权
- [ ] 一次性实验脚本（`gtja_*` / `diagnose_*` / `compare_*` 等）保留原样，
  不迁移 cli_common（改动无收益只有风险）

---

## 建议推进顺序

1. 重跑 multiyear + freq_tune（补核心结论证据链，顺带验证整改后口径）
2. 最小 CI（防回归，约一天）
3. 分钟频挖掘 pipeline（现成数据的最大增量）
4. cli_common 批量推广 + 报告模板收编（机械清理，可穿插）
5. 攻"跑输基准"研究问题本身
