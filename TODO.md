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
  - **2026-09-11 第三批补齐另两类遗漏**：`data/cache_helpers` 的
    `_pit_universe_codes`（被 **6** 个 scripts 引用）/ `_apply_membership_mask`
    （**4** 个）+ `factor/genetic_mining` 的 5 个适应度组件
    （`_ls_net_stats` / `_monthly_forward_returns` / `_mutual_info_series` /
    `_top_excess_series` / `_htai_preprocess`，共 4 个 scripts 引用）此前漏解，
    已全部公开化；新增 AST 守卫防回潮。见 3.1 末尾「口径统一第三批」。
- [x] **最小 CI**：无 `.github/workflows`。加最简 GitHub Actions
  （pytest + ruff check + tests/test_layering.py 门禁），
  把口径守卫和分层守卫变成强制约束（测试漂移到无法收集才被发现，
  根因就是无 CI）。
  - 2026-09-10 收敛范围：push/PR 只留 Ruff 真 bug 规则 + 三个分层守卫
    （约 1 分钟）；全量 pytest 移出 push 流程 → 仅手动触发的
    `full-tests.yml`。原因：全量要 7 分钟、依赖 `reports/` 等不入库产物
    （CI 上只能靠 skip 兜住），且含已知 flaky 的 risk_parity 数值测试
    （`test_solver.py`，SCS 近似解波动），每次 push 都报红纯噪声。
  - 2026-09-10 后续（`7ea73c1`）：去掉 torch 后快检查降到 60 秒；随后两个
    工作流**都改为仅 `workflow_dispatch`**（`Fast checks (manual)` /
    `Full tests (manual)`），push 不再触发任何 CI —— 从源头消除红叉、失败
    邮件与等待。代价：自动检查彻底消失，改为手动触发或本机全量 pytest。

- [x] **口径分裂第一批（2026-09-10 完成）**：删 `scripts/run_backtest.py`
  （跨层混用收益面板、已被口径守卫拦下跑不起来）；可交易性掩码收敛到
  `data/tradability.py` 单模块（删死代码 `build_executable_mask`）；
  交易成本真源上提到 `settings.yaml` 顶层 `costs` 段 + `Config.costs()`
  （原 `backtest` 段万1/5bp 与 `model_portfolio` 段万3/10bp 差 2~3 倍、
  不可比）。新增 `tests/test_tradability.py` / `tests/test_cost_config.py`
  锁死两种口径的唯一语义差别。全量 629 passed（基线 618）。详见
  `.workbuddy/memory/2026-09-10.md`。

- [x] **口径统一第二批（2026-09-10 架构审计立项；四项待拍板决策与"权重生产
  唯一入口"已于 2026-09-11 全部落定）**：

  - [x] **显著性判定收口（2026-09-10 完成，`24837bb`）**：新增
    `stats/significance.py` 作为"判定层"单一真源（与 `stats/robust_stats` 的
    "估计层"分工）——`t_pvalue`（t→双侧 p）/ `benjamini_hochberg`（BH-FDR）/
    `mean_inference`（一步式 OLS+NW 两套 t/p）。先前散落的实现全部改走它：
    `factor/mining.py` 的私有 `_benjamini_hochberg` 与 worker/串行两条路径
    各自内联的 p 值公式、`research/factor_analysis.py`、`research/attribution.py`、
    `model/evaluation.py`（2 处）的内联 p 值公式。
    **两处检验族本就不同**，故保留两种语义、但共用一个实现：批量挖掘族 =
    这一批候选 → BH-FDR(q=0.05)；单因子入库无族 → 存原始 NW 显著性，
    并**新增 `p_value_nw` 列**把原始 p 落盘 + 提供
    `FactorLibrary.significance_table(q)` /
    `load_significant_features(correction="fdr")` 在"整库"这个族上校正。
    默认仍走 raw，**不改动已落盘结论**。
    实测切换代价（hs300_2025 的 244 因子）：raw 显著 **84** → 整库 BH-FDR(q=0.05)
    **62**（22 个会翻）；hs300_2022_2025 的 862 个：397 → 339。
    **决策（2026-09-11）：默认保持 raw，FDR 降到报告层并排展示**。
    `research/report_pipeline.collect_factor_library` 与
    `scripts/factor_library_full_report.py` 已接入 `significance_table`，
    同时显示 raw / 整库 FDR 两个计数。理由：
    ① 全仓 `load_significant_features` 只有 3 个消费点（`e2e_common` /
    `diagnose_neutralized_compare` / 一个测试），**主实验（全A滚动主线
    `rolling_grid_alla` → `alla_daily_rank`）不读因子库**——它用预构建的
    alpha panels，换默认只会静默改 e2e 家族数字、破坏可比性；
    ② 因子库是**累积库**（GP / GFlowNet / exhaustive / 论文复现 / 手工因子
    多轮混入），不是同一次多重检验的"族"，整库 BH-FDR 的族语义本身不严谨；
    ③ raw 与 FDR 回答不同问题（单因子自身有无 alpha vs 这批候选里有多少是
    真的），并排展示即可，不该互相替代。
  - [x] **统计 / 预处理 / 绩效原语收口（2026-09-10，`859494e` + `1a83d77`）**：
    - **rank IC**：`factor/gflownet/reward.rank_ic_series` 改为
      `stats.ic.calc_ic_series` 的薄封装（此前自实现一份）。`calc_ic_series`
      新增 `returns_rank` 参数承载原性能捷径，并把"**捷径不总是等价**"的边界
      写进 docstring + 用测试钉住：因子**整行**缺失（窗口预热）不影响；
      **行内散点**缺失才分叉，30% 散点缺失时日均 IC 差约 1.1e-2
      （IC 量级 3e-2~5e-2，**同阶**）。
      **决策（2026-09-11）：维持现状**——训练走捷径、入库/评估走 canonical，
      并把该分层**显式声明**（`factor/gflownet/reward.py` 模块 docstring 新增
      「IC 口径分层」段，`parallel.py` 模块头与 `run_gflownet_phase1.py` 调用处
      各加一处指引）。理由：训练奖励只是 batch 内的**相对排序**信号（决定哪些
      公式进 hof），1e-2 偏差不改变公式间相对次序，而重跑 Phase 0/1 代价大且
      破坏既有结论可比性；写进因子库的 IC 本来就是 canonical
      （`FactorLibrary.register` → `calc_ic_series` 不传 `returns_rank`）。
      **两处数字不可直接对比**，已写进 docstring。
    - **市值中性化**：向量化版上移为 `factor/preprocessing.neutralize_single`
      （含截距），`reward.neutralize_market_cap` 改为薄封装，新旧 max|Δ| = 0。
      **决策（2026-09-11）：补 ones 列**。`neutralize` 在**无行业哑变量**时
      （未传行业，或当天样本不足以容纳行业哑变量而被丢弃）补一列 `_intercept`；
      含行业哑变量时**不补**（其列和已是全 1 向量、已 span 截距，重复加列会共线）。
      至此与含截距的 `neutralize_single` 口径一致，消除了"同名市值中性化存在
      两个口径"的分裂。
      **影响面实测**（`scripts/oneoff/probe_size_only_exposure.py`，真实 HS300
      1853 个交易日 × 520 股）：因样本不足而降级到 size-only 的天数 **= 0**
      （行业面板 100% 交易日覆盖、93.2% 非 NaN），故只有"完全没传行业"的调用方
      受影响，**主实验不受影响**（主线信号层主口径 `raw`、对照 `neut` 传市值+行业、
      因子层 `panels_neu` 走 `preprocess_factor(mc, ind)` 两者同传）。
      原 characterization 测试已改写为 `test_neutralize_size_only_matches_with_intercept_version`。
    - **内联 t 统计量（2026-09-11 决策：全部收口）**：此前记的"5 处"是**漏数**
      ——搜索模式用了 `np.sqrt(n)`，而真实写法多为 `np.sqrt(len(ic))`。以宽模式
      复查后实际 **13 处**：`factor/synthesis.py`（`composite_stats`）、
      `factor/genetic_mining.py`（`_seg_ic_stats` / `:1129` / `:1669`）、
      `research/factor_analysis.py`（`standard_factor_summary`）、
      `research/attribution.py`（`fama_macbeth` 的 `t_ols`）、
      `scripts/archive/compare_htai_fitness.py`、`scripts/archive/compare_ml_synthesis.py`、
      `scripts/gp_tune_budget.py`（2 处）、`scripts/walk_forward.py`（2 处）、
      `scripts/build_minute_panel.py`。全部改走 `mean_inference(robust=False)`，
      并保住旧的 `n < 2 → 0.0` 边界（`mean_inference` 该情况返回 NaN）。
      等价性探针 `scripts/oneoff/probe_ols_t_unify.py`：A 型（调用方均已 dropna）
      **9/10 逐位一致**，唯一分叉是**常数序列**——旧写法因 pandas `Series.std()`
      对全等值返回 7e-18（`s > 0` 保护失效）给出 ~2e16 的荒谬值，统一实现走
      `np.std(ddof=1)` 给精确 0.0，属**修复**（IC 序列逐日截面相关，几乎不可能恒定）。
      两个例外：① `build_minute_panel.py` 旧写法**分子 skipna、分母却用含 NaN 的
      `len(ic)`**，两者本就不匹配，收口必然改数（已在代码注释中标注）；
      ② `attribution` 的 `se_ols` 是标准误、不是 t，保留。
      **是否补 NW 列 → 决策：不补**。这 13 处的 t 只用于报告 / 诊断输出，不参与
      因子库显著判定、不进任何回测收益；补 NW 只改 CSV schema 而无实际收益。
      新增静态守卫 `tests/test_significance.py::test_no_inline_ols_t_left_in_repo`
      防回潮（只匹配 `X / (Y / np.sqrt(n))` 的除法嵌套，标准误 `sd / np.sqrt(n)`
      不算），并有 5 个函数级测试锁定各入口 == `mean_inference`。
    - **顺带修复（2026-09-11，a 项实跑真实因子库时暴露）**：
      `stats.significance.t_pvalue` 只支持标量 `df`，而 `significance_table` 对
      缺 `p_value_nw` 的旧行用 `df = n_dates - 1`（**数组**）补算 →
      `not np.isfinite(df)` 对数组直接抛 ValueError。构造出的临时库每个因子都带
      p 值，该分支一直没被测到，**真实库一跑就炸**（hs300_2025 有 244 行旧数据）。
      已支持数组 `df`（语义与标量一致：有限且 >0 → t 分布；有限但 <=0 → NaN；
      非有限 → 正态近似），并加回归测试 `test_t_pvalue_supports_array_df` /
      `test_missing_pvalue_nw_is_backfilled`；顺手加固 `n_dates` 缺列时的回退。
      修复后实测 raw→FDR：**hs300_2025 244 个 84→62**、
      **hs300_2022_2025 863 个 397→337**。
    - **绩效指标**：`scripts/e2e_backtest.perf_stats`（5 个脚本消费）的三个基础量
      改调 `backtest.metrics` 原语，逐位一致；保留两处报告层独有约定
      （短样本 <0.3 年不年化、月胜率——与 metrics 的日胜率不是同一指标）。
      ⚠️ 唯一数字变化：`max_drawdown` 由负值改**正值**（全库其余口径均取正值），
      报告渲染由 `-38.7%` 变 `38.7%`，幅度不变。
    - 收益面板构造 3 处（`cli_common.returns_from_daily` /
      `data/cache_helpers` / `build_panel` 内联）——**注意**：这不是"要统一成
      一套写法"，IC 口径与引擎口径是分层设计、数学等价（`A = B.shift(-1)`），
      只需保证"每个口径一份实现 + 跨层显式声明"，见 MEMORY.md 收益面板条。
    - **反例（不要动）**：`strategy.constraints.neutralize_industry`（2026-09-11
      自 `optimize/portfolio.py` 下沉）是**权重级**投影，与因子级残差中性化范畴
      不同，不算重复。
  - [x] **确定权重生产的唯一入口（2026-09-11 完成，方案 A）**：
    诊断结论是**不合并模块，而是统一契约 + 消除真重复**。逐函数核对后，
    `strategy/` 与 `optimize/` 的重叠面**只有 2 处、约 12 行**：
    `optimize_weights(method="equal_topk")` ↔ `TopKLongOnly`、
    `method="factor_weighted"` ↔（strategy 层无对应物）。其余 133 行
    （行业中性投影 / 上下限 / 换手收缩）是 strategy 完全没有的能力，
    `solver.py` 的 QP/HRP/BL 与 `risk.py` 的风险分解更无从重叠。
    **不合并的三条理由**：① 依赖重量不对称——`backtest/engine.py:32` 只 import
    `strategy.base`（零三方依赖），而 `optimize.solver` 要 cvxpy，合并会让回测
    引擎背重依赖；② 契约粒度不同（单截面 `Series→Series` vs 面板
    `DataFrame→DataFrame`）；③ 层次不同（业界 alpha → 组合构建 → 执行）。
    落地：新建 `strategy/constraints.py`（面板级纯函数真源 =
    `build_signal_weights` / `neutralize_industry` / `apply_bounds` /
    `apply_turnover` / `apply_constraints`），`optimize/portfolio.py` 退化为
    **薄门面**（不再持有任何独立实现）；探针
    `scripts/oneoff/probe_portfolio_move.py` 穷举 72 个约束组合验证新旧
    **max|Δ|=0**（纯搬迁）。契约显性化：`PrecomputedWeightsStrategy` 从
    `optimize.multi_period` 导出为公开适配器并补 docstring——它是"优化产物 →
    `Strategy` 契约 → 回测引擎"的唯一通道。守卫：`tests/test_layering.py`
    加 strategy 层依赖守卫（不得 import optimize/backtest 及 cvxpy 等重依赖）+
    约束真源守卫；新增 `tests/test_constraints.py`（16 例）。
    **⚠️ 一处刻意不合并**：`equal_topk` 与 `TopKLongOnly` 的 **tie-break 不同**
    —— 前者 `rank(method="first")`（按列序**确定性**），后者 `sort_values()`
    （quicksort，**不稳定**）。探针 `scripts/oneoff/probe_tie_at_topk.py` 实测
    真实因子库（HS300 2025，40 面板 / 46930 截面）**4.4% 的截面**在 top-k 边界
    存在 tie（两个离散型因子接近 100%），委托会改这些截面的持仓集合，且是
    **向不确定实现退化**，故保留确定性实现并写进 docstring。
    **新发现（2026-09-11 第三批已完成）**：同一 tie 隐患也在
    `TopFracLongOnly` / `BufferedTopFracLongOnly` 等**主线用到的策略类**里
    （`sort_values()` 不稳定 → 同输入在不同平台/版本可能选出不同股票）。
    第三批已统一为 `rank(ascending=False, method="first")`（列序确定性），
    详见 3.1 末尾「口径统一第三批」。
  - [x] **`factor/synthesis.py` 归属倒挂（2026-09-11 完成，`a6a166c`）**：
    诊断发现该模块是**两类职责混装**——`ic_weighted` / `pca` / `orthogonal` /
    `build_components` 是**确定性因子组合**（挖掘闭环最后一环、直接喂因子库），
    而 `synthesize_stacking` 系列拟合**有监督模型**（ridge / LightGBM /
    LambdaRank + 时序 CV），是模型层的活。故**未整体搬迁**（会把因子组合错放进
    模型层，并让 `synthesize_library` / `gflownet_library_ingest` 这些纯因子库
    ingest 反向依赖 model），改为**按职责拆分**：
    - 新增 `model/stacking.py`：四个 stacking 合成器 + 私有辅助
      （`_make_target` / `_time_fold_masks` / `_inner_split_by_day` /
      `_rank_ic_by_day`）原样迁入；探针 `scripts/oneoff/probe_stacking_move.py`
      对四个合成器与折掩码逐位验证 **max|Δ|=0**（真正的纯搬运）；
    - `factor/synthesis.py` 只留确定性组合；`_long_matrix` 因被跨层复用提升为
      公开 `long_matrix`（与 `model.predictor._long_matrix` 同口径声明）；
    - **依赖方向固定为 model → factor**（与 `model/predictor.py` 早已 import
      `factor.cv` / `factor.preprocessing` 一致），`factor/` 不得反向依赖 model。
    改动面：`model/training.py` + 4 个脚本（`compare_ml_synthesis` /
    `gflownet_library_ingest` / `synthesize_factors` / `synthesize_library`）的
    import 站点；`model/features.py` / `model/predictor.py` / `factor/cv.py` /
    `factor/__init__.py` 的交叉引用注释；新建 `tests/test_stacking.py`（stacking
    + 时序 CV 测试迁入），`tests/test_synthesis.py` 只留因子层，共享 fixture
    `synth_parts` 上移 `tests/conftest.py`；`tests/test_layering.py` 新增两条守卫
    （factor ↛ model；stacking 实现必须在 model 且 factor 不得留转发口）。
    **顺带修复**：`tests/test_metrics.py` 上一批重复追加了同一批 3 个测试
    （F811 重定义 → pytest 静默去重，等于测试从未真正跑；CI 快检查的
    `ruff --select F811` 会红），已去重（`7a75235`）。
  - 备注：第二批动的是**口径与依赖边界**，与本批"先跑全量测试定基线、
    改完对比"的做法一致；建议一次只动一项并单独提交，便于定位是哪一项
    改变了哪些数字。

- [x] **口径统一第三批 —— 工程卫生（2026-09-11 完成，五项独立提交）**：

  基线：全量 **692 passed**（2411s / 40min，串行；改前跑通终态）。改后收集 **712**。

  - [x] **① tie-break 确定化（`1bd9100`）**：`TopKLongShort` / `TopKLongOnly` /
    `TopFracLongOnly` 由 `vals.sort_values()`（quicksort，不稳定）改为
    `rank(ascending=False, method="first")`（列序确定性），与
    `strategy.constraints.build_signal_weights` 及 `BufferedTopFracLongOnly`
    统一为全仓唯一 tie 规则。
    **影响面实测**（`scripts/oneoff/probe_tie_stable_sort.py` +
    `probe_alla_tie_impact.py`）：① 因子库面板（HS300 2025，40 面板 / 46930 截面）
    约 **4.4%** 截面在 top-k 边界并列，且几乎每个并列截面新旧持仓集合不同，
    最坏 Jaccard = 0（top-k 全换）——**离散型因子链路需按新口径重跑**；
    ② 主实验（全A正交化 ens_h1h5，2107 日 × 5801 股）tie 率 4.95%、月频调仓日
    8.57%，但差异日平均 Jaccard 0.995、对称差仅 **2.0 只**（占持仓 0.5%），
    月频调仓换手 0.7303 → **0.7303（零变化）** → **主实验数字不变**，属可复现性加固。
    顺带修复：`TopKLongOnly` 空截面 `1.0/0` 抛 ZeroDivisionError；旧实现
    `index[-0:]` 等价于全部索引（k=0 时给出 inf 权重）。
    新增 `tests/test_strategy_tie.py`（10 例）。
  - [x] **② scripts 层中性化去重（`6ebc6ae`）**：核查确认三份同名实现
    （`e2e_common.neutralize_predictions` / `optimize_e2e.neutralize_predictions_local`
    / `run_model_portfolio.neutralize_panel`）**都走 `factor.preprocessing.neutralize`
    同一真源，不存在口径分裂**。真正的问题是 `scripts/run_model_portfolio.py`
    这个**实验入口脚本被主实验/生产反向 import**：`DEFAULT_MODEL_PARAMS`
    → `model/params.py`（`037324e`）、`default_costs` → `backtest/costs.py`
    （`47b6976`）、4 个 legacy 组件（`neutralize_panel` /
    `load_index_benchmark` / `build_style_covariates_panel` / `build_model_panel`）
    → 新建 `scripts/portfolio_common.py`（`6ebc6ae`）。共 17 个 import 站点改向。
    守卫：`test_model_params_live_in_model_layer` /
    `test_default_costs_source_is_the_backtest_layer` /
    `test_experiment_entry_scripts_are_not_imported_by_other_scripts`。
    **遗留**：`run_model_portfolio` 仍 import `rolling_grid_alla`（复用其
    `FeatureStore` / `select_features_for_year`，属"管线组件复用"），未纳入守卫
    目标，待评估是否把该管线件也下沉。
  - [x] **③ 私有函数倒挂收口（`acdb07e`）**：`data/cache_helpers` 2 个
    （`pit_universe_codes` / `apply_membership_mask`）+ `factor/genetic_mining`
    5 个（`ls_net_stats` / `monthly_forward_returns` / `mutual_info_series` /
    `top_excess_series` / `htai_preprocess`）公开化，10 个 scripts + 测试站点改向。
    仍保留私有（**仅测试白盒引用**）：`_adjust_crowding` / `_restore_crowding` /
    `_dedup_hof_by_correlation` / `_ensure_creator` / `_seg_ic_stats`。
    守卫 `test_no_private_import_from_factor_and_data`（AST 解析，scripts 层一律
    禁 import 下划线名；tests 层白名单）。
  - [x] **④ 文本链路 IC 口径核查（仅核查，代码改动按纪律延后）**：探针
    `scripts/oneoff/probe_text_ic_convention.py` 在真实文本因子数据上实测
    `evaluate_senti.rank_ic_stats`（月度）/ `evaluate_sue_txt.rank_ic`（季度末）
    与 `stats.ic.calc_ic_series` **逐位一致**（共同有效期 max|ΔIC| 分别为
    2.8e-17 / 5.6e-17，期数 92/92、32/32 完全对齐）→ **无实质分叉**。
    三条真实差异（非数值分叉，需声明）：① 无显式「有效观测 <5 门槛」（canonical
    剔、legacy 只靠 `.corr()` 对 <2 样本返 NaN 兜底，本数据恰好未触发但 2~4 只
    股票的期会漏网 → 潜在假显著性）；② `evaluate_sue_txt.rank_ic` 的字段
    `n_months` 实际是**期数**且粒度是**季度末**（32 期 ≈ 8 年季度），命名误导；
    ③ 年化常数：senti 月度若误用 `calc_ir` 的 ×√252 得 ICIR 1.996（正确 ×√12
    = 0.436，虚高 4.6 倍），legacy 用不年化的 mean/std 规避了误用但使两条线
    ICIR 不可比。**延后原因**：`scripts/textmining/*` 正被另一会话编辑
    （8 个文件未提交），现在改会污染他会话的 diff。待其落地后补 <5 门槛 + 修字段名。
  - [x] **⑤ P2 清洁项（`1e36456`）**：删死代码 `factor/synthesis.py::_align_sign`
    （全仓零引用，活的是 `_align_sign_by_ic`）；**cvxpy 真·惰性导入**
    （`optimize/solver.py` 原模块级 try/except 名为"延迟"实则在 `import optimize`
    时执行，冷启 7.3s → 改 `_require_cvxpy()` 函数内按需加载）；
    **`optimize/__init__` 惰性导入**（PEP 562 `__getattr__` + `__dir__`，
    `import optimize` **10.2s → 0.02s**，cvxpy/openpyxl/scipy.stats 均不再进
    `sys.modules`）。新增 `tests/test_optimize_lazy_import.py`（4 例，含干净子进程
    验证）。**核实为误报**：`optimize/multi_period.py:26` 的 print 位于模块
    docstring 的「用法」示例中；AST 扫描全部核心包真实 print 调用数 **= 0**。
    **纠正审计前提**：11.7s 导入开销里 cvxpy 只占约 2s，其余是
    `stats.ic→scipy.stats`(4.0s) + `research.xlsx_report→openpyxl`(2.4s) +
    pandas(2.7s)。

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
2. ~~最小 CI~~（09-10 完成：已转为手动触发；3.2 机械清理亦已批量收口）
3. **口径统一第二批（见 3.1）—— 已全部完成**：显著性判定收口、统计/预处理/绩效原语
   收口、`factor/synthesis.py` 归属倒挂拆分、**四项待拍板决策全部落定**
   （a 保持 raw + FDR 降报告层 / b GFlowNet 训练捷径 + 入库 canonical 并显式声明 /
   c size-only 补截距 / d 13 处内联 t 收口且不补 NW 列）、**权重生产入口定案**
   （方案 A：约束算子下沉 `strategy.constraints`，`optimize/portfolio` 退化为薄门面，
   不合并模块），各自独立提交 + 全量测试比对
4. **口径统一第三批（见 3.1）—— 工程卫生，已全部完成**：tie-break 确定化 /
   实验入口脚本反向依赖消除（下沉 model.params + backtest.costs +
   scripts.portfolio_common）/ 私有函数倒挂收口 / 文本链路 IC 口径核查
   （结论：逐位等价，代码改动延后至他会话落地）/ P2 清洁（死代码 + 重依赖惰性化，
   `import optimize` 10.2s→0.02s）。遗留两项见 3.1 对应条目。
5. 分钟频挖掘 pipeline（现成数据的最大增量）
6. cli_common 批量推广 + 报告模板收编（机械清理，可穿插）
7. 攻"跑输基准"研究问题本身
