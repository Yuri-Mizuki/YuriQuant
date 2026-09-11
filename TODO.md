# YuriQuant 待办清单

> 2026-08-29 六轮工程整改（P0–P5）完成后的基线盘点。整改内容见提交
> `c6019f7` / `2c0a159` / `3b92fd3`（测试兼容修复 → stats 公共层 → scripts 收敛）。
> 现状：核心包达"清晰"档（零循环依赖、统计工具单一真源、521 测试全绿），
> scripts 层仍有存量样板债。按「对结论可信度的影响」排序。
>
> 本文件是**待办与历史交付的唯一真源**（2026-09-11 文档合一）。
> 研报研读/复现待办见 [`RESEARCH_TODO.md`](RESEARCH_TODO.md)；
> 架构与完成度总览、安装与用法见 [`README.md`](README.md)。

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

- [x] **口径统一第四批 —— 结构卫生 + 文档合一（2026-09-11 完成，四项独立提交，
  `ff060a5` / `8cdca30` / `c84ee4d` / `a03d36b`，已推送）**：

  用户给的问题清单经 AST 核查后**有 5 处与实测不符**（已逐条纠正，见各项）：

  - [x] **① 归档 7 个零引用一次性脚本（`ff060a5`）**：`compare_htai_fitness` /
    `compare_ml_synthesis` / `diagnose_factor_vs_model` /
    `diagnose_neutralized_compare` / `gtja_discipline_eval` /
    `ml_algorithm_compare` / `ml_synthesis_experiment` 移入 `scripts/archive/`。
    **刻意不走 `scripts/oneoff/`**：该目录整目录 gitignored（62 文件 / 0 跟踪），
    移过去等于从版本控制删除，而这 7 个里 5 个**支撑已引用结论**（报告23 四象限 /
    华泰·GTJA 复现对照 / 合成对比 / 算法对比），丢弃会毁掉可复现性。
    scripts 根 64 → 57，archive 5 → 12（git 记为 rename，历史保留）。
    **顺带修一处潜伏 bug**：脚本用 `Path(__file__).resolve().parents[1]` 作
    project root——在 `scripts/` 下恰为仓库根，移入 `scripts/archive/` 后变成
    `scripts/`，非 cwd 调用即 `from factor...` 失败。10 个文件改 `parents[2]`
    （含 archive 里原本就写错的 daily_pipeline / factor_usability_stats /
    dpp_library_compare），2 个补 bootstrap。
  - [x] **② 报告渲染收口（`8cdca30`）**：**审计推翻了原假设——公共层早已存在**
    （`research/html_report.py` 的 `page` / `BASE_CSS` / `render_table` /
    `base_js` / `SORT_JS` / `svg_sparkline` / `embed_image_b64`）。6 个"各写一套
    CSS"的脚本与 BASE_CSS **逐字节相同的声明 = 0 个**（逐选择器比对：body/h1/h2/
    .card/table/th/td 配色字号字体全不同）→ 删了只改视觉、无功能收益，
    **刻意不动**。真正绕开 `page()` 自拼外壳的是另外 **3 个**脚本
    （`jq_style_report` / `rolling_grid_report` / `alla_excess_attribution`），
    已收编；自拼外壳模块 4 → **1**（仅基座）。`page()` 新增 `extra_css=`
    （BASE_CSS 在前、增量在后的**组合**语义），补上"继承而非替换"的缺口。
    **端到端等价验证**：实跑重生成 `rolling_grid_report` 与重构前产物比对——
    `<style>` 块**逐字节一致**；body 仅 3 处差异且全可解释（1 行空白 / 时间戳 /
    canvas id）。⚠️ 顺带发现原有非确定性：`cid = "chart_" + str(abs(hash(title))
    % 10**8)` —— Python 字符串 hash 每进程随机 → 报告 HTML 不可字节复现
    （**未修，留作独立项**）。
  - [x] **③ 常量收口（`c84ee4d`）**：用户口径"只收同概念重复 + 项目级策略"。
    审计纠正：`ETF_CANDIDATES` **已收口**在 `data/etf_universe.py`；
    `MIN_PERIODS` 在 `monitoring/crowding`(重叠交易日) 与
    `compare_portfolio_methods`(协方差最少期数) 是**两个不同概念**，合并会造成
    假耦合 → 均不动。真正的散落是 **对照基准 5 处且取值各不相同**（同名字面量、
    不同用途）→ 新增顶层 `benchmarks` 段 + `Config.benchmarks()` 按**用途**分键
    （default / all_a / etf / report_a_share），**刻意不合并成一个值**。
    3 个脚本改读 config，逐项验等（取值全部不变）。
    同类倒挂补漏：`rolling_grid_alla._existence_mask` → `existence_mask`
    （被生产 `alla_daily_rank` 用）、`build_intraday_factors._minute_frame` /
    `_ex_div_keys`、`build_fundamental_factors._add_ttm_yoy` 公开化；
    守卫纳管模块表扩到 7 个 scripts 模块。
    新增 `test_frozen_recipe_stays_consistent_across_layers`：把"冻结配方"三处
    声明（settings.model_portfolio ↔ rolling_grid_alla ↔ alla_daily_rank）
    钉死，此前只靠注释约定一致。
  - [x] **④ 文档合一（`a03d36b`）**：README 的「待建 / 已知缺口」169 行
    （含 3 段已完成流水账）**无损迁入 `TODO.md` 附录「历史交付记录」**；
    README 替换为指针。该节开头 4 条「待建」bullet 未随迁（逐条核对确认
    §二已有对应条目，避免两处维护，附录头已注明）。三份文档互相加链并各自声明
    唯一真源。**顺带修 README 三类失真**：3 个失效引用
    （`ml_synthesis_round2` / `ml_decay_diagnosis` / `ml_window_compare` 文件已不存在）、
    「公共库」表只列 2 个而**实测 19 个 scripts 根模块被跨模块 import**
    （另有 textmining 10 / oneoff 5 / data_tools 1）、scripts 索引未随归档同步。
    体量：README 50.6KB → **37.8KB（−25%）**。

#### 🔴 第四批审计新发现（未修，需拍板）

- [ ] **生产脚本依赖 gitignored 目录（P0，真缺陷非洁癖）**：
  `scripts/alla_daily_rank.py`（生产每日推理）有 **6 处** `from scripts.oneoff.*
  import`（`build_alla_alpha_panels` / `build_alla_fundamental_factors` /
  `build_alla_constructed_factors._build_panels` / `build_alla_pledge_factors` /
  `build_alla_holder_factors._pit_holder_num,_pit_share_holder`），而
  `.gitignore:41` 忽略整个 `scripts/oneoff/`（本地 62 文件 / 0 跟踪）——
  **干净 clone / 生产环境上该脚本必然 ImportError**。
  修法需先拍板这批 `build_all_*` 面板构造函数归哪一层（它们产出的基本面/质押/股东
  面板是主实验输入；候选：`factor/` 或受跟踪的 `scripts/`）。
  已用 `tests/test_layering.py::test_production_scripts_do_not_depend_on_gitignored_oneoff`
  （xfail）记录在案，防"看起来全绿"把缺陷忘掉。
- [ ] **报告 HTML 不可字节复现**：`rolling_grid_report` 用
  `abs(hash(title)) % 10**8` 生成 canvas id，Python 字符串 hash 每进程随机
  （`PYTHONHASHSEED`）→ 同输入两次运行产物不同。改用 `hashlib` 摘要即可。
- [ ] **其余 scripts 层私有名跨模块 import（约 19 处）**：
  主要是 `scripts/textmining/*` 内部互引（另一会话在改，已豁免）与
  `scripts.oneoff.*`（同上条）。待两条落地后把守卫豁免项清零。
- [ ] **根目录 5 个 `_probe_*.py`**（`_probe_cols` / `_probe_pledge` /
  `_probe_quality` / `_probe_senti_artifacts` / `_probe_status`）：被
  `.gitignore:38 /_*.py` 覆盖、未跟踪、零引用。但**其中 2 个是另一会话当天在用**
  （`_probe_senti_artifacts` 12:26 / `_probe_status` 13:28），**本批未动**。
- [ ] **`strategy/enhanced.py` 零 import**（2533 字符）：确认无人引用，
  删或补文档说明归属待定。

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

---

## 附录：历史交付记录（2026-09-11 自 README 迁入）

> 原在 `README.md` 的「待建 / 已知缺口」节。这些是**已完成**批次的细节与实测数字，
> 保留在此以便追溯；**待办**请看正文 §一~§四。
> （该节开头的 4 条「待建」bullet——生产级执行 / 文本挖掘合规化 / 在线 dashboard /
> 分钟频挖掘层——未随迁：它们与正文 §二「研究功能缺口」逐条对应，避免同一件事两处维护。）

#### 已完成（2026-08-25）

- **组合级风险分解**（`optimize/risk.py: risk_decomposition`）：Euler 分解（MRC + CR + 占比）+ 风格/行业因子方差贡献（B'ΣB 分解）+ VaR/CVaR 成分分解（历史模拟法）+ 风险预算校验。报告脚本 `scripts/risk_decomposition_report.py`。
- **h=1 模型 CPCV 无偏评估**（`scripts/cpcv_h1_eval.py`）：固定 h=1 的 10 特征 + gbdt 超参，跑 15 条 CPCV 路径产出 IC 分布 + 路径间 t-test + horizon 对比（h=1/h=5/h=20），消除 horizon 选择偏差。
- **端到端选股工作流固化**（`scripts/e2e_common.py` + `e2e_stock_picks.py` + `e2e_backtest.py`）：

  ```
  今日选股:  D:/python/Python312/python.exe scripts/e2e_stock_picks.py --real --top 30
  策略回测:  D:/python/Python312/python.exe scripts/e2e_backtest.py --real --top 50 --freq M
  ```

  共享模块 `e2e_common.py` 统一：数据加载（因子库股票池 ~420 股）、经典量价因子、
  因子库 significant 加载（排除 `model:*` 防循环）、`build_feature_set` 三级漏斗选择、
  滞后面板新鲜度守卫（单因子停更不再拖短预测日，实测剔除 `alpha101_007` 后预测日
  从 2025-12-31 修复到 2026-08-21）。回测协议：特征选择只用回测前窗口 → 月频重训
  GBDT（embargo=5）→ top-N 等权/risk_parity（SCS 解经 `_enforce_caps` L2 投影
  修正约束违反）→ VectorBacktest 记账（含佣金/印花税/滑点）。测试
  `tests/test_e2e_pipeline.py`（7 passed，含 mock 端到端）。诚实结论：2024-01~
  2026-08 月频回测，等权 top50 +16.2%（Sharpe 0.41）、risk_parity top50 +22.7%
  （Sharpe 0.60），**均跑输全池等权基准 +30.8%**——现有信号强度不足以支撑
  top-50 集中持仓跑赢全池。
- **投资收益报告**（`scripts/investment_report.py`）：最终交付物，含两部分——
  ① **模型预测作为因子的检验**（`standard_factor_summary`：IC/ICIR/NW-t/IC 衰减
  + `quantile_backtest` 分层收益图 + 月度 IC 图；双口径：稀疏=调仓日信号、
  持仓=ffill 到日频与组合一致）；② **组合 vs 大盘指数基准**（沪深300池→000300.SH、
  全A→000001.SH，`--index` 指定；绩效含相对基准 alpha/β/信息比/超额）。
  实测（2024-01~2026-08）：稀疏口径 IC=0.067（NW-t=2.50 显著）、持仓口径
  IC=0.028（NW-t=1.70 边缘）；**预测分数默认做五因子中性化**（市值/行业/动量/
  波动/换手残差，`--no-neutralize` 关闭）——中性化后等权 +27.0%（Sharpe 0.66，
  回撤 −12.1%）、risk_parity +29.3%（Sharpe 0.73，已超指数 0.68），仍跑输
  沪深300 指数 +36.0%（超额 −8%~−10%），但较未中性化（等权 16.2%/Sharpe 0.41）
  大幅改善——剥离风格后的纯 alpha 风险调整后接近指数。
- **回测引擎对齐修复**（`backtest/engine.py`，2026-08-26）：结算-调仓顺序重构——
  当日收益由**上一调仓日设定的权重**赚取，新权重次日生效。此前调仓日 t 用新权重
  赚 rp[t]（t-1→t 信号日当天收益）是**前视一天**，A 股日内反转效应下产生系统性
  方向偏差（模型反向 top50 曾虚高 +56.5%）；修复后权重(t)↔收益(t→t+1)，与 IC/
  分层口径统一，且 daily_returns 日期标签与指数对齐（beta 由 -0.03 修正为 0.56）。
- **回测引擎三项修复 + 口径守卫**（`backtest/engine.py`，2026-08-27）：
  ① **h>1 区间结算修复**——此前 horizon>1 只在调仓日乘一次 (1+seg)^(1/span)、
  区间中间日收益记 0，长持有净值被系统性压缩（每日+2% 股票 h=5 月频仅得 ~0%）；
  现改为区间内逐日几何均摊复利，回归测试固化数值对账。**此前 freq_tune /
  multiyear_oos 的所有 h>1 结论为伪结果，需重跑**（真实复测：2025 段 gbdt 中性化
  Top20% 月频 h=5 超额 −20.0%、Sharpe 1.16 vs h=1 超额 +3.07%/Sharpe 1.60，
  h=1 仍显著占优但差幅可信）。② **avg_turnover 修复**——按调仓事件平均
  （turnover_series 仅调仓日有值），weights_history 改为每日记录真实持仓；
  此前 diff 口径被非调仓日零行稀释 ~7 倍。③ **收益面板口径守卫**——h=1 传入
  shift(-h) 前视面板直接报错；run_model_portfolio/freq_tune/multiyear_oos
  同步修正（h=1 传未 shift 的 pct_change()，与基准指数标签严格对齐）。
  诊断脚本 `scripts/archive/diagnose_factor_vs_model.py`：单因子 vs 模型同口径对比——
  **range20/alpha191_159/vol60 等负 IC 单因子月频 top50 超额 +20%~+36%，实为
  高波动/高 beta 风格在 2024-2026 上行期的暴露，非预测力**；模型 IC=0.067 最高
  （预测力最强）但 beta 0.56 偏低，牛市跑输风格。
- **优化实验**（`scripts/optimize_e2e.py`）：调仓频率 × 风格中性化四组对比
  （2024-01~2026-08 top-50 等权）——① **风格中性化有效**：M 月频 + 五因子残差
  16.2%→27.0%（Sharpe 0.41→0.66，回撤 −15.6%→−12.1%）；② **周频调仓被证伪**：
  W + 原分数 6.2%（Sharpe 0.19）远差于 M——换手成本与信号噪声吃掉 h=5 的短视野
  优势，减少信号衰减的假设不成立；③ 中性化对 W 同样有效（6.2%→18.4%）。
  最优配置 = **月频 + 中性化**，已并入 `investment_report.py`（默认开启）。

#### 固化配置与正式入口（2026-08-25，最终交付物）

- **模型增强组合固化配置**（`config/settings.yaml` 的 `model_portfolio` 段 +
  `strategy/examples.py: TopFracLongOnly`，入口 `scripts/run_model_portfolio.py`，
  输出 `reports/model_portfolio/`）：把「模型信号 → 风格中性化 → Top20% 重仓多头」
  固化为可复用正式入口，配置只此一处真源。默认 horizon=1 / gbdt / frac=0.20 /
  月度调仓，2025 test 段成本后超额沪深300 **+3.07%**（Sharpe 1.60，MaxDD −11.3%；
  引擎口径修复后复测值）。关键规律：只支持 h=1（h=5 组合难变现）、Top20%
  是 alpha 密度甜点、中性化是信号变现的关键（raw IC 高但跑输风格）。
- **修复 LGBRankerPredictor**（`model/predictor.py`）：`fit()` 曾硬编码
  `N_BINS=30`+`objective=lambdarank`、绕过构造参数导致负 IC(−0.055)；改为使用
  实例参数、默认 `labels_bins=2`（截面中位数二分）+`rank_xendcg` 后样本内 IC
  由负转正至 +0.207（434/484 天为正），现有调用方无需改动。
- **调仓频率精修**（`scripts/freq_tune.py` → `reports/freq_tune/freq_tune.csv`）：
  h=1 时 M 月度最优（超额 +6.1%/Sharpe 1.55）；频率越高换手越严重侵蚀 alpha
  （日频超额 −40.6%）。2026-09-01 已在修复后引擎下重跑：h5×M 从 BUG 下的伪结果
  （Sharpe −3.1）修正为超额 −16.6%/Sharpe 0.95，h=5 仍显著劣于 h=1 但差幅可信。
  修复前旧结果存档于 `reports/freq_tune/freq_tune_prefix_engine_fix_20260825.csv`。
  注：turnover 列自 BUG-2 修复后为"按调仓事件平均"口径，与修复前的稀释口径不可比。
- **多年度 OOS 稳健性**（`scripts/multiyear_oos.py` → `reports/multiyear/`）：
  gbdt/ridge/ranker × h1/h5 × D/W/M 在 2023/2024/2025 分年 walk-forward
  （特征在定型期 fixed，防前视选择）。2026-09-01 修复后引擎重跑：
  **h1×M 仍是唯一三年一致稳健的频率解**——gbdt +5.5%±1.7%、ranker +5.4%±4.6%
  （均 3/3 年正超额）；ridge 全负；h5×M 三年平均 ≈ −15%（与 freq_tune 2025
  单年 −16.6% 互证）；日频 −33%~−46% 全灭。h>1 旧结论系 bug 伪结果的注记就此了结。

#### 已完成（2026-09-01）全A多年度滚动训练实验（最终交付物）

- **数据回补**（`scripts/oneoff/fetch_alla_history.py` + `fetch_status_batched.py`）：
  全A日线 2015-01~2026-09-01（1098 万行 × 5549 股，分批断点续拉，SDK 大清单
  单查会挂死的对策）、状态表回补 2015-2018、交易日历/上证指数基准 000001.SH。
  backward_factor / equity_structure / 行业分类原本已全A覆盖。
- **全A公因子数据集 `all_a_2018_2026`**（`scripts/oneoff/build_alla_alpha_panels.py`，
  54 分钟 4 进程）：alpha101/158/191/360 共 **798 因子 × 2471 日 × 5549 股**
  （float32 面板，~24GB）。截面算子要求全截面在场——按因子分片并行而非按股票分块；
  逐因子落盘 + 预计算 4 个 horizon 的日频 IC 缓存（特征漏斗零 IO 复用）。
  工程教训两条：float64 巨值在 astype(float32) 时溢出成 inf（须转换后再
  replace）；爆炸量级因子（alpha191_017 最大 ~3e38）必须加载时 zscore+clip。
- **实验主链**（`scripts/rolling_grid_alla.py`，四阶段断点续跑，单测
  `tests/test_rolling_grid_alla.py`）：每年特征选择（过去 500 日 horizon 匹配
  IC + 覆盖率 + 项目正典 cross-spearman 去冗余）→ 季度（h≥10 半年）walk-forward
  滚动训练（embargo=horizon，500 日滚动窗）→ **幽灵股守卫**（LightGBM 对全 NaN
  特征照样输出预测，未上市股会占满信号顶部——预测面板按"当日有行情且 ≥1/4 特征
  可用"掩码，2019 年实测可用截面 3393 只/日）→ 含成本与涨跌停/停牌过滤的
  向量化回测。**120 组合**：horizon{1,5,10,20} × 频率{D,W,M,2M} ×
  模型{ridge,gbdt,ranker,+h1 超参/窗口变体} × 中性化{on,off} × TopFrac{20%,10%}。
- **报告**（`scripts/rolling_grid_report.py` → `reports/alla_rolling/report.html`）：
  总览排序表 / 模型 IC 分年表 / 分年超额热力图（红涨绿跌）/ 五组维度对比净值
  曲线（含上证与全A等权双基准）/ 分年超额柱状图 / 稳健性小结，六条诚实披露。
- **核心结论**（全部含成本、样本外 2018-01~2026-09）：全A截面 OOS IC 显著强于
  HS300（gbdt h1 全期 **IC=0.108**、9 年全正 0.078~0.157，ICIR 年化 14.6）；
  **h=1 + 月频/周频 + gbdt 系 + Top10% raw 是一致稳健解**（gbdt h1×M 年化
  14.0%、超额上证 +11.8%/年、超额全A等权 +5.2%/年、正超额 7/9 年；gbdt_deep
  h1×W 正超额 8/9 年）；**日频调仓被成本全灭**（超额 −13%~−35%）；**风格中性化
  在全A上稀释收益**（raw 14.0% vs neut 5.5%，与 HS300 上"中性化是变现关键"
  结论相反——全A宽截面上 raw 信号本身即含可变现 alpha）；h5/h10/h20 组合偏弱
  部分源于 0.7 去冗余下长 horizon 仅剩 3~16 个特征（披露⑥）。
- **局限**：超参未在本数据重调（防二次窥探）。
- **2026-09-07 数据修复（幸存者偏差消除 + 卫生清理）**：① 按 SDK 历史清单
  （沪深A 2015 至今含退市 ∪ 当前全A，5810 只）回补退市股 K 线/股本/复权因子
  （238 只退市类、237 只有完整历史；000562.SZ 2015-01 换股退市无窗口内行情），
  面板与实验全链路重建后重跑（09-08 完成，156 组合）：头部配置超额较修复前
  回落（gbdt h1×M raw Top10% +11.8% → +10.2%/年，量级合理）；**h1+h5 秩平均
  集成（ens_h1h5）成为新头部**（年化 13.8%、超额上证 +11.5%/年、超额全A等权
  +6.4%/年）；消融显示基本面/股东族带来 ~+0.7pp/年的一致增量（含族 +10.2% vs
  纯量价 +9.5%，同口径对照）；
  ② 状态表按完整历史清单全量重拉（分年覆盖 2814→5572 单调递增，修复
  2019-2021 只有 ~500 只的锯齿，掩码全年份生效）；③ 基本面因子面板混入的
  32 只 ETF 列已剔除并重算 IC；④ 样本末端 2026-09-02 盘中半拉数据已从全链路
  裁剪（水位回退 20260901，下次 update_data 整日重拉）；⑤ 特征筛选升级为
  DPP（`research.dpp_selection`）+ 基本面/股东族保留席位
  （`RESERVED_FUNDAMENTAL_SLOTS`），并新增 smallcap/ensemble/ablation/ortho
  子实验（`--stage smallcap/ensemble`、`--ablation`、`--preproc ortho`）。
- **数据更新守卫（2026-09-07）**：① 盘中拉数守卫——`update_data` 默认把"未到
  17:00 的今天"从拉取终点摘除（`--allow-intraday` 跳过），杜绝 daily 水位被
  盘中半拉数据永久污染（daily 增量起点=last+1，不会自愈）；② 状态表拉取在
  cache 层统一 200 只/批 + 3 次重试（SDK 大清单单查会挂死）。
- **全A面板刷新流程**（日线有新日期后，按序）：① `update_data --pool all_a`；
  ② `python scripts/oneoff/build_alla_alpha_panels.py --workers 6`（全量重算
  ~1h，不加 --resume 才会覆盖延伸）；③ 基本面族按需重跑 `build_alla_fundamental_
  factors / _holder / _pledge / _constructed`（源头已过滤非股票列）；
  ④ `rolling_grid_alla --stage prep` 重建 _base；⑤ 实验各阶段按产物断点续跑。
- **因子层正交化三臂对比（2026-09-09，`--preproc ortho/mixed`）**：zscore（不正交）
  vs ortho（全因子中性化）vs mixed（仅基本面族中性化+量价 raw），120 同配置配对：
  **全正交 +1.6pp/年 > 混合 +0.1pp ≈ 不正交**——正交化的增量几乎全部来自量价因子
  （隐形小盘/流动性/波动暴露是大头），单独对基本面做 ≈ 零增量；40/42 个 h1 配置
  全正交占优，牛市微让、2022 后震荡年份保护明显。结论：ortho 口径为全A管线默认。仓位 overlay 网格（2026-09-09，
  `position_overlay_grid.py`）：波动率目标/趋势MA200/回撤状态机均**不提升
  Sharpe**（0.52~0.61 vs 基线 0.63）但显著压回撤——趋势MA200减半 MaxDD
  38.7%→23.1%、年化 +11.8%，组合 overlay MaxDD 18.8%、年化 +8.2%——按
  Calmar（0.40→0.51）取舍。另发现引擎特性：传 executable_mask 时多头权重
  会被重归一到满仓，仓位缩放型 overlay 须改用信号预掩码表达可交易性。ortho 臂补跑 h1+h5 集成后复现集成增益
  （单模型 +11.8% → 集成 +13.3%），且高于 zscore 臂同款集成的 +11.5%——
  正交化与集成的收益叠加成立。stage_ensemble 回测窗口 bug 已修（原用 close
  全索引，2016-2017 无信号空仓期稀释年化/超额：+9.9% → 修复后 +13.3%，
  与主口径逐位一致）。
- **生产化每日推理**（2026-09-08，`scripts/alla_daily_rank.py`，单测
  `tests/test_alla_daily_rank.py`）：最优方案每天盘后自动产出全A选股排名——
  与实验同代码路径尾部重算当年入选 50 因子 → gbdt 500 日窗重训预测最新截面 →
  幽灵股守卫 + 信号日可交易性标注 → `reports/alla_daily/`。一致性验证：与实验
  冻结 OOS 面板 2026-09-01 截面 Spearman 0.92（每日重训比实验季度折更新鲜，
  非完全一致属预期）。
