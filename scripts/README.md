# scripts 入口索引

> 自动生成（122 个活跃入口）。改脚本 docstring 后运行 `python -m scripts.data_tools.gen_scripts_index` 更新；`--check` 校验是否过期。

约定：`python -m scripts.<目录>.<文件名>` 运行；`scripts/archive/` 已于 2026-11-18 整体下线（活代码已上移，其余删除）；实验结论查 `reports/README.md` §一「结论速查」，产物落盘规则见其 §1.6。

## builders

| 脚本 | 说明 |
|---|---|
| `builders/build_alla_alpha_panels.py` | 全A公开因子面板重建（alpha101/158/191/360）→ all_a_2018_2026 数据集。 |
| `builders/build_alla_altfactor_neu.py` | alt 族（altf 20 因子）的 panels_neu / panels_neu_fundind 回填（family-aware）。 |
| `builders/build_alla_altfactors.py` | 全A 另类数据三族因子（altf 族）→ 并入 all_a_2018_2026。 |
| `builders/build_alla_barra_styles.py` | Barra CNE6 风格因子原始暴露面板（简化版）→ all_a_2018_2026/panels_barra/。 |
| `builders/build_alla_constructed_factors.py` | 构造型基本面因子族（B++）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_disc_holder_dyn.py` | 大宗折价率 + 股东明细动态因子 → 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_ev_rd_factors.py` | EV 分母价值因子 + 研发加回盈利因子（华泰 AI95 两项数据改造，P0 增量）。 |
| `builders/build_alla_event_factors.py` | 全A事件驱动短周期因子面板（evt 族）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_factor_neutralized.py` | 预计算全注册因子的因子层中性化面板（#3 优化）。 |
| `builders/build_alla_factor_neutralized_fundind.py` | E5 变体：基本面族「只对行业中性化」的面板变体（TODO_ARCHIVE 册Ⅲ 批次 6）。 |
| `builders/build_alla_fundamental_factors.py` | 全A财务报表基本面因子面板（B族）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_holder_dyn_factors.py` | 全A 股东/内部人**动态持股**因子面板（holder_dyn 族）→ 并入 all_a_2018_2026。 |
| `builders/build_alla_holder_factors.py` | 全A股东结构行为因子面板 → 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_llmmcts_factors.py` | LLM-MCTS 候选公式入库（all-A 域求值，2026-10-08）。 |
| `builders/build_alla_margin_factors.py` | 全A两融资金流因子面板（margin 族）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_moneyflow_factors.py` | 全A机构资金流因子面板（moneyflow 族）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_p5_factors.py` | P5 补缺基本面因子族（B5）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_pledge_factors.py` | B+ 基本面族因子面板 → 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_status_factors.py` | 停牌 / ST状态因子 → 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_style_factors.py` | 基本面风格因子族（C·Style）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_sue_pledge_factors.py` | SUE 标准化意外盈余 + 质押深度因子 → 并入 all_a_2018_2026 数据集。 |
| `builders/build_replica_factors.py` | 研报复现专用个股因子构建（只在 `reports/report_replica/` 落盘，不进 920 因子库）。 |
| `builders/common.py` | 全A builder 共用件（2026-09-11 自 11 个 build_alla_* builder 收敛）。 |
| `builders/fetch_alla_history.py` | 全A历史数据回填（2015~now）——分批断点续拉版。 |
| `builders/industry_context.py` | 申万一级行业上下文：PIT 行业归属 + 自由流通市值权重 + 调仓日 |

## common

| 脚本 | 说明 |
|---|---|
| `common/alla_common.py` | 全A滚动实验的**共享底座**（2026-10-19 架构清理）。 |
| `common/cli_common.py` | 脚本层公共 CLI 骨架 |
| `common/feature_funnel.py` | 特征选择漏斗（覆盖率 → 去冗余 → 质量分截断）—— 口径中性的单一实现。 |
| `common/fundamental_common.py` | 财报长表的 TTM / 单季拆解公共函数（2026-09-11 收敛）。 |
| `common/im_features.py` | ``im_*`` 分钟特征装载与评估窗口对齐（GFlowNet / GP 第二层共用口径）。 |
| `common/portfolio_common.py` | 跨实验脚本共享的组合编排组件（无 CLI 入口） |
| `common/power.py` | 任务运行期间请求系统保持唤醒（**在本机 S0 连接待机下无效，见下**）。 |
| `common/task_scheduler.py` | Windows 计划任务注册收口 + 调度体检（单一真源）。 |

## data_tools

| 脚本 | 说明 |
|---|---|
| `data_tools/audit_fundind_split.py` | E5 覆盖层体检：``panels_neu_fundind`` vs 基准层 ``panels_neu``。 |
| `data_tools/docs_lint.py` | 文档规范校验（docs 治理 P0，2026-10-28）。 |
| `data_tools/gen_scripts_index.py` | 生成 ``scripts/README.md`` 入口索引（2026-10-19 审计 ③）。 |
| `data_tools/memory_lint.py` | 记忆层治理校验（memory_lint，2026-11-03）。 |
| `data_tools/new_experiment.py` | 新建实验目录：一条命令完成「建目录 + 复制模板 + 登记判定」。 |
| `data_tools/recycle.py` | 走 Windows 回收站删除（**不永久删除**），供维护类脚本复用。 |

## evaluation

| 脚本 | 说明 |
|---|---|
| `evaluation/buffer_arm_920.py` | 批次 1.4：定版信号（h1020）的 buffer(20/30) 臂 vs naive top20。 |
| `evaluation/caliber.py` | 统一评估口径 —— 全项目**唯一真源**（2026-10-13 建立）。 |
| `evaluation/cheap_lever_revalidate_920.py` | cheap_lever 胜者的 920 定版信号复验（好机器批次 11，2026-10-17）。 |
| `evaluation/fundamental_blend.py` | E3 基本面慢信号 ⊕ 快信号双层融合（信达式分域建模 + 预测层 λ 加权）。 |
| `evaluation/horizon_mix.py` | E1'' horizon 混合消融：把慢 horizon OOS 预测混入 ens_h1h5 的对照实验。 |
| `evaluation/solo_live.py` | yqcore-focus-k8 生产持仓生成器（2026-11-10 建立）。 |
| `evaluation/yq_core_report.py` | `yqcore` 族（YQ 核心）产品线报数脚本 —— 三档的统一口径对照 + 净值曲线。 |

## factors

| 脚本 | 说明 |
|---|---|
| `factors/backfill_library.py` | 因子库回填：体检（dry-run）+ 并行补 canonical 评估。 |
| `factors/backfill_library_metrics.py` | 因子库指标批量回填：canonical 分层回测 + 显著性 + 路径/公式/族标签（维护工具）。 |
| `factors/backfill_metadata.py` | registry 元数据回填：``formula`` 与 ``family``（只补空缺，不动任何统计列）。 |
| `factors/build_fd_factors.py` | 飞书《因子字典》候选因子 → 因子库体检入库 |
| `factors/build_intraday_factors.py` | 日内特征 → 日频因子构建与入库 |
| `factors/build_intraday_stat_factors.py` | 分钟统计特征 → 因子库体检入库（降维路线第一层） |
| `factors/build_minute_panel.py` | 分钟面板构建入口 |
| `factors/build_top10_factors.py` | 「中国因子十强」复现与统一评估（JKP 口径榜单，2026-11-02） |
| `factors/dup_audit.py` | 全库冗余审计：一次性算清 965 个因子的两两截面相关，并把结论写回 registry。 |
| `factors/dup_precheck.py` | **登记前**冗余预检：这批因子该不该进库，先看一眼它最像谁。 |
| `factors/factor_board_data.py` | 因子看板数据层：为因子库里每个因子算「分位组合净值 + IC 序列 + 行业 IC」。 |
| `factors/factor_library.py` | 因子库 CLI |
| `factors/formula_extract.py` | 从 factor/ 源码反解析因子的**可读公式**，供 registry ``formula`` 列与看板展示。 |
| `factors/library_dedup_apply.py` | 库内去重的**执行层**：把决策表的 ``safe_drop`` 从 registry 退登记。 |
| `factors/library_dedup_plan.py` | 因子库去重决策表：把「相关 → 质量 → 生产依赖」三条判据并成一张可拍板的清单。 |
| `factors/mine_factors.py` | 因子挖掘 CLI |
| `factors/panel_leak_audit.py` | 全库面板前视体检（只读）：把「哪几块面板还在用未来数据」从猜测变成清单。 |
| `factors/panel_leak_audit_diff.py` | 把两次体检的 CSV 放到**同一套判据**下重判，输出可比的前后对照。 |
| `factors/prune_registry_backups.py` | registry 备份轮转：只保留「有用的快照」，其余走回收站。 |
| `factors/run_alphapool_ppo.py` | 华泰 AI97《大模型+强化学习因子挖掘》—— AlphaPool + MaskablePPO 训练入口 |
| `factors/run_factor_pipeline.py` | 因子流水线编排：实现 → 评估 → 看板数据 → 看板页面（2026-11-19）。 |
| `factors/run_gflownet_phase1.py` | GFlowNet Phase 1：真实 HS300 对齐研报 |
| `factors/run_llm_mcts.py` | 东吴 LLM-MCTS 因子迭代 Phase 0 —— 同题四引擎对照 runner |
| `factors/run_portfolio_phase2.py` | stage2 Phase 2：zz1000 主实验（信号管线 + PPO 滚动集成 / 银河口径 QP / 基准）。 |
| `factors/run_portfolio_ppo.py` | stage2 Phase 1：hs300 平价验证（PPO vs QP vs 等权基准 vs TopN）。 |

## ingest

| 脚本 | 说明 |
|---|---|
| `ingest/extend_factor_library.py` | 因子库数据延长 —— 面板刷新到数据源最新交易日 |
| `ingest/fetch_minute_all.py` | 全 A 分钟数据拉取（5 分钟 / 1 分钟） |
| `ingest/fetch_minute_gap.py` | 补齐 HS300 动态池（PIT 成分）缺失的分钟数据 |
| `ingest/fetch_status_batched.py` | 分段拉取全A历史状态表（涨跌停/停牌/ST/除权标记）——增量断点续拉版。 |
| `ingest/update_data.py` | 每日数据更新脚本 |
| `ingest/update_etf.py` | ETF 数据更新脚本 |
| `ingest/update_sw_industry.py` | 更新申万一级行业指数日线（31 条，1999-12 起） |

## live

| 脚本 | 说明 |
|---|---|
| `live/inc_pure_live.py` | `inc_pure` 实盘出单器：董监高主动增持 → T+1 开盘集合竞价买入，持 20 个交易日 |
| `live/show_holdings.py` | yqcore-focus-k8 持仓查询（只读，随时可跑） |
| `live/yqcore.py` | yqcore-focus-k8 生产入口：出榜 / 看持仓，两个独立子命令 |

## model_train

| 脚本 | 说明 |
|---|---|
| `model_train/decomp_dl_industry.py` | 诊断「个股预测聚合到行业后 IC 归零」：是信息不存在，还是聚合口径丢掉了？ |
| `model_train/eval_dl_industry.py` | 把自训 GRU / TCN 个股预测聚合成行业因子，出 IC / 组合表 + 诊断报告。 |
| `model_train/replica_deep.py` | 按研报 §2.1.1 训练 GRU / TCN 行业轮动深度学习因子（个股加权到行业）。 |

## pipelines

| 脚本 | 说明 |
|---|---|
| `pipelines/alla_daily_rank.py` | 全A滚动模型每日选股排名 —— 主实验信号形态（gbdt h1+h5 秩平均 × M raw Top10%）的生产化推理。 |
| `pipelines/alla_excess_attribution.py` | 全A主策略超额收益归因：α/β 分解 + Brinson 行业归因。 |
| `pipelines/fetch_altdata_daily.py` | 另类数据每日抓取/存档入口（P0 Step 1a + Step 3） |
| `pipelines/rebuild_panel_library.py` | 因子库面板月度重建（守卫式入口，计划任务用）。 |
| `pipelines/rolling_grid_alla.py` | 全A多年度滚动训练实验：horizon × 调仓频率 × 模型/超参 × 年份（2018~now）。 |

## portfolio

| 脚本 | 说明 |
|---|---|
| `portfolio/compare_portfolio_methods.py` | 组合优化方法对比脚本（P2，2026-08-14） |
| `portfolio/corr_sector.py` | 相关性行业归属面板（"长得像谁"）—— 构建 + 与申万标签的重叠率诊断 |
| `portfolio/etf_champ_signals.py` | ETF 轮动生产信号板（monthly / balanced） |
| `portfolio/etf_contest5_eval.py` | ETF 轮动 · 档位评测（80 个滚动 2 月窗口，口径统一 T+1 open） |
| `portfolio/etf_contest_spec.py` | ETF 轮动定版参数（单一真源，2026-10-27 锁定 / 2026-11-19 去比赛化） |
| `portfolio/etf_pool_monitor.py` | 池内环境监测：策略「机制条件」是否还成立的唯一事前指标（2026-10-30）。 |
| `portfolio/etf_rotation_panels.py` | ETF 轮动扩展面板（生产组件） |
| `portfolio/etf_rotation_probe.py` | ETF 轮动回测引擎（数据装载 / 面板 / 回测 / 出场调度） |
| `portfolio/industry_neff_probe.py` | 31 个申万一级的「有效独立方向数」诊断（P0-1） |
| `portfolio/industry_rotation.py` | 行业池轮动线（独立实验线，不与 ETF 三锋混跑）。 |
| `portfolio/multi_period_backtest.py` | 多期组合执行回测脚本（P3，2026-08-20） |
| `portfolio/report_replica.py` | 研报复现线（国泰海通 2026-10-29《七维共振》）· 只做可行的部分。 |

## reporting

| 脚本 | 说明 |
|---|---|
| `reporting/attribution.py` | 收益归因 CLI |
| `reporting/build_daily_email_body.py` | 从出榜产物生成每日邮件正文（markdown + 配套 HTML）。 |
| `reporting/build_monitor_seed.py` | 构造 monitor_production_ic 的生产口径历史种子面板（一次性，2026-10-17）。 |
| `reporting/factor_board.py` | 因子看板前端（仿聚宽因子库：列表页 + 因子详情页）。 |
| `reporting/factor_dashboard.py` | 聚宽式因子看板：筛选 + 排序表 + 点行看详情（自包含 HTML）。 |
| `reporting/factor_library_full_report.py` | 全因子库检验报告：821 个因子的标准检验汇总（自包含 HTML）。 |
| `reporting/jq_style_report.py` | 最佳策略收益曲线页（聚宽模拟交易风格）——策略 vs 中证全指口径基准。 |
| `reporting/md_to_email_html.py` | 日报 markdown → 邮件 HTML（内联样式，适配 163 / QQ / Gmail 等客户端）。 |
| `reporting/monitor_factor_health.py` | 定版因子库健康度巡检（因子 + 模型预测）。 |
| `reporting/monitor_production_ic.py` | 生产口径每日 IC / 风格暴露监控 CLI（全A · 定版 dingban · 2026-10-17 起）。 |
| `reporting/picks_board.py` | 本期出榜解释页（为什么买这些）—— 接在模拟交易页之后。 |
| `reporting/rolling_grid_report.py` | 全A滚动训练实验报告生成器：收益曲线 + 分年绩效 + 维度对比（自包含 HTML）。 |
| `reporting/strategy_home.py` | 策略首页（历史回测详情）+ 模拟持仓页 |
| `reporting/yqcore_dashboard.py` | `yqcore` 定版策略前端页面（自包含 HTML，零后端）——**历史回测**与**模拟交易**分两页。 |

## textmining

| 脚本 | 说明 |
|---|---|
| `textmining/_paths.py` | textmining 输出路径约定。 |
| `textmining/build_senti_factors.py` | AI 41 研报情感四因子构建（senti / senti_adj / report_num / report_score） |
| `textmining/build_senti_scores.py` | 研报摘要 FinBERT 情感打分（对齐华泰 AI 41 研报情感因子） |

## timing

| 脚本 | 说明 |
|---|---|
| `timing/rev_shell_probe.py` | 周频均值回归壳稳健性探针（纯择时线缝合清单③，reports/rev_shell_probe/）。 |
| `timing/shell.py` | 纯择时研究线共享件：调仓壳 + 指数日K装载（2026-11-21 自 sws_regime_probe 抽取）。 |
| `timing/sws_regime_probe.py` | 申万 0922 择时研报门槛探针（纯择时研究线首篇，reports/sws_timing_probe/）。 |
