# scripts 入口索引

> 自动生成（154 个活跃入口；archive/oneoff 不计）。改脚本 docstring 后运行 `python -m scripts.data_tools.gen_scripts_index` 更新；`--check` 校验是否过期。

约定：`python -m scripts.<目录>.<文件名>` 运行；退役入口在 `scripts/archive/`（仍可运行，但不再维护）。

## builders

| 脚本 | 说明 |
|---|---|
| `builders/backfill_financial_alla.py` | 回补全A财务报表四表（income/balance_sheet/cash_flow/dividend）到含退市。 |
| `builders/backfill_holder_alla.py` | 回补股东结构数据（holder_num / share_holder）到全 A 口径（含退市）。 |
| `builders/backfill_pledge_profit_alla.py` | 回补全A（含退市）股权质押 / 业绩预告 / 业绩快报三张新表到本地 parquet。 |
| `builders/build_alla_alpha_panels.py` | 全A公开因子面板重建（alpha101/158/191/360）→ all_a_2018_2026 数据集。 |
| `builders/build_alla_altfactor_neu.py` | alt 族（altf 20 因子）的 panels_neu / panels_neu_fundind 回填（family-aware）。 |
| `builders/build_alla_altfactors.py` | 全A 另类数据三族因子（altf 族）→ 并入 all_a_2018_2026。 |
| `builders/build_alla_barra_styles.py` | Barra CNE6 风格因子原始暴露面板（简化版）→ all_a_2018_2026/panels_barra/。 |
| `builders/build_alla_constructed_factors.py` | 构造型基本面因子族（B++）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_disc_holder_dyn.py` | 大宗折价率 + 股东明细动态因子 → 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_event_factors.py` | 全A事件驱动短周期因子面板（evt 族）→ 并入 all_a_2018_2026 数据集。 |
| `builders/build_alla_factor_neutralized.py` | 预计算全注册因子的因子层中性化面板（#3 优化）。 |
| `builders/build_alla_factor_neutralized_fundind.py` | E5 变体：基本面族「只对行业中性化」的面板变体（GOOD_MACHINE_TASKS 批次 6）。 |
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
| `builders/common.py` | 全A builder 共用件（2026-09-11 自 11 个 build_alla_* builder 收敛）。 |
| `builders/fetch_alla_history.py` | 全A历史数据回填（2015~now）——分批断点续拉版。 |
| `builders/run_evt_margin_build.py` | 事件/两融/资金流数据回补 + 因子构建编排器（串行，防止 SDK 单连接竞争）。 |

## common

| 脚本 | 说明 |
|---|---|
| `common/alla_common.py` | 全A滚动实验的**共享底座**（2026-10-19 架构清理）。 |
| `common/atomic_io.py` | 原子写面板 + 抗瞬时文件占用（Windows 原生锁）。 |
| `common/cli_common.py` | 脚本层公共 CLI 骨架 |
| `common/fundamental_common.py` | 财报长表的 TTM / 单季拆解公共函数（2026-09-11 收敛）。 |
| `common/portfolio_common.py` | 跨实验脚本共享的组合 / 模型编排组件（无 CLI 入口） |
| `common/power.py` | 任务运行期间保持系统唤醒，防止被「现代待机」中断。 |
| `common/task_scheduler.py` | Windows 计划任务注册收口 + 调度体检（单一真源）。 |

## data_tools

| 脚本 | 说明 |
|---|---|
| `data_tools/dump_pdf_text.py` | 把研报 PDF 抽成带页码标记的纯文本，便于按关键词定位口径参数。 |
| `data_tools/fix_meta.py` | 修复 e:/data/parquet/_meta.json：对齐实际日期 + 补文本表 + 加 pool 口径。 |
| `data_tools/gen_scripts_index.py` | 生成 ``scripts/README.md`` 入口索引（2026-10-19 审计 ③）。 |

## evaluation

| 脚本 | 说明 |
|---|---|
| `evaluation/backtest_two_periods.py` | sig_composite_pca 两时段样本外回测：2025 全年 vs 2026 上半年。 |
| `evaluation/buffer_arm_920.py` | 批次 1.4：定版信号（h1020）的 buffer(20/30) 臂 vs naive top20。 |
| `evaluation/buffer_tune.py` | 执行阻尼实验：排名缓冲带 vs 硬切 Top20%（2026-09-01） |
| `evaluation/caliber.py` | 统一评估口径 —— 全项目**唯一真源**（2026-10-13 建立）。 |
| `evaluation/capacity_table.py` | 组合层容量表：集中度 × AUM（统一口径，2026-10-13）。 |
| `evaluation/cheap_lever_revalidate.py` | cheap_lever_scan 候选的多窗口复验（2026-10-14）。 |
| `evaluation/cheap_lever_revalidate_920.py` | cheap_lever 胜者的 920 定版信号复验（好机器批次 11，2026-10-17）。 |
| `evaluation/cost_penalty_probe.py` | 成本感知优化探针（P0-2，2026-10-13）。 |
| `evaluation/cpcv_eval.py` | CPCV 多路径评估 —— 挖因子方法选型决策工具。 |
| `evaluation/cpcv_h1_eval.py` | h=1 模型 CPCV 无偏评估 |
| `evaluation/cvar_arm_920.py` | 批次 12⑤：CVaR 约束组合臂 vs Top10% 等权 vs buffer 形态（production 口径）。 |
| `evaluation/deliverable_portfolio.py` | 可交付组合：集中度 × 行业约束（P0 落地，2026-10-10 固化）。 |
| `evaluation/factor_pool_composition.py` | 因子池构成体检：量价占比到底多少、口径在哪、哪些族是真被用上了。 |
| `evaluation/freq_tune.py` | 调仓频率补充精修：horizon ∈ {1,5} × rebalance ∈ {D,W,M} |
| `evaluation/fundamental_blend.py` | E3 基本面慢信号 ⊕ 快信号双层融合（信达式分域建模 + 预测层 λ 加权）。 |
| `evaluation/fuse_placebo.py` | 融合增益的**安慰剂检验**：区分「真信号」与「覆盖率掩码 / 暴露」假象。 |
| `evaluation/galaxy_l1.py` | 银河 0608 复现 L1：三标签 GBDT + 验证集 RankIC 质量门槛（stage2 Phase 2 前置）。 |
| `evaluation/gp_tune_budget.py` | GP 评估预算 × 多 seed 调参实验（三段式，test 段受保护） |
| `evaluation/gtja_repro_eval.py` | 国君研报复现：因子回测评估与研报基准对比 |
| `evaluation/horizon_cross_eval.py` | A1 同目标交叉评估 + 组合层对照（2026-10-19）。 |
| `evaluation/horizon_mix.py` | E1'' horizon 混合消融：把慢 horizon OOS 预测混入 ens_h1h5 的对照实验。 |
| `evaluation/horizon_probe.py` | Horizon 探针：主信号在 h1–h120 上的 IC 衰减与独立样本数（2026-10-19）。 |
| `evaluation/hp_search_ic.py` | 超参 × 预测 IC 小规模对照（**滚动口径**，只比预测 IC，不跑组合回测）。 |
| `evaluation/hp_sensitivity.py` | 模型层超参敏感度探针（P2）。 |
| `evaluation/index_enhance_probe.py` | B1 探针：现有 α 能不能在**中证500 成分约束下**做出可投的指数增强？ |
| `evaluation/intraday_analysis.py` | 日内收益分解 + 时段效应分析 |
| `evaluation/jq_factor_probe.py` | 聚宽因子看板候选探针（2026-10-26）。 |
| `evaluation/market_direction_probe.py` | 大盘方向准入检验：**新信息集**能不能预测下月沪深300？ |
| `evaluation/member_blend12.py` | 多模型×多标签预测合成对照（Task ⑥ / AI29·国金19 语义）。 |
| `evaluation/mf10_t_scan.py` | 多因子10 T 扫描：合成权重回看窗 T 的参数敏感性实验（研报 §2.1 口径）。 |
| `evaluation/model_objective_scan.py` | P2 —— 模型层「目标函数 / 样本权重」扫描。 |
| `evaluation/model_state_probe.py` | 模型自诊断仓位探针（"信号自身状态 → 仓位乘数"）。 |
| `evaluation/multiyear_oos.py` | 多年度 OOS 稳健性实验：模型 × horizon × 调仓频率 × 年份 |
| `evaluation/news_alpha_probe.py` | news 族独立建模 + 可交易口径验真伪 + 组合层融合（2026-10-10，P0 路径）。 |
| `evaluation/pbo_report.py` | 实验产物接 PBO 出报告（TODO §二；华泰 AI19/22 方法论落地）。 |
| `evaluation/pead_alpha_probe.py` | 业绩预告 / 快报事件因子探针（P0 新信息源：PEAD / SUE 口径）。 |
| `evaluation/portfolio_weighting.py` | 组合层权重对照（P1）：等权 vs 波动率倒数加权 vs 风险平价。 |
| `evaluation/prod_baseline_compare.py` | P1b —— 生产基线同口径复验：定版因子 gbdt vs alpha360 gbdt / GRU。 |
| `evaluation/pv_marginal_curve.py` | 量价因子边际增益曲线 —— 阶段 B（DPP 组合口径）。 |
| `evaluation/regime_probe.py` | Regime 条件化探针（Phase 0/2 前置诊断，2026-10-15）。 |
| `evaluation/residual_ic_probe.py` | 残差目标探针：现有 920 因子池里还有没有「主模型漏掉」的正交信息（2026-10-19）。 |
| `evaluation/residual_overlay_probe.py` | A2 残差负向因子 → 组合层剔除/低配 overlay 探针（2026-10-19）。 |
| `evaluation/residual_stack_probe.py` | A2 二阶 GBDT：残差收益目标 walk-forward + 融合组合层（2026-10-19）。 |
| `evaluation/retrain_freq_ic.py` | 重训频率 × 预测 IC 小规模对照（只比预测 IC，**不跑组合回测**）。 |
| `evaluation/sdk_table_audit.py` | B2 一次性审计：SDK 未用表（ETF 份额 / 指数日权重 / 可转债 / 期权 / 国债）。 |
| `evaluation/select_horizon_features.py` | A1 长 horizon 特征清单选择器：h60 IC 质量窗 + DPP 去冗余（2026-10-19）。 |
| `evaluation/seq_predictor_compare.py` | P1 —— 时序预测器对照：序列归纳偏置 vs 截面/树归纳偏置。 |
| `evaluation/stack_blend.py` | E6 stacking：滚动元学习器**学**融合权重（对齐 CFA ensemble / MSIF-OEM 两阶段 |
| `evaluation/tabfm_rolling.py` | 批次 8：表格基础模型全量滚动对照（TabPFN-3 / TabICL V2 vs LightGBM）。 |
| `evaluation/tabpfn_rolling_compare.py` | 批次 8：表格基础模型对照（TabPFN-3.5 / TabICL 2.2 vs LightGBM）滚动 runner |
| `evaluation/text_factor_probe.py` | 论文族 `pap_*` / 情绪族 `txt_*` 的生产口径收尾评估（P0）。 |
| `evaluation/timing_probe.py` | 大盘择时（市场状态 → 仓位乘数）探针。 |
| `evaluation/views_arm_920.py` | 批次 12④：观点注入三臂对照（BL 优化层 vs 预测层 vs 不注入）。 |
| `evaluation/walk_forward.py` | 三段样本外验证（walk-forward）：train 挖 → valid 选 → test 验 |
| `evaluation/walk_forward_model.py` | 模型层滚动训练 —— Predictor 的 walk-forward 编排 |

## factors

| 脚本 | 说明 |
|---|---|
| `factors/build_alpha_factors.py` | 公开经典因子集（Alpha101 / GTJA Alpha191）→ 构建入库 |
| `factors/build_fundamental_factors.py` | 基本面因子构建与入库（价值/质量/成长/规模） |
| `factors/build_intraday_factors.py` | 日内特征 → 日频因子构建与入库 |
| `factors/build_intraday_stat_factors.py` | 分钟统计特征 → 因子库体检入库（降维路线第一层） |
| `factors/build_minute_panel.py` | 分钟面板构建入口 |
| `factors/build_paper_factors.py` | 论文复现因子构建与入库（awesome-systematic-trading 21 式） |
| `factors/build_technical_factors.py` | 技术面迭代/累积类指标 → 日频因子构建与入库 |
| `factors/factor_library.py` | 因子库 CLI |
| `factors/mine_factors.py` | 因子挖掘 CLI |
| `factors/mine_im_combos.py` | 第二层：im_* 统计特征之上的 GP 组合公式挖掘（降维路线主产出） |
| `factors/run_alphapool_ppo.py` | 华泰 AI97《大模型+强化学习因子挖掘》—— AlphaPool + MaskablePPO 训练入口 |
| `factors/run_gflownet_phase1.py` | GFlowNet Phase 1：真实 HS300 对齐研报 |
| `factors/run_llm_mcts.py` | 东吴 LLM-MCTS 因子迭代 Phase 0 —— 同题四引擎对照 runner |
| `factors/run_portfolio_phase2.py` | stage2 Phase 2：zz1000 主实验（信号管线 + PPO 滚动集成 / 银河口径 QP / 基准）。 |
| `factors/run_portfolio_ppo.py` | stage2 Phase 1：hs300 平价验证（PPO vs QP vs 等权基准 vs TopN）。 |
| `factors/synthesize_factors.py` | 多因子合成 CLI |
| `factors/synthesize_library.py` | 从因子库加载 raw 因子 → 四种合成 → 评估对比 + 入库（带血缘） |

## ingest

| 脚本 | 说明 |
|---|---|
| `ingest/check_data_quality.py` | 数据质量检查 CLI |
| `ingest/extend_factor_library.py` | 因子库数据延长 —— 面板刷新到数据源最新交易日 |
| `ingest/fetch_status_batched.py` | 分段拉取全A历史状态表（涨跌停/停牌/ST/除权标记）——增量断点续拉版。 |
| `ingest/update_data.py` | 每日数据更新脚本 |
| `ingest/update_etf.py` | ETF 数据更新脚本 |
| `ingest/update_etf_flow.py` | ETF 资金流 / 净值数据更新脚本 |

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
| `portfolio/etf_alpha_probe.py` | ETF 层 α 可行性探针（S2 持仓映射 α 的前置验证） |
| `portfolio/etf_capacity_probe.py` | ETF 轮动容量表（单只下单 ≤ 60 日均额 5%，2026-10-27） |
| `portfolio/etf_p4_probe.py` | ETF 轮动样本外验证（IS/OOS 切片 + 多起点，2026-10-27） |
| `portfolio/etf_rotation_p2.py` | ETF 轮动优化验证（P2/P3 前置，2026-10-26） |
| `portfolio/etf_rotation_probe.py` | ETF 轮动升级探针（P1） |
| `portfolio/etf_survivorship_probe.py` | ETF 轮动幸存者偏差一次性探针 |
| `portfolio/generate_signals.py` | 每日交易信号生成脚本（P3，2026-08-20） |
| `portfolio/multi_period_backtest.py` | 多期组合执行回测脚本（P3，2026-08-20） |
| `portfolio/run_etf_rotation.py` | ETF 轮动最小闭环回测脚本 |

## reporting

| 脚本 | 说明 |
|---|---|
| `reporting/attribution.py` | 收益归因 CLI |
| `reporting/build_daily_email_body.py` | 从出榜产物生成每日邮件正文（markdown + 配套 HTML）。 |
| `reporting/build_monitor_seed.py` | 构造 monitor_production_ic 的生产口径历史种子面板（一次性，2026-10-17）。 |
| `reporting/factor_explorer_report.py` | 交互式因子检测报告：时间段选择 + 来源筛选 + 指标排序 + 因子详情。 |
| `reporting/factor_library_full_report.py` | 全因子库检验报告：821 个因子的标准检验汇总（自包含 HTML）。 |
| `reporting/generate_report.py` | 一键端到端研究报告生成 |
| `reporting/jq_style_report.py` | 最佳策略收益曲线页（聚宽模拟交易风格）——策略 vs 中证全指口径基准。 |
| `reporting/md_to_email_html.py` | 日报 markdown → 邮件 HTML（内联样式，适配 163 / QQ / Gmail 等客户端）。 |
| `reporting/monitor_performance.py` | 生产化监控调度 CLI —— 因子与模型预测的性能监控。 |
| `reporting/monitor_production_ic.py` | 生产口径每日 IC / 风格暴露监控 CLI（全A · 定版 dingban · 2026-10-17 起）。 |
| `reporting/rolling_grid_report.py` | 全A滚动训练实验报告生成器：收益曲线 + 分年绩效 + 维度对比（自包含 HTML）。 |

## textmining

| 脚本 | 说明 |
|---|---|
| `textmining/_paths.py` | textmining 输出路径约定。 |
| `textmining/analyze_word_importance.py` | 词重要性分析（对齐华泰 AI 51/AI 57 模型可解释性） |
| `textmining/build_daily_factor.py` | 日频因子面板构建 |
| `textmining/build_fadt_samples.py` | FADT 事件样本构建（对齐华泰 AI 57 文本FADT选股） |
| `textmining/build_senti_factors.py` | AI 41 研报情感四因子构建（senti / senti_adj / report_num / report_score） |
| `textmining/build_senti_scores.py` | 研报摘要 FinBERT 情感打分（对齐华泰 AI 41 研报情感因子） |
| `textmining/build_sue_txt_samples.py` | SUE.txt 事件样本构建（对齐华泰 AI 51 文本PEAD） |
| `textmining/derive_pooler_encoding.py` | Pooler（全连接层）编码推导（AI 63 消融：CLS vs 全连接层） |
| `textmining/encode_fadt_bert.py` | FinBERT CLS 编码（对齐华泰 AI 63 文本表示升级） |
| `textmining/evaluate_fadt.py` | FADT 因子评估（对齐华泰 AI 57） |
| `textmining/evaluate_senti.py` | AI 41 研报情感因子族评估 |
| `textmining/evaluate_short_window.py` | 短窗口日频超额评估（T+1~T+5） |
| `textmining/evaluate_sue_txt.py` | SUE.txt 因子评估（对齐华泰 AI 51） |
| `textmining/train_fadt.py` | FADT 训练与因子构建（对齐华泰 AI 57 文本FADT选股） |
| `textmining/train_fadt_bert.py` | BERT 版 FADT 训练（对齐华泰 AI 63 文本表示升级） |
| `textmining/train_sue_txt.py` | SUE.txt 训练与因子构建（对齐华泰 AI 51 文本PEAD） |

## 2026-10-19 归档（三类引用全零，无文档/产物引用）

| 原路径 | 归档到 | 说明 |
|---|---|---|
| `scripts/common/e2e_common.py` | `scripts/archive/e2e_common.py` | HS300 时代 e2e 编排（数据/中性化/特征漏斗） |
| `scripts/evaluation/ai97_pbo_report.py` | `scripts/archive/ai97_pbo_report.py` | AI97 PBO 报告（一次性） |
| `scripts/evaluation/pv_marginal_quality.py` | `scripts/archive/pv_marginal_quality.py` | PV 边际质量探针（一次性） |
| `scripts/evaluation/tabfm_head_portfolio.py` | `scripts/archive/tabfm_head_portfolio.py` | TabPFN/TabICL 头部组合验证 |
| `scripts/evaluation/tabfm_head_study.py` | `scripts/archive/tabfm_head_study.py` | TabPFN/TabICL 头部诊断 |
| `scripts/factors/phase2_ppo_diag.py` | `scripts/archive/phase2_ppo_diag.py` | Phase2 PPO 诊断 |
| `scripts/factors/run_gflownet_phase0.py` | `scripts/archive/run_gflownet_phase0.py` | GFlowNet phase0 入口（已被 phase1 取代） |
| `scripts/reporting/factor_correlation.py` | `scripts/archive/factor_correlation.py` | 因子相关性报告 |
| `scripts/reporting/risk_decomposition_report.py` | `scripts/archive/risk_decomposition_report.py` | 风险分解报告 |
| `scripts/evaluation/cheap_lever_probe.py` | `scripts/archive/cheap_lever_probe.py` | 廉价杠杆探针 |
| `scripts/evaluation/impact_cost_probe.py` | `scripts/archive/impact_cost_probe.py` | 冲击成本探针 |
| `scripts/evaluation/multihorizon_fusion_probe.py` | `scripts/archive/multihorizon_fusion_probe.py` | 多 horizon 融合探针 |
| `scripts/builders/backfill_delisted_backward.py` | `scripts/archive/backfill_delisted_backward.py` | 退市股后复权回补 |
| `scripts/builders/backfill_delisted_equity.py` | `scripts/archive/backfill_delisted_equity.py` | 退市股股本结构回补 |
| `scripts/builders/backfill_delisted_kline.py` | `scripts/archive/backfill_delisted_kline.py` | 退市股 K 线回补 |
| `scripts/builders/build_alla_e4_interaction_factors.py` | `scripts/archive/build_alla_e4_interaction_factors.py` | E4 交互因子构建 |
| `scripts/builders/refetch_status_all_a.py` | `scripts/archive/refetch_status_all_a.py` | 全A状态重取 |
| `scripts/factors/fetch_textmining.py` | `scripts/archive/fetch_textmining.py` | 文本挖掘数据拉取 |
| `scripts/factors/gflownet_library_ingest.py` | `scripts/archive/gflownet_library_ingest.py` | GFlowNet 因子入库 |
| `scripts/data_tools/cleanup_data.py` | `scripts/archive/cleanup_data.py` | 数据清理工具 |
| `scripts/data_tools/split_zz1000.py` | `scripts/archive/split_zz1000.py` | ZZ1000 拆分工具 |
| `scripts/textmining/build_enhanced_top25.py` | `scripts/archive/build_enhanced_top25.py` | 增强 Top25 构建 |
| `scripts/textmining/build_rawtone_factor.py` | `scripts/archive/build_rawtone_factor.py` | 原始情绪因子构建 |
| `scripts/textmining/encode_fadt_seg.py` | `scripts/archive/encode_fadt_seg.py` | FADT 分段编码 |
| `scripts/textmining/evaluate_fadt_ablation.py` | `scripts/archive/evaluate_fadt_ablation.py` | FADT 消融评估 |
| `scripts/textmining/fetch_all_a_backfill.py` | `scripts/archive/fetch_all_a_backfill.py` | 全A文本数据回补 |
| `scripts/textmining/register_text_factors.py` | `scripts/archive/register_text_factors.py` | 文本因子入库 |
