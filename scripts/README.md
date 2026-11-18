# scripts 入口索引

> 自动生成（284 个活跃入口；archive/oneoff 不计）。改脚本 docstring 后运行 `python -m scripts.data_tools.gen_scripts_index` 更新；`--check` 校验是否过期。

约定：`python -m scripts.<目录>.<文件名>` 运行；`scripts/archive/` 已于 2026-11-18 整体下线（活代码已上移，其余删除）；实验结论查 `reports/README.md` §一「结论速查」，产物落盘规则见其 §1.6。

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
| `builders/run_evt_margin_build.py` | 事件/两融/资金流数据回补 + 因子构建编排器（串行，防止 SDK 单连接竞争）。 |

## common

| 脚本 | 说明 |
|---|---|
| `common/alla_common.py` | 全A滚动实验的**共享底座**（2026-10-19 架构清理）。 |
| `common/atomic_io.py` | 原子写面板 + 抗瞬时文件占用（Windows 原生锁）。 |
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
| `data_tools/dump_pdf_text.py` | 把研报 PDF 抽成带页码标记的纯文本，便于按关键词定位口径参数。 |
| `data_tools/fix_meta.py` | 修复 e:/data/parquet/_meta.json：对齐实际日期 + 补文本表 + 加 pool 口径。 |
| `data_tools/gen_scripts_index.py` | 生成 ``scripts/README.md`` 入口索引（2026-10-19 审计 ③）。 |
| `data_tools/memory_lint.py` | 记忆层治理校验（memory_lint，2026-11-03）。 |
| `data_tools/new_experiment.py` | 新建实验目录：一条命令完成「建目录 + 复制模板 + 登记判定」。 |
| `data_tools/recycle.py` | 走 Windows 回收站删除（**不永久删除**），供维护类脚本复用。 |

## evaluation

| 脚本 | 说明 |
|---|---|
| `evaluation/backtest_two_periods.py` | sig_composite_pca 两时段样本外回测：2025 全年 vs 2026 上半年。 |
| `evaluation/blend_verdict_920.py` | 定版信号慢腿权重 λ 的正式裁决：0.5 要不要动。 |
| `evaluation/blend_weight_920.py` | 定版信号「快慢融合权重」λ 在**正确底座（h1020）**上的补齐扫描。 |
| `evaluation/buffer_arm_920.py` | 批次 1.4：定版信号（h1020）的 buffer(20/30) 臂 vs naive top20。 |
| `evaluation/buffer_exit_scan.py` | 缓冲带（``exit_mult``）× K 二维扫描：成本能省多少，代价是多少 |
| `evaluation/caliber.py` | 统一评估口径 —— 全项目**唯一真源**（2026-10-13 建立）。 |
| `evaluation/capacity_table.py` | 组合层容量表：集中度 × AUM（统一口径，2026-10-13）。 |
| `evaluation/cheap_lever_revalidate.py` | cheap_lever_scan 候选的多窗口复验（2026-10-14）。 |
| `evaluation/cheap_lever_revalidate_920.py` | cheap_lever 胜者的 920 定版信号复验（好机器批次 11，2026-10-17）。 |
| `evaluation/chip_behavior_probe.py` | 筹码 / 处置效应 / 凸显理论：现有算子造廉价近似，先证残差有没有空间（2026-10-27）。 |
| `evaluation/cost_penalty_probe.py` | 成本感知优化探针（P0-2，2026-10-13）。 |
| `evaluation/cpcv_eval.py` | CPCV 多路径评估 —— 挖因子方法选型决策工具。 |
| `evaluation/cvar_arm_920.py` | 批次 12⑤：CVaR 约束组合臂 vs Top10% 等权 vs buffer 形态（production 口径）。 |
| `evaluation/deliverable_portfolio.py` | 可交付组合：集中度 × 行业约束（P0 落地，2026-10-10 固化）。 |
| `evaluation/dup_caliber_probe.py` | 判重口径探针：同一批因子在三种相关口径下到底有多像。 |
| `evaluation/ev_rd_backtest.py` | EV / 研发 四因子的组合层验证：IC 好不等于赚钱。 |
| `evaluation/ev_rd_validate.py` | EV 分母 / 研发加回 四个候选因子的三段式 + 增量检验。 |
| `evaluation/event_chain_probe.py` | 事件链与事件前状态：增持的「来历」是否决定收益（2026-11-16） |
| `evaluation/event_driven_probe.py` | 事件驱动组合的独立回测（2026-11-15） |
| `evaluation/event_industry_probe.py` | L4 事件聚合：行业内增持「密度」本身是不是信号（2026-11-16） |
| `evaluation/event_opt_probe.py` | `inc_pure` 的优化实测：金额分层 / 延迟建仓 / 负面排除 / 单日限额 / 与主线叠加。 |
| `evaluation/event_reach_probe.py` | 事件族「决策触及度 + 上帝视角天花板」一次性诊断（2026-11-15） |
| `evaluation/exec_fill_probe.py` | 事件驱动策略的执行可行性检验（2026-11-15） |
| `evaluation/exec_phase_probe.py` | 成交日相位：能不能把成交提前到「当月第 1 个交易日开盘」。 |
| `evaluation/factor_half_life_probe.py` | 因子信息半衰期探针：有没有一批因子在「周频尺度」上显著更强？ |
| `evaluation/factor_half_life_verdict.py` | 因子信息半衰期的**判定**（读 `factor_half_life_probe` 的 CSV，不重算 IC）。 |
| `evaluation/factor_library_diversity.py` | 因子库有效独立维度（n_eff）度量：判断"挖出很多因子"里有多少是新的。 |
| `evaluation/factor_pool_composition.py` | 因子池构成体检：量价占比到底多少、口径在哪、哪些族是真被用上了。 |
| `evaluation/focus_holding_probe.py` | 持仓数与持仓权重探针 —— `yqcore-focus-k15` 能压到几只、权重该不该动。 |
| `evaluation/focus_micro_probe.py` | 小资金（10 万）整手实跑选型 —— K × 缓冲带 × 行业去重 × 仓位。 |
| `evaluation/focus_risk_budget.py` | focus 内部风险预算：三种降回撤手段（小资金口径优先） |
| `evaluation/focus_sector_probe.py` | focus + 行业去重（每申万一级 ≤N 只）的生产口径完整评估。 |
| `evaluation/focus_upgrade_probe.py` | 集中档信号选型探针 —— horizon 集 × 稠密 K 网格 × IS/OOS 切分。 |
| `evaluation/fundamental_blend.py` | E3 基本面慢信号 ⊕ 快信号双层融合（信达式分域建模 + 预测层 λ 加权）。 |
| `evaluation/fuse_placebo.py` | 融合增益的**安慰剂检验**：区分「真信号」与「覆盖率掩码 / 暴露」假象。 |
| `evaluation/galaxy_l1.py` | 银河 0608 复现 L1：三标签 GBDT + 验证集 RankIC 质量门槛（stage2 Phase 2 前置）。 |
| `evaluation/gp_tune_budget.py` | GP 评估预算 × 多 seed 调参实验（三段式，test 段受保护） |
| `evaluation/gtja_repro_eval.py` | 国君研报复现：因子回测评估与研报基准对比 |
| `evaluation/horizon_cross_eval.py` | A1 同目标交叉评估 + 组合层对照（2026-10-19）。 |
| `evaluation/horizon_h40_probe.py` | h=40 新周期腿的对照探针（2026-11-15）。 |
| `evaluation/horizon_mix.py` | E1'' horizon 混合消融：把慢 horizon OOS 预测混入 ens_h1h5 的对照实验。 |
| `evaluation/horizon_probe.py` | Horizon 探针：主信号在 h1–h120 上的 IC 衰减与独立样本数（2026-10-19）。 |
| `evaluation/horizon_verdict_920.py` | 定版 horizon 结构的正式裁决：h20-only 该不该取代 h1020 坐上快信号腿。 |
| `evaluation/hp_search_ic.py` | 超参 × 预测 IC 小规模对照（**滚动口径**，只比预测 IC，不跑组合回测）。 |
| `evaluation/hp_sensitivity.py` | 模型层超参敏感度探针（P2）。 |
| `evaluation/ic_decay_profile.py` | 因子 IC 衰减剖面：有效期到底有多长（纯信号口径，**不含成本、不含组合**） |
| `evaluation/index_enhance_probe.py` | B1 探针：现有 α 能不能在**中证500 成分约束下**做出可投的指数增强？ |
| `evaluation/intraday_analysis.py` | 日内收益分解 + 时段效应分析 |
| `evaluation/intraday_factor_probe.py` | 日内微观结构因子：增量信息体检（2026-11-15）。 |
| `evaluation/jq_factor_probe.py` | 聚宽因子看板候选探针（2026-10-26）。 |
| `evaluation/market_direction_probe.py` | 大盘方向准入检验：**新信息集**能不能预测下月沪深300？ |
| `evaluation/member_blend12.py` | 多模型×多标签预测合成对照（Task ⑥ / AI29·国金19 语义）。 |
| `evaluation/minute_monthly_probe.py` | 分钟频因子的月频可兑现性（2026-11-15） |
| `evaluation/minute_weekly_probe.py` | 分钟频因子的周频可兑现性（2026-11-15） |
| `evaluation/model_objective_scan.py` | P2 —— 模型层「目标函数 / 样本权重」扫描。 |
| `evaluation/model_state_probe.py` | 模型自诊断仓位探针（"信号自身状态 → 仓位乘数"）。 |
| `evaluation/network_centrality_probe.py` | 跨股票网络中心度：P1 探针（算子空间造不出的那一类，2026-10-27）。 |
| `evaluation/news_alpha_probe.py` | news 族独立建模 + 可交易口径验真伪 + 组合层融合（2026-10-10，P0 路径）。 |
| `evaluation/p0_daily_overlay.py` | P0：日频信号能不能给 `yqcore-focus` 做**风控叠加**（而非选股替代） |
| `evaluation/p0_timing_precheck.py` | 前置检验：分钟频结构能不能预测 **focus 未来的相对表现**（不是涨跌） |
| `evaluation/pbo_report.py` | 实验产物接 PBO 出报告（TODO_ARCHIVE 册Ⅰ §二；华泰 AI19/22 方法论落地）。 |
| `evaluation/pead_alpha_probe.py` | 业绩预告 / 快报事件因子探针（P0 新信息源：PEAD / SUE 口径）。 |
| `evaluation/pit_freq_sweep.py` | 调仓频率 × 持有期 **全扫描**（PIT 动态池） |
| `evaluation/pit_intraday_probe.py` | PIT 动态池（421 只）上重算 im_* 因子并验证 1+2 |
| `evaluation/pit_weekly_probe.py` | `im_*` 短周期因子的**周频**组合检验（PIT 动态池） |
| `evaluation/pit_xs_nl_probe.py` | 挖两个缺维度：横截面相对量（xs）+ 时序非线性结构（nl） |
| `evaluation/pool_ic_diagnose.py` | 分池 IC 诊断 —— 解释「为什么信号在指数池上衰减」。 |
| `evaluation/pool_maxcorr_probe.py` | 新因子 vs 生产池：最大截面相关扫描（判「是不是老东西换皮」，2026-10-27）。 |
| `evaluation/pool_scope_probe.py` | 股票池范围探针 —— 同一选股信号在 hs300 / zz500 / zz1000 / zz2000 上的业绩。 |
| `evaluation/portfolio_weighting.py` | 组合层权重对照（P1）：等权 vs 波动率倒数加权 vs 风险平价。 |
| `evaluation/prod_baseline_compare.py` | P1b —— 生产基线同口径复验：定版因子 gbdt vs alpha360 gbdt / GRU。 |
| `evaluation/pv_marginal_curve.py` | 量价因子边际增益曲线 —— 阶段 B（DPP 组合口径）。 |
| `evaluation/rating_probe.py` | 卖方评级族因子探针（预注册判据，2026-11-15） |
| `evaluation/rating_reach_probe.py` | 卖方评级的「决策触及度」与天花板（诊断，不是新因子，2026-11-15） |
| `evaluation/rd_overlay_probe.py` | `rd_to_cap`（研发强度）的生产口径叠加验证 —— P0。 |
| `evaluation/rebalance_granularity_probe.py` | 调仓粒度探针：solo 档的信息半衰期 + 换手预算（P0 / P1）。 |
| `evaluation/rebuild_impact_920.py` | 重建前后影响对比：把「重建会改多少」变成可判定的数（2026-11-07）。 |
| `evaluation/regime_probe.py` | Regime 条件化探针（Phase 0/2 前置诊断，2026-10-15）。 |
| `evaluation/residual_ic_probe.py` | 残差目标探针：现有 920 因子池里还有没有「主模型漏掉」的正交信息（2026-10-19）。 |
| `evaluation/residual_overlay_probe.py` | A2 残差负向因子 → 组合层剔除/低配 overlay 探针（2026-10-19）。 |
| `evaluation/residual_stack_probe.py` | A2 二阶 GBDT：残差收益目标 walk-forward + 融合组合层（2026-10-19）。 |
| `evaluation/retrain_freq_ic.py` | 重训频率 × 预测 IC 小规模对照（只比预测 IC，**不跑组合回测**）。 |
| `evaluation/rtl_phase_tiers.py` | 三档 21 相位（RTL）补齐 —— focus / breadth 的相位风险有多大。 |
| `evaluation/rtl_placebo_phase.py` | 行业去重安慰剂：跨相位重跑（原安慰剂只在 phase=0，无法排除相位选择偏差）。 |
| `evaluation/rtl_probe.py` | 调仓时点运气（Rebalance Timing Luck, RTL）—— 相位极差的文献口径复现。 |
| `evaluation/rtl_variant_extra.py` | `rtl_variant_probe` 的补充：换年度轴复核行业去重，并量化分腿能降多少 RTL。 |
| `evaluation/rtl_variant_probe.py` | 行业去重的增益在多相位下是否还在 —— 以及哪个变体 RTL 最小。 |
| `evaluation/sdk_table_audit.py` | B2 一次性审计：SDK 未用表（ETF 份额 / 指数日权重 / 可转债 / 期权 / 国债）。 |
| `evaluation/sector_constraint_tier.py` | 行业约束（每申万一级 ≤N 只）放在**哪个档位**才值钱 —— 七臂 × 相位。 |
| `evaluation/sector_dedup_probe.py` | 行业去重：逻辑检验 + 不去重的效果对比（五臂权衡）。 |
| `evaluation/sector_risk_probe.py` | 行业集中度约束：从「标签去重」升级为「风险同源去重」（2026-11-11）。 |
| `evaluation/select_horizon_features.py` | A1 长 horizon 特征清单选择器：h60 IC 质量窗 + DPP 去冗余（2026-10-19）。 |
| `evaluation/seq_predictor_compare.py` | P1 —— 时序预测器对照：序列归纳偏置 vs 截面/树归纳偏置。 |
| `evaluation/solo_freq_probe.py` | 周频可行性 —— 小资金 solo 方案换调仓频率后，参数量是否需要重估。 |
| `evaluation/solo_live.py` | yqcore-focus-k8 生产持仓生成器（2026-11-10 建立）。 |
| `evaluation/solo_signal_probe.py` | solo 档的**周期组合**再检验：在 solo 真实执行层下重做 horizon 消融。 |
| `evaluation/sop_timing_probe.py` | SOP 收益分解择时（P1）：文献唯一通过多重检验的方法，本地化测试。 |
| `evaluation/stack_blend.py` | E6 stacking：滚动元学习器**学**融合权重（对齐 CFA ensemble / MSIF-OEM 两阶段 |
| `evaluation/tabfm_rolling.py` | 批次 8：表格基础模型全量滚动对照（TabPFN-3 / TabICL V2 vs LightGBM）。 |
| `evaluation/tabpfn_rolling_compare.py` | 批次 8：表格基础模型对照（TabPFN-3.5 / TabICL 2.2 vs LightGBM）滚动 runner |
| `evaluation/text_factor_probe.py` | 论文族 `pap_*` / 情绪族 `txt_*` 的生产口径收尾评估（P0）。 |
| `evaluation/tier_sector_concentration.py` | 三档形态的**行业集中度 vs 个股集中度**逐调仓日截面诊断。 |
| `evaluation/timing_conditioned_probe.py` | 条件化月频择时（P0）：复现并本地化「newsy 月」结构。 |
| `evaluation/timing_daily_train_probe.py` | 日频训练 → 月频策略上的择时乘数（第三个格子）。 |
| `evaluation/timing_probe.py` | 大盘择时（市场状态 → 仓位乘数）探针。 |
| `evaluation/top10_sharpe_eval.py` | 十因子增量检验（复用 jq_factor_probe 的评估函数） |
| `evaluation/views_arm_920.py` | 批次 12④：观点注入三臂对照（BL 优化层 vs 预测层 vs 不注入）。 |
| `evaluation/vol_target_920.py` | 定版信号上的「目标波动」仓位 overlay 复验（风控层，production 口径）。 |
| `evaluation/walk_forward.py` | 三段样本外验证（walk-forward）：train 挖 → valid 选 → test 验 |
| `evaluation/walk_forward_model.py` | 模型层滚动训练 —— Predictor 的 walk-forward 编排 |
| `evaluation/yearly_breakdown.py` | 分年拆解 —— 策略 / 基准 / IC 的逐年表现，用于解释某一年为什么异常。 |
| `evaluation/yq_core_report.py` | `yqcore` 族（YQ 核心）产品线报数脚本 —— 三档的统一口径对照 + 净值曲线。 |

## factors

| 脚本 | 说明 |
|---|---|
| `factors/audit_panel_window.py` | 按**定版回测窗口**核对因子库每个成员的面板覆盖（只读）。 |
| `factors/backfill_library.py` | 因子库回填：体检（dry-run）+ 并行补 canonical 评估。 |
| `factors/backfill_library_metrics.py` | 因子库指标批量回填：canonical 分层回测 + 显著性 + 路径/公式/族标签（维护工具）。 |
| `factors/backfill_metadata.py` | registry 元数据回填：``formula`` 与 ``family``（只补空缺，不动任何统计列）。 |
| `factors/build_alpha_factors.py` | 公开经典因子集（Alpha101 / GTJA Alpha191）→ 构建入库 |
| `factors/build_fundamental_factors.py` | 基本面因子构建与入库（价值/质量/成长/规模） |
| `factors/build_intraday_factors.py` | 日内特征 → 日频因子构建与入库 |
| `factors/build_intraday_stat_factors.py` | 分钟统计特征 → 因子库体检入库（降维路线第一层） |
| `factors/build_minute_panel.py` | 分钟面板构建入口 |
| `factors/build_paper_factors.py` | 论文复现因子构建与入库（awesome-systematic-trading 21 式） |
| `factors/build_technical_factors.py` | 技术面迭代/累积类指标 → 日频因子构建与入库 |
| `factors/build_top10_factors.py` | 「中国因子十强」复现与统一评估（JKP 口径榜单，2026-11-02） |
| `factors/dup_audit.py` | 全库冗余审计：一次性算清 965 个因子的两两截面相关，并把结论写回 registry。 |
| `factors/dup_criteria.py` | 因子重复性判据——**全仓唯一真源**。 |
| `factors/dup_precheck.py` | **登记前**冗余预检：这批因子该不该进库，先看一眼它最像谁。 |
| `factors/factor_board_data.py` | 因子看板数据层：为因子库里每个因子算「分位组合净值 + IC 序列 + 行业 IC」。 |
| `factors/factor_library.py` | 因子库 CLI |
| `factors/formula_extract.py` | 从 factor/ 源码反解析因子的**可读公式**，供 registry ``formula`` 列与看板展示。 |
| `factors/library_dedup_apply.py` | 库内去重的**执行层**：把决策表的 ``safe_drop`` 从 registry 退登记。 |
| `factors/library_dedup_plan.py` | 因子库去重决策表：把「相关 → 质量 → 生产依赖」三条判据并成一张可拍板的清单。 |
| `factors/mine_factors.py` | 因子挖掘 CLI |
| `factors/panel_leak_audit.py` | 全库面板前视体检（只读）：把「哪几块面板还在用未来数据」从猜测变成清单。 |
| `factors/panel_leak_audit_diff.py` | 把两次体检的 CSV 放到**同一套判据**下重判，输出可比的前后对照。 |
| `factors/promote_maturity.py` | 成熟度晋升：把 :data:`research.factor_library.MATURITY_GATES` 落到全库。 |
| `factors/prune_registry_backups.py` | registry 备份轮转：只保留「有用的快照」，其余走回收站。 |
| `factors/run_alphapool_ppo.py` | 华泰 AI97《大模型+强化学习因子挖掘》—— AlphaPool + MaskablePPO 训练入口 |
| `factors/run_gflownet_phase1.py` | GFlowNet Phase 1：真实 HS300 对齐研报 |
| `factors/run_llm_mcts.py` | 东吴 LLM-MCTS 因子迭代 Phase 0 —— 同题四引擎对照 runner |
| `factors/run_portfolio_phase2.py` | stage2 Phase 2：zz1000 主实验（信号管线 + PPO 滚动集成 / 银河口径 QP / 基准）。 |
| `factors/run_portfolio_ppo.py` | stage2 Phase 1：hs300 平价验证（PPO vs QP vs 等权基准 vs TopN）。 |
| `factors/sync_factor_artifacts.py` | 入库后补齐「三件套」：中性化面板 + ic_h 列 + registry 统计。 |
| `factors/synthesize_factors.py` | 多因子合成 CLI |
| `factors/synthesize_library.py` | 从因子库加载 raw 因子 → 四种合成 → 评估对比 + 入库（带血缘） |

## ingest

| 脚本 | 说明 |
|---|---|
| `ingest/check_data_quality.py` | 数据质量检查 CLI |
| `ingest/extend_factor_library.py` | 因子库数据延长 —— 面板刷新到数据源最新交易日 |
| `ingest/fetch_minute_gap.py` | 补齐 HS300 动态池（PIT 成分）缺失的分钟数据 |
| `ingest/fetch_status_batched.py` | 分段拉取全A历史状态表（涨跌停/停牌/ST/除权标记）——增量断点续拉版。 |
| `ingest/update_data.py` | 每日数据更新脚本 |
| `ingest/update_etf.py` | ETF 数据更新脚本 |
| `ingest/update_etf_flow.py` | ETF 资金流 / 净值数据更新脚本 |
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
| `model_train/diag_adaptive_window.py` | 「不固定窗口」到底能不能成立：三个前置前提的实测。 |
| `model_train/diag_capacity.py` | 容量与真实成本诊断：把「10bp」这个假设换成按 AUM 估出来的成本。 |
| `model_train/diag_daily_rebal.py` | 日频调仓诊断：既然「alpha 只活一天」，那把调仓频率提到每天能不能收到钱？ |
| `model_train/diag_exec_timing.py` | 成交时点诊断：把行业因子的超额拆到「隔夜 / 日内 / 整周」三段上。 |
| `model_train/diag_long_window.py` | 长窗腿：把「固定窗口选错了长度」这件事量化，并分辨 IC 增益与组合层拟合。 |
| `model_train/diag_raft_knn.py` | RAFT 思路的轻量原型：行业层 k-NN 检索预测（不做 MLP，只看「检索本身有没有信息」）。 |
| `model_train/diag_raft_weight.py` | RAFT 式样本加权在全A 主模型上的裁决探针。 |
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
| `portfolio/etf_breaker_h2h.py` | ETF 轮动 · 熔断触发后「降档五衡」能不能救：条件前瞻头对头 |
| `portfolio/etf_breaker_probe.py` | ETF 轮动 · 熔断降仓实验：「回撤触 −15% 就减仓」到底划不划算 |
| `portfolio/etf_champ_maxw_scan.py` | ETF 磐石（``champ``）· 单只权重上限 ``max_weight`` 扫描（相位平均口径）。 |
| `portfolio/etf_champ_phase_matrix.py` | ETF 磐石（``champ``）· 「月内调仓相位 × 权重规则」交叉矩阵（相位平均口径）。 |
| `portfolio/etf_champ_phase_robust.py` | 磐石（``champ``）· 相位交叉矩阵的**稳健性诊断**（配套 ``etf_champ_phase_matrix``）。 |
| `portfolio/etf_champ_signals.py` | ETF 轮动生产信号板（三锋 / 五衡 / 磐石） |
| `portfolio/etf_champ_tvol_scan.py` | ETF 磐石（``champ``）· ``target_vol`` 扫描 + 相位独立信息量（n_eff）实测。 |
| `portfolio/etf_contest5_buffer_oos.py` | buffer 档位选择验证（网格 + 样本外 + 权重稳健性） |
| `portfolio/etf_contest5_eval.py` | ETF 轮动 · 档位评测（80 个滚动 2 月窗口，口径统一 T+1 open） |
| `portfolio/etf_contest_spec.py` | 正式赛定版参数（单一真源，2026-10-27 锁定） |
| `portfolio/etf_deepdip_probe.py` | ETF 轮动 · 「深跌建仓」收益腿的崩盘区制存活检验（2026-10-30） |
| `portfolio/etf_entry_diag.py` | ETF 轮动 · 两个「入场前提」诊断：HRP 集中度是否异常、回测起点是否合理。 |
| `portfolio/etf_hrp_concentration_diag.py` | 三锋 HRP「单押 80%」溯源诊断（2026-10-31） |
| `portfolio/etf_index_proxy.py` | ETF → 跟踪指数代理：为长样本（2013–2019）提供价格序列（README §4.6 前置）。 |
| `portfolio/etf_intraday_clock_probe.py` | ETF 盘中决策时点探针（14:50 / 14:55 / 收盘） |
| `portfolio/etf_leak_probe.py` | ETF 轮动 · 权重层 PIT（信息渗漏）验证与回归探针。 |
| `portfolio/etf_long_sample.py` | 指数代际长样本（2013–2019）：HRP「集中放大器」的区制存活检验（README §4.6）。 |
| `portfolio/etf_outside_source_probe.py` | 池外信息源廉价近似探针（2026-10-30） |
| `portfolio/etf_pbo_audit.py` | ETF 轮动 · 策略级过拟合审计（CSCV/PBO + DSR，2026-10-30） |
| `portfolio/etf_phase_power.py` | ETF 轮动 · 三锋「相位检验力」分析：多相位重测到底值多少样本（2026-10-31）。 |
| `portfolio/etf_phase_rule_matrix.py` | ETF 轮动 · 「权重规则 × 调仓相位」交叉矩阵（**相位平均口径**）。 |
| `portfolio/etf_pool_monitor.py` | 池内环境监测：策略「机制条件」是否还成立的唯一事前指标（2026-10-30）。 |
| `portfolio/etf_rotation_panels.py` | ETF 轮动扩展面板（生产组件） |
| `portfolio/etf_rotation_probe.py` | ETF 轮动回测引擎（数据装载 / 面板 / 回测 / 出场调度） |
| `portfolio/etf_signal_latency.py` | ETF 轮动信号链端到端耗时探针 |
| `portfolio/etf_signal_preload_probe.py` | ETF 信号链「预加载」路径耗时探针 |
| `portfolio/etf_weight_probe.py` | 权重层质疑核查（2026-10-29） |
| `portfolio/etf_weight_source.py` | 收益来源分解：选标、集中、池子各自贡献多少（README §4.5）。 |
| `portfolio/generate_signals.py` | 每日交易信号生成脚本（P3，2026-08-20） |
| `portfolio/industry_neff_probe.py` | 31 个申万一级的「有效独立方向数」诊断（P0-1） |
| `portfolio/industry_rotation.py` | 行业池轮动线（独立实验线，不与 ETF 三锋混跑）。 |
| `portfolio/multi_period_backtest.py` | 多期组合执行回测脚本（P3，2026-08-20） |
| `portfolio/neutralize_group_ab.py` | 分组哑变量 A/B：申万行业 vs 相关性簇 vs 随机分组（P1-1） |
| `portfolio/report_replica.py` | 研报复现线（国泰海通 2026-10-29《七维共振》）· 只做可行的部分。 |
| `portfolio/run_etf_rotation.py` | ETF 轮动最小闭环回测脚本 |

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
| `reporting/model_performance.py` | 主模型回测表现页（净值 / 回撤 / 分年 / 月度热图，单文件 HTML）。 |
| `reporting/monitor_factor_health.py` | 定版因子库健康度巡检（因子 + 模型预测）。 |
| `reporting/monitor_production_ic.py` | 生产口径每日 IC / 风格暴露监控 CLI（全A · 定版 dingban · 2026-10-17 起）。 |
| `reporting/picks_board.py` | 本期出榜解释页（为什么买这些）—— 接在模拟交易页之后。 |
| `reporting/rolling_grid_report.py` | 全A滚动训练实验报告生成器：收益曲线 + 分年绩效 + 维度对比（自包含 HTML）。 |
| `reporting/yqcore_dashboard.py` | `yqcore` 定版策略前端页面（自包含 HTML，零后端）——**历史回测**与**模拟交易**分两页。 |

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
