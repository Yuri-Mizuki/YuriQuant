# reports/ — 产出物索引与结论速查

> 本目录整体被 gitignore（`.gitignore:23`），**任何删除都无法经 git 恢复**。
> 研究结论的唯一真源是根目录 [`TODO.md`](../TODO.md)（收口结论归档于 [`TODO_ARCHIVE.md`](../TODO_ARCHIVE.md) 三册）。
> 本文件只做索引：每个目录一行——是什么 / 一句话结论 / 判定。口径、证据、复现命令以各目录 `README.md` 为准。
> 各子目录如另有 README（如 `textmining/README.md`），以子 README 为准。

- 判定口径：`采纳`（进生产/定版）/ `判负`（实测无增量或被证伪）/ `归档`（有结论但无后续动作）/ `工具保留`（脚本或表仍有读取方）/ `事故记录`。
- **成本口径分水岭**：2026-11-11 起 27bp 往返时变成本；36bp / 10bp 旧口径数字一律不可相减、不可引用。
- **复现命令多为失效存档**：历史实验脚本随结论固化陆续清理，报告里的 `python -m scripts.*` 命令多数已跑不了，仅保留「当时怎么算」的记录。可运行的入口只以 [`scripts/README.md`](../scripts/README.md) 为准。

## 一、目录地图（按功能分区）

### 1.1 生产运行（每日/每日计划任务在用，勿动勿改名）

| 目录 | 内容 | 产出脚本 |
|---|---|---|
| `alla_daily_defv/` | 全A每日选股排名、候选、解释（**现行**，Windows 计划任务 `YuriQuant AllaDailyRank --preset dingban`，2026-10-08 起） | `scripts/pipelines/alla_daily_rank.py` |
| `alla_daily_solo/` | `yqcore-solo` 的**手工出榜**产物（`--preset legacy --horizons 1,5,10,20 --horizon-weights 2,1,1,2 --out-tag _solo`）；不在计划任务里，按需跑 | `scripts/pipelines/alla_daily_rank.py` + `scripts/evaluation/solo_live.py` |
| `alla_daily/` | ⚠️ legacy 口径产物（旧 h1+h5 ortho），停于 2026-09-22；仅历史追溯 | `scripts/pipelines/alla_daily_rank.py --preset legacy` |
| `alla_rolling/` | 全A滚动实验**主数据枢纽**（zscore 臂）：`_base/` 喂因子构建器、`pred/` 喂归因、`selection/` 喂生产日排名，8+ 组件读取 | `scripts/pipelines/rolling_grid_alla.py`（默认臂）+ `rolling_grid_report.py` + `jq_style_report.py` |
| `alla_rolling_ortho/` | 正交化口径臂——**全A管线默认**（TODO_ARCHIVE 册Ⅰ 附录：全正交 +1.6pp/年 > 不正交），ens_h1h5 集成年化 +13.3% | `scripts/pipelines/rolling_grid_alla.py --preproc ortho` |
| `alla_attribution/` | 主策略 α/β 分解 + Brinson 归因（结论录于 TODO_ARCHIVE 册Ⅰ §一） | `scripts/pipelines/alla_excess_attribution.py` |
| `model_portfolio/` | 固化组合交付物（`pipeline=alla_ortho`；legacy 对照，非现行生产） | 脚本已删，2026-11-19（产物保留） |
| `monitoring/` | 监控台账 + 报告。两条现役线：① `factor_health_report.html` + `snapshots.csv` + `alerts.csv` = 定版库因子/模型健康度巡检（2026-11-18 重建，绑 `all_a_2018_2026`）；② `production_ic_daily.csv` = 生产口径每日 IC 台账。HS300 旧链 2026-09-21 退休，证据 `legacy_monitor_*_20260921.log` | `scripts/reporting/monitor_factor_health.py` + `scripts/reporting/monitor_production_ic.py` |
| `models/` | 模型注册台账（旧监控链读取；现役链不依赖） | 脚本已删，2026-11-19（台账只读） |
| `textmining/` | 文本因子管线（**活跃开发中**，结构见其自带 README） | `scripts/textmining/*` |
| `alla_rolling_ortho920fundind/` | 定版 `dingban_fundind` 的滚动产物；`selection/` 供每日出榜（`--preset dingban`）读取，**勿动** | `scripts/pipelines/rolling_grid_alla.py --preproc ortho_fundind` |

### 结论速查：每个目录一行（判定见末列）

| 目录 | 结论一句话 | 判定 |
|---|---|---|
| `altf_neu_backfill` | alt 20 因子回填 `panels_neu`，门槛过但入选集逐位不变（死重） | 判负 |
| `altf_p1`, `altf_p1/model_ablation` | alt 三族 20 因子入库；等权合成稀释，席位 0/36；`model_ablation` 旁路 ΔIC≈0 | 判负 |
| `alphasage_probe` | AlphaSAGE 编码器 GRU→RGCN 增量不可分辨（ΔIC +0.009 < 0.02），换图仅 +3.3% 成本、不需 GPU | 工具保留 |
| `blend_verdict_920` | 慢腿 λ 预注册七门裁决：九档无一过门（整轴不可分辨），维持 λ=0.5；λ 是换手旋钮 | 判负 |
| `blend_weight_920` | 快慢融合 λ 补扫九档：λ≤0.4 是平台、0.5 起单调衰减；0.5 是换手选择非最优点 | 工具保留 |
| `cheap_lever_scan` | 低成本仍有效：小 buffer entry + 高 h5 权重，四窗口一致 | 采纳 |
| `consensus_availability` | 分析师一致预期数据不可得，不立项 | 判负 |
| `cost_penalty_probe` | 成本感知 QP 自救后仍输简单 buffer | 判负 |
| `data_root_incident` | 数据根切残缺副本致出榜停摆，已按选项 A 修复留痕 | 事故记录 |
| `deliverable_portfolio` | 可交付组合甜点 35–45 只；行业硬只数上限拖累 | 采纳 |
| `downfreq_realize_probe` | 降频兑现收线：隔夜信息在开盘缺口定价、缺口后反转；跨市场隔夜族不立项，vr 转预注册 | 判负 |
| `etf_contest5` | ETF 轮动**唯一文档真源**：生产 `monthly` + `balanced`（ERC），`core` 已退役 | 采纳 |
| `etf_champ` | 长线 `monthly`（原「磐石」）信号板运行产物（口径见 `etf_contest5` §五、§十四） | 采纳 |
| `etf_coverage` | 行业轮动→ETF 可投性时点化覆盖率表（ρ 初筛 + TE 判据） | 工具保留 |
| `factor_dashboard` | 聚宽式全A因子看板（ICIR/NW-t/月度 IC 现场算），全A库无分层业绩 | 工具保留 |
| `factor_board` | 因子看板 v2：真回测补分层组合业绩，13 逻辑族展示口径，**不构成改进结论** | 工具保留 |
| `factor_dict_probe` | 飞书因子字典 5 候选：4 个实为同一低波因子，只留 `fd_tcc` + `fd_ent30` | 归档 |
| `factor_ev_rd` | EV 分母组合层有效（+3.63pp）、研发强度 `rd_to_cap` 最强；研发加回不合格 | 采纳 |
| `factor_library_backfill` | 全A库 P0+P1 维护：批量补分层回测 / 显著性 / formula 回填（2026-11-05 起） | 进行中 |
| `factor_pool_composition` | 量价占 86%；非量价 28% 席位只贡献 9.5% gain | 归档 |
| `factor_half_life` | 周频方向从因子层封死：944 个中性化因子无短周期原料，因子越强周频越吃亏 | 判负 |
| `focus_holding` | 持仓数 K∈[5,25] 年化无显著差异，等权近最优 | 采纳 |
| `focus_upgrade` | 集中档换信号 `ens_h1h5`→`gbdt_mid13`，9 臂 × 12 K 同层对照 | 采纳 |
| `buffer_exit_scan` | 缓冲带 `exit_mult` × K：成本维度无余量 | 判负 |
| `focus_riskbudget` | focus 内部风险预算（个股配权 vs 整仓缩放）无空间 | 判负 |
| `focus_micro` | 10 万本金整手实跑的小资金方案选型（四维一次扫清） | 采纳 |
| `focus_sector` | focus 加行业去重（每申万一级 ≤N 只），11 臂 × 21 相位 | 归档 |
| `four_paths` | 四条提升路径全收口，无一过门槛 | 判负 |
| `ls_arm_probe_2026-11-24` | 多空口径研究笔记：空头腿年化与 1x/2x 口径（查表题，未跑） | 归档 |
| `founder_hf_probe_2026-11-24` | 方正高频因子低频化 16 因子 × 因子字典覆盖核对（未跑因子） | 归档 |
| `frontier_timing_probe` | 择时前沿源 L1：`us_ret1` 是纯择时线首个样本外稳定日度信号 | 归档 |
| `futures_basis_probe` | 股指期货基差方向预测判负（纯择时线第八篇） | 判负 |
| `gap_fade_probe_2026-11-24` | 「高开要跑」验证为真；「主力兑现/埋伏」叙事证伪 | 归档 |
| `hf_minagg_probe` | 华福 1114 高频聚合族：L1 IC 层复现过，L2 策略层判负 | 判负 |
| `hf_vol_timing_probe` | 华福放量/缩量择时：口径复现成立、风控叠加判负；`vr` 成线内最强候选腿 | 归档 |
| `industry_rotation` | 原 `sue_low` alpha 来自前视 bug，结论已推翻 | 判负 |
| `legs_combo_probe` | 三腿兑现与组合：同资产主门不过，`vr` 取代 `us_ret1` 成最强腿 | 判负 |
| `limitup_halfway` | 半路（盘中+5%）：事件研究 +0.96% 是虚高，组合口径为负 | 判负 |
| `limitup_relay` | 次日接力逐笔 −0.474% / 年化 −71%，9 年全负 | 判负 |
| `mask_order_audit` | 掩码作用顺序 off-by-one 审计 + 修复 | 事故记录 |
| `multisrc_timing_probe` | 多源日频择时合成判负；非价格源首出真信号（`margin_buy_mom`） | 判负 |
| `pool_scope_probe` | 全A信号下沉指数池：同池等权超额大幅衰减 | 工具保留 |
| `rd_overlay` | `rd_to_cap` 叠加进定版：生产口径复核（27bp + T+1 open + 整手，2026-11-13 起） | 进行中 |
| `rev_shell_probe` | 周频均值回归壳六门全过 → 采纳候选（≠定版） | 采纳 |
| `revshell_stitch_probe` | `rev_shell` 组合层缝合判负，壳保持独立候选腿 | 判负 |
| `sws_timing_probe` | 申万 regime 择时复现判负（纯择时线首篇） | 判负 |
| `tier_sector_concentration` | 三档形态的行业/个股集中度诊断 | 工具保留 |
| `valuation_position_rule` | 估值定仓位（PE 分位平滑 + 波动缩放 + 趋势加成） | 工具保留 |
| `vol_control_review` | 回撤控制六类方法核查，定版回撤拆解 | 归档 |
| `vr_realize_probe` | `vr` 量能腿预注册兑现探针判负 | 判负 |
| `sector_constraint_tier` | 行业约束档位（7 臂 × 21 相位），已被 `focus_sector` 覆盖 | 归档 |
| `rebalance_granularity` | solo 档调仓粒度：换手在赚钱，换手预算收口 | 判负 |
| `report_replica`, `report_replica/industry_direct`, `report_replica/ic_bridge`, `report_replica/p0a` | 七维共振复现：7 类只做出动量 1 类（子目录为行业直训 / IC 经济桥接 / P0a） | 判负（框架保留） |
| `frontier_audit_2026-10-19` | h60 预测力存活但组合层无增量；残差无正交信息 | 判负 |
| `horizon_verdict_920` | h20-only 正式裁决：维持定版 | 判负 |
| `hp_sensitivity` | 13 配置超参扫描无被漏掉的低垂果实 | 判负 |
| `impact_cost_probe` | 逐股冲击 vs 固定滑点：1e8 下≈10bp，成本是 AUM 强函数 | 工具保留 |
| `index_enhance_probe` | 中证500 IC 0.069 但 3–5% TE 净超额仅 1.5~1.9%，2025 −7% | 判负 |
| `index_rebalance_flow_probe` | ETF 申赎流与指数调样双线判负（纯择时线第九篇） | 判负 |
| `jq_amount_roll` | 成交金额族加进 gbdt 无增量 | 判负 |
| `jq_factor_probe` | 285 因子复核：仅成交金额族未被覆盖且本地过线 | 采纳 |
| `library_metadata_backfill` | registry 元数据回填，`family` 965/965 | 工具保留（治理） |
| `library_governance` | 因子库治理：register 拆 ingest/evaluate，965 因子有评估 | 工具保留（治理） |
| `market_direction_probe` | 本机可取 3 个新状态全不通过 | 判负 |
| `margin_weekly_probe` | 两融日频新造 15 因子：周频匹配没过；附两个强慢因子 | 判负（周频） |
| `margin_monthly_incr` | 两个两融慢因子换皮过、组合层没过 | 判负 |
| `model_performance` | 主模型回测表现页（净值/回撤/分年/热图），引擎调 `caliber` 同源 | 工具保留（产物） |
| `model_state_probe` | 五类模型自诊断 0/5 显著，8 个仓位臂全败 | 判负 |
| `multihorizon_fusion` | 多 horizon 前融合「年化没变高 = 没用」 | 判负 |
| `news_alpha_probe` | news-only 可交易 IC 仅 0.0098(h1)/0.0174(h20)，融合在噪声内 | 判负 |
| `p1_sequence_predictor` | 曾报 GRU +57%，自证核心结论不成立 | 判负 |
| `p1b_prod_baseline` | 同口径复验：P1 GRU 优势不成立（IC≠可交易收益） | 判负 |
| `p2_objective` | 损失函数 / 样本权重双维扫描无增量 | 判负 |
| `panels_rebuild_incident` | `panels_neu` 三度中断，缺陷全在守卫/写盘层，已修复 | 事故记录 |
| `pead_probe` | 预告/快报/SUE 14 因子已在生产池，事件族 IR 0.08–0.11 判否 | 判负 |
| `st_guard` | 状态表断更致 ST 剔除静默失效；回测侧正确、实时链已修 | 事故记录 |
| `portfolio_weighting` | 优化层空白维：TopK45 + inv_vol + buf1.5 | 采纳 |
| `pv_marginal` | 量价边际曲线并非平滑见顶 | 归档 |
| `raft_weight` | RAFT 式样本加权在全A主模型显著为负 | 判负 |
| `new_source_scan` | 新数据源复核：本地数据已榨干 | 判负 |
| `intraday_source` | 日内因子可交易性复核判负 | 判负 |
| `horizon_h40` | 补训 h=40 无新信息（IC 仅 +0.0008） | 判负 |
| `em_rating_probe` | 卖方评级 8 个构造全部未过预注册 | 判负 |
| `event_driven_probe` | 事件驱动独立组合九族一次性判生死，`inc_pure` 立项 | 采纳（`inc_pure` 立项） |
| `event_opt_probe` | `inc_pure` 优化 6 臂 + 5 新事件族：无效或噪声，非独立腿 | 工具保留 |
| `exec_fill_probe` | `inc_pure` 能买进，但必须买在开盘集合竞价 | 采纳（执行口径） |
| `exec_fill_probe_h10` | `inc_pure` H=10 对照：可成交性干净、净额更高但容错少（2026-11-16 起） | 进行中 |
| `event_chain_probe` | 事件链 + 事件前状态双否：增持的来历不解释收益 | 判负 |
| `event_industry_probe` | 行业密度信号判负，结构性稀释而非调参 | 判负 |
| `inc_pure_playbook` | `inc_pure` 实盘落地清单 + 出单器（操作手册） | 工具保留 |
| `event_reach_probe` | 事件族触及度 + 上帝视角天花板：7 族无一过 | 判负 |
| `regime_probe` | Phase 0 见肉，但条件化最终不立项 | 判负 |
| `risk_budget` | 风险维度已探：市值中性化 −6.4pp/年，行业中性化近乎免费 | 归档 |
| `seg_weight_probe` | 华泰学分钟权重：12 段动量信息只在 seg12，判负 | 判负 |
| `size_neut_strength` | α 0→1 年化单调降，市值暴露是收益来源非旋钮 | 判负 |
| `solo_signal` | `gbdt_mid13` 维持，首次在 solo 自身执行层验证 | 采纳 |
| `solo_weekly` | 周频判负：信号没错，换手 4 倍吃掉 19.5pp | 判负 |
| `exec_phase_month_open` | 「成交提前到月首开盘」判负 | 判负 |
| `rtl_phase_tiers` | 三档相位补齐：对外数字是抽签最大值（27bp 口径） | 采纳（改收益口径） |
| `rtl_rebalance_timing` | 相位极差是文献定名的不可补偿风险，非策略缺陷 | 采纳（改收益口径） |
| `sector_risk_budget` | 行业集中度约束：从标签去重升级到风险同源 | 判负（该维度收口） |
| `style_exposure` | 定版持仓的 Barra 风格体检：广度=低波+价值，k8=微盘低波价值（非高 β） | 工具保留 |
| `text_factor_probe` | `txt_*` IR 0.008~0.105；融合增益被安慰剂证伪 | 判负 |
| `textmining` | 文本因子管线产物目录的分层布局（非实验结论） | 工具保留 |
| `yq_core` | YQ 核心产品线（solo/focus 合并为「集中档」两版本） | 采纳（产品线） |
| `vol_target_920` | 目标波动 overlay 在定版底座复验判负 | 判负 |
| `timing_probe` | 择时族入口：日收益自相关 +0.048、月度方向 0/7，可实现上界≈满仓 | 判负 |
| `timing_daily_train` | 日频训练 + 月频执行补测判负 | 判负（该格子收口） |
| `timing_lit_review` | 择时 / 大盘预测外部文献核查，共识与实测一致 | 工具保留 |
| `kronos_probe` | Kronos 零样本 A 股截面排序判负（DL 择时第一例） | 判负 |
| `timing_conditioned` | P0 条件化择时 + P1 SOP 收益分解，两者均判负 | 判负（P0、P1 均收口） |
| `top10_sharpe` | 中国因子十强复现：强的全是换皮 | 判负 |
| `panel_leak_audit` | 全库 970 面板前视体检：39/965 → 11/970 | 工具保留 |
| `structural_fake_panels` | 11 个「结构性无救」假面板的归因与处置 | 事故记录 |
| `strategies` | ETF 策略入口页（`strategy_home.py` 生成） | 工具保留 |
| `rebuild_impact_920` | 前视修复后定版重测 16.31%→14.96%，人工拍板 | 采纳（门槛未过，人工拍板） |
| `caliber` | 统一口径基线落盘（`baseline_prod/fair/impact/flat.json`） | 工具保留 |
| `cpcv_h1` | h=1 CPCV 无偏评估（2026-09-11 重生成） | 工具保留 |
| `sop_timing` | P1 SOP 收益分解产物；分析合并写 `timing_conditioned/` | 工具保留 |
| `freq_tune` | 调仓频率精修（含引擎修复前存档） | 工具保留 |
| `multiyear` | 多年度 OOS 稳健性（2026-09-01 修复后重跑） | 工具保留 |
| `buffer_tune` | 换手缓冲带调参 | 工具保留 |
| `gtja_repro` | 国金因子复现对照（读自身 `baseline.csv`） | 工具保留 |
| `risk_decomposition` | 组合风险分解报告（Euler 分解 / 因子方差贡献 / VaR 成分） | 工具保留 |
| `ml_synthesis_h1` | 归档时代产物（原读取方 2026-11-19 已删，面板保留） | 工具保留 |
| `investment_report` | HS300 时代交付物存档（年化口径 ~3% 偏移待重跑） | 工具保留 |
| `walk_forward`, `two_periods` | 早期回测产物（原读取方 2026-11-19 已删） | 工具保留 |
| `_htai_gp` | 华泰 GP 挖掘对照数据 + 手写方法论总结 | 工具保留 |
| `neutralize_vectorization` | neutralize 逐日循环向量化的等价性证据与计时 | 工具保留 |
| `e2e_p0_intraday` | 日内 P0 A/B 链路（2026-09-23 五合一） | 工具保留 |
| `holder_dyn` | 股东动态 9 因子三探针（2026-09-23 二合一） | 工具保留 |
| `limit_pos` | 涨跌停纸面因子治理（2026-09-23 二合一） | 工具保留 |
| `portfolio_ppo_phase1` | stage2 Phase 1 hs300 平价验证（ppo/qp/ew/topn 四臂） | 工具保留 |
| `galaxy_l1` | 银河 0608 研报 L1 复现（脚本已删，2026-11-19） | 工具保留 |
| `fundamental_blend` | 基本面族慢速权重融合对照 | 工具保留 |
| `stack_blend` | stacking 融合权重实验（脚本已删，2026-11-19） | 工具保留 |
| `alla_arms` | 四臂（alt/base/rre_0.3/rre_0.5）特征集对比 | 工具保留 |
| `alla_daily_ortho`, `alla_daily_h1h5`, `alla_daily_det_c`, `alla_daily_ortho_check` | 定版前快照（已作废，勿再新增） | 工具保留 |
| `panels_neu_backfill` | `panels_neu` 882→920 补齐 38 缺的证据 | 工具保留 |
| `pbo_rolling_ortho` | ortho 臂 PBO（过拟合概率）检验 | 工具保留 |
| `horizon_mix` | horizon 混选（h1+h5 集成 vs 纯 h1）+ close 口径对照 | 工具保留 |
| `prod_pipeline_gap` | 生产管线 vs 回测管线 2×2 缺口诊断 | 工具保留 |
| `daily_review` | 每日复盘与探针日志 | 工具保留 |
| `daily_determinism` | 出榜链确定性判定：链路严格确定，偏差来自未定稿代码 | 工具保留 |
| `altdata_cls_audit` | 财联社电报全量回补审计（119.2 万条） | 工具保留 |
| `altdata_macro` | 东财宏观日历三次补抓运行日志 | 工具保留 |
| `alt_paper_fingerprint` | alt 族 28 因子纸面指纹体检（仅 1 个通过） | 工具保留 |
| `clean_label_alt` | 可交易掩码修标签后，28 个 alt 因子 gain 重测 | 工具保留 |
| `signal_mktcap_drift` | 入选信号市值漂移（月/年）追踪 | 工具保留 |
| `structured_evidence_recalc` | qcut 分箱坍缩扫描 | 工具保留 |
| `kronos_smoke`, `tabpfn3_smoke` | Kronos / TabPFN3 模型冒烟结论 | 工具保留 |
| `portfolio_phase2_smoke` | 组合优化 Phase 2 冒烟 | 工具保留 |
| `portfolio_phase2_mask_smoke` | 掩码注入验证冒烟（ppo/qp/ew 全臂） | 工具保留 |
| `llm_mcts_phase0` | LLM-MCTS Phase 0 四臂冒烟 | 工具保留 |
| `tabpfn_rolling_compare_smoke` | 表格基础模型对照 runner 的 gbdt 臂冒烟 | 工具保留 |
| `ic_headroom` | IC 可达上界扫描：最优线性合成 0.0785 | 工具保留 |
| `dpp_sigma_sweep` | DPP sigma × quality 扫描：s≥0.2 全平 | 工具保留 |
| `sync` | 产物同步 zip（本地保留，2026-10-27 已移出跟踪） | 工具保留 |
| `capacity_table` | 集中度 × AUM 二维容量表（脚本已删） | 工具保留 |
| `barra_cne6` | CNE6 风格回归产物（因子收益 / t / 残差面板） | 工具保留 |
| `chip_behavior_probe_2026-10-27` | 行为金融「筹码/换手」廉价探针 | 判负 |
| `cyq_probe_2026-11-18` | CYQ 筹码分布三因子，L1~L5 无一过筛 | 判负 |
| `network_probe_2026-10-27` | 跨股票网络中心度 P1 探针 | 判负 |
| `cohort_mom_probe` | 建仓后分组动量探针 | 判负 |
| `hp_search_ic` | 超参扫描（只比可交易 IC 的收口证据） | 采纳 |
| `retrain_freq_ic` | 重训频率对照（逐日/周/月/季） | 采纳 |
| `intraday_daily` | 日内因子日频组合（期限对齐）——第一次真正跑日频信号×日频调仓 | 判负 |
| `intraday_half_life_probe` | 日内因子半衰期单表探针 | 归档 |
| `member_blend12` | stacking 成员 12 选融合的指标快照 | 判负 |
| `text_factor_probe_smoke` | `text_factor_probe` 的冒烟子集 | 判负（随父实验） |
| `ai97_pbo` | AI97 因子集的 PBO / DSR 报告 | 工具保留 |
| `alphapool_ppo` | AlphaPool PPO 三臂 × 三种子训练/评估记录 | 工具保留 |
| `gflownet_ckpt` | GFlowNet 因子挖掘的模型权重存档（12 个 `.pt`） | 工具保留 |
| `batch_920_consolidated` | 920 批次收尾对照表与结果速览 | 归档 |
| `e2e_backtest` | 批次 7 年化口径重出，HS300 时代链路备份证据 | 归档 |
| `mf10_t_scan` | MF10 单因子 t 值扫描（hs300 2022–2025） | 归档 |
| `tabfm_rolling` | TabFM 滚动预测产物（all_a / hs300 / zz1000） | 归档 |
| `views_arm_920_smoke` | views 臂的 smoke 试跑 | 归档 |
| `vol_target_920_const` | 目标波动 overlay 的恒定敞口对照臂 | 归档 |
| `yelaoshi_first_board_2026-10-25` | 首板策略回测产物（脚本已删） | 归档 |
| `_rebuild_logs` | 前视修复后重建的全程日志 + `HANDOFF.md` | 归档 |
| `portfolio_phase2`, `portfolio_phase2_full`, `portfolio_phase2_full_parts` | 组合 Phase-2 实验与消融分解臂 | 归档 |
| `mf_fundind_check` | 批次 14：mf 干净单臂补测不达验收，mf 线收口 | 工具保留 |
| `frontier_audit_920_reval` | 批次 13 920 复验（A1/A2 三臂全败线封） | 工具保留 |
| `cvar_arm_920`, `views_arm_920` | 批次 12：CVaR 组合臂（不升级）与观点注入三臂（零增量） | 工具保留 |
| `alla_rolling_ortho920mf`, `alla_rolling_ortho920fundind_lblco` | 批次 12：市场状态特征（唯一正结果，待拍板）与隔夜标签臂（判负） | 工具保留 |
| `alla_rolling_jq_base`, `alla_rolling_jq_amt` | 成交金额族 A/B 两臂（结论真源 `jq_amount_roll/README.md`） | 判负 |
| `alla_rolling_jqs_base`, `alla_rolling_jqs_amt` | 同上两臂的 M 频单轨快照 | 判负（随父实验） |
| `alla_rolling_macrof`, `alla_rolling_newsf` | altf P1 宏观/新闻族臂（`_base` 已清，pred 保留） | 工具保留 |
| `alla_rolling_macrof_base`, `alla_rolling_macrof_h20`, `alla_rolling_newsf_h20` | altf P1 旁路特征臂：ΔIC −0.0007 / −0.0005，判负 | 判负 |
| `alla_rolling_ortho920tl`, `alla_rolling_ortho920v2`, `alla_rolling_ortho920e4`, `alla_rolling_ortho920e4force`, `alla_rolling_ortho920nolabels`, `alla_rolling_ortho920fundind2`, `alla_rolling_ortho920fundind3`, `alla_rolling_ortho920fundind4`, `alla_rolling_ortho920fundind5`, `alla_rolling_ortho920fundind_h40`, `alla_rolling_ortho920fundind_mf`, `alla_rolling_ortho920tl_lblco`, `alla_rolling_ortho920tl_lblgauss`, `alla_rolling_ortho920tl_lblir` | 920 批次治本/消融/标签臂（结论见 `batch_920_consolidated/`） | 归档 |
| `buffer_arm_920`, `buffer_arm_920v2`, `buffer_arm_920v2_defv`, `buffer_arm_920fundind_defv`, `buffer_arm_920fundind3_defv`, `buffer_arm_920fundind4_defv` | 缓冲带参数扫描臂（见 `batch_920_consolidated/`） | 归档 |
| `buffer_sweep_p2`, `buffer_sweep_p2_v2` | 缓冲带二阶段网格扫描（每格一目录） | 归档 |
| `fundamental_blend_920`, `fundamental_blend_920_hd`, `fundamental_blend_920v2`, `fundamental_blend_920v2_T3`, `fundamental_blend_920v2_T6`, `fundamental_blend_920v2_T12`, `fundamental_blend_920fundind`, `fundamental_blend_920fundind2_altnews`, `fundamental_blend_920fundind3`, `fundamental_blend_920fundind4`, `fundamental_blend_920fundind5`, `fundamental_blend_920fundind_full` | 基本面慢腿混入权重臂（见 `batch_920_consolidated/`） | 归档 |
| `horizon_mix_920`, `horizon_mix_920v2`, `horizon_mix_920fundind`, `horizon_mix_920fundind3`, `horizon_mix_920fundind4`, `horizon_mix_920fundind5` | h1/h5/h20 周期混合权重臂（见 `batch_920_consolidated/`） | 归档 |
| `stack_blend_920`, `stack_blend_920v2` | stacking 权重混合臂 | 归档 |

### 1.3 文档与散件

- `docs/` 自成体系，导航见 `reports/docs/README.md`；研报研读笔记在 `docs/research_notes/`（被 TODO_ARCHIVE 册Ⅱ 指针引用，不做合并）。
- 主要笔记：`docs/research_notes/文本选股研报解读与数据源调研.md`、`docs/research_notes/另类数据调研三部曲_2026-09-20.md`、`docs/research_notes/短板匹配与微信文章研读_2026-09-22.md`、`docs/research_notes/国泰海通七维共振_方法论评述_2026-11-02.md`（其余篇目见该目录）。
- 关键文档：`docs/design/yuriquant_project_overview/`、`docs/design/yuriquant_model_layer_design/`、`docs/design/yuriquant_training_discipline/`（三条设计说明）；`docs/另类数据管道_数据源说明.md`；`docs/consistency_checks/`（研报口径与公平性核对）；`archive/ml-synthesis-hs300-report/`（已归档 ML 合成交付报告）。
- 历史清理记录（原 §四，2026-11-25 迁出）：`reports/docs/archive/清理记录_2026-11-25.md`。

### 1.4 根部散件（**接口文件，勿移动**）

| 文件 | 说明 | 硬引用点 |
|---|---|---|
| `experiments.csv` | 实验台账 | `research/experiments.py:35`（`DEFAULT_PATH`） |
| `intraday_stat_checkup.csv` | 分钟统计因子体检报告（结论录于 TODO_ARCHIVE 册Ⅰ §一） | `scripts/factors/build_intraday_stat_factors.py:271` |

> 根目录除 `README.md` 外只允许上述 2 个接口文件，移动即静默断链。

### 1.5 日志与内部工作区

| 位置 | 说明 |
|---|---|
| `logs/` | **不属于任何单一实验**的跨主题运行日志。属于某实验的日志一律放该实验目录内 |

## 二、主线实验谱系（读报告前先看这个）

```
全A滚动实验（rolling_grid_alla.py，全A 5549 股 × 900 因子 × 2018-2026）
├── alla_rolling/          zscore 臂（主数据枢纽，_base/pred/selection 被生产读取）
├── alla_rolling_mixed/    mixed 臂   → 已删（2026-09-11 删主产物 / 09-16 残存 _base 亦移除，结论见 §三.1）
├── alla_rolling_nofund/   基本面消融 → 已删（同上，结论见 §三.2）
└── alla_rolling_ortho/    ortho 臂 = 管线默认（最终臂）
     └── 下游生产：model_portfolio/（组合交付）、alla_daily/（每日排名）、
         alla_attribution/（归因复核）
HS300 时代（已被全A线取代；仅存 walk_forward / two_periods / investment_report /
model层 / ML合成，其余已删、可重跑复现——结论见 §三.13~17）
```

---

## 三、已清理目录的结论备份（2026-09-11 删除，原文不可恢复）

1. **`alla_rolling_mixed/`**（约 1.2GB）——mixed 臂（仅基本面族中性化）。结论（册Ⅰ 附录）：
   全正交 +1.6pp/年 > 混合 +0.1pp ≈ 不正交；增量几乎全来自量价，40/42 个 h1 配置全正交占优。
   重生成 `rolling_grid_alla.py --preproc mixed`。*（09-16 残存 `_base/` 446MB 亦删）*
2. **`alla_rolling_nofund/`**（约 1.2GB）——排除基本面族消融。结论（同上）：基本面/股东族
   ~+0.7pp/年（含族 +10.2% vs 纯量价 +9.5%）。重生成 `--ablation`。*（`_base/` 亦删）*
3. **`alla_rolling_mixed_run.log`** —— 随臂删除的手动重定向日志。
4. **`gbdt_tune/`** —— 引擎修复前（2026-08-25）网格：h=1 × frac=0.2 最优（成本后 27.0% /
   Sharpe 1.70），h=3/5 全负；后证实是 h>1 引擎 bug 伪结果，不可引用。其「h1 + Top20% 甜点」
   已演进为 `model_portfolio` 固化配置。
5. **`gp_tune/`** —— 旧版 GP 预算网格：seed 间方差大、最优 test IC 0.025（train 0.063）；
   已被重写后的 `gp_tune_budget.py` 取代。
6. **`cpcv_eval/`** —— horizon 混选 CPCV（修复前）：ridge IC 0.044 / gbdt 0.050，15/15 路径为正，
   但含 horizon 选择偏差；已被无偏版 `cpcv_h1/` 取代。
7. **`adaptive_long_validation/`** —— 自适应调仓 ov30–ov55 Sharpe 全负（−0.15 ~ −0.78）vs 月频
   基线 0.06，阈值越敏感损耗越大 ⇒ **不采用自适应调仓**。
8. **`adaptive_rebalance/`** —— ov40 最优（Sharpe 0.84 vs 基线 0.57），但整体超额为负、样本约
   2 年 ⇒ 未采纳进固化配置。
9. **`timing_vs_crosssection/`** —— 目标波动 / 状态机择时均不优于纯截面（Sharpe 0.57 / 0.54 /
   0.29，2025 年段）；与 `position_overlay_grid` 一致：overlay 只压回撤、不提升 Sharpe。
10. **`select_hs300_2025/`** —— HS300 选股演示（幸存者偏差修复前，数字虚高：top30 年化 48% /
    Sharpe 2.54）；原 `select_stocks.py` 已于 2026-11-18 随 `scripts/archive/` 删除，只能从 git 历史取。
11. **`textmining_txt/`** —— 文本挖掘早期 txt 堆放处，已被 `textmining/` 分层布局取代，无引用。
12. **`models_mock/`** —— 旧模型台账（生成脚本已删，2026-11-19；产物只读）。
13. **`diagnose/`** —— 单因子 vs 模型同口径诊断：负 IC 单因子 top50 超额 +20%~+36%，实为
    高波动 / 高 beta 风格暴露（册Ⅰ 附录）；`diagnose_factor_vs_model.py` 可复现。
14. **`diagnose_neu/`** —— 中性化前后：模型 IC 0.059→0.015 而超额 −17.8%→−8.9% ⇒ 剥离风格后
    趋近纯 alpha（与 investment_report「中性化是变现关键」同源）；`diagnose_neutralized_compare.py`。
15. **`screening/`** —— AlphaEval 筛选漏斗：821 → PPS 234 → DPP 164 → RRE 149，
    max|corr| 1.000→0.978；`factor_screening.py`。
16. **`ml_algorithm_compare/`** —— 9 特征 × 2019-2026 OOS：gbdt 0.051(t=5.8) > ridge 0.044 >
    TabICL 0.029 ⇒ **gbdt 系最优，表格基础模型无优势**；`ml_algorithm_compare.py`。
17. **`e2e_backtest/`、`e2e_picks/`、`optimize_e2e/`、`signals/`、`multi_period/`** —— HS300 时代
    e2e 与演示产物（数字带 ~3% 年化口径偏移，册Ⅰ §一 待重跑）。结论：等权 top50 +16.2% 跑输
    全池等权 +30.8%；风格中性化 16.2%→27.0%（Sharpe 0.41→0.66）；周频被证伪；最优 = 月频 +
    中性化（已并入 investment_report）。原脚本 2026-11-18 随 `scripts/archive/` 删除；
    `daily_hs300.parquet` 2026-08-26 停更，复现仅限该缓存窗口。
18. **`intraday_stat_checkup_run.log`** —— 手动控制台重定向日志，结论在同根 `.csv`，无独立价值。

---

## 四、清理记录

历史清理记录（2026-10-27 及各期目录增删/迁移）见 `reports/docs/archive/清理记录_2026-11-25.md`。
