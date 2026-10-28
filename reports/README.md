# reports/ — 产出物索引与报告结论速览

> 本目录整体被 gitignore（`.gitignore:23`），**任何删除都无法经 git 恢复**。
> 研究结论的唯一真源是根目录 [`TODO.md`](../TODO.md)（已收口结论归档于 [`TODO_ARCHIVE.md`](../TODO_ARCHIVE.md) 三册），本文件不重复其内容，只做三件事：
>
> ① **目录地图**——每个目录是什么、谁产出、为什么保留；
> ② **已清理目录的结论备份**——2026-09-11 清理时删除的原文，其结论提炼在 §三；
> ③ **清理记录**（§四）。
>
> 各子目录如另有 README（如 `textmining/README.md`），以子 README 为准。

---

## 一、目录地图（按功能分区）

### 1.1 生产运行（每日/每日计划任务在用，勿动勿改名）

| 目录 | 内容 | 产出脚本 |
|---|---|---|
| `alla_daily_defv/` | 全A每日选股排名、候选、解释（**现行**，Windows 计划任务 `YuriQuant AllaDailyRank --preset dingban`，2026-10-08 起） | `scripts/pipelines/alla_daily_rank.py` |
| `alla_daily/` | ⚠️ legacy 口径产物（旧 h1+h5 ortho），停于 2026-09-22；仅历史追溯 | `scripts/pipelines/alla_daily_rank.py --preset legacy` |
| `alla_rolling/` | 全A滚动实验**主数据枢纽**（zscore 臂）：`_base/` 喂因子构建器、`pred/` 喂归因、`selection/` 喂生产日排名，8+ 组件读取 | `scripts/pipelines/rolling_grid_alla.py`（默认臂）+ `rolling_grid_report.py` + `jq_style_report.py` |
| `alla_rolling_ortho/` | 正交化口径臂——**全A管线默认**（TODO_ARCHIVE 册Ⅰ 附录：全正交 +1.6pp/年 > 不正交），ens_h1h5 集成年化 +13.3% | `scripts/pipelines/rolling_grid_alla.py --preproc ortho` |
| `alla_attribution/` | 主策略 α/β 分解 + Brinson 归因（结论录于 TODO_ARCHIVE 册Ⅰ §一） | `scripts/pipelines/alla_excess_attribution.py` |
| `model_portfolio/` | 固化组合交付物（`pipeline=alla_ortho`；legacy 对照，非现行生产） | `scripts/pipelines/run_model_portfolio.py` |
| `monitoring/` | 监控台账 + 报告。⚠️ 旧链 2026-09-21 退休（计划任务已删，证据 `legacy_monitor_*_20260921.log` / `legacy_task_xml_20260921.xml`）；`production_ic_daily.csv` 为**现役**每日 IC 台账 | `scripts/reporting/monitor_performance.py`（退休，手动可跑）+ `scripts/reporting/monitor_production_ic.py`（现役） |
| `models/` | 模型注册台账（旧监控链读取；现役链不依赖） | `scripts/evaluation/walk_forward_model.py` |
| `textmining/` | 文本因子管线（**活跃开发中**，结构见其自带 README） | `scripts/textmining/*` |

### 结论速查：43 份实验 README（每个目录一行）

> **只做扫描用**：口径、证据与复现命令以各目录 `README.md` 为准；§1.2 说明的是目录职责与
> 产出脚本，本表给的是**判定**。判定口径：`采纳`（进生产/进定版）/ `判负`（实测无增量或被
> 证伪）/ `归档`（有结论但无后续动作）/ `工具保留`（脚本或表仍有读取方）/ `事故记录`。
> 新增实验目录时**必须同时**在本表加一行（与 §1.6 落盘判定表一起登记）。

| 目录 | 结论一句话 | 判定 | 日期 |
|---|---|---|---|
| `altf_neu_backfill` | alt 20 因子回填 `panels_neu` 后门槛 strict 通过，但入选集逐位不变（死重） | 判负 | 2026-10-10 |
| `altf_p1` | alt 三族 20 因子入库；等权合成稀释（news ΔIC 随权重单调降），席位 0/36 | 判负 | 2026-10-09 |
| `altf_p1/model_ablation` | macro / news 作旁路特征 ΔIC −0.0007 / −0.0005，pred 秩相关 0.995+ ⇒ 框架吸收不了 | 判负 | 2026-10-10 |
| `cheap_lever_scan` | 低成本仍有肉：小 buffer entry + 高 h5 权重，两个微调四窗口全一致 | 采纳 | 2026-10-14 |
| `consensus_availability` | 分析师一致预期数据不可得（先验尽调），不立项 | 判负 | 2026-10-13 |
| `cost_penalty_probe` | 成本感知 QP 自救 +11.7pp 后仍输简单 buffer；本目录两版旧结论已作废 | 判负 | 2026-10-13 |
| `data_root_incident` | 数据根切到残缺副本致出榜停摆，已按选项 A 修复并留痕 | 事故记录 | 2026-10-14 |
| `deliverable_portfolio` | 可交付组合甜点 35–45 只（+1.9~2.3pp）；行业硬只数上限 −2.64pp | 采纳 | 2026-10-09/10 |
| `etf_alpha_probe` | 个股 α 按 ETF 成分权重聚合到 ETF 层，无截面预测力 | 判负 | 2026-10-26 |
| `etf_capacity` | 100–500 万无压力（0% 超限）、1000 万拆两笔可执行 | 工具保留 | 2026-10-27 |
| `etf_champ` | 长线「磐石」信号板；接入动态去重后年化 13.75% / Sharpe 1.08 / MDD 14.0% | 采纳 | 2026-10-27 |
| `etf_contest5` | 比赛定版「三锋 v2」：80 窗均值 +5.59% / 最差 −12.34% / 胜率 72.5% | 采纳 | 2026-10-27 |
| `etf_dedup` | 选标层动态去重（13 对双胞胎）：champ 年化 12.37→13.75%、MDD 16.1→14.0% | 采纳 | 2026-10-27 |
| `etf_p4` | 样本外 champ_hrp 年化 20.60% / Sharpe 1.27 PASS；P1 基线 OOS 失效 | 采纳 | 2026-10-26 |
| `etf_pool_expand` | 41→50 扩容：基线 +0.03、champ +0.09 Sharpe，方向正确幅度有限 | 采纳 | 2026-10-26 |
| `etf_rotation_p2` | 防御腿 + 拥挤度叠加到年化 11.23% / Sharpe 0.85 / MDD 16.70%；趋势 R² 判负 | 采纳 | 2026-10-26 |
| `etf_rotation_probe` | P1：波动率目标是唯一有效增量，年化 7.14% / Sharpe 0.62 / MDD 19.01% | 采纳 | 2026-10-26 |
| `etf_survivorship_probe` | 退市偏差结构性可忽略（不可能进池）；池选择偏差已量化 | 归档 | 2026-10-26 |
| `factor_pool_composition` | 量价占比 86.0%（入选 84% → gain 90.5%）；非量价 28% 席位只贡献 9.5% | 归档 | 2026-10-10 |
| `four_paths` | 四条提升路径全收口，无一过门槛（新信息源 / 新归纳偏置均空） | 判负 | 2026-10-10 |
| `frontier_audit_2026-10-19` | h60 预测力存活（IC 0.161）但组合层无增量；残差无正交信息 | 判负 | 2026-10-19 |
| `hp_sensitivity` | 13 配置超参扫描无被漏掉的低垂果实，当前点已在地板 | 判负 | 2026-10-10 |
| `impact_cost_probe` | 逐股冲击 vs 固定滑点：1e8 下冲击 ≈ 固定 10bp，成本是 AUM 强函数 | 工具保留 | 2026-10-13 |
| `index_enhance_probe` | 中证500 内 IC 0.069 / t=4.3，但 3–5% TE 净超额仅 +1.5~1.9%（t≈0.8）、2025 −7% | 判负 | 2026-10-26 |
| `jq_amount_roll` | 成交金额族加进 gbdt 无增量（950 特征库已重构掉这份信息） | 判负 | 2026-10-26 |
| `jq_factor_probe` | 285 因子复核：仅成交金额族未被覆盖且本地过线，换手/流动性边缘 | 采纳 | 2026-10-26 |
| `market_direction_probe` | 本机可取 3 个新状态全不通过（|t|≤0.71、五分位不单调、样本外为负） | 判负 | 2026-10-26 |
| `model_state_probe` | 五类模型自诊断状态 0/5 显著，8 个仓位臂无一通过 | 判负 | 2026-10-23 |
| `multihorizon_fusion` | 多 horizon 前融合「年化没变高 = 没用」；集中度裁决同步收口 | 判负 | 2026-10-14 |
| `news_alpha_probe` | news-only 可交易 IC 仅 0.0098(h1)/0.0174(h20)，融合在 ±0.3pp 噪声内 | 判负 | 2026-10-10 |
| `p1_sequence_predictor` | 曾报 GRU +57%；10-13 自证核心结论不成立 | 判负 | 2026-10-13 |
| `p1b_prod_baseline` | 同口径复验：P1 的 GRU 优势不成立（IC 提升 ≠ 可交易收益） | 判负 | 2026-10-13 |
| `p2_objective` | 损失函数 / 样本权重双维扫描只轻微扰动，无增量 | 判负 | 2026-10-13 |
| `panels_rebuild_incident` | `panels_neu` 三度中断：四个缺陷全在守卫/写盘脚手架层，已修复 | 事故记录 | 2026-10-16 |
| `pead_probe` | 预告/快报/SUE 14 因子已在生产池；事件族可交易 IR 0.08–0.11 ⇒ 判否 | 判负 | 2026-10-10 |
| `portfolio_weighting` | 优化层空白维探明：TopK45 + inv_vol + buf1.5 = 18.07% / Sharpe 0.736 | 采纳 | 2026-10-10 |
| `pv_marginal` | 量价边际曲线并非平滑见顶；前作「DPP50→150 −24%」对照不公平 | 归档 | 2026-10-10 |
| `regime_probe` | Phase 0 见肉（IC 0.127 vs 0.110、t=3.55），但条件化最终不立项 | 判负 | 2026-10-14 |
| `risk_budget` | 风险维度已探过：市值中性化 −6.4pp/年、行业中性化近乎免费 | 归档 | 2026-10-10 |
| `size_neut_strength` | α 0→1 年化单调 15.52→13.80%，无内部最优 ⇒ 市值暴露是收益来源不是旋钮 | 判负 | 2026-10-09 |
| `text_factor_probe` | `txt_*` 5 个 IR 0.008~0.105；融合增益被安慰剂证伪；106 项 FDR 仅 1 条存活且为负 | 判负 | 2026-10-13 |
| `textmining` | 文本因子管线产物目录的分层布局约定（非实验结论） | 工具保留 | — |
| `timing_probe` | 择时收口：日收益自相关 +0.048、月度方向 0/7 ⇒ 可实现上界 ≈ 满仓 | 判负 | 2026-10-14/15 |

### 1.2 保留实验与报告（有代码读取、或支撑已归档脚本引用的结论；均可重跑复现）

| 目录 | 说明 |
|---|---|
| `cpcv_h1/` | h=1 CPCV 无偏评估（2026-09-11 重生成） |
| `freq_tune/` | 调仓频率精修（含引擎修复前存档 `freq_tune_prefix_engine_fix_20260825.csv`） |
| `multiyear/` | 多年度 OOS 稳健性（2026-09-01 修复后重跑） |
| `buffer_tune/` | 换手缓冲带调参 |
| `gtja_repro/` | 国金因子复现对照（读自身 `baseline.csv`） |
| `etf_rotation/` | ⚠️ legacy（2026-08-26，15 只池、无 PIT 掩码）；已被 `etf_rotation_probe/` 取代 |
| `etf_rotation_probe/` | **ETF 轮动 P1（2026-10-26，现行）**：41 池 + PIT 掩码；基线 `dual_ivvt15_m5` 年化 7.14% / Sharpe 0.62 / MDD 19.01%（结论真源其 README） |
| `etf_rotation_p2/` | **ETF 轮动 P2 前置（2026-10-26）**：防御腿 + 拥挤度 + 趋势 R² 三对照；`def_mix_crowd_f20` 年化 11.23% / Sharpe 0.85 / MDD 16.70% |
| `etf_{champ,contest5,contest5_opt,contest5_wl,contest_wl,contest_wl2,p4,dedup,capacity,pool_expand,survivorship_probe,alpha_probe}/` | **ETF 赛事 / 长线全家桶（2026-10-26~27）**：三锋 v2 真源 = `etf_contest5/README.md` §十三（参数 `etf_contest_spec.FINAL`）；长线「磐石」= `etf_champ/`；`contest5_opt/` 定版证据、`contest_wl*/` 白名单复核、`p4/` 样本外、其余专项探针；作废初赛目录已回收（§四） |
| `risk_decomposition/` | 组合风险分解报告（Euler 分解 / 因子方差贡献 / VaR 成分） |
| `ml_synthesis_h1/` | ⚠️ 归档时代产物，但 `monitor_performance.py` 读其预测面板 ⇒ **不可删** |
| `investment_report/` | HS300 时代交付物存档（年化口径 ~3% 偏移待重跑）；归档脚本以其为输入 |
| `walk_forward/`、`two_periods/` | 早期回测产物；`research/report_pipeline.py`（generate_report）读取其 `summary.csv` |
| `_htai_gp/` | 华泰 GP 挖掘对照数据 + 手写方法论总结（`scripts/archive/compare_ml_synthesis.py` 读取） |
| `neutralize_vectorization/` | neutralize 逐日循环向量化的等价性证据与计时（2026-09-20；`factor/preprocessing.py` 的 `neutralize_batch` / `neutralize_grouped`，默认已切 2.96x 逐位一致版） |
| `e2e_p0_intraday/` | 日内 P0 A/B 链路（2026-09-23 五合一：`daily/` 每日链+placebo、`with_im/` `no_im/` 分钟因子对照、`w2023h2/` `w2025h1/` 双窗口日志；产出 `scripts/oneoff/_p0_intraday_ab*.py`） |
| `holder_dyn/` | 股东动态 9 因子三探针（2026-09-23 二合一：`ablation/` A/B 消融、`forced/` 强制纳入臂 Δ=−0.85pp，结论已归档 TODO；pred 中间产物已清理） |
| `limit_pos/` | 涨跌停纸面因子治理（2026-09-23 二合一：`tradable_p0/` 池内 −55.6%/年缺口 +126.7pp、`ablation_p1b/` 全项改善 +0.3~1.2pp；结论驱动生产默认 `--exclude-features limit_pos --tradable-labels`） |
| `portfolio_ppo_phase1/` | stage2 Phase 1 hs300 平价验证（ppo/qp/ew/topn 四臂；`scripts/factors/run_portfolio_ppo.py`） |
| `galaxy_l1/` | 银河 0608 研报 L1 复现：三标签 GBDT RankIC（valid 正 / test 负；`scripts/evaluation/galaxy_l1.py`） |
| `fundamental_blend/` | 基本面族慢速权重融合对照（`scripts/evaluation/fundamental_blend.py`；`hd_*` 为同脚本的 hd 版口径产物） |
| `stack_blend/` | stacking 融合权重实验（`scripts/evaluation/stack_blend.py`，复用 fundamental_blend 的 ICIR 慢速面板） |
| `alla_arms/` | 四臂（`alt/` `base/` `rre_0.3/` `rre_0.5/`）特征集对比；各臂仅存 `arm_config.json`（口径配置），pred 中间产物 2026-09-23 清理 |
| `alla_daily_ortho/`、`alla_daily_h1h5/`、`alla_daily_det_c/`、`alla_daily_ortho_check/` | 09-16 出榜链定版的四个候选臂快照（history/leaders/explain/feature_importance；定版口径见 `alla_daily/`） |
| `panels_neu_backfill/` | panels_neu 882→920 补齐 38 缺的证据（build.log + selection_diff.log；派生 TODO_ARCHIVE 册Ⅰ §一 P0 重跑项） |
| `pbo_rolling_ortho/` | ortho 臂 PBO（策略过拟合概率）检验（M/W 频率 × neut/raw 口径） |
| `horizon_mix/` | horizon 混选（h1+h5 集成 vs 纯 h1）+ close 口径对照 |
| `prod_pipeline_gap/` | 生产管线 vs 回测管线 2×2 缺口诊断（结论推进 TODO_ARCHIVE 册Ⅰ §一 P0 项） |
| `daily_review/` | 每日复盘（09-18 review、09-20 四源 Phase0 探针日志、事件驱动数据接入评估） |
| `daily_determinism/` | 出榜链「同天两跑结果不同」判定：链路本身严格确定，偏差来自用重构未定稿的代码跑 |
| `altdata_cls_audit/` | 财联社电报全量回补审计（119.2 万条；推翻另类数据 P0 计划「新闻快讯流无历史可回补」前提） |
| `altdata_macro/` | 东财宏观日历三次补抓运行日志（64,978 行） |
| `alt_paper_fingerprint/` | alt 族 28 因子「纸面指纹」批量体检（口径照抄 `limit_pos/tradable_p0`；28 个仅 1 个通过） |
| `clean_label_alt/` | 可交易掩码修标签后，28 个 alt 因子 gain 重测 |
| `signal_mktcap_drift/` | 入选信号市值漂移（月/年）追踪 |
| `structured_evidence_recalc/` | qcut 分箱坍缩扫描 |
| `kronos_smoke/`、`tabpfn3_smoke/` | Kronos / TabPFN3 模型冒烟结论 |
| `portfolio_phase2_smoke/` | 组合优化 Phase 2 冒烟（`signals/`） |
| `llm_mcts_phase0/` | LLM-MCTS Phase 0 四臂冒烟（mcts/gp/gflownet/llm_oneshot + `*_realsmoke` hs300 轮；全量产物落本目录，命令见册Ⅲ 批次 9.2） |
| `tabpfn_rolling_compare_smoke/` | 表格基础模型对照 runner 的 gbdt 臂 hs300 冒烟；tabpfn/tabicl 臂待 GPU |
| `portfolio_phase2_mask_smoke/` | 掩码注入验证冒烟（ppo/qp/ew 全臂 + `run.log`） |
| `size_neut_strength/` | 市值中性化强度曲线（2026-10-09，P0）：α 0→1 年化单调递减（15.52%→13.80%），无内部最优 ⇒ 市值暴露是收益来源不是旋钮 |
| `ic_headroom/` | IC 可达上界扫描（2026-10-09）：最优线性合成 0.0785；候选扩容 150 → 0.0598（多样性过头稀释） |
| `dpp_sigma_sweep/` | DPP sigma × quality 扫描（2026-10-09）：s≥0.2 全平；DPP vs 直选最强 50 仅 +0.0014（噪声内），价值在降相关 |
| `altf_p1/` | alt 三族 20 因子首轮入库 + 增量测试（2026-10-09，registry 955→975）：news 族含独立信息（相关 0.139）但等权合成稀释 |
| `deliverable_portfolio/` | 可交付组合 集中度 × 行业约束（2026-10-09/10，**P0 落地**）：甜点 35~45 只（+1.9~2.3pp）；行业硬只数上限 −2.64pp、m≥8 才回本；脚本已固化 `scripts/evaluation/deliverable_portfolio.py` |
| `alla_rolling_macrof*/`、`alla_rolling_newsf*/`（5 臂） | altf P1 模型层 A/B 五臂（`_base` 已清、pred 保留）：macro/news 旁路特征均无增益（ΔIC −0.0007 / −0.0005）→ 见 `altf_p1/model_ablation/` |
| `news_alpha_probe/` | news 族组合层收尾（2026-10-10，**P0 路径关闭**）：可交易 IC 仅 0.0098(h1)/0.0174(h20)，融合在 ±0.3pp 噪声内 |
| `factor_pool_composition/` | 因子池体检（2026-10-10）：量价占 86.0%；非量价 28% 席位只贡献 9.5% gain |
| `altf_neu_backfill/` | alt 族回填 `panels_neu`（2026-10-10，数据面问题 #1）：门槛 strict 通过，但入选集逐位不变（死重） |
| `frontier_audit_920_reval/` | **批次 13 920 复验（2026-10-21，A1/A2 三臂全败线封）**：defv 定版信号面板（caliber 登记 `dingban_fundind`）+ 残差首轮两 CSV + overlay/stack/h60 三臂产物；结论真源其 README。 |
| `cvar_arm_920/` / `views_arm_920/` | 批次 12⑤/12④（2026-10-21）：CVaR 组合臂（不升级）与观点注入三臂（零增量）负结果归档。 |
| `alla_rolling_ortho920mf/` / `alla_rolling_ortho920fundind_lblco/` | 批次 12②/12③（2026-10-21/22）：市场状态特征（**唯一正结果 +1.44pp 建议采纳挂拍板**）与隔夜标签臂（判负）。 |
| `mf_fundind_check/` | 批次 14（2026-10-26）：mf 干净单臂补测——不达验收（+0.49pp < +1pp），mf 线收口。 |
| `sync/` | 产物同步 zip（本地保留，2026-10-27 已移出跟踪；大件可由册Ⅲ 命令重建，§四） |
| `frontier_audit_2026-10-19/` | frontier audit（2026-10-19）：主信号 IC 到 h60 反升至 0.161；残差四口径无正交信息（详见其 README） |
| `text_factor_probe/` | 论文族 + 情绪族生产口径收尾（2026-10-13）：`txt_*` IR 0.008~0.105 ≪ 0.3 门槛；融合增益经安慰剂证伪；FDR 106 项仅 1 条存活且为负 |

### 1.2.1 臂目录与探针产物（2026-10-28 补登，之前靠通配符扫过 ⇒ 等于没登记）

> 本节由 `scripts/data_tools/docs_lint.py` 规则 1 逼出来：上表用 `etf_contest5*`、
> `alla_rolling_macrof*/` 这类**通配写法**，新臂建目录时不加行也不会被发现，
> 于是 22 个目录长期处于「索引里查不到、只能靠 ls」的状态。此后**每臂一行**。
> 判定列口径同 §一 速查表（这些目录无独立 README，结论挂在其父实验的 README）。

| 目录 | 是什么 | 判定 |
|---|---|---|
| `alla_rolling_jq_base` / `alla_rolling_jq_amt` | 成交金额族 A/B 两臂（50 vs 52 特征），结论真源 `jq_amount_roll/README.md` | 判负 |
| `alla_rolling_jqs_base` / `alla_rolling_jqs_amt` | 同上两臂的 **M 频单轨快照**（只留月频，写作本章节前无任何文档提及，`jqs` 后缀在代码里零命中 ⇒ 由同批手工重跑产生） | 判负（随父实验） |
| `alla_rolling_macrof_base` / `alla_rolling_macrof_h20` | altf P1 宏观族臂（`_base` 已清，剩 pred/equity/metrics） | 判负 |
| `alla_rolling_newsf_h20` | altf P1 新闻族 h20 臂 | 判负 |
| `etf_contest5_nocap` / `etf_contest5_nofuel` / `etf_contest5_wl` | 定版信号的三个变体臂（无单只上限 / 无燃料项 / 白名单池）的信号板产物 | 工具保留 |
| `capacity_table` | 集中度 × AUM 二维容量表（`scripts/evaluation/capacity_table.py`） | 工具保留 |
| `barra_cne6` | CNE6 风格回归产物（因子收益 / t / 残差面板 / sigma 诊断） | 工具保留 |
| `chip_behavior_probe_2026-10-27` | 行为金融「筹码/换手」廉价探针（全 A 面板 + 三口径 IC） | 判负 |
| `network_probe_2026-10-27` | 跨股票网络中心度 P1 探针 | 判负 |
| `cohort_mom_probe` | 建仓后分组动量探针（`group_ic*` / `gate.parquet`） | 判负 |
| `hp_search_ic` | 超参扫描（**只比可交易 IC** 的收口证据，`scripts/evaluation/hp_search_ic.py`） | 采纳 |
| `retrain_freq_ic` | 重训频率对照（逐日/周/月/季） | 采纳 |
| `intraday_half_life_probe` | 日内因子半衰期单表探针 | 归档 |
| `member_blend12` | stacking 成员 12 选融合的指标快照 | 判负 |
| `text_factor_probe_smoke` | `text_factor_probe` 的冒烟子集 | 判负（随父实验） |

> ⚠️ `alla_rolling_mixed/` `alla_rolling_nofund/` 已移除（2026-09-16，结论见 §三.1/§三.2）。
> 其 `_base/` 会因 `rolling_grid_alla.py --stage all` 重算而"复活"（四臂字节相同，各 446MB）；
> 无生产代码读取，重跑约 1 分钟/臂。
>
> 二轮清理（同日）已删归档脚本产物、HS300 时代 e2e 族与演示产物；结论见 §三.13~17。

### 1.3 文档型报告（2026-09-11 按功能归类）

| 位置 | 内容 |
|---|---|
| `docs/design/yuriquant_project_overview/` | 项目总览（自包含 HTML） |
| `docs/design/yuriquant_model_layer_design/` | 模型层蓝图（`model/__init__.py`、`walk_forward_model.py`、`tests/test_model_layer.py` docstring 引用） |
| `docs/design/yuriquant_training_discipline/` | 训练纪律 |
| `docs/research_notes/文本选股研报解读与数据源调研.md` | 研报解读与数据源调研笔记（2026-08-18） |
| `docs/另类数据管道_数据源说明.md` | 另类数据管道 7 源落地实况 + PIT 对齐规则（§四，TODO 引用） |
| `docs/research_notes/另类数据调研三部曲_2026-09-20.md` | 另类数据调研合并册（方向 → 全景分类 → 现成源实测；2026-09-24 由 3 篇合并，原文件快照见 §四） |
| `docs/research_notes/短板匹配与微信文章研读_2026-09-22.md` | 短板盘点×前沿方案匹配 + 微信文章逐篇研读合并册（2026-09-24 由 2 篇合并） |
| `docs/research_notes/`（其余 12 篇） | 研报研读笔记（一篇一报，被 TODO_ARCHIVE 册Ⅱ 指针引用，**不做合并**）+ P1 速读批/CNEquity 调研 |
| `docs/consistency_checks/` | **研报口径核对 & 公平性核对报告**（2026-09-16 新建）：`AI97_预算与复现一致性核对.md`、`口径核对_AI39_多因子10.md`、`gflownet_im_faircheck.md` + 对应三臂原始日志 |
| `archive/ml-synthesis-hs300-report/` | 已归档的 ML 合成交付报告（根 README 能力表引用） |

### 1.4 根部散件（**接口文件，勿移动**）

> 这 4 个文件的位置被代码/路径约定**硬编码**，移动到子目录会静默断链（详见 §1.6 判定表）。
> 它们就是根目录「允许存在的散件」全集——除此之外根目录**只应有 `README.md`**。
> 2026-09-20 复核：根目录已收敛为下列 5 项，此前两处散件已归位（`prod_pipeline_gap.stdout.log` →
> `prod_pipeline_gap/run_stdout.log`、`_tmp_alla_daily_auto_0916_rerun.log` → `alla_daily/` 下同名日志）。

| 文件 | 说明 | 硬引用点 |
|---|---|---|
| `experiments.csv` | 实验台账 | `research/experiments.py:35`（`DEFAULT_PATH`）、`research/report_pipeline.py:141,341` |
| `all_a_pit_union_2019.json` | 全A PIT 池代码清单 | textmining `build_*_samples.py` 按 `reports/{pool}_pit_union_2019.json` 路径约定读取 |
| `intraday_stat_checkup.csv` | 分钟统计因子体检报告（结论录于 TODO_ARCHIVE 册Ⅰ §一） | `scripts/common/e2e_common.py:163`、`scripts/factors/build_intraday_stat_factors.py:271` |
| `mine_im_combos.csv` | 分钟特征 GP 组合挖掘结果 | `scripts/factors/mine_im_combos.py:124` |

### 1.5 日志与内部工作区

| 位置 | 说明 |
|---|---|
| `logs/` | **不属于任何单一实验**的跨主题运行日志（如 `mine_im_combos_run.log`）。属于某实验的日志一律放该实验目录内 |
| ~~`_pdf_dump/`~~ | 研报 PDF 文字提取中间产物，2026-09-23 已删（17 个 txt，md5 清单见 §四）。今后提取件落 `_tmp/` 当天清，勿常驻 |
| ~~`_tmp/`~~ | 当日临时工作区，2026-09-23 已删（含仓库根 `_tmp/` 共 54 个文件）；再出现则会话结束整体清掉 |

### 1.6 归类约定（**长期规则，新产出必须遵守**）

> 原则：**根目录只放接口文件和 README**。任何新产出，落盘前先按下面的表选目录。

#### 落盘判定表（按"这是什么"选，不按"谁跑的"选）

| 产出类型 | 去处 | 命名 |
|---|---|---|
| 单次实验的完整产物（回测/评估/合成） | `reports/<实验名>/` | 目录名 = 脚本名去 `run_`/`_run`；同族实验合并为父子目录（如 `e2e_p0_intraday/daily/`、`limit_pos/tradable_p0/`） |
| 该实验的运行日志 | **同实验目录内**，不另建 `logs/` | `<脚本名>_run.log`；多臂加后缀 `_placebo.log` |
| 口径核对 / 一致性核对 / 公平性核对报告（.md） | `docs/consistency_checks/` | `<标的>_<核对主题>.md`，如 `AI97_预算与复现一致性核对.md` |
| 设计文档 / 蓝图 / 方法论 | `docs/design/<主题>/` | — |
| 研报解读、数据源调研笔记 | `docs/research_notes/` | — |
| 研报 PDF 的文字提取中间产物 | `_tmp/`（可重生成，勿当作结论引用，**当天清**） | 产出工具 `scripts/data_tools/dump_pdf_text.py`（输出路径显式传入，无硬编码默认目录） |
| 跨实验的台账 / 全局报告 | 根部（**仅此 4 个，见 §1.4**） | 不改名、不移动 |
| 不属于任何单一实验的日志 | `logs/` | — |
| 已归档、不再重跑的交付报告 | `archive/` | — |
| 当日临时的 pytest 输出 / smoke JSON / 探针日志 | **`_tmp/`，会话结束后删除** | 以 `_` 开头 |

#### 三条硬规则

1. **新实验自建目录**：跑一个新实验前先 `mkdir reports/<实验名>/`，所有产物（含日志）进该目录。
   **不要**把 `xxx_run.log` 丢在根目录——这是 2026-09-16 之前根目录 38 个散件的主因。
2. **临时件必须带 `_` 前缀**且只放在 `_tmp/`。当天调试产生的 `_pytest_*.log` / `_smoke_*.json`
   **当天清**，不要留在根目录过夜。落根目录的 `_` 文件视为待清理。
3. **移动前先查引用**：任何 `reports/` 下文件的移动，先执行：
   ```bash
   grep -rn "<文件名>" --include=*.py --include=*.md --include=*.yaml .
   ```
   有命中 → 要么保持原位（推荐，符合 §1.4），要么**同步改代码并跑相关测试**。
   `reports/` 整体 gitignored，移动/删除**不可经 git 恢复**——动之前先 `cp` 快照到仓库外。

#### 定期体检（建议每月一次）

```bash
# 1) 根文件应只有 5 行（README + §1.4 四个接口文件），多于 5 行说明有散件未归类
python -c "import os;r='reports';print('\n'.join(sorted(f for f in os.listdir(r) if os.path.isfile(os.path.join(r,f)))))"

# 2) 各臂 _base 重复检测（各约 446MB；已归档的臂应删掉其 _base，见 §1.2 注）
python -c "
import hashlib,os,pathlib
def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        while (b:=f.read(1<<20)): h.update(b)
    return h.hexdigest()
base=pathlib.Path('reports'); seen={}
for d in sorted(base.iterdir()):
    p=d/'_base'/'tradable_mask.parquet'
    if p.exists(): seen.setdefault(md5(p),[]).append(d.name)
for k,v in seen.items():
    if len(v)>1: print('重复 _base:', v)
"
```
后者若打印出多个臂名，说明那些臂的 `_base` 内容相同（**必然如此**，因为 prep 与 `--preproc` 无关）；
已归档不再使用的臂可删其 `_base/` 省空间，活跃臂请保留。

---

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
    Sharpe 2.54）；`scripts/archive/select_stocks.py` 可重生成。
11. **`textmining_txt/`** —— 文本挖掘早期 txt 堆放处，已被 `textmining/` 分层布局取代，无引用。
12. **`models_mock/`** —— `walk_forward_model.py --mock` 模型台账，按需重生成。
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
    中性化（已并入 investment_report）。重跑见 `scripts/archive/` 同名脚本；
    `daily_hs300.parquet` 2026-08-26 停更，复现仅限该缓存窗口。
18. **`intraday_stat_checkup_run.log`** —— 手动控制台重定向日志，结论在同根 `.csv`，无独立价值。

---

## 四、清理记录

- **2026-10-28（文档规范化的第一批清理：旧待办副本 + 被取代的冒烟报告）**：
  8 + 2 文件 / 约 350KB，全部走 `scripts/oneoff/_recycle_batch.py` 回收站。
  快照（路径 + md5 + 原件副本）在 `~/yuriquant_snapshots/2026-10-28_docs_cleanup/`。
  - **旧待办 6 份**：`_tmp/split169/{TODO,RESEARCH_TODO,GOOD_MACHINE_TASKS,TODO_merged,
    RESEARCH_TODO_merged}.md` 与 `_tmp/TODO.staged.md`（10-13 中间稿）。起因：09-23
    拆分待办时的中间副本、10-27 三文件整合后已全部逐字并进 `TODO_ARCHIVE.md` 三册，
    官方快照另存 `2026-10-27_todo_consolidation/`；这 6 份是**同一批内容的更早版本**，
    留着会让「待办唯一真源」这条失效。空目录 `_tmp/split169/` 一并移除。
  - **旧会话计划 2 份**：`.zcode/plans/plan-sess_{0b2456f9,96502b56}*.md`（09-11 的
    「三项任务实施方案」与「每日选股排名实现计划」）。两份计划均已落地（前者对应
    指标 t 统计量与年限，后者对应 `alla_daily_rank.py`），保留那份 10-27 的
    `plan-session_0fb6435e*.md`（当次整合的唯一记录）。
  - **被取代的冒烟报告 2 份**：`galaxy_l1/smoke_report.md` 与
    `portfolio_ppo_phase1/smoke_report.md`。判定依据 = **同目录正式报告已含其全部信息
    且数字更新**：`galaxy_l1/report.md` 覆盖 2024+2025 两年（smoke 只有 2024 单年，
    判定段结论方向相同但数值是旧的）；`portfolio_ppo_phase1/report.md` 是四臂全量
    （smoke 只有 ppo 单臂 12 个决策日）。两者都零引用、无脚本生成、README 未提及。
  - **刻意未动**：`cheap_lever_scan/README_prefix_execfix_20261019.md`（执行价修复前
    旧版 README，与现行版仅差 6 行 exec-fix 警示块 ⇒ 属**存在信息损失**，等你拍板）；
    `model_portfolio/equity_ens_h1h5_pre_{open,vwap}.csv`（pre=税前 / net=税后，
    是**口径对照**不是版本过旧）；`reports/daily_determinism/report.md` 与
    `report_BC.md`（分别记 A vs B、B vs C 两轮，非重复）；`reports/etf_rotation/`
    （被 4 处引用，含设计文档「现状基线」）。


- **2026-10-27（作废的初赛方案目录 `reports/etf_contest/`）**：15 文件 / 0.5MB，
  含 20 日动量 Top3 方案的回测产物（`contest_summary.csv`、`contest_k5_*`、`board.html`、
  当日 `signals.csv`/`orders_*.csv`、`meta.json`、`run_config*.json`）。删除起因：该方案
  （20 日动量、周频、Top3、等权）已作废，profile `contest` 同步从代码移除（提交 `77bccbf`）。
  快照清单（path+size+md5）在 `%USERPROFILE%\etf_contest_archive_snapshot_20261027.md5`；
  引用扫描命中 5 处（`reports/etf_contest5/README.md` §一 与开篇、`reports/etf_champ/README.md`、
  设计文档 §P5、`scripts/portfolio/etf_contest_spec.py` docstring），已全部改为指向
  `reports/etf_contest5*` 或标注"已删除"；目录走 `_recycle_batch.py` 回收站（可从回收站恢复）。
  `etf_contest_probe.py --specs base|k5` 仍可复跑并重建同名目录。

- **2026-10-26（jq 成交金额族 A/B 两臂 `_base/`）**：`reports/alla_rolling_jq_base/_base/`
  与 `.../jq_amt/_base/` 共 28 文件 / **903.1MB**，是 prep 的确定性产物，两臂逐字节
  相同且不被任何分析产物引用（预测/回测用共享 `reports/alla_rolling/_base`；执行价面板
  只在本臂 `--stage backtest` 内读，重跑 prep 65s 可重建）。快照清单（path+size+md5）
  在 `%USERPROFILE%\_yq_backup\20261026_jq_amount_roll_base\manifest.csv`；引用扫描
  `alla_rolling_jq|jq_base|jq_amt` 仅命中 `TODO.md`（描述性文字，无路径依赖）；
  删除走 `_recycle_batch.py` 回收站（未另存副本，可再生）。

- **2026-09-23（二）全项目缓存/暂存清理**：目标 `scripts/` 之外还有没有冗余，扫完全项目后执行。
  - **删 venv 外 16 个 `__pycache__/`**（222 个 `.pyc`）+ `.pytest_cache` + `.ruff_cache`
    （`tests/ factor/ factor\{gflownet,rl}/ research/ data/ data\{altdata,textmining}/ monitoring/
    model/ optimize/ stats/ backtest/ strategy/ config/ _tmp/`）——可再生缓存，零引用风险。
  - **删 `_tmp/` 的 pytest 暂存**（`pytest_tmp*` 5 个目录，含 158 个空目录 + ~74MB parquet/csv/npy
    夹具）与 `gtja_repro` 3 个空 `.log`；`_tmp` 1791 → 114 文件、124.2 → 50.3MB。venv 内空目录
    已无残留（venv 外 0）。
  - **刻意未动**：`_tmp/hf_weights/`（Kronos 权重，`_smoke_kronos.py` 引用）、`_tmp/Kronos/`
    （smoke 脚本 sys.path 依赖的克隆）、`_tmp/fl_6ff.py`（66KB 因子库模块草稿，无跟踪版本，
    唯一副本）、全部 `.log`（daily_rank/kronos 运行证据，总量 <1MB 不值一动）、fadt 语料
    （`train_fadt_bert.py` 等 3 脚本引用）、`alla_rolling_mixed/nofund` 之外的所有 `_base/`。
  - **留待决策（未删）**：`reports/alla_rolling_ortho/_base/`（446.5MB）与主臂
    `alla_rolling/_base/` **字节级完全相同**（md5 已核），但 README 定 ortho 为「管线默认
    （最终臂）」而非已归档臂，按 §1.2「活跃臂请保留」规则**未自动删**；若确认可删，
    省 446.5MB（prep 会按确定性重建）。
  - **快照**：1992 文件，含 size+md5（清单文件已清理）；
    删除件副本 `%USERPROFILE%\_yq_backup\20260923_project_cleanup\`。

- **2026-10-19（脚本归档 + 索引）**：`scripts/` 去重收尾。
  - **归档 26 个入口**（`git mv` 到 `scripts/archive/`，历史保留）：`ai97_pbo_report`、
    `pv_marginal_quality`、`tabfm_head_{study,portfolio}`、`phase2_ppo_diag`、
    `run_gflownet_phase0`、`factor_correlation`、`risk_decomposition_report`、
    3 个 `cheap_lever/impact_cost/multihorizon` 探针、3 个退市回补、
    `build_alla_e4_interaction_factors`、`refetch_status_all_a`、
    2 个 `fetch_textmining/gflownet_library_ingest`、2 个 data_tools、
    6 个 textmining 数据准备脚本——均为 tracked 引用全零、无文档/产物引用。
  - **`e2e_common` 迁出公共层**：`scripts/common/e2e_common.py` →
    `scripts/archive/e2e_common.py`（HS300 停更缓存链路；20 处 import 同步）。
  - **新增 `scripts/README.md`**：155 个活跃入口分组索引 + 归档映射；
    `scripts/data_tools/gen_scripts_index.py --check` 进 CI。
  - **未归档**：textmining 7 个（只被 gitignored reports README 记录）、退市/财务
    回补与数据工具（仍有人工流程）——逐条确认后再动，避免断现网操作。

- **2026-09-24（docs 合并 + 冒烟去重）**：
  - **research_notes 17 → 14 篇**：两组同主题合并——`另类数据调研三部曲_2026-09-20.md`
    （方向调研/全景分类/现成源实测对照三篇，演进链同日成册）与
    `短板匹配与微信文章研读_2026-09-22.md`（短板盘点为锚 + 微信文章逐篇为应用）。
    内容**逐字保留**仅加分隔导航；9 篇研报研读一篇一报被 TODO_ARCHIVE 册Ⅱ 指针引用，
    **刻意不合**。引用同步：CNEquity 调研、design/另类数据P0_实施计划、管道_数据源说明
    共 3 文件改指合并后文件名。
  - **冒烟去重**：`llm_mcts_phase0/` 删 3 个调试中途轮（断链修复前/中的 superseded 运行），
    留四臂最终轮 + realsmoke；`llm_mcts_phase0_realsmoke/` 并入 `llm_mcts_phase0/`
    （`*_realsmoke` 后缀）；根部散件 `portfolio_phase2_mask_smoke.log` 归位为该目录 `run.log`。
  - **快照**：17 文件 / 148.1KB，含 md5（清单文件已清理）；
    删除均走 `_recycle_batch.py` 回收站。根部文件复核恰 5 项。
- **2026-09-11**：
  - 删除 §三所列 12 项（约 2.4GB，其中 alla_rolling_mixed / nofund 占 98%）。
  - 新建 `docs/design/`、`docs/research_notes/`、`archive/` 并移入文档型报告；
    同步更新根 README 两处路径与 3 处代码 docstring。
  - 刻意**未动**：`_tmp/`（当日另一会话的工作日志与备份）、`textmining/`（他会话
    开发中，8 个脚本未提交）、§1.2 全部目录（有读取方或支撑归档脚本可复现性）。
- **2026-09-11 二轮**：再删 9 目录 + 1 散件（`diagnose` `diagnose_neu` `screening`
  `ml_algorithm_compare` `e2e_backtest` `e2e_picks` `optimize_e2e` `signals`
  `multi_period` + `intraday_stat_checkup_run.log`，共约 14MB——本轮目标是目录数
  精简，结论均已先录 §三。二轮后本目录保留 23 项）。`_tmp` 与 `textmining` 当时有
  会话实时写入（17:39 仍有新日志），继续未动。
- **2026-09-16 散件治理（根目录 38 → 5）**：本轮**只做 A 档（纯归类 + 清临时件），不碰代码**。
  - **回收站删除 22 个临时件**（当天 pytest/smoke/探针残留，共 61.6KB，零引用）：
    `_pytest_*.log`×7、`_smoke_*.{json,log}`×11、`alphapool_real_smoke_{prevfix,vwapfix}.{json,log}`×4。
    删除前已 `cp` 全量快照到仓库外（38 个根文件；该快照目录已清理）。
  - **新建 `docs/consistency_checks/`**，移入 3 份核对报告 + 3 个 GFlowNet 公平性核对日志。
  - **7 个运行日志归位到所属实验目录**（不再堆根目录）：`e2e_p0_intraday_daily*`→其目录、
    `p0_ab_w*`→对应窗口目录、`mine_im_combos_run.log`→新建 `logs/`。
  - **刻意未动**：§1.4 四个接口文件（有硬编码引用路径，移动会静默断链）；
    `scripts/oneoff/`（67 个脚本，按约定本就该乱，gitignored，另案处理）。
  - **验证**：`pytest tests/test_experiments.py tests/test_report_pipeline.py -q` → **17 passed**，
    确认未被移动破坏引用。
  - **新增 §1.6「归类约定」**——把本轮做法固化为长期规则（落盘判定表 + 三条硬规则 + 月度体检命令）。
- **清理工具**：`scripts/oneoff/_recycle_batch.py`——走 `SHFileOperationW(FOF_ALLOWUNDO)` 送回收站，
  避免永久删除；分批 ≤10 个。后续清理可复用。
- **2026-09-16 二轮（残留臂目录）**：移除 `alla_rolling_mixed/` 与 `alla_rolling_nofund/`
  **共 892.3 MB**（各含一个 446.2MB 的 `_base/`）。
  - **性质澄清**：这不是 09-11 的漏网文件，而是 `stage_prep()` 每次必然重建的缓存
    （见 §1.2 注）。四臂 `_base` 实测**字节级完全相同**，09-15 曾被四臂依次重建一遍。
  - **安全性**：全仓无生产代码读这两个目录；唯一读方 `scripts/oneoff/compare_three_arms.py`
    读的 `mixed/metrics_overall.csv` 早已不存在（该脚本本身也已随本轮 oneoff 清理移除）。
    需要时 `--stage all` 自动重建，**能力无损**。
  - **存档**：md5 清单（清单文件已清理；删除件副本在 `%USERPROFILE%\_yq_backup\`）。
  - **验证**：`pytest --collect-only` → **883**（与治理前一致）；全仓引用扫描确认无残留。
- **2026-09-23 一次性中间产物清理 + 同族目录合并**（目标：顶层目录 62→53）：
  - **删除 93 个一次性文件、共 1026.9 MB**（删除前逐项登记路径+大小+MD5，
    清单文件已清理）：
    ① **20 个 pred parquet**（`alla_arms` 四臂 8 + `holder_dyn_ablation` 4 + `holder_dyn_forced` 4
    + `limit_pos_ablation_p1b` 4，各约 52–55MB）——回测/集成早已结算，结论都在同目录
    `report.md` 与 metrics csv，pred 无二次读取方；
    ② **`_pdf_dump/` 全 17 个研报 txt**——`dump_pdf_text.py` 输出路径显式传入、无硬编码
    默认目录，重抓可再生产（且两篇相关研报的结论已落 `docs/research_notes/` 研读笔记）；
    ③ **仓库根目录 `_tmp/` 54 个研报分页抽取 txt**（quantaalpha / llm_mcts 两篇的
    `pages/page_*.txt` + `_dirs.txt`）。
  - **合并 6 组同族目录**（同级散目录收进父目录，1.2 表已注明合并日期）：
    `e2e_p0_intraday/{daily,no_im,with_im,w2023h2,w2025h1}` 五合一、`holder_dyn/{ablation,forced}`
    二合一、`limit_pos/{ablation_p1b,tradable_p0}` 二合一，及 `galaxy_l1`、`portfolio_ppo_phase1`、
    `fundamental_blend` 三组的带后缀散件归位。**结论性文件一个没动**（report.md / summary /
    metrics / arm_config.json 全部保留）。
  - **同步改引用**：7 个 `scripts/oneoff/` 探针脚本的路径常量、4 处生产代码注释
    （`alla_daily_rank.py`、`labels.py` 等）、TODO.md 与根 README 的路径串，以及 reports 内
    7 份 md 的旧路径链接（`alt_paper_fingerprint` / `clean_label_alt` / `alla_daily_h1h5` /
    `limit_pos/*` 等）。全仓 rg 扫描确认旧目录名（`*_tradable_p0` / `*_ablation_p1b` /
    `holder_dyn_*` / `e2e_p0_intraday_daily` 等）在 py/md/yaml 中**零残留**。
  - **验证**：根文件仍恰 5 个（README + §1.4 四接口文件）；
    `pytest tests/test_experiments.py tests/test_report_pipeline.py -q` → **17 passed**
    （与 09-16 治理后一致，确认目录移动未破坏引用）。
  - **结果**：约 1764 → **1725 个文件**；总量约 5.9GB → **4.75 GiB**（净减约 1GB，精确删除量以 manifest 的 1026.9MB 为准）。

### 2026-10-27 待办三文件整合 + 仓库跟踪清理（记录）

- **退役两个待办文件**：`RESEARCH_TODO.md` / `GOOD_MACHINE_TASKS.md` 内容逐字
  归档进根目录 `TODO_ARCHIVE.md`（册Ⅱ/册Ⅲ；册Ⅰ = 原 TODO.md 已收口部分），
  原文件 `git rm`。快照（路径+md5）存仓库外
  `~/yuriquant_snapshots/2026-10-27_todo_consolidation/`：
  `TODO.md e7321f3f…`、`RESEARCH_TODO.md e9f1f70e…`、
  `GOOD_MACHINE_TASKS.md 8a51949c…`。
- **`ai97_100k.json` 移出 git**（62KB，AI97 100k 步归因 dump，全库零引用，
  属漏入库的实验产物）→ 物理移至 `reports/ai97_100k.json`（本目录 gitignore，
  md5 `d5051d74…` 见上快照），`git rm --cached`。
- **删除 `.github/workflows/`**（ci.yml + full-tests.yml，均手动触发、2026-09-10
  起 push 静音）：用户拍板 CI 能力归零；`uv.lock` 同步守卫改本机
  `uv lock --check`（README 工程质量节已改写）。
- **引用同步**：全仓 54+ 文件（代码 docstring / config/baselines.yaml 的行号级
  引用 / README / AGENTS / reports 内 20+ md）统一改指 `TODO_ARCHIVE.md`
  「册 + 节名」锚点；两处行号级硬引用（`baselines.yaml:110`、
  `neutralize_vectorization/report.md:5`）改章节锚，不再依赖行号。
  全仓扫描旧文件名在 TODO_ARCHIVE 之外零残留。
- **`uv.lock` 移出跟踪**（800KB，跟踪文件中最大）：工作流全用系统解释器 + pip install -e，
  CI 删除后零消费方（schedule.yaml 亦明确"系统 Python 不受 uv sync 影响"）→
  本地保留并 gitignore，README 工程质量节改写；需要精确环境时 `uv lock` 重生成。
- **`reports/sync/` 两个 zip 移出跟踪**（batch1213 304KB / batch14）：曾由用户指示
  force-add，2026-10-27 应"尽量精简"要求 `git rm --cached`（本地保留、可由
  TODO_ARCHIVE 册Ⅲ 存档命令重建）；reports/ 下跟踪项仅剩本 README。
