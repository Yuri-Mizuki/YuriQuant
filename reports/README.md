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
| `monitoring/` | 因子库/模型预测监控台账 + 报告。**⚠️ 旧链 2026-09-21 退休、计划任务已 `/delete`**（数据集行情源 08-26 停更）：`snapshots.csv`/`alerts.csv`/`monitor_report.html` 停在 hs300 时代；`production_ic_daily.csv` 是**现役**生产口径每日 IC 台账。停用决策证据留档：`legacy_monitor_hs300_run_20260921.log`（旧链直跑）/ `legacy_monitor_alla_probe_20260921.log`（改指全A 试跑，920 因子被跳过）/ `legacy_task_xml_20260921.xml`（任务条件 XML） | `scripts/reporting/monitor_performance.py`（退休，手动可跑）+ `scripts/reporting/monitor_production_ic.py`（现役） |
| `models/` | 模型注册台账（旧监控链读取；现役链不依赖） | `scripts/evaluation/walk_forward_model.py` |
| `textmining/` | 文本因子管线（**活跃开发中**，结构见其自带 README） | `scripts/textmining/*` |

### 1.2 保留实验与报告（有代码读取、或支撑已归档脚本引用的结论；均可重跑复现）

| 目录 | 说明 |
|---|---|
| `cpcv_h1/` | h=1 CPCV 无偏评估（2026-09-11 重生成） |
| `freq_tune/` | 调仓频率精修（含引擎修复前存档 `freq_tune_prefix_engine_fix_20260825.csv`） |
| `multiyear/` | 多年度 OOS 稳健性（2026-09-01 修复后重跑） |
| `buffer_tune/` | 换手缓冲带调参 |
| `gtja_repro/` | 国金因子复现对照（读自身 `baseline.csv`） |
| `etf_rotation/` | ⚠️ legacy ETF 轮动报告（2026-08-26，15 只池 + 周频合成动量，无 PIT 掩码）；仅历史追溯，**被 `etf_rotation_probe/` 取代** |
| `etf_rotation_probe/` | **ETF 轮动 P1 探针（2026-10-26，现行）**：41 只跨资产池 + PIT 流动性掩码，13 变体主表 + 7 变体稳健网格；基线 `dual_ivvt15_m5` 年化 7.14% / Sharpe 0.62 / MDD 19.01%（结论真源其 README）；脚本 `scripts/portfolio/etf_rotation_probe.py` |
| `etf_rotation_p2/` | **ETF 轮动 P2 前置（2026-10-26）**：防御腿 + 份额 PIT 拥挤度 + 趋势 R² 三项对照；新基线候选 `def_mix_crowd_f20` 年化 11.23% / Sharpe 0.85 / MDD 16.70%，邻域 0.76–0.94（脚本 `scripts/portfolio/etf_rotation_p2.py`） |
| `etf_{champ,contest5,contest5_opt,contest5_wl,contest_wl,contest_wl2,p4,dedup,capacity,pool_expand,survivorship_probe,alpha_probe}/` | **ETF 轮动赛事与长线定版全家桶（2026-10-26~27）**：比赛定版「三锋 v2」一页纸真源 = `etf_contest5/README.md` §十三（参数真源 `etf_contest_spec.py`）；长线「磐石」= `etf_champ/`；`contest5_opt/` 含定版对照/稳健/反转三档/v2 采纳证据，`contest_wl*/` 白名单复核，`p4/` 样本外，`dedup/capacity/pool_expand/survivorship_probe/alpha_probe/` 各专项探针；作废的初赛目录 `etf_contest/` 已回收（§四 10-27） |
| `risk_decomposition/` | 组合风险分解报告（Euler 分解 / 因子方差贡献 / VaR 成分） |
| `ml_synthesis_h1/` | ⚠️ 虽属归档时代，但 `monitor_performance.py` 读其预测面板，删除会破坏监控 |
| `investment_report/` | HS300 时代投资收益报告（最终交付物存档；数字有 ~3% 年化口径偏移待重跑，TODO_ARCHIVE 册Ⅰ §一第 3 条；归档脚本 `diagnose_factor_vs_model` 以其为输入） |
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
| `llm_mcts_phase0/` | 东吴 LLM-MCTS Phase 0 四臂冒烟（2026-09-24：mcts/gp/gflownet/llm_oneshot 各留最终一轮 + `*_realsmoke` 真实 hs300 离线缓存轮；全量产物将来落本目录，命令见 TODO_ARCHIVE 册Ⅲ 批次 9.2） |
| `tabpfn_rolling_compare_smoke/` | 批次 8 表格基础模型对照 runner 的 gbdt 臂 hs300 冒烟（2026-09-24；tabpfn/tabicl 臂全量待 GPU） |
| `portfolio_phase2_mask_smoke/` | 批次 5 前置掩码注入验证冒烟（2026-09-24，ppo/qp/ew 全臂 + `run.log`） |
| `size_neut_strength/` | 市值中性化强度曲线（2026-10-09，P0）：α 0→1 年化单调递减（15.52%→13.80%），无内部最优 ⇒ 市值暴露是收益来源不是旋钮 |
| `ic_headroom/` | IC 可达上界扫描（2026-10-09）：最优线性合成 0.0785；候选扩容 150 → 0.0598（多样性过头稀释） |
| `dpp_sigma_sweep/` | DPP sigma × quality 扫描（2026-10-09）：s≥0.2 全平；DPP vs 直选最强 50 仅 +0.0014（噪声内），价值在降相关 |
| `altf_p1/` | alt 三族 20 因子首轮入库 + 增量测试（2026-10-09，registry 955→975）：news 族含独立信息（相关 0.139）但等权合成稀释 |
| `deliverable_portfolio/` | 可交付组合 集中度 × 行业约束（2026-10-09/10，**P0 落地**）：甜点 35~45 只（+1.9~2.3pp）；行业硬只数上限 −2.64pp、m≥8 才回本；脚本已固化 `scripts/evaluation/deliverable_portfolio.py` |
| `alla_rolling_macrof*/`、`alla_rolling_newsf*/`（5 臂） | altf P1 模型层 A/B 临时臂（2026-10-10；A=`macrof_base`、B1=`macrof`、B2=`newsf`/`newsf_h20`；`_base` 已清理、pred/metrics 保留）。**结论：macro/news 旁路特征均无增益**（ΔIC −0.0007 / −0.0005，pred 秩相关 0.995+）→ 见 `altf_p1/model_ablation/` |
| `news_alpha_probe/` | news 族**验真伪 + 组合层融合**（2026-10-10，P0 路径**关闭**）：news-only 可交易 IC 仅 0.0098(h1)/0.0174(h20) = 主模型 1/8~1/12，集中持仓为负（TopK45 −8.6%/−6.4%）；融合增益 h1 在 ±0.3pp 噪声内、h20 系统性负 → 脚本 `scripts/evaluation/news_alpha_probe.py` |
| `factor_pool_composition/` | 因子池构成体检（2026-10-10）：**三层占比** 池 84% → 入选 72% → 归因 gain 90.5%（非量价拿 28% 席位只贡献 9.5%）；「83.6%」= 旧 955 口径、今 `panels_neu` 实为 86.0%；查出 **alt 20 因子未进 panels_neu** + 26 无面板 + 26 哈希未登记 → 脚本 `scripts/evaluation/factor_pool_composition.py` |
| `altf_neu_backfill/` | alt 族 `panels_neu` 回填（2026-10-10，**修数据面问题 #1**）：family-aware（零膨胀→跳过 MAD / 广播→时序 z）+ `bonus_freq_3y`；门槛 **strict 通过** ⇒ 不再需要 `YURIQUANT_PANEL_CHECK=warn`。A/B 实测**入选集逐位不变**（alt 不在 `ALT_FAMILY_SETS`、窗内 |IC| 排名 783–948 ⇒ 死重）→ 脚本 `scripts/builders/build_alla_altfactor_neu.py` |
| `frontier_audit_920_reval/` | **批次 13 920 复验（2026-10-21，A1/A2 三臂全败线封）**：defv 定版信号面板（caliber 登记 `dingban_fundind`）+ 残差首轮两 CSV + overlay/stack/h60 三臂产物；结论真源其 README。 |
| `cvar_arm_920/` / `views_arm_920/` | 批次 12⑤/12④（2026-10-21）：CVaR 组合臂（不升级）与观点注入三臂（零增量）负结果归档。 |
| `alla_rolling_ortho920mf/` / `alla_rolling_ortho920fundind_lblco/` | 批次 12②/12③（2026-10-21/22）：市场状态特征（**唯一正结果 +1.44pp 建议采纳挂拍板**）与隔夜标签臂（判负）。 |
| `mf_fundind_check/` | 批次 14（2026-10-26）：mf 干净单臂补测——不达验收（+0.49pp < +1pp），mf 线收口。 |
| `sync/` | **产物远端同步捆绑**（`batch1213_products_20261022.zip` / `batch14_products_20261026.zip`；大件 parquet 不入 zip、由 TODO_ARCHIVE 册Ⅲ 存档命令可重建）——曾由用户指示 force-add 入库，**2026-10-27 应"尽量精简"要求移出跟踪**（本地保留，§四 有记录）。 |
| `frontier_audit_2026-10-19/` | **增量方向 frontier audit（2026-10-19，A1/A2）**：`horizon_probe` 主信号 IC h1–h120 衰减（h60 反而升到 0.161）+ `residual_ic_probe` 920 因子对主信号残差的四口径 IC；脚本 `scripts/evaluation/{horizon_probe,residual_ic_probe}.py`，结论回写 TODO_ARCHIVE 册Ⅰ §六。 |
| `text_factor_probe/` | 论文族 `pap_*`(21) + 情绪族 `txt_*`(5) **生产口径收尾（2026-10-13，P0）**：口径修正——面板实体**存在**（哈希名在 `panels/`），缺的是未进 `panels_neu` ⇒ 从未中性化/竞争；`txt_*` 5 个自身可交易 IR **0.008~0.105 ≪ 0.3 门槛**、独立 Sharpe 0.47~0.57 < 主 0.631；**融合层「Sharpe +0.05~0.12」经安慰剂证伪**（值置换复现中位 71% ⇒ 覆盖率掩码/市值暴露，非 alpha）；`pap_*` 无一条过「正交+够强」；106 项 BH-FDR **仅 1 条存活且为负** → 脚本 `scripts/evaluation/text_factor_probe.py` + `fuse_placebo.py` |

> ⚠️ **`alla_rolling_mixed/` 与 `alla_rolling_nofund/` 已彻底移除（2026-09-16）**。
> 这两个臂的**结论早已归档**（见 §三.1 / §三.2），但它们的 `_base/` 缓存曾长期残留——
> **不是漏网，是设计使然**：`rolling_grid_alla.py` 的 `stage_prep()`（第 232 行）每次执行
> 都把同一份数据源缓存**重算一遍写进各臂自己的 `_base/`**，不共享。故 09-11 删臂后，
> 只要有人再跑一轮 `--stage all`，`_base/` 就会"复活"。实测四臂 `_base` **字节级完全相同**
> （md5 一致，各 446.2MB），09-15 14:58~15:02 曾被四臂依次重建一遍。
> 移除时登记过 md5 清单（清单文件已清理，删除件副本在 `%USERPROFILE%\_yq_backup\`），
> **无任何生产代码读取它们**；如需重跑该臂，`--stage all` 会自动重建（约 1 分钟/臂）。

> **二轮清理（同日）已删**：归档脚本产物（`diagnose/` `diagnose_neu/` `screening/`
> `ml_algorithm_compare/`）、HS300 时代 e2e 族（`e2e_backtest/` `e2e_picks/`
> `optimize_e2e/`）与演示产物（`multi_period/` `signals/`）——结论见 §三.13~17，
> 重跑对应脚本即可复现。

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
>
> **2026-09-20 复核**：根目录已收敛为下列 5 项。此前两处根目录散件已归位（内容不丢）：
> `prod_pipeline_gap.stdout.log` → `prod_pipeline_gap/run_stdout.log`（该实验 2×2 的原始 stdout；
> 目录内另有成稿 `run.log`）、`_tmp_alla_daily_auto_0916_rerun.log` →
> `alla_daily/alla_daily_rank_auto_20260916_rerun.log`（09-16 19:03 降级态那轮，09-17 曾"暂留作对照证据"）。

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
| ~~`_pdf_dump/`~~ | 研报 PDF 文字提取中间产物，**已于 2026-09-23 删除**（17 个 txt，md5 清单见 §四）。`dump_pdf_text.py` 无硬编码路径，今后提取件落 `_tmp/` 当天清，勿再常驻本目录 |
| ~~`_tmp/`~~ | 当日工程会话临时工作区。**2026-09-23 已删除**（连同仓库根目录 `_tmp/` 的研报分页抽取，共 54 个文件）；今后如再出现，会话结束后整体清掉 |

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

1. **`alla_rolling_mixed/`**（约 1.2GB）——三臂对比的 mixed 臂（仅基本面族中性化）。
   结论已录 TODO_ARCHIVE 册Ⅰ 附录：全正交 +1.6pp/年 > 混合 +0.1pp ≈ 不正交；正交化增量几乎全部
   来自量价因子，40/42 个 h1 配置全正交占优。重生成：`rolling_grid_alla.py --preproc mixed`。
   *（2026-09-16 补：残留的 `_base/`（446MB）亦已移除，见 §1.2 注。）*
2. **`alla_rolling_nofund/`**（约 1.2GB）——排除基本面族的消融对照。
   结论已录 TODO_ARCHIVE 册Ⅰ 附录：基本面/股东族带来 ~+0.7pp/年一致增量（含族 +10.2% vs
   纯量价 +9.5%）。重生成：`rolling_grid_alla.py --ablation`。
   *（2026-09-16 补：残留的 `_base/`（446MB）亦已移除。）*
3. **`alla_rolling_mixed_run.log`** —— mixed 臂手动重定向日志，随臂删除。
4. **`gbdt_tune/`** —— 引擎修复前（2026-08-25）的 gbdt 网格：h=1 × frac=0.2 最优
   （成本后年化 27.0%、超额 +4.8%、Sharpe 1.70）；h=3/h=5 全负——后被证实是
   h>1 引擎 bug 的伪结果，不可引用。其「h1 + Top20% 是甜点」结论已演进为
   `model_portfolio` 固化配置（`strategy/examples.py` docstring 仍引用该实验名）。
5. **`gp_tune/`** —— GP 预算网格（旧版脚本，2026-08-17）：seed 间方差大、最优公式
   test IC 0.025（train 0.063，过半衰减）。已被重写后的 `gp_tune_budget.py` 取代，
   重跑可复现。
6. **`cpcv_eval/`** —— horizon 混选 CPCV（2026-08-24，引擎修复前）：ridge IC
   0.044±0.007、gbdt 0.050±0.004，15/15 路径为正。存在 horizon 选择偏差，
   已被 `cpcv_h1/`（无偏版）取代。
7. **`adaptive_long_validation/`** —— 自适应调仓变体（ov30–ov55）Sharpe 全部为负
   （−0.15 ~ −0.78），月频基线 0.06；阈值越敏感调仓越频、损耗越大。**不采用自适应调仓**。
8. **`adaptive_rebalance/`** —— 4 变体中 ov40 最优（Sharpe 0.84 vs 基线 0.57，
   超额 −1.0% vs −5.1%），但整体超额为负、样本仅约 2 年，**未采纳进固化配置**。
9. **`timing_vs_crosssection/`** —— 目标波动 / 状态机择时叠加均不优于纯截面
   （Sharpe 0.57 / 0.54 / 0.29，2025 年段）；与后续 `position_overlay_grid` 结论
   （TODO_ARCHIVE 册Ⅰ 附录：择时 overlay 不提升 Sharpe、只压回撤）一致。
10. **`select_hs300_2025/`** —— HS300 时代选股演示（2026-08-05，幸存者偏差修复前，
    数字虚高不可引用：rev_growth_yoy top30 年化 48%、Sharpe 2.54）。
    `scripts/archive/select_stocks.py` 随时可重生成。
11. **`textmining_txt/`** —— 文本挖掘原始 txt 早期堆放处，已被 `textmining/` 分层
    布局取代，无任何代码引用。
12. **`models_mock/`** —— `walk_forward_model.py --mock` 的 mock 模型台账，按需重生成。
13. **`diagnose/`** —— 单因子 vs 模型同口径诊断：range20 / alpha191_159 / vol60 等
    负 IC 单因子月频 top50 超额 +20%~+36%，实为高波动/高 beta 风格暴露、非预测力
    （结论已录 TODO_ARCHIVE 册Ⅰ 附录）。重跑 `scripts/archive/diagnose_factor_vs_model.py` 可复现。
14. **`diagnose_neu/`** —— 中性化前后对照：模型预测中性化后 IC 0.059→0.015 但
    超额 −17.8%→−8.9%——剥离风格暴露后模型趋近纯 alpha（与 investment_report
    「中性化是变现关键」结论同源）。重跑 `scripts/archive/diagnose_neutralized_compare.py`。
15. **`screening/`** —— AlphaEval 式筛选漏斗（hs300_2022_2025）：821 → PPS 234 →
    DPP 164 → RRE 149，max|corr| 1.000→0.978。重跑 `scripts/archive/factor_screening.py`。
16. **`ml_algorithm_compare/`** —— 9 经典特征 × 2019-2026 OOS：gbdt IC 0.051
    （NW t=5.8）> ridge 0.044（t=4.1）> TabICL 0.029（t=3.2）——**gbdt 系最优，
    表格基础模型（TabICL）无优势**。重跑 `scripts/archive/ml_algorithm_compare.py`。
17. **`e2e_backtest/`、`e2e_picks/`、`optimize_e2e/`、`signals/`、`multi_period/`**
    —— HS300 时代 e2e 链路与演示产物，历史数字带 ~3% 年化口径偏移（TODO_ARCHIVE 册Ⅰ §一
    第 3 条待重跑项）。核心结论已录 TODO_ARCHIVE 册Ⅰ 附录：等权 top50 +16.2% 跑输全池等权
    +30.8%；风格中性化 16.2%→27.0%（Sharpe 0.41→0.66）；周频调仓被证伪；
    最优=月频+中性化，已并入 investment_report。分别重跑 `e2e_backtest.py` /
    `scripts/archive/e2e_stock_picks.py` / `optimize_e2e.py` / `generate_signals.py` /
    `multi_period_backtest.py` 可复现。
    **2026-09-21**：`e2e_stock_picks.py` 与 `select_stocks.py` 已移入 `scripts/archive/`
    （理由见根 README「`scripts/archive/`」节）；重跑路径随之上移到 archive。
    注意 `e2e_common.load_daily_data` 读的 `daily_hs300.parquet` **2026-08-26 起停更**，
    上述 e2e 族"重跑可复现"仅限于该缓存覆盖的窗口。
18. **`intraday_stat_checkup_run.log`** —— 手动控制台重定向日志；体检结论与数据
    在同根 `.csv`（TODO 引用的是 csv），日志无独立价值。

---

## 四、清理记录

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
