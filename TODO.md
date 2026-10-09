# YuriQuant 待办清单

> **本文件是待办的唯一真源**（研报复现线见 [`RESEARCH_TODO.md`](RESEARCH_TODO.md)；
> 架构与完成度总览、安装与用法见 [`README.md`](README.md)）。
>
> 2026-09-21 大清理：对照已实现代码归档已完成项（另类数据 P0 / 调度收口 / 监控退休 /
> HS300 入口归档 / panels_neu 920 / holder_dyn 消融 / 中性化向量化 / stage2 Phase 0-2 管线等）。
> 清理前全文见 git 历史（`e70ac96` 为 09-20 清理前基线）；历史交付细节压缩于附录。
>
> **2026-10-09 刷新**：好机器批次 1–10 全部出数收官（GOOD_MACHINE_TASKS 已标终态），
> 主模型定版 = fundind 快信号 h1020⊕慢信号 s0.5 + buffer(15/40) 执行层
> （`reports/batch1_920_收尾对照表.md` §⑭⑯）。原 §一 P0 治本重跑归档、§五批次
> 后续立项全部收官；本文件剩项 = **定版切生产链（唯一高优先）** + 小工程收尾 +
> 资源墙后置线。

---

## 一、研究验证欠账（最高优先级——影响结论可信度）

- [ ] **定版口径切换生产出榜链**（当前唯一挡在「定版=生产」前面的事）：
  定版信号 = fundind 快信号 h1020⊕慢信号 s0.5 + **buffer(15/40)** 执行层
  （§⑭ 拍板、§⑯ 参数上修；v2 数据态 18.68%/0.813，buffer top20 口径
  18.07%/0.805/换手 19.1%）。生产链 `alla_daily_rank.py` 现状 = 标准正交特征
  自算 + 无 buffer 策略，需三处切换：① 特征口径对齐 `panels_neu_fundind`
  （基本面族只剥行业、保市值风格，= rolling_grid `--preproc ortho_fundind` 同参）；
  ② 信号合成切 h1020⊕s0.5（慢信号=标准面板 fundamental 族滚动 ICIR 线性、T=24）；
  ③ 策略类换 `BufferedTopFracLongOnly(0.15, 0.40)`——**`strategy/examples.py`
  缺省仍 (0.20, 0.30)，勿用缺省**。切换后建议双链并行对照一周再退役旧链。
- [ ] **重跑 e2e 族报告对齐年化口径**：`perf_stats` 年化 244→252 后，
  `reports/e2e_backtest/`、`reports/investment_report/` 中 e2e 链路历史数字
  与现行口径存在 ~3% 系统性偏移。已挂好机器队列末尾。
- [ ] **all_a_2018_2026 数据集纳入正式因子库监控**（目前为轻量 registry）。
  关联：`monitor_performance` 旧链复活的前提之一（另一前提 = runner 行情源参数化），
  现役生产监控已由 `monitor_production_ic` 承担，本条非阻塞。
- [ ] **方案 B 缺面板对齐**（小工程，自 §五并入）：ic_h 缓存 824 列中 4 列无对应
  panels_neu 文件（coverage 过滤与 ic 融合的口径差：alpha191_138 / alpha101_096 /
  alpha101_097 / unlock_ratio_20d），当前靠 `YURIQUANT_PANEL_CHECK=warn` 容忍；
  补齐面板或 ic 融合按 coverage 同步过滤，消除长期 warn。

### 已完成归档（一行一条）

- [x] **P0 新口径（920 面板）治本口径基线重跑（09-24~10-02 完成，三段式）**：
  v1（ortho920tl）出数即发现基本面族 99 张面板建于 backward_factor 全A修复之前
  （网格污染，§①-⑥ 降级为「量价+部分另类」口径）→ 99 面板清除重建 → v2
  （ortho920v2）完整口径定版：h1020⊕s0.5 17.87%/Sharpe 0.781 四项全面占优翻案
  → fundind 快信号 18.68%/0.813（§⑫）→ §⑭ 拍板 fundind+buffer 执行层 →
  §⑯ 网格上修 buffer(15/40)（18.07%/0.805/换手 19.1%）。
  全程 `reports/batch1_920_收尾对照表.md` §①–⑯。
- [x] 另类数据 P0 管道（09-20~21）：`data/altdata/` 传输层 7 源落地（巨潮快讯 119.2 万条
  自建历史存档 / 东财宏观日历 64,978 行含补洞 / 巨潮增减持 26.6 万行全历史等，
  见 `reports/docs/另类数据管道_数据源说明.md`）；日增量入口 `fetch_altdata_daily`
  进计划任务（18:00，`.venv` 解释器）
- [x] holder_dyn 9 因子三探针消融（09-21）：强制纳入臂 **Δ=−0.85pp 无增量价值**
  （`reports/holder_dyn/forced/`），Step 1b 全历史回补结案不推进；表保留（增量继续抓）
- [x] panels_neu 882→920 补齐（09-21）：38 缺面板补齐（含 ALT 28 + 基本面/股东 10，
  修复主实验 select 崩溃点），证据 `reports/panels_neu_backfill/` → 由此派生 P0 重跑项
- [x] 每日计划任务静默失败修复 + 调度收口（09-17/09-21）：三任务同日 `0xC000013A`
  被杀根因 = schtasks 裸建继承电池条件；新建 `config/schedule.yaml` 单一登记表 +
  `scripts/common/task_scheduler.py` 单点注册（XML 导入，显式关电池条件）+ `doctor` 体检
- [x] `YuriQuant Monitor` 旧监控链退休（09-21）：任务 `/delete`；数据集行情源 08-26 停更，
  每轮 144 critical / 235 warning 纯噪音；改指全A 实测不可行；生产监控由
  `monitor_production_ic` 承担（09-18 起）
- [x] HS300 时代选股清单入口归档（09-21）：`e2e_stock_picks` / `select_stocks` →
  `scripts/archive/`（数据口径停更、职责被 `alla_daily_rank` 取代）
- [x] 执行价模式恢复并接入主入口（09-21）：当日下线当晚自 HEAD 恢复（恢复无损：41 测试 +
  主形态 open/vwap 重算对齐留存 CSV ≤0.0001pp）；`run_model_portfolio.py` 新增 `--execution`
  {close,open,vwap}（默认 close = 历史行为逐位不变，T+1 臂 opt-in）；重建
  `_base/{open,vwap}_adj.parquet`（收益率层校验通过）；冒烟 open 8.03% / vwap 7.97% vs close 8.60%
  （2026 短样本，方向幅度与 09-15 全样本校准一致）。证据：
  `reports/docs/consistency_checks/口径核对_AI39_多因子10.md` §1.5「恢复记录」+
  6 份 metrics CSV（09-15 全样本：T+1 open 14.66 / VWAP 14.60 vs close 15.54）
- [x] 中性化向量化（09-20）：`batch` 2.96x 默认（逐位一致 max|Δ|=0）、`grouped` 8.3x opt-in
- [x] 出榜链口径对齐主实验 ortho（09-17）+ `limit_pos` 硬剔 + 申万一级行业排名进日报（09-18）
- [x] 生产化每日推理 `alla_daily_rank`（09-08，含计划任务注册）
- [x] 全A超额来源复核（09-08）：对全A等权纯选股 α=+5.1%/年，对上证超额约半数为风格敞口
- [x] 超额/夏普显著性检验进回测指标（09-08）；论文复现因子族入库（09-08）
- [x] 全A数据卫生修复（09-07）：退市股回补 238 只（幸存者偏差消除）、状态表重拉、
  ETF 列剔除、半拉日裁剪
- [x] 全A滚动训练实验（09-01）；重跑 multiyear_oos / freq_tune（09-01）：
  h1×M 唯一三年一致稳健解、h5×M 伪结果修正

---

## 二、研究功能缺口（新能力）

### 已就绪待接入 / 待全量

- [x] **实验产物接 PBO 出报告**（09-23 固化 `scripts/evaluation/pbo_report.py`）：
  rolling_grid 三网格已出（`reports/pbo_rolling_ortho/`：M_raw 0.174 / M_neut 0.227 /
  W_raw 0.000，均远低于 0.5 红线）；后续各线已全部接齐——AI97 0.704（负结果主证）、
  GFlowNet 114 因子族 0.114（稳健）、LLM-MCTS 14 因子族 0.039（稳健）。
- [x] **labels.py IR/Calmar 标签分支**（09-23，AI29 转译；`build_labels(method=)`，
  超额口径需 `bench_close_panel`）。**实验已跑完判负（10-04）**：ir 臂 IC 全线
  −52~−191×10⁻⁴、回撤中位 +1.27pp 恶化（AI29 自述代价 + Label Alchemy vol_scaled
  预警双应验）；calmar 同族分母更病态不再跑。三臂总结论（gauss_rank 零增量 /
  co 隔夜判负 / ir 判负）见 RESEARCH_TODO §一 标签臂记录。
- [ ] **分钟频挖掘第三层扩原料**：时段切片矩阵化（20 分钟窗 × 动量/波动/量）、
  量价 lead-lag、tsfresh 计算器移植、事件日条件化（与文本 PEAD 线交叉）；
  扩完重跑 mine_im_combos 看组合上限是否抬升（二层结论：增益≈0，信号在
  时段动量，原料不够）。
- [ ] **分钟频扩容（资源墙）**：all_a 池物化（~5500 码 × 48 bar，MemMap 分块性能待验）、
  1 分钟档位（存储 ×5）；先跑吞吐探针（`scripts/oneoff/_probe_minute_throughput.py`）估成本。
- [x] **另类数据因子轮次（altf 首轮，10-07 完成）**：`scripts/builders/build_alla_altfactors.py`
  三族 20 因子入 all_a_2018_2026（panels + factor_stats_altf.jsonl + registry 922→942 +
  ic_h{1,5,10,20} 融合）——增减持 7（公告日对齐 lag0，金额比流通市值）/ 宏观日历 7
  （15:00 收盘闸门 + 方向表，广播族截面 IC 按构造 NaN，1 个 ×波动率交互变体可测）/
  快讯情绪 6（cls 词典代理情绪 + 条数/加红计数；**|IC| 全场最高**：attention 反转
  h1 −0.021 → h20 −0.043）。坑与备注：① 09-23 缓存重建后 cls/宏观存档丢失，本次
  经通道回补（宏观 65,419 行全量；cls 回补至 2022-09 共 63 万条后停，快讯族覆盖
  2022-09~2026-09，每日 17:30 续传任务会继续推早水位，重跑 builder 即扩覆盖）；
  ② `altf` 尚未登记 `SET_TO_FAMILY`/`SOURCE_PREFIXES`（本轮禁改该文件），registry
  family=其他 + subfamily 区分三族，下次动 `research/factor_library.py` 时补。
  **完整判读（10-08 结案）**：席位竞争 36 轮 0 入池；慢信号家族并入（--include-alt-news）
  定版 18.68%→16.27% 双路径双败——**alt 三族对主模型无增量**（news 族 h20 反转方向
  与 macro 择时特征留待未来新信号线），证据 `reports/batch2385_结果速览.md` P-C 节。
- [x] **llm_pool 机制抽取（QuantaAlpha 转译，09-23 完成）**：① AST 结构去重
  （`structure_similarity` 子树 Jaccard ≥0.6 判同族变体——去窗口、终端占位、交换律排序，
  `_update` 求值前拦截省评估预算，`struct_dedup=False` 可关）；② 复杂度三维约束两项新校验
  `param_heavy`（自由参数占比 ≥50%，当前空间上保险性质）+ `too_many_features`
  （底层特征 >`MAX_BASE_FEATURES=6`，固定常数不随字段表扩容放松）；③ 语义一致性前置校验
  `llm_semantic_check`（LLM-judged，opt-in 依赖注入设计，解析失败=未通过不静默放行）。
  `tests/test_ai97_llm_pool.py` 73→118 用例全绿；机制④归 MCTS Phase 0、机制⑤备选未做。
- [x] **批次 8 表格基础模型对照 runner（09-24 完成；全量已收官见 §五批次 8 条）**：
  `scripts/evaluation/tabpfn_rolling_compare.py`：逐月重训 × 月内逐日预测
  （test_step=1，v1 抽样高估项直接修掉）× W∈{2m,3m} 窗口扫描；四臂
  （gbdt / tabpfn-3.5 / tabicl-2.2 / 秩平均，走 PREDICTORS 注册零分叉）；
  评估 = 可交易 IC（T+1 掩码）+ NW t + Top10% 月频 **open 正名口径**组合 +
  两臂逐日秩相关诊断；`--device cuda` 好机器全量入口。本机 hs300 gbdt 臂
  冒烟通过（tradableIC 0.0255/0.0271、NWt 3.0/3.2，W=3m>2m 与 08-08 方向
  一致）。**顺手修窗口语义 bug**：`MonthBegin(W)` 从月内日期回退只覆盖
  W−1 个月（2m 窗实测仅 17~20 日），改 `DateOffset(months=W)`。
- [x] **东吴 LLM-MCTS Phase 0 代码落地（09-24 完成）→ 全量已收官（10-08/09）**：
  `factor/mcts/` 五模块（seeds：29 Seed = Alpha158 rolling 全类 × w=20，
  **求值走原生 callable 零转译**；reward：周度六项 + 防前视切片评测器；
  tree：UCT + virtual expansion；proposer：MCTS 上下文 prompt + LLM 扩展器
  + 离线变异兜底；engine：主循环 + 三层去重）+ `scripts/factors/run_llm_mcts.py`
  四臂 runner（mcts/llm_oneshot/gp/gflownet，双口径验收 + PBO/DSR 自动出数）。
  吸收 RD-Agent 机制①（周度 IC ≥0.99 数值去重）④（失败换向）⑦（JSON 纪律）。
  **all-A 正式跑**（all_a_2019_2026，IS 2019-2023/OOS 2024-2026-09，h=5，
  3 seed × 50 iters × 5 variants）：750 候选收编 14、**PBO=0.039 族层稳健**、
  OOS≥IS 泛化一致；`build_alla_llmmcts_factors.py` 入库 llm_* 后 v4 席位
  **4/14 入池（唯一产出超门槛因子的挖掘引擎）**，但组合层 +0.16pp 噪声级
  **不采纳**——「席位突破、组合无增量」归档（量价族信息在 DPP 下与既有 alpha
  高度重叠），证据 `reports/batch2385_结果速览.md` P-A 节 + 终审节。
- [x] **多模型×多标签预测合成对照（09-23 完成）**：12 成员（gbdt/ranker/ridge ×
  h1/h5/h10/h20）等权逐级 + walk-forward 学权（`scripts/evaluation/member_blend12.py`，
  与 stack_blend 同参）。结论：**增量在 horizon（标签）维度不在模型族维度**
  （eq_h20_3 16.00% vs eq_h1_3 12.27%；同 horizon 混族反而稀释）；全 12 等权
  15.12% < h1020 16.89%（−1.77pp）；学权 stack_all12 15.53% ≈ 生产 ens_h1h5
  15.52%，对 12 成员学权仍无增量。**h1020 已在 920 口径裁决采纳**（horizon_mix_920：
  16.80% vs baseline 13.49%，+3.31pp——后经 §⑩-⑪ 成为定版信号成分；12 成员
  全量复验随之不再必要）。证据 `reports/member_blend12/`、`reports/horizon_mix_920/`。

### 生产化 / 长期

- [ ] **生产级执行**：实盘下单对接、实时行情驱动（当前日频盘后信号，
  "信号→次日执行"滑点假设未经真实成交验证）。
- [ ] **文本挖掘合规化**：同花顺/巨潮爬虫依赖（页面变更即断），生产化需
  商业数据源或明确"研究性补充"定位。
- [ ] **在线 dashboard**：报告均为静态 HTML，无增量刷新监控页面。
- [ ] **监控链收敛（非阻塞余项）**：`monitoring/` 包与 `monitor_performance.py` 留在仓库
  供手动跑；复活前提 = ① 全A 库 evals 补齐 ② runner 行情源参数化。
- [x] **HS300 时代入口三口同裁归档（09-24 完成）**：`e2e_backtest` /
  `optimize_e2e` / `investment_report` → `scripts/archive/`（证据与已归档者
  相同：数据口径停更、产物已清理）；同步退休 `test_metrics` perf_stats
  一致性用例与 `test_layering` 旧路径钉子；mock 链路测试按 09-21
  e2e_stock_picks 先例改指 archive 保活；归档件/oneoff/README 引用全部改指。
  **顺手清欠**：`test_layering` 私有名守卫在 master 上的存量红灯（altdata/
  builders/p5/galaxy/rolling_grid 白盒用例 + mcts `_fmt_const`）——业务侧
  公开化（`fmt_const`/`std_code6`/`pit`/`safe`），测试侧白名单精确登记，
  守卫恢复全绿。
- [x] **调度余项（09-24 完成）**：`.workbuddy/memory/automations/71e2d7a5`
  退役记忆文件已删（现役 `3b31c720` 日报自动化保留不动）。

### 已完成归档（2026-09-18~20 交付批次）

- [x] **RL 组合优化线（stage2 三阶段管线）**：`factor/rl/portfolio_env.py`
  （银河 0706 口径主动权重空间 env + tradable_masks，22 用例）→
  `run_portfolio_ppo.py`（Phase 1 hs300 平价验证四臂，行为形态与银河 HS300 结论一致）→
  `run_portfolio_phase2.py`（Phase 2 zz1000 主实验 runner：三标签 GBDT 信号管线 +
  PPO 滚动集成 + 银河口径 QP + 消融钩子；冒烟全链路通过；**全档终判已出（10-05）：
  负结果归档，RL 线整体不成立**，见 §五同期交付归档与 RESEARCH_TODO §二）
- [x] **银河 0608 L1（风险标签可测成立）**：`factor/classic.compute_galaxy_features`
  （36 特征）+ `build_galaxy_labels`（三标签）+ `scripts/evaluation/galaxy_l1.py`
  （mdd test 0.25/0.40 超研报量级；sharpe/alpha 年度翻符号 → 质量门槛+回退为必要设计）
- [x] **策略级过拟合检验**：`stats/pbo.py`（14 用例，统计锚点测试）
- [x] **多因子10 两主力合成方法 + T 扫描脚本**：`synthesize_ic_ir_max`/`synthesize_ic_max`/
  `half_life_weights`/等权分支（26 用例）+ `scripts/evaluation/mf10_t_scan.py`
- [x] **QP 求解器三件套（AI39 对齐）**：`industry_target_from_benchmark` /
  `max_weight_change` / `solve_lambda_grid`；顺手修 industry_target Series 歧义 bug
  与 risk_aversion 反向 docstring
- [x] **每日邮件正文链路** + **研报 PDF 抽文本工具** + markdown 依赖显式化
- [x] **数据链路健壮性（09-16 事故治理）**：cache 宽表合并 OOM 修复、状态表分片落盘 +
  子进程硬超时
- [x] **分钟频一/二层链（09-09~09-14）**：MinutePanelStore + 42 个 im_* 特征
  （37/42 显著，尾盘动量反转 t=-23.4）+ A/B 对照（with_im 0.583 vs placebo 0.291，
  日内因子真实有效）+ GP 二层挖掘（6 公式入库但组合增益≈0）+ 挖掘层接入 GP/GFlowNet

---

## 三、工程层剩余债务

- [x] **.venv rl 依赖债核销（09-24 实测）**：gymnasium/sb3/sb3-contrib/torch
  在 .venv 已齐（torch 2.13）；真实堵点是 pandas 3.x 下 `Series.rank().to_numpy()`
  返回只读视图令 `portfolio_env.rank_pct` 原位覆写炸——已修（显式 copy），
  双解释器（sys pandas 2.3.3 / venv pandas 3.0.5）rl 测试全绿，**.venv 保留
  pandas 3 作兼容性金丝雀**（日更计划任务运行中，不降级）。
- [x] **portfolio_env 涨跌停/停牌掩码注入（09-24 完成）**：
  `build_tradable_mask` 增 `execution_lag` 参数（0=T 日状态，决策日收盘调仓
  自洽；默认 1 零回归）；`run_portfolio_phase2.make_env` 掩码 = 指数成分 ∩
  T 日可交易 + 整行全 False 回退防炸训练；QP 臂 `valid` 过滤同口径 + 信号
  reindex 兜底（mdd_66 长标签尾部覆盖不足不再 KeyError）。测试
  `tests/test_phase2_tradability.py`；hs300 ppo/qp/ew 冒烟通过。
- [x] **`stage_backtest`/`stage_ensemble` eq 缓存失效（P0 附带，09-23 完成 commit f4d3172）**：
  rolling_grid_alla 三处 exists-skip（backtest eq / select json / pred parquet）全部接
  产物指纹 sidecar（`.fp.json`），失配自动重算（eq/selection 分钟级自动；pred warning 后
  重训）；eq 指纹绑 pred 内容（重训→eq 失效，断点续跑仍命中）；`--exclude-features` 补硬
  校验须配 --out-tag；**附带发现并修复 --quick 冒烟产物可被全量复用**（years 入 pred 指纹）。
  注意 stage_ensemble/smallcap 本就总重跑无此病；指纹抓不住不改常量的纯代码逻辑变更
  （此类走 --out-tag）。
- [ ] **邮件链路 Skills 化**（可选，速读批借鉴）：build_daily_email_body 的口径约束固化 +
  渐进式披露；技能总量控制在上限 20–30 内。
- [ ] **scripts 层私有名跨模块 import**：仅剩 `scripts/textmining/*`（他会话管辖，不动）。

### 已完成归档

- [x] 六轮工程整改 P0–P5（2026-08-29 基线）：stats 公共层、scripts 收敛、521 测试全绿
- [x] 口径统一第一~七批（09-10~09-11）：run_backtest 删除、四项决策拍板、
  权重生产入口定案、scripts 全量重排、同名指标公式对齐、分层收口、HTML 字节复现
- [x] 私有名倒挂清零、根目录 probe 清理、零引用模块删除；cli_common 推广、
  HTML 模板收编、超长文件拆分、最小 CI（09-10 起转手动触发）

---

## 四、低优先级（知情即可）

- [ ] `factor/technical.py` 与 `factor/technical_indicators.py` 双口径指标
  （有意保留；SAR 有两份实现）
- [ ] GFlowNet 自写 PPO（`factor/gflownet/ppo.py`）与 AlphaPool MaskablePPO 双轨
- [ ] optimize 标注的 P3 待建：完整多期最优执行、风险预算非等权
- [ ] 一次性实验脚本（`gtja_*` / `diagnose_*` / `compare_*` 等）保留原样，不迁移 cli_common

---

## 五、好机器批次出数后的后续立项（2026-09-25 立，2026-10-09 全部收官）

> 证据与数字：`reports/batch1_920_收尾对照表.md` §⑫–⑯、`reports/batch2385_结果速览.md`。
> 定版演进：10-02 v2 口径拍板 h1020⊕s0.5（17.87%/0.781）→ 10-06 §⑭ fundind 快信号
> +buffer(20/30)（18.68%/0.813）→ 10-08 §⑯ buffer 网格上修 15/40（18.07%/0.805/
> 换手 19.1%，top20 口径）。五项立项 + 后续 P-A/P-B/P-C/P1 提升路径全部裁决完毕：

- [x] **批次 8 组合层传导研究**：头部诊断确认「ICL 的 IC 优势在截面中腰部不在头部、
  跨方法头部 Jaccard 0.17 真互补」→ H1 三臂集成被否 / **H3 头部内加权成立**
  （tabicl_w3m 加权 11.97%/0.739 中小池组合冠军）→ zz1000 复验跨池成立 + 新发现
  「gbdt+tabicl 对集成 20.03%/1.25」→ 全A 大截面裁决：**ICL 优势不跨大截面**
  （gbdt IC 4.4 倍反超），主模型基座维持 gbdt、tabicl 定位中小池组件。
  余项（对集成组合层验证 / H3 跨池复核）留中小池场景非主线。
- [x] **GFlowNet 旋钮增量（国金24 残余③④）**：RRE≥0.3 门槛 |IC| 中性略负
  （0.113 vs 0.117，种子方差增大）；预算 2x 饱和（1200 iters 反而 −0.004）——
  **raw+im 无门槛 600 iters 为该线定版形态**；⑤入库收官：114 个 gf_* 因子入
  hs300_2022_2025 库（33 个 |t_nw|>2）、**PBO=0.114 族层稳健**（vs AI97 0.704）。
- [x] **AI97 步长归因**：100k 步（10 倍预算）训练段复合 IC 仅 −0.05→+0.02、
  训练/测试符号翻转依旧 → **真实样本外失效而非未收敛**，该线正式归档；
  若重启需换池/构建机制，不再加预算。
- [x] **E4 交互特征换注入方式**：池竞争路径证伪在先（15 个交互因子 |IC| 不入
  top-150、选股逐位同基线）；强制注入臂（ortho920e4force）gbdt h1 M raw f0.10
  11.81% vs v2 基线 11.67%（**+0.14pp 噪声级**）——E4 线结案，「基本面当条件
  变量」假设在本构造下不成立。
- [x] **方案 B 缺面板对齐**：唯一存活余项，上移 §一（小工程）。

### 同期交付归档（2026-10-04~09，定版冲刺批）

- [x] **标签工程三臂全量闭环（10-04/05）**：gauss_rank 零增量（主线 rank 已在最优
  族）/ co 隔夜判负（−1.0~−1.2pp、回撤换手双恶化；ICIR 升经逐日 IC 配对检验
  t=−5.2 判定为无价值副产品，日频组合 28/28 恶化）/ ir 判负（AI29 回撤代价 +
  Label Alchemy vol_scaled 预警双应验）——**现行 rank+cc+return 即标签维局部最优**，
  预算转投 horizon 混合（已兑现：h1020 为定版成分）。
- [x] **phase2 修复三连 + 全档终判（10-02~05）**：env er 基准改可交易基准
  （0/4 全灭真凶=口径 bug 非 RL 无效，零动作 −10.9%→0.00% 实证）+ PPO 门槛按
  zz1000 重校（原值 0.02/0.08 系研报科创50口径，PDF 取证）+ 候选集成 rollout
  per_model 形状修正 → 全档 24 臂终判：**银河口径 RL/QP 在 zz1000 无稳健超额、
  reward 六项分量贡献不稳健——RL 路线在本项目数据域整体有效性不成立**（与
  AI97 互证）。已知残留：QP-2025 solver 失败、TopN-2026 日期缺口、no_alpha
  消融未生效（三项记录在案）。
- [x] **主模型提升路径 ②①④③ 全部裁决归档（10-06）**：buffer×定版采纳（+0.40pp/
  换手 41.3→27.0%）/ fundind 快信号采纳（18.68%/0.813 新定版）/ fundind 慢信号
  负（慢信号偏好标准中性化）/ T 窗扫描维持 T=24 / tabicl 全A 冒烟裁决 gbdt 基座
  不可替代。工具沉淀：`buffer_arm_920.py` defv 变体、`fundamental_blend
  --panels-subdir`、`tabfm_rolling.py` 全A 预筛路径（ic_h top-N 防 OOM）。
- [x] **D 盘数据治理（10-06）**：sdk_cache 137G→~3G（balance_sheet/pledge_freeze/
  restricted/margin 四表 116G 经 parquet 完整性验证后删）；income/cash_flow 补齐
  （income 4383 只=SDK 源上限）→ 引出 P0 income 复核（负结果 §⑮：信息已被
  balance 系+原池覆盖，定版维持 v2）与**环境漂移更正**（同配置如今重跑
  ~17.0–17.2%，不回退数据换数字，引用注明数据态）。
- [x] **alt 三族×主模型双路径终审（10-08）**：席位竞争 0/36 + 慢信号家族并入
  （--include-alt-news）18.68%→16.27% 双败——**alt 三族对主模型无增量结案**；
  news 族 h20 反转方向（|IC| 随 horizon 单调增强）与 macro 择时特征留待未来
  新信号线。
- [x] **GP 预算放大裁决（10-07）**：pop400×gen25 拉满仍停早期平台、去冗余仅 1 因子
  入库——与 GFlowNet 饱和互证：**现有特征空间上做公式重组合的两类算法都已触及
  上限，增量 alpha 需新信息源或新归纳偏置**。
- [x] **LLM-MCTS all-A 正式跑→入库→终审（10-08/09）**：750 候选收编 14、PBO=0.039、
  入库 llm_*、v4 席位 4/14（唯一挖掘引擎席位突破）、组合层 +0.16pp 噪声级不采纳
  ——「席位突破、组合无增量」归档（见 §二东吴条目详录）。

---

## 建议推进顺序（2026-10-09 刷新；研报线见 RESEARCH_TODO 第六节）

1. **定版口径切换生产出榜链**（§一，当前唯一高优先——特征源 fundind +
   h1020⊕s0.5 + buffer(15/40) 三处切换 + 双链并行对照一周）
2. 小工程收尾：e2e 族历史报告年化口径重跑（~3% 系统性偏移，队列末尾）+
   方案 B 缺面板对齐（4 列）
3. 分钟频扩容吞吐探针 → 视成本决定 all_a 分钟物化与第三层扩原料
   （资源墙；华泰3128 依赖此线，RESEARCH_TODO §六）
4. 中小池组件线（可选，非主线）：tabicl 头部内加权跨池复核 +
   gbdt+tabicl 对集成组合层验证
5. 生产级执行（实盘对接、实时行情）

---

## 附录：历史交付记录（压缩版）

> 原 README「待建 / 已知缺口」节 + 完整交付细节压缩至此。**待办**请看正文 §一~§四。

**2026-08-25 批**：组合级风险分解（Euler + VaR/CVaR 成分）；CPCV h=1 无偏评估；
端到端选股工作流固化（现结论：2024-01~2026-08 月频等权 top50 +16.2% 跑输全池基准
+30.8%——信号强度不足以支撑集中持仓）；投资收益报告（中性化后等权 +27.0%/Sharpe 0.66，
仍跑输沪深300 指数 +36.0%）；回测引擎三项修复（结算-调仓顺序前视一天、h>1 区间结算、
avg_turnover 稀释）+ 收益面板口径守卫（h=1 传 shift(-h) 直接报错）；风格中性化有效 /
周频被证伪（optimize_e2e 四组对比）；模型增强组合固化配置（`run_model_portfolio`）；
LGBRankerPredictor 修复（负 IC −0.055 → +0.207）。

**2026-09-01 批（全A 多年度滚动训练）**：全A数据回补（1098 万行 × 5549 股）；
`all_a_2018_2026` 数据集（798 因子 × 2471 日 × 5549 股 float32 ~24GB）；120 组合网格
主链（四阶段断点续跑 + 幽灵股守卫）；核心结论：全A截面 OOS IC 显著强于 HS300
（gbdt h1 全期 IC=0.108）、h1+M/W+gbdt+Top10% raw 一致稳健解、日频被成本全灭、
风格中性化在全A上稀释收益（与 HS300 相反）。

**2026-09-07~09 批**：幸存者偏差消除（退市回补 238 只，超额回落 ~2pp）；ens_h1h5 集成
成为新头部（13.8%）；DPP 特征选择 + 基本面保留席位；正交化三臂对比（全正交 +1.6pp
> 混合 ≈ 不正交，增量几乎全部来自量价因子）；ortho 臂集成复现（+13.3%，stage_ensemble
回测窗口 bug 修复）；仓位 overlay 网格（趋势 MA200 减半 MaxDD，按 Calmar 取舍）；
数据更新守卫（盘中半拉 K 线防污染）。

**关键工程教训（全量见 `.workbuddy/memory/`）**：`build_panel` 复权只作用 OHLC
（vwap 用 amount/volume 须先复权）；SDK 大清单单查会挂死（分批+重试）；float64 巨值
astype(float32) 溢出成 inf；状态表缺预测日数据使因子全 NaN；`os.replace` 在 Windows
遇共享读句柄报 WinError 5（写盘须原子优先+占用降级）。
