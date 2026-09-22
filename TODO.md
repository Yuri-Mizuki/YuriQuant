# YuriQuant 待办清单

> **本文件是待办的唯一真源**（研报复现线见 [`RESEARCH_TODO.md`](RESEARCH_TODO.md)；
> 架构与完成度总览、安装与用法见 [`README.md`](README.md)）。
>
> 2026-09-21 大清理：对照已实现代码归档已完成项（另类数据 P0 / 调度收口 / 监控退休 /
> HS300 入口归档 / panels_neu 920 / holder_dyn 消融 / 中性化向量化 / stage2 Phase 0-2 管线等）。
> 清理前全文见 git 历史（`e70ac96` 为 09-20 清理前基线）；历史交付细节压缩于附录。

---

## 一、研究验证欠账（最高优先级——影响结论可信度）

- [ ] **P0 新口径（920 面板）治本口径基线重跑**（**执行环境：另一台更强的机器，本机不跑**）：
  `panels_neu` 882→920 补齐 38 缺（`reports/panels_neu_backfill/`）后，真 DPP 对照
  磁盘主实验 selection **0/18 相同**（每轮 +6~11/−6~11；`lhb_count_20d`/`notice_*`/
  `unlock_ratio_20d`/`limit_pos`/`st_days`/`suspend_*` 进选中，挤出等量 alpha360 量价族）
  ⇒ **15.54%/15.52% 与当前库已脱钩**：当前代码+数据已产不出旧数字，下次重跑
  （即使零改动）落新口径。**命令**（挂机 ≈5–7h：select 36 轮 ≈2h + predict ≈2–3h
  + ensemble/backtest ≈0.5h）——**治本口径**（09-21 拍板；09-22 拍板主跑即
  T+1 open 主披露，close 降为乐观上限对照）：
  `python scripts/pipelines/rolling_grid_alla.py --preproc ortho --out-tag ortho920tl \
     --exclude-features limit_pos --tradable-labels --execution open`
  （`--execution` 仅作用于 backtest 阶段、产物加 `_open` 后缀；新目录的
  `_base/open_adj.parquet` 由 `--stage all` 的 prep 自动建出。⚠️ 执行价分段
  只能挂 h=1 结算（§1.7 推论①："h=1 回测"≠"只用 h1 模型"，ens/生产链全是
  h=1 结算、open 全覆盖），故 `metrics_overall_open` 仅含 h=1 行（定版口径
  全在其中）；h>1 native 结算诊断行由 close 对照臂保留。**补齐对照臂**——
  主跑完成后各 +0.5h：`--stage backtest`（close，与 882 旧基线对表）+
  `--stage backtest --execution vwap`（次披露）。）
  **生产入口披露臂**：`python scripts/pipelines/run_model_portfolio --no-train
  --execution open` 为主，`--execution vwap` 次披露，close 对照可选
  （09-15 旧库校准 close→open −0.88pp / →vwap −0.94pp，换手不变）。
  三个开关各自的意义（一次重跑同时落三个维度，差异归因靠 2x2 探针补齐）：
  - `--out-tag ortho920tl`：产物另存，旧 `alla_rolling_ortho/` 原样保留作对照，禁覆盖；
  - `--exclude-features limit_pos`：因子层剔除纸面因子（09-16 已证 +0.3~1.2pp/年、
    4/4 格全改善；新口径下它 13/18 轮会重新进选中，不剔则白占 1/50 名额）；
    与生产出榜链 `alla_daily_rank.py` 口径对齐；
  - `--tradable-labels`：**标签治本**（09-17 接入的 P1-a）——训练标签把「T 日封板 →
    T+1 买不进」的样本按 T+1 成交口径掩掉，模型从源头学不到纸面关系，
    不再依赖逐因子拉黑（ST/停牌/一字板同理受益）。
  （可选对照臂：`--out-tag ortho920 --exclude-features limit_pos`（无 labels）与
  `--out-tag ortho920pure`（全默认），各 ≈5–7h。）
  **收尾必做**：① 新旧对照表（沿用 `prod_pipeline_gap` 2x2 格式），**主口径列 =
  `metrics_overall_open`（T+1 open）**，close 仅作乐观上限对照（注意
  `metrics_ensemble.csv` 固定 close 口径，仅作集成内部对照）；
  ② open 口径年化变化 >±1pp ⇒ 更新 MEMORY.md 主实验口径段（主数字换 open），
  旧数字补「882 口径」标注；
  ③ 顺带落地 `_extra` 方案 B（面板存在性门槛；当前 920=920 下是空操作，纯防复发）；
  ④ 若治本臂不及旧基线，先看 `--tradable-labels` 单开关消融再定口径去留——
  **不允许因「治本臂数字低」而回退标签掩码**（掩掉纸面收益是修正偏差，不是损失）；
  ⑤ E1'' horizon 定版（RESEARCH_TODO §一）在新 pred 上以 **open 口径**复核
  （`scripts/evaluation/horizon_mix.py` 需先接 execution 臂）。
- [ ] **重跑 e2e 族报告对齐年化口径**：`perf_stats` 年化 244→252 后，
  `reports/e2e_backtest/`、`reports/investment_report/` 中 e2e 链路历史数字
  与现行口径存在 ~3% 系统性偏移。已挂好机器队列末尾。
- [ ] **all_a_2018_2026 数据集纳入正式因子库监控**（目前为轻量 registry）。
  关联：`monitor_performance` 旧链复活的前提之一（另一前提 = runner 行情源参数化），
  现役生产监控已由 `monitor_production_ic` 承担，本条非阻塞。

### 已完成归档（一行一条）

- [x] 另类数据 P0 管道（09-20~21）：`data/altdata/` 传输层 7 源落地（巨潮快讯 119.2 万条
  自建历史存档 / 东财宏观日历 64,978 行含补洞 / 巨潮增减持 26.6 万行全历史等，
  见 `reports/docs/另类数据管道_数据源说明.md`）；日增量入口 `fetch_altdata_daily`
  进计划任务（18:00，`.venv` 解释器）
- [x] holder_dyn 9 因子三探针消融（09-21）：强制纳入臂 **Δ=−0.85pp 无增量价值**
  （`reports/holder_dyn_forced/`），Step 1b 全历史回补结案不推进；表保留（增量继续抓）
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

- [ ] **实验产物接 PBO 出报告**：`stats/pbo.py` 已就绪（CSCV PBO + DSR + deflate_best），
  把 rolling_grid 各臂 / AI97 三臂 / stage2 Phase 2 各配置的期间收益矩阵接入
  作为固定出报告环节。
- [ ] **分钟频挖掘第三层扩原料**：时段切片矩阵化（20 分钟窗 × 动量/波动/量）、
  量价 lead-lag、tsfresh 计算器移植、事件日条件化（与文本 PEAD 线交叉）；
  扩完重跑 mine_im_combos 看组合上限是否抬升（二层结论：增益≈0，信号在
  时段动量，原料不够）。
- [ ] **分钟频扩容（资源墙）**：all_a 池物化（~5500 码 × 48 bar，MemMap 分块性能待验）、
  1 分钟档位（存储 ×5）；先跑吞吐探针（`scripts/oneoff/_probe_minute_throughput.py`）估成本。
- [ ] **另类数据因子轮次**：快讯/宏观/公告表通道就绪但因子未建（刻意留白），
  等下一次因子挖掘轮次消费；PIT 对齐规则见 `reports/docs/另类数据管道_数据源说明.md` §四。

### 生产化 / 长期

- [ ] **生产级执行**：实盘下单对接、实时行情驱动（当前日频盘后信号，
  "信号→次日执行"滑点假设未经真实成交验证）。
- [ ] **文本挖掘合规化**：同花顺/巨潮爬虫依赖（页面变更即断），生产化需
  商业数据源或明确"研究性补充"定位。
- [ ] **在线 dashboard**：报告均为静态 HTML，无增量刷新监控页面。
- [ ] **监控链收敛（非阻塞余项）**：`monitoring/` 包与 `monitor_performance.py` 留在仓库
  供手动跑；复活前提 = ① 全A 库 evals 补齐 ② runner 行情源参数化。
- [ ] **HS300 时代入口剩余项裁决**：同族 `e2e_backtest` / `optimize_e2e` 证据与已归档者相同
  （`daily_hs300.parquet` 停更、产物 09-16 清理）但属**独立回测实验**，待拍板是否归档。
  09-22 全仓死代码普查补充：`investment_report` 同属本族——零引用入口（仅 mock 测试
  保活）且 import `e2e_backtest` 四个函数（perf_stats / 两只组合回测 / walk_forward
  预测），归档需三口同裁并同步退休 `test_metrics` 的 perf_stats 一致性用例与
  `test_layering` 的 `_enforce_caps` 钉子。普查其余结论：核心库包零孤儿，墓地已收敛在
  scripts/archive + scripts/oneoff 两区，无新增待清项。
- [ ] **调度余项**：`.workbuddy/memory/automations/` 下残留的旧自动化 `71e2d7a5`
  记忆文件（已加退役标注，可择机清）。

### 已完成归档（2026-09-18~20 交付批次）

- [x] **RL 组合优化线（stage2 三阶段管线）**：`factor/rl/portfolio_env.py`
  （银河 0706 口径主动权重空间 env + tradable_masks，22 用例）→
  `run_portfolio_ppo.py`（Phase 1 hs300 平价验证四臂，行为形态与银河 HS300 结论一致）→
  `run_portfolio_phase2.py`（Phase 2 zz1000 主实验 runner：三标签 GBDT 信号管线 +
  PPO 滚动集成 + 银河口径 QP + 消融钩子；冒烟全链路通过，**全量待好机器**，命令见 RESEARCH_TODO）
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

- [ ] **.venv 缺 rl 依赖**：gymnasium/sb3 等只在系统解释器（D:/Python/Python312）可用，
  .venv 跑不了 RL 测试（test_ai97_llm_pool / test_portfolio_env / test_gflownet_im_features
  需用系统 python）；统一环境或在 .venv 补装 rl extra。
- [ ] **portfolio_env 涨跌停/停牌掩码注入**：当前 tradable_masks 只剔非成分；
  Phase 2 全量前接 `data/tradability`（一字板不可成交语义）——zz1000 的 ST/停牌量
  会放大该边界。
- [ ] **`stage_backtest`/`stage_ensemble` eq 缓存失效（P0 附带）**：旧 eq CSV 即跳过回测
  → metrics 可能静默陈旧；P0 治本重跑时顺带核对。
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

## 建议推进顺序（2026-09-21 刷新；研报线见 RESEARCH_TODO 第六节）

1. **P0 新口径（920 面板）治本口径基线重跑**（见一，命令已写明，挂机 ≈5–7h，
   **执行环境：另一台更强的机器，本机不跑**——迁移清单复用 RESEARCH_TODO
   第六节序 0：git clone + `E:\data` 数据面（panels_neu 920 已含）+ Python 环境；
   本任务无 API 依赖）。**唯一挡在「所有全A结论可信引用」前面的事，优先于其他一切长实验**
2. **好机器长实验队列**（清单与命令见 RESEARCH_TODO 第六节序 0）：
   920 治本重跑 → AI97 三臂 ×3 seed → 国金24 残余⑤ 校准轮 → mf10 T 扫描全量 →
   stage2 Phase 2 zz1000 全量（快档→全档）
3. **本机半天级穿插**（均挂已有基础设施）：CVaR 约束进 solve_portfolio、
   labels.py IR/Calmar 标签分支、预测层观点注入等价实现、实验产物接 PBO
4. e2e 族历史报告年化口径重跑（~3% 系统性偏移）
5. 分钟频第三层扩原料 + all_a 分钟数据扩容 + 另类数据因子轮次（资源墙/挖掘轮次，后置）
6. 生产级执行（实盘对接、实时行情）

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
