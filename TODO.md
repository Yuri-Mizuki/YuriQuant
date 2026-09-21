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

> 2026-09-20 清理：本节原 12 项中 10 项已完成，压缩归档于节末；清理前的
> 条目全文见提交 `e70ac96` 的 TODO.md 与附录历史交付记录。
> 2026-09-21 二次清理：另类数据 P0 管道 + holder_dyn 消融 + panels_neu 补齐完成，归档于节末。

- [ ] **P0 新口径（920 面板）全量基线重跑**（09-21 新增，取代原「ortho 臂 alt 口径重跑」；
  **执行环境：另一台更强的机器，本机不跑**）：
  `panels_neu` 882→920 补齐 38 缺（`reports/panels_neu_backfill/`）后，真 DPP 对照
  磁盘主实验 selection **0/18 相同**（每轮 +6~11/−6~11；`lhb_count_20d`/`notice_*`/
  `unlock_ratio_20d`/`limit_pos`/`st_days`/`suspend_*` 进选中，挤出等量 alpha360 量价族）
  ⇒ **15.54%/15.52% 与当前库已脱钩**：当前代码+数据已产不出旧数字，下次重跑
  （即使零改动）落新口径。**命令**（挂机 ≈5–7h：select 36 轮 ≈2h + predict ≈2–3h
  + ensemble/backtest ≈0.5h）——**治本口径**（由莉酱 09-21 拍板）：
  `python scripts/pipelines/rolling_grid_alla.py --preproc ortho --out-tag ortho920tl \
     --exclude-features limit_pos --tradable-labels`
  三个开关各自的意义（一次重跑同时落三个维度，差异归因靠 2x2 探针补齐）：
  - `--out-tag ortho920tl`：产物另存，旧 `alla_rolling_ortho/` 原样保留作对照，禁覆盖；
  - `--exclude-features limit_pos`：因子层剔除纸面因子（09-16 已证 +0.3~1.2pp/年、
    4/4 格全改善；新口径下它 13/18 轮会重新进选中，不剔则白占 1/50 名额）；
    与生产出榜链 `alla_daily_rank.py` 口径对齐；
  - `--tradable-labels`：**标签治本**（09-17 接入的 P1-a）——训练标签把「T 日封板 →
    T+1 买不进」的样本按 T+1 成交口径掩掉，模型从源头学不到纸面关系，
    不再依赖逐因子拉黑（ST/停牌/一字板同理受益）。
  （若想分离三开关的净贡献：跑完治本臂后，可用
  `--out-tag ortho920 --exclude-features limit_pos`（无 labels）与
  `--out-tag ortho920pure`（全默认）补两个对照臂，每个 ≈5–7h，可选。）
  **收尾必做**：① 新旧 `metrics_ensemble` 对照表（沿用 `prod_pipeline_gap` 2x2 格式，
  探针 `scripts/oneoff/probe_prod_pipeline_gap.py` 改两行路径即得；治本臂 vs
  `alla_rolling_ortho` 的差异 = 920 面板 + exclude + labels 三者合计）；
  ② 集成年化变化 >±1pp ⇒ 更新 MEMORY.md 主实验口径段，旧数字补「882 口径」标注；
  ③ 顺带落地 `_extra` 方案 B（面板存在性门槛；当前 920=920 下是空操作，纯防复发）；
  ④ 若治本臂不及旧基线，先看 `--tradable-labels` 单开关消融再定口径去留——
  **不允许因「治本臂数字低」而回退标签掩码**（掩掉纸面收益是修正偏差，不是损失）。
- [ ] **重跑 e2e 族报告对齐年化口径**：`perf_stats` 年化 244→252 后，
  `reports/e2e_backtest/`、`reports/investment_report/` 中 e2e 链路历史数字
  与现行口径存在 ~3% 系统性偏移。已挂好机器队列末尾。
- [ ] **all_a_2018_2026 数据集纳入正式因子库监控**（目前为轻量 registry）。

### 已完成归档（一行一条）

- [x] ~~ortho 臂 alt 口径 `stage_predict/stage_backtest` 重跑~~（09-12 挂起项）——
  **并入上条新口径全量重跑**：alt 28 面板已随 09-21 补齐进入候选池，旧口径单独重跑已无意义
- [x] 另类数据 P0（09-20~21）：抓取层 7 源落地（cls 119.2万条 / 宏观 64,978 行含补洞
  2,275 天 / 巨潮增减持 26.6 万行全历史等，见 `reports/docs/另类数据管道_数据源说明.md`）；
  holder_dyn 9 因子三探针消融 **Δ=−0.85pp 无增量价值**（`reports/holder_dyn_forced/`，
  Step 1b 全历史回补不做）；panels_neu 882→920 补齐（`reports/panels_neu_backfill/`）
- [x] 重跑 multiyear_oos（09-01，620s）：h1×M 是唯一三年一致稳健解
- [x] 重跑 freq_tune（09-01）：h1 排序不变，h5×M 伪结果修正
- [x] 全A滚动训练实验（09-01）：选股宽度是"跑输基准"的答案——Top10% 等权
  超额全A等权 +5.2%/年，gbdt h1×M 一致稳健
- [x] 全A数据卫生修复（09-07）：退市股回补 238 只（幸存者偏差消除，超额
  回落 ~2pp 排序不变）、状态表重拉、ETF 列剔除、半拉日裁剪
- [x] 生产化每日推理 `alla_daily_rank`（09-08，含计划任务注册）
- [x] 出榜链口径对齐主实验 ortho（09-17）
- [x] 每日计划任务静默失败修复（09-17）
- [x] 全A超额来源复核（09-08）：对全A等权纯选股 α=+5.1%/年，对上证超额
  约半数为小盘风格敞口
- [x] 超额/夏普显著性检验进回测指标（09-08）
- [x] 论文复现因子族入库（09-08，awesome-systematic-trading 21 式）

---

## 二、研究功能缺口（新能力）

### 2026-09-18~20 交付批次（RL 组合优化线 + 统计层，详见 RESEARCH_TODO stage2 条目）

- [x] **RL 组合优化线（stage2 三阶段）**：`factor/rl/portfolio_env.py`
  （银河 0706 口径主动权重空间 env + tradable_masks，28 用例）→
  `scripts/factors/run_portfolio_ppo.py`（Phase 1 hs300 平价验证四臂：
  PPO 贴基准打平/QP·TopN 主动偏离者输，与银河 HS300 形态一致）→
  `scripts/factors/run_portfolio_phase2.py`（Phase 2 zz1000 主实验 runner：
  三标签 GBDT 信号管线 + PPO 滚动集成 + 银河口径 QP + 消融钩子；冒烟全链路
  通过，生产命令两档见 RESEARCH_TODO，待好机器全量）。
- [x] **策略级过拟合检验**：`stats/pbo.py`——CSCV PBO（12870 组合秒级）+
  缩水夏普 DSR + deflate_best（14 用例，统计锚点测试）；待接入各实验产物
  作为出报告固定环节。
- [x] **银河 0608 特征族与三标签（复现 L1）**：`factor/classic` 新增
  `compute_galaxy_features`（36 特征，分组差异化预处理）+
  `build_galaxy_labels`（alpha22/sharpe22/mdd66）+ `scripts/evaluation/galaxy_l1.py`
  滚动评估（6 用例）。**结论：风险标签可测成立**（mdd test 0.25/0.40 超
  研报量级）、sharpe/alpha 年度翻符号→质量门槛+回退为必要设计。
- [x] **合成权重 T 扫描**：`scripts/evaluation/mf10_t_scan.py`（研报 T∈
  {3,6,9,12,24,36} 月网格；防未来函数=训练窗截标签实现期；7 用例）。
- [x] **QP 求解器三件套（AI39 对齐）**：`optimize/solver` 新增
  industry_target_from_benchmark / max_weight_change（逐股 |w−w₀|≤δ）/
  solve_lambda_grid；顺手修 industry_target 传 Series 的真值歧义 bug 与
  risk_aversion 反向 docstring。
- [x] **每日邮件正文链路**：`scripts/reporting/build_daily_email_body.py` +
  `md_to_email_html.py` + pyproject report extra（markdown 依赖显式化）。
- [x] **研报 PDF 抽文本工具**：`scripts/data_tools/dump_pdf_text.py`。
- [x] **数据链路健壮性（09-16 事故治理）**：cache 宽表合并 OOM 修复
  （内存友好合并+upto 短路）、状态表分片落盘 + 子进程硬超时
  （settings fetch.status_table）。

### 剩余缺口

- [ ] **分钟频挖掘第三层扩原料**：时段切片矩阵化（20 分钟窗 × 动量/波动/量）、
  量价 lead-lag、tsfresh 计算器移植、事件日条件化（与文本 PEAD 线交叉）；
  扩完重跑 mine_im_combos 看组合上限是否抬升（二层结论：增益≈0，信号在
  时段动量，原料不够）。
- [ ] **分钟频扩容（资源墙）**：all_a 池物化（~5500 码 × 48 bar，MemMap
  分块性能待验）、1 分钟档位（存储 ×5）；先跑吞吐探针估成本。
- [ ] **生产级执行**：实盘下单对接、实时行情驱动（当前日频盘后信号，
  "信号→次日执行"滑点假设未经真实成交验证）。
- [ ] **文本挖掘合规化**：同花顺/巨潮爬虫依赖（页面变更即断），生产化需
  商业数据源或明确"研究性补充"定位。
- [ ] **在线 dashboard**：报告均为静态 HTML，无增量刷新监控页面。

### 已完成归档（分钟频一/二层链，2026-09-20 压缩）

- [x] 数据层 MinutePanelStore + 特征层 42 个 im_* + 单因子体检（09-09：
  37/42 显著，尾盘动量反转 t=-23.4 为全库最大增量）+ A/B 对照（09-14：
  with_im 0.583 vs placebo 0.291，日内因子真实有效）+ GP 二层挖掘
  （09-14：6 公式入库但**组合增益≈0**，全部围绕尾盘动量代数重述）+
  挖掘层接入 GP/GFlowNet（09-14 `mine_im_combos` / 09-16 `--feat-source`，
  原"挖掘层"待办就此关闭）。

---

## 三、工程层剩余债务

> 2026-09-20 清理：本节原 20 项已全部完成（六轮工程整改 + 口径统一七批 +
> cli_common 推广 + HTML 模板收编 + 超长文件拆分等），压缩归档于节末；
> 清理前全文见提交 `e70ac96`。

### 剩余（5 项）

- [ ] **.venv 缺 rl 依赖**：gymnasium/sb3 等只在系统解释器
  （D:/Python/Python312）可用，.venv 跑不了 RL 测试（test_ai97_llm_pool /
  test_portfolio_env / test_gflownet_im_features 需用系统 python）；
  统一环境或在 .venv 补装 rl extra。
- [ ] **portfolio_env 涨跌停/停牌掩码注入**：当前 tradable_masks 只剔
  非成分；Phase 2 全量前接 `data/tradability`（一字板不可成交语义，
  与研报环境层一致）——zz1000 的 ST/停牌量会放大该边界。
- [ ] **实验产物接 PBO 出报告**：`stats/pbo.py` 已就绪，把 rolling_grid
  各臂 / AI97 三臂 / Phase 2 各配置的期间收益矩阵接入 cscv_pbo +
  deflate_best 作为固定出报告环节。
- [ ] **邮件链路 Skills 化**（可选，速读批借鉴）：build_daily_email_body
  的口径约束固化 + 渐进式披露；技能总量控制在上限 20–30 内。
- [ ] **其余 scripts 层私有名跨模块 import**：仅剩 `scripts/textmining/*`
  （他会话管辖，不动）。

### 已完成归档（2026-09-20 压缩）

- [x] 六轮工程整改 P0–P5（2026-08-29 基线）：stats 公共层、scripts 收敛、
  521 测试全绿
- [x] 口径统一第一~七批（09-10~09-11）：run_backtest 删除、四项决策拍板、
  权重生产入口定案、scripts 全量重排、同名指标公式对齐、分层收口、
  HTML 字节复现
- [x] 私有名倒挂清零（守卫泛化）、根目录 probe 脚本清理、零引用模块删除
- [x] cli_common 推广、HTML 报告模板收编（8 套收敛）、超长文件/函数拆分、
  核心 print→logging、最小 CI

---

## 四、低优先级（知情即可）

- [ ] `factor/technical.py` 与 `factor/technical_indicators.py` 双口径指标
  （有意保留；SAR 有两份实现，`calc_sar` / SDK 版）
- [ ] GFlowNet 自写 PPO（`factor/gflownet/ppo.py`）与 AlphaPool MaskablePPO
  （`factor/rl/`）双轨；`factor/rl` 目前仅测试消费、无 scripts 入口
- [ ] optimize 标注的 P3 待建：完整多期最优执行、风险预算非等权
- [ ] 一次性实验脚本（`gtja_*` / `diagnose_*` / `compare_*` 等）保留原样，
  不迁移 cli_common（改动无收益只有风险）

---

## 建议推进顺序（2026-09-21 刷新；研报研读线见 RESEARCH_TODO 第六节）

1. ~~重跑 multiyear + freq_tune~~（09-01 完成）；口径统一二/三批（全部完成）
2. **P0 新口径（920 面板）治本口径基线重跑**（见一，命令已写明，挂机 ≈5–7h，
   **执行环境：另一台更强的机器，本机不跑**——机器迁移清单复用 RESEARCH_TODO
   第六节序 0：git clone + `E:\data` 数据面（panels_neu 920 已含）+ Python 环境；
   本任务无 API 依赖）。**唯一挡在「所有全A结论可信引用」前面的事，优先于其他一切长实验**
3. **好机器长实验队列**（清单与命令见 RESEARCH_TODO 第六节序 0）：
   AI97 三臂 ×3 seed → 国金24 残余⑤ 校准轮 → mf10 T 扫描全量 →
   stage2 Phase 2 zz1000 全量（快档→全档，命令见 RESEARCH_TODO）
4. **本机半天级穿插**（均挂已有基础设施）：CVaR 约束进 solve_portfolio、
   labels.py IR/Calmar 标签分支、预测层观点注入等价实现、实验产物接 PBO
5. e2e 族历史报告年化口径重跑（见一，~3% 系统性偏移）
6. 分钟频第三层扩原料 + all_a 分钟数据扩容（资源墙，后置）
7. 生产级执行（实盘对接、实时行情）

### 历史顺序（2026-08-29 版，仅存档）

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
   scripts.common.portfolio_common）/ 私有函数倒挂收口 / 文本链路 IC 口径核查
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

- **组合级风险分解**（`optimize/risk.py: risk_decomposition`）：Euler 分解（MRC + CR + 占比）+ 风格/行业因子方差贡献（B'ΣB 分解）+ VaR/CVaR 成分分解（历史模拟法）+ 风险预算校验。报告脚本 `scripts/reporting/risk_decomposition_report.py`。
- **h=1 模型 CPCV 无偏评估**（`scripts/evaluation/cpcv_h1_eval.py`）：固定 h=1 的 10 特征 + gbdt 超参，跑 15 条 CPCV 路径产出 IC 分布 + 路径间 t-test + horizon 对比（h=1/h=5/h=20），消除 horizon 选择偏差。
- **端到端选股工作流固化**（`scripts/common/e2e_common.py` + `e2e_stock_picks.py` + `e2e_backtest.py`）：

  ```
  今日选股:  D:/python/Python312/python.exe scripts/pipelines/e2e_stock_picks.py --real --top 30
  策略回测:  D:/python/Python312/python.exe scripts/pipelines/e2e_backtest.py --real --top 50 --freq M
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
- **投资收益报告**（`scripts/reporting/investment_report.py`）：最终交付物，含两部分——
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
- **优化实验**（`scripts/portfolio/optimize_e2e.py`）：调仓频率 × 风格中性化四组对比
  （2024-01~2026-08 top-50 等权）——① **风格中性化有效**：M 月频 + 五因子残差
  16.2%→27.0%（Sharpe 0.41→0.66，回撤 −15.6%→−12.1%）；② **周频调仓被证伪**：
  W + 原分数 6.2%（Sharpe 0.19）远差于 M——换手成本与信号噪声吃掉 h=5 的短视野
  优势，减少信号衰减的假设不成立；③ 中性化对 W 同样有效（6.2%→18.4%）。
  最优配置 = **月频 + 中性化**，已并入 `investment_report.py`（默认开启）。

#### 固化配置与正式入口（2026-08-25，最终交付物）

- **模型增强组合固化配置**（`config/settings.yaml` 的 `model_portfolio` 段 +
  `strategy/examples.py: TopFracLongOnly`，入口 `scripts/pipelines/run_model_portfolio.py`，
  输出 `reports/model_portfolio/`）：把「模型信号 → 风格中性化 → Top20% 重仓多头」
  固化为可复用正式入口，配置只此一处真源。默认 horizon=1 / gbdt / frac=0.20 /
  月度调仓，2025 test 段成本后超额沪深300 **+3.07%**（Sharpe 1.60，MaxDD −11.3%；
  引擎口径修复后复测值）。关键规律：只支持 h=1（h=5 组合难变现）、Top20%
  是 alpha 密度甜点、中性化是信号变现的关键（raw IC 高但跑输风格）。
- **修复 LGBRankerPredictor**（`model/predictor.py`）：`fit()` 曾硬编码
  `N_BINS=30`+`objective=lambdarank`、绕过构造参数导致负 IC(−0.055)；改为使用
  实例参数、默认 `labels_bins=2`（截面中位数二分）+`rank_xendcg` 后样本内 IC
  由负转正至 +0.207（434/484 天为正），现有调用方无需改动。
- **调仓频率精修**（`scripts/evaluation/freq_tune.py` → `reports/freq_tune/freq_tune.csv`）：
  h=1 时 M 月度最优（超额 +6.1%/Sharpe 1.55）；频率越高换手越严重侵蚀 alpha
  （日频超额 −40.6%）。2026-09-01 已在修复后引擎下重跑：h5×M 从 BUG 下的伪结果
  （Sharpe −3.1）修正为超额 −16.6%/Sharpe 0.95，h=5 仍显著劣于 h=1 但差幅可信。
  修复前旧结果存档于 `reports/freq_tune/freq_tune_prefix_engine_fix_20260825.csv`。
  注：turnover 列自 BUG-2 修复后为"按调仓事件平均"口径，与修复前的稀释口径不可比。
- **多年度 OOS 稳健性**（`scripts/evaluation/multiyear_oos.py` → `reports/multiyear/`）：
  gbdt/ridge/ranker × h1/h5 × D/W/M 在 2023/2024/2025 分年 walk-forward
  （特征在定型期 fixed，防前视选择）。2026-09-01 修复后引擎重跑：
  **h1×M 仍是唯一三年一致稳健的频率解**——gbdt +5.5%±1.7%、ranker +5.4%±4.6%
  （均 3/3 年正超额）；ridge 全负；h5×M 三年平均 ≈ −15%（与 freq_tune 2025
  单年 −16.6% 互证）；日频 −33%~−46% 全灭。h>1 旧结论系 bug 伪结果的注记就此了结。

#### 已完成（2026-09-01）全A多年度滚动训练实验（最终交付物）

- **数据回补**（`scripts/builders/fetch_alla_history.py` + `fetch_status_batched.py`）：
  全A日线 2015-01~2026-09-01（1098 万行 × 5549 股，分批断点续拉，SDK 大清单
  单查会挂死的对策）、状态表回补 2015-2018、交易日历/上证指数基准 000001.SH。
  backward_factor / equity_structure / 行业分类原本已全A覆盖。
- **全A公因子数据集 `all_a_2018_2026`**（`scripts/builders/build_alla_alpha_panels.py`，
  54 分钟 4 进程）：alpha101/158/191/360 共 **798 因子 × 2471 日 × 5549 股**
  （float32 面板，~24GB）。截面算子要求全截面在场——按因子分片并行而非按股票分块；
  逐因子落盘 + 预计算 4 个 horizon 的日频 IC 缓存（特征漏斗零 IO 复用）。
  工程教训两条：float64 巨值在 astype(float32) 时溢出成 inf（须转换后再
  replace）；爆炸量级因子（alpha191_017 最大 ~3e38）必须加载时 zscore+clip。
- **实验主链**（`scripts/pipelines/rolling_grid_alla.py`，四阶段断点续跑，单测
  `tests/test_rolling_grid_alla.py`）：每年特征选择（过去 500 日 horizon 匹配
  IC + 覆盖率 + 项目正典 cross-spearman 去冗余）→ 季度（h≥10 半年）walk-forward
  滚动训练（embargo=horizon，500 日滚动窗）→ **幽灵股守卫**（LightGBM 对全 NaN
  特征照样输出预测，未上市股会占满信号顶部——预测面板按"当日有行情且 ≥1/4 特征
  可用"掩码，2019 年实测可用截面 3393 只/日）→ 含成本与涨跌停/停牌过滤的
  向量化回测。**120 组合**：horizon{1,5,10,20} × 频率{D,W,M,2M} ×
  模型{ridge,gbdt,ranker,+h1 超参/窗口变体} × 中性化{on,off} × TopFrac{20%,10%}。
- **报告**（`scripts/reporting/rolling_grid_report.py` → `reports/alla_rolling/report.html`）：
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
  ② `python scripts/builders/build_alla_alpha_panels.py --workers 6`（全量重算
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
- **生产化每日推理**（2026-09-08，`scripts/pipelines/alla_daily_rank.py`，单测
  `tests/test_alla_daily_rank.py`）：最优方案每天盘后自动产出全A选股排名——
  与实验同代码路径尾部重算当年入选 50 因子 → gbdt 500 日窗重训预测最新截面 →
  幽灵股守卫 + 信号日可交易性标注 → `reports/alla_daily/`。一致性验证：与实验
  冻结 OOS 面板 2026-09-01 截面 Spearman 0.92（每日重训比实验季度折更新鲜，
  非完全一致属预期）。
