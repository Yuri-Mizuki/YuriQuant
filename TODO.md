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
>
> 🧭 **2026-10-13 起：统一口径 + 定版登记 = 强制约定**。所有「改进 vs 基线」对照**必须**走
> `scripts/evaluation/caliber.py`（口径 7 要素唯一真源：窗口 prod=2018+ / 信号登记 / 逐股冲击成本 /
> `Config.costs()` 费率 / **几何** Sharpe / 每月首日调仓 / Σ 暖机不截断）与 `config/baselines.yaml`
> （基线登记表：`v1-dingban` 生产 · `v1w-ortho882` 本机代理）。**基线不得自造。**
> `caliber.py --check` 自检；详见 `reports/docs/design/定版主模型与统一口径.md`。

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
- [ ] **执行层摩擦 + 全A 组合优化器调查（10-13 立）**：链路优化点差距分析已完成
  （`reports/docs/design/链路优化调查_预测组合执行_2026-10-13.md`）。**结论 = 组合层
  「调参」已收口**（9 项全在 ±0.7pp 内，最大 +0.63pp），**真空白只剩 2 处**，且都在下游。
  待办（本机可跑、成本低）：**P0-a 滑点分档压力测试**（10/20/40bp，复用
  `deliverable_portfolio --slippage-bp`；已知 10→40bp 基线年化 **15.52%→9.32%**）→
  **P0-b 整手约束**（回测引擎无 `lot`，仅 `optimize/signals.py::_to_lot` 下单层 ⇒ 集中组合
  净值虚高）→ P1 全A 组合优化器对照（QP 五法/HRP 代码齐备但 `compare_portfolio_methods.py`
  **只跑过 mock**，`reports/` 无产物）→ P2-a 掩码口径自洽化拍板（T 信号/T+1 掩码/T 成交
  三日子混用，已有实测：月频噪声级、周频 +3.76pp）。附带发现两处口径未对齐：生产形态
  （全池 Top10% 等权）≠ 研究最优（TopK35~45+inv_vol，+1.9~2.3pp）**未裁决**；慢信号滞后须披露。
  **10-13 同日方案定稿（外部先进方案扫描后）**：检索组合/执行层前沿（含 **NBER w34861
  《ML Meets Markowitz》**——2010-2023 中国 A 股 5489 只 / 400+ 特征实证）⇒
  **重心从「换组合优化器」修正为「成本内生化」**：**P0** = ① 冲击成本模型替代固定 10bp
  （A股幂律/平方根，买卖非对称 ×1.5；`optimize/solver.py` 的 `quadratic_cost`/`turnover_penalty`
  **参数已存在、默认 0**，待校准）+ ② 成本感知重寻优 + ③ α 衰减 aim（Gârleanu-Pedersen 简版）；
  **关闭** MVO/BL/风险预算非等权（Chopra-Ziemba「μ 误差 ≫ Σ」+ DeMiguel 1/N + 项目 zz1000 QP
  已否，三重否证）与日内执行算法（月频 + T+1 开盘不适用）；HRP 降 P2（价值在低换手非 Sharpe）。
  核心互证：项目成本 **36bp 单边 = 0.36%**，**已在 A股论文 0.3% 临界点之上**，与项目自身
  「滑点 40bp 时集中化优势抹平 −6.2pp」同向。详见报告 §八–§九。
  **10-13 当日已按优先级落地（§十）**：
  - **P0-1 逐股冲击成本模型** ✅ 落地 + 实测（`backtest/costs.py::ImpactCostModel`、
    `engine.run(adv_panel=,vol_panel=)`、`config/settings.yaml::costs.impact_*`）。**实测推翻
    §9.2 前提**：1e8 资金下冲击 ≈ 固定 10bp（15.37% vs 15.70%，−0.33pp）⇒ 10bp 假设在 1e8
    大致正确；但**强 AUM 依赖**（1e8→1e9 掉 −4.55pp），且**集中化吃容量**（45 只 @1e8 仅 10.67%）。
    产物 `reports/impact_cost_probe/`。
  - **P0-2 成本感知优化** ✅ 工具 + 校准值（`optimize/cost_penalty.py`；1e8 下
    **k₁=2.32e-4 / k₂=6.95e-2**）。**公平成本口径下结论 = 关闭切生产**：裸 QP 输等权
    **−12.6~−15.4pp**（机制 = 集中度×平方根律，10~43 只 vs 423 只）；**成本感知惩罚自救
    +8.96pp**（9.24%→18.20%，换手 0.901→0.678，成本 14700bp→6184bp **低于等权 6236bp**，
    `scale=0.5` 即平台）——**但自救后仍差等权 −3.5pp**（18.28%/0.835 vs 21.80%/0.934），
    此时成本已非主因、差距在**收益**（不等权 206~284 只 < 等权 423 只）⇒ 与 DeMiguel 一致。
    产物 `reports/cost_penalty_probe/`。
  - ⚠️ **本版修正上一版两处错误（方向都是高估 QP，务必知悉）**：① 上一版表 1 标题写 `mvo`，
    但命令**漏了 `--method`** ⇒ 实际是 `min_var`（**真 `mvo` 在本轮之前从未跑过**）；
    ② `run_precomputed` **未传 `adv_panel/vol_panel`** ⇒ QP 臂冲击成本恒 0、只扣佣金印花
    （QP 成本被低估约 6500bp）。修正后**结论反向**（−1.5pp → −12.6~−15.4pp）。
    ⇒ **教训：`--method` 必须显式；任何 QP 臂的成本必须传 `adv/vol`。**
  - 🔴 **P0 级引擎 bug 已修（本项最重要产出）**：`backtest/engine.py::_apply_executable_mask`
    把 QP 求解器**数值尘埃负值**（−1e-9 量级）放大成全仓空头 ⇒ 组合 gross 恒为 2.0、换手突破
    long-only 上界（1.0）。修法 = 空头 gross 相对多头 **>1e-3** 才认作真空头腿（实测尘埃上界 2.1e-4）；
    等权/真实多空输入逐位不变（`max|Δ|=0`）。
    ✅ **回归扫描已完成**：`factor/synthesis.py`/`model/views.py` 仅 docstring 提及不实调；
    `run_portfolio_ppo.py`/`run_portfolio_phase2.py` 用自建评估（不走 `VectorBacktest`）
    ⇒ **无历史数字受污染**。⚠️ 但 **AI97 的 QP 对照臂**（`mvo`+`λ=5.0`、未指定 `alpha_scale`）
    本身受量级失配影响（`TE=18%` vs PPO 4.12% 即其症状），**那条对照须打 caveat**
    （不改 RL 线「整体不成立」的终判）。
  - ✅ **原「未修缺陷②」已修（P0-4）**：`solve_portfolio(method="mvo")` 目标量级失配
    （alpha 项比 `wᵀΣw` 大 ~3 个数量级）⇒ 顶点解（真实数据实测 10 只满仓、`n_active` 11↔300 摆动）。
    **修法 = 新增 `alpha_scale` 参数**：`None`（默认，**零回归、逐位一致**）/ `"auto"`
    （标准化 z + 自动标定，使 λ=1 时 α 项与风险项在等权点**梯度同量级**）/ 数值（显式）。
    效果：`nz>1e-4` 10→17 只、`n_cap` 9~10→4~6。测试 `tests/test_solver.py` 新增 8 项。
  - **P0-3 α 衰减 aim** ✅ 代码就绪（`optimize/aim.py`）；量级问题已修，但 §10.2.4 证明
    该家族瓶颈在**「集中度→成本」**（aim 同族、预期同命）⇒ 标记「待容量表有结论后再判」。
  - **P1-a 整手约束** ✅ 落地 + 实测：**降级**——只在「小资金 × 集中」显著
    （1e5×45 只 ⇒ 欠配 16.7%、−1.15pp）；≥1e7 资金下仅 0.2%，可忽略。
  - **P1-b 掩码口径** ✅ 开关落地（`rolling_grid_alla.py --mask-lag {0,1}`，默认 1 零回归，
    非默认强制 `--out-tag`）；**拍板建议**：报数走 `--mask-lag 1` + 成交改 T+1（唯一自洽且
    可执行），研究中间口径用 0；**默认不动，改动须由莉酱拍板 + 全链重跑**。
  - **生产形态**：**建议不切集中化**（§10.1 证明它吃 AUM 容量）；要切换须先做
    「集中度 × AUM」二维容量表。
  - **下一步（本轮收口后，按 ROI）**：① **P1 组合层「集中度 × AUM」二维容量表**——
    本轮证明**集中度是成本放大器**（平方根律：424 只等权 vs 10~43 只 QP ⇒ 成本差 2~3 倍），
    这把「集中化是否进生产」从**收益维**升级为**容量维**，且**主链口径可直接做**；
    ② P1 HRP 对照（预期同 QP 家族受同一「集中度→成本」约束，优先级降低）；
    ③ P2 掩码口径拍板后全链重跑（由莉酱决策）。
    **关闭**：切生产到 QP（mvo/min_var/成本感知任何变体）/ 集中化进生产 / 日内执行算法。

### 已完成归档（一行一条）

- [x] **论文族 `pap_*`21 + 情绪族 `txt_*`5 · 生产口径收尾（10-13，P0）—— 结论 = 全线封底**：
  **① 口径修正**：26 个因子的面板**实体存在**（哈希名落 `<ds>/panels/<hash>.parquet`，
  `evals/` 亦有逐日 IC + 净值，registry 的 IC/IR/t 列已填）⇒ 旧记「registry 有 / 面板无
  ⇒ 从未评估」**不成立**；真缺口 = 未进 `panels_neu/`（949 = 975−21−5）⇒ 从未中性化、
  从未与量价池 DPP 竞争。**② `txt_*` 5 个全部不达标**：自身可交易 IR（逐日 IC mean/std，
  主模型同尺度 1.076）= 0.008 / 0.020 / 0.071 / 0.101 / 0.105 ≪ **0.3 门槛**（差 3~37 倍）；
  独立组合 Sharpe 0.486~0.568 **全部低于主模型 0.631**，独立超额 3.2~5.1pp < 8.2pp。
  **③ 融合层的「Sharpe 改善」是假象**：加 `txt` 后 TopFrac0.10 Sharpe 0.631→0.678~0.747，
  20/20 格子都改善；但组合数学不允许（`txt_report_score` IR 0.008 ⇒ 应 +0.00005，实测
  +0.116），且 **IC 最弱的因子改善最大**。**安慰剂检验**（把因子值在被覆盖列内逐行随机
  置换、零信息）**复现了中位 71%（1%~121%）**，残差中位仅 1.38σ、70% 格子在 ±2σ 内、
  且随权重正负跳变 ⇒ **增益来自「覆盖率掩码 / 市值暴露」**（研报覆盖 = 大盘股，任何带此
  掩码的融合都会减波动/降回撤），**不是 alpha**。**④ `pap_*` 无一条过双条件**：
  `pap_lowvol_1y`(neu Sharpe 0.758)/`pap_smallcap`(0.747) 虽超主模型但都带市值暴露且与主
  rank 相关 0.12~0.29；`pap_rev_5d` 虽过 IC-IR 0.330 却**组合层为负**（Sharpe 0.098、
  超额 −4.7pp）——**「IC 尺度骗人」的教科书案例**；`pap_breakout_atr` 残差 IC +0.1145(t=50.5)
  是**纸面**（原始可交易 IC −0.0130）。**⑤ 多重检验**：106 项组合测试 BH-FDR q=0.05
  **仅 1 条存活且为负**。与 `four_paths` §三既判「文本线 ≈0.19 ❌」同量级同结论。
  证据 `reports/text_factor_probe/`；脚本 `scripts/evaluation/{text_factor_probe,fuse_placebo}.py`。

- [x] **分析师一致预期修正线 · 数据可得性尽调（10-13）—— 结论 = 不可立项**：
  直调东财 `reportapi.eastmoney.com/report/list`（免费/无鉴权），拿到**逐篇研报的盈利预测**
  原始字段（`predictThisYearEps` / `predictNextYearEps` / `predictNextTwoYearEps` + 评级
  `emRatingName`/`ratingChange` + 目标价 `indvAimPriceT`），全市场 **148,338 条（2017 起，
  ≈15 分钟可回补）**。但 **`predict*` 字段的历史起点 = 2024-04（当年）/ 2025-04（次年）**
  （2017–2023 全空）⇒ 当年 EPS ~30 个月、次年 EPS ~18 个月；**更硬的是每个 FY 的预测窗口
  只有「当年」（FY2024/FY2025 各 9 个月，仅 FY2026 因 2025 研报的 next 跨到 14 个月）**
  ⇒ 无法构成「横截面 × 时间」的稳定 revision 因子，**不满足准入判据（IR ≥ 0.3，需 ≥36 月）**。
  属**东财侧数据生成时点限制**，换源/自建不可绕过。附带三点：① 🚨 **`notna=1.0` 是空串假象**
  （实际覆盖 EPS 0.68 / 评级 0.67 / **目标价 0.08** / `actualLastYearEps` 0）；② **akshare 的
  `stock_research_report_em` 封装口径错误**（把 `predictThisYearEps` 按查询年硬编码列名，
  历史行被贴错年度标签）⇒ **必须直调 API**；③ **顺手发现 `txt_*` 5 因子 registry 有/面板无
  ⇒ 从未评估**（研报线的真实剩余价值）。证据 `reports/consensus_availability/`。
- [x] **P1b 复验：P1 的「GRU 优势」被推翻（10-13）**：承接 10-10 P1 结论段自留的复验要求
  （「必须在定版全 A / 920 因子口径上复验」），把**生产基线**放进 P1 的同口径
  （池/折/标签/评估逐项复用，**只换输入**），得到三个否定结论 ——
  **① 基线弱**：P1 对照的 `gbdt_raw` = 0.0388 是全表最弱一档，而生产在用的定版 50 因子
  （`y2026__h5` DPP 选中）同口径 **IC 0.0682 / IR 10.07**；**sanity check 逐位复现**
  （`gbdt_a360` 与 P1 `gbdt_raw` 的 ic_mean/ic_ir/t_nw 三者完全相同）⇒ 对照有效。
  **② 同输入反转**：把 50 因子做成 T=10 时序窗口，`gbdt_prod_seq` **0.0679** vs
  `gru_prod_seq` **0.0554（−18%）** —— 而 P1 里是 +57% ⇒ 该结论是**输入特定**的，非架构普适。
  **③ 集成被组合层否决**：GRU 与生产基线几乎正交（rank 相关仅 **0.068**，同族 seed 间 0.58），
  集成 **IC 确实 +14%**（0.0682→0.0777），但 **Top10% 超额从 0.474% 降至 0.312%/0.242%**，
  **权重扫描 w_gru=0.1~0.5 每一档都下降**（0.1 档就 −0.064pp）⇒ 不是等权稀释。
  ⇒ **「新归纳偏置」这一路在生产口径 + 可交易检验下无落地价值**；P1 报告已回改
  （§〇 修正块 / §3.7 / caveat#2 标注死因）。证据 `reports/p1b_prod_baseline/`，
  脚本 `scripts/evaluation/prod_baseline_compare.py`。
  **方法论沉淀**：「隔离归纳偏置」的对照设计会让**基线失去代表性** —— 凡给「A 比 B 强」的结论，
  B 必须是生产真在用的那个东西；等价性验证只证明口径对齐，不证明基线有代表性。
- [x] **时序预测器对照（P1）+ 目标函数扫描（P2）收口（10-10）**：补掉 10-10 盘点出的模型层
  两处真空白。**⚠️ P1 部分已被 10-13 P1b 推翻（见上条），此处保留原始记录**。
  **P1 = 本轮调查第一个明确正结果**（`reports/p1_sequence_predictor/`）：
  同输入 alpha360（6 字段×60 滞后，逐日截面 z-score）下，**GRU OOS rank IC 0.0608~0.0632
  vs gbdt 同输入 0.0388（+57%~+63%）、vs gbdt_a158（158 工程化因子）0.0493（+24%~+28%）**，
  Transformer 0.0554（+43%）；GRU seed 散布仅 0.0024 ⇒ 远超噪声。**但机理经四层控制被三次
  推翻后收敛**：① `gru_shuf`（时间轴随机置换）0.0595 ≈ 有序 ⇒ **时序顺序不是来源**；② 大 MLP
  0.0307/0.0387 **不优于** gbdt ⇒ 不是「NN vs 树」；③ 小 MLP(h=12) 训练 MSE 0.0823 ≈ GRU
  0.0825 但 OOS 差 +0.021 ⇒ **正则/欠拟合假设被否**；④ `gbdt_raw_big`(1000 树/63 叶) 0.0420
  ⇒ 不是基线欠调。**⇒ 来源 = 「循环结构逐时间步权重共享」这一架构归纳偏置**（与顺序、正则、
  容量均无关）。**旁证（最有价值）**：GRU 与 gbdt 预测秩相关为**负**（gru vs gbdt_raw −0.138 /
  transformer vs gbdt_raw −0.650）⇒ 两族提取**近乎相反**的截面成分，秩平均集成把 IC 从
  0.0632 推到 **0.0671（+6%）**、gbdt 边际≈0。**caveat**：池 = HS300 并集（RAM 33.8GB 装不下
  全 A alpha360 张量 ≈20.7GB），月频 h=5、test 2023-08~2026-08 单窗口 ⇒ **须在定版全A/920 口径
  复验**方可谈落地（候选用法 = 混入信号层集成）。**P2 正式收口**（`reports/p2_objective/`）：
  10 臂（l2/l1/huber×2/quantile50/ranker + decay/volinv/abs/day 权重 + 标签 z-score）IC
  **全落在 ±1.3e-3**，**噪声地板 1.0e-3**（数学等价对 `l1_rank` vs `quant50_rank` 实测
  ΔIC=0.0010、预测秩相关 0.9898）⇒ 目标函数这一维确认无低垂果实；**明确有害**：huber(α=0.05)
  −4.8e-3、标签换 z-score −19.1e-3。脚本 `scripts/evaluation/{seq_predictor_compare,model_objective_scan}.py`。
- [x] **四条收益提升路径全部实测收口（10-10）**：承接「量价到顶」直接证据，把候选提升
  方向压成四条并**全部跑完**——**四条全空，且缺口因同一个机理**。
  **① 风险维度 ❌**（`reports/risk_budget/`）：复利归因（算术 17.50% → 几何 15.54%，
  拖累 1.96pp）；**波动率目标化 Sharpe 0.633→0.581（负）**、市值中性化 **−6.4pp/年**
  （09-18 四臂：市值只占 28%、量价风格 72%）、行业中性化 −0.06pp、回撤状态机更差；
  **唯一改善 Calmar 的是趋势 MA200（0.402→0.512，代价 −3.8pp 年化，未在定版复验）**。
  **② 新信息源 ❌**（`reports/pead_probe/`）：**推翻「PEAD/SUE 是空白」的前提**——
  审计发现 **14 个预告/快报/SUE 因子已在 `panels_neu`**（事件族 31 个 mean|IC| 0.0068
  = 全池最弱）；独立重建 6 面板（含池内没有的**事件漂移窗**口径）复现同强度，
  **可交易 IC ≈ 全样本 IC（0.011–0.017）⇒ 非纸面但弱**；**PEAD-only 模型合成后归零**
  （IC −0.0021）、组合层最好 +0.06pp、TopK45 集中持仓仅 3.54%/回撤 54.7%
  —— 与 news 族一字不差。**③ 模型层超参 ❌**（`reports/hp_sensitivity/`）：13 配置单维扫描，
  **所有小幅调整全落在 seed 噪声地板内**（7.6e-4），**所有「更大/更快」明确劣化**
  （n_est 150→600 −70e-4、leaves 15→31 −28e-4、lr 0.03→0.06 −39e-4）⇒ **过拟合主导，
  当前点即平台**，超参这一维正式收口。**④ 不做**（已负结果，勿重投）。
  **统一机理 + 准入判据（本次最有价值的沉淀）**：$S=\sqrt{S_1^2+S_2^2}$ ⇒
  **自身 IR 0.11 的正交信号最多值 +1.2%**；要把 IR 从 0.714 提到 0.80 需 **ρ=0 且
  自身 IR ≥ 0.36** 的新收益流。**全部非量价族合起来上界仅 IR 0.738（+3.4%）**
  ⇒ 所有历史失败**不是相关性问题，是绝对强度差一个数量级**。
  **⇒ 新数据源准入门槛 = 自身可交易 IR ≥ 0.3**（而非「低相关」）。
  汇总 `reports/four_paths/README.md`。
- [x] **量价因子边际增益曲线 + 组合层短板诊断（10-10，P0+P1）**：直接测量「量价候选池
  大小 K → 性能」这条曲线（修正 `ic_headroom` 扩池同时换选择机制的不公平对照），
  **首次给「量价到顶」以直接证据**（此前只有挖掘引擎饱和的间接推断）。
  **P0 阶段 A**（池质量，零额外 IO）——量价质量是**缓坡非悬崖**（1-25 名 Y 年 OOS IC
  0.0532 → 501-791 名 0.0203 = 38%）；**P0 阶段 B**（DPP 组合口径，7 年 2019-2025，
  1960s）——**K≥200 逐位完全相同**（`mixed_K200 = K350 = K500 = K791 = 0.0702`，且
  **每一年单独看都逐位相同**）、**拐点在 K≈36**（0.0700 = 平台值 99.7%）、
  **排序仅值 +0.0025**（vs `random_K200` 均值 0.0677）、**非量价贡献 ≈ 0**
  （`pure_pv_K200` 0.0701 ≈ `mixed_K200` 0.0702；K≥200 时 49-50/50 席位是量价）。
  **P1**：模型层低垂果实已摘完（**超参从未搜索**=唯一剩余，低预期）；优化层首次补测
  inv_vol/erc + 固定 K 缓冲带 → **TopK45 + inv_vol + buf1.5 = 18.07%/Sharpe 0.736**
  （vs 等权基线 17.44%/0.687，四项同向改善），只值 **+0.63pp = 微调非新增长极**。
  ⇒ **增量必须来自「新信息源」或「新归纳偏置」**。证据 `reports/pv_marginal/` +
  `reports/portfolio_weighting/`（口径 = 882 ortho 预测，绝对数字属 882、相对结论可迁移）。
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

- [x] **大盘择时（市场状态 → 仓位乘数）可行性与收口（10-14）**：项目此前只有
  截面选股一条腿，「加择时」= 组合层乘总仓位 ω_t（选股排序不动）。工程零成本
  （`Strategy.get_weights_at(date,...)` 天然拿到日期），**但判定为不立项**。
  探针 `scripts/evaluation/timing_probe.py`（秒级，含 4 项口径自检）→ `reports/timing_probe/`：
  - **上界测试（决定性）**：日频完美 foresight 年化 **9.03%**，**低于满仓 15.54%
    达 −6.51pp** ⇒ 「提前一天知道大盘涨跌」本身对这个组合就是亏的。
  - **beta 结构**：全期 beta 0.977、年化 alpha 7.73% + beta 贡献 9.76%；
    **下行/上行捕获 = 0.923/0.982 = 0.940**（经验门槛 0.8）⇒ 上下行已对称，
    **无结构性空间**；beta 漂移仅 +0.010/年 ⇒ 也不存在「择时 beta 漂移」这个次级机会。
  - **7 条因果规则实测**（MA20/60/120/200 · mom60/120 · vol 目标 15%，含 10bp
    切换成本）：**7/7 Δ年化 < 0** ⇒ 按唯一判据全否。唯一双正的 MA20
    （ΔSharpe +0.138、MDD 38.8%→15.8%）本质是「长期半仓运行」：平均仓位 0.56、
    成本拖累 −2.73%/年，是减年化换回撤、**不是 alpha**（与 09-15 position_overlay_grid
    「趋势 MA200 减半 Calmar +27%、年化 −3.8pp」同源复现）。
  - **重启触发条件**（三条任一）：① 定版形态变（转 long-short/中性 ⇒ 择时意义自动消失）；
    ② 出现新择时信息源且**自身可交易 IR ≥ 0.3**、且先过**上界测试**（不是先写规则）；
    ③ 由莉酱明确要「低回撤版」⇒ MA20 半仓是现成方案，但须按「减年化 3.9pp 换回撤」记账。
  - **口径陷阱 4 个已踩并被 self_check 锁住**：① 🚨拿 `bench_eqw`（**日收益**）当价格
    序列比均线 = 纯前视 → 跑出 **275% 年化 / Sharpe 19.5** 的荒谬结果；
    ② 🚨成本项打印时多乘/除 1e4；③ 🚨上下行捕获写成 `mean(p|p≷0)/mean(p|m≷0)`
    **恒等于 1** ⇒ 假「完美对称」；④ 🚨上界表按固定行号取值 → 改标签匹配。
  - ⚠️ 被测臂 = **882 ortho 满仓无 buffer**（本机唯一「逐日净值+基准列」齐全实体），
    **非登记基线** `v1w-ortho882`（带 buffer 16.46%/0.685）⇒ 写进定版前须
    `--equity` 换 920 定版重跑（本机无 920 pred）。
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
- [ ] **私有名跨模块 import 存量红灯（`test_layering` 在 HEAD 即红，19 处）**：原记录「仅剩
  `scripts/textmining/*`」已过期。实测 offender 聚成 6 族：`rolling_grid_alla._ann`（被 5 个
  `scripts/evaluation/*` 引用）、`seq_predictor_compare.{_cs_z,_build_px_panels,_panel_from_vec,_row_idx}`、
  `build_alla_holder_dyn_factors.{_apply_lag,_rolling_sum_panel}`、`build_alla_altfactor_neu._classify`、
  `factor.rl.alphapool_env.{_ic_np,_panels_to_np,_standardize_panel}`、`model.market_features._expanding_z`。
  修法 = **定义模块**公开化（或给公开别名）+ 改引用方；建议一次 `mass-path-refactor` 收口
  （`factor/rl/` 涉 AI97 复现，须跑求值回归）。2026-10-13 已把 6 处**测试白盒**登记进
  `_TESTS_ONLY_PRIVATE`（先前遗漏，属既有欠债、与本轮无关）。
  → 详见 `reports/docs/design/定版主模型与统一口径.md` §六.4。

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

> **2026-10-10 战略结论（四条路全部实测）**：量价线（上游 ≥36 即平台、≥200 逐位相同、
> 组合层 +0.63pp）、风险维度、模型层超参、PEAD/SUE 新信息源 **四条全部收口**，
> 缺口因**同一机理**：$S=\sqrt{S_1^2+S_2^2}$ ⇒ 自身 IR 只有 0.11 的正交信号最多值 +1.2%；
> **已探明的「低相关」不能替代「足够强」**。⇒ **新数据源准入判据 = 自身可交易 IR ≥ 0.3**
> （见 §一 10-10 条目、`reports/four_paths/README.md`）。
> **⚠️ 10-13 更新：唯一候选「分析师一致预期修正」已尽调完毕 → 数据不可行**
> （免费源 EPS 预测历史仅 ~2.5 年且每 FY 窗口只有当年，见上「已完成归档」首条）。
> ⇒ **本阶段「新信息源」这条腿正式封底**：剩下的只有工程侧确定性收益
> （出榜链切生产）与已有资产的低成本收尾（`txt_*` / `pap_*` 从未评估）。
> 不改变「定版切生产链」的工程优先级（确定性收益，与本研究线并行不冲突）。

1. ~~**分析师一致预期修正线尽调**~~ ✅ **10-13 完成 → 不立项**（数据见上）；
   替代动作 = 评估已有 `txt_*` 5 因子（零数据成本，见下第 4 项）。
2. **定版口径切换生产出榜链**（§一，当前唯一高优先——特征源 fundind +
   h1020⊕s0.5 + buffer(15/40) 三处切换 + 双链并行对照一周）
3. 小工程收尾：e2e 族历史报告年化口径重跑（~3% 系统性偏移，队列末尾）+
   方案 B 缺面板对齐（4 列）
4. ~~论文族 `pap_*` 21 + 情绪族 `txt_*` 5 **收尾评估**~~ ✅ **10-13 完成 → 全线封底**
   （详见下方「已完成归档」与 `reports/text_factor_probe/README.md`）：`txt_*` 5 个自身
   可交易 IR **0.008~0.105 ≪ 0.3**、独立 Sharpe 0.47~0.57 < 主 0.631；融合层出现的
   +0.05~0.12 Sharpe **经安慰剂证伪**（= 覆盖率掩码/市值暴露）；`pap_*` 无一条过
   「正交+够强」；106 项 BH-FDR 仅 1 条存活且为负。**研报线（论文族+情绪族）正式封底。**
5. 趋势 MA200 overlay **在 920 定版复验**（882 口径实测 `reports/alla_rolling_ortho/position_overlay_grid.csv`：
   年化 15.58%→**11.80%**、Sharpe 0.633→**0.607**、回撤 38.7%→**23.1%**、**Calmar 0.402→0.512**。
   ⚠️ **2026-10-14 措辞更正**：此处原文写"唯一改善风险调整后收益的现存项"——**不准确**，
   它**只改善 Calmar，年化与 Sharpe 都是降的**；按「年化」判据（10-14 唯一口径指令）应直接否。
   若要复验请连同判据一起拍板）
6. 分钟频扩容吞吐探针 → 视成本决定 all_a 分钟物化与第三层扩原料
   （资源墙；华泰3128 依赖此线，RESEARCH_TODO §六）
7. 中小池组件线（可选，非主线）：tabicl 头部内加权跨池复核 +
   gbdt+tabicl 对集成组合层验证
8. 生产级执行（实盘对接、实时行情）

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
