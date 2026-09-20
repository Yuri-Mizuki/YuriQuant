# YuriQuant 研报研读与复现待办

> 工程/研究待办见 [`TODO.md`](TODO.md)；架构与用法见 [`README.md`](README.md)。

> 2026-09-10 基于 `E:\研报` 全库（85 个 PDF，去重后约 60 篇不同报告）筛选建立。
> **2026-09-16 按代码证据核实复现状态并重排优先级**（原勾选状态全面失真，已修正）。
> **2026-09-18 增量**：西南 T2RL 研读完成；AI39 三处低成本对齐 / 多因子10 T 扫描脚本 /
> AI19+22 PBO 检验落地（本轮 108 测试全绿）。**长实验一律不跑、写成显式待办**：
> AI97 三臂 ×3 seed、国金24 残余⑤、T 扫描全量（命令见各条目）。
> **2026-09-18 二次重排**：推进顺序改为「0 跑就绪长实验 → 1 银河0706+华安226 研读 →
> 3 Mamba2 半天成本核算 → 4 AI43/AI29 穿插 → 5 大模型投研速读」，详见第六节。
> **2026-09-20 增量**：银河 0706 + 华安 226 研读完成，**stage2 复现设计定稿**
> （主蓝图=银河 PPO、env 建主动权重空间、主实验池改 zz1000、评估双门槛，
> 见 research_notes 报告第四节）；**长实验执行环境改另一台更强的机器**
> （迁移清单见第六节序 0）；新增 CVaR 约束低成本待办（华安 226 转译）。
> **2026-09-20 二批**：**stage2 Phase 0 完成**（`factor/rl/portfolio_env.py` +
> 22 用例全绿）；国金19 Mamba2 研读归档（不复现，模型层补短板改写为市场状态
> 特征 + 多模型×多标签合成）；AI43/AI29 研读完成（落地待办见序 4）。
> **2026-09-20 三批**：**P1 速读批 7 篇全部处置完毕**（国金大模型投研 4 + 招商 +
> 江海 + AI30，合篇报告见 research_notes）。剩余未研读仅：暂缓同族 RL（东方 DFQ、
> DQN，等 stage2 落地后归并）与 P2 远期清单。
> 筛选维度：① 当前（2026）业界是否仍先进/主流；② 是否落在项目四条主线
> （因子挖掘 → 因子合成/模型 → 文本因子 → 组合优化/RL）。
>
> **核心判断：全库 85 篇中 2025–2026 年仅 12 篇，恰好覆盖项目全部短板方向。**
>
> ### 状态标记
> - `[x]` 已完成复现（代码 + 产物落地）
> - `[~]` 部分完成（有基础，缺关键环节）
> - `[ ]` 未开始

---

## 〇、复现状态盘点（2026-09-16 核实，按代码证据）

**一句话结论：P0 十六条目里 6 条已完成、4 条半成品、6 条未开始；文本线（51/57/63/41）
已被另一会话整体完成，不再是待办；真正的空白集中在「端到端模型 + RL 组合优化」两处短板。**

| 研报 | 状态 | 核实证据 |
|---|---|---|
| 国金22 GFlowNet 低相关量子 | `[x]` 完成 | `factor/gflownet/`（env/net/tb/reward/parallel/selection/expr/ppo 9 模块）+ `scripts/factors/run_gflownet_phase1.py` + 118 因子入库 + 池间对比 |
| 国泰君安 GP 解构 | `[x]` 完成 | `factor/gtja.py` + `mine_factors --gp-gtja` + `scripts/evaluation/gtja_repro_eval.py`（含研报基准对标：多空 22.90%/夏普 2.33） |
| 申万 ML 合成非线性因子 | `[x]` 完成 | `scripts/archive/compare_ml_synthesis.py` 2×2 四象限（非线性×GBDT IC/IR=0.0162/2.18 最高） |
| 华泰 AI51 文本 PEAD | `[x]` 完成 | `build_sue_txt_samples` + `train_sue_txt` + `evaluate_sue_txt` + `evaluate_short_window` |
| 华泰 AI57 文本 FADT | `[x]` 完成 | `build_fadt_samples` + `train_fadt` + `evaluate_fadt` + `analyze_word_importance` |
| 华泰 AI63 BERT FADT | `[x]` 完成 | `encode_fadt_bert` / `encode_fadt_seg` / `derive_pooler_encoding` + `train_fadt_bert` + **8 种编码消融**（`evaluate_fadt_ablation.py`） |
| 华泰 AI37/41 BERT 情感 | `[x]` 完成 | `build_senti_scores` + `build_senti_factors` + `evaluate_senti` + `build_rawtone_factor`（AI63 扩展测试 5） |
| 华泰 AI14/16 时序交叉验证 | `[x]` 完成 | `scripts/evaluation/cpcv_eval.py` + `cpcv_h1_eval.py`（N=6 k=2，15 路径） |
| 华泰 AI6 Boosting 基线 | `[x]` 完成 | `model/` GBDT 已成主实验与合成对照基线 |
| 华泰 AI11 stacking | `[x]` 完成 | `model/stacking.py`（4 合成器，09-11 从 factor 层迁出） |
| **国金24 GFlowNet+AlphaEval 分钟频** | `[x]` **主体完成 + 残余①已补（09-16）** | AlphaEval 三步漏斗 ✅ `scripts/archive/factor_screening.py`；RRE ✅ `factor/gflownet/selection.py`；DPP ✅ `rolling_grid_alla.py`；**42 特征已接入 GP** ✅ `scripts/factors/mine_im_combos.py`（库内 42 `im_*` + 6 `gp_im_*`）；e2e 三臂 ✅ `reports/e2e_p0_intraday_daily/`；**GFlowNet 已接 `im_*`** ✅ `--feat-source`（09-16）+ 共享装载层 `e2e_common.attach_im_features`。残余：仅覆盖 hs300 无全A分钟数据、RRE 门槛默认关闭、挖掘预算未跑满、GFlowNet 正式实验未跑（现仅 5 iters 冒烟） |
| **华泰 AI97 大模型+RL** | `[x]` **P0 环境 + LLM 池 + 真实试跑 + 一致性核对（09-16）** | `factor/rl/` 四模块 + `scripts/factors/run_alphapool_ppo.py`；真实 deepseek-flash 验证 8/8；真实 HS300 小预算试跑完成（3000 步未收敛）；**核对报告 `reports/AI97_预算与复现一致性核对.md`：8 处不一致（3 结构性）+ 研报仅两臂 + 预算不吃资源** |
| **华泰 AI39 组合优化实证** | `[x]` 口径核对完成（09-16） | 核对报告 `reports/口径核对_AI39_多因子10.md`；**最大差异=主动权重空间 vs 绝对权重空间**（结构性）；3 项低成本对齐项 |
| **华泰多因子10 因子合成口径** | `[x]` 口径核对 + **两主力方法已补齐（09-16）** | 报告同上；`factor/synthesis.py` 新增 `synthesize_ic_ir_max`/`synthesize_ic_max`/`half_life_weights`/等权分支，26 用例；**剩余 T 扫描 + 多窗口对比实验** |
| **国金19 Mamba2 端到端选股** | `[x]` **研读归档（09-20），不复现** | 三条硬墙（分钟数据/CUDA/无超参）+ 研报自证端到端边际价值低于合成；补短板改写为市场状态特征 + 多模型×多标签合成，见 research_notes 报告 |
| **华泰3128 全频段量价** | `[ ]` **未开始** | 无多频段融合实现 |
| **华泰 LLM_FADT** | `[ ]` **未开始** | `scripts/textmining/` 无 LLM 相关脚本 |
| **西南 T2RL 端到端 RL** | `[~]` **研读完成（09-18），复现未开始** | 研读报告 `reports/docs/research_notes/T2RL_研读_两阶段RL因子挖掘与组合优化.md`：实为两阶段级联（TFAC 挖掘 + TFSAC 组合优化）；MDP 五要素 / SAC 稳定性 6 件套 / 复现边界 9 条已提取 |
| **东方 DFQ RL 因子组合** | `[ ]` **未开始** | 无 DFQ 匹配 |
| **华泰 DQN 择时** | `[ ]` 未开始 | （RL 入门基线，优先级最低） |
| **华泰 AI19/22 PBO 过拟合** | `[x]` **完成（09-18）** | `stats/pbo.py`（CSCV PBO + 缩水夏普 DSR + `deflate_best`）+ 14 用例（统计锚点：纯噪声 PBO≈0.5 / 唯一真信号 PBO=0 / DSR 零点=0.5） |
| **华泰 AI32/34 AlphaNet** | `[ ]` 未开始 | 无 alphanet 匹配 |

> ⚠️ **文本线（51/57/63/41）由另一会话整体完成**，产物在 `reports/textmining/`
> （fadt/sue 双任务 × hs300/zz1000 双池，含 README + 复现报告）。
> **不要重复投入**；若要继续，方向是升级版 LLM_FADT，且需先确认他会话是否已启动。

---

## 一、P0 待推进（按 ROI 排序）

### 🥇 第一档：半成品 / 残余收尾

- [x] ~~**国金24：42 个分钟特征接入 GP 挖掘框架**~~ —— **已于 09-14 完成，09-16 核实更正**
  （20260708，`因子挖掘/`）—— 上一版清单误判为未接入，实际证据：
  - **挖掘层已接通**：`scripts/factors/mine_im_combos.py`（**可复用脚本**，非 oneoff）
    把 17 个独立显著 `im_*` 当 GP 终端集，搜索时序×截面算子组合。
    实测 `reports/mine_im_combos_run.log`：09-14 19:49:55 → 21:42:55（**1h53m**，
    pop300/gen15，gen11 早停）→ 产出 6 条公式，全部入库 `gp_im_*`
    （`source=mining:im_combos:pop300gen15`）。
  - **库内已落地**：`E:\data\factor_library\hs300_2022_2025\registry.csv` 869 行中，
    `im_*` **42 个** + `gp_im_*` **6 个** = 48 个月内因子。
  - **e2e 已跑**：`reports/e2e_p0_intraday_daily/{with_im,no_im,placebo_im}`
    三臂齐备（with_im 24.8min / no_im 54.2min / placebo 69.7min）。
  - **资源实测**：分钟缓存 308MB（`min5_hs300.parquet`）+ MemMap 485MB，
    特征提取秒级（纯 numpy 整块向量化）→ **分钟数据层本身极轻**；
    真正耗时在下游 e2e 回测（三臂 ≈2.5h/轮）与 GP 挖掘（≈2h/轮），
    与日频实验同量级，非国金24 引入的额外负担。

- [ ] **国金24 残余（四条，均非"接入"而是"扩容 / 补交叉 / 调旋钮"）**
  - [x] ~~**① GFlowNet 接分钟特征**~~ —— **2026-09-16 完成**
    - `run_gflownet_phase1.py` 新增 `--feat-source {raw,im_independent,im_all,raw+im_independent,raw+im_all}`
      ＋ `--eval-begin/--eval-end/--keep-window`；`FEATURES` 由模块常量改为运行时动态。
    - 共享装载层 `scripts/common/e2e_common.py`：`load_im_panels`（单一真源，`mine_im_combos` 已委托）
      ＋ `attach_im_features` ＋ `aligned_eval_masks`。搬迁等价性探针
      `scripts/oneoff/_probe_im_loader_equiv.py` 逐位验证 max|Δ|=0。
    - 单测 `tests/test_gflownet_im_features.py`（20 用例）＋ 三种模式真实小预算跑通。
    - **关键坑（已固化进脚本打印）**：单靠 `--train-begin` 裁窗口会**连带改变面板列并集**
      （实测 520→421 列）与 Barra 风格参照系 → 对照臂引入第二个差异。故把评估窗口
      与面板构建解耦，raw 对照臂用 `--eval-begin/--eval-end`；已验证两臂
      面板 shape、训练段、测试段三者完全一致。
    - **已披露的复现边界**：im 特征只覆盖 HS300 每日在册 300 只中的 **255 只（85%）**，
      含 im 的公式截面天然比纯 raw 窄 → 横向比较须计入。
  - **② 全A 分钟数据** —— 唯一分钟缓存是 `min5_hs300.parquet`（仅 hs300/5min/2022–2026）。
    扩到全A 才是真正的资源墙，`scripts/oneoff/_probe_minute_throughput.py`
    就是为外推该成本而写（需真实 SDK 跑一次才有数字）。
  - **③ RRE 门槛开启** —— `--min-autocorr` 默认 0.0（关闭），已实现但未在分钟池上开过；
    分钟特征换手高，这是降换手的现成旋钮。
  - **④ 挖掘预算未跑满** —— 默认 `--pop 200 --gen 25`，上次跑 pop300 但 gen11 就早停；
    加大预算或调 `--patience` 可能出更多公式（成本 ≈2h/轮，线性可估）
    【注：GP 侧，与 ① 的 GFlowNet 侧并列】。
  - **⑤ 正式实验（① 之后的新增项）** —— 建议跑 `--iters 600 --batch 12`（研报口径）的
    im / raw＋im / raw 三臂，各 ≥3 seed。**当前只有 5 iters 冒烟，无任何可引用数字**：
    5 iters 下 im 臂 OOS |IC| 0.1009、raw 臂 0.1271，但同臂三次重复波动 0.0767~0.1009，
    噪声量级已超过臂间差异 → 不得据此下结论。
    - 成本估算（**外推，非实测**）：5 iters×4 batch 实测 54s，但其中约 40s 是面板构建＋
      均匀基线等固定开销（reward 求值约 100 次 ≈ 0.5s/次）。600 iters×12 batch ≈ 7200 次
      求值 → **粗估 60~90 分钟/轮**（子树奖励缓存命中会低于此）。三臂 ×3 seed ≈ 9~14 小时，
      建议先跑 1 臂 1 seed 校准真实耗时再铺开。

- [x] ~~**华泰 AI97 后段：接 LLM 初始因子池**~~ —— **2026-09-16 完成**
  （20251204，`因子挖掘/` `强化学习/`）
  - 研报角色分工落地：**LLM 只负责"构造基础池"与"定期去弱留强"，RL 是筛选器不是生成器**。
  - 落点：`factor/rl/llm_pool.py`（约 700 行）
    - 研报语法 ⇄ 项目语法互转：`parse_report_formula` / `to_project` / `to_report` / `canonical`；
      研报的 `Mean($close, 20)` ↔ 项目 `ts_mean_20(close)`，覆盖全部 21 个算子。
    - **3 项语义校验**，一一对应研报点名的三类 RL 失败模式：
      `trivial_terminal`（只输出单字段或套一层 Abs/Log，对应"构造过于简单"）、
      `dimension_mismatch`（价格与成交量相减，对应"不符合逻辑"）、
      `redundant_op` / `degenerate`（`Abs(Abs(x))`、`Sub(x,x)`，对应"符号冗余"）。
      量纲用 (价格指数, 成交量指数) 二元向量做加减乘除，**常数量纲为 None 与任意维度兼容**。
    - `build_prompt`：研报**未给出** prompt 原文，这是自拟提示词（已在 docstring 标注）。
    - 提案器：`TemplateProposer`（离线确定性，20 模板×窗口）与
      `OpenAICompatibleProposer`（纯 urllib，需 `DEEPSEEK_API_KEY`）；`make_proposer` 工厂。
    - 注入原语：`seed_pool`（构造基础池）+ `refresh_pool`（先剔 RL 因子再注入新因子）。
  - 池侧改动：`AlphaPool` 新增 `origins` 来源标记 / `origin_counts()` /
    **`drop_worst(origin, n)`**（只淘汰 `origin="rl"`，保证大模型因子"刚注入不被踢"；
    剔除后重优化权重，`best_obj` 保持单调不回退）。
  - 网络：`alphapool_nets.py` 提供研报图表15 的 `LSTMSharedNet` / `TransformerSharedNet`
    （`n_layers=1, d_model=128, dropout=0.1`），token 走 Embedding、
    3 个手工特征投影成**追加在序列末位的"状态 token"**（末位恒有效，两个网络读手工特征
    的方式一致 → 差异只来自时序结构本身）。
  - 训练入口：`scripts/factors/run_alphapool_ppo.py`（`--policy mlp|lstm|transformer`、
    `--llm none|template|openai`、`--llm-init/--llm-every/--drop-rl-n/--llm-new`、
    `--llm-model/--llm-base-url/--llm-max-tokens/--llm-timeout/--llm-rounds`）。
  - 单测：`tests/test_ai97_llm_pool.py`（76 用例）＋ `tests/test_htai_rl_p0.py` 全绿。
  - **真实大模型已联网验证（09-16 同日追加）**：`--llm openai --llm-model deepseek-flash`
    （key 走环境变量 `DEEPSEEK_API_KEY`），探针 `scripts/oneoff/_probe_deepseek_conn.py`。
    3 次独立调用**均 8/8 通过**语法/窗口/长度/3 项语义校验，全部入池；
    完整链路（初始池 + RL + 2 轮真实 API 定期刷新）跑通。
  - ⚠️ **推理模型陷阱（唯一的大坑，已固化进默认值）**：`deepseek-flash` 的
    `content` 与 `reasoning_content` **共享 `max_tokens`**；给 32/2000/3000 时思维链
    把预算吃光 → `finish_reason="length"` 且 **`content` 为空字符串**，而 HTTP 200、
    无异常 → 极易误判成 key/模型名错误。**实测 `max_tokens=16000` 可用**，默认值已设。
    `_post` 现在遇空 content 会显式报错并带 `finish_reason`/`reasoning_tokens` 诊断。
  - **冒烟实测（mock 合成面板，非真实数据）**：template 提案 20 条→入池 9、
    best_obj 0→0.0829；真实 DeepSeek 提案 8 条→**入池 8**、best_obj 0→0.0833，
    RL 1024 步训到池内 10 因子（llm 5 / rl 5，训练段复合 IC 0.0831）。
  - 大模型产出因子（真实）：`div(sub(open,ts_ref_1(close)),ts_ref_1(close))` 隔夜反转
    **w=0.63**、`div(ts_delta_20(close),ts_std_20(close))` 波动调整动量 w=0.47、
    量比、非流动性、5 日反转取负 —— 全是研报经典范式，窗口只用 20、量纲自洽。
  - **已披露的复现边界**：
    1. `template` 路径**不是大模型**，只是同一提示词口径下的模板兜底；
       真实调用走 `--llm openai`（已联网验证）。
    2. **研报未指明 deepseek 的具体模型版本**，本项目用 `deepseek-flash`，
       属"同族不同版本"的边界，引用数字须注明。
    3. `--llm-init` / `--llm-new` 研报**无对应超参**，属自拟参数，不得当研报口径引用。
    4. `VWAP` 用 `amount/volume` 构造（项目日线面板无该字段），是**替代口径**。
    5. `--horizon` 研报**未给出**，默认 10 交易日，引用须同时注明。
    6. 冒烟数字来自合成面板，**不可作为行情结论引用**。
  - **顺带修掉一个生产 bug**：`factor/formula.py` 的 `parse_formula` 只认整数常量，
    研报 13 个常数里 3 个是小数（`0.5 / -0.5 / -0.01`）→ 被当成特征名 → KeyError →
    **这些 token 在整个 RPN 空间里恒拿 -1 奖励**。修复后 `mul(0.5,close)` 正常求值，
    整数分支行为不变。回归用例：`test_rpn_decimal_constant_now_evaluates`。

### 🥈 第二档：补短板（模型层 30% / 优化层 20%）

- [~] **西南 T2RL** —— **研读完成（2026-09-18），复现未开始**
  （20260331，`强化学习/`）—— 报告：`reports/docs/research_notes/T2RL_研读_两阶段RL因子挖掘与组合优化.md`
  - **关键澄清**：标题"端到端"实为**两阶段级联**（阶段一 TFAC=Transformer+AC 方向奖励
    挖因子，阶段二 TFSAC=SAC 组合权重），研报展望自认"仍处于级联阶段" → 与本项目
    `optimize/`（预测→QP）是**同生态位竞争方案**；对照实验形态 =
    同一预测信号下 SAC 权重 vs QP 权重 vs Top-100 等权三臂
  - 已提取：MDP 五要素（状态=40 日窗口+预测因子简单合并 / 动作=Top-100 连续权重 /
    奖励=log 收益−θ·方差（β=50）/ episode=10 日）、SAC 稳定性 6 件套（双 Critic、
    Polyak 目标网、回放池、自动温度、奖励缩放、有界回合）、调仓两层换手控制
    （因子换手不限 / 期内双边 ≤10%）
  - **复现边界 9 条**（研报未给，引用数字必须注明）：θ 数值、SAC 全部超参、
    网络维度、全A 段费率、滚动重训细节、正文 63.35% vs 表 6 的 64.67% 不一致、
    特征可得性（换手率+Barra10 精确定义）、无任何过拟合检验、无冲击成本模型
  - **建议路线（stage2-only）→ 已被 09-20 银河 0706 研读升级定稿**：主蓝图换为
    银河 0706（PPO + 奖励 6 项 + 全超参），SAC 降为可选第四臂；env 建在**主动权重
    空间**；主实验池改 zz1000。详见 `reports/docs/research_notes/银河0706_华安226_
    研读_RL组合优化全景与stage2设计定稿.md` 第四节。本篇的 SAC 设计（Top-100 动作
    压缩、双 Critic、自动温度）保留为第四臂口径。研报数字口径（费率 0.00025、
    开盘价成交）与项目 C0 不可比

- [x] ~~**国金19 Mamba2 端到端选股框架**~~ —— **研读完成 + 归档决策（09-20）：
  不复现**。报告：`reports/docs/research_notes/国金19_Mamba2_研读_成本核算与归档决策.md`
  - **三条硬墙**：① 数据墙——合成架构 7 个子模型中 6 个依赖 60/30/10min 分钟线，
    项目只有 HS300 5min（2022 起）；② 算力墙——Mamba2 官方实现依赖 CUDA
    selective-scan 核，项目无 GPU；③ 研报**未给任何模型超参/训练配置**，
    1:1 复现自始不可能。
  - **研报自身的证据支持归档**：Mamba2 单模型对 GRU 提升有限（全A RankIC
    10.91%→11.67%），真正增益来自"频率×模型×因子"合成与指数信息注入——
    端到端模型边际价值 < 合成与市场状态特征。
  - **模型层短板（30%）补法改写**（低成本）：① 三大宽基指数点位市场状态特征
    进 GBDT 面板（研报最可搬结论，半天）；② 多模型×多标签预测合成
    （复用 `model/stacking.py` + `factor/synthesis.py`，见 AI29 笔记）；
    ③ 损失保持 MSE（研报对比结论）；④ 30min 方向挂回全A 分钟扩容资源墙之后。
  - **注意**：先量训练成本 —— Mamba2 在 CPU 环境可能不可行（项目无 GPU 描述）

- [ ] **华泰3128 全频段量价特征选股模型**
  （20231208，`因子合成/`）—— **依赖国金24 后段先完成**
  - 关联：5 分钟线 2022-01~2026-06 至今未进建模层
  - 待提取：全频段（日/分钟/分笔）如何统一建模、频段间信息融合方式
  - 落点：与 `intraday_features.py` 的"分钟→日频统计特征"路线做对照（降维 vs 端到端）

- [ ] **东方 DFQ：RL 因子组合挖掘系统**（20230817，`强化学习/`）
  - 待提取：RL 用于因子**组合**（而非个股）的系统设计、状态空间如何表达因子池
  - 定位：与西南 T2RL 同族，可合并为一轮 RL 优化研究

### 🥉 第三档：低成本校准（读研报核对，不写新代码）

- [x] ~~**华泰 AI39 组合优化实证 —— 口径核对**~~ —— **2026-09-16 完成**（`组合优化/`）
  - 交付：`reports/口径核对_AI39_多因子10.md` 第一节（13 项逐条对照）
  - **核心发现（最该动手的一处）**：研报全程在**主动权重空间** `x = w − w_b` 相对中证500
    做约束；项目在**绝对权重空间**。第 4 项最关键 —— 研报 `X_mkt·(w−w_b) = 0` 是
    「组合市值暴露 = 基准市值暴露」，项目 `|B'w| ≤ tol` 是把**绝对**暴露压到 0；
    因 B 已全A zscore，大盘基准自身 `B'w_b ≠ 0`，**两者在指增场景会给出明显不同组合**
    → 要复刻 AI39 必须引入「基准相对」口径。
  - 能力层面项目**超出**研报（QP + ERC + HRP + BL 四种，研报只有线性/二次规划）。
  - 三处低成本对齐：① 行业中性目标改基准行业权重（现默认等权）；
    ② 逐股权重变动上限 `|w − w_0| ≤ δ`（现只有组合级换手，研报控的是单股跳变）；
    ③ λ 网格 {0,0.2,0.5,1,2} —— **注意 `_alpha_score` 是 rank pct 无量纲，
    λ 有效区间与研报不同，不能照抄数值**。
  - **暂不投入**：结构化风险模型 `V = XFX' + D`（含 NW 预测期限调整 / 特征值调整 /
    贝叶斯压缩 / B 统计量），等于引入整套 Barra CNE5，工程量大 → **建议单独立项**。

- [x] ~~**华泰多因子10 因子合成方法实证 —— 口径核对**~~ —— **2026-09-16 完成**
  （20190104，`因子合成/`）
  - 交付：`reports/口径核对_AI39_多因子10.md` 第二节（12 项逐条对照）
  - **核心缺口**：研报两个**主力方法完全缺失** ——
    **最大化 IC_IR**（`w = Σ⁻¹·ĪC`，Σ=历史 IC 协方差，带 `w ≥ 0`）与
    **最大化 IC**（`w = V⁻¹·ĪC`，V=当前截面因子值相关阵）；另**无半衰加权**
    （研报 `w_t = 2^((t−T−1)/H)`）、**无 T 参数扫描**（研报 T ∈ {3,6,9,12,24,36}）。
  - 项目**更严格**的两处（须在对比时披露）：PCA 方向只用前 60% 段 SVD 防未来函数、
    `rebuild_train_weights` 只用训练段重算权重 → 研报未讨论此问题。
  - 补齐顺序（按 ROI）：**① 最大化 IC_IR**（研报主力，`estimate_covariance` 已有收缩能力，
    只需把输入从收益面板换成 IC 矩阵；`optimize/solver.py` 已有 cvxpy 解带约束 QP
    → **半天内可落地**）→ ② 半衰加权（改 `rebuild_train_weights` 一处，IC 加权与
    收益率加权同时受益）→ ③ 等权合成（一行，`weight_by="equal"`）→ ④ 最大化 IC
    （与 ① 共用求解器，只换 Σ 来源）→ ⑤ T 扫描（依赖前四项）。
  - **复现边界**：研报「历史因子收益率」是 WLS（行业+市值中性）回归产物，项目**无该链路**
    → 第 2 项（收益率半衰加权）成本远高于第 1/3/5 项，这是它排第 2 而非第 1 的原因；
    研报 6 个因子组合是**同类风格内部**合成，与项目「挖掘量价因子→合成」的相关结构不同，
    其「强相关用 PCA」的结论**不能直接搬**。

- [x] ~~**华泰 AI19 重采样 + AI22 回测过拟合概率（PBO）**~~ —— **2026-09-18 完成**
  - `stats/pbo.py`：**CSCV PBO**（Bailey et al. 2017；块聚合向量化，C(16,8)=12870
    组合秒级；ω̄=秩/(N+1)，PBO=P(λ≤0)）+ **缩水夏普 DSR**（Bailey & López de Prado
    2014；SR₀=√V[SR]·[(1−γ)Φ⁻¹(1−1/N)+γΦ⁻¹(1−1/(Ne))]）+ `deflate_best` 一步到位
  - 定位：CPCV 是模型层检验，PBO/DSR 是**策略/参数选择层**检验——互补不重复
  - 单测 14 例（统计锚点：纯噪声 PBO≈0.5、唯一真信号 PBO=0、DSR 零点=0.5、
    负偏度惩罚、试验数压显著）
  - 用法：N 个候选配置的 (T×N) 收益矩阵 → `cscv_pbo(M)` + `deflate_best(M)`；
    读法：`pbo` 低 **且** `dsr ≥ 0.95` → 选择过程可信。待办：接入现有实验产物
    （rolling_grid 各臂 / AI97 三臂）作为出报告的固定环节

---

## 二、P1 高度参考（6 篇）—— **全部处置完毕（09-20）**

- [x] ~~**银河 0706 基于深度学习预测与强化学习优化的指数增强**~~ —— **研读完成（09-20）**
  报告：`reports/docs/research_notes/银河0706_华安226_研读_RL组合优化全景与stage2设计定稿.md`
  - **超参/状态清单全给**（表7/表8），PPO+GAE、奖励 6 项分解（超额/信号暴露/风险/
    TE/换手/集中度，λ 全给出）、训练成本=0 防双重扣费 + Banach 自融资投影
  - **RL vs 凸优化直接对比一赢一输**：科创50 RL 赢（超额 12.52% vs 5.24%，权重右偏
    长尾单股>30%）、HS300 RL 输（4.26% vs 7.56%，集中度反而更低）——
    **RL 超额来自集中度自由度，成分多/偏离空间小的池子无从发挥**
  - 消融 6 组：**去实际超额收益崩到 5.85%**；"平均 reward 高 ≠ 模型好"→ 双门槛筛选
- [x] ~~**华安 226 风险规避型 RL**~~ —— **研读完成（09-20）**（学海拾珠转述，
  原文 ICT Express 2024-04；同上报告）实证仅 3 只美股，**不值得复现**；
  借鉴两点：① Dirichlet 策略处理权重单纯形（SAC 臂行动分布变体）；
  ② **CVaR 惩罚不必走 RL**——可直接做成 `solve_portfolio` 可选约束
  （RU 线性化，挂 `optimize/risk.py`，见下文新增待办）
- [x] ~~**招商证券 因子筛选与投资组合构建**~~ —— **速读完成（09-20）**，
  合篇报告见 research_notes。确认性读物无落地项（项目筛选+QP 能力均超出）；
  "逐层增量信息解释"可作 factor_analysis 辅助诊断视角备查
- [x] ~~**江海证券 机器学习在多因子组合中的应用**~~ —— **速读完成（09-20）**（同上合篇）。
  确认性读物；两条互证：时序交叉优于 K 折、**线性基线不可轻视**（其负结果大概率
  是 12 个月窗+月频+大持仓的弱设定所致，不构成对 GBDT 主栈反证，
  但新模型对照必须带线性基线）
- [x] ~~**国金 大模型赋能投研系列（4 篇）**~~ —— **速读完成（09-20）**，
  合篇报告见 research_notes（含对邮件日报链路的四条工具链借鉴：
  Skills 化判据与渐进式披露、交付三件套惯例、LLM 回测代码检查清单
  （信号日/执行日/成本/不可交易——与项目 tradable labels 口径互证）、
  技能数量上限 20–30）。可选项，非必须
- [x] ~~**华泰 AI43 因子观点融入 ML**~~ —— **研读完成（09-20）**，报告见 research_notes
  （AI43_AI29 合篇）。改 sklearn 源码的路线**不复现**（与 LightGBM 主栈不符）；
  落地项 = 预测层观点注入的**低成本等价实现**（特征复制 ×k / 两段模型），
  与已有 BL 优化层注入构成"注入点对照"实验，见第六节序 4 ③

---

## 三、P2 可选（未开工部分）

- [ ] 华泰 AI32/34 AlphaNet（神经网络因子挖掘）—— GP/GFlowNet 之外第三条路径对照（等主线完成）
- [ ] 华泰 AI26 GP 在 CTA 信号挖掘中的应用 —— 挖掘框架跨资产泛化验证
- [ ] 华泰 AI42 图神经网络选股与 Qlib 实践 —— GNN 结构化关系建模
- [ ] 华泰 AI29 另类标签与集成学习 —— **研读完成（09-20）**，报告见 research_notes
  （AI43_AI29 合篇）。不复现（CatBoost 月频场景不同）；落地项 =
  `model/labels.py` 加 **IR / Calmar 标签分支**（半天，注意研报如实披露的代价：
  另类标签的超额最大回撤**更差**，实验须同口径报回撤）+ 多标签子模型合成对照
  （复用 stacking/synthesis，零新代码）；67 种训练期长度胜率统计 = CPCV 之外的
  轻量稳健性协议，可选用
- [ ] 华泰 AI13 损失函数改进 —— lambdarank 合成细节参考
- [ ] 华泰 AI27 ML 黑箱可解释性 —— 辅助手段
- [ ] 华泰 AI40 微软 Qlib 体验 —— 工程架构参考
- [ ] 华泰 AI45 cGAN 资产配置 / AI35 WGAN 时序生成 —— 生成模型方向，研究导向可读

---

## 四、已从清单移出（不必再读）

- **已完成口径核对（研报不必再读，落地项见第六节）**：华泰 AI39（组合优化实证）、
  华泰多因子系列10（因子合成方法实证）、华泰 AI97（大模型+强化学习因子挖掘）
  —— 三者核对报告/实现见 `reports/口径核对_AI39_多因子10.md` 与 `factor/rl/llm_pool.py`
- **已完成复现**：华泰 AI21/23（GP 因子挖掘）、国金22（GFlowNet）、
  国泰君安 GP 解构、申万 ML 合成非线性、**华泰 AI51/57/63/41（文本线全部）**、
  华泰 AI14/16（CPCV）、AI11（stacking）、AI6（Boosting 基线）
- **建议不细读（历史基线）**：华泰 AI1–AI9（除 AI5 RF / AI6 Boosting 作对照）、
  AI2 GLM 作下限基线可选；AI7 Python 实战、AI10 宏观+RF、AI12 特征选择、
  AI17/18/20 工程细节、AI24/25 有效性检验、AI31/36/38/44 GAN 其余、
  AI3 SVM / AI4 朴素贝叶斯 / AI9 RNN-LSTM（业界已弃用或被取代）

---

## 五、复现范围与验收标准（每篇开工前先定）

参照项目既有惯例（Phase 0 → Phase 1 → 缺陷修复 → 收敛优化）：

1. **复现边界**：严格 1:1 复现 / 骨架 + 差异分析（须显式标注）
2. **数据可行性**：现有数据面（日 K 2019–2026、5 分钟线 2022–2026、财务三表、
   股东数据）能否支撑；缺什么、是否影响结论
3. **公平对照**：必须有统一基线（均匀策略 / GBDT / 等权），配对条件下多窗口
   一致性检验，禁止单年小样本下结论
4. **验收口径**：IC/IR + Newey-West t + CPCV 路径分布 + 换手成本 +
   **年化数字必须带臂名**（zscore 13.76% vs ortho 15.54%，同脚本不同 `--preproc`）

---

## 六、建议推进顺序（2026-09-18 二次重排）

> 排序原则：**先跑已就绪的长实验（挂机不占人力），跑批期间做研读，半天级成本核算
> 决定是否立项新复现**。2026-09-16 版的 1/2（T2RL 研读、AI39/多因子10 核对）已落地。

| 序 | 任务 | 理由 |
|---|---|---|
| **0** | **跑已就绪长实验（执行环境：移至另一台更强的机器，本机不再排队）**：① AI97 三臂 ×3 seed（命令见上文 AI97 条）→ ② 国金24 残余⑤ 校准轮（1 臂 1 seed）→ 视耗时铺三臂 → ③ T 扫描全量。**迁移清单**：代码仓库（git clone）+ `E:\data` 数据面（因子库 parquet / min5_hs300 / 日线缓存，~GB 级）+ Python 环境（系统解释器 D:/Python/Python312 有全依赖；.venv 缺 rl 依赖）+ `DEEPSEEK_API_KEY`（llm 臂需要）；产物 CSV/报告拷回本机入库归档 | 三者代码/前置全就绪；AI97 跑完接 `stats.pbo` 做过拟合检验 |
| 1 | ~~银河 0706 + 华安 226 研读~~ **完成（09-20）** | 报告见 research_notes。**stage2 复现设计已定稿**：主蓝图=银河 0706（PPO+奖励 6 项+全超参），T2RL SAC 降为可选第四臂；env 建在**主动权重空间**（同时解决 AI39 绝对/主动口径问题）；训练成本=0+回测统一扣费；逐年滚动+双窗口双门槛+多种子集成+回退；评估双门槛（"平均 reward 高≠好"）；**主实验池改 zz1000**（RL 超额来自集中度自由度，hs300 只做平价验证）；消融照抄银河 6 组 |
| 2 | ~~西南 T2RL~~ **研读完成（09-18）** | 报告见 research_notes；SAC 设计（Top-100 动作压缩/双 Critic/自动温度）保留为 stage2 第四臂口径，主蓝图已让位银河 0706（见序 1） |
| 3 | ~~国金19 Mamba2：半天成本核算~~ **研读归档（09-20），不复现** | 三条硬墙（60/30/10min 数据墙 / CUDA 算力墙 / 研报无超参）；研报自证端到端边际价值低于合成。**模型层补短板改写**：指数点位市场状态特征（半天）+ 多模型×多标签合成（复用 stacking/synthesis），见 research_notes 报告 |
| 4 | ~~穿插：AI43 + AI29 研读~~ **完成（09-20）** | 报告见 research_notes。**落地待办（均为半天级）**：① `model/labels.py` 加 IR/Calmar 标签分支（AI29，注意披露"超额回撤更差"代价）；② 指数点位市场状态特征（国金19 转译）；③ 预测层观点注入等价实现（AI43：特征复制/两段模型，与 BL 优化层注入做三臂对照）；④ 多标签子模型合成对照（复用 stacking，零新代码） |
| ~~5~~ | ~~国金大模型投研 4 篇速读~~ **完成（09-20）**（含招商/江海/AI30 全部速读批） | 合篇报告见 research_notes；工具链借鉴为可选项（邮件链路 Skills 化等四条） |
| ~~4a~~ | ~~补最大 IC_IR / 最大化 IC 合成方法 + 半衰加权~~ **已完成（09-16/09-18）** | `synthesize_ic_ir_max`/`synthesize_ic_max`/`half_life_weights` + T 扫描脚本；剩余全量扫描见序 0 ③ |
| ~~5~~ | ~~华泰 AI97 后段（LLM 初始池）~~ **已完成（09-16）** | `factor/rl/llm_pool.py` + `alphapool_nets.py` + `run_alphapool_ppo.py`，真实 deepseek-flash 联网验证 8/8 |
| 6 | 华泰3128 全频段 | 依赖分钟特征扩容，与国金24 残余②同源，资源墙后置 |
| 7 | 国金24 残余②：全A 分钟扩容 | **先跑吞吐探针估成本**，再决定是否投入 |

**新增待办（由本次口径核对产出）**：

- [~] **AI39 基准相对口径** —— **3 处低成本对齐已完成（2026-09-18）**：
  ① `industry_target_from_benchmark`（基准权重 → 行业目标，`optimize/solver.py`）；
  ② `solve_portfolio(max_weight_change=δ)` 逐股 `\|w−w₀\|≤δ`（面板级
  `optimize_weights_qp` 已透传；⚠️ NaN 退出持仓权重>δ 会不可行，docstring 已注明）；
  ③ `solve_lambda_grid`（λ 网格，缺省研报 {0,0.2,0.5,1,2}）。
  **顺手修掉两个存量问题**：`industry_target` 传 Series 触发真值歧义 ValueError
  （签名声称支持 Series 但 `industry_target or {}` 从未真正支持——AI39 目标辅助函数
  首触即炸）；`risk_aversion` docstring「越大越保守」与实现相反——项目 mvo 是
  `min w'Σw − λ·score'w`，λ 越大越**激进**，与 AI39 风险厌恶 λ 恰为倒数关系
  （λ_项目 ≈ 1/λ_AI39），已在 docstring 钉死。测试 40→46 用例。
  **剩余（单独立项）**：① 主动权重空间的结构性口径（x=w−w_b，需把约束/目标整体
  搬进主动空间，工程量中等）；② 结构化风险模型（Barra CNE5）。
- [ ] **CVaR 约束进 solve_portfolio**（华安 226 研读转译，09-20 新增，低成本穿插档）：
  历史模拟场景 + Rockafellar-Uryasev 线性化（cvxpy 可直接写），可选约束
  `CVaR_α(loss) ≤ c`；`optimize/risk.py` 已有 VaR/CVaR 度量，缺的是把它变成
  QP 约束。**不依赖 RL、独立有价值**；华安 226 本身不值得复现（3 只美股实证）。
- [x] ~~**stage2 组合优化 RL 实现——Phase 0（env 确定性骨架）**~~ —— **完成（09-20）**：
  `factor/rl/portfolio_env.py`（银河 0706 口径：主动权重空间、动作→权重映射
  clip+softmax+active_share 混合+持仓数上限、奖励六项分解 λ=表 7 缺省、
  Banach 自融资投影训练成本=0、银河表 8 状态向量 8N+14、逐决策日 gymnasium env）
  + `tests/test_portfolio_env.py` 22 用例（合成面板确定性验证）。
  Phase 1 期间追加：`tradable_masks`（非成分/不可持仓权重清零重归一）+
  回看收益窗只取期间末端日（首决策日无已实现期间收益的 off-by-one）。
- [x] ~~**stage2 组合优化 RL 实现——Phase 1（hs300 平价验证）**~~ —— **完成（09-20）**
  - `scripts/factors/run_portfolio_ppo.py`：真实数据四臂入口（PPO/QP/等权/TopN，
    统一评估口径防漂移）；env 扩展 `tradable_masks`（非成分权重清零重归一）。
  - **结果**（2023-01~2026-06，85 期，信号代理=rev5，基准=成分等权，单 seed）：
    PPO 超额 −0.82%/TE 4.12%（贴近基准）vs 等权 −0.47% vs QP −3.00%（TE 18%）
    vs TopN −2.32%。**链路自洽 ✅，且行为形态与银河 HS300 结论一致**
    （成分多/偏离空间小的池子：RL 贴基准打平、大幅主动偏离者输）——
    这正是 Phase 1 想要的验证。数字不外引（信号是代理、不调参）。
  - **踩掉三个工程坑（Phase 2 直接受益）**：① `daily_hs300.parquet` 是
    (date,code) MultiIndex；② `rolling_covariance` 的 `dropna(how="any")`
    在含数据不全历史成分的宽面板上把窗口行全杀光（513 码全 None → QP 臂
    静默退化为基准）——须逐期取「当期成分∩窗内数据完整」子集估 Σ、
    在子集内解小稠密 QP 再散回（全尺寸带零块 Σ 会让 OSQP user_limit）。
  - **Phase 2 待办（zz1000 主实验）**：信号换 GBDT h=10 预测（f）+ 风险信号
    （z，可先 −vol20）；滚动训练协议（逐年 + 双窗口双门槛 + 多种子集成 + 回退）；
    银河 6 组消融；结果接 `stats/pbo.py`。预计大算力，随长实验同机排队。
  **Phase 3（全A 规模化，Phase 2 验证 RL 超额机制后才启动）**：池=全A，
  **基准=全A 等权**（T2RL 口径——全A 无自然指数权重面板，指增语义改相对
  等权基准）；额外工程项=可交易掩码注入（全A ST/停牌量大）、动作维度 ~5800
  的训练稳定性、信号覆盖核查（分钟特征等增广信号全A 缺失）。
  设计意图：hs300→zz1000→全A 是"集中度自由度"递增梯度，三点点位可画出
  「RL 相对 QP 优势 vs 池子集中度空间」关系，比单独一个全A 结果更有信息量。
- [x] ~~**多因子10 补齐两主力合成方法**~~ —— **2026-09-16 完成前两项**：
  最大化 IC_IR（`synthesize_ic_ir_max`，`w=Σ⁻¹ĪC` + 半衰加权 + 对角收缩 + `w≥0`
  截断）/ 最大化 IC（`synthesize_ic_max`，`w=V⁻¹ĪC`，V=决策时点截面）/
  半衰权重（`half_life_weights`，逐点数值已钉死）/ 等权分支。
  测试 `tests/test_synthesis.py` 9→26 用例。**T 扫描脚本已落地（09-18）**：
  `scripts/evaluation/mf10_t_scan.py`（核心函数 `t_scan_composite` + CLI；
  防未来函数 = 训练窗再截标签实现期 horizon 天，单测验证"测试段收益错位不改权重"；
  合成面板 7 用例）。**剩余（长实验，命令已定待跑）**：全量 T 扫描
  `python -m scripts.evaluation.mf10_t_scan --dataset hs300_2022_2025 --top 8`
  + 两方法与现有 pipeline 的多窗口对比实验。
- [x] ~~**AI97 真实大模型路径**~~ —— **2026-09-16 已完成联网验证**：`--llm openai` +
  `DEEPSEEK_API_KEY`（`deepseek-flash`），3 次调用均 8/8 通过校验；详见上文「推理模型陷阱」。
- [x] ~~**AI97 真实数据小预算试跑**~~ —— **2026-09-16 完成**（HS300 后复权面板，
  `(1097,421)`，train 726 日 / test 371 日，h=10）：LLM 提案 13 → 入池 13，
  `best_obj` 0→0.0526；RL 3000 步后池 13（llm 3/rl 10），`best_obj`→0.0739；
  **训练段复合 IC +0.0535 / 测试段 −0.0538（符号翻转）**，总耗时 418.5s。
  **注**：3000 步远未收敛，此结果不构成"LLM 池无效"的证据；需跑满预算再判。
  **副产品**：暴露并修复 `vwap` 复权口径 bug（见下）。
- [x] ~~**vwap 量纲 bug**~~ —— **2026-09-16 修复**：`build_panel` 的复权只作用于
  OHLC，`amount/volume` 是原始价口径 → `vwap/close` 中位数 0.2851（应为 1）。
  `attach_vwap(panel, *, backward=...)` 现乘复权因子，实测修复后 **0.9999**。
  **凡从 amount/volume 构造价格类量都必须复权**（该特征直接进 RL 动作空间）。
- [x] ~~**AI97 预算成本 & 复现一致性核对**~~ —— **2026-09-16 完成**，
  报告 `reports/AI97_预算与复现一致性核对.md`（逐页核 34 页研报）。要点：
  **① 研报只有两臂**（RL 16.41%/7.17% vs RL+LLM 17.85%/9.78%，且 LLM 增强
  **回撤变大**＝用回撤换收益）—— **项目脚本无臂参数，"三臂"要新写**；
  **② 预算不吃资源**：10000 步≈12–15 分钟/臂（3000 步实测 418.5s），
  3 臂×3 seed≈2–2.5h，纯 CPU；**③ 8 处不一致（3 结构性）**：数据区间差一个量级
  （研报 train 2010–2018/valid/test 三段）、回测层完全不同（研报每 5 日调仓+带约束
  指增+费率 0.15%+vwap 成交）→ **研报数字与项目不可比**、大模型版本未指明；
  **④ 一致 10 项**（6 字段/13 常数/**22 算子**/`MAX_EXPR_LENGTH=15`/网络超参/
  5 档奖励/`llm_every`/`drop_rl_n`/`total_timesteps`/`n_steps`/`batch_size`/LLM 两角色）。
- [ ] **AI97 正式实验（长实验，09-18 定稿命令，待跑；执行环境移至另一台更强的机器，迁移清单见第六节序 0）**：跑满 10000 步 +
  **两臂对照（研报口径：RL alone / RL+LLM）+ 第三臂"随机池"（自造，用于分离
  "LLM 知识"与"初始池非空"效应）** × 3 seed；并核查测试段 IC 符号翻转是
  "未收敛"还是"真实样本外失效"。成本 ≈2–2.5h（09-16 核对：10000 步 ≈12–15 分钟/臂）。
  **注意**：`--policy transformer` 须用研报的 `d_model=16`（不是 LSTM 的 128）。

  ```bash
  # 跑前先 1 臂 1 seed 校准真实耗时与产物路径拼写
  for arm in none random; do for seed in 0 1 2; do
    python -u scripts/factors/run_alphapool_ppo.py --panel real --arm $arm \
      --seed $seed --total-timesteps 10000 --policy mlp
  done; done
  for seed in 0 1 2; do   # llm 臂需 DEEPSEEK_API_KEY（deepseek-flash, max_tokens=16000 已固化默认）
    python -u scripts/factors/run_alphapool_ppo.py --panel real --arm llm \
      --llm openai --seed $seed --total-timesteps 10000 --policy mlp
  done
  ```
  跑完把三臂结果接 `stats/pbo.py`（`deflate_best`/`cscv_pbo`）做过拟合检验。
  **评估口径（银河 0706 教训，09-20 追加）**：验收须**同时看**训练目标
  （`best_obj`/平均 reward）与**测试段实际组合表现**（超额/回撤/换手）——
  银河消融实证"平均 reward 高 ≠ 样本外好"（去实际超额收益组 reward 第三、
  累计超额垫底 5.85%）；模型筛选用双门槛而非单一目标值。
  - [x] **前置①：臂参数已落地（09-16）**：`run_alphapool_ppo.py --arm {llm,none,random}`
    \+ `_make_arm_proposer()`；三臂共用一条代码路径，唯一分叉点是 proposer 是否为
    `None` / 是否有知识。**设计见 `reports/docs/design/AI97_三臂实验设计.md`**。
  - [x] **前置②：冒烟通过（09-16）**：`scripts/oneoff/_probe_arm_smoke.py` —— 三臂
    接线断言全过（none 起始池=0、llm/random 非空、random proposer=uniform）。
  - [x] **前置③：修一个静默覆盖 bug（09-16）**：输出目录原为
    `..._{panel}_{policy}_{llm}`，**三臂同 `--llm template` 会互相覆盖且不报错**；
    已改为含 `--arm` 与 `--seed`。
  - [ ] **待办**：跑正式三臂 × 3 seed（≥3 seed 是硬要求 —— 早前 5-iter 冒烟的
    噪声带 0.0767~0.1009 已**超过**臂间差异本身）。
- [ ] **AI97 校准器补第四个校验（已完成，09-16）**：`scalar_operand` /
  `scalar_formula`。**这是均匀采样臂暴露的真实缺陷**：`_dim` 对常数返回 `None`
  （"与任何量纲兼容"），导致「常数出现在 ts_* 算子实参位」这类子树**量纲上永远合法**，
  `dimension_mismatch` 一条都拦不住，直到运行时才抛
  `AttributeError: 'int' object has no attribute 'rolling'`。
  **修复前实测 200 条均匀采样：73% 抛异常且校验器全部放行，真正可用仅 23.5%；
  修复后 0% 抛异常、可用 87.5%，采样效率 5.5%→14.5%。**
  探针：`scripts/oneoff/_probe_uniform_{true_yield,efficiency}.py`、
  `_probe_const_subtree_hypothesis.py`；测试：`test_constant_operand_is_rejected` 等
  （含 `test_rejected_const_operand_would_really_crash` 反向对照，防校验器过度收紧）。
- [ ] **AI97 研报消融补齐**（研报做了 5 组，项目一个没做）：**因子复杂度约束**
  （`[3,7]` 崩到 −1.56%、`[3,8]` 最高 18.71%、原始 `[0,15]` 最稳 → 极敏感，
  值得实现+对照，代码半天）/ LSTM vs Transformer / IC vs ICIR Pool /
  奖励 6 版（增量奖励最差）/ 因子数量扫描（**10 个最优**）。
- [ ] **Pool 权重优化等价性**：项目闭式解 `w∝Σ⁻¹μ`（Ledoit 50% 收缩）vs 研报
  Adam 迭代——同池同时点比较 `best_obj` / 组合 IC 是否等价；另若要跑 MeanStd Pool
  对照，需先实现 ICIR 权重优化（当前 `--metric icir` 只换**评估指标**）。

**穿插项**：~~华泰 AI19/22（PBO 过拟合检验）~~ **已完成（09-18，`stats/pbo.py`）**；
~~国金大模型投研 4 篇速读~~ **已完成（09-20）**。

**暂缓**：东方 DFQ / 华泰 DQN 择时（与西南 T2RL 同族，等 T2RL 落地后归并）；
AlphaNet / GNN / GAN 系列（P2，主线完成后再说）。

---

## 附：全库分类速查

| 目录 | 篇数 | 已完成 | 待推进 | 不细读 |
|---|---|---|---|---|
| 华泰人工智能（主目录） | 44 | 21/23 GP、14/16 CPCV、11 stacking、6 Boosting、**AI39 口径核对**、**AI29 研读（09-20）** | AI32/AI34/AI42/AI13/AI27/AI40/AI45 | ~19 |
| 因子挖掘 | 7 | 国金22、国泰君安 GP、**国金24 主体**、**华泰 AI97（P0 环境 + LLM 初始池）** | AI26、国金24 残余③④ | — |
| 因子合成 | 9 | 申万 ML 合成、**多因子10 口径核对 + 两主力方法补齐（09-16）+ T 扫描脚本（09-18）**、**国金19 Mamba2 研读归档（09-20）** | 华泰3128、**T 扫描全量 + 多窗口对比（待跑）** | AI28 |
| 文本挖掘 | 6 | **51/57/63/41 全部** | 华泰 LLM_FADT | — |
| 强化学习 | 6 | **西南 T2RL 研读（09-18）、银河0706+华安226 研读（09-20）** | T2RL stage2 复现（设计已定稿）、东方 DFQ、DQN | — |
| 组合优化 | 1 | **AI39 三处低成本对齐（09-18）** | 主动空间结构性对齐 + Barra CNE5（均单独立项） | — |
| 传统机器学习 | 7 | AI6/AI11/AI14/AI16、**AI19/AI22 PBO（09-18）**、**AI43/AI29 研读（09-20）** | —（落地待办见序 4） | 6（华泰重复归档） |
| 大模型投研 | 4 | **4 篇速读（09-20）** | — | — |
| 探索 | 1 | — | AI42 | — |

> 注：华泰篇目在多个分类目录重复归档共 15 篇，全库实际不同报告约 60 篇。
