# YuriQuant 研报研读与复现待办

> 工程/研究待办见 [`TODO.md`](TODO.md)；架构与用法见 [`README.md`](README.md)。

> 2026-09-10 基于 `E:\研报` 全库（85 个 PDF，去重后约 60 篇不同报告）筛选建立。
> **2026-09-16 按代码证据核实复现状态并重排优先级**（原勾选状态全面失真，已修正）。
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
| **国金19 Mamba2 端到端选股** | `[ ]` **未开始** | 全仓无 mamba 匹配（首轮 `ssm` 子串命中为假阳性） |
| **华泰3128 全频段量价** | `[ ]` **未开始** | 无多频段融合实现 |
| **华泰 LLM_FADT** | `[ ]` **未开始** | `scripts/textmining/` 无 LLM 相关脚本 |
| **西南 T2RL 端到端 RL** | `[ ]` **未开始** | 无 T2RL 匹配 |
| **东方 DFQ RL 因子组合** | `[ ]` **未开始** | 无 DFQ 匹配 |
| **华泰 DQN 择时** | `[ ]` 未开始 | （RL 入门基线，优先级最低） |
| **华泰 AI19/22 PBO 过拟合** | `[ ]` **未开始** | 无 deflated / pbo 匹配 |
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

- [ ] **西南 T2RL：端到端深度强化学习因子挖掘与组合优化**
  （20260331，`强化学习/`）—— **双线命中（挖掘+优化），优化层最大短板**
  - 关联：`optimize/` 当前是"两阶段（预测→cvxpy 优化）"，研报是端到端 RL
  - 待提取：状态/动作/奖励设计、与两阶段的收益对比、训练稳定性处理
  - 落点：RL 端到端组合构建（路线图第 4 项）的完整参考实现

- [ ] **国金19 Mamba2 端到端选股框架**
  （20251125，`因子合成/`）—— **模型层最大短板，"独立端到端合成模型缺失"**
  - 待提取：Mamba2 输入构造（时序面板如何喂入）、端到端 vs 两阶段差异、训练成本
  - 落点：与 TabICL / GBDT / ridge / lambdarank 做第 N 种对照；
    评估协议对齐 `scripts/evaluation/cpcv_h1_eval.py`（N=6 k=2）
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

- [ ] **华泰 AI19 重采样 + AI22 回测过拟合概率（PBO）—— 穿插项**
  - **策略级过拟合检验尚未上**（已有 CPCV 是模型层，PBO 是策略层）
  - 独立于主线，可随时补齐统计严谨性

---

## 二、P1 高度参考（6 篇，未开工）

- [ ] **银河证券 基于深度学习预测与强化学习优化的指数增强**（20260706，`强化学习/`）
  —— 与本项目 optimize/ 层工程实现最接近
- [ ] **华安证券 风险规避型强化学习模型在投资组合优化中的应用**（20250305，`强化学习/`）
  —— 对接 `optimize/risk.py`
- [ ] **招商证券 因子筛选与投资组合构建**（20181023，`因子合成/`）—— 优化层约束规范
- [ ] **江海证券 机器学习在多因子组合中的应用**（20240912，`因子合成/`）
- [ ] **国金 大模型赋能投研系列（4 篇）**（20260430–20260723，`大模型投研/`）
  —— 工具链方向，非 alpha 来源，可穿插速读
- [ ] **华泰 AI30 因果推断初探 / AI43 因子观点融入 ML**
  —— AI43 衔接本项目 Black-Litterman 观点融合

---

## 三、P2 可选（未开工部分）

- [ ] 华泰 AI32/34 AlphaNet（神经网络因子挖掘）—— GP/GFlowNet 之外第三条路径对照
- [ ] 华泰 AI26 GP 在 CTA 信号挖掘中的应用 —— 挖掘框架跨资产泛化验证
- [ ] 华泰 AI42 图神经网络选股与 Qlib 实践 —— GNN 结构化关系建模
- [ ] 华泰 AI29 另类标签与集成学习 —— 收益标签体系扩展（`model/labels.py` 可对照）
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

## 六、建议推进顺序（2026-09-16 重排）

| 序 | 任务 | 理由 |
|---|---|---|
| 1 | **西南 T2RL（端到端 RL）** | 补优化层最大短板（20%），2026-03 最新 |
| 2 | **国金19 Mamba2（端到端选股）** | 补模型层最大短板（30%），先量训练成本 |
| 3 | ~~国金24 残余①：GFlowNet 接 `im_*`~~ **已完成（09-16）** | 代码已通（`--feat-source`）。**下一步是残余⑤：跑正式三臂实验（`--iters 600 --batch 12`，≥3 seed）** |
| 4 | ~~华泰 AI39 + 多因子10 口径核对~~ **已完成（09-16）** | 报告 `reports/口径核对_AI39_多因子10.md`。**产出的两个可执行项见下行** |
| 4a | **补最大 IC_IR / 最大化 IC 合成方法 + 半衰加权** | 口径核对的高 ROI 落地项：研报两主力方法全缺，`estimate_covariance` + cvxpy 已就绪，半天可落地 |
| 5 | ~~华泰 AI97 后段（LLM 初始池）~~ **已完成（09-16）** | `factor/rl/llm_pool.py` + `alphapool_nets.py` + `run_alphapool_ppo.py`，73 用例全绿。**未做真实 LLM 联网验证** |
| 6 | 华泰3128 全频段 | 依赖分钟特征扩容，与国金24 残余②同源 |
| 7 | 国金24 残余②：全A 分钟扩容 | **先跑吞吐探针估成本**，再决定是否投入 |

**新增待办（由本次口径核对产出）**：

- [ ] **AI39 基准相对口径**（主动权重空间）—— 3 处低成本对齐：基准行业目标 /
  逐股变动上限 `|w−w_0| ≤ δ` / λ 网格遍历（λ 数值不可照抄，见报告 1.3）。
  **结构化风险模型（Barra CNE5）建议单独立项**，不塞进本轮。
- [x] ~~**多因子10 补齐两主力合成方法**~~ —— **2026-09-16 完成前两项**：
  最大化 IC_IR（`synthesize_ic_ir_max`，`w=Σ⁻¹ĪC` + 半衰加权 + 对角收缩 + `w≥0`
  截断）/ 最大化 IC（`synthesize_ic_max`，`w=V⁻¹ĪC`，V=决策时点截面）/
  半衰权重（`half_life_weights`，逐点数值已钉死）/ 等权分支。
  测试 `tests/test_synthesis.py` 9→26 用例。**剩余**：T 扫描（研报 T∈{3,6,9,12,24,36}
  个月，由调用方切 `train_dates`）与两方法与现有 pipeline 的多窗口对比实验。
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
- [ ] **AI97 正式实验**：跑满 10000 步 + **两臂对照（研报口径：RL alone / RL+LLM）
  \+ 第三臂"随机池"（自造，用于分离"LLM 知识"与"初始池非空"效应）** × 3 seed；
  并核查测试段 IC 符号翻转是"未收敛"还是"真实样本外失效"。成本 ≈2–2.5h。
  **注意**：`--policy transformer` 须用研报的 `d_model=16`（不是 LSTM 的 128）。
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

**穿插项**：华泰 AI19/22（PBO 过拟合检验）—— 独立于主线，可随时补；
国金大模型投研 4 篇速读 —— 工具链方向。

**暂缓**：东方 DFQ / 华泰 DQN 择时（与西南 T2RL 同族，等 T2RL 落地后归并）；
AlphaNet / GNN / GAN 系列（P2，主线完成后再说）。

---

## 附：全库分类速查

| 目录 | 篇数 | 已完成 | 待推进 | 不细读 |
|---|---|---|---|---|
| 华泰人工智能（主目录） | 44 | 21/23 GP、14/16 CPCV、11 stacking、6 Boosting、**AI39 口径核对** | AI32/AI34/AI42/AI29/AI13/AI27/AI40/AI45 | ~19 |
| 因子挖掘 | 7 | 国金22、国泰君安 GP、**国金24 主体**、**华泰 AI97（P0 环境 + LLM 初始池）** | AI26、国金24 残余③④ | — |
| 因子合成 | 9 | 申万 ML 合成、**多因子10 口径核对** | 国金19 Mamba2、华泰3128、**多因子10 两主力方法补齐** | AI28 |
| 文本挖掘 | 6 | **51/57/63/41 全部** | 华泰 LLM_FADT | — |
| 强化学习 | 6 | — | 西南 T2RL、银河0706、华安226、东方 DFQ、DQN | — |
| 组合优化 | 1 | — | **AI39 基准相对口径对齐（结构化风险模型单独立项）** | — |
| 传统机器学习 | 7 | AI6/AI11/AI14/AI16/AI19(部分) | AI19/AI22 PBO、AI43 | 6（华泰重复归档） |
| 大模型投研 | 4 | — | 4 篇速读 | — |
| 探索 | 1 | — | AI42 | — |

> 注：华泰篇目在多个分类目录重复归档共 15 篇，全库实际不同报告约 60 篇。
