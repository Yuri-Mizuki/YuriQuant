# YuriQuant 研报研读与复现待办

> 工程/研究待办见 [`TODO.md`](TODO.md)；架构与用法见 [`README.md`](README.md)。
> **细节真源**：研读报告全在 `reports/docs/research_notes/`（本文件只留指针与待办）；
> 逐日日志 `.workbuddy/memory/`。

> 2026-09-10 基于 `E:\研报` 全库（85 个 PDF，去重后约 60 篇）筛选建立；09-16 按代码证据
> 核实复现状态；09-18/09-20 五批推进（P1 全部处置 + stage2 设计定稿 + Phase 0/1 落地 +
> Phase 2 管线冒烟）；**09-21 增量**：panels_neu 补齐至 920 → 主实验数字脱钩，
> 治本口径重跑定为最高优先级（见 TODO §一，先于本文件全部长实验）。
> **长实验一律不跑、写成显式待办，执行环境 = 另一台更强的机器**。

> **核心判断：全库 85 篇中 2025–2026 年仅 12 篇，恰好覆盖项目全部短板方向。**
>
> ### 状态标记
> - `[x]` 已完成复现（代码 + 产物落地） · `[~]` 部分完成 · `[ ]` 未开始

---

## 〇、复现状态盘点（2026-09-16 核实，按代码证据）

**一句话结论：P0 十六条目里 6 条已完成、4 条半成品、6 条未开始；文本线（51/57/63/41）
已被另一会话整体完成；真正的空白集中在「端到端模型 + RL 组合优化」两处短板。**

| 研报 | 状态 | 核实证据 |
|---|---|---|
| 国金22 GFlowNet 低相关量子 | `[x]` | `factor/gflownet/`（9 模块）+ `run_gflownet_phase1` + 118 因子入库 |
| 国泰君安 GP 解构 | `[x]` | `factor/gtja.py` + `mine_factors --gp-gtja` + `gtja_repro_eval`（研报对标：多空 22.90%/夏普 2.33） |
| 申万 ML 合成非线性因子 | `[x]` | `scripts/archive/compare_ml_synthesis.py` 2×2 四象限 |
| 华泰 AI51/57/63/37/41 文本线 | `[x]` | `scripts/textmining/` 全链（FADT/SUE/BERT/情感）；⚠️ **另一会话维护，不要重复投入** |
| 华泰 AI14/16 CPCV / AI11 stacking / AI6 Boosting | `[x]` | `cpcv_eval` + `cpcv_h1_eval`；`model/stacking.py`；GBDT 已成主基线 |
| 国金24 GFlowNet+AlphaEval 分钟频 | `[x]` 主体 + 残余 | AlphaEval 漏斗 / RRE / DPP / 42 `im_*` 接 GP / e2e 三臂 / GFlowNet 接 im；**残余见 §一** |
| 华泰 AI97 大模型+RL | `[x]` P0 完成 | `factor/rl/` 四模块 + `run_alphapool_ppo` + LLM 池真实联网验证；**正式三臂实验待跑（§一）** |
| 华泰 AI39 组合优化实证 | `[x]` 口径核对 | `reports/口径核对_AI39_多因子10.md`；最大差异=主动 vs 绝对权重空间；3 项低成本对齐已落地 |
| 华泰多因子10 合成口径 | `[x]` + 补齐 | 两主力方法（`synthesize_ic_ir_max`/`synthesize_ic_max`）+ 半衰加权 + T 扫描脚本；全量扫描待跑 |
| 银河 0608 时序截面三层预测 | `[x]` 研读 + L1 落地 | L1 结论：**风险标签可测成立**（mdd test 0.25/0.40）；L2 不触发，倾向归档；L3 并入 stage2 |
| 西南 T2RL 端到端 RL | `[~]` 研读完成 | 两阶段级联（TFAC+TFSAC）；SAC 设计保留为 stage2 可选第四臂；复现未开始（归并 stage2 后再议） |
| 国金19 Mamba2 | `[x]` 研读归档，**不复现** | 三条硬墙（分钟数据/CUDA/无超参）；模型层补短板改写为市场状态特征 + 多模型多标签合成 |
| 华泰 AI19/22 PBO | `[x]` | `stats/pbo.py`（CSCV PBO + DSR + deflate_best，14 用例）；待接入实验产物 |
| 银河 0706 + 华安 226 | `[x]` 研读 + stage2 定稿 | **RL vs 凸优化直接对比一赢一输 → RL 超额来自集中度自由度**；stage2 主蓝图定稿并已落地管线 |
| 华泰3128 全频段量价 | `[ ]` | 依赖全A 分钟扩容（资源墙） |
| 华泰 LLM_FADT | `[ ]` | 需先确认他会话是否已启动 |
| 东方 DFQ / 华泰 DQN | `[ ]` | RL 同族，暂缓，等 stage2 落地后归并 |
| 华泰 AI32/34 AlphaNet 等 | `[ ]` | P2（见§三） |

---

## 一、P0 待推进（按 ROI 排序）

> ⚠️ **920 治本口径重跑（TODO §一）优先于本节全部条目**——它是所有全A结论可信引用的前提。

### 🥇 国金24 残余（非"接入"而是"扩容 / 补交叉 / 调旋钮"）

- **② 全A 分钟数据**：唯一分钟缓存是 `min5_hs300.parquet`；扩到全A 才是真正的资源墙，
  `scripts/oneoff/_probe_minute_throughput.py` 为外推成本而写（需真实 SDK 跑一次）。
- **③ RRE 门槛开启**：`--min-autocorr` 默认 0.0（已实现未开过）；分钟特征换手高，
  这是降换手的现成旋钮。
- **④ 挖掘预算未跑满**：默认 `--pop 200 --gen 25`，上次 pop300 但 gen11 早停；
  加大预算或调 `--patience`（≈2h/轮）。
- **⑤ 正式实验（GFlowNet）**：`--iters 600 --batch 12`（研报口径）的 im / raw＋im / raw
  三臂各 ≥3 seed。**当前只有 5 iters 冒烟，无任何可引用数字**（臂间差异 < 噪声带）。
  成本粗估（外推）：60~90 分钟/轮，三臂 ×3 seed ≈ 9~14h，**先跑 1 臂 1 seed 校准**。

### 🥇 AI97 正式实验（长实验，命令已定稿，待好机器）

两臂对照（研报口径：RL alone / RL+LLM）+ 第三臂"随机池"（自造，分离"LLM 知识"与
"初始池非空"效应）× 3 seed；核查测试段 IC 符号翻转是"未收敛"还是"真实样本外失效"
（3000 步试跑：训练段 +0.0535 / 测试段 −0.0538，**未收敛前不构成"LLM 池无效"证据**）。
成本 ≈2–2.5h（10000 步 ≈12–15 分钟/臂，纯 CPU）。

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

跑完接 `stats/pbo.py`（`deflate_best`/`cscv_pbo`）做过拟合检验。
**评估口径（银河 0706 教训）**：验收须**同时看**训练目标（`best_obj`/reward）与
测试段实际组合表现（超额/回撤/换手）——"平均 reward 高 ≠ 样本外好"，双门槛筛选。
**注意**：`--policy transformer` 须用研报的 `d_model=16`（不是 LSTM 的 128）。

复现边界（引用数字必须注明）：① `template` 不是大模型，真实调用走 `--llm openai`；
② 研报未指明 deepseek 具体版本，项目用 `deepseek-flash` 属同族不同版本；③ `--llm-init`/
`--llm-new` 为自拟参数；④ VWAP 用 amount/volume 构造是替代口径（须复权——`build_panel`
复权只作用 OHLC 的 bug 已修）；⑤ `--horizon` 研报未给，默认 10 交易日；⑥ 研报只有两臂
且回测层完全不同（研报数字与项目不可比）；⑦ **推理模型陷阱**：`content` 与
`reasoning_content` 共享 `max_tokens`，思维链吃光预算 → 空 content + HTTP 200 无异常，
实测 `max_tokens=16000` 可用（已固化默认）。

### 🥈 多因子10 T 扫描全量（长实验，命令已定）

`python -m scripts.evaluation.mf10_t_scan --dataset hs300_2022_2025 --top 8`
+ 两方法（IC_IR 最大 / IC 最大）与现有 pipeline 的多窗口对比实验。
防未来函数 = 训练窗再截标签实现期 horizon 天（单测已锁）；**已知 caveat**：`synthesize_ic_max`
曾用全样本 V（隐式 look-ahead）——全量跑前先核对该点。

### 🥉 补短板低成本项（半天级，本机穿插）

- [ ] **指数点位市场状态特征进 GBDT 面板**（国金19 转译，半天）：三大宽基指数点位。
- [ ] **多模型×多标签预测合成对照**（复用 `model/stacking.py` + `factor/synthesis.py`，零新代码）。
- [ ] **`model/labels.py` 加 IR/Calmar 标签分支**（AI29，半天；注意研报如实披露的代价：
  另类标签的超额最大回撤**更差**，实验须同口径报回撤）。
- [ ] **预测层观点注入等价实现**（AI43：特征复制×k / 两段模型，与 BL 优化层注入做三臂对照）。
- [ ] **CVaR 约束进 solve_portfolio**（华安 226 转译）：历史模拟场景 + RU 线性化，
  可选约束 `CVaR_α(loss) ≤ c`；`optimize/risk.py` 已有度量，缺的是变成 QP 约束。
  不依赖 RL、独立有价值。
- [ ] **AI97 研报消融补齐**（5 组）：因子复杂度约束（研报示 `[3,8]` 最高 18.71%、
  `[3,7]` 崩 −1.56%，极敏感）/ LSTM vs Transformer / IC vs ICIR Pool / 奖励 6 版 /
  因子数量扫描（10 个最优）。
- [ ] **AI39 剩余（单独立项）**：① 主动权重空间结构性口径（x=w−w_b，工程量中等）；
  ② 结构化风险模型（Barra CNE5）。低成本 3 件套已落地（09-18）。

---

## 二、RL 组合优化 stage2（银河 0706 复现线）—— 主线定稿

> 报告真源：`reports/docs/research_notes/银河0706_华安226_研读_RL组合优化全景与stage2设计定稿.md`

**设计要点**：主蓝图=银河 0706（PPO+奖励 6 项+全超参）；env 建在**主动权重空间**
（同时解决 AI39 绝对/主动口径问题）；训练成本=0（Banach 自融资投影）+回测统一扣费；
主实验池改 **zz1000**（RL 超额来自集中度自由度，hs300 只做平价验证）；逐年滚动 +
双窗口双门槛 + 多种子集成 + 全败回退；消融照抄银河 6 组；T2RL SAC 降为可选第四臂。

**进度**：
- [x] Phase 0：`factor/rl/portfolio_env.py`（22 用例，含 tradable_masks + 回看窗 off-by-one 修复）
- [x] Phase 1：`run_portfolio_ppo.py` hs300 平价验证——PPO 贴基准打平（超额 −0.82%/TE 4.12%）
  vs QP −3.00%（TE 18%）/TopN −2.32%，行为形态与银河 HS300 结论一致 ✅
  （踩坑存档：`rolling_covariance` 的 `dropna(how="any")` 在宽面板上把窗口行全杀光 →
  须逐期取「当期成分∩窗内完整」子集估 Σ 再解小稠密 QP）
- [x] 前置 L1（银河 0608）：`compute_galaxy_features`（36 特征）+ `build_galaxy_labels`
  + `galaxy_l1.py` → **mdd_66 稳定显著（test 0.25/0.40）**；sharpe/alpha 年度翻符号 →
  质量门槛+回退是必要设计；L2 不触发
- [~] **Phase 2（zz1000 主实验）：管线完成（09-20），冒烟通过，待好机器全量**
  `scripts/factors/run_portfolio_phase2.py`：三标签 GBDT 信号（质量门槛/回退）+ PPO
  银河滚动协议（双窗双门槛/多种子 Softmax 集成/全败回退）+ 银河口径 QP（主动空间+
  TE≤10%+换手≤20%+top5 后处理）+ EW/TopN 基准 + 消融组。冒烟：全链路通、
  银河 QP zz1000 2024 超额 +3.28%、no_alpha 消融装配正确。

**生产命令（好机器执行）**：

```bash
# 快档（约 6-9 小时）：先出三臂主对照
python -m scripts.factors.run_portfolio_phase2 --pool zz1000 \
  --begin 2019-06-01 --end 2026-06-30 --model-years 2024,2025,2026 \
  --arms ppo,qp,ew,topn --ablations full \
  --ppo-seeds 5 --ppo-retries 3 --ppo-timesteps 50000 \
  --out reports/portfolio_phase2
# 全档（约 1-2 天）：+ 银河消融组
python -m scripts.factors.run_portfolio_phase2 --pool zz1000 \
  --ablations full,no_alpha,no_risk,no_excess,no_te_turn \
  --ppo-timesteps 100000 --out reports/portfolio_phase2_full
```

跑完接 `stats/pbo.py` 做选择偏差检验；判读守双门槛口径。
**前置工程项**：涨跌停/停牌掩码注入 `data/tradability`（TODO §三）。

**Phase 3（全A 规模化，Phase 2 验证 RL 超额机制后才启动）**：池=全A、基准=全A 等权
（T2RL 口径）；工程项=可交易掩码（全A ST/停牌量大）、动作维度 ~5800 训练稳定性、
信号覆盖核查。设计意图：hs300→zz1000→全A 是"集中度自由度"递增梯度，
三点画出「RL 相对 QP 优势 vs 集中度空间」关系。

---

## 三、P2 可选（未开工部分）

- [ ] 华泰 AI32/34 AlphaNet——GP/GFlowNet 之外第三条路径对照（等主线完成）
- [ ] 华泰 AI26 GP 在 CTA 信号挖掘——跨资产泛化验证
- [ ] 华泰 AI42 图神经网络选股——GNN 结构化关系建模
- [ ] 华泰 AI13 损失函数 / AI27 可解释性 / AI40 Qlib / AI45 cGAN / AI35 WGAN——辅助参考

---

## 四、已从清单移出（不必再读）

- **已完成口径核对**：华泰 AI39、多因子10、AI97（核对报告见 `reports/口径核对_AI39_多因子10.md`
  与 `reports/AI97_预算与复现一致性核对.md`）
- **已完成复现**：华泰 AI21/23 GP、国金22 GFlowNet、国泰君安 GP、申万 ML 合成、
  文本线全部（51/57/63/41）、AI14/16 CPCV、AI11 stacking、AI6 Boosting
- **建议不细读（历史基线）**：AI1–AI9 大部分、AI10/12/17/18/20/24/25/31/36/38/44、
  AI3/4/9（业界已弃用或被取代）

---

## 五、复现范围与验收标准（每篇开工前先定）

参照项目既有惯例（Phase 0 → Phase 1 → 缺陷修复 → 收敛优化）：

1. **复现边界**：严格 1:1 / 骨架 + 差异分析（须显式标注）
2. **数据可行性**：现有数据面能否支撑；缺什么、是否影响结论
3. **公平对照**：必须有统一基线（均匀策略 / GBDT / 等权），配对条件多窗口一致性，
   禁止单年小样本下结论；**长实验 ≥3 seed**（5-iter 冒烟噪声带可超臂间差异）
4. **验收口径**：IC/IR + Newey-West t + CPCV/PBO + 换手成本 +
   **年化数字必须带臂名与口径版本**（zscore 13.76% vs ortho 15.54%，同脚本不同 `--preproc`；
   ⚠️ 09-21 起 15.54%/15.52% 为 **882 面板口径**，与 920 当前库脱钩——引用旧数字须带
   「882 口径」标注，治本重跑落新数字后以新数字为准）

---

## 六、建议推进顺序（2026-09-21 刷新）

> 排序原则：**先跑已就绪长实验（挂机不占人力），跑批期间做研读，半天级成本核算决定立项**。

| 序 | 任务 | 说明 |
|---|---|---|
| **0** | **920 治本口径基线重跑**（TODO §一，命令已写明，≈5–7h）→ **好机器长实验队列**：① AI97 三臂 ×3 seed → ② 国金24 残余⑤ 校准轮（1 臂 1 seed）→ 视耗时铺三臂 → ③ mf10 T 扫描全量 → ④ stage2 Phase 2 zz1000 全量（快档→全档，命令见 §二） | **迁移清单**：代码仓库（git clone）+ `E:\data` 数据面（因子库 parquet / min5_hs300 / 日线缓存，~GB 级；panels_neu 920 已含）+ Python 环境（系统解释器 D:/Python/Python312 有全依赖；.venv 缺 rl 依赖）+ `DEEPSEEK_API_KEY`（AI97 llm 臂需要）；产物 CSV/报告拷回本机入库归档 |
| 1 | 本机半天级穿插（§一 🥉 六项） | 均挂已有基础设施 |
| 2 | ~~研读批~~ **全部完成**：银河 0706+华安 226（09-20）、西南 T2RL（09-18）、国金19 Mamba2 归档（09-20）、AI43/AI29（09-20）、大模型投研速读批（09-20） | 报告均见 `reports/docs/research_notes/` |
| 3 | 华泰3128 全频段 | 依赖分钟特征扩容，资源墙后置 |
| 4 | 国金24 残余②：全A 分钟扩容 | 先跑吞吐探针估成本，再决定是否投入 |

**暂缓**：东方 DFQ / 华泰 DQN 择时（等 stage2 落地后归并）；AlphaNet / GNN / GAN 系列（P2）。

---

## 附：全库分类速查

| 目录 | 篇数 | 已完成 | 待推进 | 不细读 |
|---|---|---|---|---|
| 华泰人工智能（主目录） | 44 | 21/23 GP、14/16 CPCV、11 stacking、6 Boosting、AI39 核对、AI29 研读、AI19/22 PBO、AI43 研读 | AI32/34/42/13/27/40/45 | ~19 |
| 因子挖掘 | 7 | 国金22、国泰君安 GP、国金24 主体、AI97 P0、银河 0608 研读 | AI26、国金24 残余②③④⑤、AI97 正式实验 | — |
| 因子合成 | 9 | 申万 ML、多因子10 补齐+T 扫描脚本、国金19 归档 | T 扫描全量+多窗口对比、华泰3128 | AI28 |
| 文本挖掘 | 6 | 51/57/63/41 全部 | 华泰 LLM_FADT | — |
| 强化学习 | 6 | T2RL 研读、银河0706+华安226 研读、stage2 Phase 0/1 | **stage2 Phase 2 全量**、东方 DFQ、DQN | — |
| 组合优化 | 1 | AI39 三处低成本对齐 | 主动空间结构性对齐 + Barra CNE5（单独立项） | — |
| 传统机器学习 | 7 | AI6/11/14/16、AI19/22 PBO、AI43/AI29 研读 | —（落地项见 §一 🥉） | 6 |
| 大模型投研 | 4 | 4 篇速读 | — | — |
| 探索 | 1 | — | AI42 | — |

> 注：华泰篇目在多个分类目录重复归档共 15 篇，全库实际不同报告约 60 篇。
