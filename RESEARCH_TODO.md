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

### 基本面因子权重过低的诊断与双频融合（09-22 研读+本机诊断完成）

**现象**（`reports/alla_daily_ortho/latest_feature_importance.csv`）：gain Top5
全量价、累计 71.5%（alpha158_KLEN 23.4%），`ln_mktcap` 5.0% 第 6，其余基本面
单因子仅 0.02–0.2%，与量价头部差 2–3 个数量级。

**诊断（已用数据证实，三层）**：
1. **期限错配（主因，D1 已证）**：`ic_h{1,5,10,20}.parquet` 分族弹性——
   fundamental mean|IC| h1→h20 ×1.143、holder ×1.150，量价族全负
   （alpha158 ×0.932 / alpha101 ×0.955 / alpha360 ×0.976 / alpha191 ×0.983）；
   个体 bp h20 |IC| 0.140（全库 24/946）、ln_mktcap 0.157（第 10）。
   两个 caveat：① `div_consecutive_years` raw |IC|=0.32 全库第一是风格假象
   （中性化后 coverage 仅 0.0009，被 MIN_COVERAGE 正确挡掉）；② selection 实录
   h1/h20 两臂每年均 7–8 基本面席位、另类族 0 席（该批 reports 早于 09-12
   另类接入，920 重跑后复核另类席位是否兑现）。
2. **正交化二次压制**：bp/ep/div_yield 的主信息恰是风格暴露本身，剥完行业+市值
   残差短周期更弱（与"信号层必须 raw"同源）。
3. **头部归因（D2）**：`latest_explain_top.csv` Top20 选股前三驱动全为量价+市值
   （alpha360_LOW0 / alpha191_070 / ln_mktcap），基本面不构成头部直接驱动。

**E1' horizon 混合回测（09-22 本机已跑，复用 882 口径 pred，固化脚本
`scripts/evaluation/horizon_mix.py`）**：口径=stage_ensemble（raw/M月频/equal/
Top10%），基线精确复现（15.52% vs 存档 15.54%）：

| 变体 | OOS IC | 年化 | 超额(指) | Sharpe | 换手 |
|---|---|---|---|---|---|
| baseline_ens_h1h5 | 0.1048 | 15.52% | +13.29pp | 0.63 | 72.9% |
| ens_h1h5h20 | 0.0978 | 16.75% | +14.52pp | 0.68 | 68.4% |
| ens_h1h5h10h20 | 0.0935 | 16.89% | +14.66pp | 0.69 | 66.6% |
| blend_h20_l25 | 0.1000 | 16.62% | +14.38pp | 0.68 | 69.6% |
| blend_h20_l40 | 0.0957 | 16.82% | +14.58pp | 0.68 | 67.2% |
| gbdt_h20_only | 0.0721 | **16.94%** | +14.71pp | 0.69 | **58.1%** |

结论：慢 horizon 混入一致改善（+1.2~1.4pp、换手 −4~15pp）；h20_only 年化最高
换手最低，但逐年非全胜（2020/2022/2025 大胜 +17.1/+25.8/+36.8 vs 2019/2024/2026
落后）、回撤 41.1% 最深；ens_h1h5h10h20 曲线最平滑。全截面 IC 降而组合升 =
慢信号优势集中在持仓头部。caveat：882 口径 + 6 变体同窗比较，正式采纳需 920
重跑后复核（含 buffer 20/30 臂）。产物 `reports/horizon_mix/`（不入库）。
**09-22 口径换主（默认 open）后 open 口径复跑已出**（`horizon_mix.py` 已接
execution 臂，baseline 14.66% 与 09-21 留存 `metrics_overall_open` 逐位一致）：
ens_h1h5h10h20 15.75%（+1.09pp）/ gbdt_h20_only 16.41%（+1.75pp）、换手 58.0%
——结论在可执行口径下保持。正名 CSV=open、`_close`=旧 close 口径。

**业界融合方案匹配（09-22 搜索）**：最匹配=信达《深度学习揭秘之一：量价与基本面
结合》"基本面偏线性+量价非线性"分域建模 → **E3**（基本面滚动 ICIR 线性慢信号 ⊕
LightGBM 快信号，预测层 λ 加权）；华泰多任务/多期限方向已被 E1' 验证有效；
DoubleEnsemble（Qlib，样本×特征重加权）符合 LightGBM 栈但侵入 predictor，列 P6；
MASTER/HIST/Transformer 系与轻量滚动架构不匹配，暂缓。

**优先级队列（刷新）**：
- [x] D1 分族 IC-horizon 弹性（09-22 完成，证期限错配）
- [x] D2 头部归因核验（09-22 完成）
- [x] E1' horizon 混合消融（09-22 完成，+1.2~1.4pp）
- [ ] **P1 E1''**：920 基线重跑后对新 pred 目录复跑 `horizon_mix.py`（已接
  execution 臂、默认 open）+ buffer(20/30) 臂，定版月频信号口径是否切到
  ens_h1h5h10h20（好机器，挂基线重跑后）
- [x] **P2 E3 双层融合（09-22 本机完成，open 口径，固化脚本
  `scripts/evaluation/fundamental_blend.py`）**：基本面族（62/63 因子，
  goodwill_ratio 面板缺）滚动 ICIR 线性慢信号（月频、embargo 2 月末、
  trailing 24 月、ICIR clip[0,3]；末月 Top 权重 bp 0.060 / float_ratio 0.057 /
  main_profit_ratio 0.046 / 单季成长族 / holder_num_chg，ICIR 噪声大→近分散
  等权）⊕ 快信号秩空间 λ 加权。结果（882 pred / M / Top10% / open）：
  **slow_only 年化 10.17%、超额上证 +7.94pp、换手 26.8%**（线性基本面合成
  自身有真 alpha）；**blend_s0.3 全指标占优基线**（年化 15.04% vs 14.66%、
  Sharpe 0.64 vs 0.59、回撤 35.4% vs 37.2%、换手 64.1% vs 73.0%）；blend_s0.5
  把 2024 大差年 −5.5pp 修到 +2.2pp（防御器，代价是强年少赚）；**与 h20 混合
  信息冗余**：h1020⊕s0.5 年化 15.02% < h1020_baseline 15.75%，但 Sharpe 0.66
  全场最高、回撤 33.5% 最低。结论：慢信号是**风险调整改进器而非增量 alpha**
  （h20 模型已隐式吃到基本面）；定版取舍=要年化选 h1020 混合、要 Sharpe/回撤
  选 h1020⊕s0.5——留 920 重跑后与 E1'' 一并裁决。
  产物 `reports/fundamental_blend/`（不入库）。
- [ ] **P3 E5**：基本面只对行业中性化臂（panels 变体+重跑，好机器）
- [ ] **P4 E4**：交互特征（基本面分桶×量价组内 zscore 显式入池，重训）
- [x] **E6 stacking + max-ICIR 对照（09-22 本机完成，open 口径）**：
  **E6**（`scripts/evaluation/stack_blend.py`，7 成员 OOS 预测滚动元学习，
  秩空间凸约束、逐月 walk-forward）：stack_fast4（4 horizon 学权）年化 15.86%
  vs h1020 等权 15.75%——**学权未胜等权**（+0.11pp 在噪声内、回撤 40.2% 更深）；
  stack_all7 12.82% 更差（元学习器在 fwd20 目标上把 slow_icir 学到 0.35、
  ridge_h1 0.19——目标错配+成员相关的小样本过拟合），**但学到的权重结构独立
  复现了诊断链结论"慢信号+h20 是最有价值成员"**。**max-ICIR**
  （fundamental_blend --weighting 双权重，收缩 Σ+active-set 迭代）：
  slow_only_max 7.87% << ICIR 版 10.17%——24 个月度 IC 样本估 50 因子协方差
  病态，华泰"max-ICIR 最优"在小样本基本面域**不成立**；blendmax_s0.5 14.97%
  略优于 blend_s0.5 但不及 blend_s0.3。**系列总结论：等权/简单规则稳健性
  压倒学权/优化（与 1/N 文献一致），定版候选收敛为 h1020_baseline（要年化）
  与 h1020⊕s0.5（要 Sharpe/回撤/换手）二选一，留 920 裁决。**
- [x] **P5 数据源盘点（09-22 完成，结论：4/5 可自建或已建，无需付费源）**：
  ① **增减持/高管持股已建好入库**——holder_dyn 族 9 因子（alt_inner_trade
  2.6 万条 + alt_mgmt_hold 3.0 万条，mgmt_netbuy 系 h20 IC 0.012~0.013 随
  horizon 走强），6 个已进 panels_neu，**但不在 FUNDAMENTAL_FAMILY_SETS
  保护席位族→静默缺席（与当初另类族同病）**；注意 inner_* 系数据 2025 年
  才开始（active_day_frac 0.167），mgmt_* 系全历史可用。**最便宜补缺动作=
  把 holder_dyn 族并入慢信号家族/席位机制，可并入 920 后 E3'' 对照**。
  ② SUE 财报版：income 表全字段（NET_PRO_EXCL_MIN_INT_INC 等 95 列）可自建。
  ③ 总资产增速：TOTAL_ASSETS ✓。④ 送转预期：CAP_RESV ✓ + TOT_SHARE ✓ +
  dividend 表（bonus_rate/base_share 送转历史）✓。⑤ capex：cash_flow 117 列
  无"购建固定资产"专项科目（可 NET_CASH_FLOWS_INV_ACT/总资产 作投资强度代理，
  或 akshare 免费接口补拉）。分析师一致预期=唯一真数据墙（tushare 未装、
  akshare 东财接口历史浅，暂缓）。
- [x] **P5 补缺构建落地——两批共 9 因子（09-22 本机完成，
  `scripts/builders/build_alla_p5_factors.py`；registry 955、panels_neu 928）**：
  第一批 5 个 + 第二批 4 个（ccc / report_delay / piotroski_f / inv_rev_gap，
  全面性审计的"最后一批"）。强弱分明：**sue_q −0.068、asset_growth_yoy −0.046、
  inv_rev_gap −0.017（压货）** 为强因子；piotroski_f −0.013（**负号=A股高质量
  反转，与美股文献相反，诚实结果**）；ccc/delay −0.007~−0.008 弱但方向符合
  预期（占用久/晚披露=负）；capex_intensity/spsr 弱、bonus_freq_3y 覆盖 0.21
  被门槛挡。9 个全部入 FUNDAMENTAL_SETS（B5 族）参加 920 席位竞争。慢信号
  全家族（81 因子）对照：slow_only 10.24%、h1020⊕s0.5 15.08%/Sharpe 0.66/
  回撤 33.4%——定版候选结论不变。**顺手修了库级 bug**：`upsert_rows`
  fill_missing_only 分支在混合批次（新+已有）时静默丢弃纯新增行
  （"新增 N"日志与实际写入不符），已修。
  基本面挖掘到此**全面性收口**：覆盖 ~95%+最后一批，剩余全是数据墙。
- [x] **基本面因子全面性审计（09-22 搜索核对，结论：覆盖 ~95%，无大类缺失）**：
  对照 [Barra CNE6 风格体系](https://www.fxbaogao.com)（16 风格：Size/Beta/Mom/
  ResVol/NLSize/Liq 量价域管辖 + Value/EarnYield/Growth/Leverage/Quality/Dividend/
  Sentiment 全有）与学术模型（[q-factor I/A+ROE](https://www.nber.org)、
  [FF5 RMW/CMA](https://english.ckgsb.edu.cn)、[Sloan 应计](https://www.anderson.ucla.edu)、
  [Piotroski F](https://alphaarchitect.com)）：A股特色（质押/商誉/预告快报/送转/
  股东户数/增减持/解禁/两融）全齐；研发/费用率因子已存在但弱且 rd 系 coverage
  0.44（不动作）。**最后一批可自建增量 4 个（全部现有字段，半天级）**：
  ccc 现金转换周期（ACCT_PAYABLE 在表）、report_delay 披露时滞（ann_date−
  report_period）、piotroski_f（9 项组件齐全差合成）、inv_rev_gap 存货异动；
  **数据墙 4 类（不做）**：分析师一致预期、审计意见、ESG、客户供应链。
  操纵性应计（修正 Jones）可近似但工程中等，列后。
- [ ] **P6**：DoubleEnsemble 式特征×样本重加权（若 P1–P3 后基本面贡献仍低）

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
| **0** | **920 治本口径基线重跑**（TODO §一，命令已写明，≈5–7h）→ **好机器长实验队列**：① AI97 三臂 ×3 seed → ② 国金24 残余⑤ 校准轮（1 臂 1 seed）→ 视耗时铺三臂 → ③ mf10 T 扫描全量 → ④ stage2 Phase 2 zz1000 全量（快档→全档，命令见 §二）；**920 重跑完成后追加 E1'' horizon 定版臂**（`python -m scripts.evaluation.horizon_mix --pred-dir <920产物目录>` + buffer 20/30 臂，§一 基本面诊断 P1）。**全部批次/命令/验收已汇总定稿 → [GOOD_MACHINE_TASKS.md](GOOD_MACHINE_TASKS.md)** | **迁移清单**：代码仓库（git clone）+ `E:\data` 数据面（因子库 parquet / min5_hs300 / 日线缓存，~GB 级；panels_neu 920 已含）+ Python 环境（系统解释器 D:/Python/Python312 有全依赖；.venv 缺 rl 依赖）+ `DEEPSEEK_API_KEY`（AI97 llm 臂需要）；产物 CSV/报告拷回本机入库归档 |
| 1 | 本机半天级穿插（§一 🥉 六项；+基本面 E3 双层叠加回测，零重训） | 均挂已有基础设施 |
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
