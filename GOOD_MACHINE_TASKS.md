# GOOD_MACHINE_TASKS — 好机器大任务队列（2026-09-22 定稿；**2026-10-09 批次 1–10 全部出数收官**）

> 汇总全部需要好机器执行的大任务：批次顺序、完整命令、耗时估计、依赖与验收。
> 本机半天级穿插项不在本清单（见 TODO §一 🥉）。**约定：批次内任务可并行，
> 批次间按序；每批产物 CSV/报告拷回本机入库归档（reports/ 不进 git）；
> 全链成交口径已定版 open=T+1 可执行（正名产物），close=乐观上限对照（_close）。**
> **收官状态（10-09）**：批次 1–6、8–10 全部终判（下表），**唯一存活余项 = 批次 7
> e2e 族年化口径重跑（队列末尾，无阻塞）**；结果速览真源 `reports/batch2385_结果速览.md`
> + `reports/batch1_920_收尾对照表.md`。命令与验收口径全文保留作复现存档。
> **资源标注**：每批标 🖥️CPU / 🎮GPU / 💾大内存。

| 批次 | 内容 | 终态（10-09） | 资源 |
|---|---|---|---|
| 1 | 920 治本重跑 + 定版臂 | ✅ **定版落定**：v1 面板污染事故→v2 完整口径（h1020⊕s0.5 17.87%/0.781）→fundind 快信号 18.68%/0.813→§⑯ buffer(15/40) 终版（18.07%/0.805/换手 19.1%）；**环境漂移更正：income 补齐后同配置 ~17.0–17.2%** | 🖥️💾 CPU 密集 + 大内存（全A 面板 float32 ~24GB + 滚动工作副本） |
| 2 | AI97 三臂 ×3 seed | ✅ **负结果归档**：PBO=0.704 / DSR 不显著；100k 步归因=真实失效非未收敛 | 🖥️ 纯 CPU |
| 3 | GFlowNet 正式实验 | ✅ **raw+im 定版形态**（0.117 > im 0.109 > raw 0.099，臂间差>种子噪声）；RRE 中性、预算饱和；114 因子入库 PBO=0.114 | 🖥️ CPU 可跑 |
| 4 | mf10 T 扫描全量 | ✅ 出数：两法均 T=3 最优、单调变差，NW t≤1.4 不显著——不构成升级依据 | 🖥️ CPU |
| 5 | stage2 Phase 2 zz1000 | ✅ **负结果归档**（全档 24 臂）：PPO 三年不一致、消融臂普遍反超 full、QP 仅 2024 正超额——RL 线整体不成立（与批次 2 互证） | 🖥️ CPU 多核 |
| 6 | 口径修复重训 E5/E4/P5 | ✅ E5 fundind **采纳为定版成分**（+0.81pp）；E4 池竞争+强制注入双证伪（+0.14pp 噪声级）；P5 两批 9 因子收口 | 🖥️💾 同批次 1 |
| 7 | e2e 族年化口径重跑 | ⬜ **唯一存活余项**（队列末尾，未实测耗时） | 🖥️ CPU |
| 8 | 表格基础模型对照 | ✅ 三级收官：hs300（tabicl_w3m 头部内加权 11.97%/0.739 组件冠军）/ zz1000（tabicl 跨池最优 + gbdt+tabicl 对集成 20.03%/1.25 新发现）/ 全A（**ICL 优势不跨大截面，gbdt IC 4.4 倍反超，主模型基座维持 gbdt**） | 🎮 GPU（已执行） |
| 9 | 东吴 LLM-MCTS Phase 0 | ✅ all-A 正式跑（750 候选收编 14、PBO=0.039、OOS≥IS）→ 入库 llm_* → 席位 4/14 **唯一超门槛引擎** → 组合层 +0.16pp 噪声级**不采纳**——席位突破、组合无增量，归档（hs300 29 Seed 四臂协议未按原样执行，跨引擎对照由既有线结论替代，如实记录） | 🖥️ CPU + LLM API |
| 10 | 920 后收尾对照包 | ✅ horizon_mix/stack/fundamental_blend/buffer/T 窗/income 复核/alt 终审全出（§⑫–⑯）；member_blend12 12 成员复验因 h1020 已裁决采纳不再必要 | 🖥️ CPU |

---

## 〇、当前结论快照（2026-10-09，全部 open 可执行口径）

1. **主模型定版（§⑭⑯）**：**fundind 快信号 h1020⊕慢信号 s0.5 + buffer(15/40)
   执行层**——TopFrac 月频口径 18.68%/Sharpe 0.813/回撤 34.0%/换手 49.6%，
   buffer top20 口径 18.07%/0.805/**换手 19.1%**（帕累托占优原 20/30：
   +1.20pp/Sharpe +0.06/换手 −9.9pp，剂量-反应单调非网格运气）。
   慢信号标准面板 T=24、fundamental 族滚动 ICIR 线性。
2. **环境漂移更正（10-09）**：income 补齐（4383 只）已永久改变 panels_neu/慢信号
   家族权重——同配置如今重跑约 **17.0–17.2%**（v3/v4 双证），v2 存档 18.68%
   属补齐前数据态；**不回退数据换数字**，引用注明数据态（income 4383 前后）。
3. **负结果清单（与正结果同权重归档）**：AI97 RL 挖掘（PBO 0.704 + 100k 步真实
   失效）；stage2 RL/QP 组合优化（24 臂无稳健超额，reward 分量不稳健）；标签
   工程三臂（gauss_rank 零增量 / co、ir 判负）；E4 交互双路径（池竞争 + 强制注入
   +0.14pp）；alt 三族（席位 0/36 + 慢信号并入 −2.4pp）；GP/GFlowNet 预算饱和；
   LLM-MCTS 组合层（+0.16pp 噪声级）；mf10 T 扫描（不显著）；tabicl 全A 大截面
   （ICL 不跨大截面）。**现有特征空间上公式重组合已触及上限，增量需新信息源
   或新归纳偏置。**
4. **组件层资产**：tabicl 头部内加权 = 中小池（hs300/zz1000）组合组件
   （11.97%/0.739，H3 跨池复核与 gbdt+tabicl 对集成验证留中小池场景）；
   gf_* 114 因子（PBO 0.114）与 llm_* 14 因子（PBO 0.039）= 分散器/家族池
   资产（与慢信号同定位）；Alt 三族中 news 反转方向与 macro 择时特征留未来
   新信号线。
5. **诊断链存档（882 时代结论，已被 v2 部分翻案）**：基本面权重低主因 = 期限
   错配（fundamental h1→h20 ×1.143）；「ortho 二次压制」的两次测量——v1 污染期
   E5 仅 +0.03pp（基本面族缺席时无从生效，该「证伪」不成立），v2 完整口径下
   fundind 快信号 **+0.81pp**（快信号侧压制真实）；慢信号侧反之偏好标准中性化
   （fundind 慢信号 11.49% < 标准 12.84%）——定版取 fundind 快 + 标准慢的组合。

---

## 一、迁移清单（一次性，缺一不可）

| 项 | 内容 |
|---|---|
| 代码 | `git clone` master（≥ `93d7b0a`） |
| 数据面 | 整个 `E:\data`：`parquet/`（原始表）+ `factor_library/all_a_2018_2026/`（panels + panels_neu 920 已含 + registry + ic_h*.parquet）+ `min5_hs300` + 日线缓存；~GB 级 |
| Python | 系统解释器（本项目机为 D:/Python/Python312，含全部依赖；**.venv 缺 rl 依赖**，RL 测试/实验必须用系统解释器） |
| 密钥 | `DEEPSEEK_API_KEY`（批次 2 的 llm 臂、批次 9 MCTS 均需要） |
| 硬件建议 | 内存 ≥64GB（批次 1/5/6 全A 面板 + 滚动工作副本）；批次 8 需 GPU：单卡 ≥8GB 显存（TabPFN-3 官方口径单张 H100 推理 1M 行，hs300 规模 8GB 足够） |
| Python 补装（仅批次 8） | `pip install tabpfn==9.0.0`（TabPFN-3.5）+ `pip install tabicl==2.2.0`；HF 镜像 `HF_ENDPOINT=https://hf-mirror.com`（torch 用系统解释器现有版本即可）；**载荷包别动**（pandas 2.x 等保持原版，见批次 8 版本口径） |

---

## 批次 1：920 治本口径基线重跑 + 全部定版臂（≈8–10h，一切定版裁决的前提）

> ✅ **已收官（09-24~10-08，含事故更正与三轮定版演进）**：v1（ortho920tl）出数即
> 发现基本面族 99 张面板网格污染（§⑨）→ 重建后 v2（ortho920v2）完整口径定版
> （§⑩-⑪ h1020⊕s0.5 17.87%/0.781）→ fundind 快信号 +0.81pp（§⑫①a，18.68%/0.813）
> → §⑭ 拍板 + §⑯ buffer(15/40) 终版。**P0 income 补齐复核负结果（§⑮）+ 环境漂移
> 更正（同配置 ~17.0–17.2%）**。全部数字与裁决见 `reports/batch1_920_收尾对照表.md`。
> 下方命令存档备复现。

### 1.1 主跑（≈5–7h：select 36 轮 ≈2h + predict ≈2–3h + ensemble/backtest ≈0.5h）

```bash
python scripts/pipelines/rolling_grid_alla.py --preproc ortho --out-tag ortho920tl \
   --exclude-features limit_pos --tradable-labels --execution open
```

三开关：`--out-tag` 产物另存（禁覆盖旧 alla_rolling_ortho）；`--exclude-features
limit_pos` 剔纸面因子（与生产出榜链对齐）；`--tradable-labels` 标签治本（T 日封板
→T+1 买不进的样本按 T+1 成交口径掩掉）。`--execution open` 与新默认同值、自文档。

### 1.2 执行价对照臂（主跑完成后各 +0.5h）

```bash
python scripts/pipelines/rolling_grid_alla.py --preproc ortho --out-tag ortho920tl \
   --exclude-features limit_pos --tradable-labels --stage backtest --execution close
python scripts/pipelines/rolling_grid_alla.py --preproc ortho --out-tag ortho920tl \
   --exclude-features limit_pos --tradable-labels --stage backtest --execution vwap
```

（close 臂兼保 h>1 native 结算诊断行；open 正名产物仅含 h=1 行——定版口径全在其中。）

### 1.3 E1'' horizon 定版 + E3'' blend 定版（各 ≈2–10min，复用 1.1 的 pred）

```bash
python -m scripts.evaluation.horizon_mix \
    --pred-dir reports/alla_rolling_ortho920tl --out reports/horizon_mix_920
python -m scripts.evaluation.fundamental_blend \
    --pred-dir reports/alla_rolling_ortho920tl --out reports/fundamental_blend_920
python -m scripts.evaluation.stack_blend \
    --pred-dir reports/alla_rolling_ortho920tl --out reports/stack_blend_920
python -m scripts.evaluation.member_blend12 \
    --pred-dir reports/alla_rolling_ortho920tl --out reports/member_blend12_920
```

### 1.4 buffer(20/30) 臂

对 1.3 胜出信号加 `BufferedTopFracLongOnly(entry 0.20/exit 0.30)` 臂
（复用 `scripts/evaluation/buffer_tune.py` 思路；对照 naive top20：882 口径实测
buffer 年化 25.32%/换手 53.9% 优于 naive 24.50%/66.1%）。

### 1.5 附带复核（零额外训练）

- 另类族 6 席位是否兑现（882 时代 selection 0 席，疑因 reports 早于 09-12 接入）；
- `goodwill_ratio` panels_neu 缺面板（920 补齐清单外）；
- `stage_backtest/stage_ensemble` eq 缓存 exists-skip（旧 eq CSV 跳过回测 → metrics
  静默陈旧）——新目录首跑无此问题，核对即可。

### 1.6 收尾必做（TODO §P0 全文为准）

① 新旧对照表（`prod_pipeline_gap` 2x2 格式），主口径列 = 正名 `metrics_overall.csv`
（open）；② open 口径年化变化 >±1pp ⇒ 更新 MEMORY 主实验口径段、旧数字补「882 口径」
标注；③ `_extra` 方案 B 落地；④ 治本臂不及旧基线先看 `--tradable-labels` 单开关
消融——**不允许因「治本臂数字低」回退标签掩码**；⑤ 定版二选一裁决（结论快照 #2）
+ holder_dyn 族并入慢信号家族对照（见 §批次 6 P5 落地）。

---

## 批次 2：AI97 三臂 ×3 seed（≈2–2.5h，纯 CPU，可与批次 1 后半并行）

> ✅ **已收官（09-25 出数、09-26 步长归因）——负结果归档**：九臂臂间差异 < seed
> 噪声、PBO=0.704、DSR 不显著；100k 步归因 = 真实样本外失效非未收敛。详见
> `reports/batch2385_结果速览.md` 批次 2 节 + 步长归因补充节。命令存档备复现。

```bash
# 跑前先 1 臂 1 seed 校准真实耗时与产物路径拼写
for arm in none random; do for seed in 0 1 2; do
  python -u scripts/factors/run_alphapool_ppo.py --panel real --arm $arm \
    --seed $seed --total-timesteps 10000 --policy mlp
done; done
for seed in 0 1 2; do   # llm 臂需 DEEPSEEK_API_KEY
  python -u scripts/factors/run_alphapool_ppo.py --panel real --arm llm \
    --llm openai --seed $seed --total-timesteps 10000 --policy mlp
done
```

跑完接 `stats/pbo.py`（deflate_best/cscv_pbo）；验收双门槛（训练目标 + 测试段组合
表现）；3000 步试跑测试段 IC 翻转是"未收敛"还是"真实失效"在此裁决。
（复现边界与陷阱清单见 RESEARCH_TODO §一 AI97 节，含 max_tokens=16000 已固化。）

## 批次 3：国金24 残余⑤ GFlowNet 正式实验（校准 1–1.5h → 全量 9–14h）

> ✅ **已收官（09-25 三臂出数，10-05/06 ③④旋钮+⑤入库收官）**：raw+im 0.117 >
> im 0.109 > raw 0.099（臂间差>种子噪声）为定版形态；RRE≥0.3 中性（0.113）、
> 预算 2x 饱和（0.1189）；114 个 gf_* 入库（33 个 |t_nw|>2）、PBO=0.114。
> 详见 `reports/batch2385_结果速览.md` 批次 3 节 + RRE/预算/入库三补充节。

```bash
# 先 1 臂 1 seed 校准真实耗时（60~90 分钟/轮为外推值）
python -m scripts.factors.run_gflownet_phase1 ... --iters 600 --batch 12   # im / raw＋im / raw 三臂
# 校准后铺三臂 ×3 seed
```

（当前只有 5 iters 冒烟、无任何可引用数字；`--min-autocorr` RRE 门槛与挖掘预算
旋钮见 RESEARCH_TODO §一 国金24 节。）

## 批次 4：mf10 T 扫描全量（半天级）

> ✅ **已出数**（`reports/mf10_t_scan/`）：两法（ic_ir_max / ic_max）均 T=3 最优、
> T 增大单调变差；NW t ≤1.4 不显著——不构成对现 pipeline 的升级依据，负结果存档。

```bash
python -m scripts.evaluation.mf10_t_scan --dataset hs300_2022_2025 --top 8
```

两方法（IC_IR 最大 / IC 最大）与现 pipeline 多窗口对比。~~已知 caveat~~
**已清（09-24）**：09-16 的"决策时点截面"修复在「全时段面板 + train_dates」
调用形态下未生效（V 恒取 `panel.index[-1]` = 测试段末）——已改为决策时点
跟随 `train_dates[-1]` + 回归锚（`tests/test_synthesis.py::
test_ic_max_v_decision_point_follows_train_dates`），mf10 T 扫描可放心全量。

## 批次 5：stage2 Phase 2 zz1000 全量（快档 6–9h → 全档 1–2 天）

> ✅ **已收官（10-05 全档 24 臂终判）——负结果归档**：修复链三连（env er 可交易
> 基准 / 门槛 zz1000 重校 / 断点持久化+降级）后，PPO full +1.07/+0.25/−3.17 三年
> 不一致、2025/2026 消融臂普遍反超 full、QP 仅 2024 正超额——**银河口径 RL/QP
> 在 zz1000 无稳健超额，RL 线整体不成立**（与批次 2 互证）。Phase 3 不启动。
> 详见 `reports/batch2385_结果速览.md` 批次 5 快档/终局更正/env 修复/门槛重校/全档判读五节。

```bash
# 快档：先出三臂主对照（约 6-9 小时）
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

**前置工程债已清（09-24）**：`portfolio_env`/QP 臂接入 `data/tradability`
（`build_tradable_mask` 新增 `execution_lag=0` T 日状态口径，决策日收盘调仓
自洽）；`make_env` 掩码 = 成分 ∩ 可交易 + 整行全 False 回退；QP `valid` 过滤
同口径 + 信号 reindex 兜底（mdd_66 长标签尾部覆盖不足不再 KeyError）。
hs300 全臂冒烟通过（`reports/portfolio_phase2_mask_smoke`）。跑完接
`stats/pbo.py`、判读守双门槛。

## 批次 6：口径修复重训实验（依赖批次 1 裁决，各 ≈5–7h）

> ✅ **已收官**：E5 fundind（v2 完整口径下 +0.81pp，**采纳为定版特征源**；v1 污染期
> +0.03pp 系基本面族缺席无从生效）；E4 池竞争证伪 + 强制注入臂 +0.14pp 噪声级
> （ortho920e4force，结案）；P5 两批 9 因子收口（sue_q −0.068 最强）。

- **E5 基本面只对行业中性化臂**（panels_neu 变体：基本面族跳过市值中性化，
  保价值/红利风格信息）——检验"ortho 二次压制"诊断，单变量对照；
- **E4 交互特征**（基本面状态分桶 × 量价因子组内 zscore 显式入池）——检验
  "基本面当条件变量"假设；
- **P5 收尾（09-22 全部构建入库：两批 9 因子，sue_q/asset_growth_yoy/
  inv_rev_gap 强、piotroski_f 负号=A股质量反转、其余弱；已入 B5 族+慢信号
  81 因子家族）**：剩 holder_dyn 族并入 FUNDAMENTAL_FAMILY_SETS 的拍板
  （本机对照温和改善）与 E5/E4 重训对照。

## 批次 7（队列末尾）——⬜ 唯一存活余项

- e2e 族报告对齐年化口径（`perf_stats` 244→252 后 ~3% 系统性偏移）。

## 批次 8：表格基础模型对照 🎮（TabPFN-3.5 / TabICL 2.2 vs LightGBM；09-22 立项，09-23 版本口径升级）

> ✅ **已收官（三级：hs300 全量 → 头部诊断/组合 → zz1000 复验 → 全A 冒烟裁决）**：
> ① hs300：tabpfn3/tabicl OOS IC 全面高于 gbdt（0.055/0.051 vs 0.042）但 IC 优势
> 未传导组合层；头部诊断确认 ICL 优势在截面中腰部——H1 三臂集成被否、**H3 头部
> 内加权成立（tabicl_w3m 11.97%/0.739 组件冠军，两窗口稳健）**；② zz1000：tabicl
> 跨池最优 + **gbdt+tabicl 对集成 20.03%/1.25 全场最优**（集成可行性依池而定）；
> ③ 全A 大截面：**ICL 优势不跨大截面**（gbdt IC 4.4 倍反超）——主模型基座维持
> gbdt，tabicl 定位中小池组件。详见 `reports/batch2385_结果速览.md` 批次 8 三节 +
> `reports/batch1_920_收尾对照表.md` §⑫③。

**背景**：项目 2026-08 已实测 TabICL v1（ICML 2025，`model/predictor.py::TabICLPredictor`
基建现成——ICL 语义、chunk 推理、device 参数齐全）：**滚动 3 个月是唯一 4 窗口全正 IR 的
方法（+2.48~+4.88，均值≈3.9）**，静态全年在 2025→2026H1 切换期翻车（−2.72）——ICL 模型
与滚动短窗口天然适配。**v1 结论只作方向假设来源（09-22 由莉酱拍板：有 V2 不做 v1）**——
v1 的遗留局限（test_step=3 抽样高估、HS300 单池、47 因子、n_estimators=2）**直接在 V2 上
修复，不在 v1 上补验**：版本差异与抽样差异混在一起会污染对照，且在旧模型上加固的数字
对用 V2 的正式对照没有参考价值。**版本线**：TabPFN-3（2026-05-12，arXiv:2605.13986，
Prior Labs）→ **TabPFN-3.5（2026-09-15 随 tabpfn 9.0.0 发布，成为默认模型**，1M 行 ×
2 万特征、KV cache、SHAP 提速；TabArena 单次 forward 1850 Elo 为 v3 数字，对照：
LightGBM tuned+ensembled 1600 / TabICL V2 default 1700 / TabPFN-2.5 1550）。
License = 研究与内部评估免费（TabPFN-3.5 license，本项目研究用途无碍）。
**本次对照全部用最新版：TabPFN-3.5（tabpfn 9.0.0）+ TabICL 2.2.0（V2 大版本），
不测任何旧版（09-23 版本口径升级，替代 09-22 的"TabPFN-3 + TabICL V2"表述）。**

**版本口径（09-23 定，生产机按此装）**：本机 09-23 已实测升级通道全通——
`tabpfn==9.0.0`（PyPI 2026-09-15 发布，**默认模型 = TabPFN-3.5**，`ModelVersion.V3_5` /
`V3_5_FAST` 在 `tabpfn.constants`；3.5 支持 1M 行 × 2 万特征、CPU 上限 5000 行，与 v3 相同）
+ `tabicl==2.2.0`（V2 大版本最新补丁）。License 全部就绪（本机 API 直推 accepted：
v3 串 `tabpfn-3-5-license-v1.0` 与 3.5 串 `tabpfn-3.5-license-v1.0` 均已接受）；
生产机复用同一 token（`TABPFN_TOKEN` 环境变量）或到 ux.priorlabs.ai License 页自取。
**注意：升级/安装时 uv 会连带拉新 pandas/torch/sklearn——生产环境载荷包（pandas 2.x 等）
必须钉回原版本，仅 tabpfn/tabicl 装新版（两者依赖下限兼容旧 pandas）**，09-23 本机
踩过一次 pandas 3.0.6 令 portfolio_env 测试连环挂的坑。

**臂设计**（滚动协议沿用 08-08 结论）：
- 臂 1：`gbdt`（现役基线，同窗口滚动对照）；
- 臂 2：`tabpfn35`（TabPFN-3.5 = 9.0.0 默认模型；CPU 兜底参数照抄 TabICLPredictor：
  n_estimators=2 起步、chunk 推理；如 3.5-Fast 在同窗口耗时显著更优则加测该臂）；
- 臂 3：`tabicl_v2`（tabicl 2.2.0 同口径跑；v1 的旧数字不进对照，只作
  「ICL+滚动短窗口值得测」的方向依据）；
- 臂 4：`tabpfn35+gbdt 秩平均`（h1h5 集成同款逻辑：两模型族信号秩相关 <0.3 时
  ensemble 才有增量空间——先算秩相关再决定此臂价值）。
- 窗口扫描：W∈{2m, 3m}（08-08：2m 强趋势更强、3m 唯一全正，须在 920/全A 复验）。

**评估口径**：可交易掩码 + 可交易 IC（防纸面三判据）；OOS IC + NW t；test_step=1
全量测试日（v1 的 IR 高估问题直接在 V2 上修掉，不在 v1 上补验）；组合层对照
（Top10% 等权，口径=批次 1 正名 open）。

**两步走**：
1. **本机冒烟（半天，CPU 可跑，不必等好机器）✅ 已完成（09-22）**：hs300 单窗口单 W、
   n_estimators=2，校准 tabpfn v3 API 与耗时，验证 `TabICLPredictor` 模式可平移（新
   Predictor 类半天）。**实测校准数据**（`reports/tabpfn3_smoke/`，quick 模式 2024H2
   训练 → 2025–2026H1 测试，~6000 训练样本）：
   | 臂 | IC | IC_IR | fit | pred（CPU） |
   |---|---|---|---|---|
   | gbdt | 0.0241 | 2.84 | 1.5s | 0.5s |
   | tabpfn3（v3 权重） | 0.0243 | 1.59 | 169.8s | 1659.6s |
   | tabicl | 0.0271 | 1.80 | 1.0s | 864.4s |
   - **API/环境已全通**：tabpfn 8.4.0（冒烟当时最新，**7.1.1 不含 v3**，ModelVersion 只到
     V2_6）+ license 走 `TABPFN_TOKEN` 环境变量（API 直接 POST `/account/license`
     {"version":...} 即接受，无需浏览器）；CPU 大样本须 `ignore_pretraining_limits=True`
     （v3 上限 5000 行，v2.x 是 1000）。权重 ckpt 已缓存 `%APPDATA%\tabpfn\`。
     **09-23 已再升 tabpfn 9.0.0（3.5）+ tabicl 2.2.0，冒烟数字为 v3 权重所出，正式对照
     用 3.5 重跑（冒烟只作耗时量级与 API 校准依据）**。
   - **耗时外推**：CPU 上 tabpfn3 pred ≈ 27.7min/窗口（6000 训练样本 × 5000 测试行）
     → 全量逐月滚动不可行，**GPU 是批次 8 硬前置**；tabicl CPU 减半（864s）也撑不住
     全A 逐月。冒烟 IC 层面三臂同量级（0.024–0.027），v3 未见优势——方向假设（ICL+
     滚动短窗）仍待 GPU 全量裁决。
2. **好机器全量（本批次）**：全A（~5549 股 × 逐月滚动 × test_step=1）+ hs300/zz1000
   多池 + 多窗口。耗时未知——**先跑 hs300 全量外推**再决定全A 是否铺满。
   runner 新写（复用 PREDICTORS 注册 + `scripts/evaluation` 滚动协议，半天工程量）。

**预期管理（诚实标注）**：TabArena 数字是通用表格 benchmark，不等于 A 股截面有优势；
v1 实测在 HS300 上静态 IR 最差（−2.72）、全靠滚动短窗口翻盘（方向假设）→ 对照的假设是
「TabPFN-3 更快 + 更大 context 后，滚动短窗口协议下的优势能否复现并放大到全A」，
**假设可能不成立，负结果同样归档**（参照 ml_algorithm_compare 先例：TabICL v1 0.029
< gbdt 0.051，静态口径已被证伪过一次）。

**与批次 1 的关系**：正式对比用 920 新口径 pred 与特征池；冒烟不依赖批次 1。

---

## 批次 9：东吴 LLM-MCTS Phase 0（同题四引擎对照；**代码已就绪 09-24**）

> ✅ **已收官（10-08/09）——席位突破、组合无增量，归档**：正式跑落在 all-A 域
> （口径变更：all_a_2019_2026、3 seed × 50 iters，hs300 29 Seed 四臂协议未按原样
> 执行——跨引擎对照由既有线结论替代：GP/GFlowNet 饱和、AlphaPool 失效）。
> 750 候选收编 14、PBO=0.039、OOS≥IS；`build_alla_llmmcts_factors` 入库 llm_* 后
> v4 席位 4/14（唯一超门槛引擎）；组合层 v4 17.17% vs 同环境 v3 17.01% = +0.16pp
> 噪声级**不采纳**。含环境漂移更正（income 补齐后同配置 ~17.0–17.2%）。
> 详见 `reports/batch2385_结果速览.md` P-A 节 + 终审节、`reports/llm_mcts_phase0/`。
> 设计真源：`reports/docs/research_notes/东吴0623_研读_LLM_MCTS因子迭代框架与Phase0设计.md`；
> 立项理由与复现边界见 RESEARCH_TODO §一 🥈。

### 9.1 前置（已完成，09-24 本机）

- [x] `factor/mcts/` 五模块：`seeds.py`（29 Seed = Alpha158 rolling 全类 × w=20，
  **求值走原生 ALPHA158 callable 零转译风险**，展示伪代码仅供 LLM 阅读）、
  `reward.py`（周度六项 reward + 防前视切片评测器）、`tree.py`（UCT +
  virtual expansion + 深度上限）、`proposer.py`（MCTS 上下文 prompt + LLM
  扩展器 + 离线变异兜底 `SeedMutationProposer`）、`engine.py`（主循环 +
  三层去重：静态校验/结构同族→logic_review→**周度 IC 相关 ≥0.99 数值去重**
  （RD-Agent 机制①）+ 机制④失败换向 + 机制⑦ JSON 纪律双防线）
- [x] `scripts/factors/run_llm_mcts.py`：四臂 runner（mcts / llm_oneshot /
  gp / gflownet，gp/gflownet 复用项目引擎、报告层统一走周度六项评测）；
  双口径验收（正式选择对标 65.5% / 候选池诊断对标 75.4%）+ 两两相关分布 +
  PBO/DSR 自动出数
- [x] 冒烟：`--panel mock` 四臂全通（mcts 0.5s / oneshot 4s / gp 42s /
  gflownet 74s）+ **真实 hs300 离线缓存冒烟通过**（vwap 复权重建自检
  中位数 1.0003；Corr(close,volume,20) IS/OOS RankIC −0.031/−0.031 方向一致，
  `reports/llm_mcts_phase0/mcts_template_0924_115617_realsmoke/`）
- [x] `tests/test_llm_mcts.py` 23 用例全绿（含 **IS/OOS 子树缓存隔离回归锚**：
  `formula_builder` 的 node_cache 键不含面板，共用会把 IS 面板错配给 OOS）
- ⚠️ 已知边界：本机 hs300 日线缓存 2019 起（设计口径 2015），`--is` 切片
  自动适配为 2019-2023；好机器若拉全 2015 起缓存则 IS 自动扩满，无需改参

### 9.2 全量（命令定稿，09-24）

```bash
# 主臂（LLM 扩展 + logic_review 自动开；需 DEEPSEEK_API_KEY）
python -u scripts/factors/run_llm_mcts.py --panel hs300_2015_2026 \
  --is 2016-01-01:2023-12-31 --oos 2024-01-01:2026-08-21 --horizon 5 \
  --arm mcts --iterations 10 --variants 5 --max-depth 3 --seeds-n 29 \
  --llm openai --out reports/llm_mcts_phase0
# 对照臂（不需要 key，template=离线变异兜底；GP/GFlowNet 走项目引擎）
for arm in llm_oneshot gp gflownet; do
  python -u scripts/factors/run_llm_mcts.py --panel hs300_2015_2026 \
    --arm $arm --iterations 10 --variants 5 --seeds-n 29 \
    --out reports/llm_mcts_phase0
done
# 链路自检（零数据/零网络，秒级）：--panel mock --arm mcts --smoke
```

（`--seeds-n 29` = 东吴全量；试点可先 `--seeds-n 3`。llm_oneshot 臂想测真实
LLM 时加 `--llm openai`。全A 920 平行臂待接 panels 数据面后另行挂载。）

### 9.3 口径与验收

- hs300_2015_2026 主 + 全A 920 平行；IS 2016-2023 / OOS 2024-2026-08-21；**h=5 对齐东吴**
  （非生产口径）；搜索 reward → top50 全量复评（不抽样）；**IS 选型、OOS 只验证**
- 双口径对标东吴：正式选择 OOS 提升比例 vs **65.5%**；候选池诊断率 vs **75.4%**
- 四臂比双优公式数 / 去重唯一数 / 两两相关性 / token 成本；接 `stats/pbo.py` 出 PBO/DSR
- 成本粗估：≈1450 次 LLM 扩展/臂 + 1450 次面板评测（可多进程）；token ≈ AI97 llm 臂量级
- **否定结果同样归档**：若 llm_oneshot ≈ mcts，结论即「LLM 语义生成为主、MCTS 控制增益有限」
- 高频部分不做（min5_hs300 资源墙）

---

## 附：本机已完成实验速览（2026-09-22 诊断批——882 口径，v2 定版裁决见 §〇 快照与收尾对照表 §⑩-⑯）

| 实验 | 一句话结论 | v2/920 后终态 |
|---|---|---|
| D1 分族 IC-horizon | 期限错配证实：fundamental +14.3%/holder +15.0%（h1→h20），量价全负 | 诊断成立，h1020/h20 模型成为定版成分 |
| D2 头部归因 | Top20 选股前三驱动全为量价+市值，基本面不直接驱动头部 | 成立 |
| E1' horizon 混合 | h1020 等权 +1.1pp（close）/+1.09pp（open），全截面 IC 降而组合升 | 920 复验 +3.31pp（§②），采纳 |
| E3 双层融合 | slow_only +7.9pp 真 alpha；blend_s0.3 全指标占优基线；慢信号=防御分散器 | v2 slow_only 12.84% 印证；⊕s0.5 进定版信号 |
| E6 stacking | 学权未胜等权；权重结构独立复现"慢信号+h20 最有价值" | 920 复验同向（stack_fast4 +0.56pp 噪声级） |
| max-ICIR | 小样本域劣于简单 ICIR（7.87% vs 10.17%）——负结果归档 | 结论不变 |
| P5 数据盘点 | 4/5 可自建或已建；增减持因子已入库但缺席席位；唯一数据墙=分析师类 | 两批 9 因子收口（09-22 落地） |
