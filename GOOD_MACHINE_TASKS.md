# GOOD_MACHINE_TASKS — 好机器大任务队列（2026-09-22 定稿，09-23 增补批次 9）

> 汇总全部需要好机器执行的大任务：批次顺序、完整命令、耗时估计、依赖与验收。
> 本机半天级穿插项不在本清单（见 TODO §一 🥉）。**约定：批次内任务可并行，
> 批次间按序；每批产物 CSV/报告拷回本机入库归档（reports/ 不进 git）；
> 全链成交口径已定版 open=T+1 可执行（正名产物），close=乐观上限对照（_close）。**
> **资源标注（09-22 补）**：每批标 🖥️CPU / 🎮GPU / 💾大内存；GPU 强受益任务见批次 8。

| 批次 | 内容 | 耗时 | 资源 |
|---|---|---|---|
| 1 | 920 治本重跑 + 定版臂 | ≈8–10h | 🖥️💾 CPU 密集 + 大内存（全A 面板 float32 ~24GB + 滚动工作副本） |
| 2 | AI97 三臂 ×3 seed | ≈2–2.5h | 🖥️ 纯 CPU（多核受益小） |
| 3 | GFlowNet 正式实验 | 校准 1–1.5h → 9–14h | 🖥️ CPU 可跑（GPU 可加速非必需，模型小） |
| 4 | mf10 T 扫描全量 | 半天级 | 🖥️ CPU（caveat 已清：ic_max 决策时点 bug 09-24 修复+回归锚，见批次 4 节） |
| 5 | stage2 Phase 2 zz1000 | 快档 6–9h → 全档 1–2 天 | 🖥️ CPU 多核（sb3 PPO 常规 CPU 训练） |
| 6 | 口径修复重训 E5/E4/P5 | 各 ≈5–7h | 🖥️💾 同批次 1 |
| 7 | e2e 族年化口径重跑 | 未实测 | 🖥️ CPU，队列末尾 |
| **8** | **表格基础模型对照（TabPFN-3.5 / TabICL 2.2 vs LightGBM）** | **runner 已就绪**（09-24 `scripts/evaluation/tabpfn_rolling_compare.py`：逐月滚动 W∈{2,3}×test_step=1 + 可交易 IC/NWt + Top10% 月频 open 组合 + 秩相关诊断，gbdt 臂 hs300 真实冒烟通过 IC 0.026~0.027/NWt≈3）→ tabpfn/tabicl 臂全量待 GPU | **🎮 GPU 强受益（实测后升级为硬前置：CPU pred 27.7min/窗口，全量不可行）**；全量版建议单卡 ≥8GB 显存；hs300 小窗口 CPU 也能跑 |
| **9** | **东吴 LLM-MCTS Phase 0**（同题四引擎对照） | **代码已就绪**（09-24 本机落地：`factor/mcts/` 五模块 + `run_llm_mcts.py` 四臂 runner，23 用例 + mock/真实双冒烟通过）→ 全量待跑（耗时未实测，template 四臂 mock 冒烟 mcts 0.5s/gflownet 74s/onestot 4s/gp 42s@60 求值） | 🖥️ CPU + LLM API（key 同批次 2，仅 mcts/llm_oneshot 臂需要）；候选评测互相独立可多进程 |
| 10 | 920 后收尾对照包（member_blend12 / 消融 5 组等） | 各分钟级~小时级 | 🖥️ CPU，依赖批次 1 新 pred |

---

## 〇、当前结论快照（09-22，全部实验为 open 可执行口径、882 pred 复用）

1. **诊断闭环**：基本面权重低的主因是**期限错配**（fundamental 族 mean|IC| h1→h20
   ×1.143、holder ×1.150，量价族全负）；h20 模型已隐式吃到基本面，慢信号显式
   叠加不再提供增量 alpha，只剩风险分散价值。
2. **定版候选收敛为二选一**（920 重跑后裁决）：
   - **h1020 等权混合**（要年化）：15.75% / Sharpe 0.64 / 换手 66.6%；
   - **h1020⊕慢信号 λ0.5**（要风险调整）：15.02% / Sharpe 0.66 / 回撤 33.5% / 换手 51.1%。
3. **学权/优化路线证伪**：E6 stacking 学权未胜等权（15.86% vs 15.75%，回撤更深）；
   max-ICIR 在小样本基本面域劣于简单 ICIR（7.87% vs 10.17%）——与 1/N 文献一致，
   等权/简单规则稳健性胜出。**09-23 增证**：member_blend12 把成员扩到
   3 模型 × 4 horizon = 12 个，学权 stack_all12 15.53% ≈ 生产 ens_h1h5 15.52%，
   仍低于 h1020 等权 16.89%——学权无增量在更大成员集上再次复现。
4. **慢信号自身是真 alpha**：纯基本面 ICIR 线性合成年化 10.17%、超额上证 +7.94pp、
   换手 26.8%，是合格的防御性分散器。
5. **P5 数据盘点**：补缺清单 4/5 可自建或已建（增减持已入库待接席位），唯一数据墙
   =分析师一致预期类。详见 RESEARCH_TODO §一 基本面诊断。
6. **member_blend12（09-23，882 存量 pred）**：合成增量几乎全在 horizon（标签）维度
   不在模型族维度——eq_h20_3 16.00% > eq_h1_3 12.27%，同 horizon 混模型族是稀释；
   全 12 等权 15.12% < h1020 16.89%（弱成员净拖累）。**h1020 已在 882 口径
   member_blend12 上再次胜出（16.89%），920 后复验**。证据 `reports/member_blend12/`。

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

```bash
# 先 1 臂 1 seed 校准真实耗时（60~90 分钟/轮为外推值）
python -m scripts.factors.run_gflownet_phase1 ... --iters 600 --batch 12   # im / raw＋im / raw 三臂
# 校准后铺三臂 ×3 seed
```

（当前只有 5 iters 冒烟、无任何可引用数字；`--min-autocorr` RRE 门槛与挖掘预算
旋钮见 RESEARCH_TODO §一 国金24 节。）

## 批次 4：mf10 T 扫描全量（半天级）

```bash
python -m scripts.evaluation.mf10_t_scan --dataset hs300_2022_2025 --top 8
```

两方法（IC_IR 最大 / IC 最大）与现 pipeline 多窗口对比。~~已知 caveat~~
**已清（09-24）**：09-16 的"决策时点截面"修复在「全时段面板 + train_dates」
调用形态下未生效（V 恒取 `panel.index[-1]` = 测试段末）——已改为决策时点
跟随 `train_dates[-1]` + 回归锚（`tests/test_synthesis.py::
test_ic_max_v_decision_point_follows_train_dates`），mf10 T 扫描可放心全量。

## 批次 5：stage2 Phase 2 zz1000 全量（快档 6–9h → 全档 1–2 天）

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

- **E5 基本面只对行业中性化臂**（panels_neu 变体：基本面族跳过市值中性化，
  保价值/红利风格信息）——检验"ortho 二次压制"诊断，单变量对照；
- **E4 交互特征**（基本面状态分桶 × 量价因子组内 zscore 显式入池）——检验
  "基本面当条件变量"假设；
- **P5 收尾（09-22 全部构建入库：两批 9 因子，sue_q/asset_growth_yoy/
  inv_rev_gap 强、piotroski_f 负号=A股质量反转、其余弱；已入 B5 族+慢信号
  81 因子家族）**：剩 holder_dyn 族并入 FUNDAMENTAL_FAMILY_SETS 的拍板
  （本机对照温和改善）与 E5/E4 重训对照。

## 批次 7（队列末尾）

- e2e 族报告对齐年化口径（`perf_stats` 244→252 后 ~3% 系统性偏移）。

## 批次 8：表格基础模型对照 🎮（TabPFN-3.5 / TabICL 2.2 vs LightGBM；09-22 立项，09-23 版本口径升级）

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

## 附：本机已完成实验速览（2026-09-22，细节见 RESEARCH_TODO §一）

| 实验 | 一句话结论 |
|---|---|
| D1 分族 IC-horizon | 期限错配证实：fundamental +14.3%/holder +15.0%（h1→h20），量价全负 |
| D2 头部归因 | Top20 选股前三驱动全为量价+市值，基本面不直接驱动头部 |
| E1' horizon 混合 | h1020 等权 +1.1pp（close）/+1.09pp（open），全截面 IC 降而组合升 |
| E3 双层融合 | slow_only +7.9pp 真 alpha；blend_s0.3 全指标占优基线；慢信号=防御分散器 |
| E6 stacking | 学权未胜等权；权重结构独立复现"慢信号+h20 最有价值" |
| max-ICIR | 小样本域劣于简单 ICIR（7.87% vs 10.17%）——负结果归档 |
| P5 数据盘点 | 4/5 可自建或已建；增减持因子已入库但缺席席位；唯一数据墙=分析师类 |
