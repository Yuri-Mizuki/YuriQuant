# YuriQuant

个人量化研究系统（日频 + 日内）。数据 → 因子 → 模型 → 优化 → 回测 → 监控 → 报告 全链路，
数据源可插拔（AmazingData SDK / CSV / 另类数据自建通道），本地 Parquet 增量缓存，
向量化回测引擎带交易成本、涨跌停/停牌过滤和 point-in-time 股票池。

> **主实验口径速览**（2026-09-21）：全A(含退市回补) → 因子层全正交(ortho) → DPP 选择 +
> 基本面保留席位 → gbdt h1+h5 秩平均集成 → raw 信号 → 月频 Top10% 等权。
> 2018–2026 样本外 ortho 臂年化 **15.54%**（超额上证 +13.30 / Sharpe 0.632），
> zscore 臂 13.76%。⚠️ **引用必须带臂名**；且该数字为 **882 面板口径**——09-21
> `panels_neu` 补齐至 920 后已与当前库脱钩（详见「主要实验与交付物」）。

## 目录结构

```
config/     配置加载（settings.yaml + schedule.yaml 调度登记表 + 环境变量占位符）
data/       数据源抽象、本地缓存、股票池、可执行性掩码、财务PIT、文本挖掘、另类数据(altdata/)
factor/     因子层：算子空间、挖掘（exhaustive/GP/GFlowNet/RL）、合成、公式解析
model/      模型层：特征漏斗、标签、预测器、训练、评价、模型账本、因子回写
optimize/   优化层：组合优化（QP/HRP）、风险归因、信号生成、多期执行
monitoring/ 生产化监控：指标、告警规则、账本、生产口径 IC 台账、自包含HTML报告
strategy/   因子值 → 组合权重的策略
backtest/   向量化回测引擎、交易成本、绩效指标
research/   研究工具：因子检验（IC/分层）、归因、基准、DPP、实验记录、报告渲染
scripts/    命令行入口，按功能分 7 个子包（common / ingest / factors / pipelines /
            portfolio / evaluation / reporting）+ builders / textmining / archive /
            data_tools / oneoff（详见下方「scripts 目录索引」）
tests/      pytest 测试套件
reports/    实验与交付物（模型/监控/因子库/设计文档/HTML报告）
```

## 数据源

本项目当前实际使用的数据源有以下几类：

| 数据源 | 类型 | 接入方式 | 说明 |
|---|---|---|---|
| **AmazingData SDK** | 商业授权 | `data/datasource.py` 的 `AmazingDataSource` | **主数据源**。覆盖行情（日/分钟）、复权、财务三表、行业、股本、股息、十大股东/户数、涨跌停/停牌、指数成分、交易日历等。银河证券私有分发的本地 wheel，不在 PyPI，需按 `AmazingData开发手册.pdf` 单独安装；运行需 `AMAZINGDATA_USER/PWD/HOST/PORT` 凭证登录（非公开免费） |
| **另类数据自建通道** | 免费公开接口 | `data/altdata/` + `scripts/pipelines/fetch_altdata_daily.py` | 2026-09-20 落地的 7 源（详见下节）。免费但自建存档，无 SLA |
| **文本挖掘（同花顺研报 / 巨潮公告）** | 网页抓取 | `data/textmining/` | 从 `basic.10jqka.com.cn` 研报页与 `cninfo.com.cn` 公告接口抓取。免费但需爬取，页面结构可能变更需维护；定位为研究性补充 |
| **CSV 数据源** | 本地文件 | `CSVDataSource` | 备用，无 SDK 凭证时自动回退的离线开发模式 |
| **Mock 数据源** | 合成 | 脚本 `--mock` 开关 | 无凭证快速验证管线的模拟数据 |

主数据流：AmazingData SDK → 自建 Parquet 缓存 `DataCache`（`data/cache.py`，根目录 `e:/data/parquet/`）→
业务层（股票池 / 行业 / 财务 PIT / 可执行掩码等）。`DataCache` 在数据源之上提供增量更新、透明访问与离线研究能力。

### 数据层缓存表

| 当前文件 | 内容 | 缓存模式 | 索引/结构 |
|---|---|---|---|
| `daily.parquet` | 日K线 OHLCV+amount | 长表增量 | (date, code) |
| `min{period}.parquet` | 分钟K线（如 min5） | 长表增量 | (kline_time, code) |
| `adj_factor.parquet` / `backward_factor.parquet` | 单次/累积后复权因子 | 宽表全量刷新 | date×code 宽表 |
| `history_stock_status.parquet` | 涨跌停/停牌/ST/除权除息 | 长表增量 | (date, code) |
| `income.parquet` / `balance_sheet.parquet` / `cash_flow.parquet` | 财务三表 | 整表覆盖 | 长事件表 |
| `calendar.parquet` | 交易日历 | 合并去重 | date 列表 |
| `index_constituent_{code}.parquet` | 指数成分 | 整表覆盖 | 长事件表 |
| `industry_classification_level{N}.parquet` | 行业分类（申万 N 级） | 整表覆盖 | 长事件表 |
| `equity_structure.parquet` / `dividend.parquet` / `share_holder.parquet` / `holder_num.parquet` | 股本/分红/股东 | 整表覆盖 | 长事件表 |
| `alt_*.parquet` / `alt_cls/` | **另类数据 7 源**（快讯/宏观日历/增减持等） | 前向增量 + 断点续抓 | 统一水位 `_alt_meta.json` |
| `intraday/min{period}_{pool}/` | **分钟稠密面板**（物化视图，供日内挖掘） | 整体重写 | 按年 `y{YYYY}/*.npy` [day,bar,code] float32 + meta.json，np.memmap 按需页调入 |
| `_meta.json` | 各表增量水位 last_date + 数据指纹 | — | — |

### 数据层核心能力

- **数据源抽象** `DataSource` 定义 17 个抽象方法，行情统一返回 multi-index `(date, code)` DataFrame，日期统一 `pandas.Timestamp`，代码统一 `XXXXXX.SH/SZ/BJ`；切换数据源只改配置，不动业务代码。
- **Point-in-time**：`data/universe.py`（Universe 按指数成分构建股票池，PIT 取成分）；`data/financials.py`（三表 PIT 展开为日频面板，无未来函数）。
- **可执行性掩码** `data/tradability.py`：处理停牌、涨停封板、跌停封板等约束。
- **另类数据管道** `data/altdata/`（2026-09-20）：传输层（抓取+列名防御+统一 schema+原子落盘+断点续抓+前向增量）
  与因子层解耦。7 源实测：巨潮快讯 `alt_cls/`（119.2 万条，2014-12~今，自建历史存档）、
  东财宏观日历 `alt_macro_calendar`（64,978 行 / 2018 起，公布值/预期值/前值+重要性分级）、
  巨潮增减持 `alt_cninfo_holder`（26.6 万行 / 2010 起全历史，PIT 质量最高）、
  东财董监高 `alt_mgmt_hold`（仅 1 年，交叉验证用）、新浪亲属买卖 `alt_inner_trade`（约 1.7 年，滞后 2 交易日）、
  实控人变动 `alt_holder_control`（2010 起，稀疏事件型，滞后 60 交易日）、新闻快照 `news`（冗余备份，未建表）。
  PIT 对齐规则与逐源坑位见 `reports/docs/另类数据管道_数据源说明.md`。
  因子层已消费增减持族（holder_dyn 9 因子，**消融结论无增量价值**，表保留作增量）；其余表通道就绪、因子待建。
- **文本挖掘** `data/textmining/`：同花顺研报主源 + 巨潮公告辅源，统一 `fetch_docs()` 入口，parquet 增量缓存。
- **分钟频数据层（日内挖掘底座）** `data/intraday.py` + `factor/intraday_features.py`：parquet 分钟长表 → 按年分区的内存映射稠密面板 `[日,bar,码]`（MemMap，全市场 GB 级数据不进内存），之上是 42 个"分钟→日频"统计特征。入口 `scripts/factors/build_minute_panel.py`。

## 研发流程与完成度

> 三层 14 阶段。✅ 已就绪 · ⚠️ 雏形 · ❌ 待建。每阶段统一格式：**入口 → 输入 → 输出 → 验收标准**。

### 01 因子层（✅ 本系统最完整主干）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 研究分析 | ✅ `data/`、`scripts/evaluation/intraday_analysis.py`、`scripts/ingest/check_data_quality.py` | SDK → 行情/财务/日内结构画像，数据质量检查无 ERROR |
| 2 | 提出想法 | ✅ `scripts/factors/mine_factors.py`（`--exhaustive`/`--gp`）、`factor/gflownet/`、`factor/rl/` | 穷举 + 遗传规划 + GFlowNet(TB/PPO) + AlphaPool RL 自动生成候选公式 |
| 3 | 开发准备 | ✅ `scripts/ingest/update_data.py` | SDK → Parquet 缓存 + PIT 面板 + 股票池，增量水位正确、PIT 无未来函数 |
| 4 | 开发实现 | ✅ `scripts/factors/build_{technical,fundamental,intraday}_factors.py` + `scripts/builders/` 全A构建器 13 个 | 技术面/基本面/日内因子 + 全A数据集 `all_a_2018_2026`（920 因子 × 2471 日 × 5549 股） |
| 5 | 因子分析 | ✅ `research/factor_analysis.py`、`scripts/reporting/factor_correlation.py` | 因子面板 → IC/IR/衰减/分层/NW t/FDR，显著性基于 Newey-West t |
| 6 | 因子构建 | ✅ `scripts/factors/synthesize_factors.py`、`synthesize_library.py` | IC 加权 / PCA / IC_IR 最大 / IC 最大 / 半衰加权 / 等权 / 正交 / ML Stacking |
| 7 | 因子入库 | ✅ `scripts/factors/factor_library.py` | registry + panels + evals 三件套，血缘可追溯、去冗余预检、六维标签 |

**算子与指标**：`factor/operators.py` 算子注册表（约 51 算子）；`factor/technical_indicators.py`
通达信口径指标 57 个；`factor/technical.py` 自研 pandas 技术指标 9 个（离线可用）；
`factor/classic.py` 经典因子 7 个 + 银河 0608 特征族 36 个（`compute_galaxy_features`）；
`factor/preprocessing.py` 去极值(MAD/分位)→中性化(行业+对数市值)→标准化(zscore/rank)。
**中性化已向量化**（2026-09-20）：`batch` 实现 2.96x 提速且与历史逐日实现逐位一致（`max|Δ|=0`），
已设为默认；`grouped` 8.3x 为 opt-in。

**自动化挖掘算法**：
- **GP 遗传规划** `factor/genetic_mining.py`：DEAP，HallOfFame 精英保留，门诊/滚动 IC 评估。
- **GFlowNet** `factor/gflownet/`：`FactorMDP` 因子构造 MDP + `TBPolicy`/`PPONet`，Trajectory Balance 与 PPO 对照训练。
- **AlphaPool RL** `factor/rl/`：`AlphaPool` 环境 + gymnasium 包装，MaskablePPO + LSTMSharedNet；含 LLM 初始池（`llm_pool.py`，真实 deepseek-flash 联网验证）；组合优化 env（`portfolio_env.py`，银河 0706 口径主动权重空间）。

### 02 模型层（✅ 主要能力已就绪）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 模型设计 | ✅ `model/registry.py` | 持久化 CSV 注册表，原子写，list/view/compare/delete，同名再注册=新版本 |
| 2 | 模型训练 | ✅ `model/training.py`（`train_and_register`） | `ml_stacking` 与 `predictor` 双路径，滚动时序 CV |
| 3 | 模型评价 | ✅ `model/evaluation.py` + `scripts/evaluation/walk_forward_model.py` | IC/IR/NW t/p、IC 衰减、分层多空；三段样本外 + 滚动再训练 |
| 4 | 模型迭代 | ✅ 同名再注册 + `research/experiments.py` | 新版本自动入 registry，实验留痕 |
| 5 | 模型上线 | ✅ `model/serving.py`（`register_model_as_factor`） | 模型预测面板回写因子库，血缘双向溯源 |

**预测器与算法**：`model/features.py` 三级特征漏斗；`model/labels.py` horizon 前瞻收益（rank/zscore/raw，embargo=horizon）；
`model/predictor.py` Ridge / LightGBM / TabICL + `fit_predict_oos()` 扩展窗口 CV；`model/stacking.py` 四种合成器。

### 03 优化层（✅ 组合优化/风险归因/多期执行已就绪）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 组合优化 | ✅ `optimize/portfolio.py` + `optimize/solver.py` | 启发式投影 + 求解器双通道；QP 五法（min_var/tev/mvo/risk_parity/bl）+ HRP；约束含预算/上下限/行业中性/**基准行业权重目标**/逐股权重变动上限/λ 网格 |
| 1b | 多期执行 | ✅ `optimize/multi_period.py` + `scripts/portfolio/` | D/W/M 调仓频率，调仓日重解 QP；`generate_signals.py` 转每日可执行指令 |
| 2 | 风险归因 | ✅ `optimize/risk.py`（`risk_attribution`） | α/β 分解（NW t）、Brinson（Carino 链接）、风险分解（Euler + VaR/CVaR 成分） |
| 3 | 持续监控 | ✅ `monitoring/production_ic.py` + `scripts/reporting/monitor_production_ic.py` | **现役 = 生产口径每日 IC/风格暴露台账**（见 §04）；旧 `monitor_performance` 链已退役 |

### 04 生产化监控（✅ 现役台账 + ⚠️ 旧链退役）

**现役**：`monitoring/production_ic.py` + `scripts/reporting/monitor_production_ic.py`
（2026-09-18 起）——全A · ens_h1h5 · ortho 口径的每日 IC / 中性化 IC / 风格暴露台账 →
`reports/monitoring/production_ic_daily.csv`，由 19:00 日报自动化带跑。

**旧链**（`monitoring/` 包 + `scripts/reporting/monitor_performance.py`）：`YuriQuant Monitor`
计划任务 **2026-09-21 已删除**——其绑定数据集 `hs300_2022_2025` 行情源自 2026-08-26 停更，
每轮产出 144 critical / 235 warning 纯噪音告警；改指全A 实测不可行（runner 行情源硬编码 hs300
+ 全A 库 evals 缺失，920 因子全部被跳过）。脚本手动仍可跑，但结果只代表历史窗口。
复活前提：① 补齐全A 库 evals；② runner 行情源参数化（TODO §一/§二）。

旧链能力保留说明：`metrics.py`（IC 漂移/覆盖率/新鲜度/分位单调性）、`alerts.py`（因子级 5 规则 +
信号级 5 规则 + 库级拥挤度）、`ledger.py`（幂等账本）、`runner.py`（编排 + 自包含 HTML 报告）。

### 05 研究层与报告

- **因子检验** `research/factor_analysis.py`：Spearman/Pearson IC、IR、IC 衰减、分层回测。
- **归因** `research/attribution.py`：Fama-MacBeth、Brinson、α/β 分解。
- **基准** `research/benchmarks.py`：等权/买入持有基准。
- **多样性筛选** `research/dpp_selection.py`：log-det 最大化的 DPP 集合级筛选 + 贪心 pairwise 去重。
- **实验记录** `research/experiments.py`：CSV 实验档案，run_id/指纹/metrics/note。
- **报告渲染** `research/html_report.py` + `xlsx_report.py` + `report_pipeline.py`：
  自包含 HTML（Chart.js，A 股红涨绿跌）与 XLSX；一键端到端报告 `scripts/reporting/generate_report.py`。
- **稳健统计** `research/robust_stats.py`：Newey-West t、OLS+HAC、自动滞后阶选择。
- **过拟合检验** `stats/pbo.py`（2026-09-18）：CSCV PBO（12870 组合秒级）+ 缩水夏普 DSR + `deflate_best`；待接入各实验产物作为出报告固定环节。

## 主要实验与交付物

> 实验结论按最新口径排序；历史 HS300 时代实验已压缩。年化数字**必须带臂名**引用。

### 全A 主线（现行口径）

| 实验 | 入口 | 结论 / 交付 |
|---|---|---|
| **全A多年度滚动训练（主实验）** | `scripts/pipelines/rolling_grid_alla.py` + `rolling_grid_report.py` | 全A 5549 股，2018~2026 分年 walk-forward。**最优口径 = ortho 因子层全正交 + DPP 选择(基本面保留席位) + gbdt h1+h5 秩平均集成 + raw Top10% 月频**：年化 15.54%、超额上证 +13.30、Sharpe 0.632（2026-09-20 定版）。⚠️ **该数字为 882 面板口径**——09-21 `panels_neu` 882→920 补齐后真 DPP 对照选中集 0/18 相同，**旧数字与当前库已脱钩**，治本口径重跑（`--out-tag ortho920tl`）待执行，新数字出来前引用须带「882 口径」标注。关键消融：正交化 +1.6pp、DPP+基本面席位 +0.7pp、h1h5 集成 +1.5pp、信号层必须 raw（zscore 臂 13.76%）、月频可行/日频 −52pp。产物 `reports/alla_rolling/` |
| **2×2 口径探针** | `scripts/oneoff/probe_prod_pipeline_gap.py` → `reports/prod_pipeline_gap/report.md` | 实验链 vs 生产链口径差异隔离：ortho+h1h5 15.52 / zscore+h1h5 13.75 / ortho+h1 13.51 / zscore+h1 12.47；信号秩相关仅 0.828——**"15.54 已不可复现"的量化证据** |
| **全A每日选股排名（生产化推理）** | `scripts/pipelines/alla_daily_rank.py` | 主实验最优口径的每日盘后推理（17:30 计划任务）：增量更新 → 尾部重算当年入选 ~50 因子 → 500 日窗重训 → 全A排名 + Top10% 可交易候选。**09-17 起默认 ortho 口径**（实时正交化，面板等价性 85 特征秩相关 median 0.9999）；`--preproc zscore` 回退；硬剔 `limit_pos` 纸面因子；**BJ 整体不入榜**（无行业分类）。09-21 起任务注册走 `task_scheduler`（真源 `config/schedule.yaml`） |
| **全A超额归因与显著性复核** | `scripts/pipelines/alla_excess_attribution.py` | 对上证超额 +10.2%/年中约一半来自风格敞口（β=1.10），对全A等权纯选股 α=+5.1%/年（t=2.14 显著）；Brinson：超额全部来自行业内选股（选择 +64.1%）→ `reports/alla_attribution/` |
| **模型增强组合（正式投产入口）** | `scripts/pipelines/run_model_portfolio.py` | 全A · ortho · ens_h1h5 · raw · 月频 Top10%，参数真源 `config/settings.yaml` `model_portfolio` 段；2026 样本外实测（net）年化 +8.6%、超额上证 +8.2%；导出当日 Top10% 选股清单 → `reports/model_portfolio/` |
| **论文复现因子族（awesome 21 式）** | `factor/paper_factors.py` + `scripts/factors/build_paper_factors.py` | awesome-systematic-trading 复现库 21 个可 A 股实现策略入库：短期反转 IC=0.038/t=11.3、低波 t=8.2、价值 t=8.5 显著为正；月频动量族为负（A 股动量反转复现） |
| **另类数据管道 P0** | `data/altdata/` + `scripts/pipelines/fetch_altdata_daily.py` | 7 源落地（快讯 119.2 万条 / 宏观日历 6.5 万行 / 巨潮增减持 26.6 万行全历史等）；**holder_dyn 9 因子强制纳入消融 Δ=−0.85pp，无增量价值结案**（`reports/holder_dyn_forced/`）；其余表通道就绪待因子挖掘轮次 |
| **RL 组合优化 stage2（银河 0706 复现线）** | `factor/rl/portfolio_env.py` + `scripts/factors/run_portfolio_ppo.py` / `run_portfolio_phase2.py` | Phase 0 env 骨架（22 用例）+ Phase 1 hs300 平价验证（PPO 贴基准打平/QP 大偏离者输，与银河 HS300 形态一致）+ 银河 0608 L1 落地（**风险标签可测成立**：mdd test 0.25/0.40）+ Phase 2 zz1000 runner 冒烟通过；**全量待好机器**（命令见 RESEARCH_TODO） |
| **生产口径 IC 监控（现役）** | `scripts/reporting/monitor_production_ic.py` | 全A · ens_h1h5 · ortho 每日 IC / 中性化 IC / 风格暴露台账 → `reports/monitoring/production_ic_daily.csv` |

### HS300 时代（历史基线，已压缩）

| 实验 | 入口 | 结论 / 交付 |
|---|---|---|
| 模型 walk-forward | `scripts/evaluation/walk_forward_model.py` | mock/real 滚动再训练，模型因子回写因子库 → `reports/models(_mock)/` |
| 端到端策略回测 | `scripts/pipelines/e2e_backtest.py` | walk-forward 月频回测（2024-01~2026-08 跑输全池基准——信号强度不足以支撑集中持仓）→ `reports/e2e_backtest/` |
| ML 因子合成 | `scripts/archive/ml_synthesis_experiment.py` | h=5 valid IC 0.055-0.061 但 test 归零（29/35 特征方向翻转）；h=1 OOS IC 0.039-0.041 → `reports/archive/ml-synthesis-hs300-report/` |
| 算法对比 | `scripts/archive/ml_algorithm_compare.py` | Ridge/GBDT/TabICL + 窗口/再训频率对比 → `reports/ml_algorithm_compare/` |
| 组合方法对比 | `scripts/portfolio/compare_portfolio_methods.py` | projection/min_var/tev/risk_parity/hrp 五法 → `reports/portfolio_methods_compare.csv` |
| 多期执行 | `scripts/portfolio/multi_period_backtest.py` | 2025 与 2026H1 两段样本外 → `reports/multi_period/`、`reports/two_periods/` |
| 投资收益报告 | `scripts/reporting/investment_report.py` | 模型预测作因子检验 + 组合 vs 大盘指数基准（沪深300→000300.SH，全A→000001.SH）→ `reports/investment_report/` |
| 因子库检验/交互报告 | `scripts/reporting/factor_library_full_report.py`、`factor_explorer_report.py` | 全量检验表 + 交互检测（时间段×来源×指标排序）→ `reports/factor_library_report_*.html` |
| 调仓频率精修 / 多年度 OOS | `scripts/evaluation/freq_tune.py`、`multiyear_oos.py` | h1×M 是唯一三年一致稳健解；日频超额 −40% → `reports/freq_tune/`、`reports/multiyear/` |
| HS300 选股清单入口（已归档） | `scripts/archive/e2e_stock_picks.py`、`select_stocks.py` | 09-21 归档：数据口径停更、职责被 `alla_daily_rank` 取代 |
| 旧链监控（已退役） | `scripts/reporting/monitor_performance.py` | `YuriQuant Monitor` 任务 09-21 删除（原因见 §04），脚本手动可跑 |
| 日内研究 | `scripts/evaluation/intraday_analysis.py` | 隔夜 vs 日内分解、时段效应 → `reports/intraday_analysis_*.png` |
| 自动因子挖掘 | `scripts/evaluation/gp_tune_budget.py`、`run_gflownet_phase0/1.py` | GP 调参 / GFlowNet 最小闭环 → `reports/gp_tune/`、`reports/_htai_gp/` |
| 文本挖掘 | `scripts/factors/fetch_textmining.py` + `scripts/textmining/` | 研报/公告 → FADT/SUE-文本、BERT 编码 → `reports/textmining*/`（另一会话维护） |
| 设计文档 | — | 模型层/训练纪律/项目总览 → `reports/docs/design/` |

### 执行价口径（已收口）

- **C0 = 调仓日收盘成交 + T+1 可交易掩码**（2026-09-14 终审）。C0 实盘不可行（因子用 T 日收盘数据），
  实测 T+1 开盘 14.66 / VWAP 14.60（vs close 15.54，代价 −0.9pp、换手不变）。
  **执行价模式代码（`--execution`/open/vwap）已于 2026-09-21 整体移除**，仅留 6 份 metrics CSV 证据。

## 待办 / 缺口 / 历史交付

> **单一真源：本节不再在这里维护。**
>
> - **待办与已知缺口** → [`TODO.md`](TODO.md)
> - **研报研读与复现待办** → [`RESEARCH_TODO.md`](RESEARCH_TODO.md)
> - **逐日工作日志与踩坑** → `.workbuddy/memory/YYYY-MM-DD.md`

## scripts 目录索引

**2026-09-11 按功能重排**：root 原有 56 个扁平 .py 已全部归入 7 个受跟踪子目录。
调用方式统一为 `python -m scripts.<group>.<script>`，`python scripts/<group>/<script>.py` 亦可。

> ⚠️ **不用 `scripts/data/` 与 `scripts/reports/`**：与顶层 `data/` 包、`reports/`
> 产物目录同名，`import data` 会被劫持——故改为 `ingest/` 与 `reporting/`。

### 调度入口速查（真源 = `config/schedule.yaml`）

> **2026-09-21 收口**：注册、核对、体检全部由 `scripts/common/task_scheduler.py` 单点负责
> （历史上散着三份重复的 `install_task` 实现，且 schtasks 裸建继承电池条件导致任务
> `0xC000013A` 静默被杀——现改 XML 导入，显式关电池条件、设 `StartWhenAvailable` /
> `ExecutionTimeLimit=PT3H` / `IgnoreNew`）。搬目录/改参数后必须重注册。

```bash
# 体检：登记↔系统漂移 / 上次结果码 / 产物新鲜度（秒级，只读）
python -m scripts.common.task_scheduler doctor

# 按登记表重注册（改完 schedule.yaml 再跑；参数不再靠手敲）
python -m scripts.common.task_scheduler install alla_daily_rank
```

| key | 谁的活 | 任务名 | 时间 | 体系 |
|---|---|---|---|---|
| `alla_daily_rank` | `scripts.pipelines.alla_daily_rank` | `YuriQuant AllaDailyRank` | 每日 17:30 | schtasks |
| `altdata_daily` | `scripts.pipelines.fetch_altdata_daily`（**须 `.venv` 解释器**） | `YuriQuant AltDataDaily` | 每日 18:00 | schtasks |
| `daily_report` | 日报邮件 + 生产口径 IC 台账 | — | 工作日 19:00 | **WorkBuddy 自动化** |
| ~~`monitor`~~ | ~~`scripts.reporting.monitor_performance`~~ | ~~`YuriQuant Monitor`~~ | — | **2026-09-21 已删除**（原因见 §04） |
| `altdata_cls_backfill` | 财联社历史回补续跑（回补完自动空转） | `YuriQuant AltDataClsBackfill` | — | **planned 未启用** |
| `altdata_news` | 高频新闻快照 | `YuriQuant AltDataNews` | — | **planned 未启用** |

> 🚨 任务里内嵌的是**绝对路径**，搬目录后必须重注册。
>
> ⚠️ **WorkBuddy 自动化刻意不并入 schtasks**：日报要判交易日、按退出码分级重试、附件缺失
> 降级、异常写记忆——分层原则：**确定性流水线走 schtasks，需要判断的走 WorkBuddy**。
>
> ⚠️ **日报与出榜的时序缓冲只有 30 分钟**。自动化已改为"产物日期必须等于最近交易日、
> 且今天是工作日时日历不得落后于今天"才放行——防链断后把旧榜当当日榜发出去。

### 命名相近但职责不同的脚本（老规矩，别弄混）

`walk_forward`（因子挖掘三段验证）≠ `walk_forward_model`（模型层滚动 OOS）；
`synthesize_factors`（SDK 在线合成）≠ `synthesize_library`（离线合成）；
`e2e_backtest` / `backtest_two_periods` / `multi_period_backtest` 为三件不同的事。

### `scripts/common/` — 公共库（被多方 import，**非入口**，改它影响面最大）

| 模块 | 职责 |
|---|---|
| `cli_common` | 脚本层公共 CLI 骨架（argparse / logging / 三态数据源样板）+ build 家族助手 |
| `e2e_common` | e2e 家族编排（数据加载 / 特征选择 / im 特征装载 `attach_im_features` 等） |
| `portfolio_common` | 跨实验组合编排（legacy HS300 口径组件） |
| `fundamental_common` | 基本面构建器公共层 |
| `task_scheduler` | **Windows 计划任务单点注册**（XML 导入，显式关电池条件）+ `doctor` 体检；消费 `config/schedule.yaml` |

### `scripts/ingest/` — 数据更新 / 回补 / 体检

| 脚本 | 职责 |
|---|---|
| `update_data` | 增量更新行情/财务缓存（`--pool` / `--no-minute`；盘中守卫自动回退终点） |
| `update_etf` | ETF 行情更新（需系统 Python 3.12 + SDK） |
| `fetch_status_batched` | 批量拉全A状态表，断点续拉 |
| `check_data_quality` | 拉数后质量体检：缺失率/复权跳变/价格异常 |
| `extend_factor_library` | 因子库面板延长到数据源最新交易日 |

### `scripts/factors/` — 因子构建 / 合成 / 入库 / 挖掘

| 脚本 | 职责 |
|---|---|
| `build_alpha_factors` | Alpha101 / GTJA Alpha191 公开因子构建入库 |
| `build_fundamental_factors` / `build_technical_factors` | 财务 PIT / 技术指标因子 |
| `build_paper_factors` | 论文复现因子 21 式 |
| `build_intraday_factors` / `build_intraday_stat_factors` / `build_minute_panel` | 日内因子三件套 |
| `synthesize_factors` / `synthesize_library` | 多因子合成（在线版 / 离线版） |
| `factor_library` | 因子库 CLI（list / compare / export / delete） |
| `mine_factors` | 因子挖掘 CLI（exhaustive / GP / GTJA 预设 / im 特征） |
| `run_gflownet_phase0` / `run_gflownet_phase1` / `gflownet_library_ingest` | GFlowNet 挖掘与入库 |
| `run_portfolio_ppo` / `run_portfolio_phase2` | RL 组合优化 stage2（Phase 1 平价验证 / Phase 2 zz1000 主实验） |
| `fetch_textmining` | 拉取文本挖掘数据 |

### `scripts/pipelines/` — 主线管线（主实验 / 生产 / 端到端）

| 脚本 | 职责 |
|---|---|
| `alla_daily_rank` | 全A每日模型选股排名（计划任务 `YuriQuant AllaDailyRank`） |
| `fetch_altdata_daily` | 另类数据日增量抓取（计划任务 `YuriQuant AltDataDaily`，`.venv`） |
| `rolling_grid_alla` | 全A多年度滚动训练实验（主实验入口） |
| `run_model_portfolio` | 模型增强组合正式入口（ortho · ens_h1h5 · 月频 Top10%） |
| `alla_excess_attribution` | 全A主策略超额归因（α/β + Brinson） |
| `e2e_backtest` | 端到端选股 walk-forward 回测 |

### `scripts/portfolio/` — 组合构建 / 信号 / 执行

| 脚本 | 职责 |
|---|---|
| `generate_signals` | 每日可执行交易信号导出（`--index` 参数化） |
| `multi_period_backtest` | 多期组合执行回测（QP + 成本 + 约束） |
| `compare_portfolio_methods` | 组合法对比 |
| `run_etf_rotation` | ETF 轮动最小闭环 |
| `optimize_e2e` | 调仓频率 × 风格中性化端到端优化（HS300 口径） |

> 注：需要 Σ 的 QP / HRP / BL 在 `optimize/` 包；本目录只做「无风险模型的权重生产」。

### `scripts/evaluation/` — 回测 / 评估 / 调参 / 实验网格

| 脚本 | 职责 |
|---|---|
| `walk_forward_model` | 模型层滚动 OOS（唯一生产切分入口） |
| `walk_forward` | 因子挖掘三段样本外验证 |
| `cpcv_eval` / `cpcv_h1_eval` | CPCV 评估 / h=1 专项 |
| `multiyear_oos` | 多年度 OOS（2023/24/25 × h1/h5 × D/W/M） |
| `freq_tune` / `buffer_tune` | 调仓频率精修 / 缓冲带调参 |
| `mf10_t_scan` | 合成权重 T 参数扫描（多因子10 复现） |
| `galaxy_l1` | 银河 0608 特征族+三标签滚动 GBDT 评估 |
| `backtest_two_periods` | 两期因子面板回测 |
| `gtja_repro_eval` | 国金 Alpha 掘金复现评估 |
| `gp_tune_budget` | GP 挖掘预算调参 |
| `intraday_analysis` | 日内收益分解 + 时段效应 |

### `scripts/reporting/` — 报告渲染 / 归因 / 监控

| 脚本 | 职责 |
|---|---|
| `investment_report` | 投资收益报告 |
| `factor_explorer_report` / `factor_library_full_report` | 因子库交互检测 / 全量检验报告 |
| `rolling_grid_report` | 全A滚动实验收益曲线 + 分年绩效 HTML |
| `jq_style_report` | 聚宽风格收益曲线页 |
| `risk_decomposition_report` | 组合级风险分解报告 |
| `generate_report` | 一键汇总 reports/ 产物为单个 HTML |
| `attribution` / `factor_correlation` | 收益归因 CLI / 因子相关性矩阵 |
| `build_daily_email_body` + `md_to_email_html` | 每日邮件正文构建（日报自动化后端） |
| `monitor_production_ic` | **现役**：生产口径每日 IC / 风格暴露台账 |
| `monitor_performance` | 旧链监控（任务已删除，脚本手动可跑） |

> 所有报告统一走 `research/html_report.page()` 外壳，守卫 `tests/test_report_shell.py`。

### `scripts/builders/` — 全A 数据集构建 / 回补管线（受跟踪）

生产入口 `alla_daily_rank` 直接 import 它（干净 clone 必须在库）。

| 分组 | 模块 |
|---|---|
| 因子面板构建器（→ `all_a_2018_2026`） | `build_alla_alpha_panels`（alpha101/158/191/360）、`build_alla_fundamental_factors`（闭包基座）、`build_alla_constructed_factors`、`build_alla_pledge_factors`、`build_alla_holder_factors`、`build_alla_status_factors`、`build_alla_style_factors`、`build_alla_event_factors`、`build_alla_margin_factors`、`build_alla_moneyflow_factors`、`build_alla_sue_pledge_factors`、`build_alla_disc_holder_dyn`（大宗折价+股东动态）、`build_alla_holder_dyn_factors`（另类数据增减持 9 因子）、`build_alla_factor_neutralized`（因子层中性化面板） |
| 数据回补器 | `fetch_alla_history`、`refetch_status_all_a`、`backfill_financial_alla`、`backfill_holder_alla`、`backfill_pledge_profit_alla`、`backfill_delisted_kline` / `_backward` / `_equity` |
| 编排器 | `run_evt_margin_build`（回补 + 构建串行，防 SDK 单连接竞争） |

用法示例：`python -m scripts.builders.build_alla_alpha_panels --workers 6`；多数支持 `--resume`。

### `scripts/archive/` — 归档区（受跟踪，"已产出结论、不再迭代"）

`daily_pipeline`、`factor_screening`、`factor_usability_stats`、`dpp_library_compare`、
`compare_htai_fitness`、`compare_ml_synthesis`、`diagnose_factor_vs_model`、
`diagnose_neutralized_compare`、`gtja_discipline_eval`、`ml_algorithm_compare`、
`ml_synthesis_experiment`；2026-09-21 新增 `e2e_stock_picks`、`select_stocks`
（HS300 时代选股清单入口：数据口径停更、被 `alla_daily_rank` 取代）。

> 未随批归档的同名家族：`e2e_backtest` / `optimize_e2e` 属独立回测实验入口，保留原处。

### `scripts/textmining/` / `scripts/data_tools/` / `scripts/oneoff/`

| 目录 | 状态 |
|---|---|
| `textmining/` | 文本因子线（22 个），**另一会话维护** |
| `data_tools/` | 数据工具（含 `dump_pdf_text.py` 研报 PDF 抽文本） |
| `oneoff/` | 🚫 **整目录 gitignored**（本地一次性脚本/探针，0 跟踪）。别往这里放被 import 的代码。规则见 `scripts/oneoff/README.md` |

## 安装

本项目依赖用 `pyproject.toml` 管理，可直接装成可编辑包：

```bash
pip install -e ".[dev]"
```

**复现锁**：仓库提交 `uv.lock`。需要精确复现环境可用：

```bash
uv sync --frozen --extra dev --extra ml --extra gp --extra solver
```

CI 以 `uv lock --check` 守卫锁文件同步；`rl` 组含 torch，CI 与本地默认不装，需要时单独 `--extra rl`。

CI 分两层：push/PR 只跑**快检查**（ruff 真 bug 规则 + 三个分层守卫测试，约 1 分钟）；
全量 pytest 移到仅手动触发的 `full-tests.yml`。

**例外：AmazingData SDK 不在上述依赖里。** 银河证券私有 wheel，按开发手册 3.3 节单独安装：

```bash
pip install AmazingData-*.whl
```

没有这个 SDK 也能用——`config/settings.yaml` 把 `datasource.type` 切成 `csv`，或跑 Mock 模式。

> ⚠️ **双环境**：生产固定**系统 Python 3.12**（`D:/Python/Python312`，SDK wheel 进不了
> uv.lock）；`.venv` 是 uv 锁定的**测试**环境（pandas 版本与系统不一致，详见
> `.workbuddy/memory/2026-09-20.md`）。**另类数据抓取只装 `.venv`**。

## 快速开始

跑一次 Mock 数据端到端回测（不需要数据源凭证）：

```bash
python scripts/pipelines/e2e_backtest.py --top 20 --model ridge --n-days 400 --n-codes 30
```

用真实数据（需先配置 `AMAZINGDATA_USER/PWD/HOST/PORT` 环境变量）：

```bash
python -m scripts.ingest.update_data                  # 先建/增量更新缓存
python scripts/pipelines/e2e_backtest.py --real       # 端到端选股 walk-forward 回测
python -m scripts.pipelines.run_model_portfolio       # 正式投产入口（全A 正交化管线）
```

更新本地数据缓存：

```bash
python -m scripts.ingest.update_data                  # 日K线 + 配置的分钟档位
python -m scripts.ingest.update_data --minute 1,5,15  # 指定分钟档位
python -m scripts.ingest.update_data --no-minute      # 只更新日频数据
```

### 模型增强组合（固化配置，正式投产入口）

```bash
python -m scripts.pipelines.run_model_portfolio                  # 全A·ortho·ens_h1h5·raw·月频Top10%
python -m scripts.pipelines.run_model_portfolio --frac 0.20      # 覆盖持仓比例
python -m scripts.pipelines.run_model_portfolio --refresh-base   # 日线更新后先重建基础面板
python -m scripts.pipelines.run_model_portfolio --no-train       # 复用预测缓存（快速回测/选股）
python -m scripts.pipelines.run_model_portfolio --pre-cost       # 同时输出成本前口径
```

2026 样本外实测（net）：年化 +8.6%、超额上证 +8.2%、Sharpe 0.33；导出当日
Top10% 选股清单 `picks_YYYY-MM-DD.csv`。参数真源在 `config/settings.yaml` 的
`model_portfolio` 段（**费率除外**——交易成本唯一真源是顶层 `costs` 段）；
结果落 `reports/model_portfolio/`。

### 全A每日模型选股排名（生产化推理）

```bash
python scripts/pipelines/alla_daily_rank.py                    # 全流程：更新数据→重训→排名（ortho ~60 分钟）
python scripts/pipelines/alla_daily_rank.py --skip-update      # 离线（数据已更新）
python scripts/pipelines/alla_daily_rank.py --preproc zscore   # 回退旧口径（北交所保留）
python scripts/pipelines/alla_daily_rank.py --window 750       # gbdt_w750 变体
```

> 任务注册推荐入口：`python -m scripts.common.task_scheduler install alla_daily_rank`
> （`--install-task` 保留只为兼容）。细节：增量更新带盘中半拉 K 线守卫（拉取终点自动
> 回退上一交易日）+ SDK 拉表失败重试 3 次；幽灵股守卫 + 信号日停牌/ST/封板标注；
> 跨年无选择文件时回退最近年份并告警；**北交所无行业分类 → ortho 口径不入榜**。

### 生产口径监控（现役）

```bash
python scripts/reporting/monitor_production_ic.py                        # 生产口径每日 IC 台账
python scripts/reporting/monitor_production_ic.py --summary-only         # 只打印近期 vs 全期 IC
python scripts/reporting/monitor_performance.py                          # 旧链（手动，历史窗口）
```

### 另类数据日增量

```bash
python scripts/pipelines/fetch_altdata_daily.py --tables all   # 全表前向增量（断点续抓）
```

## 日内研究：收益分解 + 时段效应分析

```bash
python -m scripts.evaluation.intraday_analysis --offline --begin 20250101 --end 20251231   # 读缓存（推荐）
python -m scripts.evaluation.intraday_analysis          # 真实数据（需 SDK 登录）
python -m scripts.evaluation.intraday_analysis --mock   # mock 验证管线
```

输出（默认到 `reports/`）：`intraday_analysis_{year}.png`（4 面板）、
`intraday_ts_{year}.csv`（日频时间序列）、`intraday_summary_{year}.csv`（汇总指标）。

## 训练纪律（L2）

所有搜索/训练类算法（穷举/GP/GFlowNet/RL/stacking/predictor）遵循三段式纪律，
由 `config/settings.yaml` 的 `discipline` 段冻结，`embargo = 标签 horizon`：
- **train 段**（挖掘 / 粗筛）
- **valid 段**（筛选 / 定权重）
- **test 段**（valid_end 之后，冻结）：只允许 walk_forward 型最终验证与上线后监控，绝不参与挖掘/调参/入库决策。

## 测试

```bash
pytest tests/ -v
```

测试套件覆盖数据层、回测引擎、因子（挖掘/合成/GFlowNet/RL）、模型层、监控、
优化/求解器/信号/多期、文本挖掘、报告与稳健统计等。

## Lint / Type-check

```bash
ruff check .
mypy .
```
