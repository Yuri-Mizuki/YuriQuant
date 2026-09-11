# YuriQuant

个人量化研究系统（日频 + 日内）。数据 → 因子 → 模型 → 优化 → 回测 → 监控 → 报告 全链路，
数据源可插拔（AmazingData SDK / CSV），本地 Parquet 增量缓存，
向量化回测引擎带交易成本、涨跌停/停牌过滤和 point-in-time 股票池。

## 目录结构

```
config/     配置加载（settings.yaml + 环境变量占位符）
data/       数据源抽象、本地缓存、股票池、可执行性掩码、财务PIT、文本挖掘
factor/     因子层：算子空间、挖掘（exhaustive/GP/GFlowNet/RL）、合成、公式解析
model/      模型层：特征漏斗、标签、预测器、训练、评价、模型账本、因子回写
optimize/   优化层：组合优化（QP/HRP）、风险归因、信号生成、多期执行
monitoring/ 生产化监控：指标、告警规则、账本、自包含HTML报告、调度
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
| **CSV 数据源** | 本地文件 | `CSVDataSource` | 备用，无 SDK 凭证时自动回退的离线开发模式，指向本地目录 |
| **Mock 数据源** | 合成 | 脚本 `--mock` 开关 | 无凭证快速验证管线的模拟数据 |
| **文本挖掘（同花顺研报 / 巨潮公告）** | 网页抓取 | `data/textmining/` | 从 `basic.10jqka.com.cn` 研报页与 `cninfo.com.cn` 公告接口抓取。**免费但需爬取**，无官方开放 API、无 SLA，页面结构可能变更需维护；建议定位为研究性补充而非生产依赖 |

主数据流：AmazingData SDK → 自建 Parquet 缓存 `DataCache`（`data/cache.py`，根目录 `e:/data/parquet/`）→
业务层（股票池 / 行业 / 财务 PIT / 可执行掩码等）。`DataCache` 在数据源之上提供增量更新、透明访问与离线研究能力。

### 数据层缓存表

| 当前文件 | 内容 | 缓存模式 | 索引/结构 |
|---|---|---|---|
| `daily.parquet` | 日K线 OHLCV+amount | 长表增量 | (date, code) |
| `min{period}.parquet` | 分钟K线（如 min5） | 长表增量 | (kline_time, code) |
| `adj_factor.parquet` | 单次复权因子 | 宽表全量刷新 | date×code 宽表 |
| `backward_factor.parquet` | 累积后复权因子 | 宽表全量刷新 | date×code 宽表 |
| `history_stock_status.parquet` | 涨跌停/停牌/ST/除权除息 | 长表增量 | (date, code) |
| `income.parquet` | 利润表 | 整表覆盖 | 长事件表 |
| `balance_sheet.parquet` | 资产负债表 | 整表覆盖 | 长事件表 |
| `cash_flow.parquet` | 现金流量表 | 整表覆盖 | 长事件表 |
| `calendar.parquet` | 交易日历 | 合并去重 | date 列表 |
| `index_constituent_{code}.parquet` | 指数成分（000300SH） | 整表覆盖 | 长事件表 |
| `industry_classification_level{N}.parquet` | 行业分类（申万 N 级） | 整表覆盖 | 长事件表 |
| `equity_structure.parquet` | 股本结构变动事件 | 整表覆盖 | 长事件表 |
| `dividend.parquet` | 分红送转 | 整表覆盖 | 长事件表 |
| `share_holder.parquet` | 十大股东 | 整表覆盖 | 长事件表 |
| `holder_num.parquet` | 股东户数 | 整表覆盖 | 长事件表 |
| `intraday/min{period}_{pool}/` | **分钟稠密面板**（物化视图，供日内挖掘） | 整体重写 | 按年 `y{YYYY}/*.npy` [day,bar,code] float32 + meta.json，np.memmap 按需页调入 |
| `_meta.json` | 各表增量水位 last_date + 数据指纹 | — | — |

### 数据层核心能力

- **数据源抽象** `DataSource` 定义 17 个抽象方法，行情统一返回 multi-index `(date, code)` DataFrame，日期统一 `pandas.Timestamp`，代码统一 `XXXXXX.SH/SZ/BJ`；切换数据源只改配置，不动业务代码。
- **Point-in-time**：`data/universe.py`（Universe 按指数成分构建股票池，沪深300/中证500/中证1000，PIT 取成分）；`data/financials.py`（三表 PIT 展开为日频面板，无未来函数）。
- **可执行性掩码** `data/tradability.py`：处理停牌、涨停封板、跌停封板等约束。
- **离线模式** `data/offline.py`：`OfflineDataSource`（缺缓存抛错）/ `OfflineQuietDataSource`（缓存完整时建面板）。
- **文本挖掘** `data/textmining/`：同花顺研报主源 + 巨潮公告辅源，统一 `fetch_docs()` 入口，parquet 增量缓存（ths/cninfo），PIT 日期过滤。
- **分钟频数据层（日内挖掘底座）** `data/intraday.py` + `factor/intraday_features.py`：parquet 分钟长表 → 按年分区的内存映射稠密面板 `[日,bar,码]`（MemMap，全市场 GB 级数据不进内存；参照 Alpha掘金 24 的工程方案），之上是 tsfresh 风格的向量化"分钟→日频"统计特征层（42 个：动量/收益分布/波动/形态自相关/量价/蜡烛，与 Alpha掘金 22 的 40 指标降维路线一致）。入口 `scripts/factors/build_minute_panel.py`（`--features` 提取特征长表，`--demo-ic` 次日 IC 快照）。

## 研发流程与完成度

> 三层 14 阶段。✅ 已就绪 · ⚠️ 雏形 · ❌ 待建。每阶段统一格式：**入口 → 输入 → 输出 → 验收标准**。

### 01 因子层（✅ 本系统最完整主干）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 研究分析 | ✅ `data/`、`scripts/evaluation/intraday_analysis.py`、`scripts/ingest/check_data_quality.py` | SDK → 行情/财务/日内结构画像，数据质量检查无 ERROR |
| 2 | 提出想法 | ✅ `scripts/factors/mine_factors.py`（`--exhaustive`/`--gp`）、`factor/gflownet/`、`factor/rl/` | 穷举 + 遗传规划 + GFlowNet(TB/PPO) + AlphaPool RL 自动生成候选公式 |
| 3 | 开发准备 | ✅ `scripts/ingest/update_data.py` | SDK → Parquet 缓存 + PIT 面板 + 股票池，增量水位正确、PIT 无未来函数 |
| 4 | 开发实现 | ✅ `scripts/build_{technical,fundamental,intraday}_factors.py` | 技术面 9 因子 + 基本面 32 因子 + 日内 14 因子，面板 date×code 口径一致 |
| 5 | 因子分析 | ✅ `research/factor_analysis.py`、`scripts/reporting/factor_correlation.py`、`scripts/pipelines/e2e_backtest.py` | 因子面板 → IC/IR/衰减/分层/NW t/FDR，显著性基于 Newey-West t |
| 6 | 因子构建 | ✅ `scripts/factors/synthesize_factors.py`、`scripts/factors/synthesize_library.py` | IC 加权 / PCA / 正交 / ML Stacking(ridge/gbdt/lambdarank) 四种合成 |
| 7 | 因子入库 | ✅ `scripts/factors/factor_library.py` | registry + panels + evals 三件套，血缘可追溯、可选 `check_dup` 去冗余预检、六维标签、`set-tag`/`monitor`/`regime`/`select_diverse` |

**算子与指标**：`factor/operators.py` 算子注册表（约 51 算子，元素/时序/截面算子）；`factor/technical_indicators.py`
通达信口径指标 57 个（复用 AmazingData 算子库）；`factor/technical.py` 自研 pandas 技术指标 9 个（离线可用）；
`factor/classic.py` 经典因子 7 个；`factor/preprocessing.py` 去极值(MAD/分位)→中性化(行业+对数市值)→标准化(zscore/rank)。

**自动化挖掘算法**：
- **GP 遗传规划** `factor/genetic_mining.py`：DEAP，HallOfFame 精英保留，门诊/滚动 IC 评估。
- **GFlowNet** `factor/gflownet/`：`FactorMDP` 因子构造 MDP + `TBPolicy`/`PPONet`，Trajectory Balance（Phase 0 简化）与 PPO 对照训练，Phase 1 市值中性化 + 10 日调仓 + 低相关筛选。
- **AlphaPool RL** `factor/rl/`：`AlphaPool` 环境 + gymnasium 包装，MaskablePPO + LSTMSharedNet，均值-方差协方差池。

### 02 模型层（✅ 主要能力已就绪）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 模型设计 | ✅ `model/registry.py`（`ModelRegistry`） | 持久化 CSV 注册表（`reports/models/registry.csv`），原子写，支持 list/view/compare/delete，同名再注册=新版本 |
| 2 | 模型训练 | ✅ `model/training.py`（`train_and_register`） | `ml_stacking` 与 `predictor` 双路径，滚动时序 CV |
| 3 | 模型评价 | ✅ `model/evaluation.py` + `scripts/evaluation/walk_forward_model.py` | IC/IR/NW t/p、IC 衰减、分层多空；三段样本外 + 上线期滚动再训练 |
| 4 | 模型迭代 | ✅ 同名再注册 + `research/experiments.py` | 新版本自动入 registry，实验留痕 |
| 5 | 模型上线 | ✅ `model/serving.py`（`register_model_as_factor`） | 模型预测面板回写因子库，命名 `model:<name>_h<horizon>`，血缘双向溯源 |

**预测器与算法**：
- `model/features.py` `FeatureStore`：白/黑名单 → 覆盖率过滤 → 相关性去冗余 → 上限截断的三级特征漏斗。
- `model/labels.py` `LabelBuilder`：horizon 前瞻收益，rank/zscore/raw 三种标签，embargo=horizon。
- `model/predictor.py`：`RidgePredictor`（闭式解）、`LGBMPredictor`（LightGBM）、`TabICLPredictor`（TabICL in-context learning）；`fit_predict_oos()` 扩展窗口时序 CV。
- `model/training.py` stacking 合成分支：ridge / gbdt / gbdt_tuned / lambdarank。

**模型滚动入口** `scripts/evaluation/walk_forward_model.py`：
`--mock/--real`、`--methods ridge,gbdt`、`--horizon`、`--mode rank|zscore|raw`、`--n-folds`、`--min-train-days`、
`--dedup-corr`、`--max-features`、`--save-library`（OOS 面板回写因子库）、`--registry-root`。
mock 落 `reports/models_mock`，真实落 `reports/models`。

### 03 优化层（组合优化/风险归因/多期执行已就绪）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 组合优化 | ✅ `optimize/portfolio.py` + `optimize/solver.py` + `scripts/portfolio/compare_portfolio_methods.py` | 启发式投影 + 求解器双通道（详见下）；对比脚本覆盖 projection/min_var/tev/risk_parity/hrp |
| 1b | 多期执行 | ✅ `optimize/multi_period.py` + `scripts/portfolio/multi_period_backtest.py`、`scripts/portfolio/generate_signals.py` | D/W/M 调仓频率，调仓日重解 QP，不可交易标的 α=NaN；`generate_signals.py` 转为每日可执行指令（含涨停不可买/跌停不可卖/停牌冻结/整手化） |
| 2 | 风险归因 | ✅ `optimize/risk.py`（`risk_attribution`） | α/β 分解（CAPM/多因子 + Newey-West）、基准对照、Brinson 归因（Carino 链接） |
| 3 | 持续监控 | ✅ `optimize/monitor.py` + `monitoring/` + `scripts/reporting/monitor_performance.py` | 滚动 IC/漂移/衰减/自相关 + 生产化监控（见「生产化监控」）；定时自动化已实现 |

**组合优化算法**（`optimize/solver.py`）：
- 协方差估计：滚动窗口 + Ledoit-Wolf 收缩（严格防前视，只用调仓日前数据）。
- cvxpy QP 五种方法：`min_var` / `tev` / `mvo` / `risk_parity` / `bl`（Black-Litterman 均衡先验 + 观点后验）。
- 约束：预算、个股上下限、行业中性/偏离、风格中性化、换手（线性投影 + 二次 Almgren-Chriss 成本）、多空（short_limit/gross_limit）、保证金占用。
- HRP（`hrp_weights`，Ward 聚类 + 递归二分 + 逆方差，免矩阵求逆）。

### 04 生产化监控（✅ 已落地）

`monitoring/` 包 + `scripts/reporting/monitor_performance.py`：

- **指标** `metrics.py`：IC 漂移、覆盖率、数据新鲜度、分位单调性、多空日均；模型因子以注册时 `ic_mean` 为期望基线。
- **告警** `alerts.py`：因子级 5 规则（stale_data/coverage_drop/ic_decay/significance_loss/monotonicity_break）+
  信号级 5 规则（signal_stale/coverage/concentration/turnover/blocked）+ 库级拥挤度（factor_crowding，IC 相关 + PC1 解释度）；`rollup_status` → normal/warning/critical。
- **账本** `ledger.py`：`snapshots.csv` + `alerts.csv`，同 `run_date` 幂等覆盖，`history()` 跨运行追踪。
- **调度与报告** `runner.py`：编排 + 自包含 HTML 报告（inline SVG sparkline，无外部依赖）+ `next_run_time()` 调度函数。

**CLI** `scripts/reporting/monitor_performance.py`：单次运行 / `--daemon HH:MM` 常驻 / `--register-model-factors`
（把 h=1 模型预测回写因子库为 `model:*`）/ `--task-cmd`（生成 Windows 计划任务）/ `--signal-path`（信号层监控）。
调度方式支持常驻进程、Windows 计划任务或 cron。

### 05 研究层与报告

- **因子检验** `research/factor_analysis.py`：Spearman/Pearson IC、IR、IC 衰减、分层回测。
- **归因** `research/attribution.py`：Fama-MacBeth 两步回归、Brinson（Carino 链接）、α/β 分解。
- **基准** `research/benchmarks.py`：等权/买入持有基准。
- **多样性筛选** `research/dpp_selection.py`：log-det 最大化的 DPP 集合级筛选 + 贪心 pairwise 去重。
- **实验记录** `research/experiments.py`：CSV 实验档案，run_id/指纹/metrics/note。
- **报告渲染** `research/html_report.py` + `research/xlsx_report.py` + `report_pipeline.py`：
  自包含 HTML（Chart.js，A 股红涨绿跌）与 XLSX（openpyxl + 图表嵌入）；一键端到端报告 `scripts/reporting/generate_report.py`。
- **稳健统计** `research/robust_stats.py`：Newey-West t、OLS+HAC、自动滞后阶选择。

## 主要实验与交付物

| 实验 | 入口 | 结论 / 交付 |
|---|---|---|
| **ML 因子合成**（HS300 2022-2025 三段） | `scripts/archive/ml_synthesis_experiment.py` | h=5 valid IC 0.055-0.061 但 test 归零（29/35 特征方向翻转，低波/低价风格 2025 反转）；h=1 样本外 IC 0.039-0.041（NW-t 2.5-2.9）；诚实披露 h=1 选择存在数据窥探 → `reports/archive/ml-synthesis-hs300-report/` |
| **算法对比** | `scripts/archive/ml_algorithm_compare.py` | 2019-2026 长时段 Ridge/GBDT/TabICL + 窗口/再训频率对比 → `reports/ml_algorithm_compare/` |
| **模型 walk-forward** | `scripts/evaluation/walk_forward_model.py` | mock/real 滚动再训练，模型因子回写因子库 → `reports/models(_mock)/` |
| **组合方法对比** | `scripts/portfolio/compare_portfolio_methods.py` | projection/min_var/tev/risk_parity/hrp 五法对比 → `reports/portfolio_methods_compare.csv` |
| **多期执行** | `scripts/portfolio/multi_period_backtest.py`、`backtest_two_periods.py` | 2025 与 2026H1 两段样本外回测 → `reports/multi_period/`、`reports/two_periods/` |
| **选股与信号** | `scripts/portfolio/select_stocks.py`、`generate_signals.py` | 每日选股明细 + 可执行交易信号 → `reports/select_hs300_2025/`、`reports/signals/` |
| **端到端选股（今日信号）** | `scripts/pipelines/e2e_stock_picks.py` | 因子筛选 → GBDT 预测 → risk_parity 组合 → 选股清单 → `reports/e2e_picks/` |
| **端到端策略回测** | `scripts/pipelines/e2e_backtest.py` | walk-forward 月频回测（2024-01~2026-08 跑输全池基准，见报告）→ `reports/e2e_backtest/` |
| **投资收益报告** | `scripts/reporting/investment_report.py` | 模型预测作因子检验（IC/IR/NW-t/分层图）+ 组合 vs 大盘指数基准（沪深300→000300.SH，全A→000001.SH）→ `reports/investment_report/` |
| **因子库检验报告** | `scripts/reporting/factor_library_full_report.py` | 821 因子全量检验表（IC/ICIR/NW-t/分层收益/换手，可排序+搜索，top 因子分层净值图），借用公开库标注"无挖-验分离" → `reports/factor_library_report_hs300_2022_2025.html` |
| **交互式因子检测** | `scripts/reporting/factor_explorer_report.py` | 821 因子交互报告：时间段选择 × 来源筛选 × 指标排序，详情含 5/10 层 × 月/周分层净值+多空线、IC 序列+MA、IC 衰减、月度热力图、各层绩效表 → `reports/factor_explorer_hs300_2022_2025.html` |
| **模型增强组合（固化配置）** | `scripts/pipelines/run_model_portfolio.py` | 模型信号 → 风格中性化 → TopFrac 重仓多头，`reports/model_portfolio/`；默认 gbdt+中性化+Top20%+月频，2025 test 段成本后超额沪深300 **+4.85%**（Sharpe 1.70） |
| **调仓频率精修** | `scripts/evaluation/freq_tune.py` | gbdt+中性化+Top20% 上放开 h×freq 网格，验证换手吞噬收益 → `reports/freq_tune/`；h1×M 最优，日频超额 −40% |
| **多年度 OOS 稳健性** | `scripts/evaluation/multiyear_oos.py` | gbdt/ridge/ranker × h1/h5 × D/W/M，2023/2024/2025 分年 walk-forward（特征定型期固定防前视）→ `reports/multiyear/`；h1×M 唯一三年一致稳健解 |
| **全A多年度滚动训练** | `scripts/pipelines/rolling_grid_alla.py` + `rolling_grid_report.py` | 全A 5549 股 × 798 公因子（all_a_2018_2026 数据集）2018~2026 分年 walk-forward 网格：horizon{1,5,10,20}×频率×模型/超参×中性化×集中度 = 120 组合 → `reports/alla_rolling/report.html`；h1+月/周频+gbdt 一致最优（Top10% raw 年化 14.0%、超额上证 +11.8%、正超额 7~8/9 年），日频成本全灭，风格中性化在全A上反而稀释 alpha（与 HS300 结论相反） |
| **全A每日选股排名（生产化推理）** | `scripts/pipelines/alla_daily_rank.py` | 上述最优方案（gbdt h1×M raw Top10%）的每日盘后推理：数据增量更新 → 尾部重算当年入选 50 因子（同一复权基准自洽）→ 500 日窗重训 → 全A排名 + Top10% 可交易候选 → `reports/alla_daily/`；与实验 OOS 面板末日截面 Spearman 0.92；`--install-task 17:30` 注册每日计划任务 |
| **全A超额归因与显著性复核** | `scripts/pipelines/alla_excess_attribution.py`（2026-09-08） | 复跑主策略取每日权重：对上证超额 +10.2%/年中约一半来自风格敞口（β=1.10、R²=0.61），对全A等权纯选股 α=+5.1%/年（t=2.14 显著）；Brinson（申万一级）：主动收益 +35.6% = 选择 +64.1% + 配置 −15.5% + 交互 −12.3%——超额全部来自行业内选股。回测指标同步新增 `sharpe_t_stat` / `years_to_prove`=(1.96/\|SR\|)² / `excess_t_stat`（主策略 8.35 年：Sharpe t=1.69、超额 t=1.947 压线）→ `reports/alla_attribution/`，网格报告已含 t 列 |
| **论文复现因子族（awesome 21 式）** | `factor/paper_factors.py` + `scripts/factors/build_paper_factors.py`（2026-09-08） | awesome-systematic-trading 复现库 61 策略中 21 个可在 A 股数据面实现者翻译入库（all_a_2018_2026 达 900 因子）：短期反转 IC=0.038/t=11.3、低波 t=8.2、价值 t=8.5、研发强度 t=5.6、质量/FSCORE t≈4.4 显著为正；月频动量族为负（A 股动量反转复现）→ registry `source=paper:awesome-systematic-trading:*` |
| **日内研究** | `scripts/evaluation/intraday_analysis.py` | 隔夜 vs 日内收益分解、成交量/波动率时段效应 → `reports/intraday_analysis_{year}.png`、`intraday_summary_{year}.csv` |
| **自动因子挖掘** | `scripts/evaluation/gp_tune_budget.py`、`run_gflownet_phase0/1.py`、`train_htai_rl_p0.py`、`gflownet_library_ingest.py` | GP 调参 / GFlowNet TB+PPO / AlphaPool RL 最小闭环 → `reports/gp_tune/`、`reports/_htai_gp/` |
| **文本挖掘** | `scripts/factors/fetch_textmining.py` + `scripts/textmining/` | 研报/公告抓取 → FADT/SUE-文本 样本、BERT 编码、训练评估 → `reports/textmining*/` |
| **生产化监控** | `scripts/reporting/monitor_performance.py` | 因子与模型预测性能监控 → `reports/monitoring/` |
| **设计文档** | — | 模型层/训练纪律/项目总览 → `reports/docs/design/yuriquant_{model_layer_design,training_discipline,project_overview}/` |

## 待办 / 缺口 / 历史交付

> **单一真源：本节不再在这里维护。**
>
> - **待办与已知缺口**（研究验证欠账 / 功能缺口 / 工程债务 / 低优先级 / 推进顺序）
>   → [`TODO.md`](TODO.md)
> - **研报研读与复现待办**（P0/P1/P2 分层 + 复现验收标准）
>   → [`RESEARCH_TODO.md`](RESEARCH_TODO.md)
> - **历史交付记录**（2026-08-25 / 2026-09-01 各批完成的细节与实测数字）
>   → [`TODO.md`](TODO.md) 的「附录：历史交付记录」
> - **逐日工作日志与踩坑** → `.workbuddy/memory/YYYY-MM-DD.md`

架构与完成度总览见上节「研发流程与完成度」；交付物总览见「主要实验与交付物」。

## scripts 目录索引

**2026-09-11 按功能重排**：root 原有 **56 个扁平 .py** 已全部归入 7 个受跟踪子目录
（root 现在只剩 `__init__.py`）。调用方式统一为 `python -m scripts.<group>.<script>`，
`python scripts/<group>/<script>.py` 亦可（两者等价）。

> ⚠️ **不用 `scripts/data/` 与 `scripts/reports/`**：与顶层 `data/` 包、`reports/`
> 产物目录同名。以「文件路径」方式跑脚本时解释器会把脚本所在目录放进 `sys.path[0]`，
> `import data` 会被 `scripts/data/` 劫持 —— 故改为 `ingest/` 与 `reporting/`。

### 调度入口速查（计划任务指向这里）

| 脚本 | 新路径 | 计划任务 |
|---|---|---|
| `alla_daily_rank` | `scripts.pipelines.alla_daily_rank` | `YuriQuant AllaDailyRank` |
| `monitor_performance` | `scripts.reporting.monitor_performance` | `YuriQuant Monitor`（每日 17:30） |

> 🚨 任务里内嵌的是**绝对路径**。搬目录后必须重注册：
> `python -m scripts.pipelines.alla_daily_rank --install-task 17:30`

### 命名相近但职责不同的脚本（老规矩，别弄混）

`walk_forward`（因子挖掘三段验证）≠ `walk_forward_model`（模型层滚动 OOS）；
`synthesize_factors`（SDK 在线合成）≠ `synthesize_library`（离线合成）；
`e2e_backtest`（端到端选股 walk-forward）/ `backtest_two_periods`（两期面板回测）/
`multi_period_backtest`（多期组合执行）为三件不同的事。

### `scripts/common/` — 公共库（被多方 import，**非入口**，改它影响面最大）

| 模块 | 职责 |
|---|---|
| `cli_common` | 脚本层公共 CLI 骨架（argparse / logging / 三态数据源样板）+ build 家族助手 |
| `e2e_common` | e2e 家族编排（`load_daily_data` / `select_features` / `drop_stale_factors` / 风格中性化） |
| `portfolio_common` | 跨实验组合编排（`neutralize_panel` / `load_index_benchmark` / `build_model_panel` 等 legacy HS300 口径组件） |

### `scripts/ingest/` — 数据更新 / 回补 / 体检

| 脚本 | 职责 |
|---|---|
| `update_data` | 增量更新行情/财务缓存（`--pool` / `--no-minute`） |
| `update_etf` | ETF 行情更新（需系统 Python 3.12 + SDK） |
| `fetch_status_batched` | 批量拉全A状态表（涨跌停/停牌/ST），断点续拉 |
| `check_data_quality` | 拉数后质量体检：缺失率/复权跳变/价格异常告警 |
| `extend_factor_library` | 因子库面板延长到数据源最新交易日 |

### `scripts/factors/` — 因子构建 / 合成 / 入库 / 挖掘

| 脚本 | 职责 |
|---|---|
| `build_alpha_factors` | Alpha101 / GTJA Alpha191 公开因子构建入库 |
| `build_fundamental_factors` | 财务 PIT 因子（价值/质量/成长/规模） |
| `build_technical_factors` | 技术迭代/累积类指标因子 |
| `build_paper_factors` | 论文复现因子 21 式（awesome-systematic-trading 经典策略 → A 股截面因子） |
| `build_intraday_factors` | 5 分钟 K 线 → 日频因子 |
| `build_intraday_stat_factors` | 日内统计因子（时段效应族） |
| `build_minute_panel` | 分钟长表 → MemMap 稠密面板 + 覆盖统计（`--features` / `--demo-ic`） |
| `synthesize_factors` | 多因子合成 CLI（SDK 在线版） |
| `synthesize_library` | 因子库合成（离线版，免 SDK） |
| `factor_library` | 因子库 CLI（list / compare / export / delete） |
| `mine_factors` | 因子挖掘 CLI（exhaustive / GP / GTJA 预设） |
| `run_gflownet_phase0` / `run_gflownet_phase1` | GFlowNet 挖掘：最小闭环 / 真实 HS300 对齐研报 |
| `gflownet_library_ingest` | GFlowNet 因子入库 |
| `fetch_textmining` | 拉取文本挖掘数据（研报/公告） |

### `scripts/pipelines/` — 主线管线（主实验 / 生产 / 端到端）

| 脚本 | 职责 |
|---|---|
| `alla_daily_rank` | 全A每日模型选股排名（最优方案生产化推理，计划任务 `YuriQuant AllaDailyRank`） |
| `rolling_grid_alla` | 全A多年度滚动训练实验（2018~now × horizon × 频率 × 模型网格） |
| `run_model_portfolio` | 模型增强组合正式入口（h=1 / gbdt / Top20% 月度调仓） |
| `alla_excess_attribution` | 全A主策略超额归因（α/β + Brinson）→ `reports/alla_attribution/` |
| `e2e_backtest` / `e2e_stock_picks` | 端到端选股 walk-forward 回测 / 今日选股流水线 |

### `scripts/portfolio/` — 组合构建 / 信号 / 执行

| 脚本 | 职责 |
|---|---|
| `generate_signals` | 每日可执行交易信号导出 |
| `optimize_e2e` | 调仓频率 × 风格中性化端到端优化 |
| `multi_period_backtest` | 多期组合执行回测（QP + 成本 + 约束） |
| `select_stocks` | 因子库 → 选股回测演示入口 |
| `run_etf_rotation` | ETF 轮动最小闭环回测 |
| `compare_portfolio_methods` | 组合法对比（等权/因子加权/TopK + 中性投影） |

> 注：需要 Σ 的 QP / HRP / BL 在 `optimize/` 包，不在本目录。
> 本目录只做「无风险模型的权重生产」（属于组合构建）。

### `scripts/evaluation/` — 回测 / 评估 / 调参 / 实验网格

| 脚本 | 职责 |
|---|---|
| `walk_forward_model` | 模型层滚动 OOS（`rolling_oos`，唯一生产切分入口） |
| `walk_forward` | 因子挖掘三段样本外：train 挖 → valid 选 → test 验 |
| `cpcv_eval` / `cpcv_h1_eval` | CPCV（组合purged CV）评估 / h=1 专项 |
| `multiyear_oos` | 多年度 OOS（2023/24/25 × h1/h5 × D/W/M） |
| `freq_tune` | 调仓频率精修（h=1 时 M 月度最优） |
| `buffer_tune` | 缓冲带（buffered top-k）调参 |
| `backtest_two_periods` | 两期（预热+研究）基本面/技术因子面板回测 |
| `gtja_repro_eval` | 国金 Alpha 掘金系列复现评估 |
| `gp_tune_budget` | GP 挖掘预算调参（DEAP 迭代/种群规模） |
| `intraday_analysis` | 日内收益分解 + 时段效应分析 |

### `scripts/reporting/` — 报告渲染 / 归因 / 监控

| 脚本 | 职责 |
|---|---|
| `investment_report` | 投资收益报告（端到端最终交付物） |
| `factor_explorer_report` | 交互式因子检测报告（时间段/来源筛选/指标排序） |
| `factor_library_full_report` | 全因子库标准检验汇总报告（自包含 HTML） |
| `risk_decomposition_report` | 组合级风险分解报告 |
| `generate_report` | 一键汇总 reports/ 产物为单个 HTML 报告（`research.report_pipeline` 的 CLI 壳） |
| `rolling_grid_report` | 全A滚动实验的收益曲线 + 分年绩效 HTML |
| `jq_style_report` | 聚宽风格收益曲线页（vs 国证A指≈中证全指，`--run-id` 选组合） |
| `attribution` | 收益归因 CLI（三大归因框架） |
| `factor_correlation` | 因子两两相关性矩阵报告 |
| `monitor_performance` | 生产化监控调度（计划任务 `YuriQuant Monitor` 每日 17:30） |

> 所有报告统一走 `research/html_report.page()` 外壳（`css=` 整体替换 /
> `extra_css=` BASE 在前增量在后），守卫 `tests/test_report_shell.py`。

### `scripts/builders/` — 全A 数据集构建 / 回补管线（受跟踪）

2026-09-11 自 gitignored 的 `scripts/oneoff/` 迁入（22 个模块）。
⚠️ 之所以必须入库：生产入口 `alla_daily_rank` 直接 import 它，迁出前**干净 clone
上生产会 ImportError**。详见该包 `__init__.py`。

| 分组 | 模块 |
|---|---|
| 因子面板构建器（→ `all_a_2018_2026`） | `build_alla_alpha_panels`（alpha101/158/191/360）、`build_alla_fundamental_factors`（B族基本面，**闭包基座**）、`build_alla_constructed_factors`（B++构造型）、`build_alla_pledge_factors`（B+质押/预告）、`build_alla_holder_factors`（股东结构）、`build_alla_status_factors`（停牌/ST）、`build_alla_style_factors`（C·Style）、`build_alla_event_factors`（事件驱动）、`build_alla_margin_factors`（两融）、`build_alla_moneyflow_factors`（机构资金流）、`build_alla_sue_pledge_factors`（SUE+质押深度）、`build_alla_disc_holder_dyn`（大宗折价+股东动态）、`build_alla_factor_neutralized`（因子层中性化面板） |
| 数据回补器（面板的上游输入） | `fetch_alla_history`、`refetch_status_all_a`、`backfill_financial_alla`、`backfill_holder_alla`、`backfill_pledge_profit_alla`、`backfill_delisted_kline` / `_backward` / `_equity` |
| 编排器 | `run_evt_margin_build`（事件/两融/资金流：回补 + 构建串行，防 SDK 单连接竞争） |

用法示例：`python -m scripts.builders.build_alla_alpha_panels --workers 6`；
多数支持 `--resume` 断点续跑。

### `scripts/archive/` — 归档区（受跟踪，"已产出结论、不再迭代"）

`daily_pipeline`（被 monitor/update/extend 拆散取代）、`factor_screening`（消费端未接线）、
`factor_usability_stats`（无 main，功能被覆盖）、`dpp_library_compare`（历史对比实验）。
2026-09-11 新增 7 个（零 import 引用的一次性/对比/诊断脚本）：
`compare_htai_fitness`、`compare_ml_synthesis`、`diagnose_factor_vs_model`、
`diagnose_neutralized_compare`、`gtja_discipline_eval`、`ml_algorithm_compare`、
`ml_synthesis_experiment`。

### `scripts/textmining/` / `scripts/data_tools/` / `scripts/oneoff/`

| 目录 | 状态 |
|---|---|
| `textmining/` | 文本因子线（22 个），**另一会话维护**，本批未动 |
| `data_tools/` | 数据工具（3 个） |
| `oneoff/` | 🚫 **整目录 gitignored**（本地一次性脚本 / 探针，0 跟踪）。**别往这里放被 import 的代码** —— 干净 clone 上是空的。规则见 `scripts/oneoff/README.md` |

> **子目录脚本的 project root 引导**：`scripts/<group>/x.py` 用
> `Path(__file__).resolve().parents[2]`（比 `scripts/` 根下多一层）；
> `scripts/` 根只剩 `__init__.py`，一般不需要。

## 安装

本项目依赖用 `pyproject.toml` 管理，可直接装成可编辑包：

```bash
pip install -e ".[dev]"
```

**复现锁**：仓库提交 `uv.lock`（`uv lock` 生成的跨平台解析锁，覆盖全部可选依赖组，
关键版本与实测兼容矩阵一致）。需要精确复现环境可用：

```bash
uv sync --frozen --extra dev --extra ml --extra gp --extra solver
```

CI 以 `uv lock --check` 守卫锁文件与 `pyproject.toml` 保持同步；`rl` 组含
torch（CUDA 体积过大），CI 与本地默认不装，有需要时单独 `--extra rl`。

CI 分两层（2026-09-10 起）：push/PR 只跑**快检查**——`ruff check
--select F821,F811,F522,F523,F632`（未定义名/重复定义这类真 bug）+
三个分层守卫测试，约 1 分钟；**全量 pytest 移到仅手动触发**的
`full-tests.yml`（Actions 页 → Run workflow）。原因是全量套件要 7 分钟，
且依赖 `reports/` 等不入库的实验产物（CI 上必然缺文件，只能靠 skip 兜住），
本机数据齐全时跑更有意义。

**例外：AmazingData SDK 不在上述依赖里。** 它是银河证券私有分发的本地 wheel，
不在 PyPI 上，需要按开发手册（`AmazingData开发手册.pdf` 3.3 节）单独安装：

```bash
pip install AmazingData-*.whl
```

没有这个 SDK 也能用——`config/settings.yaml` 里把 `datasource.type` 切成 `csv`，
或直接跑下面的 Mock 数据模式，不需要任何真实凭证。

## 快速开始

跑一次 Mock 数据端到端回测（不需要数据源凭证）：

```bash
python scripts/pipelines/e2e_backtest.py --top 20 --model ridge --n-days 400 --n-codes 30
```

用真实数据（需要先在环境变量里配置好 `AMAZINGDATA_USER` / `AMAZINGDATA_PWD` /
`AMAZINGDATA_HOST` / `AMAZINGDATA_PORT`）：

```bash
python -m scripts.ingest.update_data                  # 先建/增量更新缓存
python scripts/pipelines/e2e_backtest.py --real          # 端到端选股 walk-forward 回测
python -m scripts.pipelines.run_model_portfolio          # 正式投产入口（全A 正交化管线）
```

更新本地数据缓存：

```bash
python -m scripts.ingest.update_data                  # 日K线 + 配置的分钟档位（默认 5 分钟）
python -m scripts.ingest.update_data --minute 1,5,15  # 指定分钟档位
python -m scripts.ingest.update_data --no-minute      # 只更新日频数据
```

分钟频率（日内研究）：`data/datasource.get_minute_kline` + `data/cache.get_minute_kline`
按 AmazingData 手册 `query_kline` / `Period.minN` 实现，支持
1/3/5/10/15/30/60/120 分钟八档，缓存文件 `min{period}.parquet`
（索引 `(kline_time, code)`），按交易日增量更新、半拉天自动补全。

### 模型滚动训练（快速验证）

```bash
python scripts/evaluation/walk_forward_model.py --mock   # 无 SDK，mock 数据跑通管线
python scripts/evaluation/walk_forward_model.py --real --methods ridge,gbdt --horizon 5 --save-library
```

### 模型增强组合（固化配置，正式投产入口；2026-09-09 升级为全A正交化管线）

```bash
python -m scripts.pipelines.run_model_portfolio                  # 全A·ortho因子层·ens_h1h5集成·raw信号·月频Top10%
python -m scripts.pipelines.run_model_portfolio --frac 0.20      # 覆盖持仓比例
python -m scripts.pipelines.run_model_portfolio --refresh-base   # 日线更新后先重建基础面板
python -m scripts.pipelines.run_model_portfolio --no-train       # 复用上次预测缓存（快速回测/选股）
python -m scripts.pipelines.run_model_portfolio --pre-cost       # 同时输出成本前口径
```
2026 样本外实测（net）：年化 +8.6%、超额上证 +8.2%、Sharpe 0.33；同步导出当日
Top10% 选股清单 `picks_YYYY-MM-DD.csv`。参数真源在 `config/settings.yaml` 的
`model_portfolio` 段（**费率除外**——交易成本的唯一真源是顶层 `costs` 段，2026-09-10
起引擎缺省与 `default_costs()` 共用它，全项目不再有第二份费率）；
结果落 `reports/model_portfolio/`。旧版 HS300 口径
（单模型+信号层中性化）已退役，函数保留为 legacy 供 buffer_tune/freq_tune 复用。

### 全A每日模型选股排名（最优方案生产化推理，2026-09-08）

```bash
python scripts/pipelines/alla_daily_rank.py                    # 全流程：更新数据→重训→排名（~5分钟）
python scripts/pipelines/alla_daily_rank.py --skip-update      # 离线（数据已更新）
python scripts/pipelines/alla_daily_rank.py --window 750       # gbdt_w750 变体
python scripts/pipelines/alla_daily_rank.py --install-task 17:30   # 注册每日盘后 Windows 计划任务
python scripts/pipelines/alla_daily_rank.py --remove-task
```

把全A滚动实验的最优方案（gbdt h1 + 当年入选 50 因子 + 500 日窗 + raw Top10%）
变成每日盘后一条命令：增量更新全A缓存（盘中运行由 update_data 的守卫自动把
拉取终点回退到上一交易日，防当日半拉K线进缓存——2026-09-02 事故的对策；
SDK 拉表瞬时失败自动重试 3 次）
→ 尾部（训练窗+260 日预热）重算入选因子（量价走 alpha 注册表，B族财务/
B+族商誉质押/B++族构造/A族股东复用 oneoff 构建器；训练窗与预测截面同一次
重算、同一复权基准，不改写冻结的 `all_a_2018_2026` 数据集）→ gbdt 重训预测
最新截面 → 幽灵股守卫 + 信号日停牌/ST/封板标注。输出
`reports/alla_daily/ranking_YYYYMMDD.csv`（全A 排名）、`picks_YYYYMMDD.csv`
（Top10% 可交易候选，等权参考）、`history.csv`（逐日漂移监控）、
`latest_ranking.csv`（稳定路径副本）。口径披露：训练段用发布时点已知的全部
标签（实时预测无未来可窥，实验的 embargo 是回测隔离）；可交易性为信号日
状态估计（T+1 一字板不可预知）；跨年无当年选择文件时回退最近年份并告警。

> 顺带修复数据层一个实际 bug（2026-09-08）：`DataCache.get_calendar` 在
> `end=None` 时永不回源，本地日历被历史某次显式 end 调用封顶（实测卡在
> 20260902）后，所有"更新到最新"的调用永远看不到新交易日——已改为 end=None
> 语义 = "覆盖到今天"（`tests/test_data_layer.py::test_calendar_end_none_covers_today`）。

### 生产化监控

```bash
python scripts/reporting/monitor_performance.py                          # 单次监控
python scripts/reporting/monitor_performance.py --daemon 18:30            # 每日常驻
python scripts/reporting/monitor_performance.py --task-cmd                # 生成 Windows 计划任务命令
python scripts/reporting/monitor_performance.py --register-model-factors  # 注册 h=1 模型因子
```

## 日内研究：收益分解 + 时段效应分析

第一步（解释性分析）：拆解隔夜 vs 日内收益结构、刻画成交量与波动率的
时段 U 型、首根 bar 对全天的预测能力。复用现有因子库方法论但产出日频
特征作为可入库因子。

```bash
python -m scripts.evaluation.intraday_analysis --offline --begin 20250101 --end 20251231   # 读缓存（推荐）
python -m scripts.evaluation.intraday_analysis          # 真实数据（需 SDK 登录）
python -m scripts.evaluation.intraday_analysis --mock   # mock 验证管线
```

输出（默认到 `reports/`）：
- `intraday_analysis_{year}.png`：收益分解 + 成交量 U 型 + 时段波动率 + 首根预测（4 面板）
- `intraday_ts_{year}.csv`：日频时间序列（隔夜/日内/全日收益）
- `intraday_summary_{year}.csv`：汇总指标（NW t、方差占比、时段均值等）

## 训练纪律（L2）

所有搜索/训练类算法（穷举/GP/GFlowNet/RL/stacking/predictor）遵循三段式梳理纪律，
由 `config/settings.yaml` 的 `discipline` 段（begin/train_end/valid_end）冻结，
`embargo = 标签 horizon`：
- **train 段**（挖掘 / 粗筛）
- **valid 段**（筛选 / 定权重）
- **test 段**（valid_end 之后，冻结）：只允许 walk_forward 型最终验证与上线后监控，绝不参与挖掘/调参/入库决策。

## 测试

```bash
pytest tests/ -v
```

测试套件覆盖数据层、回测引擎、因子（挖掘/合成/GFlowNet/RL）、模型层、监控、
优化/求解器/信号/多期、文本挖掘、报告与稳健统计等 36 个文件 / 379 个用例。

## Lint / Type-check

```bash
ruff check .
mypy .
```