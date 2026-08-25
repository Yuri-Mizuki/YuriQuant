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
scripts/    命令行入口（更新数据 / 挖掘 / 回测 / 实验 / 监控）
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
| `_meta.json` | 各表增量水位 last_date + 数据指纹 | — | — |

### 数据层核心能力

- **数据源抽象** `DataSource` 定义 17 个抽象方法，行情统一返回 multi-index `(date, code)` DataFrame，日期统一 `pandas.Timestamp`，代码统一 `XXXXXX.SH/SZ/BJ`；切换数据源只改配置，不动业务代码。
- **Point-in-time**：`data/universe.py`（Universe 按指数成分构建股票池，沪深300/中证500/中证1000，PIT 取成分）；`data/financials.py`（三表 PIT 展开为日频面板，无未来函数）。
- **可执行性掩码** `data/tradability.py`：处理停牌、涨停封板、跌停封板等约束。
- **离线模式** `data/offline.py`：`OfflineDataSource`（缺缓存抛错）/ `OfflineQuietDataSource`（缓存完整时建面板）。
- **文本挖掘** `data/textmining/`：同花顺研报主源 + 巨潮公告辅源，统一 `fetch_docs()` 入口，parquet 增量缓存（ths/cninfo），PIT 日期过滤。

## 研发流程与完成度

> 三层 14 阶段。✅ 已就绪 · ⚠️ 雏形 · ❌ 待建。每阶段统一格式：**入口 → 输入 → 输出 → 验收标准**。

### 01 因子层（✅ 本系统最完整主干）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 研究分析 | ✅ `data/`、`scripts/intraday_analysis.py`、`scripts/check_data_quality.py` | SDK → 行情/财务/日内结构画像，数据质量检查无 ERROR |
| 2 | 提出想法 | ✅ `scripts/mine_factors.py`（`--exhaustive`/`--gp`）、`factor/gflownet/`、`factor/rl/` | 穷举 + 遗传规划 + GFlowNet(TB/PPO) + AlphaPool RL 自动生成候选公式 |
| 3 | 开发准备 | ✅ `scripts/update_data.py` | SDK → Parquet 缓存 + PIT 面板 + 股票池，增量水位正确、PIT 无未来函数 |
| 4 | 开发实现 | ✅ `scripts/build_{technical,fundamental,intraday}_factors.py` | 技术面 9 因子 + 基本面 32 因子 + 日内 14 因子，面板 date×code 口径一致 |
| 5 | 因子分析 | ✅ `scripts/run_backtest.py`、`scripts/factor_correlation.py`、`research/factor_analysis.py` | 因子面板 → IC/IR/衰减/分层/NW t/FDR，显著性基于 Newey-West t |
| 6 | 因子构建 | ✅ `scripts/synthesize_factors.py`、`scripts/synthesize_library.py` | IC 加权 / PCA / 正交 / ML Stacking(ridge/gbdt/lambdarank) 四种合成 |
| 7 | 因子入库 | ✅ `scripts/factor_library.py` | registry + panels + evals 三件套，血缘可追溯、可选 `check_dup` 去冗余预检、六维标签、`set-tag`/`monitor`/`regime`/`select_diverse` |

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
| 3 | 模型评价 | ✅ `model/evaluation.py` + `scripts/walk_forward_model.py` | IC/IR/NW t/p、IC 衰减、分层多空；三段样本外 + 上线期滚动再训练 |
| 4 | 模型迭代 | ✅ 同名再注册 + `research/experiments.py` | 新版本自动入 registry，实验留痕 |
| 5 | 模型上线 | ✅ `model/serving.py`（`register_model_as_factor`） | 模型预测面板回写因子库，命名 `model:<name>_h<horizon>`，血缘双向溯源 |

**预测器与算法**：
- `model/features.py` `FeatureStore`：白/黑名单 → 覆盖率过滤 → 相关性去冗余 → 上限截断的三级特征漏斗。
- `model/labels.py` `LabelBuilder`：horizon 前瞻收益，rank/zscore/raw 三种标签，embargo=horizon。
- `model/predictor.py`：`RidgePredictor`（闭式解）、`LGBMPredictor`（LightGBM）、`TabICLPredictor`（TabICL in-context learning）；`fit_predict_oos()` 扩展窗口时序 CV。
- `model/training.py` stacking 合成分支：ridge / gbdt / gbdt_tuned / lambdarank。

**模型滚动入口** `scripts/walk_forward_model.py`：
`--mock/--real`、`--methods ridge,gbdt`、`--horizon`、`--mode rank|zscore|raw`、`--n-folds`、`--min-train-days`、
`--dedup-corr`、`--max-features`、`--save-library`（OOS 面板回写因子库）、`--registry-root`。
mock 落 `reports/models_mock`，真实落 `reports/models`。

### 03 优化层（组合优化/风险归因/多期执行已就绪）

| # | 阶段 | 状态与入口 | 说明 |
|---|---|---|---|
| 1 | 组合优化 | ✅ `optimize/portfolio.py` + `optimize/solver.py` + `scripts/compare_portfolio_methods.py` | 启发式投影 + 求解器双通道（详见下）；对比脚本覆盖 projection/min_var/tev/risk_parity/hrp |
| 1b | 多期执行 | ✅ `optimize/multi_period.py` + `scripts/multi_period_backtest.py`、`scripts/generate_signals.py` | D/W/M 调仓频率，调仓日重解 QP，不可交易标的 α=NaN；`generate_signals.py` 转为每日可执行指令（含涨停不可买/跌停不可卖/停牌冻结/整手化） |
| 2 | 风险归因 | ✅ `optimize/risk.py`（`risk_attribution`） | α/β 分解（CAPM/多因子 + Newey-West）、基准对照、Brinson 归因（Carino 链接） |
| 3 | 持续监控 | ✅ `optimize/monitor.py` + `monitoring/` + `scripts/monitor_performance.py` | 滚动 IC/漂移/衰减/自相关 + 生产化监控（见「生产化监控」）；定时自动化已实现 |

**组合优化算法**（`optimize/solver.py`）：
- 协方差估计：滚动窗口 + Ledoit-Wolf 收缩（严格防前视，只用调仓日前数据）。
- cvxpy QP 五种方法：`min_var` / `tev` / `mvo` / `risk_parity` / `bl`（Black-Litterman 均衡先验 + 观点后验）。
- 约束：预算、个股上下限、行业中性/偏离、风格中性化、换手（线性投影 + 二次 Almgren-Chriss 成本）、多空（short_limit/gross_limit）、保证金占用。
- HRP（`hrp_weights`，Ward 聚类 + 递归二分 + 逆方差，免矩阵求逆）。

### 04 生产化监控（✅ 已落地）

`monitoring/` 包 + `scripts/monitor_performance.py`：

- **指标** `metrics.py`：IC 漂移、覆盖率、数据新鲜度、分位单调性、多空日均；模型因子以注册时 `ic_mean` 为期望基线。
- **告警** `alerts.py`：因子级 5 规则（stale_data/coverage_drop/ic_decay/significance_loss/monotonicity_break）+
  信号级 5 规则（signal_stale/coverage/concentration/turnover/blocked）+ 库级拥挤度（factor_crowding，IC 相关 + PC1 解释度）；`rollup_status` → normal/warning/critical。
- **账本** `ledger.py`：`snapshots.csv` + `alerts.csv`，同 `run_date` 幂等覆盖，`history()` 跨运行追踪。
- **调度与报告** `runner.py`：编排 + 自包含 HTML 报告（inline SVG sparkline，无外部依赖）+ `next_run_time()` 调度函数。

**CLI** `scripts/monitor_performance.py`：单次运行 / `--daemon HH:MM` 常驻 / `--register-model-factors`
（把 h=1 模型预测回写因子库为 `model:*`）/ `--task-cmd`（生成 Windows 计划任务）/ `--signal-path`（信号层监控）。
调度方式支持常驻进程、Windows 计划任务或 cron。

### 05 研究层与报告

- **因子检验** `research/factor_analysis.py`：Spearman/Pearson IC、IR、IC 衰减、分层回测。
- **归因** `research/attribution.py`：Fama-MacBeth 两步回归、Brinson（Carino 链接）、α/β 分解。
- **基准** `research/benchmarks.py`：等权/买入持有基准。
- **多样性筛选** `research/dpp_selection.py`：log-det 最大化的 DPP 集合级筛选 + 贪心 pairwise 去重。
- **实验记录** `research/experiments.py`：CSV 实验档案，run_id/指纹/metrics/note。
- **报告渲染** `research/html_report.py` + `research/xlsx_report.py` + `report_pipeline.py`：
  自包含 HTML（Chart.js，A 股红涨绿跌）与 XLSX（openpyxl + 图表嵌入）；一键端到端报告 `scripts/generate_report.py`。
- **稳健统计** `research/robust_stats.py`：Newey-West t、OLS+HAC、自动滞后阶选择。

## 主要实验与交付物

| 实验 | 入口 | 结论 / 交付 |
|---|---|---|
| **ML 因子合成**（HS300 2022-2025 三段） | `scripts/ml_synthesis_experiment.py`、`ml_synthesis_round2.py`、`ml_decay_diagnosis.py` | h=5 valid IC 0.055-0.061 但 test 归零（29/35 特征方向翻转，低波/低价风格 2025 反转）；h=1 样本外 IC 0.039-0.041（NW-t 2.5-2.9）；诚实披露 h=1 选择存在数据窥探 → `reports/ml-synthesis-hs300-report/` |
| **算法对比** | `scripts/ml_algorithm_compare.py`、`ml_window_compare.py` | 2019-2026 长时段 Ridge/GBDT/TabICL + 窗口/再训频率对比 → `reports/ml_algorithm_compare/` |
| **模型 walk-forward** | `scripts/walk_forward_model.py` | mock/real 滚动再训练，模型因子回写因子库 → `reports/models(_mock)/` |
| **组合方法对比** | `scripts/compare_portfolio_methods.py` | projection/min_var/tev/risk_parity/hrp 五法对比 → `reports/portfolio_methods_compare.csv` |
| **多期执行** | `scripts/multi_period_backtest.py`、`backtest_two_periods.py` | 2025 与 2026H1 两段样本外回测 → `reports/multi_period/`、`reports/two_periods/` |
| **选股与信号** | `scripts/select_stocks.py`、`generate_signals.py` | 每日选股明细 + 可执行交易信号 → `reports/select_hs300_2025/`、`reports/signals/` |
| **端到端选股（今日信号）** | `scripts/e2e_stock_picks.py` | 因子筛选 → GBDT 预测 → risk_parity 组合 → 选股清单 → `reports/e2e_picks/` |
| **端到端策略回测** | `scripts/e2e_backtest.py` | walk-forward 月频回测（2024-01~2026-08 跑输全池基准，见报告）→ `reports/e2e_backtest/` |
| **日内研究** | `scripts/intraday_analysis.py` | 隔夜 vs 日内收益分解、成交量/波动率时段效应 → `reports/intraday_analysis_{year}.png`、`intraday_summary_{year}.csv` |
| **自动因子挖掘** | `scripts/gp_tune_budget.py`、`run_gflownet_phase0/1.py`、`train_htai_rl_p0.py`、`gflownet_library_ingest.py` | GP 调参 / GFlowNet TB+PPO / AlphaPool RL 最小闭环 → `reports/gp_tune/`、`reports/_htai_gp/` |
| **文本挖掘** | `scripts/fetch_textmining.py` + `scripts/textmining/` | 研报/公告抓取 → FADT/SUE-文本 样本、BERT 编码、训练评估 → `reports/textmining*/` |
| **生产化监控** | `scripts/monitor_performance.py` | 因子与模型预测性能监控 → `reports/monitoring/` |
| **设计文档** | — | 模型层/训练纪律/项目总览 → `reports/yuriquant_{model_layer_design,training_discipline,project_overview}/` |

## 待建 / 已知缺口

- **完整生产级执行**：代客下单/撮合对接、实时行情驱动（当前为日频 + 盘后信号）。
- **文本挖掘合规化**：当前依赖爬虫，如需稳定生产化建议对接 iFinD/Wind/Choice 等商业源。
- **前端可视化 / 在线dashboard**：当前报告以静态 HTML/XLSX 为主。
- **分钟频日内因子挖掘**：5 分钟数据已缓存（2022-2026）但缺分钟级挖掘 pipeline（现有日内因子本质是日频化）。

### 已完成（2026-08-25）

- **组合级风险分解**（`optimize/risk.py: risk_decomposition`）：Euler 分解（MRC + CR + 占比）+ 风格/行业因子方差贡献（B'ΣB 分解）+ VaR/CVaR 成分分解（历史模拟法）+ 风险预算校验。报告脚本 `scripts/risk_decomposition_report.py`。
- **h=1 模型 CPCV 无偏评估**（`scripts/cpcv_h1_eval.py`）：固定 h=1 的 10 特征 + gbdt 超参，跑 15 条 CPCV 路径产出 IC 分布 + 路径间 t-test + horizon 对比（h=1/h=5/h=20），消除 horizon 选择偏差。
- **端到端选股工作流固化**（`scripts/e2e_common.py` + `e2e_stock_picks.py` + `e2e_backtest.py`）：

  ```
  今日选股:  D:/python/Python312/python.exe scripts/e2e_stock_picks.py --real --top 30
  策略回测:  D:/python/Python312/python.exe scripts/e2e_backtest.py --real --top 50 --freq M
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

## 安装

本项目依赖用 `pyproject.toml` 管理，可直接装成可编辑包：

```bash
pip install -e ".[dev]"
```

**例外：AmazingData SDK 不在上述依赖里。** 它是银河证券私有分发的本地 wheel，
不在 PyPI 上，需要按开发手册（`AmazingData开发手册.pdf` 3.3 节）单独安装：

```bash
pip install AmazingData-*.whl
```

没有这个 SDK 也能用——`config/settings.yaml` 里把 `datasource.type` 切成 `csv`，
或直接跑下面的 Mock 数据模式，不需要任何真实凭证。

## 快速开始

跑一次 Mock 数据回测（不需要数据源凭证）：

```bash
python scripts/run_backtest.py --factor momentum_20
```

多因子对比：

```bash
python scripts/run_backtest.py --factors momentum_20,volatility_20,turnover_20
```

用真实数据（需要先在环境变量里配置好 `AMAZINGDATA_USER` / `AMAZINGDATA_PWD` /
`AMAZINGDATA_HOST` / `AMAZINGDATA_PORT`）：

```bash
python scripts/run_backtest.py --real --factors all
```

更新本地数据缓存：

```bash
python -m scripts.update_data                  # 日K线 + 配置的分钟档位（默认 5 分钟）
python -m scripts.update_data --minute 1,5,15  # 指定分钟档位
python -m scripts.update_data --no-minute      # 只更新日频数据
```

分钟频率（日内研究）：`data/datasource.get_minute_kline` + `data/cache.get_minute_kline`
按 AmazingData 手册 `query_kline` / `Period.minN` 实现，支持
1/3/5/10/15/30/60/120 分钟八档，缓存文件 `min{period}.parquet`
（索引 `(kline_time, code)`），按交易日增量更新、半拉天自动补全。

### 模型滚动训练（快速验证）

```bash
python scripts/walk_forward_model.py --mock   # 无 SDK，mock 数据跑通管线
python scripts/walk_forward_model.py --real --methods ridge,gbdt --horizon 5 --save-library
```

### 生产化监控

```bash
python scripts/monitor_performance.py                          # 单次监控
python scripts/monitor_performance.py --daemon 18:30            # 每日常驻
python scripts/monitor_performance.py --task-cmd                # 生成 Windows 计划任务命令
python scripts/monitor_performance.py --register-model-factors  # 注册 h=1 模型因子
```

## 日内研究：收益分解 + 时段效应分析

第一步（解释性分析）：拆解隔夜 vs 日内收益结构、刻画成交量与波动率的
时段 U 型、首根 bar 对全天的预测能力。复用现有因子库方法论但产出日频
特征作为可入库因子。

```bash
python -m scripts.intraday_analysis --offline --begin 20250101 --end 20251231   # 读缓存（推荐）
python -m scripts.intraday_analysis          # 真实数据（需 SDK 登录）
python -m scripts.intraday_analysis --mock   # mock 验证管线
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