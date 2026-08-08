# YuriQuant

个人量化研究系统（日频）。数据 → 因子 → 策略 → 回测 → 报告 全链路，
数据源可插拔（AmazingData SDK / CSV），本地 Parquet 增量缓存，
向量化回测引擎带交易成本、涨跌停/停牌过滤和 point-in-time 股票池。

## 目录结构

```
config/     配置加载（settings.yaml + 环境变量占位符）
data/       数据源抽象、本地缓存、股票池、可执行性掩码
factor/     因子层：算子空间、挖掘（exhaustive/GP）、合成、公式解析
model/      模型层：模型注册表、训练、评价（对齐聚宽 02 模型层）
optimize/   优化层：组合优化、风险归因、持续监控（对齐聚宽 03 优化层）
strategy/   因子值 → 组合权重的策略
backtest/   向量化回测引擎、交易成本、绩效指标
research/   研究工具：因子检验（IC/分层）、归因、基准、实验记录、报告
scripts/    命令行入口（更新数据 / 挖掘 / 回测 / 归因）
tests/      pytest 测试套件
```

## 研发流程（对齐聚宽 AI 投研流程）

三层 14 阶段。✅ 已就绪 · ⚠️ 雏形 · ❌ 待建。每阶段统一格式：
**入口 → 输入 → 输出 → 验收标准**。

### 01 因子层（≈95% 完备，本系统最完整主干）

| # | 阶段 | 入口 | 输入 → 输出 | 验收标准 |
|---|---|---|---|---|
| 1 | 研究分析 ✅ | `data/`、`scripts/intraday_analysis.py`、`scripts/check_data_quality.py` | SDK → 行情/财务/日内结构画像 | 数据质量检查无 ERROR |
| 2 | 提出想法 ✅ | `scripts/mine_factors.py`（`--gp` 遗传规划） | 43 算子空间 → 候选公式 | 候选自动生成，无需手写 |
| 3 | 开发准备 ✅ | `scripts/update_data.py` | SDK → Parquet 缓存 + PIT 面板 + 股票池 | 增量水位正确、PIT 无未来函数 |
| 4 | 开发实现 ✅ | `factor/base.py`、`scripts/build_{technical,fundamental,intraday}_factors.py` | 特征 → 因子面板 | 面板 date×code 口径一致 |
| 5 | 因子分析 ✅ | `scripts/run_backtest.py`、`scripts/factor_correlation.py` | 因子面板 → IC/IR/衰减/分层/NW t | 显著性基于 Newey-West t |
| 6 | 因子构建 ✅ | `scripts/synthesize_factors.py`、`synthesize_library.py` | raw 因子 → 复合因子 | IC 加权/PCA/正交/Stacking 四种 |
| 7 | 因子入库 ✅ | `scripts/factor_library.py` | 因子 → registry + panels + evals | 血缘可追溯、可迭代再挖掘 |

### 02 模型层（≈30%，ML 当前仅用于因子合成）

| # | 阶段 | 入口 | 说明 |
|---|---|---|---|
| 1 | 模型设计 ⚠️ | `model/registry.py`（`ModelRegistry`） | 模型规格/超参/特征以 spec JSON 落盘 |
| 2 | 模型训练 ⚠️ | `model/training.py`（`train_and_register`） | 薄封装 factor/synthesis 的 ML stacking（ridge/gbdt/lambdarank），训练即注册 |
| 3 | 模型评价 ⚠️ | `model/evaluation.py`（`evaluate_model`）+ `scripts/walk_forward.py` | IC/IR/NW t/衰减/分层；三段样本外（train挖→valid选→test验） |
| 4 | 模型迭代 ⚠️ | 同名再注册 + `research/experiments.py` | 新版本自动入 registry，实验留痕 |

### 03 优化层（≈20%，缺口最大）

| # | 阶段 | 入口 | 说明 |
|---|---|---|---|
| 1 | 组合优化 ⚠️ | `optimize/portfolio.py`（`optimize_weights`） | 简单加权 + 约束增强已就绪（行业中性/权重上下限/换手约束）；均值方差/风险平价 ❌ 待建 |
| 2 | 风险归因 ⚠️ | `optimize/risk.py`（`risk_attribution`） | α/β 分解 + 基准对照 + Brinson（Carino 链接）已就绪；组合级风险拆解 ❌ 待建 |
| 3 | 持续监控 ⚠️→❌ | `optimize/monitor.py`（`monitor_report`） | 滚动 IC/漂移/衰减/自相关骨架已就绪；预警阈值 + 定时自动化 ❌ 待建 |

## 数据层缓存

数据流：AmazingData SDK（`data/datasource.py`）→ 自建 Parquet 缓存
`DataCache`（`data/cache.py`，根目录 `e:/data/parquet/`）→ 业务层
（股票池 / 行业 / 财务 PIT / 可执行掩码等）。缓存层扁平存放，每表一个
parquet，`_meta.json` 记录各表增量水位（`last_date`）。

### 表清单与命名映射

| 当前文件 | 内容 | 缓存模式 | 索引/结构 | 建议新名（未来迁移） |
|---|---|---|---|---|
| `daily.parquet` | 日K线 OHLCV+amount | 长表增量 | (date, code) | `quote_daily` |
| `min{period}.parquet` | 分钟K线（如 min5） | 长表增量 | (kline_time, code) | `quote_min{period}` |
| `adj_factor.parquet` | 单次复权因子 | 宽表全量刷新 | date×code 宽表 | `quote_adj_factor` |
| `backward_factor.parquet` | 累积后复权因子 | 宽表全量刷新 | date×code 宽表 | `quote_backward_factor` |
| `history_stock_status.parquet` | 涨跌停/停牌/ST/除权除息 | 长表增量 | (date, code) | `status_history` |
| `income.parquet` | 利润表 | 整表覆盖 | 长事件表 | `fin_income` |
| `balance_sheet.parquet` | 资产负债表 | 整表覆盖 | 长事件表 | `fin_balance_sheet` |
| `cash_flow.parquet` | 现金流量表 | 整表覆盖 | 长事件表 | `fin_cash_flow` |
| `calendar.parquet` | 交易日历 | 合并去重 | date 列表 | `ref_calendar` |
| `index_constituent_{code}.parquet` | 指数成分（000300SH） | 整表覆盖 | 长事件表 | `ref_index_constituent_{code}` |
| `industry_classification_level{N}.parquet` | 行业分类（申万 N 级） | 整表覆盖 | 长事件表 | `ref_industry_level{N}` |
| `equity_structure.parquet` | 股本结构变动事件 | 整表覆盖 | 长事件表 | `ref_equity_structure` |
| `dividend.parquet` | 分红送转 | 整表覆盖 | 长事件表 | `ref_dividend` |
| `share_holder.parquet` | 十大股东 | 整表覆盖 | 长事件表 | `ref_share_holder` |
| `holder_num.parquet` | 股东户数 | 整表覆盖 | 长事件表 | `ref_holder_num` |
| `_meta.json` | 各表增量水位 last_date | — | — | 保持 `_meta.json` |

缓存模式说明：
- **长表增量**：只从数据源拉取本地缺失的日期段，合并去重后全量写盘
  （早期实现曾把过滤子集写回导致丢历史，2026-07-28 已修复）；水位在
  `_meta.json`，增量起点 = `last_date + 1`（分钟线 `last_inclusive` 当天回补）。
- **宽表全量刷新**：SDK 自身维护增量，本地整表重拉去重（复权因子类）。
- **整表覆盖**：稀疏事件表无"增量"概念，每次拉取整表覆盖；水位记录最近
  数据日期（事件表取表内最大日期列，宽表取文件写入时间）。

### 命名规范（约束后续新增表，存量文件名保持不变）

```
<域>_<表>[_参数].parquet
```

- 域前缀：`quote`=行情 / `fin`=财务 / `status`=状态 / `ref`=参考事件 / `meta`=元数据
- 参数后缀统一放末尾、语义一致：频率 `quote_min15`、级别 `ref_industry_level1`、
  标的代码 `ref_index_constituent_000300SH`
- 索引约定：行情/状态长表用 MultiIndex `(time, code)`；事件表保留 `code` 列 + 默认索引
- 若未来迁移存量文件到新命名：同步修改 `data/cache.py` 内对应文件名并重命名数据文件，
  跑一遍 `tests/test_data_layer.py` 回归

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

## 日内研究：收益分解 + 时段效应分析

第一步（解释性分析）：拆解隔夜 vs 日内收益结构、刻画成交量与波动率的
时段 U 型、首根 bar 对全天的预测能力。复用现有因子库方法论但产出日频
特征作为可入库因子（路径 A）。

```bash
python -m scripts.intraday_analysis --offline --begin 20250101 --end 20251231   # 读缓存（推荐）
python -m scripts.intraday_analysis          # 真实数据（需 SDK 登录）
python -m scripts.intraday_analysis --mock   # mock 验证管线
```

输出（默认到 `reports/`）：
- `intraday_analysis_{year}.png`：收益分解 + 成交量 U 型 + 时段波动率 + 首根预测（4 面板）
- `intraday_ts_{year}.csv`：日频时间序列（隔夜/日内/全日收益）
- `intraday_summary_{year}.csv`：汇总指标（NW t、方差占比、时段均值等）

## 测试

```bash
pytest tests/ -v
```

## Lint / Type-check

```bash
ruff check .
mypy .
```
