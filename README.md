# YuriQuant

个人量化研究系统（日频）。数据 → 因子 → 策略 → 回测 → 报告 全链路，
数据源可插拔（AmazingData SDK / CSV），本地 Parquet 增量缓存，
向量化回测引擎带交易成本、涨跌停/停牌过滤和 point-in-time 股票池。

## 目录结构

```
config/     配置加载（settings.yaml + 环境变量占位符）
data/       数据源抽象、本地缓存、股票池、可执行性掩码
factor/     因子基类与因子库
strategy/   因子值 → 组合权重的策略
backtest/   向量化回测引擎、交易成本、绩效指标
research/   因子检验（IC/分层）、图表与 Excel 报告
scripts/    命令行入口（更新数据 / 跑回测）
tests/      pytest 测试套件
```

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
