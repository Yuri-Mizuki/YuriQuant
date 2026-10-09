# YuriQuant

**从原始行情到每日选股清单——一套盘后自动跑数据、出榜单、发报告的全链路 A 股量化研究系统。**

大多数量化仓库止步于回测脚本。YuriQuant 覆盖 **数据 → 因子 → 模型 → 组合优化 →
回测 → 监控 → 报告** 的完整研发生命周期，全部围绕一条铁律构建——
**无未来函数、可样本外复现、可直接投产**。

## 🏆 实战战绩

全 A 市场（5549 只股票，含退市回补）、因子层全正交、2018–2026 逐年 walk-forward
样本外、**T+1 开盘可执行口径**（含交易成本，非纸面收益）：

| 配置 | 年化 | Sharpe | 最大回撤 | 换手 |
|---|---|---|---|---|
| **定版信号**：fundind 面板 h1+h5+h10+h20 混合 ⊕ 慢信号 blend + buffer(15/40) 执行层 | **18.07%** | 0.81 | — | **19.1%** |
| 同信号 · 月频 TopFrac 口径（无 buffer） | 18.68% | 0.813 | 34.0% | 49.6% |
| 主基线：GBDT 单周期 · 月频 Top10% | 11.67% | 0.49 | — | — |

慢信号单独成军也有年化 12.8%、换手仅 22% 的防御性配置。定版演进与逐项消融见
`reports/batch1_920_收尾对照表.md`（§⑩–⑯：污染事故更正 → v2 完整口径 →
fundind 快信号采纳 → buffer 参数网格上修 15/40）。

> **口径诚实注记**：定版数字为 income 财报数据补齐（4383 只）**之前**的 v2 数据态
> 存档；同一配置在补齐后的当前数据态重跑约 **17.0–17.2%**（v3/v4 双证）。本项目
> 按纪律不回退数据换数字，引用时须注明数据态（income 4383 前后）。

系统同样诚实地告诉你什么不行——日频调仓在交易成本下直接转负；AlphaPool RL
挖掘被自己的 PBO 检验判负（0.704）后追加 10 倍步长归因确认真实失效才归档；
zz1000 上的 RL/QP 组合优化 24 臂消融无稳健超额；LLM-MCTS 是唯一产出超门槛
因子的挖掘引擎（席位 4/14 突破）但组合层增量 +0.16pp 噪声级、不采纳；另类
数据三族因子席位 0/36；GP/GFlowNet 双双预算饱和。基本面族面板污染事故连更正
重跑的全程记录都在 `reports/` 里。**没有幸存者偏差的战绩表。**

## ✨ 能力巡礼

### 📊 数据层：一切研究的地基

- **数据源可插拔**：商业 SDK / CSV / Mock 三态切换，只改配置不动业务代码
- **本地 Parquet 增量缓存**：15+ 类表（行情/分钟/财务三表/行业/股本/股东/涨跌停停牌/指数成分…），增量水位 + 数据指纹自动管理
- **Point-in-time 纪律**：指数成分 PIT 取股票池、财务三表 PIT 展开为日频面板——回测里不存在未来函数
- **可交易性掩码**：停牌、涨停封板、跌停封板、ST 全部显式建模
- **另类数据自建通道**：7 个免费公开源落地——巨潮快讯 **119 万条**（2014 年起自建存档）、增减持 26.6 万行全历史、宏观日历 6.5 万行、董监高/亲属买卖/实控人变动
- **另类数据三族因子（altf）**：`scripts/builders/build_alla_altfactors.py` 一轮产出 20 个面板因子（增减持/宏观日历 surprise/快讯情绪三族，公告日与 15:00 收盘闸门双 PIT 口径），并入 all_a_2018_2026 registry + ic_h
- **分钟数据内存映射面板**：GB 级分钟行情按年分区 `np.memmap` 稠密化 `[日, bar, 股票]`，全市场分钟数据不进内存，之上直接跑 42 个"分钟→日频"统计特征

### 🧬 因子层：本系统最完整的主干

- **多因子谱系**：50+ 算子注册表、通达信口径指标 57 个、Alpha101/158/191/360、经典 + 券商特征族 36 个、论文复现 21 式
- **全 A 因子库**：**920+ 因子 × 5549 股（含退市）× 2018–2026**，registry + panels + evals 三件套，血缘全程可溯
- **五种自动挖掘引擎**：穷举搜索、遗传规划（DEAP）、GFlowNet（Trajectory Balance 与 PPO 对照训练）、AlphaPool 强化学习（MaskablePPO + LSTM，含 LLM 生成的初始因子池）、LLM-MCTS（UCT 搜索 + 三层去重）——全部引擎已跑到可引用终审：GFlowNet raw+im 臂 114 因子入库（族层 PBO 0.114 稳健），LLM-MCTS all-A 正式跑 750 候选收编 14（族层 PBO 0.039，唯一产出超 top-150 门槛因子的引擎），GP/AlphaPool 预算饱和与真实失效分别归档
- **合成七法**：IC 加权 / PCA / IC_IR 最大 / 半衰加权 / 等权 / 正交 / ML Stacking
- **预处理流水线**：去极值 → 行业 + 市值中性化 → 标准化，向量化实现 **2.96x 提速且与逐日实现逐位一致**

### 🤖 模型层：从因子到预测

- **预测器**：Ridge / LightGBM / TabICL，滚动时序 CV；表格基础模型三级对照收官——TabICL 在 hs300/zz1000 为最优单臂（头部内加权成中小池组合冠军），全A 大截面上 GBDT 不可替代（IC 反超 4.4 倍）
- **标签工程**：多 horizon × 多变换（rank / gauss_rank / zscore / raw）× 超额口径（IR / Calmar）× 隔夜窗口，embargo = horizon 防泄漏；gauss_rank / 隔夜 / IR 三臂全量裁决均为负结果，现行 rank+cc 口径即标签工程维局部最优（两篇 2026 外部文献转译闭环）
- **模型注册表**：持久化版本管理，同名再注册即新版本，实验全留痕
- **模型即因子**：预测面板一键回写因子库，模型 ↔ 因子血缘双向溯源

### ⚖️ 优化层：从预测到持仓

- **组合优化**：QP 五法（min_var / tev / mvo / risk_parity / Black-Litterman）+ HRP；约束覆盖预算 / 上下限 / 行业中性 / 基准行业偏离 / 逐股权重变动上限 / λ 网格
- **风险归因**：α/β 分解（Newey-West t）、Brinson（Carino 链接）、Euler 风险分解 + VaR/CVaR 成分
- **多期执行**：日 / 周 / 月调仓频率，调仓日重解优化，每日可执行信号导出

### 🎯 回测与反过拟合：数字必须经得起拷问

- **向量化回测引擎**：交易成本、涨跌停 / 停牌过滤、**T+1 开盘可执行口径为全链默认**（收盘价成交的乐观数字只作上限对照）
- **稳健统计**：Newey-West t、OLS+HAC、自动滞后阶、FDR 多重检验校正
- **反过拟合三件套**：CSCV PBO 过拟合概率（12870 个组合秒级算完）、Deflated Sharpe Ratio、三段式训练纪律（train / valid / test 分段 + embargo，配置冻结）

### 🏭 生产化：研究系统的最后一公里

- **每日盘后自动推理**：计划任务增量更新数据 → 尾部重算 → 重训 → 全 A 排名 + Top10% 可交易候选
- **生产口径 IC 台账**：每日 IC / 中性化 IC / 风格暴露持续记录，信号衰减一眼可见
- **报告自动化**：自包含 HTML（Chart.js，A 股红涨绿跌）+ XLSX + 邮件日报，一键端到端汇总
- **调度单点管理**：Windows 计划任务 XML 导入式注册 + `doctor` 体检（登记↔系统漂移检测）

## 🏗️ 目录结构

```
data/       数据层：数据源抽象、增量缓存、PIT、可交易掩码、另类数据、分钟面板
factor/     因子层：算子库、五大挖掘引擎、合成、因子库
model/      模型层：特征漏斗、标签、预测器、训练评价、注册表
optimize/   优化层：QP/HRP/BL、风险归因、信号、多期执行
backtest/   向量化回测引擎
strategy/   因子值 → 组合权重
research/   研究工具：IC/分层、归因、DPP、实验台账、报告渲染
stats/      稳健统计与反过拟合（PBO / DSR）
monitoring/ 生产监控：IC 台账、告警规则、账本
scripts/    150+ 脚本入口（ingest / factors / pipelines / portfolio /
            evaluation / reporting / builders / common …）
config/     settings.yaml（参数真源）+ schedule.yaml（调度登记表）
tests/      86 个文件 / 1186 个用例
reports/    实验产物、监控台账与设计文档
```

> 深入细节：任务与缺口看 [`TODO.md`](TODO.md)，研报复现线看
> [`RESEARCH_TODO.md`](RESEARCH_TODO.md)，实验产物与设计文档在 `reports/`。

## 🚀 快速开始

```bash
git clone https://github.com/Yuri-Mizuki/YuriQuant.git
cd YuriQuant
pip install -e ".[dev]"        # 或 uv sync --frozen --extra dev 精确复现锁定环境
pytest tests/ -q              # 1186 个用例，无需任何数据凭证
```

**没有商业数据源也能开发**：`config/settings.yaml` 把 `datasource.type` 切成
`csv`，或给任意管线加 `--mock`——全链路在合成数据上跑通。

接真实数据的主线（需 AmazingData SDK 凭证，见下方说明）：

```bash
python -m scripts.ingest.update_data             # 建立/增量更新 Parquet 缓存
python -m scripts.pipelines.rolling_grid_alla    # 主实验：全A 多年度 walk-forward 网格
python -m scripts.pipelines.run_model_portfolio  # 投产组合：正交化 · h1+h5 集成 · 月频 Top10%
python -m scripts.pipelines.alla_daily_rank      # 每日盘后推理出榜
python -m scripts.reporting.generate_report      # 一键端到端 HTML 报告
```

> **数据源说明**：主数据源 AmazingData 为银河证券私有分发的商业 SDK（非公开，
> 需自行获取授权）。这不妨碍你阅读、复用全部架构与算法代码——数据层抽象让任何
> 数据源（包括你自己的 CSV）都能接进同一条管线。

## 🧪 工程质量

- **测试**：1186 个用例覆盖数据层、回测、因子挖掘、模型、优化、监控、报告全栈
- **CI 两层**：push/PR 快检查（ruff 真 bug 规则 + 分层守卫，约 1 分钟）；全量测试手动触发
- **复现锁**：`uv.lock` 提交入库，`uv lock --check` 守卫；可选依赖按功能域分组（ml / gp / solver / rl），缺失时优雅降级
- **Lint / 类型**：`ruff check .` + `mypy .`

## 📜 免责声明

本项目仅供量化研究与方法论学习，不构成任何投资建议。历史回测收益不代表未来表现。
