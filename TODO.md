# YuriQuant 待办清单

> 2026-08-29 六轮工程整改（P0–P5）完成后的基线盘点。整改内容见提交
> `c6019f7` / `2c0a159` / `3b92fd3`（测试兼容修复 → stats 公共层 → scripts 收敛）。
> 现状：核心包达"清晰"档（零循环依赖、统计工具单一真源、521 测试全绿），
> scripts 层仍有存量样板债。按「对结论可信度的影响」排序。

---

## 一、研究验证欠账（最高优先级——影响结论可信度）

口径变更后，以下历史实验结论需重跑才能继续引用：

- [ ] **重跑 `scripts/multiyear_oos.py`**：README 已标注"h>1 结论基于回测引擎
  bug 下的伪结果，需重跑"；`reports/multiyear/` 至今为空。
  "h1×M 唯一稳健解"这一核心结论目前只有 2025 单年支撑。
- [ ] **重跑 `scripts/freq_tune.py`**：同上，h×freq 网格结论需在修复后引擎下
  重新生成（`reports/freq_tune/` 为空）。
- [ ] **重跑 e2e 族报告对齐年化口径**：`perf_stats` 年化 244→252 后，
  `reports/e2e_backtest/`、`reports/investment_report/` 中 e2e 链路历史数字
  与现行口径存在 ~3% 系统性偏移。
- [ ] **攻"跑输全池基准"的研究问题**：等权 top50 +16.2% vs 全池等权 +30.8%
  （中性化后接近但仍负超额，见 README「已完成 2026-08-25」）。信号强度不足，
  可探索方向：特征集扩充（当前仅经典 12 + 因子库）、风格暴露管理、
  选股宽度（top-50 集中度 vs 分散）。

## 二、研究功能缺口（新能力）

- [ ] **分钟频因子挖掘 pipeline**：5 分钟数据已缓存 2022–2026（`min5.parquet`），
  但现有日内因子本质是日频化。现成数据面上最大的未开发资产。
- [ ] **生产级执行**：实盘下单对接、实时行情驱动（当前为日频 + 盘后信号；
  "信号→次日执行"的滑点假设未经真实成交验证）。
- [ ] **文本挖掘合规化**：当前依赖同花顺/巨潮爬虫（页面变更即断），
  生产化需对接 iFinD/Wind/Choice 等商业源，或明确接受"研究性补充"定位。
- [ ] **在线 dashboard**：报告均为静态 HTML，无增量刷新的监控页面。

## 三、工程层剩余债务

### 3.1 结构性（应先做）

- [x] **实验脚本私有函数倒挂残余**：`cpcv_eval.py` / `cpcv_h1_eval.py` /
  `ml_algorithm_compare.py` 仍 import `ml_synthesis_experiment` 的
  `_eval_row` / `_px_panels` / `_fit_predict_valid` / `_monthly_ic` 等私有函数
  （P3 仅解掉了 `_classic_features`）。公共函数应迁至 `e2e_common` 或独立模块。
- [x] **最小 CI**：无 `.github/workflows`。加最简 GitHub Actions
  （pytest + ruff check + tests/test_layering.py 门禁），
  把口径守卫和分层守卫变成强制约束（测试漂移到无法收集才被发现，
  根因就是无 CI）。

### 3.2 机械性（可批量清理）

- [x] **cli_common 推广**（骨架已建、采用率 <15%）：
  - 53 处 `logging.basicConfig` 手写样板 → `setup_logging()`（仅 scripts 层）
  - 22 处手写 `--real`/`--mock` add_argument → `add_real_mock_args()`
- [x] **HTML 报告模板收编**：8 套各自内嵌的模板（`monitoring/runner.py` 的
  317 行 `generate_html_report` 与 `research/html_report.py` 同名异构、
  `investment_report` / `factor_explorer_report` / `factor_library_full_report` /
  `risk_decomposition_report` / `run_etf_rotation` / `report_pipeline`）。
  以 `research/html_report.py` 为基座统一，顺带拆掉超长函数。
- [x] **超长文件/函数拆分**：
  - `factor/genetic_mining.py` 1692 行（`run_gp_mining` 单函数 ~301 行）
  - `factor/alpha191.py` 1273 行（公式库，可辩护）
  - `scripts/mine_factors.py` 800 行、`scripts/factor_explorer_report.py` 的
    `build_factor_data` ~493 行
- [x] **核心包卫生**：25 处 `print(` → logging（factor 15、data 8）；
  10 处 `except: pass` 静默吞异常逐个审查。
- [x] **依赖锁文件**：当前仅 `>=` 下界，加 lock（pip-tools / uv）保证可复现。

## 四、低优先级（知情即可）

- [ ] `factor/technical.py` 与 `factor/technical_indicators.py` 双口径指标
  （有意保留；SAR 有两份实现，`calc_sar` / SDK 版）
- [ ] GFlowNet 自写 PPO（`factor/gflownet/ppo.py`）与 AlphaPool MaskablePPO
  （`factor/rl/`）双轨；`factor/rl` 目前仅测试消费、无 scripts 入口
- [ ] optimize 标注的 P3 待建：完整多期最优执行、风险预算非等权
- [ ] 一次性实验脚本（`gtja_*` / `diagnose_*` / `compare_*` 等）保留原样，
  不迁移 cli_common（改动无收益只有风险）

---

## 建议推进顺序

1. 重跑 multiyear + freq_tune（补核心结论证据链，顺带验证整改后口径）
2. 最小 CI（防回归，约一天）
3. 分钟频挖掘 pipeline（现成数据的最大增量）
4. cli_common 批量推广 + 报告模板收编（机械清理，可穿插）
5. 攻"跑输基准"研究问题本身
