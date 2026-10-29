# TODO_ARCHIVE — 已收口待办归档（2026-10-27 整合建立）

> **定位**：本文件是三个历史待办文件的**逐字快照合册**，整合自 `TODO.md` /
> `RESEARCH_TODO.md` / `GOOD_MACHINE_TASKS.md`（整合时点 = 2026-10-27，
> 快照 md5 见下）。活跃待办的唯一真源此后 = 根目录 `TODO.md`；
> 本文件只在条目从 `TODO.md` 收口移出时**追加**，不再改写历史内容。
> 报告产物治理规则仍 = `reports/README.md` §1.6。
>
> 整合前快照（md5，存 `~/yuriquant_snapshots/2026-10-27_todo_consolidation/`）：
> `TODO.md e7321f3f…` · `RESEARCH_TODO.md e9f1f70e…` ·
> `GOOD_MACHINE_TASKS.md 8a51949c…` · `ai97_100k.json d5051d74…`（后者已移出 git）

## 导航（三册）

| 册 | 来源 | 内容 |
|---|---|---|
| **册Ⅰ 工程线** | 原 `TODO.md` 全文 | 研究验证欠账 / 功能缺口 / 工程债务 / 好机器后续立项 / 增量方向 §六（A1/A2/B1/B2/B3/C1/D1 历史）/ 历史交付附录 |
| **册Ⅱ 研读复现线** | 原 `RESEARCH_TODO.md` 全文 | 复现状态盘点 / P0 各线终判（AI97、东吴 MCTS、mf10、链路优化、基本面诊断、RL stage2）/ P2 可选 / 验收标准 / 全库分类速查 |
| **册Ⅲ 好机器队列** | 原 `GOOD_MACHINE_TASKS.md` 全文 | 批次 1–14 全部终判 + 迁移清单（环境搭建约定）+ 批次 8 版本口径 + 收尾 SOP |

## 外部引用热点锚点表（引用本归档一律指到「册 + 节名」，不指行号）

| 被引用内容 | 原位置 | 现位置 |
|---|---|---|
| v1w2 基线出处（`config/baselines.yaml` 引用） | GMT:34-36 | **册Ⅲ §〇 当前结论快照** |
| 批次 11 cheap_lever 裁决门槛（≥+1pp 切换判据） | GMT 批次 11 | **册Ⅲ 批次 11 行 + 批次 12① 终裁** |
| 批次 9.2 东吴 MCTS 全量命令 | GMT 9.2 | **册Ⅲ §9.2** |
| 链路优化线收口结论（QP/集中化/aim/HRP 全关） | TODO §一 / RESEARCH_TODO 链路线 | **册Ⅰ §一 + 册Ⅱ §一 链路优化线** |
| 增量方向 A1/A2/B1/B2 立项与判据 | TODO §六 | **册Ⅰ §六** |
| 好机器批次出数后的后续立项（标签三臂等） | TODO §五 | **册Ⅰ §五** |
| 复现范围与验收标准四条 | RESEARCH_TODO §五 | **册Ⅱ §五**（活跃副本 = `TODO.md` §五） |
| mf10 T 扫描预算 | RESEARCH_TODO mf10 节 | **册Ⅱ §一 多因子10 + 册Ⅲ 批次 4** |
| 全库研报分类速查（~87 篇） | RESEARCH_TODO §附 | **册Ⅱ §附** |
| stage2 RL 组合优化生产命令 | RESEARCH_TODO §二 | **册Ⅱ §二 + 册Ⅲ 批次 5** |
| P5 基本面九因子来源 | RESEARCH_TODO §一 | **册Ⅱ §一 基本面诊断 + 册Ⅰ §六 B2** |
| 好机器环境搭建（迁移清单 / tabpfn 版本口径） | GMT §一 / 批次 8 | **册Ⅲ §一 + 批次 8** |

## 各册前言（整合时补注，正文未改动）

- **册Ⅰ**：快照时点仍有活跃项，已全部迁至新 `TODO.md`（D1 ETF 赛事线、
  分钟频扩容、生产化四项、RD-Agent P2、低优先研读、验收标准）。册内残留
  `- [ ]` 为历史快照原样；「知情即可」类记录（§四）视为历史决定不再跟办。
- **册Ⅱ**：**闭账补注（10-27 核对）**——基本面诊断优先级队列中三条过期
  未勾选的 open 已被好机器批次实际消化：**E1''**（horizon_mix 复跑）已由
  批次 10 执行、h1020 已采纳为定版成分；**E5**（fundind）已由批次 6 执行并采纳；
  **E4**（双路径）已由批次 6 证伪结案。「ETF 盲区/出场加速探针组」整条迁至
  新 `TODO.md` §一；§三 P2 可选四条迁至 `TODO.md` §四；§五验收标准迁至
  `TODO.md` §五（本册保留原件）。
- **册Ⅲ**：批次 1–14 全部终判；唯一未执行项 = **批次 12⑥ RD-Agent P2**
  （WSL2 环境阻塞），已迁至新 `TODO.md` §三。§一迁移清单与批次 8 版本口径
  为好机器环境搭建约定，保留作复现存档；`reports/sync/` 的 zip 捆绑
  （batch1213 / batch14）可由册内命令重建。

---

<!-- ═══════════════════════ 册Ⅰ 工程线（原 TODO.md 全文快照，逐字保留） ═══════════════════════ -->

# YuriQuant 待办清单

> **本文件是待办的唯一真源**（研报复现线见 [`RESEARCH_TODO.md`](RESEARCH_TODO.md)；
> 架构与完成度总览、安装与用法见 [`README.md`](README.md)）。
>
> 2026-09-21 大清理：对照已实现代码归档已完成项（另类数据 P0 / 调度收口 / 监控退休 /
> HS300 入口归档 / panels_neu 920 / holder_dyn 消融 / 中性化向量化 / stage2 Phase 0-2 管线等）。
> 清理前全文见 git 历史（`e70ac96` 为 09-20 清理前基线）；历史交付细节压缩于附录。
>
> **2026-10-09 刷新**：好机器批次 1–10 全部出数收官（GOOD_MACHINE_TASKS 已标终态），
> 主模型定版 = fundind 快信号 h1020⊕慢信号 s0.5 + buffer(15/40) 执行层
> （`reports/batch1_920_收尾对照表.md` §⑭⑯）。原 §一 P0 治本重跑归档、§五批次
> 后续立项全部收官；本文件剩项 = **定版切生产链（唯一高优先）** + 小工程收尾 +
> 资源墙后置线。
>
> 🧭 **2026-10-13 起：统一口径 + 定版登记 = 强制约定**。所有「改进 vs 基线」对照**必须**走
> `scripts/evaluation/caliber.py`（口径 7 要素唯一真源：窗口 prod=2018+ / 信号登记 / 逐股冲击成本 /
> `Config.costs()` 费率 / **几何** Sharpe / 每月首日调仓 / Σ 暖机不截断）与 `config/baselines.yaml`
> （基线登记表：`v1-dingban` 生产 · `v1w-ortho882` 本机代理）。**基线不得自造。**
> `caliber.py --check` 自检；详见 `reports/docs/design/定版主模型与统一口径.md`。
>
> 🧭 **2026-10-19 增量立项**：横截面选股在「当前数据 × h1–h20 × long-only Top10%」
> 格子内的边际已收口（直接证据：K≥36 平台、K≥200 逐位相同、非量价 K0 仅 26%；
> 见 `reports/pv_marginal/`、`reports/four_paths/`）。本轮**只开三个未证伪方向**：
> ① h60/h120 长 horizon + 基本面/质量（§六 A1）；② 残差目标挖掘（§六 A2，
> 探针已落 `scripts/evaluation/residual_ic_probe.py`）；③ beta hedge / 指数增强 /
> Barra Σ（§六 B1）。执行纪律不变：`caliber.py` + PBO/DSR + 多窗口 + 安慰剂；
> **不再开**量价因子 / 超参 / QP / 择时（含模型自诊断仓位，10-23 收口）/ regime /
> alt 计数六类已封底线。

---

## 一、研究验证欠账（最高优先级——影响结论可信度）

- [x] **2026-10-19 架构清理（审计 §6.6）**：① `scripts/common/alla_common.py`
  下沉，解除 `rolling_grid_alla` 被 25 个脚本反向 import；② neutralized/fundind
  builder 与 7 个 `load_panels` 样板合并；③ 归档 26 个零引用入口 + 新增
  `scripts/README.md` 索引（CI 校验）；④ `e2e_common` 迁出 common 到 archive。
  详见 `reports/docs/consistency_checks/全项目代码与口径审计_2026-10-19.md` §6.6。

- [x] **定版口径切换生产出榜链**（✅ **10-10 已完成**——schedule.yaml `--preset
  dingban` + 产物 `reports/alla_daily_defv/`；legacy 产物停于 09-22。10-13 审计曾误判
  "未切"，真源在 schedule.yaml 而非脚本 argparse 默认）：
  - ✅ tradable_labels=True 已由莉酱拍板确认（2026-10-15，与批次 1/6 定版臂命令一致）。
  - 🔴 **遗留故障（10-12 起连续失败）**：17:30 计划任务死于**会话撕裂**——任务以
    InteractiveToken 注册（仅登录会话内运行），用户傍晚断连/系统 18:01 入睡/10-14
    17:32 重启都会杀掉任务进程（rc=0xC000013A/0xC0000142/0xC0000409 三种死法同源；
    10-11 会话活着故整链成功 18:17）。**已处置（2026-10-15，由莉酱拍板：改任务时点）**：
    schedule.yaml `17:30→18:30` 避开维护窗口（S4U/白名单方案搁置）+ task_scheduler 模板
    加 RestartOnFailure 3×30min + WakeToRun + 重注册（NextRun=10-16 18:30；当晚 18:00
    的 altdata 出窗口即成功 rc=0，反向验证根因）；daily_report 目标时点登记改 20:00
    （WorkBuddy 端解封时需用户自行同步）。数据缓存 10-10~14 缺口已手动补更
    （logs/update_manual_20261015.log）；**10-10~14 五天榜单不回填**（buffer 状态
    与实时链不符，补出来反而误导）；10-15 榜单已手动产出（18:27 落盘，链 23 分钟全绿）。
  - ✅ **10-15 迁移 18:30 后的次日检验（10-20 复盘）**：**EDR/会话撕裂未再现**
    （10-16 晚整链成功；18:30 时点无 7045/0xC0000142 死法），但暴露**新故障
    10-17~19 三晚无新榜**：①根因 = `update_data` 拉取积压（10-17 起 exit=1 ×3 /
    10-18 跑满 1800s 预算被杀 / 10-19 一次 0xC0000409 + 两次 exit=1，水位停在
    10-16）→ 出榜降级用旧缓存产出重复的 20261016 榜（10-17/18）；②10-19 晚
    18:33 出榜进程日志戛然而止、零产物零重试痕迹（死因未留痕，当晚无 EDR
    窗口，疑 SDK 原生库或资源问题）。**已处置（10-20）**：手动 update 把
    daily_all_a 追平至 10-19（CPU 满载 ~30 分钟消化积压，非卡死）；手动
    `--skip-update` 补出 10-19 榜；**留痕修复**——`alla_daily_rank` 的 update
    子进程 stderr 改落 `logs/update_data_task.log`（原先进 schtasks 黑洞，
    三晚失败只剩 exit code 可猜）。**待观察**：今晚 18:30 任务水位已平、
    预算内应正常；若再失败，看 update_data_task.log 的真实异常。
  - ✅ **10-23 根因终判 + 回归修复（留痕立功）**：10-20~22 三晚再故障
    （10-20/22 任务进程被杀无榜、10-21 全量拉但日 K 先落盘侥幸出榜），
    `update_data_task.log` 首次留痕即定位——**10-19 的 `get_all_a` PIT 审计
    修改引入回归**：晚间 update 以 target=今天 调用，而缓存止于上一交易日
    ⇒ 右界 `last≥今天` 恒假 → 返回空池 → update 误判「本地无缓存」回退
    API 清单 + 日 K 按 begin_date 走 4 年全量拉（10 分钟级）→ 超 1800s
    预算被杀 → 数据永远落后一天（**死循环**）。修复 = `data/universe.py`
    右界钳制到缓存末日（PIT 语义不变），配 `tests/test_universe.py` 3 用例
    （含「查询日晚于缓存末日不得返回空池」回归锁）；实测修复后日 K 41 秒
    （原 10 分钟）、不再走回退分支。10-17~19 的积压与 10-20~22 的死循环是
    **两个独立故障**（前者真积压、后者代码回归），均已收口。
  - ✅ **联动件④⑤ 已落地（10-16）**：④ 监控种子——实时段 10-10 起已读 defv dingban
    榜（CLI 默认值同步），历史段因本机无 fundind 回测 pred 保留 882 ortho ens_h1h5
    （regen 需数小时，待面板重建腾出内存后补），口径断点已在
    `monitoring/production_ic.py` 模块 docstring + CLI 头部如实披露，`source` 列
    分界改为**动态取 hist pred 末日**（原硬编码 2026-09-12 是 legacy 榜残留，与
    defv 首榜 2026-10-08 不符）；⑤ 慢信号滞后披露——出榜链把实际信号月写入
    history.csv 新列 `slow_signal_month`，邮件 Caveat 读该列动态披露
    （滞后月数自动计算；旧产物无该列时降级为静态说明），不再硬编码日期。
- [x] **方案 B 缺面板对齐（10-16 完成）**：实测缺口与 TODO 原记录不符——4 个 horizon
  各只有 **1 列缺失（goodwill_ratio）**（alpha191_138 等原 4 列已在后续面板重建中
  补齐，unlock_ratio_20d 已不在 ic 缓存）；raw 面板存在、coverage 0.57 但 panels_neu
  历史批次漏跑，`build_alla_factor_neutralized --only goodwill_ratio` 17 秒补齐
  （cov=0.508），ic 缓存 4 个 horizon 全部 0 缺失，`YURIQUANT_PANEL_CHECK=warn`
  长期容忍可消除。**已知副作用**：该 builder 是 jsonl 全量覆写设计，单因子运行后
  `factor_stats_neutralized.jsonl` 只剩 1 行（既有行为，待后续批次重建时自然恢复）。
  - 双链并行：legacy 09-22 已停，无并行对照——以 defv 连续产出 + 下游 daily_report
    正常为验收（10-11 前已连续运行）。
  - 时序纪律：批次 11（v1w2-cheap）复验**先过（62 因子环境 5/5 窗 +1.47pp）
  后撤（本机补齐 P5 面板后 72 因子环境仅 3/5 窗，2023+/2025+ 转负）——
  🔴 2026-10-17 生产切换搁置，维持 dingban**；✅ **10-21 批次 12① 终判**
  （本工作机当前数据态第三次复验）：仍 3/5 窗（+1.22/+0.50/−0.52/−0.36，
  全窗 +0.77pp）< 门槛 ≥4/5 ⇒ **v1w2-cheap 转 superseded、链路优化线终判
  关闭**（`baselines.yaml` status 已改；证据 `reports/cheap_lever_scan/`
  §四 + `revalidate_920_gm.csv`）。
- [x] **重跑 e2e 族报告对齐年化口径**（10-17，好机器批次 7 收官）：归档链
  `scripts/archive/{e2e_backtest,investment_report}.py --real` 以 252
  口径重出报告（`reports/e2e_backtest/` + `reports/investment_report/`，含
  README 换算注记：记录历史数年化 +1~2pp、Sharpe ×1.016）。**如实记录**：
  重出数（10.37%/11.91%）与记录数（16.2%/27.0%）差距主因环境漂移（09-07
  幸存者修复 −2pp + 数据端点 08-25→09-15 + 特征池演进），旧产物已删无法
  单放口径变更；归档链数字不再作现行证据（现行 = 全A 主实验）。
  附带：本机 venv scs 3.3.0 DLL 损坏（solver/e2e mock 测试连环崩）已换
  3.3.1 修复（uv.lock 未动，本机环境偏差）。
- [ ] **all_a_2018_2026 数据集纳入正式因子库监控**（目前为轻量 registry）。
  关联：`monitor_performance` 旧链复活的前提之一（另一前提 = runner 行情源参数化），
  现役生产监控已由 `monitor_production_ic` 承担，本条非阻塞。
- [x] **方案 B 缺面板对齐**（小工程，自 §五并入；**10-16 完成，见 §一 dingban 条目**）：
  原记录「ic 缓存 824 列中 4 列缺失」已过期——实测 4 个 horizon 各只缺
  goodwill_ratio 1 列，面板补齐后 `YURIQUANT_PANEL_CHECK=warn` 容忍可消除。
- [x] **链路优化线：执行层摩擦 + 全A 组合优化器（10-13 立，10-15 全线收口）**：
  真源 = `reports/docs/design/链路优化调查_预测组合执行_2026-10-13.md`（四部 17 节，
  含两次结论自我推翻的完整记录）+ RESEARCH_TODO §一 条目。终判五条：
  - **冲击成本模型落地**（`ImpactCostModel`，η·σ·√(Q/ADV)）：1e8 资金下 ≈ 固定 10bp，
    强 AUM 依赖（1e8→1e9 −4.6pp）；容量表 = 集中度 × AUM 二维
    （`reports/capacity_table/`，buffer 形态容量 >1e9）。
  - **QP 全家族关闭**（mvo/min_var/成本感知变体）：公平成本口径下打平 Top10% 等权、
    输生产 buffer 形态 −2.7pp；机制 = 持仓只数 × 平方根律（八成来自只数本身）。
    途中修复 3 个阻断级缺陷（引擎尘埃放大 gross=2.0 / mvo 量级失配 `alpha_scale` /
    协方差暖机截断 QP 空仓 59 日），全部带回归测试；AI97 的 QP 对照臂打 caveat。
  - **集中化全否 + buffer 确认最优**：production 口径锚点对拍，TopK45/100 ×
    equal/inv_vol/erc 最好 16.02 < 现行 16.46；研究口径 +1.9~2.3pp 被 T+1 open +
    换手成本吃光。aim 关闭（机制已被 buffer 带宽工程化）、HRP 书面关闭（重开条件 =
    CNE5）、前融合 / back_union 否（三窗口净≈0，`reports/multihorizon_fusion/`）。
  - **微调四维**（`reports/cheap_lever_scan/`）：frac×buffer / 调仓日 / 成交价 /
    免费信号臂全收口（现行即最优）；h 加权单调 ⇒ **两个胜者 buf(0.08/0.40) +0.78pp、
    h1w0.25 +0.31pp、叠加 +0.85pp（882 口径 5/5 窗口）→ ⚠️ 920 复验环境分歧
    （10-17）：62 因子环境 5/5 窗 +1.47pp 通过、补齐 P5 后 72 因子环境仅 3/5 窗
    （近两窗转负）⇒ **生产切换搁置，维持 dingban**；v1w2-cheap 已登记、
    `--preset v1w2-cheap` 已入库；✅ **10-21 批次 12① 终判否决**（3/5 窗，
  转superseded、线关闭，`reports/cheap_lever_scan/` §四）**。
  - **掩码口径拍板（10-15）**：报数 = 方案② `--mask-lag 1` + T+1 open（现值即此，
    零改动不重跑）；方案①（lag 0 + T 收盘）仅研究上限，数字不得并列。

### 已完成归档（一行一条）

- [x] **论文族 `pap_*`21 + 情绪族 `txt_*`5 · 生产口径收尾（10-13，P0）—— 结论 = 全线封底**：
  **① 口径修正**：26 个因子的面板**实体存在**（哈希名落 `<ds>/panels/<hash>.parquet`，
  `evals/` 亦有逐日 IC + 净值，registry 的 IC/IR/t 列已填）⇒ 旧记「registry 有 / 面板无
  ⇒ 从未评估」**不成立**；真缺口 = 未进 `panels_neu/`（949 = 975−21−5）⇒ 从未中性化、
  从未与量价池 DPP 竞争。**② `txt_*` 5 个全部不达标**：自身可交易 IR（逐日 IC mean/std，
  主模型同尺度 1.076）= 0.008 / 0.020 / 0.071 / 0.101 / 0.105 ≪ **0.3 门槛**（差 3~37 倍）；
  独立组合 Sharpe 0.486~0.568 **全部低于主模型 0.631**，独立超额 3.2~5.1pp < 8.2pp。
  **③ 融合层的「Sharpe 改善」是假象**：加 `txt` 后 TopFrac0.10 Sharpe 0.631→0.678~0.747，
  20/20 格子都改善；但组合数学不允许（`txt_report_score` IR 0.008 ⇒ 应 +0.00005，实测
  +0.116），且 **IC 最弱的因子改善最大**。**安慰剂检验**（把因子值在被覆盖列内逐行随机
  置换、零信息）**复现了中位 71%（1%~121%）**，残差中位仅 1.38σ、70% 格子在 ±2σ 内、
  且随权重正负跳变 ⇒ **增益来自「覆盖率掩码 / 市值暴露」**（研报覆盖 = 大盘股，任何带此
  掩码的融合都会减波动/降回撤），**不是 alpha**。**④ `pap_*` 无一条过双条件**：
  `pap_lowvol_1y`(neu Sharpe 0.758)/`pap_smallcap`(0.747) 虽超主模型但都带市值暴露且与主
  rank 相关 0.12~0.29；`pap_rev_5d` 虽过 IC-IR 0.330 却**组合层为负**（Sharpe 0.098、
  超额 −4.7pp）——**「IC 尺度骗人」的教科书案例**；`pap_breakout_atr` 残差 IC +0.1145(t=50.5)
  是**纸面**（原始可交易 IC −0.0130）。**⑤ 多重检验**：106 项组合测试 BH-FDR q=0.05
  **仅 1 条存活且为负**。与 `four_paths` §三既判「文本线 ≈0.19 ❌」同量级同结论。
  证据 `reports/text_factor_probe/`；脚本 `scripts/evaluation/{text_factor_probe,fuse_placebo}.py`。

- [x] **分析师一致预期修正线 · 数据可得性尽调（10-13）—— 结论 = 不可立项**：
  直调东财 `reportapi.eastmoney.com/report/list`（免费/无鉴权），拿到**逐篇研报的盈利预测**
  原始字段（`predictThisYearEps` / `predictNextYearEps` / `predictNextTwoYearEps` + 评级
  `emRatingName`/`ratingChange` + 目标价 `indvAimPriceT`），全市场 **148,338 条（2017 起，
  ≈15 分钟可回补）**。但 **`predict*` 字段的历史起点 = 2024-04（当年）/ 2025-04（次年）**
  （2017–2023 全空）⇒ 当年 EPS ~30 个月、次年 EPS ~18 个月；**更硬的是每个 FY 的预测窗口
  只有「当年」（FY2024/FY2025 各 9 个月，仅 FY2026 因 2025 研报的 next 跨到 14 个月）**
  ⇒ 无法构成「横截面 × 时间」的稳定 revision 因子，**不满足准入判据（IR ≥ 0.3，需 ≥36 月）**。
  属**东财侧数据生成时点限制**，换源/自建不可绕过。附带三点：① 🚨 **`notna=1.0` 是空串假象**
  （实际覆盖 EPS 0.68 / 评级 0.67 / **目标价 0.08** / `actualLastYearEps` 0）；② **akshare 的
  `stock_research_report_em` 封装口径错误**（把 `predictThisYearEps` 按查询年硬编码列名，
  历史行被贴错年度标签）⇒ **必须直调 API**；③ **顺手发现 `txt_*` 5 因子 registry 有/面板无
  ⇒ 从未评估**（研报线的真实剩余价值）。证据 `reports/consensus_availability/`。
- [x] **P1b 复验：P1 的「GRU 优势」被推翻（10-13）**：承接 10-10 P1 结论段自留的复验要求
  （「必须在定版全 A / 920 因子口径上复验」），把**生产基线**放进 P1 的同口径
  （池/折/标签/评估逐项复用，**只换输入**），得到三个否定结论 ——
  **① 基线弱**：P1 对照的 `gbdt_raw` = 0.0388 是全表最弱一档，而生产在用的定版 50 因子
  （`y2026__h5` DPP 选中）同口径 **IC 0.0682 / IR 10.07**；**sanity check 逐位复现**
  （`gbdt_a360` 与 P1 `gbdt_raw` 的 ic_mean/ic_ir/t_nw 三者完全相同）⇒ 对照有效。
  **② 同输入反转**：把 50 因子做成 T=10 时序窗口，`gbdt_prod_seq` **0.0679** vs
  `gru_prod_seq` **0.0554（−18%）** —— 而 P1 里是 +57% ⇒ 该结论是**输入特定**的，非架构普适。
  **③ 集成被组合层否决**：GRU 与生产基线几乎正交（rank 相关仅 **0.068**，同族 seed 间 0.58），
  集成 **IC 确实 +14%**（0.0682→0.0777），但 **Top10% 超额从 0.474% 降至 0.312%/0.242%**，
  **权重扫描 w_gru=0.1~0.5 每一档都下降**（0.1 档就 −0.064pp）⇒ 不是等权稀释。
  ⇒ **「新归纳偏置」这一路在生产口径 + 可交易检验下无落地价值**；P1 报告已回改
  （§〇 修正块 / §3.7 / caveat#2 标注死因）。证据 `reports/p1b_prod_baseline/`，
  脚本 `scripts/evaluation/prod_baseline_compare.py`。
  **方法论沉淀**：「隔离归纳偏置」的对照设计会让**基线失去代表性** —— 凡给「A 比 B 强」的结论，
  B 必须是生产真在用的那个东西；等价性验证只证明口径对齐，不证明基线有代表性。
- [x] **时序预测器对照（P1）+ 目标函数扫描（P2）收口（10-10）**：补掉 10-10 盘点出的模型层
  两处真空白。**⚠️ P1 部分已被 10-13 P1b 推翻（见上条），此处保留原始记录**。
  **P1 = 本轮调查第一个明确正结果**（`reports/p1_sequence_predictor/`）：
  同输入 alpha360（6 字段×60 滞后，逐日截面 z-score）下，**GRU OOS rank IC 0.0608~0.0632
  vs gbdt 同输入 0.0388（+57%~+63%）、vs gbdt_a158（158 工程化因子）0.0493（+24%~+28%）**，
  Transformer 0.0554（+43%）；GRU seed 散布仅 0.0024 ⇒ 远超噪声。**但机理经四层控制被三次
  推翻后收敛**：① `gru_shuf`（时间轴随机置换）0.0595 ≈ 有序 ⇒ **时序顺序不是来源**；② 大 MLP
  0.0307/0.0387 **不优于** gbdt ⇒ 不是「NN vs 树」；③ 小 MLP(h=12) 训练 MSE 0.0823 ≈ GRU
  0.0825 但 OOS 差 +0.021 ⇒ **正则/欠拟合假设被否**；④ `gbdt_raw_big`(1000 树/63 叶) 0.0420
  ⇒ 不是基线欠调。**⇒ 来源 = 「循环结构逐时间步权重共享」这一架构归纳偏置**（与顺序、正则、
  容量均无关）。**旁证（最有价值）**：GRU 与 gbdt 预测秩相关为**负**（gru vs gbdt_raw −0.138 /
  transformer vs gbdt_raw −0.650）⇒ 两族提取**近乎相反**的截面成分，秩平均集成把 IC 从
  0.0632 推到 **0.0671（+6%）**、gbdt 边际≈0。**caveat**：池 = HS300 并集（RAM 33.8GB 装不下
  全 A alpha360 张量 ≈20.7GB），月频 h=5、test 2023-08~2026-08 单窗口 ⇒ **须在定版全A/920 口径
  复验**方可谈落地（候选用法 = 混入信号层集成）。**P2 正式收口**（`reports/p2_objective/`）：
  10 臂（l2/l1/huber×2/quantile50/ranker + decay/volinv/abs/day 权重 + 标签 z-score）IC
  **全落在 ±1.3e-3**，**噪声地板 1.0e-3**（数学等价对 `l1_rank` vs `quant50_rank` 实测
  ΔIC=0.0010、预测秩相关 0.9898）⇒ 目标函数这一维确认无低垂果实；**明确有害**：huber(α=0.05)
  −4.8e-3、标签换 z-score −19.1e-3。脚本 `scripts/evaluation/{seq_predictor_compare,model_objective_scan}.py`。
- [x] **四条收益提升路径全部实测收口（10-10）**：承接「量价到顶」直接证据，把候选提升
  方向压成四条并**全部跑完**——**四条全空，且缺口因同一个机理**。
  **① 风险维度 ❌**（`reports/risk_budget/`）：复利归因（算术 17.50% → 几何 15.54%，
  拖累 1.96pp）；**波动率目标化 Sharpe 0.633→0.581（负）**、市值中性化 **−6.4pp/年**
  （09-18 四臂：市值只占 28%、量价风格 72%）、行业中性化 −0.06pp、回撤状态机更差；
  **唯一改善 Calmar 的是趋势 MA200（0.402→0.512，代价 −3.8pp 年化，未在定版复验）**。
  **② 新信息源 ❌**（`reports/pead_probe/`）：**推翻「PEAD/SUE 是空白」的前提**——
  审计发现 **14 个预告/快报/SUE 因子已在 `panels_neu`**（事件族 31 个 mean|IC| 0.0068
  = 全池最弱）；独立重建 6 面板（含池内没有的**事件漂移窗**口径）复现同强度，
  **可交易 IC ≈ 全样本 IC（0.011–0.017）⇒ 非纸面但弱**；**PEAD-only 模型合成后归零**
  （IC −0.0021）、组合层最好 +0.06pp、TopK45 集中持仓仅 3.54%/回撤 54.7%
  —— 与 news 族一字不差。**③ 模型层超参 ❌**（`reports/hp_sensitivity/`）：13 配置单维扫描，
  **所有小幅调整全落在 seed 噪声地板内**（7.6e-4），**所有「更大/更快」明确劣化**
  （n_est 150→600 −70e-4、leaves 15→31 −28e-4、lr 0.03→0.06 −39e-4）⇒ **过拟合主导，
  当前点即平台**，超参这一维正式收口。**④ 不做**（已负结果，勿重投）。
  **统一机理 + 准入判据（本次最有价值的沉淀）**：$S=\sqrt{S_1^2+S_2^2}$ ⇒
  **自身 IR 0.11 的正交信号最多值 +1.2%**；要把 IR 从 0.714 提到 0.80 需 **ρ=0 且
  自身 IR ≥ 0.36** 的新收益流。**全部非量价族合起来上界仅 IR 0.738（+3.4%）**
  ⇒ 所有历史失败**不是相关性问题，是绝对强度差一个数量级**。
  **⇒ 新数据源准入门槛 = 自身可交易 IR ≥ 0.3**（而非「低相关」）。
  汇总 `reports/four_paths/README.md`。
- [x] **量价因子边际增益曲线 + 组合层短板诊断（10-10，P0+P1）**：直接测量「量价候选池
  大小 K → 性能」这条曲线（修正 `ic_headroom` 扩池同时换选择机制的不公平对照），
  **首次给「量价到顶」以直接证据**（此前只有挖掘引擎饱和的间接推断）。
  **P0 阶段 A**（池质量，零额外 IO）——量价质量是**缓坡非悬崖**（1-25 名 Y 年 OOS IC
  0.0532 → 501-791 名 0.0203 = 38%）；**P0 阶段 B**（DPP 组合口径，7 年 2019-2025，
  1960s）——**K≥200 逐位完全相同**（`mixed_K200 = K350 = K500 = K791 = 0.0702`，且
  **每一年单独看都逐位相同**）、**拐点在 K≈36**（0.0700 = 平台值 99.7%）、
  **排序仅值 +0.0025**（vs `random_K200` 均值 0.0677）、**非量价贡献 ≈ 0**
  （`pure_pv_K200` 0.0701 ≈ `mixed_K200` 0.0702；K≥200 时 49-50/50 席位是量价）。
  **P1**：模型层低垂果实已摘完（**超参从未搜索**=唯一剩余，低预期）；优化层首次补测
  inv_vol/erc + 固定 K 缓冲带 → **TopK45 + inv_vol + buf1.5 = 18.07%/Sharpe 0.736**
  （vs 等权基线 17.44%/0.687，四项同向改善），只值 **+0.63pp = 微调非新增长极**。
  ⇒ **增量必须来自「新信息源」或「新归纳偏置」**。证据 `reports/pv_marginal/` +
  `reports/portfolio_weighting/`（口径 = 882 ortho 预测，绝对数字属 882、相对结论可迁移）。
- [x] **P0 新口径（920 面板）治本口径基线重跑（09-24~10-02 完成，三段式）**：
  v1（ortho920tl）出数即发现基本面族 99 张面板建于 backward_factor 全A修复之前
  （网格污染，§①-⑥ 降级为「量价+部分另类」口径）→ 99 面板清除重建 → v2
  （ortho920v2）完整口径定版：h1020⊕s0.5 17.87%/Sharpe 0.781 四项全面占优翻案
  → fundind 快信号 18.68%/0.813（§⑫）→ §⑭ 拍板 fundind+buffer 执行层 →
  §⑯ 网格上修 buffer(15/40)（18.07%/0.805/换手 19.1%）。
  全程 `reports/batch1_920_收尾对照表.md` §①–⑯。
- [x] 另类数据 P0 管道（09-20~21）：`data/altdata/` 传输层 7 源落地（巨潮快讯 119.2 万条
  自建历史存档 / 东财宏观日历 64,978 行含补洞 / 巨潮增减持 26.6 万行全历史等，
  见 `reports/docs/另类数据管道_数据源说明.md`）；日增量入口 `fetch_altdata_daily`
  进计划任务（18:00，`.venv` 解释器）
- [x] holder_dyn 9 因子三探针消融（09-21）：强制纳入臂 **Δ=−0.85pp 无增量价值**
  （`reports/holder_dyn/forced/`），Step 1b 全历史回补结案不推进；表保留（增量继续抓）
- [x] panels_neu 882→920 补齐（09-21）：38 缺面板补齐（含 ALT 28 + 基本面/股东 10，
  修复主实验 select 崩溃点），证据 `reports/panels_neu_backfill/` → 由此派生 P0 重跑项
- [x] 每日计划任务静默失败修复 + 调度收口（09-17/09-21）：三任务同日 `0xC000013A`
  被杀根因 = schtasks 裸建继承电池条件；新建 `config/schedule.yaml` 单一登记表 +
  `scripts/common/task_scheduler.py` 单点注册（XML 导入，显式关电池条件）+ `doctor` 体检
- [x] `YuriQuant Monitor` 旧监控链退休（09-21）：任务 `/delete`；数据集行情源 08-26 停更，
  每轮 144 critical / 235 warning 纯噪音；改指全A 实测不可行；生产监控由
  `monitor_production_ic` 承担（09-18 起）
- [x] HS300 时代选股清单入口归档（09-21）：`e2e_stock_picks` / `select_stocks` →
  `scripts/archive/`（数据口径停更、职责被 `alla_daily_rank` 取代）
- [x] 执行价模式恢复并接入主入口（09-21）：当日下线当晚自 HEAD 恢复（恢复无损：41 测试 +
  主形态 open/vwap 重算对齐留存 CSV ≤0.0001pp）；`run_model_portfolio.py` 新增 `--execution`
  {close,open,vwap}（默认 close = 历史行为逐位不变，T+1 臂 opt-in）；重建
  `_base/{open,vwap}_adj.parquet`（收益率层校验通过）；冒烟 open 8.03% / vwap 7.97% vs close 8.60%
  （2026 短样本，方向幅度与 09-15 全样本校准一致）。证据：
  `reports/docs/consistency_checks/口径核对_AI39_多因子10.md` §1.5「恢复记录」+
  6 份 metrics CSV（09-15 全样本：T+1 open 14.66 / VWAP 14.60 vs close 15.54）
- [x] 中性化向量化（09-20）：`batch` 2.96x 默认（逐位一致 max|Δ|=0）、`grouped` 8.3x opt-in
- [x] 出榜链口径对齐主实验 ortho（09-17）+ `limit_pos` 硬剔 + 申万一级行业排名进日报（09-18）
- [x] 生产化每日推理 `alla_daily_rank`（09-08，含计划任务注册）
- [x] 全A超额来源复核（09-08）：对全A等权纯选股 α=+5.1%/年，对上证超额约半数为风格敞口
- [x] 超额/夏普显著性检验进回测指标（09-08）；论文复现因子族入库（09-08）
- [x] 全A数据卫生修复（09-07）：退市股回补 238 只（幸存者偏差消除）、状态表重拉、
  ETF 列剔除、半拉日裁剪
- [x] 全A滚动训练实验（09-01）；重跑 multiyear_oos / freq_tune（09-01）：
  h1×M 唯一三年一致稳健解、h5×M 伪结果修正

---

## 二、研究功能缺口（新能力）

### 已就绪待接入 / 待全量

- [x] **大盘择时（市场状态 → 仓位乘数）可行性与收口（10-14 立，**10-15 修正重写**）**：
  项目此前只有截面选股一条腿，「加择时」= 组合层乘总仓位 ω_t（选股排序不动）。
  工程零成本（`Strategy.get_weights_at(date,...)` 天然拿到日期），**判定不立项**。
  探针 `scripts/evaluation/timing_probe.py`（秒级）→ `reports/timing_probe/README.md`。
  🚨 **初版核心结论「上界 9.03% < 满仓」是错的（foresight 方向写反：拿明天的答案判
  今天的仓），10-15 由莉酱追问时发现并修正；修正后收口结论不变，但理由更硬**：
  - **信息层（决定性）**：「昨日方向·次日执行」——**把「昨日方向」这一个 bit 用到
    极致的窄类上界**（该 bit 收盘后本就免费可知，臂=一条规则而非 oracle；勿当成
    「一切因果择时」的上界，三次修正 10-15）——年化 **14.96% ≈ 满仓 15.54%
    （−0.58pp）**。根因：**大盘日收益自相关仅 +0.048 ≈ 随机游走**，昨天的涨跌里
    几乎不包含明天的涨跌；更丰富信息集靠月度 0/7 + 资金面 0/4 + 规则 7/7 的
    **零结果咬合**支撑（咬合表见 README §六）。物理不可能的两个上界（开盘前知
    当日 294.8%、月初知当月 42.2%）说明**空间存在、钥匙不存在**。
  - **靶子可预测性**：等权全A 105 个月，7 个常用时序因子（动量 1/2/3/6/12 月、
    波动率、反转）rank IC 全部 |IC| ≤ 0.11、**0/7 显著**（最强 t=1.13）；
    月频方向为**负**（弱反转）⇒ 直接判死趋势追踪类规则。
  - **选股 vs 择时的本质区别**（为什么你的模型帮不上忙）：选股是横截面（每天
    ~5000 样本、IC 0.10 按日累计 t>8 显著、月频 Top10% 复利）；择时是时序
    （**每月只有 1 个观测**、105 个月 IC 不显著、单资产判断错一次=整仓暴露）。
  - **beta 结构（辅证）**：全期 beta 0.977、alpha 7.73% + beta 贡献 9.76%、
    **下行捕获 0.923 < 上行捕获 0.982** ⇒ 选股模型下跌日本来就抗跌，
    择时想干的活选股已干了一半；beta 漂移 +0.010/年 ⇒ 也无「动态 beta」次级机会。
  - **规则层（7 条因果规则实测**，MA20/60/120/200 · mom60/120 · vol 目标 15%，
    含 10bp 切换成本）：**7/7 Δ年化 < 0** ⇒ 全否。唯一双正的 MA20（ΔSharpe
    +0.138、MDD 38.8%→15.8%）是「长期半仓」（仓位 0.56、成本 −2.73%/年），
    减年化换回撤、不是 alpha（与 09-15 position_overlay_grid 同源互证）。
  - **重启触发条件**：① 新的**月度大盘预测信息源**（价格衍生类全测完），先过
    IR ≥ 0.3 + 上界测试；② 定版转 long-short/中性；③ 明确要低回撤版 ⇒ MA20
    半仓现成，按「−3.9pp 年化换 MDD 腰斩」记账。
  - **方法论沉淀**：仓位层 overlay 评估顺序 = **信息上界（可实现口径）→ 靶子
    可预测性 → 规则实测**，第一层能否掉就不进第二层。🔑 **结论反直觉时（完美
    择时反而亏损）第一反应是查自己，不是给市场找解释**——初版 bug 就是反例。
  - **口径陷阱 5 个**（已锁进 self_check + 6 条测试）：① foresight 方向反
    （初版主 bug）；② 拿 `bench_eqw` 日收益当价格比均线 = 275% 年化纯前视；
    ③ 上下行捕获写成 `mean(p|p≷0)/mean(p|m≷0)` 恒等于 1；④ 上界表按行号取值；
    ⑤ 月数打印误用因子数。
  - ⚠️ 被测臂 = **882 ortho 满仓无 buffer**（非登记基线 `v1w-ortho882`）⇒
    相对结论可迁移，写进定版前须 `--equity` 换 920 重跑。
- [x] **模型自诊断仓位（信号自身状态 → 仓位乘数）探针与收口（10-23 立并完成）**：
  市场侧择时（10-15）与 regime（10-15）收口后，剩下唯一未被证伪的仓位信息集是
  **模型自身状态**——「这一期我的信号有多可信」而非「大盘会涨会跌」。探针
  `scripts/evaluation/model_state_probe.py` → `reports/model_state_probe/README.md`。
  5 个状态量（滚动 IC60 / ICIR60 / 集成两臂秩相关分歧 / 预测强度=头部均值−全均值 /
  预测离散度），基线 = **登记 `v1w-ortho882`（production/open 现跑，16.39%/0.681 锚点逐位复现）**。
  - **预测力 0/5**：五状态 → **下月组合收益** 的月度 rank IC |t| ≤ 1.06（102–105 个月），
    五分位全不单调 —— 与市场侧 0/7 同一个死穴：月频时序只有 1 观测/月。
  - **上界**：完美预知下月 **IC 符号** = **与满仓逐位相同**（avg_pos 0.995）——
    因为 **IC 符号 105/105 个月全正**，「等信号失效再降仓」天然**没有触发空间**。
  - **规则层 11 臂（4 先验方向 + 4 事后反转诊断 + 3 杠杆形态）无一通过**：先验方向最好的
    「分歧<中位→满仓」Δ年化 +0.13pp / **2-5 窗口** / 安慰剂 p=0.092；
    「强度>中位→满仓」ΔSharpe +0.122、4/5 窗口，但 **Δ年化 −2.79pp**（长期降仓）。
  - **同平均仓位分解（本轮新增口径，关键）**：分歧门相对同仓位（0.876）+1.93pp、
    强度门（0.719）+1.40pp ⇒ **状态里确有"哪个月更好"的信息，但信息价值
    付不起长期降仓的 beta 代价（1.8~4.2pp）**；强度门额外多砍 MDD 4.7pp
    （−21.1% vs 同仓位 −25.8%），分歧门 MDD 反而比同仓位更深（−34.3% vs −30.7%）。
  - **「不降仓能不能择时」也已测完（10-23 追加，`--spread` 模式）**：
    ① **动态集中度**（满仓只改 K，预测目标换成「下月 Top30 − Top80 收益差」）——
    5 状态 0/5 过门、最强 |t| = 1.15；且 K30 月均 +1.19% vs K80 +1.16%，
    **spread 仅 +0.04%/月、胜率 45.7%**（无缓冲口径）⇒ 集中度收益差本身近乎为零，
    `portfolio_weighting` 的 K45 优势主要来自 buffer 而非"更集中"。
    ② **杠杆形态**（ω∈{1.5,0.5}，平均仓位 0.938）：强度门含 4.5% 融资 **+0.85pp / 2-5 窗口 /
    p=0.068**（融资归零的上界才 +2.01pp / 4-5）；分歧门杠杆 +5.39pp 但 Sharpe +0.011、
    MDD −47.7% ⇒ **赚的是平均仓位 1.25 的 beta**。两条「不降仓」路径均不达标。
  - **判定：收口，不进引擎**。重启条件：① 新状态量（预测分布熵 / 个股级不确定度 /
    跨模型族分歧）须先过月频 |t| ≥ 2.5；② 转 long-short / 市场中性。
    （原「杠杆形态」条件已测完并关闭。）
  - **口径 bug（已锁进 8 条测试）**：`omega_daily` 首版用 `shift(1)+ffill` 把月末 ω
    映射到当月，实际**整体滞后一个月**；修正前强度门安慰剂 p=0.002（假阳性），
    修正后 p=0.080。**一个月的错位足以把噪声变成"显著"。**
  - ⚠️ 882 口径；920 定版（`dingban_fundind`）本机缺 `defv_sig__h1.parquet`，未复验。
- [x] **大盘方向准入检验（新信息集，10-26 立并完成）**：
  `timing_probe`（价格/资金面/日历）、`regime_probe`、`model_state_probe` 三条线收口后，
  换**不重叠的信息集**再测一次：`scripts/evaluation/market_direction_probe.py` →
  `reports/market_direction_probe/README.md`。目标 = 下月沪深300（**次月首日收盘起算**，
  不用月末收盘成交），状态全 PIT，判据先验固定（|t|≥2.5 + 五分位单调 + `oos_t>0`）。
  - **本机可取的 3 个状态 0/3 过门**：`realvol60`（60 日已实现波动）IC −0.070 / t −0.71；
    `stock_bond`（300 − 国债ETF 20 日动量差）IC +0.055 / t +0.57；
    `mom20` 对照组 IC +0.062 / t +0.64 —— **对照组复现了 timing_probe 的已知结论**，
    证明这个"零"不是取数/对齐 bug。
  - **`stock_bond` 的陷阱已记录**：样本外命中率 59.6% 看似可用，但 `oos_t = −0.48`
    （小跌月猜对、大涨月猜错）⇒ **命中率不能当信号，必须看符号×幅度的 edge**。
  - **未测（数据不可得，非被证伪）**：期权隐含 QVIX / VRP（optbbs SSL 间歇）、
    宏观 PMI/社融/M1-M2/10Y/SHIBOR（eastmoney 整段 ReadTimeout）。数据可得性表在 README §二；
    拿到稳定离线缓存后，同一脚本补 4 个候选即可，判据不变。
  - 工程坑：akshare 部分接口无超时挂死，`ThreadPoolExecutor` 兜底仍会因非 daemon 线程
    导致进程退不出去；已改 **daemon 线程 + queue**。
  - **[10-26 追加] 个人投资口径（`--valuation`）：换长周期估值，结论相反——有信息**。
    PE-TTM **滚动 10 年分位**（PIT）→ 未来 1/3 年收益的极值条件检验
    （legulegu 月度 PE：沪深300 2005-04+ / 全A 2005-01+，已缓存 `data/pe_*.csv`）：
    沪深300 便宜 20% 分位后 1 年均值 **+15.4%**、贵 20% 分位后 **−7.3%**；3 年 **+39.0% / −28.6%**。
    **但去重叠后独立"贵"样本只有 0~1 个**（即 2007/2015/2021 三次泡沫、2008/2018/2024 三次底部）
    ⇒ **这是极值纪律，不是预测**。等本金定投对照（200 个月、同等总投入）：
    恒定 1.21x / 1.15% vs 估值加权（便宜 2x、贵 0.5x）1.31x / **1.64%**（+0.49pp/年）。
  - 口径坑：① 扩张分位在本序列最高只到 0.75（2008 低 PE 永久压低后续分位）⇒ "贵"桶为空，
    已改滚动 10 年分位；② 定投必须**归一化到等本金**，否则"便宜加倍"只是多投 20% 钱。
- [x] **市场状态特征（mf）采纳判定（10-26 复核，批次 12② 收尾）**：
  证据 `reports/sync/batch1213_products_20261022.zip::mf/`（本机无 pred 副本）。
  **判定：方向为正，但不直接切生产**，差一条干净单臂。
  - **达线项**：同态配对 buffer(15/40) **17.74%/0.769/换手 21.0% vs 锚 16.31%/0.738/20.6%**
    = +1.44pp、Sharpe +0.032、MDD −0.9pp；5/5 窗年化为正（+0.7~+2.3pp）、4/5 Sharpe 正；
    单臂、无挑选偏差，过「>±1pp 才值得并入」线；单模型 h1 层 −0.04pp ⇒ 增量来自长 horizon。
  - **不能直接切的两个硬问题**：① **对照不是单变量** —— mf 臂命令是 `--preproc ortho`
    （`ortho920tl 同配置 + --market-features`），锚是定版 `--preproc ortho_fundind`，
    +1.44pp 混着「去掉 fundind」一项（该项历史 +0.81pp）；**真正要上线的
    `ortho_fundind + mf` 从未跑过**。② mf README「生产出榜链无需动」**是错的**：
    生产链 `alla_daily_rank.py` 每日自 `compute_features()` + `fit()`，原本零 mf 接线。
  - **10-26 已做（接线，默认关、零回归）**：`alla_daily_rank.py` 加 `--market-features`
    （注入 18 个 `mkt_*` 面板、须显式配 `--out-tag`、`history.csv` 记
    `n_market_features` / `market_feature_last_date`），锁 1 条 CLI 守卫测试；
    指数缓存本机实测 18 面板、逐行同值广播可用（000001_SH/399317_SZ 均 2015+）。
  - **❌ 批次 14 已收官（10-26，本机）**：干净单臂 `ortho920fundind_mf` defv
    = 16.80%/0.728 vs 锚（同数据态复测逐位复现）16.31%/0.738——**全窗 Δ
    +0.49pp < +1pp 门槛且 Sharpe −0.010**（分窗 ΔSharpe 仅 2025+ 为正）。
    **判定：不切 `v1-mf`，市场状态特征线收口**；+1.44pp 拆解 = 去 fundind
    ≈ +0.8pp + 干净 mf 增量 ≈ +0.5pp。接线保留（默认关）。证据
    `reports/mf_fundind_check/`（window_check.csv + buffer_arm.csv）。
  - **新增生产依赖**：`update_data` 只维护个股池，**指数日线无日更任务**——采用 mf
    前必须补指数缓存更新，否则预测日整行 NaN（代码已告警留痕）。
  - 幅度折扣：Sharpe 仅 +0.032（年化 +1.44pp 含波动 22.1%→23.1%）、2023+ 窗 ΔSharpe
    −0.026、5 窗互相嵌套≈1 个样本、无存档权益序列算不出配对 t；v1w2 有 5/5→3/5 先例。
- [x] **实验产物接 PBO 出报告**（09-23 固化 `scripts/evaluation/pbo_report.py`）：
  rolling_grid 三网格已出（`reports/pbo_rolling_ortho/`：M_raw 0.174 / M_neut 0.227 /
  W_raw 0.000，均远低于 0.5 红线）；后续各线已全部接齐——AI97 0.704（负结果主证）、
  GFlowNet 114 因子族 0.114（稳健）、LLM-MCTS 14 因子族 0.039（稳健）。
- [x] **labels.py IR/Calmar 标签分支**（09-23，AI29 转译；`build_labels(method=)`，
  超额口径需 `bench_close_panel`）。**实验已跑完判负（10-04）**：ir 臂 IC 全线
  −52~−191×10⁻⁴、回撤中位 +1.27pp 恶化（AI29 自述代价 + Label Alchemy vol_scaled
  预警双应验）；calmar 同族分母更病态不再跑。三臂总结论（gauss_rank 零增量 /
  co 隔夜判负 / ir 判负）见 RESEARCH_TODO §一 标签臂记录。
- [x] **分钟频挖掘第三层扩原料（时段切片矩阵化）**（10-14~15 全链收口）：36 个
  `*_segNN_*`（20 分钟窗 × 动量/RV份额/量份额）入 `factor/intraday_features.py` +
  hs300_2022_2025 库（测试 15 passed；checkup `reports/intraday_stat_checkup.csv` 78 行
  合并版）。首轮：20/36 显著、**时段量/波动份额是真新维度**（与库内最高相关 ≤0.66、
  `pv_seg10/11_volshare` 残差 t 3.5/4.0 存活；动量维只是把已有信号切细，`mom_seg01` 与
  `open30` 相关 0.896）。**但 GP 组合层未抬升**：`mine_im_combos` pop300gen15×31 终端
  （同 seed 同预算复跑），HOF 6→2 条且两条仍是 `im_mom_last_bar` 同构、最好 OOS IC
  0.0523 < 基线 0.060 ⇒ **份额族独立 IC（t 4~5）进不了组合层头部**——「公式重组合
  无法把弱原料变强」的又一次确认。剩余子项（量价 lead-lag / tsfresh 移植 / 事件日
  条件化）与「时段份额族进模型层（gbdt 前融合）」—— **模型层亦负（10-15 裁决）**：
  hs300 日频 e2e（horizon=1/top50/max_features=30）三臂 no_im 6.59%/0.421 vs with_im42
  9.97%/0.583 vs **with_im78 8.26%/0.471 ⇒ 新 36 特征 −1.71pp/−0.112**（seg12 与
  last_bar 0.657 冗余占席 + 弱终端扰动特征选择；⚠️ 10-14 曾误引月频版数字当日频，
  已纠正）。**时段切片矩阵化全链（单因子✓/GP❌/模型层❌）收口**，证据
  `reports/e2e_p0_intraday/daily/with_im78/` + `logs/e2e_p0_intraday_with_im78_skiprp.log`
  （risk_parity SCS 段错误走 --skip-rp，等权列不受影响）。剩余 = 全 A 扩容吞吐探针
  （押后）与逐段新原料（须先过 0.3 可交易 IR）。
- [ ] **分钟频扩容（资源墙）**：all_a 池物化（~5500 码 × 48 bar，MemMap 分块性能待验）、
  1 分钟档位（存储 ×5）；先跑吞吐探针（`scripts/oneoff/_probe_minute_throughput.py`）估成本。
- [x] **量自相关族 + 量价方向拆分（10-23，文章《高频数据低频化：为什么量比价更重要》P0）**：
  `factor/intraday_features.py` 新增 5 个（`pv_volshare_autocorr_l1` / `_acf_disp` /
  `_ar1_resid_acf` + `pv_ret_vol_corr_up` / `_dn`，非时段 42→47），测试 15 passed，
  hs300_2022_2025 全量重跑（83 行）。**结论**：4/5 过 |t_nw|>2（IC −0.008~−0.025），
  `dup_corr1` 全 ≤0.36 ⇒ **确是新驱动源（正交性成立）**；但 10-23 复核（§2b.8）发现
  文章的"量≫价 26 倍"**在 lag1 不成立**（−0.025160 vs −0.025133）、仅 lag≥2 成立（10.4 倍）；
  ① 方向与文章相反（本地 h1/h5/h20 **全负**=拥挤反转，horizon 不能解释，见 §2b.8；
  口径拆解见 §2b.9）；
  ② 文章第一名 AR(1) 残差构造本地最弱（t −2.92）⇒ **构造强度不可跨口径移植**；
  ③ 除 `corr_up`（sharpe +0.119）外 best_sharpe 全负 ⇒ 单因子多头不可用，只能进组合层。P2 半衰期标定已完成（见下条）；
  再决定 P1（极值邻域状态/分域显著性/事件幅度族）。证据 `reports/intraday_stat_checkup.csv`
  + `logs/intraday_stat_checkup_20261023.log`；完整判读 `reports/docs/research_notes/
  短板匹配与微信文章研读_2026-09-22.md` §2b。
- [x] **im_* 半衰期标定（10-23，同文 P2，已完成）**：`scripts/oneoff/_probe_intraday_half_life.py`
  → `reports/intraday_half_life_probe/half_life.csv`（83 个 im_* × h1/5/10/20 秩 IC）。
  **结论**：67/83 可算倍数、中位 1.02、同号 82%；**分族寿命 = mom 0.10 / candle 0.49 /
  pv 1.02 / ret 1.10 / shape 1.37 / vol 1.41**。① **h1 最强的五个（last_bar/seg12/close30/
  vwap_dev/pm_ret）倍数仅 0.08~0.46** ⇒ 日内动量是"次日反转"型短寿命信号，禁止长窗平滑、
  IC −0.06 不可当可持续 alpha；② **量族被 h1 口径系统性低估**（volshare_autocorr_l1
  1.80×→h20 IC −0.045 进全库前 5、acf_disp 2.00×、corr_dn 1.88×）⇒ 修正上条"IC 弱"判断；
  ③ `ar1_resid_acf`（文章第一名）h20 转正、`candle_high_in_am` 方向翻转 ⇒ 建议退休；
  ④ with_im78 −1.71pp 的解释补充：78 特征混着寿命 0.1 与 2.0 两族，同一 h1 标签下
  **按寿命分组入模**才对。下一步 P3（未做）：量族 20 日平滑版 + h20 标签 e2e 对照 /
  动量族标 h1-only / 下线两个噪声因子。
- [x] **P1 三族落地（10-23，同文 §2b.7）**：新增 6 个特征（`path_*` 2 / `dom_vol_rv_sig` 1 /
  `evt_*` 3，非时段 47→53），测试 15 passed，checkup 89 行重跑（66/89 显著、16 个 dup>0.7）。
  **结论**：① **路径族不成立**（两特征 t −1.16 / +0.55）——5 分钟 bar 下 ±20 分钟窗=±4 bars，
  局部性被抹平，须等 1 分钟档位或缩到 ±1 bar；② **事件深度成立**（`evt_max_reversal_depth`
  t −8.70 / `evt_down_tail_share` t +7.35，寿命 1.60），但 `evt_drawdown_depth` 与
  `alpha360_HIGH0` 相关 0.797 = 冗余；③ **分域显著性成立但无增量**（t −7.02，寿命 1.34，
  但 `dup_top1`=`gp_im_d150c8f4` ⇒ 自家 GP 已挖到同类构造，只宜作对照基线）。
  ⇒ 可用 3 个（1 短窗 + 2 长窗），文章八类在 5 分钟口径下**只剩 3 类成立**。证据
  `reports/intraday_half_life_probe/half_life.csv`；判读 `research_notes/...§2b.7`。
- [x] **「与文章不一致」口径拆解（10-23，§2b.9，只做诊断不改因子）**：11 维变体逐一测试
  （池内成交额三分位 / bar 粒度 5→60min / 剔首 bar / 去共同 U 形 / 剔涨跌停 / 流动性过滤 /
  20 日平滑 / past5-past20 中性化 / 反转载荷 / AR(1) 残差版）。
  **结论**：① **强度差异可解释**——量占比自相关 20 日平滑后 |IC| 0.028→**0.050**（h5），
  与文章首位 0.0586 同量级，说明"我们弱一倍"主要是未平滑的噪声折损；② **符号差异不可解释**
  ——所有可测口径下均为负（含 2022-2026 逐年、三个规模层 −0.019/−0.018/−0.018），
  文章称同期也年年为正 ⇒ 非样本期；剩余仅 **全A 池**（无分钟数据）、**1 分钟微观结构**
  （无该档位）、**文章未公开构造**三项，本地无法验证；③ 文章冠军 AR(1) 残差版本地 ≈0
  （h5 −0.0006、平滑 +0.0049）= **无信号而非反号**，单条构造强弱不可跨口径移植。
  ⇒ 使用约束：按本地符号入模；要文章量级须加 20 日平滑；不引用文章数字做排序依据。
- [x] **分钟频 P3-① / ②（10-23 完成，实测见研读笔记 §2b.12）**：① **周/月度换仓重测**——
  毛收益不随换仓变慢而单调改善（周频最差），净收益改善（日 −78%~−88% → 月 −4%~−19%）
  全部来自调仓次数减少（1085→229→54）；**每次调仓换手不降**（日 160% vs 月 163%，月频
  相邻 Top-30 重叠仅 12~14%），因为分钟频因子值 rank 自相关只有 0.02~0.22（20 日后仍
  0.02~0.18）⇒ “量族 IC 寿命长”是预测力衰减、不是持仓稳定，**降频只是省成本不是 alpha**。
  ② **20 日平滑**——rank_ac20 提升 3~4 倍、月频多头换手砍半；`pv_volshare_autocorr_l1`
  （净年化 +1.2%→**+6.3%**，sharpe +0.08→+0.48）与 `shape_autocorr_l1`（+4.0%→**+6.2%**，
  +0.25→+0.44）显著改善，但 `ar1_resid_acf` / `ret_vol_corr_up` 平滑后更差 ⇒ 平滑只救
  “本身有信号且噪声大”的因子。**e2e 已完成**（89 个 `im_*`、h=5、forward 5 折）：平滑
  **不提升 IC**（GBDT 0.0060→0.0067 均不显著；Ridge 0.026→0.013 反而腰斩），但周频多头
  换手 40%→23%、Ridge 平滑后逐年净收益 3/4 年为正且波动小 ⇒ **平滑只作组合层预处理，
  不进 IC 筛选**。**不做**：单因子层面继续调换仓频率（应留给组合层）、把平滑当 IC 提升手段。
- [x] **分钟频 P3-③ 条件分布因子（10-23 完成，见研读笔记 §2b.13）**：新增 `cond_ret_hi/lo_mean`
  + `cond_ret_hi_lo_diff` 3 个（`im_`，共 56 个非时段特征），2/3 显著（IC −0.0127，t −2.68）
  但 **hi 与 hi_lo_diff ρ0.997、与 `vol_rsj` ρ0.83 / `ret_mean` ρ0.81 高度重叠**；唯一正交的
  `cond_ret_lo_mean` IC≈0（t −0.34）⇒ **无新增独立驱动源**。`hi_lo_diff` 已标 retired，其余
  保留 experimental 且不进候选池。结论：方法论成立，但 5 分钟粒度退化为 RSJ 变体。
- [x] **分钟频 P3-④ 量价转移熵（10-23 完成，见研读笔记 §2b.14）**：新增 `pv_te_vol_ret`
  （量→收益）+ `pv_te_ret_vol` 2 个（`im_`，共 58 个非时段特征）。**是本轮唯一独立新驱动源**：
  IC −0.0197 / **NW t −6.56** / ICIR −3.06，**全库 918 因子无一 |ρ|>0.3**（最高 0.284
  `shape_longest_neg_run`，对照组 P3-③ 与 `vol_rsj` ρ0.83）；**IC 不衰减反增强**
  （h1 −0.0197 → h20 −0.0299）⇒ 属"状态型"慢变量，不像 P0 量族那样是事件型；反向 TE
  （收益→量）弱一倍（t −3.90、IC −0.0085）⇒ **传导方向不对称，"量领先价"本地可测**。
  载体只有月频多头（+5.2%、sharpe +0.34；周频 −9.7% 再证 §2b.12 的 W 频最差），L/S 仍负；
  20 日平滑后换手 89%→23%、净 +3.7%（sh +0.28）。处置：2 个均 `experimental`。
  **组合层增量已验（10-23，见 §2b.16）**：单列加回 92 个 `im_*` 基线跑 e2e，3 种子
  ΔIC **−0.0000**（−0.0006~+0.0003）、Δ月频年化 +0.64pp 跨零 ⇒ **零增量，不进主模型**。
  **诚实声明**：三分位 27 盒 × 45 样本/日偏少，`n_bins` 敏感性与 lag>1 未测。
- [x] **P3 组合层增量检验（10-23，见 §2b.16）**：`pv_te_vol_ret` 与 `evt_down_tail_share`
  单列加回 92 列基线 → ① TE **ΔIC −0.0000（3 种子）**、Δ年化 +0.64pp 跨零 ⇒ 零增量；
  ② `evt_down_tail_share` ΔIC **3/3 稳定 −0.0045** 但 Δ月频净年化 **3/3 稳定 +1.85pp**
  （回撤反而 +1.4pp 更深、逐年 IC 2022/2026 变差）⇒ "IC 降收益升"冲突，不能由截面预测力
  解释，不采纳，列入多窗口+分池复核；③ 安慰剂（`path_extreme_rv_diff` 作第 93 列）与基线
  **逐位相同**（LGBM 忽略无信息列）⇒ +evt 不是"多一列"的路径扰动；④ **口径警告**：本轮
  基线 IC 0.0244 vs §2b.12 的 0.0060 差 4 倍（特征覆盖过滤/列集不同）⇒ **跨轮 e2e 绝对数
  不可横向引用**，只能同脚本内做臂间比较；后续动 e2e 前先固定特征快照 + 多种子协议。
- [x] **分钟频 P3-⑤ 非同步交易（10-23 完成，见研读笔记 §2b.15）**：**零成交维度不可用**
  ——`min5_hs300` 全期 32.3 万个 (日,码) 单元里 **87.8% 的零成交 bar 数恒为 0**、全期均值
  0.012%、每日截面 90%−10% 分位差 0.0000（2022~2026 逐年都在 10⁻⁴ 量级）⇒ Q 类统计量在
  5 分钟 bar 上没有载体，该支直接关闭。退而落地 **`pv_zero_ret_share`（零收益 bar 占比，
  价格粘滞度）**：IC +0.0443 / NW t **6.77** / ICIR 3.16 / best_sharpe +0.048 @lo_M，但
  **全库 80 个因子 \|ρ\|>0.3（最高 `ret_up_ratio` −0.733、`alpha191_187` −0.659）**，
  对 `ret_up_ratio`+`alpha191_187`+`vol_rv` 三元回归（平均 R² 0.687）后 **h1 残差归零**
  （+0.0036 / t 0.90），仅 h20 留 +0.0218（低波/不活跃度慢频倾斜）⇒ 是既有族的机械互补
  （涨+跌+平=1），**保留 experimental 作不活跃度诊断，不进候选池**。
- [x] **分钟频 P3 收口（10-23）**：最终排序——④ 转移熵（量→收益）是唯一独立新驱动源；
  ①② 是组合层工程（降频/平滑，不进 IC 筛选）；③⑤ 是既有族重新参数化。**不做**：
  同伴相关性、高频风险模型、成交量时钟、崩盘概率泊松（理由见 §2b.11 表）。
- [x] **另类数据因子轮次（altf 首轮，10-07 完成）**：`scripts/builders/build_alla_altfactors.py`
  三族 20 因子入 all_a_2018_2026（panels + factor_stats_altf.jsonl + registry 922→942 +
  ic_h{1,5,10,20} 融合）——增减持 7（公告日对齐 lag0，金额比流通市值）/ 宏观日历 7
  （15:00 收盘闸门 + 方向表，广播族截面 IC 按构造 NaN，1 个 ×波动率交互变体可测）/
  快讯情绪 6（cls 词典代理情绪 + 条数/加红计数；**|IC| 全场最高**：attention 反转
  h1 −0.021 → h20 −0.043）。坑与备注：① 09-23 缓存重建后 cls/宏观存档丢失，本次
  经通道回补（宏观 65,419 行全量；cls 回补至 2022-09 共 63 万条后停，快讯族覆盖
  2022-09~2026-09，每日 17:30 续传任务会继续推早水位，重跑 builder 即扩覆盖）；
  ② `altf` 尚未登记 `SET_TO_FAMILY`/`SOURCE_PREFIXES`（本轮禁改该文件），registry
  family=其他 + subfamily 区分三族，下次动 `research/factor_library.py` 时补。
  **完整判读（10-08 结案）**：席位竞争 36 轮 0 入池；慢信号家族并入（--include-alt-news）
  定版 18.68%→16.27% 双路径双败——**alt 三族对主模型无增量**（news 族 h20 反转方向
  与 macro 择时特征留待未来新信号线），证据 `reports/batch2385_结果速览.md` P-C 节。
- [x] **llm_pool 机制抽取（QuantaAlpha 转译，09-23 完成）**：① AST 结构去重
  （`structure_similarity` 子树 Jaccard ≥0.6 判同族变体——去窗口、终端占位、交换律排序，
  `_update` 求值前拦截省评估预算，`struct_dedup=False` 可关）；② 复杂度三维约束两项新校验
  `param_heavy`（自由参数占比 ≥50%，当前空间上保险性质）+ `too_many_features`
  （底层特征 >`MAX_BASE_FEATURES=6`，固定常数不随字段表扩容放松）；③ 语义一致性前置校验
  `llm_semantic_check`（LLM-judged，opt-in 依赖注入设计，解析失败=未通过不静默放行）。
  `tests/test_ai97_llm_pool.py` 73→118 用例全绿；机制④归 MCTS Phase 0、机制⑤备选未做。
- [x] **批次 8 表格基础模型对照 runner（09-24 完成；全量已收官见 §五批次 8 条）**：
  `scripts/evaluation/tabpfn_rolling_compare.py`：逐月重训 × 月内逐日预测
  （test_step=1，v1 抽样高估项直接修掉）× W∈{2m,3m} 窗口扫描；四臂
  （gbdt / tabpfn-3.5 / tabicl-2.2 / 秩平均，走 PREDICTORS 注册零分叉）；
  评估 = 可交易 IC（T+1 掩码）+ NW t + Top10% 月频 **open 正名口径**组合 +
  两臂逐日秩相关诊断；`--device cuda` 好机器全量入口。本机 hs300 gbdt 臂
  冒烟通过（tradableIC 0.0255/0.0271、NWt 3.0/3.2，W=3m>2m 与 08-08 方向
  一致）。**顺手修窗口语义 bug**：`MonthBegin(W)` 从月内日期回退只覆盖
  W−1 个月（2m 窗实测仅 17~20 日），改 `DateOffset(months=W)`。
- [x] **东吴 LLM-MCTS Phase 0 代码落地（09-24 完成）→ 全量已收官（10-08/09）**：
  `factor/mcts/` 五模块（seeds：29 Seed = Alpha158 rolling 全类 × w=20，
  **求值走原生 callable 零转译**；reward：周度六项 + 防前视切片评测器；
  tree：UCT + virtual expansion；proposer：MCTS 上下文 prompt + LLM 扩展器
  + 离线变异兜底；engine：主循环 + 三层去重）+ `scripts/factors/run_llm_mcts.py`
  四臂 runner（mcts/llm_oneshot/gp/gflownet，双口径验收 + PBO/DSR 自动出数）。
  吸收 RD-Agent 机制①（周度 IC ≥0.99 数值去重）④（失败换向）⑦（JSON 纪律）。
  **all-A 正式跑**（all_a_2019_2026，IS 2019-2023/OOS 2024-2026-09，h=5，
  3 seed × 50 iters × 5 variants）：750 候选收编 14、**PBO=0.039 族层稳健**、
  OOS≥IS 泛化一致；`build_alla_llmmcts_factors.py` 入库 llm_* 后 v4 席位
  **4/14 入池（唯一产出超门槛因子的挖掘引擎）**，但组合层 +0.16pp 噪声级
  **不采纳**——「席位突破、组合无增量」归档（量价族信息在 DPP 下与既有 alpha
  高度重叠），证据 `reports/batch2385_结果速览.md` P-A 节 + 终审节。
- [x] **多模型×多标签预测合成对照（09-23 完成）**：12 成员（gbdt/ranker/ridge ×
  h1/h5/h10/h20）等权逐级 + walk-forward 学权（`scripts/evaluation/member_blend12.py`，
  与 stack_blend 同参）。结论：**增量在 horizon（标签）维度不在模型族维度**
  （eq_h20_3 16.00% vs eq_h1_3 12.27%；同 horizon 混族反而稀释）；全 12 等权
  15.12% < h1020 16.89%（−1.77pp）；学权 stack_all12 15.53% ≈ 生产 ens_h1h5
  15.52%，对 12 成员学权仍无增量。**h1020 已在 920 口径裁决采纳**（horizon_mix_920：
  16.80% vs baseline 13.49%，+3.31pp——后经 §⑩-⑪ 成为定版信号成分；12 成员
  全量复验随之不再必要）。证据 `reports/member_blend12/`、`reports/horizon_mix_920/`。

### 生产化 / 长期

- [ ] **生产级执行**：实盘下单对接、实时行情驱动（当前日频盘后信号，
  "信号→次日执行"滑点假设未经真实成交验证）。
- [ ] **文本挖掘合规化**：同花顺/巨潮爬虫依赖（页面变更即断），生产化需
  商业数据源或明确"研究性补充"定位。
- [ ] **在线 dashboard**：报告均为静态 HTML，无增量刷新监控页面。
- [ ] **监控链收敛（非阻塞余项）**：`monitoring/` 包与 `monitor_performance.py` 留在仓库
  供手动跑；复活前提 = ① 全A 库 evals 补齐 ② runner 行情源参数化。
- [x] **HS300 时代入口三口同裁归档（09-24 完成）**：`e2e_backtest` /
  `optimize_e2e` / `investment_report` → `scripts/archive/`（证据与已归档者
  相同：数据口径停更、产物已清理）；同步退休 `test_metrics` perf_stats
  一致性用例与 `test_layering` 旧路径钉子；mock 链路测试按 09-21
  e2e_stock_picks 先例改指 archive 保活；归档件/oneoff/README 引用全部改指。
  **顺手清欠**：`test_layering` 私有名守卫在 master 上的存量红灯（altdata/
  builders/p5/galaxy/rolling_grid 白盒用例 + mcts `_fmt_const`）——业务侧
  公开化（`fmt_const`/`std_code6`/`pit`/`safe`），测试侧白名单精确登记，
  守卫恢复全绿。
- [x] **调度余项（09-24 完成）**：`.workbuddy/memory/automations/71e2d7a5`
  退役记忆文件已删（现役 `3b31c720` 日报自动化保留不动）。

### 已完成归档（2026-09-18~20 交付批次）

- [x] **RL 组合优化线（stage2 三阶段管线）**：`factor/rl/portfolio_env.py`
  （银河 0706 口径主动权重空间 env + tradable_masks，22 用例）→
  `run_portfolio_ppo.py`（Phase 1 hs300 平价验证四臂，行为形态与银河 HS300 结论一致）→
  `run_portfolio_phase2.py`（Phase 2 zz1000 主实验 runner：三标签 GBDT 信号管线 +
  PPO 滚动集成 + 银河口径 QP + 消融钩子；冒烟全链路通过；**全档终判已出（10-05）：
  负结果归档，RL 线整体不成立**，见 §五同期交付归档与 RESEARCH_TODO §二）
- [x] **银河 0608 L1（风险标签可测成立）**：`factor/classic.compute_galaxy_features`
  （36 特征）+ `build_galaxy_labels`（三标签）+ `scripts/evaluation/galaxy_l1.py`
  （mdd test 0.25/0.40 超研报量级；sharpe/alpha 年度翻符号 → 质量门槛+回退为必要设计）
- [x] **策略级过拟合检验**：`stats/pbo.py`（14 用例，统计锚点测试）
- [x] **多因子10 两主力合成方法 + T 扫描脚本**：`synthesize_ic_ir_max`/`synthesize_ic_max`/
  `half_life_weights`/等权分支（26 用例）+ `scripts/evaluation/mf10_t_scan.py`
- [x] **QP 求解器三件套（AI39 对齐）**：`industry_target_from_benchmark` /
  `max_weight_change` / `solve_lambda_grid`；顺手修 industry_target Series 歧义 bug
  与 risk_aversion 反向 docstring
- [x] **每日邮件正文链路** + **研报 PDF 抽文本工具** + markdown 依赖显式化
- [x] **数据链路健壮性（09-16 事故治理）**：cache 宽表合并 OOM 修复、状态表分片落盘 +
  子进程硬超时
- [x] **分钟频一/二层链（09-09~09-14）**：MinutePanelStore + 42 个 im_* 特征
  （37/42 显著，尾盘动量反转 t=-23.4）+ A/B 对照（with_im 0.583 vs placebo 0.291，
  日内因子真实有效）+ GP 二层挖掘（6 公式入库但组合增益≈0）+ 挖掘层接入 GP/GFlowNet

---

## 三、工程层剩余债务

- [x] **.venv rl 依赖债核销（09-24 实测）**：gymnasium/sb3/sb3-contrib/torch
  在 .venv 已齐（torch 2.13）；真实堵点是 pandas 3.x 下 `Series.rank().to_numpy()`
  返回只读视图令 `portfolio_env.rank_pct` 原位覆写炸——已修（显式 copy），
  双解释器（sys pandas 2.3.3 / venv pandas 3.0.5）rl 测试全绿，**.venv 保留
  pandas 3 作兼容性金丝雀**（日更计划任务运行中，不降级）。
- [x] **portfolio_env 涨跌停/停牌掩码注入（09-24 完成）**：
  `build_tradable_mask` 增 `execution_lag` 参数（0=T 日状态，决策日收盘调仓
  自洽；默认 1 零回归）；`run_portfolio_phase2.make_env` 掩码 = 指数成分 ∩
  T 日可交易 + 整行全 False 回退防炸训练；QP 臂 `valid` 过滤同口径 + 信号
  reindex 兜底（mdd_66 长标签尾部覆盖不足不再 KeyError）。测试
  `tests/test_phase2_tradability.py`；hs300 ppo/qp/ew 冒烟通过。
- [x] **`stage_backtest`/`stage_ensemble` eq 缓存失效（P0 附带，09-23 完成 commit f4d3172）**：
  rolling_grid_alla 三处 exists-skip（backtest eq / select json / pred parquet）全部接
  产物指纹 sidecar（`.fp.json`），失配自动重算（eq/selection 分钟级自动；pred warning 后
  重训）；eq 指纹绑 pred 内容（重训→eq 失效，断点续跑仍命中）；`--exclude-features` 补硬
  校验须配 --out-tag；**附带发现并修复 --quick 冒烟产物可被全量复用**（years 入 pred 指纹）。
  注意 stage_ensemble/smallcap 本就总重跑无此病；指纹抓不住不改常量的纯代码逻辑变更
  （此类走 --out-tag）。
- [ ] **邮件链路 Skills 化**（可选，速读批借鉴）：build_daily_email_body 的口径约束固化 +
  渐进式披露；技能总量控制在上限 20–30 内。
- [x] **私有名跨模块 import 存量红灯（`test_layering`，原 19 处；10-16 清零）**：
  实测 6 族中 5 族已被此前会话清掉（holder_dyn `_apply_lag`/`_rolling_sum_panel`
  已公开化、altfactor `_classify`→`classify`、alphapool_env 三件、market_features
  `_expanding_z`→`expanding_z`、rolling_grid/seq_predictor_compare 的引用均已转
  测试白盒或公开名）；**生产侧最后 1 处 = `hp_search_ic.py` 引
  `retrain_freq_ic._rolling_pred`**——已公开化为 `rolling_pred`（定义模块改签名+
  docstring 注明公开化缘由，引用方同步），`test_layering` 15 用例全绿。
  tests 白盒存量按 `_TESTS_ONLY_PRIVATE` 白名单管理，维持不动。

### 已完成归档

- [x] 六轮工程整改 P0–P5（2026-08-29 基线）：stats 公共层、scripts 收敛、521 测试全绿
- [x] 口径统一第一~七批（09-10~09-11）：run_backtest 删除、四项决策拍板、
  权重生产入口定案、scripts 全量重排、同名指标公式对齐、分层收口、HTML 字节复现
- [x] 私有名倒挂清零、根目录 probe 清理、零引用模块删除；cli_common 推广、
  HTML 模板收编、超长文件拆分、最小 CI（09-10 起转手动触发）

---

## 四、低优先级（知情即可）

- [ ] `factor/technical.py` 与 `factor/technical_indicators.py` 双口径指标
  （有意保留；SAR 有两份实现）
- [ ] GFlowNet 自写 PPO（`factor/gflownet/ppo.py`）与 AlphaPool MaskablePPO 双轨
- [ ] optimize 标注的 P3 待建：完整多期最优执行、风险预算非等权
- [ ] 一次性实验脚本（`gtja_*` / `diagnose_*` / `compare_*` 等）保留原样，不迁移 cli_common

---

## 五、好机器批次出数后的后续立项（2026-09-25 立，2026-10-09 全部收官）

> 证据与数字：`reports/batch1_920_收尾对照表.md` §⑫–⑯、`reports/batch2385_结果速览.md`。
> 定版演进：10-02 v2 口径拍板 h1020⊕s0.5（17.87%/0.781）→ 10-06 §⑭ fundind 快信号
> +buffer(20/30)（18.68%/0.813）→ 10-08 §⑯ buffer 网格上修 15/40（18.07%/0.805/
> 换手 19.1%，top20 口径）。五项立项 + 后续 P-A/P-B/P-C/P1 提升路径全部裁决完毕：

- [x] **批次 8 组合层传导研究**：头部诊断确认「ICL 的 IC 优势在截面中腰部不在头部、
  跨方法头部 Jaccard 0.17 真互补」→ H1 三臂集成被否 / **H3 头部内加权成立**
  （tabicl_w3m 加权 11.97%/0.739 中小池组合冠军）→ zz1000 复验跨池成立 + 新发现
  「gbdt+tabicl 对集成 20.03%/1.25」→ 全A 大截面裁决：**ICL 优势不跨大截面**
  （gbdt IC 4.4 倍反超），主模型基座维持 gbdt、tabicl 定位中小池组件。
  余项（对集成组合层验证 / H3 跨池复核）留中小池场景非主线。
- [x] **GFlowNet 旋钮增量（国金24 残余③④）**：RRE≥0.3 门槛 |IC| 中性略负
  （0.113 vs 0.117，种子方差增大）；预算 2x 饱和（1200 iters 反而 −0.004）——
  **raw+im 无门槛 600 iters 为该线定版形态**；⑤入库收官：114 个 gf_* 因子入
  hs300_2022_2025 库（33 个 |t_nw|>2）、**PBO=0.114 族层稳健**（vs AI97 0.704）。
- [x] **AI97 步长归因**：100k 步（10 倍预算）训练段复合 IC 仅 −0.05→+0.02、
  训练/测试符号翻转依旧 → **真实样本外失效而非未收敛**，该线正式归档；
  若重启需换池/构建机制，不再加预算。
- [x] **E4 交互特征换注入方式**：池竞争路径证伪在先（15 个交互因子 |IC| 不入
  top-150、选股逐位同基线）；强制注入臂（ortho920e4force）gbdt h1 M raw f0.10
  11.81% vs v2 基线 11.67%（**+0.14pp 噪声级**）——E4 线结案，「基本面当条件
  变量」假设在本构造下不成立。
- [x] **方案 B 缺面板对齐**：唯一存活余项，上移 §一（小工程）。

### 同期交付归档（2026-10-04~09，定版冲刺批）

- [x] **标签工程三臂全量闭环（10-04/05）**：gauss_rank 零增量（主线 rank 已在最优
  族）/ co 隔夜判负（−1.0~−1.2pp、回撤换手双恶化；ICIR 升经逐日 IC 配对检验
  t=−5.2 判定为无价值副产品，日频组合 28/28 恶化）/ ir 判负（AI29 回撤代价 +
  Label Alchemy vol_scaled 预警双应验）——**现行 rank+cc+return 即标签维局部最优**，
  预算转投 horizon 混合（已兑现：h1020 为定版成分）。
- [x] **phase2 修复三连 + 全档终判（10-02~05）**：env er 基准改可交易基准
  （0/4 全灭真凶=口径 bug 非 RL 无效，零动作 −10.9%→0.00% 实证）+ PPO 门槛按
  zz1000 重校（原值 0.02/0.08 系研报科创50口径，PDF 取证）+ 候选集成 rollout
  per_model 形状修正 → 全档 24 臂终判：**银河口径 RL/QP 在 zz1000 无稳健超额、
  reward 六项分量贡献不稳健——RL 路线在本项目数据域整体有效性不成立**（与
  AI97 互证）。已知残留：QP-2025 solver 失败、TopN-2026 日期缺口、no_alpha
  消融未生效（三项记录在案）。
- [x] **主模型提升路径 ②①④③ 全部裁决归档（10-06）**：buffer×定版采纳（+0.40pp/
  换手 41.3→27.0%）/ fundind 快信号采纳（18.68%/0.813 新定版）/ fundind 慢信号
  负（慢信号偏好标准中性化）/ T 窗扫描维持 T=24 / tabicl 全A 冒烟裁决 gbdt 基座
  不可替代。工具沉淀：`buffer_arm_920.py` defv 变体、`fundamental_blend
  --panels-subdir`、`tabfm_rolling.py` 全A 预筛路径（ic_h top-N 防 OOM）。
- [x] **D 盘数据治理（10-06）**：sdk_cache 137G→~3G（balance_sheet/pledge_freeze/
  restricted/margin 四表 116G 经 parquet 完整性验证后删）；income/cash_flow 补齐
  （income 4383 只=SDK 源上限）→ 引出 P0 income 复核（负结果 §⑮：信息已被
  balance 系+原池覆盖，定版维持 v2）与**环境漂移更正**（同配置如今重跑
  ~17.0–17.2%，不回退数据换数字，引用注明数据态）。
- [x] **alt 三族×主模型双路径终审（10-08）**：席位竞争 0/36 + 慢信号家族并入
  （--include-alt-news）18.68%→16.27% 双败——**alt 三族对主模型无增量结案**；
  news 族 h20 反转方向（|IC| 随 horizon 单调增强）与 macro 择时特征留待未来
  新信号线。
- [x] **GP 预算放大裁决（10-07）**：pop400×gen25 拉满仍停早期平台、去冗余仅 1 因子
  入库——与 GFlowNet 饱和互证：**现有特征空间上做公式重组合的两类算法都已触及
  上限，增量 alpha 需新信息源或新归纳偏置**。
- [x] **LLM-MCTS all-A 正式跑→入库→终审（10-08/09）**：750 候选收编 14、PBO=0.039、
  入库 llm_*、v4 席位 4/14（唯一挖掘引擎席位突破）、组合层 +0.16pp 噪声级不采纳
  ——「席位突破、组合无增量」归档（见 §二东吴条目详录）。

---

## 六、增量方向（2026-10-19 立项，按 ROI 排序）

> 背景：2026-10-19 战略复盘结论 = 「当前数据 × h1–h20 × long-only Top10%」到顶，
> 但不是「横截面预测」到顶。未证伪的只剩三类；本节是这三个方向的真源。
> 每个方向先做**可证伪的最小探针**，不直接上全量；执行纪律不变：
> `caliber.py` + PBO/DSR + 多窗口 + 安慰剂，任何「新」不豁免。
> 证据目录：`reports/frontier_audit_2026-10-19/`。

### A1. 长 horizon（h60/h120）+ 基本面/质量模型（P1，第一优先）

- **假设**：h1–h20 的 DPP 选择把基本面压到 6 席；拉长到 h60/h120 后，基本面/
  质量/价值的信息占比上升，能形成一条**低换手、高容量**的第二策略。
- **正向前置**：h20-only 16.94% vs h1 13.51%（882 口径）；h1020 在 920 口径
  +3.31pp；基本面族 IC h1→h20 ×1.143，量价族全负（RESEARCH_TODO §一 E1'）。
- **最小探针（先不做全量模型）**：
  1. 用 `close_adj` 造 h60/h120 标签，算主信号 + top50 因子的 IC / NW t / 逐年方向；
  2. 检查重叠标签的有效独立期数（8 年 / 60 日 ≈ 32 个非重叠期），确认 PBO 可用；
  3. 单窗口 2019–2022 训练 / 2023–2026 OOS 的 gbdt h60 探针（同 50 因子池，
     只换标签），与 h20 基线同口径比净 Sharpe / 换手 / 容量。
- **判据**：production 口径下净 Sharpe 或 Calmar 不低于 h20 基线，且换手显著更低；
  **不以「年化更高」为唯一判据**（长 horizon 可能降年化但升容量）。
- **状态**：`[x]` **920 复验否决（10-21 封线；882 口径曾全链闭环——h60 第二策略当时成立）**：
  ① 前置 + 首模型（h20 清单）：主信号 IC 不随 horizon 衰减（h60 0.161）；
  首模型 IC 0.1410 但配对差 t=1.62 不显著（见首轮 README §七）。
  ② **h60 清单**（`select_horizon_features.py`，质量窗 2021-09~2023-09、
  fwd 不穿透 2023 年末、DPP 50）：与 h20 清单仅重叠 18/50，**fundamental
  6 席自然上榜**（ln_mktcap/bp/sue_q/div_yield/sp_ttm）——A1 假设首层证据。
  ③ **h60 清单版模型**：IC **0.1800** / IR 1.38 / 胜率 89.7%（vs h20 清单版
  0.1410，清单贡献 +0.039）。④ **同目标交叉**（`horizon_cross_eval.py`）：
  2×2 矩阵 + 配对差——同目标 h20 **+0.0126（t=3.05）**、同目标 h60
  **+0.0324（t=7.82）**，首模型的「不显著」边界被打破。⑤ **组合层**
  （production 口径 buffer 15/40 月频，2024+）：m_h60 **Sharpe 1.028 /
  Calmar 0.83 / 换手 17.5%** vs 主信号 0.796/0.70/30.7% vs m_h20
  0.736/0.59/19.5%——**立项判据全过**（净 Sharpe↑、换手≈主信号 57%、
  成本近半），2026 弱市年 +18.5% vs 主信号 +5.1%。证据
  `reports/frontier_audit_2026-10-19/`（§十 + `selection_h60/` +
  `h60_pred/` + `cross_eval/`）。
  **边界**：882 口径、OOS 仅 2.7 年（2025 牛市）、组合形态未针对 h60 调优。
  **第三轮（融合对照 + buffer，§十一）**：m_h60 ⊕ 主信号任何 α 融合都不如
  m_h60 单独（1.028）；且 2024 年 m_h60 −1.6pp 落后主信号（优势集中在
  2025/2026）⇒ **按独立第二策略并行，不做融合成分**；buffer 网格平缓
  （10/30 → 1.068 / 20/50 → 1.003），生产形态 15/40 已是合理点。
  **❌ 920 复验未过（10-21 批次 13③，线封）**：配对差 t=0.33/0.78 不显著、
  组合 Sharpe 0.695 < 基线−0.05=0.771（defv 同窗 0.821/换手 18.0%）、
  换手未显著降（17.4%≈基线 97%）——「低换手第二策略」不立项；882 清单
  实体不在本机、920 现算清单 fundamental 6→3 席（构成漂移为候选归因）。
  证据 `reports/frontier_audit_920_reval/`（cross_eval + h60_920）。

### A2. 残差目标挖掘（P1，已开工）

- **假设**：现有 920 因子里还有「主模型漏掉」的正交信息；把残差当目标能把它找出来。
- **已落地（2026-10-19）**：`scripts/evaluation/residual_ic_probe.py` —— 四口径
  `raw / resfac / restgt / both` + Newey-West t + 逐年方向；测试
  `tests/test_residual_ic_probe.py` 3 用例锁口径（残差化与 `residual_panel` 逐位
  一致、raw IC 与 `calc_ic_series` 逐位一致、主信号成分残差化后基本消失）。
- **首轮**：top-200 by |IC|（h=1）+ 非量价 8 族 `stride=3` 筛查；结果落
  `reports/frontier_audit_2026-10-19/`。判据：`both` |NW t|≥3 且逐年方向稳定
  （≥7/9 年同号）的因子数；若为 0 ⇒ 「新因子」这条线正式封底；若 >0 ⇒ 用
  top-K 做二阶 GBDT（目标 = 残差收益）walk-forward。
- **状态**：`[x]` **920 复验否决（10-21 封线；882 口径组合层曾为正——主支撑
  = 二阶 GBDT）**：首轮残差 IC 结论不变（负向 risk/quality + 量价非线性
  成分）。① **二阶 GBDT**（`residual_stack_probe.py`，特征池按早年列重排 =
  无选择前视，标签=残差收益，walk-forward 季度重训）：stage2 OOS IC（残差）
  **0.0359 / IR 0.49**；融合 α=0.50：**Sharpe 0.796→0.950（+0.154）**、
  年化 +3.76pp、安慰剂 −0.054——**A2 组合层为正的主证据**。
  ② **负向 overlay**（`residual_overlay_probe.py`）：固定 neg6 + w=0.40 全期
  ΔSharpe +0.170、9 年 9 正、placebo 全负；但 **walk-forward 因子选择复核后
  缩水到 +0.02**（每年初用截至上年末残差 IC 重选 top-6，`overlay_wf/`）——
  固定集约 7/8 的超额是选择前视；真实增量集中在近年（2023+ 因子集与固定
  neg6 趋同，+0.8~+2pp/年），定位「真实但薄的剔除规则」。
  证据 `reports/frontier_audit_2026-10-19/` §九（含 9.1.1 修正）。
  **❌ 920 复验未过（10-21 批次 13①②，线封）**：wf overlay ΔSharpe −0.030
  （固定集 +0.085=选择前视上限，placebo −0.036 方向正确）；stack 残差 IC
  0.0742 过线但 fuse α=0.5 ΔSharpe −0.041（验收 ≥+0.08）——**残差 IC 强
  ≠ 组合层有增量**（four_paths 机理再证）。数据态换相：920 残差榜 top5
  转正向质量因子（ebit_margin/roic_ttm 等，留观不单独开线）。证据
  `reports/frontier_audit_920_reval/`（overlay + residual_stack + 两份
  residual_ic CSV）。

### B1. beta hedge / 指数增强 / Barra Σ（P2，第二优先）

- **假设**：当前 α 7.73% / β 9.76% / β 0.977，约 56% 收益是 beta；剥 beta 或
  做基准约束能改善 Sharpe/Calmar/可投性，而不需要更好的预测。
- **已有零件**：Barra CNE6 简化版 13 变量、R² 0.217（`reports/barra_cne6/`）；
  `factor_covariance` / `specific_risk` / `assemble_sigma` 雏形；
  `costs.short_borrow_rate`。
- **最小探针**：
  1. beta hedge：多头组合 + 空 IM/IC（或自定义篮子），60 日滚动 beta；
     算净 Sharpe / MDD / 基差成本敏感性；先不做 QP 全家族。
  2. 指数增强：CSI 500/1000 基准 + 3–5% TE 约束，用 Barra Σ 做风格约束；
     先看 TE 与 α 的权衡，不直接上 CSI 300（α 偏小盘）。
- **判据**：净 Sharpe/Calmar 相对 long-only 有提升，且换手/容量不恶化；
  若只是「降波动也降收益」则按风险偏好归档，不算 alpha。
- **状态**：`[~]` **纯多头探针已跑（10-26），结论：不立项为 α 线、归档为风险偏好型替代**。
  证据 `reports/index_enhance_probe/README.md`（脚本
  `scripts/evaluation/index_enhance_probe.py`）：α 在中证500 成分内**有效**
  （月度 rank IC 0.069、t=4.3、安慰剂 0/20），但 3–5% TE 档净超额仅
  **+1.5%/+1.9%（t≈0.8，未达 t≥2）**，TE 曲线已贴该 IC 的理论上限（IR≈IC×√12），
  且 **2025 单年 −7.7%（λ0.5）/ −10.3%（λ0.8）**——与主线近期 α 衰减同源。
  其中 3% TE ≈ λ0.35（+1.1%），5% TE ≈ λ0.75（+1.8%）；Sharpe 0.10→0.19、
  MDD −41.8%→−35%，换手 17%/月、容量无压力 ⇒ 只能作为「本来就要持有中证500」
  的替代配置，不是新 α。**未做**：Barra CNE6 Σ 显式约束、beta hedge（无期货权限）、
  fundind 锚口径 pred（锚那次未落盘）。
- **数据前置（10-26 实测满足）**：① 指数成分**日**权重 `get_index_weight` ——
  000300 148 万行 2006-04-10+、000905 191 万行 2010-12-10+、000852 290 万行
  2014-10-20+（无公告时间戳 ⇒ T+1）；② 股指期货日线 `get_future_code_list`
  816 合约、`IF/IC/IM` OHLCV+amount 正常（本分支未用）。
  证据 `reports/frontier_audit_2026-10-19/sdk_field_audit.md` §七。

### B2. SDK 未用表盘点（P2，非股票 α，防「数据源已尽」误判）

- **已核**：融资/龙虎榜/大宗/质押/解禁/预告已进池（资金流 12、事件 31、股东 18，
  均值 |IC| 0.01 量级，弱）；**未接**：ETF 申赎/份额、指数成分日权重、可转债、
  期权、国债收益率（SDK 手册 §3.5.10–3.5.15）。
- **判断**：这些多数**不是股票截面 α**，但能开 ETF 轮动、转债、指数事件、风险
  overlay 等别的策略；不要指望它们救选股。先做 1 周字段审计（覆盖/历史/PIT），
  再决定是否立项。
- **状态**：`[~]` **字段级审计本机实测完成（10-26）**，框架 10-20 立：
  `reports/frontier_audit_2026-10-19/sdk_field_audit.md` §七（脚本
  `scripts/evaluation/sdk_table_audit.py`，产物 `sdk_field_audit/` 每表字段 CSV）。
  结论：**高优先级两张表都过线** —— ① ETF 份额 `get_fund_share`（510300/510500/
  159915：3522/3319/3610 行、2011~2012 起、`ANN_DATE≥CHANGE_DATE` 占比 1.000
  ⇒ PIT 安全）→ **ETF 轮动线可立项**；② 指数日权重 `get_index_weight` → **B1 解锁**。
  待补/降级：可转债 `kzz_conv` 只是每债一行的条款快照（需再拉 5–8 张子表才能成线）；
  期权有 1312 只合约资料但**无 IV 字段**（要自己反解，成本高，维持「中」）；
  国债收益率 5 种代码写法全返回空 dict（代码格式未明，维持「低」）。
  附带更正：**「本机无 SDK 数据面」判断有误**，本机可 `is_local=False` 回源下载
  （缓存 `E:/data/sdk_cache/`）。

### B3. 聚宽因子看板复核：三族缺口 + 探针 + 滚动 A/B（2026-10-26，`[x]` 已闭环：不入库）

- **背景**：`joinquant.com/view/factorlib/list` 的 285 个因子是单口径算的（HS300 +
  近 10 年 + 多空组合Ⅰ + 过滤涨跌停停牌）。此前只按该口径挑过 4 个，本轮用
  **5 池 × 4 区间 = 20 口径**复核，再搬到本地因子库口径复算。
- **复核**：285 → **176 个**方向一致（16 个有效口径中 ≥15 个同号；149 个 16/16
  全同号）。数据源**不缺**：`daily_all_a`（OHLCV+amount）+ `_base`（复权价、
  流通市值、换手）能算全部量价因子；真缺的只有**分析师一致预期**类
  （earnings_yield / earnyild / predicted_EP / predicted_growth，本机无表）。
  950 特征库的真实缺口 = **成交金额族**（alpha158 的 VMA/VSTD 是成交量口径），
  以及**换手率**（alpha360_VOLUME* 是原始成交量，不是换手率）。
- **探针**：`scripts/evaluation/jq_factor_probe.py`
  （测试 `tests/test_jq_factor_probe.py` 4 例）。口径：全 A、T+1 可交易掩码、
  h1/h20 Rank IC + NW t、行业内去均值 + lncap 秩残差、与 `panels_neu` 950 面板
  去重（抽样近似 + top1 精确复核）、月度十分位多空（双边千三、5 窗）。
- **结论**：过线（残差 |IC|≥0.01、dup_corr<0.7、月净超额≥1pp 且 ≥4/5 窗）的只有
  **成交金额族**：TVSTD6 / TVSTD20 / TVMA6 / TVMA20（相对库内最相似因子的增量 IC
  −0.027~−0.050、t −24~−37、月均 1.7~2.1%、4-5/5 窗）；但四者互相 0.84~0.95
  高度相关 ⇒ 实际取 **TVSTD20**（5/5 窗、年化 +25.0%、dup 0.690）为主、
  **TVMA6**（增量最强 −0.050、dup 0.572）为备选。换手族边缘（VOL5 dup 0.734、
  增量 −0.024、4/5 窗）；波动/残差波动族 IC 稳（−0.044~−0.055）但相对
  alpha191_159 已无增量（resid_dup ≈0）；boll_down/vosc/cci/atr/mfi dup>0.7；
  momentum 方向不一致且 IC≈0 ⇒ 不通过。**边界**：金额族月换手 35~58%，
  容量/冲击成本未验；聚宽口径含涨跌停过滤，本口径只有停牌/可交易掩码。
- **证据**：`reports/jq_factor_probe/README.md` + `jq_probe_h1.csv`（27 行全指标）。
- **滚动 A/B（2026-10-26 已做，结论：不入库）**：把 TVSTD20 + TVMA6 经
  `--force-features` 强制追加进 50 特征（52 特征，绕过 DPP 竞争），只跑生产模型
  gbdt / h1 / 2018–2026 / ortho（两臂仅特征集不同）。结果：OOS IC 0.1122 → 0.1124
  （Δ +0.0002，t 3.25 可测但经济量≈0）；预测面板截面秩相关 **0.9959**；组合层
  M 频 Δ年化 −0.09pp（M/raw/f0.20），12 组合全落 −0.38~+0.32pp，逐年 Δ 3 正 6 负、
  std 0.38pp ⇒ **全在噪声内**。判读与 altf P1 同型：950 特征库已重构出成交金额
  信息（dup 0.57~0.69），给最宽松通路也无增量。证据：`reports/jq_amount_roll/README.md`
  （两臂 `reports/alla_rolling_jq_{base,amt}/`）。**未测边界**：h5/h10/h20、
  非 gbdt 模型族、生产 buffer 形态、小盘池。
- **下一步**：不再为此族排期；若日后要做，只剩「h5/h20 或换模型族/换池」两条未测
  分支，须先有明确目标场景。两张面板（`panels`/`panels_neu`）保留为实验资产，
  不写 registry / ic_h。

### C1. 模型异质性 / meta-labeling（P3，低预期）

- pool-specific（按 size/流动性/行业分域）、ensemble 分歧定仓位、GNN（仅当拿到
  真实关系型数据：供应链/基金共持）。P1b 已证同数据换架构不解决；不优先。
- **状态**：`[ ]` 不投入，除非 A/B 线出正结果后有明确接口。

### D1. ETF 轮动（2026-10-26 立项，P0+P1 已收口，P2 待开）

> 2026-10-29 二次清理：本节引用的 `etf_rotation_probe/`、`etf_rotation_p2/`、`etf_p4/`、
> `etf_dedup/`、`etf_capacity/`、`etf_pool_expand/`、`etf_alpha_probe/`、
> `etf_survivorship_probe/` 产物目录与 5 个对应探针脚本、18 个 `_tmp/_etf_*.py`
> 已全部送回收站。**现行结论与全部数字见 `reports/etf_contest5/README.md`**（三方案 +
> 25 条判否清单，口径统一 T+1 开盘成交）；本节保留立项/收口过程作为研究路径留档。
>
> 2026-10-28 清理：本节引用的 `etf_contest_probe.py` / `etf_dedup_probe.py` 与
> `reports/etf_contest5_opt/`、`reports/etf_contest_wl{,2}/` 已送回收站；采纳/判否
> 结论与关键数字保留在 `reports/etf_contest5/README.md`。

- **方案真源**：`reports/docs/design/ETF轮动_研究与实施计划_2026-10-26.md`；
  结论真源 `reports/etf_rotation_probe/README.md`。
- **P0 `[x]`（10-26）**：池扩到 **41 只**（宽基 8 / 行业 14 / 主题 8 / 债券 4 / 商品 1 /
  跨境 6；42 候选剔除 `512580.SH`，60 日均额 0.08 亿未过 3000 万门槛），入池体检
  `_pool_audit` 随 `update_etf` 自动打印；四表回补到 2026-10-26——
  `daily_etf.parquet` 79,417 行、`fund_share.parquet` 99,746 行（PIT
  `ANN_DATE≥CHANGE_DATE` = 1.000）、`fund_nav.parquet` 100,946 行、
  `fund_iopv.parquet` 97,581 行（`511090.SH` 无 IOPV）。入口
  `scripts/ingest/update_etf_flow.py`；接口 `DataSource.get_fund_{share,nav,iopv}`。
  - **10-26 二次扩池 41 → 50 `[x]`**：低相关 4 只（豆粕 159985 / 有色期货 159980 /
    能源化工 159981 / 日经 513520）+ 高弹性 5 只（半导体设备 159516 / 科创芯片 588200 /
    机器人 562500 / 电池 159755 / 创新药 159992）。四表已补齐（K 线 89,819 行、
    份额 112,338 行、净值 113,565 行、IOPV 110,118 行 / 49 只）。
    **A/B 结论**：P1 基线 7.14/0.62/19.01 → **7.56/0.65/16.23**；
    champion 10.96/0.82/16.70 → **11.95/0.91/16.14（三段 3/3）**；
    `champ_mom_dsh_f67` **14.43% / 1.06 / 17.16%**（但 f67 是事后邻近点，须进 P4）；
    Δshare 单独增量被稀释（`mom_dsh_f80` 0.80→0.68，IC 0.086→0.064 / t 3.31，仍过判据）。
    扩池改善视为**上界**（与首次扩池同源的池选择偏差）。真源
    `reports/etf_pool_expand/README.md`。幸存者偏差 A 臂已冻结为 41 只以保持可复现。
- **P1 `[x]` 有条件通过（10-26）**：13 变体 + 7 变体稳健网格，判据先验固定
  （Sharpe ≥ 0.5 / MDD ≤ 35% / 三段 ≥ 2/3）。唯一主表 PASS = `dual_ivvt15_m5`
  （12-1 + 绝对动量 + 逆波动 + 15% 波动目标）：**年化 7.14% / Sharpe 0.62 /
  MDD 19.01%**（旧周频合成基线 7.04% / 0.26 / 38.38%）。稳健网格 7/7 过线
  （target_vol 12/15/18、TopK 3/5/8、缓冲区、cost×2 中最低 Sharpe 0.52；`ivvt15_m3`
  达 0.80 且三段全正）。**Caveat**：波动目标是唯一有效增量（12-1 单用 0.13、
  逆波动单用 MDD 42.4%、缓冲区滞后），且组合项是看主表后追加 ⇒ 多重检验候选；
  池构成含幸存者偏差（仅含当前在交易的 ETF）。
- **P1 遗留**：① ✅ 用退市 ETF 清单量化幸存者偏差（10-26 完成，见下条）；
  ② PCF 每日快照存档（SDK 仅当日快照）→ 为 S2 映射校准铺路（S2 已判负，降级为可选）。
- **P2 前置三项 `[x]`（10-26）**：结论真源 `reports/etf_rotation_p2/README.md`，
  脚本 `scripts/portfolio/etf_rotation_p2.py`。
  - ✅ **防御腿**（风险关闭从 0 收益现金改为债/金）：国债腿 Sharpe 0.76 / MDD 16.83%；
    债+金 0.77。黄金单独用年化 11.61% 但 MDD 29.91%（是收益来源不是降波工具）。
  - ✅ **份额拥挤度**（PIT 面板按 `ANN_DATE` 生效，250 日分位 > 0.80 剔除）：
    单用 +0.05（未达阈值），与防御腿叠加后**年化 10.96% / Sharpe 0.82 / MDD 16.70%**
    （基线 7.14% / 0.62 / 19.01%），邻域 7 变体 Sharpe 0.74–0.93、双倍成本 0.75 ⇒ 非尖峰。
  - ❌ **趋势一致性 R²**（动量 × 60 日对数价格回归 R²）：Sharpe 0.56、换手 45% ⇒ 不采纳
    （聚宽的斜率×R² 适用于只持 1 只的强过滤场景，TopK=5 分散后只会削掉高动量段）。
  - 新基线候选 = `def_mix_crowd_f20`：目标波动 15% / TopK=5 / 防御=债+金 / 拥挤度 0.80。
  - **频率与持仓**：月频最优（双周 0.75 / 周频 0.81 vs 月频 0.82，周频只换来 MDD 15.66%）；
    `top_k=5` 为上限，PIT 掩码 + 绝对动量后**实际平均 4.76 只**，防御腿平均占 39.0% 仓位。
  - **成本**：月换手 35.4%（单边）⇒ 每次调仓成本 = `2×35.4%×(佣金+滑点)`，默认 5+5bp
    约 0.85%/年（占当前资金），回测全期 1.29%/初始资金/年。按实际佣金万分之一（1bp）+5bp 滑点：
    **0.52%/年，年化 11.36% / Sharpe 0.85**；1+2bp 为 0.26%/年、11.66%/0.88；1+10bp 为 0.94%/年、10.87%/0.82。
  - **10-26 口径修正**：选标必须发生在 PIT 掩码之后（因子先按可投集合置 NaN），
    旧"先选后掩码"会让不达标标的占坑、持仓被动缩水到 2 只；主探针已固化
    `factor = factor.where(mask)`，上表为修正后数值（最优项 11.23%/0.85 → 10.96%/0.82）。
- **S2 持仓映射 α `[结论：不通过]`（10-26）**：30/30 只 A 股 ETF 映射到跟踪指数
  （`get_index_constituent` 的 PIT 成分 + 自由流通市值代理权重，`get_index_weight`
  实测只有 000300/016/905/852），ETF-月覆盖 97.8%、每期 28.6 只。ETF 层 α 对下月收益
  **mean IC = −0.055 / ICIR −0.148 / t = −1.32**；动量前 50% 子集 IC −0.031（t −0.65）；
  与 12-1 动量相关 −0.07。个股 α 方向已核验为正（次日/5 日 IC +0.067/+0.089）⇒
  不是符号错，是线性聚合把信号稀释/反转。**不进入组合回测**；真源
  `reports/etf_alpha_probe/README.md`，脚本 `scripts/portfolio/etf_alpha_probe.py`。
  若再试，只允许一次非线性聚合（ETF 内 α 前 10% 成分权重占比），判据须先固定。
- **幸存者偏差 / 池选择偏差 `[量化完成]`（10-26）**：结论真源
  `reports/etf_survivorship_probe/README.md`，脚本 `scripts/portfolio/etf_survivorship_probe.py`
  （全 ETF 清单 3031 只；A = 现 41 只固定池，B = 机械 PIT 动态池 489 只曾入池）。
  - **退市偏差 = 结构性零**：126 只退市 ETF 中 52 只有满 3 年窗口，**0 只**曾过 3000 万成交额门槛
    （3 年后最高 60 日均额 p50 103 万 / p90 569 万 / max 2521 万 = 门槛 84%；B 池中位 2.38 亿）。
    机制自洽：退市原因≈流动性枯竭，池规则已前置排除。数据源缺 1245/1372 只退市行情 ⇒
    无法逐只反向入池，只能做门槛+分位论证。
  - **池选择偏差 = −0.16 Sharpe，必须标注**：A 7.14% / 0.62 / 19.01%（与正式探针逐位一致）、
    B 5.89% / 0.46 / 22.86%，三段 +17.4/−5.1/+7.0% 全部弱于 A。
    ⇒ 报 41 池结果时须注明其为**上界**；机械可复现版本约 5.89% / 0.46。
- **S1 残余 Δshare `[结论：通过（反转）]`（10-26）**：结论真源 `reports/etf_rotation_p2/README.md` §七，
  入口 `scripts/portfolio/etf_rotation_p2.py --variants share`。
  - **方向 = 反转**：−Δshare20 月频 rank IC **+0.086 / t +3.97**、−Δshare5 +0.080 / t +4.30，
    均通过「|IC|≥0.03、t≥2、安慰剂 0/20」；正向 Δshare 反向显著（资金流动量说被否）。
    Δshare60 只有 0.038 / t 1.44，不通过。n=81 个月。
  - **组合层**：动量 + 剔除申购最热 20%（`mom_dsh_f80`）**9.35% / 0.80 / MDD 16.89%**
    （P1 基线 7.14/0.62/19.01）⇒ ΔSharpe +0.18、MDD −2.1pp；阈值邻域 f67=0.80、f90=0.70。
    与 champion 叠加（`champ_mom_dsh`）**13.30% / 0.97 / MDD 18.24% / 换手 39.6% / 成本 1.53%/年**，
    相对旧 champion 10.96/0.82/16.70：Sharpe +0.14、年化 +2.3pp，**但 MDD 恶化 1.54pp、
    换手 +4.2pp、成本 +0.19pp/年**（三项须一并标注）；邻域 f67 0.95 / f90 0.92，非尖峰。
  - 纯 Δshare 反转排序 7.47% / 0.71 / 14.52%，但换手 61.8%、成本 2.23%/年，不作候选。
- **S3 折溢价 `[IC 显著 / 组合判负]`（10-27，50 池）**：折溢价 = 原始价 / IOPV − 1
  （49/50 覆盖）。月频 rank IC：`prem20` **−0.072 / t −3.27**、`prem` −0.067 / t −3.02，
  均过「|IC|≥0.03、t≥2、安慰剂 0/20」⇒ 高溢价后续收益低。**但组合层不采纳**：
  单增量最好 `mom_prem_f67` 仅 7.70% / 0.68 / 15.45%（基线 7.56/0.65/16.23，ΔSharpe +0.03
  未达阈值、换手 30.8→39.9%）；与 champion 叠加反而从 11.95/0.91 降到 **10.94/0.84**。
  原因：溢价最高的是跨境 QDII 与商品期货（结构性溢价），动量常在其上有效，月度剔除砍掉收益腿。
  折溢价只保留为**执行时点**辅助（不在溢价 >1% 时追高），不做月度截面过滤。
  真源 `reports/etf_rotation_p2/README.md` §八。
- **P3 容量表 `[x]`（10-27）**：结论真源 `reports/etf_capacity/README.md`，脚本
  `scripts/portfolio/etf_capacity_probe.py`。口径：单只单边下单 ≤ 60 日均额 5%。
  - **100–500 万无压力**（0% 调仓日超限）；**1000 万基本可执行**（def_mix 6.1% /
    champ 2.4% 的调仓日略超，拆 2 笔即可）；**3000 万约 1/3 超限**（需拆 2–3 天或放宽到 10%）；
    **1 亿以上按 5% 不适用**。
  - 容量分位：def_mix_crowd_f20 p5 = **994 万** / p50 4718 万；champ_mom_dsh_f67 p5 = 1209 万 /
    p50 3813 万。**硬约束是新增商品期货 ETF**（159981 能源化工 2.97 亿/日、159980 有色期货
    6.15 亿/日），不是国债/宽基。资金量大时优先 `def_mix_crowd_f20`（换手更低）。
- **P3 组合层 `[不达升级判据 / HRP 记降风险选项]`（10-27，50 池）**：ERC（等风险贡献）
  与 HRP（层次风险平价）引入 60 日完整协方差，另测单只权重上限。真源
  `reports/etf_rotation_p2/README.md` §九。
  - 基线信号上全部降 Sharpe：erc 0.59（−0.06）、hrp 0.44（−0.21）、maxw25 0.52、maxw20 0.41，
    HRP 换手还升到 39.5% ⇒ TopK=5 的入选者高度相关，完整协方差无分散可挖。
  - champion 上 HRP 才有效：**12.33% / 0.95 / MDD 12.18%**（champ 11.95/0.91/16.14，
    MDD −3.96pp、Calmar 0.74→1.01），代价月换手 +6.3pp、成本 +0.19pp/年；Sharpe 仅 +0.04，
    **不达 ΔSharpe>0.1 判据，不作主升级**。权重上限（25%/20%）一律降收益，不采纳。
- **P4 样本外 `[champ / champ_hrp / champ_erc / champ_dsh_f67 全部 PASS]`（10-27）**：
  真源 `reports/etf_p4/README.md`，脚本 `scripts/portfolio/etf_p4_probe.py`。
  IS = 2020-01→2024-12，OOS = 2025-01→2026-10，多起点 2020/21/22/23。
  - `p1_base` **样本外失效**：OOS Sharpe 0.45、2022 起点 0.31 ⇒ 基线本身不够。
  - `champ` OOS 1.02 / MDD 16.14%（年化 17.06%），起点 0.91/0.64/0.66/0.86；
    `champ_hrp` OOS **1.27 / MDD 11.48%**（年化 20.60%），起点全部 ≥0.73；
    `champ_erc` OOS 1.13 / 14.58%；`champ_dsh_f67` OOS 1.04 / 17.16%（IS 最强 1.10，但换手 46–49%）。
  - 结论：**定版主策略取 `champ`（def_mix_crowd_f20）**；HRP 作为可选的降 MDD 开关保留
    （OOS 优势可能含 2025–26 单一区制，不凭此单独定版）。
  - 口径提醒：OOS 仅 ~22 个月且是上涨区制；参数仍是在全样本上挑的，属「时间切片留出」。
- **P5 接线 `[x]`（10-27）**：新增 `scripts/portfolio/etf_champ_signals.py` +
  `reports/etf_champ/`（`signals.csv` / `candidates.csv` / `board.html` / `meta.json`）。
  输出 champ 的**目标持仓、下单金额、参与率、溢价告警、候选榜单**；`--positions` 可传
  实际持仓按差额成交。当前快照（2026-10-08 调仓、1000 万）：风险腿 57.9%（有色期货/
  红利/煤炭/有色/能源化工）+ 防御腿 42.1%（国债/黄金各半），0 项告警。
  真源 `reports/etf_champ/README.md`。S1 未做项（Δshare/成交额、连续净申购天数）暂不推进。
- **后续可选**：周更自动任务（schedule 每日/每周跑信号板并推送）、HRP 版信号板开关
  （P4 OOS 1.27，但全样本仅 +0.04，需再攒一段样本外再定）。
- **两个月比赛方案 `[x]`（10-27）**：新增 `scripts/portfolio/etf_contest_probe.py` 与比赛信号板，
  真源 `reports/etf_contest/README.md`。80 个滚动 2 月窗口：
  - A（推荐）：12-1 动量 + 拥挤度过滤、Top3、周频、等权满仓 —— 中位 **+2.44%**、
    p10 −8.1%、最差 −29.7%、>10% 概率 20%、>20% 7.5%；
  - B（进攻）：Top2、双周 —— 均值 +3.44%、>20% 概率 10%，左尾略厚；
  - C（防守）：长线 champ —— 胜率 62.5%、最差 −10.4%，但 >10% 仅 7.5%。
  - 短周期动量（20/60 日）在周频上全负；Top1 最差窗口 −50%，不采用。
  - 当前信号（2026-10-19）：煤炭/有色期货/能源化工各 1/3（三只全资源品，风险集中）。
- **池合理性审计 `[x]`（10-27）**：`scripts/portfolio/etf_pool_audit.py`，结论回写
  `reports/etf_pool_expand/README.md` §五。合规全过（50/50 满 3 年、60 日均额最低 0.9 亿）；
  **冗余是真问题**：相关 >0.95 有 13 对、有效独立资产仅 4.1/50、27.9% 持仓日持有同质双胞胎；
  静态去重 50→45 的指示性测试：年化 −0.3pp、**MDD −9pp、最差 2 月窗口 −11pp**。
  待开：**选标层动态去重**（120 日相关聚类、同簇留动量最高），预计是比继续扩池更高性价比的改进。
- **动态去重 `[x]`（10-27）**：`EtfRotationAdvanced._dedup` + `scripts/portfolio/etf_dedup_probe.py`，
  真源 `reports/etf_dedup/README.md`。规则：按动量贪心，与已选标的 120 日相关 ≥0.90 则跳过（无前视）。
  - champ：年化 12.37→**13.75%**、Sharpe 0.94→**1.08**、MDD 16.1→**14.0%**；OOS 1.04→**1.20**；
  - 比赛 A：13.85→**17.26%**、0.49→**0.66**、MDD 38.8→**26.3%**、2 月最差 −29.7→**−15.1%**；
  - 比赛 B：17.21→**20.25%**、0.56→**0.68**；阈值 0.85/0.90/0.95 结果接近。
  - 双胞胎共持日：champ 46.8→9.5%、A 27.9→5.8%。已接入 `etf_champ_signals` 两个 profile。
- **正式赛规则对齐（Top5 合规版）`[x]`（10-27）**：官方规则要求「配置不少于 5 只产品」，
  原 Top1/2/3 比赛方案全部作废（`reports/etf_contest/README.md` 已标注）。
  - 新增 `etf_contest_probe.py --specs k5`（11+6 个 ≥5 只变体）与
    `etf_champ_signals.py --profile contest5`；真源 `reports/etf_contest5/README.md`。
  - 选定 `k5_m12_1_w_dd90`（Top5/12-1/周频/等权/去重 0.90）：80 窗口均值 **+3.11%**、
    中位 +1.84%、胜率 **65%**、p10 −7.0%、最差 −18.5%；**最少持仓 5 只、0% 调仓日低于 5**。
  - 关键否决：champ 的拥挤度硬过滤在 9–10% 调仓日只剩 1–4 只（不合规）；
    核心-卫星权重在两种信号上方向相反（噪声），不采用。
  - 今日清单：按 HRP 权重（见下条）；日经 513520 溢价 +4.5% 不追，用煤炭 515220 补位。
  - ⚠️ 未闭环：**大赛指定 ETF 品种清单未确认**（用户 10-27 已确认"基本都有"，
    缺标再单独告知）；仍需用户提供白名单后按白名单重跑选标。
  - 追加 A/B「买 5 只同主题 ETF 凑数」（不去重 + 排名衰减 33/27/20/13/7）：中位
    +1.12%、最差 −27.2%、胜率 56%，全面劣于去重 0.90 + 等权（+1.84% / −18.5% / 65%）
    ⇒ **合规但亏钱**；该思路只用于"指定池缺标时的同主题补位"。
  - **权重方案对比 `[x]`（10-27，同选标只换 `weighting`）**：等权不是最优。
    中位排名：HRP+单只≤30% **+2.99%** > HRP +2.58% > inverse-vol +2.27% > ERC +2.21%
    > 等权 +1.84%；最差窗口 HRP −12.97% vs 等权 −18.45%；胜率 73.8% vs 65%。
    HRP 年化 23.4% / Sharpe 0.95（等权 14.9% / 0.59），代价是单次换手 0.271 vs 0.144、
    年化成本 3.8% vs 1.6%（收益已扣成本）。IS/OOS 中位 2.23/5.22 vs 1.06/4.38，两段都不输。
    **排名加权（5:4:3:2:1）判负**（中位 +1.47%、最差 −23.2%）：12-1 能选对方向、
    排不出 5 只之间两个月内的强弱次序。**30% 上限必加**（不加时今日能源化工单只 49.1%）。
    不利证据如实记：10 月起点的 6 个窗口 HRP 反而略差（−1.47% vs 等权 −0.99%）、
    全样本 MDD −37.7% vs 等权 −31.8%、换手翻倍。执行 = `--profile contest5h`，
    等权 `--profile contest5` 保留为低换手备选。真源 `reports/etf_contest5/README.md` §七。
  - **比赛 vs 长线同口径对比 `[x]`（10-27）**：`etf_contest_probe.py --specs lt` 把长线 champ
    放进同一 80 个 2 月窗口（`contest_lt_summary.csv`）。champ 中位 +1.31% / 最差 −8.21% /
    p10 −2.69% / Sharpe 1.08 / MDD −14.0%；比赛版 +2.99% / −15.34% / −7.64% / 0.95 / −37.7%。
    窗口收益相关 0.666，比赛版 58.8% 窗口胜出、配对均值 +1.85pp（重叠窗口 t 2.29 偏乐观）。
    左尾全部劣于 champ，右尾（>10% 20% vs 6.3%）全部优于 champ；10 月起窗口 champ +0.92%
    vs 比赛版 −1.47%。结论：比赛用短窗口版、长线用 champ，不要混用。真源同上 §八。
  - **长线定版剩余优化点 A1–A3 + B4–B5 `[x]`（10-27）**：`etf_rotation_p2.py --variants opt`
    跑 20 变体（`summary_opt.csv`，基准 champ_dd90 13.75%/1.08/−13.97%）。**A1 去重邻域**
    9 格 Sharpe 1.01–1.14（全部 > 不去重 0.94，散布 0.13）⇒ 去重真实、阈值不可调，保留
    0.90/120。**A2 Δshare×去重判负**（f67 −0.024、f80 +0.044 但 MDD +1.3pp、f90 −0.035，
    非单调）⇒ 去重已吃掉拥挤信息，旧 `champ_mom_dsh_f67` 14.43% 是去重前数字，线收口。
    **A3 缓冲区通过**：月换手 39.8%→32.1%（−19%）、成本 1.01→0.80%/年、Sharpe +0.019、
    MDD −0.27pp；15/40 与 10/30 逐位相同（去重后候选清单短、宽度不 binding）⇒ 零参数规则，
    **已切为 `champ` 生产 profile 默认（10-27 晚，用户拍板）**，信号板已重出：目标持仓
    510880 红利 → 510500 中证500，风险腿 55.8% + 防御腿 44.2%，0 告警。
    **B4 HRP**：MDD 13.97→10.82%（−3.15pp）、
    Calmar 0.98→1.31、换手 +5.5pp ⇒ 降 MDD 开关；hrp+30% 上限 15.58/1.15/11.36 属形状选择。
    **B5 防御腿**：债 70% 温和改善（−1.35pp MDD），金 70% 变差；阈值 −5% 与基准逐位相同
    （门槛不 binding），+5% 最好（16.07/1.18）但持仓 4.4 只、现金 40.6%，属轻度择时且只有
    两个点无平台 ⇒ 不采纳。多重检验：18 个 ΔSharpe 标准差 0.045、最大 2.2σ ⇒ 收益维度无
    可证实增量。真源 `reports/etf_rotation_p2/README.md` §十一。
  - **比赛指定池缺标处理 `[x]`（10-27）**：`etf_champ_signals.py` 新增 `--exclude`（把不可投
    标的 mask 置 False，选标与建仓都不可交易，无前视），并新增 `--max-premium`
    （选标层折溢价闸门，PIT）。实跑按"剔一层→看补位→再剔"不收敛（159981 → 日经 +4.5%
    → 纳指 +15.3% → 标普 +9.1% → 豆粕 +2.6%），改用闸门一次收敛。最终 5 只（100 万）：
    红利 30.0% / 煤炭 30.0% / 有色期货 30.0% / 通信 5.16% / 半导体设备 4.84%，
    0 告警、满仓、满足 ≥5 只 ≥5 笔。**结论：缺标不要手工找兄弟硬凑（159981 无同主题兄弟），
    要在剔除后的可投集合内重跑；补位连锁且会补进溢价品，溢价必须做选标前置闸门。**
    ⚠️ 该清单已偏离 §七 回测口径（全池、无闸门）⇒ §七 的中位 +2.99% 不能直接套用。
    **白名单已拿到（10-27，APP 两张长截图 OCR）**：截图内**正好 664 只**（行距恒定 128px、
    最大间隔 131px，无整行漏读；两图首尾重叠 1 行）。按**名称**重匹配（上一版只按代码精确匹配，
    结论错误）：50 池 **29 只代码命中 + 16 只可换同指数产品 + 仅 5 只无替代**
    （白酒 512690 / 30年国债 511090 / 转债 511380 / 豆粕 159985 / 能源化工 159981）。
    **上一版"主流宽基全被剔"作废**：白名单有沪深300 12 只、中证500 9 只、创业板 37 只、
    科创50 8 只、红利 27 只、A500 10 只，缺的是我们挑的那家发行人的那只。
    映射表 `reports/etf_contest5_wl/pool_map.csv`（回测用池内价格代理、下单用白名单代码）。
    按真实约束重跑（`--exclude` 5 只 + 溢价闸门）：煤炭 30.0% / 有色期货 27.0% /
    半导体设备 21.2% / 通信 19.0% / 有色金属 2.8%（下单码 159881），0 告警。
    **复核（80 个 2 月窗口，`reports/etf_contest_wl2/`）**：HRP+上限 中位 **+2.05%**、
    Sharpe **0.79**、最差 −15.3%；等权 +1.53%/0.49、inverse-vol +1.60%/0.61
    ⇒ HRP 优势仍在，不必切 IV。合规缺口 `min_holdings=4`、仅 **0.57%** 调仓日不足 5 只
    （旧版 4.58%）。仍需实现"不足 5 只用 511010 国债占位"。
    真源 `reports/etf_contest5/README.md` §九–§十一、`reports/etf_contest5_wl/whitelist_ocr.csv`。
    **30% 上限对照（10-27）**：无上限中位 +2.41%/Sharpe 0.89/最差 −13.8%，看似全面占优，
    但今日煤炭单只 **52.3%**、年成本 4.1%（有上限 3.3%）⇒ 上限保留的理由是控单一品种尾部，
    不是提高期望（上限对中位的影响已从全池 +0.41pp 翻转为 −0.36pp，属噪声）。
    **定版（用户拍板）**：比赛取**无上限**，`contest5h` 已改为无上限（临时控集中度用
    `--max-weight 0.30`）。信号板新增 `--max-weight`（传 0 = 不设上限）。
    **累计口径 → 集中版定版 `[x]`（10-27）**：用户指出「配置不少于 5 只产品」按**赛期内累计
    买入过的不同产品数**计，而非每个调仓日同时持有 ⇒ Top3 合法，主方案改集中版。新增
    `contest3` profile（Top3 + `mom12_1_crowdsoft` + 120 日 0.90 去重 + HRP 无上限 + 周一周频）。
    80 个 2 月窗口（合规 51 池 + `--exclude` 5 只，`reports/etf_contest5_opt/contest_conc_summary.csv`
    15 变体）：均值 **+4.44%** / 中位 **+3.99%** / p10 −8.16% / 最差 −23.82% / 胜率 66.3% /
    >10% 27.5% / <−10% 3.75%，优于 Top5 HRP（+3.86% / +2.41% / −5.89% / −13.8%）；Top1 右尾最高
    但 <−10% 20%、MDD −64% ⇒ 弃；Top2 均值最高但 MDD −46% ⇒ 不作默认。
    **优化点 1–5 结论**：① 国债/黄金占位 `fill_to/fill_codes` 代码保留，真实白名单仅 0.57%
    调仓日触发，累计口径下 `contest3` 不启用；② **拥挤软惩罚采用**（中位 +1.21pp、<−10% 减半
    7.5%→3.75%）；③ 6-1 动量**否决**（中位 ~0、胜率 <50%、MDD −44~−47%）；④ 白名单扩池
    237 只只利好 k=1（c1 10 月起 6 窗口全正、均值 +3.75%），k=3 中位 +1.47% 反低于 51 池、
    k=5 被稀释 ⇒ 默认仍用 51 池，扩池仅作补位池（`contest_conc_wl_summary.csv` 11 变体）；
    ⑤ 周一优于周五（c5 HRP 中位 +2.41% vs +1.69%）。
    **修执行口径**：引擎不允许在最后一根 bar 建仓（`i < n-1`），周频最新日（周一）恰是调仓日时
    旧信号板会退回上一周权重 ⇒ `run_variant` 返回 strategy，信号板在「最新日 == 调仓日」时直接
    用当日因子/掩码取权重。今日（2026-10-26，100 万）：通信 77.55% / 科创芯片 12.32% /
    半导体设备 10.12%，0 告警、满仓；三者近 60 日相关 0.68/0.84/0.93（120 日仅 0.26/0.14/0.26，
    去重窗口没挡住相关性抬升）⇒ 100% 科技主题，`--max-weight 0.50` 备选 50.0/27.5/22.5。
    ⚠️ 残余：平台若按"每时点 ≥5 只"校验则 Top3 违规 ⇒ 建仓先买 5 只（3 核心 + 2×1% 卫星）。
    **正确性核验 `[x]`（10-27 用户追问）**：实盘信号板（`build_panels+build_extra_panels`）与
    回测探针（`build_contest_panels`）的 `mom12_1 / mom12_1_crowdsoft / crowd / mask` 在
    2136×50 个格子上**逐格相同**，最近 5 个周频调仓日 Top10 排序完全一致；同 spec 重跑数值
    可复现（+4.44%/+3.99%）。PIT 复核：动量 shift(21)/shift(252)、crowd=份额 `ann_date<=t`
    的 250 日滚动分位、流动性 60 日均额、溢价 20 日均值，均为后视。
    **稳健性补测**（`reports/etf_contest5_opt/robust/`，17 变体）：去重窗口 120→60 日中位
    +3.81%（−0.18pp，噪声内）；单只上限 50% 中位 **+1.52%**（−2.5pp，代价大）⇒ 维持
    无上限 + 120 日去重，参数锁定不再改。
    **定版落地 `[x]`（10-27 整理）**：新增 `scripts/portfolio/etf_contest_spec.py` 作为定版参数
    唯一真源（`FINAL` + `EXCLUDED_5`），回测探针与实盘信号板共同引用，`tests/
    test_etf_contest_spec.py` 断言两侧一致（4 项全过）；比赛版 `--exclude` 改为自动排除 5 只
    非池品种；作废的初赛 profile（20 日动量/Top3）已从代码与文档删除，
    `reports/etf_contest/` 旧目录已按治理流程回收（快照 md5 + 引用同步，见
    `reports/README.md` §四 2026-10-27 记录），需要时可 `--specs base` 复跑重建。定版一页纸见
    `reports/etf_contest5/README.md` §十三（参数表 / 代码组织 / 每周 SOP / 冻结声明）。
    真源 `reports/etf_contest5/README.md` §十二–§十三。
    **定版升级 v2 `[x]`（10-27 夜）**：冻结后用户追问"有没有反转因子"⇒ 个股级短期反转在
    ETF 上不成立（篮子抵消特质反转），但"20 日涨幅过热"是尖峰动量特征，可只做减法。
    补测三档（`--specs rev`，`reports/etf_contest5_opt/rev/`）：剔除 mom20 >10% 为
    **均值 +5.59% / 中位 +3.76% / p10 −3.64% / 最差 −12.34% / 胜率 72.5% / >10% 31.3% /
    <−10% 2.50% / Sharpe 1.28 / MDD −25.4%**，对 v1（+4.44/3.99/−8.16/−23.82/66.3/3.75/
    0.79/−38.5）全面改善尾部与均值、中位仅 −0.23pp；>15%/>20% 逐档变差（**单调**，非单点挖矿）；
    分年 2020/21/22/23/26 改善、2024/25 强趋势年变差。代价：换手 36%→45%、成本拖累
    5.2%→9.4%/年（已含在净收益内）。**已采纳为定版 v2**：`FINAL.signal=mom12_1_crowdsoft_pre10`，
    探针定版槽更名 `c3_final`；今日三只 20 日涨幅 −30/−30/−25% 全在阈值下，**当日权重不变**
    （77.55/12.32/10.12）。真源 `reports/etf_contest5/README.md` §13.9。


---

## 建议推进顺序（2026-10-19 刷新；研报线见 RESEARCH_TODO 第六节）

> **2026-10-19 增量优先序（第三轮刷新）**：A1/A2 在 882 口径全链收口（A1
> h60 第二策略 Sharpe 1.028/换手 17.5%，融合对照确认**并行不融合**、buffer
> 网格平缓维持 15/40；A2 主支撑 = stack 融合 +0.154，overlay 经 walk-forward
→ **批次 13 已收官（10-21，本机）：三臂全败，
  A1/A2 线关闭**（overlay −0.03 / stack −0.04 / h60 三条件全败；882 正
  结果未迁移 920 数据态，详见 §六各条目 ❌ 标记）。
> B1 beta hedge / B2 SDK 字段审计顺位后移；C1 不投入。旧 1–8 项保留作历史，
> 其中「定版切生产链」仍是工程侧唯一高优先，与本研究线并行。

> **2026-10-26 新增 D1（ETF 轮动）**：跨资产线的首个正结果——P0 扩池 41 只 + 四表回补、
> P1 基线 `dual_ivvt15_m5` Sharpe 0.62 / MDD 19.01%（旧 0.26 / 38.4%），稳健网格 7/7 过线。
> 与股票选股线**不抢资源**（分钟级、本机跑）；下一步先做幸存者偏差量化 + PCF 存档，
> 再开 P2 的 S1 份额流 / S2 持仓映射 α。

> **2026-10-10 战略结论（四条路全部实测）**：量价线（上游 ≥36 即平台、≥200 逐位相同、
> 组合层 +0.63pp）、风险维度、模型层超参、PEAD/SUE 新信息源 **四条全部收口**，
> 缺口因**同一机理**：$S=\sqrt{S_1^2+S_2^2}$ ⇒ 自身 IR 只有 0.11 的正交信号最多值 +1.2%；
> **已探明的「低相关」不能替代「足够强」**。⇒ **新数据源准入判据 = 自身可交易 IR ≥ 0.3**
> （见 §一 10-10 条目、`reports/four_paths/README.md`）。
> **⚠️ 10-13 更新：唯一候选「分析师一致预期修正」已尽调完毕 → 数据不可行**
> （免费源 EPS 预测历史仅 ~2.5 年且每 FY 窗口只有当年，见上「已完成归档」首条）。
> ⇒ **本阶段「新信息源」这条腿正式封底**：剩下的只有工程侧确定性收益
> （出榜链切生产）与已有资产的低成本收尾（`txt_*` / `pap_*` 从未评估）。
> 不改变「定版切生产链」的工程优先级（确定性收益，与本研究线并行不冲突）。

1. ~~**分析师一致预期修正线尽调**~~ ✅ **10-13 完成 → 不立项**（数据见上）；
   替代动作 = 评估已有 `txt_*` 5 因子（零数据成本，见下第 4 项）。
2. **定版口径切换生产出榜链**（§一，当前唯一高优先——特征源 fundind +
   h1020⊕s0.5 + buffer(15/40) 三处切换 + 双链并行对照一周）
3. ~~小工程收尾：e2e 族历史报告年化口径重跑 + 方案 B 缺面板对齐（4 列）+
   好机器批次 11（cheap_lever 胜者 920 复验）~~ ✅ **10-17 双收官**：批次 7
   见 §一归档条目；批次 11 复验**环境分歧后切换搁置**（62 因子 5/5 通过 →
   72 因子 3/5 撤回，见 `reports/cheap_lever_scan/README.md`，最终裁决挂
   批次 12）；本机 panels_neu 已补 9 张 P5（div_yield_ttm 卡 dividend 表
   仅 520 只——数据缺口，生产机可 SDK 补拉后重建）；方案 B 的 4 列对齐仍开放。
4. ~~论文族 `pap_*` 21 + 情绪族 `txt_*` 5 **收尾评估**~~ ✅ **10-13 完成 → 全线封底**
   （详见下方「已完成归档」与 `reports/text_factor_probe/README.md`）：`txt_*` 5 个自身
   可交易 IR **0.008~0.105 ≪ 0.3**、独立 Sharpe 0.47~0.57 < 主 0.631；融合层出现的
   +0.05~0.12 Sharpe **经安慰剂证伪**（= 覆盖率掩码/市值暴露）；`pap_*` 无一条过
   「正交+够强」；106 项 BH-FDR 仅 1 条存活且为负。**研报线（论文族+情绪族）正式封底。**
5. ~~趋势 MA200 overlay 在 920 定版复验~~ ✅ **已被择时线收口覆盖（10-15）**：
   7 条因果规则（含 MA200）Δ年化全负（`reports/timing_probe/`），按年化判据直接否；
   「低回撤版」按需启用 = MA20 半仓（−3.9pp 年化换 MDD 腰斩，保险不是 alpha）
6. 分钟频扩容吞吐探针 → 视成本决定 all_a 分钟物化与第三层扩原料
   （资源墙；华泰3128 依赖此线，RESEARCH_TODO §六）
7. 中小池组件线（可选，非主线）：tabicl 头部内加权跨池复核 +
   gbdt+tabicl 对集成组合层验证
8. 生产级执行（实盘对接、实时行情）

---

## 附录：历史交付记录（压缩版）

> 原 README「待建 / 已知缺口」节 + 完整交付细节压缩至此。**待办**请看正文 §一~§四。

**2026-08-25 批**：组合级风险分解（Euler + VaR/CVaR 成分）；CPCV h=1 无偏评估；
端到端选股工作流固化（现结论：2024-01~2026-08 月频等权 top50 +16.2% 跑输全池基准
+30.8%——信号强度不足以支撑集中持仓）；投资收益报告（中性化后等权 +27.0%/Sharpe 0.66，
仍跑输沪深300 指数 +36.0%）（⚠️ 244 年化口径——252 换算 +1~2pp/×1.016 见
`reports/e2e_backtest/README.md`，批次 7）；回测引擎三项修复（结算-调仓顺序前视一天、h>1 区间结算、
avg_turnover 稀释）+ 收益面板口径守卫（h=1 传 shift(-h) 直接报错）；风格中性化有效 /
周频被证伪（optimize_e2e 四组对比）；模型增强组合固化配置（`run_model_portfolio`）；
LGBRankerPredictor 修复（负 IC −0.055 → +0.207）。

**2026-09-01 批（全A 多年度滚动训练）**：全A数据回补（1098 万行 × 5549 股）；
`all_a_2018_2026` 数据集（798 因子 × 2471 日 × 5549 股 float32 ~24GB）；120 组合网格
主链（四阶段断点续跑 + 幽灵股守卫）；核心结论：全A截面 OOS IC 显著强于 HS300
（gbdt h1 全期 IC=0.108）、h1+M/W+gbdt+Top10% raw 一致稳健解、日频被成本全灭、
风格中性化在全A上稀释收益（与 HS300 相反）。

**2026-09-07~09 批**：幸存者偏差消除（退市回补 238 只，超额回落 ~2pp）；ens_h1h5 集成
成为新头部（13.8%）；DPP 特征选择 + 基本面保留席位；正交化三臂对比（全正交 +1.6pp
> 混合 ≈ 不正交，增量几乎全部来自量价因子）；ortho 臂集成复现（+13.3%，stage_ensemble
回测窗口 bug 修复）；仓位 overlay 网格（趋势 MA200 减半 MaxDD，按 Calmar 取舍）；
数据更新守卫（盘中半拉 K 线防污染）。

**关键工程教训（全量见 `.workbuddy/memory/`）**：`build_panel` 复权只作用 OHLC
（vwap 用 amount/volume 须先复权）；SDK 大清单单查会挂死（分批+重试）；float64 巨值
astype(float32) 溢出成 inf；状态表缺预测日数据使因子全 NaN；`os.replace` 在 Windows
遇共享读句柄报 WinError 5（写盘须原子优先+占用降级）。


---

<!-- ═══════════════════════ 册Ⅱ 研读复现线（原 RESEARCH_TODO.md 全文快照，逐字保留） ═══════════════════════ -->

# YuriQuant 研报研读与复现待办

> 工程/研究待办见 [`TODO.md`](TODO.md)；架构与用法见 [`README.md`](README.md)。
> **细节真源**：研读报告全在 `reports/docs/research_notes/`（本文件只留指针与待办）；
> 逐日日志 `.workbuddy/memory/`。

> 2026-09-10 基于 `E:\研报` 全库（85 个 PDF，去重后约 60 篇）筛选建立；09-16 按代码证据
> 核实复现状态；09-18/09-20 五批推进（P1 全部处置 + stage2 设计定稿 + Phase 0/1 落地 +
> Phase 2 管线冒烟）；**09-21 增量**：panels_neu 补齐至 920 → 主实验数字脱钩，
> 治本口径重跑定为最高优先级（见 TODO §一，先于本文件全部长实验）。
> **09-23 增量**：东吴《LLM-MCTS》+ 东方《QuantaAlpha》两篇 LLM 因子挖掘研报归档
> （`reports/docs/research_notes/东吴0623_*`、`东方0407_*`）；东吴列为 **P0 试点**
> （同题四引擎对照，Phase 0 设计定稿见笔记，代码待实现），QuantaAlpha 仅抽 5 个机制
> 进 `llm_pool` 增强清单、**不复现**。
> **09-24 增量**：东吴 LLM-MCTS 代码落地（见 §〇）；兴业 0526《智能体与深度学习双范式》
> （海外文献综述 189）**速读归档不复现**——六篇被引文献三篇已有更深对应物（东吴/东方/RD-Agent），
> 唯一增量 = Hu(2026)「市场结构决定范式选择」框架（A 股结构判定 DL 占优，为增量枯竭后引入
> 非线性表达留文献动机）；笔记 `reports/docs/research_notes/兴业0526_速读_*.md`。
> **10-06 增量**：知乎双篇转读（用户推送，**二手信源均不复现**）——①《谈量化交易中被严重
> 低估的执行模块》：整手化已有（`optimize/signals.py`）、拆单不适用（月频+T+1 开盘），
> 仅 MWU 在线融合记为"合成第八法"存档备选（月频小 T + phase2 集中度教训，预期有限不立项）；
> ②BigQuant 转述华泰精工 2026-04-01《高频特征的参数化，分钟级可解释因子挖掘框架》：框架层
> 与国金24 线同题无增量，**真增量 = 矿池工程化四件套（计算下推/逐因子串行/矿池持久化+
> 断点续挖增量扩表）→ 服务 §二分钟扩容资源墙**（DuckDB/polars lazy scan 年分区 parquet
> 即仓库栈内等价物；"16GB 可行"再证先跑 `_probe_minute_throughput.py` 校准）；成交量回归
> 截距记为 `im_` 第 43 特征候选（笔数自相关因分钟面板无笔数字段不可算）。笔记
> `reports/docs/research_notes/知乎双篇_研读_执行模块与华泰分钟因子框架_2026-10-06.md`。
> **10-16 增量**：天风《BetaAgent：交易风险因子挖掘和评估系统》研读归档，**不复现**——
> LLM 多 Agent 挖掘线第 4 篇同族（东吴/东方/AI97/RD-Agent 之后），靶子转向**风险因子**
> （分钟量价+分钟资金流构造 10 个短期交易类风险因子，作 Barra 之外的 0.1σ 风险约束）。
> 数据墙 = 全 A 分钟量价已在资源墙外、分钟资金流完全没有；真增量 = **六维风险因子验收
> 框架**（ΔR² 增量解释力 + 风险型 IC 形态「|IC| 高 + IC 均值低」等），挂 Barra 风险模型
> 单独立项线备选。笔记 `reports/docs/research_notes/天风0830_研读_*.md`。
> **10-08/09 增量**：好机器批次 1–10 全部出数收官，研报复现线大面积闭环——
> AI97 正式三臂 PBO=0.704 + 100k 步归因（**真实失效，归档**）；东吴 LLM-MCTS
> all-A 正式跑（750 候选收编 14、PBO=0.039，**席位 4/14 唯一突破**）但组合层
> +0.16pp 噪声级不采纳（归档）；国金24 残余③④⑤ 闭环（raw+im 定版形态
> OOS |IC| 0.117、114 因子入库 PBO=0.114）；银河 0706 stage2 全档 24 臂终判
> （**RL/QP 在 zz1000 无稳健超额，RL 线整体归档**）；mf10 T 扫描出数（T=3 最优、
> 不显著）。数字真源：`reports/batch2385_结果速览.md`、`reports/batch1_920_收尾对照表.md`。
> **长实验一律不跑、写成显式待办，执行环境 = 另一台更强的机器**。

> **核心判断：全库 87 篇中 2025–2026 年仅 14 篇，恰好覆盖项目全部短板方向。**
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
| 国金24 GFlowNet+AlphaEval 分钟频 | `[x]` 主体 + 残余③④⑤闭环（10-05/06） | AlphaEval 漏斗 / RRE / DPP / 42 `im_*` 接 GP / e2e 三臂 / GFlowNet 接 im；正式三臂 ×3 seed：raw+im 0.117 > im 0.109 > raw 0.099（臂间差>种子噪声）=**该线定版形态（600 iters 无门槛）**；114 gf_* 入库 PBO=0.114；**残余② 全A 分钟扩容（资源墙）仍开放** |
| 华泰 AI97 大模型+RL | `[x]` 完成，**负结果归档（09-25/26）** | `factor/rl/` 四模块 + `run_alphapool_ppo` + LLM 池真实联网验证；正式三臂 ×3 seed 臂间差异 < seed 噪声、CSCV **PBO=0.704**、最优臂 DSR=0.25 不显著；100k 步归因=**真实样本外失效非未收敛**——不构成可引用 alpha 源（§一） |
| 东吴0623 LLM-MCTS 因子迭代 | `[x]` 完成，**席位突破/组合无增量，归档（10-08/09）** | `factor/mcts/` 五模块 + `run_llm_mcts.py` 四臂 runner（mcts/llm_oneshot/gp/gflownet）+ 23 用例；**all-A 正式跑**：750 候选收编 14、PBO=0.039、OOS≥IS；入库 llm_* 后 v4 **席位 4/14（唯一产出超门槛因子的引擎）**，组合层 +0.16pp 噪声级不采纳——量价信息在 DPP 下与既有 alpha 高度重叠（§一） |
| 东方0407 QuantaAlpha | `[x]` 研读归档，**不复现** | 论文主结果有测试集泄露（东方自述）、修正复现仅 21 因子 ICIR 偏低；5 机制抽取进 `llm_pool` 增强清单（TODO §二） |
| 微软 RD-Agent(Q)（NeurIPS 2025） | `[x]` 研读归档，**不复现不替换** | 量化 R&D 自动化多 Agent 框架（LLM 提假设→Co-STEER 写码→Qlib 回测→反馈，bandit 选 factor/model 方向）；**同题不同栈**（AI97/东吴 MCTS 同题）；机制抽取 8 条：数值去重/失败换向/JSON 纪律进 MCTS Phase 0 设计，`fin_factor_report` 自动 vs 手工对照列 P2（挂 920+好机器 WSL2）。笔记 `reports/docs/research_notes/微软RD-Agent_研读_*.md` |
| 兴业 0526 双范式综述（海外文献 189） | `[x]` 速读归档，**不复现** | 海外文献综述（Agentic AI 闭环 / 东吴 MCTS / QuantaAlpha / PPO 加权 / 残差分布预测 / E2EAI 六篇转述）；三篇已有更深对应物；唯一增量 = Hu(2026) 市场结构框架（A 股判定 DL 占优）；笔记 `reports/docs/research_notes/兴业0526_速读_*.md` |
| 天风 0830 BetaAgent（10-16 转读） | `[x]` 研读归档，**不复现** | LLM 多 Agent 挖掘第 4 篇同族，靶子=**风险因子**（分钟量价+分钟资金流 → 10 个短期交易类风险因子，Barra 外 0.1σ 约束使三基准指增超额 MDD −1.1~−2.7pp，但 300/500 超额年化是降的）；数据墙（分钟资金流全缺）+ 同族引擎已收口；机制抽取 = **六维验收框架**（ΔR² 增量 + 风险型 IC「\|IC\| 高 + IC 均值低」）挂 Barra 风险模型线；笔记 `reports/docs/research_notes/天风0830_研读_*.md` |
| 华泰 AI106 日内量价 NN 挖掘（10-20 转读） | `[x]` 研读归档，**不复现**；机制② 抱团动量已试**判负**（10-21） | Value–Weight 框架（1 分钟 240 根 → 50 日频因子；ZZ1000 指增超额 17.7→19.3%）；**三道墙**：数据（1 分钟×全A×2017，比华泰3128 墙还高）/ 算力（本机无 GPU）/ 优先级（排全A 分钟扩容后）；**机制抽取两条**：① 分钟扩容落地后先做「日内分段重加权」探针（对 `im_*` 等权版对照，半天级，衔接时段切片）；② ~~抱团动量因子可日频仿制~~ → **10-21 已试，日频截面层判负**：门控（行业成交额 HHI 252d 分位 >0.8 × 行业 20 日动量 >0）对动量有效性**无区分力**——分组 rank IC 差 `t=−0.91/−0.33`（h=5/20），mom_60/120 窗口同样 `|t|<0.6`；`mom×(1+gate)` 与 mom_20 截面 rank 相关 **0.992**（门控仅覆盖 8.4% 的「日×行业」，行业级乘数不改截面排序）；交互式 `(2g−1)·mom` 的 IC +0.05 只是 mom_20 的镜像翻转。⇒ 三种形式全退化为 mom_20，**无选股增量**；原文该因子的收益来自组合层 overlay + 五路径周频调仓，两者不可拆分归因，不复现；笔记 `reports/docs/research_notes/华泰AI106_研读_*.md`（§二 已补验证） |
| SSRN 7494298 Label Alchemy（10-04 转读） | `[x]` 研读归档，**转译实验完成** | 标签工程三维度（Location/Scale/Shape）；美股月频固定 GBDT：raw Sharpe 0.68 → 高斯化排名 1.69、行业 z 1.56、vol_scaled 0.36/0.66 **更差**；标签解释力 0.87 vs 模型 0.04；**10-04 三臂实验**：gauss_rank 零增量（主线 rank 已在最优族）、ir 判负（vol_scaled 预警应验）——见 §二 |
| 易方达 ICML 2026 Label Horizon Paradox（10-04 转读） | `[x]` 研读归档，**转译实验完成** | 最优训练标签 horizon δ*≠目标 Δ（日频隔夜场景 δ*≈close-to-open）；BLO 双层优化自动学 δ*；**10-04 co 隔夜臂判负**（−1.0pp/回撤换手双恶化；分钟特征+深度模型的机制不自迁移到日线 GBDT）——见 §二；BLO 留作 horizon 混合线的自动化备选 |
| 华泰 AI39 组合优化实证 | `[x]` 口径核对 | `reports/口径核对_AI39_多因子10.md`；最大差异=主动 vs 绝对权重空间；3 项低成本对齐已落地 |
| 华泰多因子10 合成口径 | `[x]` + 补齐 + T 扫描出数（10 月） | 两主力方法（`synthesize_ic_ir_max`/`synthesize_ic_max`）+ 半衰加权 + T 扫描；全量已跑：两法均 T=3 最优、T 增大单调变差，NW t≤1.4 不显著（§一） |
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

> ~~920 治本口径重跑优先于本节全部条目~~ **已完成（09-24~10-02，定版落定）**；本节
> 大部分条目已随批次收官闭环（各节带 ✅ 终态）。🥉 区观点注入三臂与
> CVaR 组合臂**已于 10-21 批次 12④⑤ 落判（均负结果归档**，证据
> `reports/views_arm_920/` + `reports/cvar_arm_920/`）；真正待推进仅剩
> 国金24 残余②（全A 分钟扩容资源墙）；另有一条**赛后探针**（ETF 轮动盲区/
> 出场加速组，判据已预注册，见本节末条目，比赛结束前只跑数不裁决）。

### 🥇 国金24 残余（③④⑤ 已闭环，仅剩② 资源墙）

- **② 全A 分钟数据**：唯一分钟缓存是 `min5_hs300.parquet`；扩到全A 才是真正的资源墙，
  `scripts/oneoff/_probe_minute_throughput.py` 为外推成本而写（需真实 SDK 跑一次）。
  （知乎华泰篇转译的矿池工程化四件套 = 此项的实现路径指针，见头部 10-06 增量。）
- [x] **③ RRE 门槛开启（10-05 闭环）**：raw+im 臂 + `--min-autocorr 0.3` × 3 seed，
  OOS |IC| 均值 0.113 vs 无门槛 0.117（种子间方差反而增大）——|IC| 口径中性略负；
  「免费降换手」价值待组合层复核，**维持 raw+im 无门槛为主臂**。
- [x] **④ 挖掘预算饱和（10-06 闭环）**：600→1200 iters 同种子 0.1231→0.1189，
  加倍预算反而 −0.004（种子噪声内）——600 iters 已饱和，「预算不足」假设被否。
- [x] **⑤ 正式实验（10-06 收官）**：三臂 ×3 seed（600 iters × batch 12）——
  **raw+im 0.117 > im 0.109 > raw 0.099**，臂间差（0.018）> 种子噪声（~0.008），
  国金24「分钟特征增强」方向成立；114 个 gf_* 因子入库 hs300_2022_2025 库
  （33 个 |t_nw|>2），**PBO=0.114 族层稳健**（vs AI97 0.704）、最优单因子
  DSR=0.040（族用稳、单用弱）——定位分散器/家族池，与慢信号同定位。
  **该线定版形态 = raw+im 无门槛 600 iters**。

### 🥇 AI97 正式实验（✅ 09-25 出数、09-26 步长归因——负结果归档）

**结果**：九臂（none/random/llm × 3 seed）测试段复合 IC 臂间差异 < seed 噪声、
训练/测试段符号普遍翻转；CSCV **PBO=0.704**（12870 组合）、最优臂 none_s2
DSR=0.25 不显著 → 10000 步下不构成可引用的稳定 alpha 源。
**步长归因（09-26）**：100k 步 × none/seed0——训练段复合 IC −0.0499→+0.0190、
测试段 +0.0677→−0.0434，10 倍预算下符号翻转依旧 → **真实样本外失效而非未收敛**，
该线正式归档；若重启需换池/构建机制而非加预算。
产物：`reports/alphapool_ppo_real_mlp_*.json`、`reports/ai97_pbo/`。

（命令与复现边界存档如下——引用数字必须注明七条边界。）

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

### 🥈 东吴 LLM-MCTS Phase 0 试点（✅ 10-08/09 全量收官——席位突破、组合无增量，归档）

> 设计真源：`reports/docs/research_notes/东吴0623_研读_LLM_MCTS因子迭代框架与Phase0设计.md`
> （reward 六项公式 / UCT / virtual expansion / 参数表 / 验收口径 / 复现边界全在笔记里）。

**终审（10-08/09）**：正式跑落在 **all-A 域**（all_a_2019_2026，IS 2019-2023 /
OOS 2024-2026-09-22，h=5，3 seed × 50 iters × 5 variants）——750 候选收编 14
（结构去重 699 + 数值去重 37）、**PBO=0.039 族层稳健**、OOS≥IS 泛化一致
（top：ts_corr_20(close,volume) OOS −0.058 超 top-150 门槛）。`build_alla_llmmcts_factors`
入库 llm_* 后 v4 席位 **4/14 入池（五个挖掘引擎中唯一产出超门槛因子者）**；
但组合层 v4 17.17%/0.746 vs 同环境 v3 17.01% → **+0.16pp 噪声级，不采纳**——
量价族信息在 DPP 去冗余下与既有 alpha 高度重叠。**「席位突破、组合无增量」归档**；
跨引擎对照由既有线结论替代（GP/GFlowNet 饱和、AlphaPool 失效），hs300 29 Seed
四臂协议未按原样执行，如实记录。证据 `reports/llm_mcts_phase0/`、
`reports/batch2385_结果速览.md` P-A 节 + 终审节。

**立项理由**：MCTS 用 UCT 把有限回测预算集中到高潜分支，直击「GP 盲目试错、因子同质化、
单 IC 选股不管换手」三个已知短板；六项 reward（0.40 IC+0.30 IR+0.02 覆盖+0.10 换手
+0.10 多样性+0.08 过拟合）全部可用项目现成件计算。**立项核心问题 = 同题四臂对照
（MCTS vs GP vs GFlowNet vs LLM-one-shot）有无增量**——不做对照则价值不成立。

**实现范围（本机可写，~600-800 行）**：`factor/mcts/`（tree/reward/engine）+
`scripts/factors/run_llm_mcts.py`；复用 `llm_pool.py`（proposer+解析+4 校验）、
`alphapool_env`（算子/窗口/字段）、`alpha158.py`（29 Seed，窗口统一 20）、既有周度
RankIC/换手/相关性统计。先写 1 Seed × 3 iters 冒烟（对齐国金24 教训：噪声带可超臂间差异）。
**吸收微软 RD-Agent(Q) 三机制**（09-24 研读，笔记 `reports/docs/research_notes/微软RD-Agent_研读_*.md` §三）：
① 评测前"与当前库逐日 IC 相关 ≥0.99 数值去重"层（AST 去重之外的数值防线）；
④ 连续 N 轮无 SOTA 改进 → prompt 层重置复杂度从简单公式起步；⑦ 因子公式禁省略号/占位文本（JSON 纪律）。

**实验口径**：hs300_2015_2026 主 + 全A 920 平行臂；IS 2016-2023 / OOS 2024-2026-08-21；
**h=5 对齐东吴**（非项目默认 h=1）；搜索 reward → top50 全量复评两段结构（不抽样）；
**IS 选型、OOS 只验证**。验收双口径：正式选择 OOS 提升比例对标 **65.5%**、候选池诊断率
对标 **75.4%**；四臂比双优公式数/去重数/相关性/token 成本；接 `stats/pbo.py`。

**执行**（~~代码就绪 + `DEEPSEEK_API_KEY` 后挂好机器~~ **已执行（10-08，正式跑口径变更见上节横幅）**；命令存档备复现，≈1450 次 LLM 扩展/臂，单臂评测可并行）：

```bash
# 命令已定稿（09-24，命令真源 = GOOD_MACHINE_TASKS 批次 9.2）；主臂：
python -u scripts/factors/run_llm_mcts.py --panel hs300_2015_2026 \
  --is 2016-01-01:2023-12-31 --oos 2024-01-01:2026-08-21 --horizon 5 \
  --arm mcts --iterations 10 --variants 5 --max-depth 3 --seeds-n 29 \
  --llm openai --out reports/llm_mcts_phase0
# 对照臂 --arm llm_oneshot/gp/gflownet（不需 key）；链路自检 --panel mock --smoke
```

**复现边界（引用必须注明）**：① 东吴未披露模型→项目 `deepseek-flash`（同族不同版本）；
② prompt 自拟（东吴/论文均未给全文）；③ 股票池不同，**指标绝对值不可跨报告比、只比同池四臂**；
④ h=5 系对齐东吴非生产口径；⑤ diversity 项搜索期压相关候选、复评期放开（东吴口径）；
⑥ 东吴 Stage1 分层抽样纯为降本，项目不抽。**明确跳过高频部分**（min5_hs300 资源墙，
同 TODO §二分钟频扩容项）。注意：若 `llm_oneshot` 臂已接近 mcts，结论即「LLM 语义生成
为主、MCTS 控制增益有限」——否定结果同样有信息量，如实报。

### 🥈 多因子10 T 扫描全量（✅ 已出数——T=3 最优、不显著）

`reports/mf10_t_scan/`（hs300_2022_2025，h10，top8，T∈{3,6,9,12,24}）：
两法均 **T=3 最优、T 增大单调变差**（ic_ir_max OOS IC 0.0222→0.0182、
ic_max 0.0215→0.0147）；但 NW t ≤1.4 不显著、绝对量级弱——该数据集/口径下
合成窗口选择不构成对现 pipeline 多窗口方案的升级依据，负结果存档。
（caveat 修复史：09-24 `synthesize_ic_max` V 决策时点前视已修 + 回归锚。）

### 🥇 链路优化线：预测 → 组合 → 执行（✅ 10-13/14 收口，10-17 批次 11 复验环境分歧→切换搁置；QP/集中度/aim/HRP 全关）

设计真源 `reports/docs/design/链路优化调查_预测组合执行_2026-10-13.md`（四部 17 节，
含两次结论自我推翻的完整记录）。三天内完成「执行层真实摩擦 + 组合优化器」两个
空白维的全部实测，**全部负结果 + 两个待复验正结果**：

- **逐股冲击成本模型落地**（`backtest/costs.py::ImpactCostModel`，η·σ·√(Q/ADV)）：
  1e8 资金下冲击 ≈ 固定 10bp（现行假设大致正确）；但**成本是 AUM 强函数**
  （1e8→1e9 掉 −4.55pp）。容量表：buffer 形态容量 >1e9（`reports/capacity_table/`）。
- **QP 全家族关闭**：公平成本口径下裸 QP 输等权 −10~−12pp（八成来自持仓只数 ×
  平方根律，§13.1 机制拆解）；成本感知惩罚自救 +11.7pp 后 ≈ 打平 Top10% 但仍输
  生产形态 buffer(15/40) −2.7pp。**修复途中揪出 2 个阻断级缺陷**：引擎
  `_apply_executable_mask` 尘埃放大 bug + `mvo` 量级失配（`alpha_scale` 修）+
  协方差暖机截断（QP 空仓 59 日，−3.3pp/年）——三处全修 + 回归测试锁死。
- **集中化裁决：全否**（production 口径锚点对拍）：TopK45/100 × equal/inv_vol/erc/buf
  最好 16.02/0.645 < 现行 buffer(15/40) 16.46/0.685；研究口径 +1.9~2.3pp 的优势
  被 T+1 open + 换手成本吃光。**buffer(15/40) 参数在两窗口均为峰值，不动**。
- **前融合多任务否 / back_union 否**（三窗口 2 胜 1 负净 ≈0，
  `reports/multihorizon_fusion/`）；**aim 关闭**（GP 偏调整机制已被 buffer 带宽
  工程化，`optimize/aim.py` 留作工具）；**HRP/risk_parity 书面关闭**（ERC≡inv_vol
  实测逐位相同 + 同族集中度机制；重开条件 = CNE5 级风险模型落地）。
- **微调四维只剩两个真胜者**（882 口径 5/5 窗口一致，`reports/cheap_lever_scan/`）：
  **buf(0.08/0.40) +0.78pp**、**h1:h5=0.25:0.75 +0.31pp**、叠加 **+0.85pp**
  （17.31%/0.707，换手还降到 31.8%）——⚠️ **10-17 批次 11 于 920 定版信号
  复验环境分歧**：62 因子慢信号环境 5/5 窗 +1.47pp 通过，补齐 P5 后 72 因子
  环境仅 3/5 窗（近两窗转负）⇒ **生产切换搁置，维持 dingban**；「h1 减半」
  操作迁移两环境均成立、字面迁移两环境均否。最终裁决挂批次 12（好机器
  自家口径重跑）。
- **口径三纪律沉淀**（代码+测试已锁）：① Sharpe 一律生产几何口径（算术恒高
  0.06~0.13）；② 涉 Σ 的探针收益面板不得截断到回测窗口（暖机教训）；
  ③ 探针基线必须含生产形态 buffer 臂（`cost_penalty_probe` 默认已改）。
- **✅ 拍板（由莉酱 2026-10-15）：报数 = 方案②**（`--mask-lag 1` + T+1 open 成交）。
  现值即此（管线默认 1、定版链 open=T+1）⇒ **零代码改动、不触发全链重跑**；
  方案①（lag 0 + T 收盘）仅作研究上限对照，**数字不得与报数并列**；
  引用历史混搭口径产物必须带口径标签。提频实验（周频+）开工前必读
  `reports/docs/consistency_checks/口径核对_掩码取样日错配.md`（月频噪声级、周频 +3.76pp）。

### 🥉 补短板低成本项（半天级，本机穿插）

- [x] **信号-市值漂移监控升格进生产 IC 监控**（09-23 完成，commit 612cf59）：
  `monitoring/production_ic.py` 新列 `signal_mktcap_spearman`（逐日
  Spearman(score, cov_size)，复用 daily_rank_ic，零新依赖）+ summarize
  mktcap_drift 全期/近窗/逐年 + CLI 摘要行。判据 = 逐年恶化趋势（raw vs
  oneoff 中性化面板口径差见 docstring）。**验证**：L1 探针 max|Δ|=1.1e-16
  （2103 日）；L2 逐年趋势与 882 红牌 Spearman=1.000；端到端 2111 行落盘，
  **近 60 日漂移 −0.353、实时段 z_size≈−1.0（小市值暴露持续），红牌属实**。
  一次性诊断脚本 `scripts/oneoff/_signal_mktcap_drift.py` 保留作 882 口径存档。
- [x] **指数点位市场状态特征进 GBDT 面板**（09-23，国金19 转译；`model/market_features.py`
  9 特征×指数（多周期动量/均线位置/20日波动/52周位置），expanding z、防前视因果锁测试）：
  `rolling_grid_alla --market-features`（须配 --out-tag）旁路注入 stage_predict——不过 select
  漏斗（逐股 IC 对日级广播特征无定义）、不做截面 zscore（同日同值 std=0→全 NaN 坑）、
  不进 existence_mask。默认上证+国证A指（2015 起缓存全期；沪深300 缺 2018 不默认）。
  ✅ **实验已跑（10-21 批次 12②）**：国证A指 399317 本地缓存已补拉，
  `ortho920mf` 全A 重训——verdict 见 GOOD_MACHINE_TASKS 批次 12②。
- [x] **llm_pool 机制抽取（QuantaAlpha 转译，09-23 完成）**：① `structure_similarity`
  AST 子树 Jaccard 去重（同族不同窗=1.0、异族=0.0 实测；`_update` 入池前拦截）；
  ② `param_heavy` + `too_many_features`（MAX_BASE_FEATURES=6 固定上限）；
  ③ `llm_semantic_check` opt-in 语义一致性（依赖注入，不触网可测）。118 用例全绿。
  机制④（假设元数据）归东吴 MCTS Phase 0；机制⑤（池容量 50% 上限）与 AlphaPool
  corr_threshold/capacity 语义重叠，备选不做。
- [x] **多模型×多标签预测合成对照**（09-23 完成，`scripts/evaluation/member_blend12.py`）：
  12 成员 = 3 模型 × 4 horizon（horizon 即不同训练标签，AI29 多标签维度的现有 pred
  等价物）。**增量全在 horizon 维**：eq_h20_3 16.00% > eq_h10_3 14.19% > eq_h5_3
  13.58% > eq_h1_3 12.27%；同 horizon 混模型族是稀释（eq_h1_3 12.27% <
  solo_gbdt_h1 13.51%）；全 12 等权 15.12% < h1020 等权 16.89%（−1.77pp）；
  walk-forward 学权 stack_all12 15.53% ≈ 生产 ens_h1h5 15.52%（学权对 12 成员
  仍无增量，与 stack_blend 7 成员旧结论一致）。**h1020 = 920 后候选升级项**
  （+1.37pp vs 生产基线，882 存量 pred 口径，重跑后复验再议——**920 已裁决采纳**
  （horizon_mix_920 +3.31pp），12 成员复验随之关闭）。
  证据 `reports/member_blend12/`、`reports/horizon_mix_920/`。
- [x] **`model/labels.py` 加 IR/Calmar 标签分支**（09-23，AI29 转译；`build_labels(method="ir"/"calmar", bench_close_panel=)`
  —— IR = 区间超额收益 ÷ 区间内日度超额 σ，Calmar = 区间超额收益 ÷ |几何超额净值 MaxDD|；
  method="return" 默认零回归（36 既有测试逐位一致）+ 7 个新测试。
  **实验已跑判负（10-04，见下方标签臂③ir 记录）**：AI29 自述回撤代价 + Label
  Alchemy vol_scaled 预警双应验；calmar 同族分母更病态不再跑。
- [x] **标签工程双论文研读**（10-04，SSRN 7494298《Label Alchemy》+ 易方达 ICML 2026
  《The Label Horizon Paradox》arXiv 2602.03395；微信/小红书转读全文，非研报库）：
  **Label Alchemy**（美股 2002-2026 月频、固定 GBDT/弹性网络/NN、扩展窗口滚动）：
  标签三维度 Location/Scale/Shape——raw 收益 L-S Sharpe 0.68 → **高斯化排名 1.69**
  （全部 15 标签最佳 1.74）、行业 z-score 1.56、普通截面 z 1.53、仅去市场均值 1.15、
  PCA/FF6 残差 1.16–1.28；**vol_scaled 0.36 / vol_scaled_rank 0.66 比 raw 更差且换手高**
  （历史波动≠未来波动，除法重排横截面削弱信号）；方差分解：标签解释力 **0.87 vs
  模型 0.04（≈20×）**；预测期越短差距越大（1m ΔSharpe 1.01 → 12m 0.30）。
  **Label Horizon Paradox**（A 股 CSI300/500/1000、10 骨干、分钟特征→日频/30m/90m 三
  场景）：最优训练标签 horizon δ*≠目标 Δ（边际信息实现 vs 边际噪声累积权衡，定理 3.2
  分解）；**日频隔夜场景 δ*≪Δ，close-to-open 代理标签优于 close-to-close**；BLO 双层
  优化（λ softmax 加权多 horizon 损失 + 3 epoch warm-up 防捷径 + 熵正则防塌缩）单次
  训练自动找 δ*，CSI500 LSTM IC 0.0845→0.1029 / ICIR 0.724→0.861；开源
  `Chenhui-Song/label-horizon-paradox`（Alpha158 demo）。
  **转译 4 点**：① 主线 rank 标签方向获外部支持，但项目是均匀百分位秩（−0.5, 0.5）、
  论文最佳是**高斯化排名**（rank→Φ⁻¹ 一行变换）——`model/labels.py` 加 `gauss_rank`
  mode 单臂对照（挂 920 后，与 ir/calmar 臂同批；注意论文是美股月频，日频外部效度
  须打折）；② **vol_scaled 证据 = AI29 IR/Calmar 臂预期管理**：除以波动类标签在该论文
  跑输 raw（口径差异：AI29 用区间未来实现波动、论文用历史波动，机制不完全等同，但
  「同口径报回撤+换手」要求再 +1，不达标即弃）；③ **隔夜标签臂**：训练标签改
  close→next open 收益、预测目标/执行不变（Scenario 1 证据），与 `--execution open`
  口径天然对齐，单臂便宜；④ 「标签 20× 模型解释力 + 短期收益最大」与 member_blend12
  「增量全在 horizon 维」互证，E1'' horizon 定版优先级获文献背书；BLO 作为 E1'' 的
  自动化升级备选（非现在做）。
  指针：SSRN `papers.ssrn.com/sol3/papers.cfm?abstract_id=7494298`；arXiv `2602.03395`；
  微信转读 `mp.weixin.qq.com/s/uKFYF6eb1Ox008ct009cNA`；小红书 QuantML 转读
  「易方达｜预测次日收益，模型要用同样标签吗」。
  **实验闭环（10-04 当日，三臂全量跑完）**：gauss_rank 零增量 / co 判负 / ir 判负
  ——现行 rank+cc+return 即标签工程维局部最优，详见下方三条臂记录。
- [x] **标签臂 ①gauss_rank**（10-04 立项，10-04 完成，Label Alchemy 转译）：
  `model/labels.py` `mode="gauss_rank"`（截面秩→Φ⁻¹）+ `rolling_grid_alla
  --label-mode`（指纹含 label 口径）。全量对照 `ortho920tl_lblgauss` vs 基线
  `ortho920tl`（同口径 ortho+剔 limit_pos+可交易标签+open/close 双执行价）：
  **零增量**——IC 差 ±0.001 内（ranker d≡0 为数学必然：排序损失对单调变换不变，
  内置一致性验证）；月频组合中位差 0.00pp（close 52 组合正负 25/15，open 28 组合
  12/12）；gbdt_deep h1 Top10 +0.6pp 为孤例不达分布；回撤/换手不变。**归因**：Label
  Alchemy 增益大头在 raw→截面变换这一步，主线 rank 已在最优族内，rank→gauss_rank
  只是分布形态微调（其美股月频证据不自迁移到 A 股日频 GBDT）。**保留 mode 代码
  不换主线**。
- [x] **标签臂 ②隔夜标签**（10-04 立项/完成，Label Horizon Paradox 转译）：`window="co"`
  （close[t]→open[t+1] 训练标签，仅 h=1，`_base/open_adj` 复用；评价仍 cc 解耦）+
  `--label-window` 防线（须显式 --horizons 1）。全量 `ortho920tl_lblco`：**负增量
  弃用**——IC 全模型 h1 −0.003~−0.009（ICIR 反升 17→18 但不转化为组合优势）；月频
  组合中位 **−1.0pp（close）/−1.2pp（open）**，回撤 +1.3~1.6pp、换手 +1.3pp 双恶化。
  **ICIR 升已专项检验为无价值副产品**（10-05，gbdt h1 逐日 IC 配对 n=2118：mean
  −0.0038 配对 t=−5.2 p=1.9e-07 高度显著降、负日占比 54%；std 同步降 9.3% 且降幅
  比例大于 mean → ICIR 净微升；corr(base,co)=0.944 同源信号。std 降是隔夜收益截面
  本身更平静的镜像，非信号变稳）——**日频组合（最该吃 ICIR 红利的场景）28/28 恶化
  −4~−20pp**，无一例外。
  **归因**：论文 Scenario 1 是分钟特征+深度时序模型；日线因子集的信息已在收盘价中
  实现，隔夜段标签 = 丢日内信息量 + 引入隔夜跳空噪声。**归档，不动主线**。
- [x] **标签臂 ③ir**（AI29 转译 09-23 落码，10-04 全量判负）：`--label-method ir
  --horizons 5,10,20`（h=1 数学无定义——区间 σ 需 ≥2 日观测，实证全 NaN 已加管线
  防线）。全量 `ortho920tl_lblir`：**判负弃用**——IC 全线 −52~−191×10⁻⁴（gbdt h5
  0.1111→0.1047、h20 0.1351→0.1256）；对标签自身 IC 也降（gbdt h5 0.1000 <
  基线对 cc 0.1111）= 学到更弱而非学到不同；组合层月频 12/12 分裂、中位 +0.04pp
  无增量、**回撤中位 +1.27pp 恶化**（ridge h5 −1.5pp/回撤 +7.6pp 典型）。AI29 研报
  自述的回撤代价 + Label Alchemy vol_scaled 预警（0.36/0.66）**双双应验**。
  **calmar 不再跑**（同族分母更病态）；AI29 标签转译线闭环。
  **三臂总结论**：现行 rank+cc+return 标签口径 = 标签工程维度的局部最优；两个 2026
  外部证据（SSRN 7494298 / ICML 2026）的可迁移部分在本口径无低垂果实——反证标签
  形态预算应转投 horizon 混合（member_blend12 +1.37pp 实证）。证据
  `reports/alla_rolling_ortho920tl_lbl{gauss,co,ir}/`（ic_stats + metrics_overall
  ×open/close + 驱动日志 predict/backtest*.log）。
- [x] **预测层观点注入等价实现**（09-23，AI43 等价实现非源码复现；`model/views.py`）：
  ① `inject_by_feature_duplication`（观点因子复制 ×k，软注入——改变分裂概率/收缩结构）；
  ② `TwoStagePredictor`（观点因子逐日中位数分层，层内各训一个底层预测器，硬注入——
  树顶部 k 层分裂的离散版；predict 分组判定用输入面板自身网格，OOS 折内可用）。
  与 BL 优化层注入（`bl_views_from_factor`）构成对照臂；**三臂对照实验待跑**（挂 920 后，
  同一观点下 BL 注入 vs 预测层注入 vs 不注入，且须披露观点因子单因子 IC——观点可能是错的）。
- [x] **CVaR 约束进 solve_portfolio**（09-23，华安 226 转译）：`scenario_returns` (S×N)
  历史模拟场景 + RU 线性化，约束 `CVaR_α(loss) ≤ c`（loss=−场景收益·w）；面板级
  `optimize_weights_qp` 自动构造场景矩阵（< t 的最近 window 行，与 rolling_covariance
  同防前视纪律）。**工程注意**：RU 辅助变量（S+1 个）使紧约束下 OSQP 默认 max_iter=4000
  不够（user_limit）→ CVaR 激活时 max_iter=200k + eps 1e-8；校验口径必须用 RU 分式
  尾部（⌊(1−α)S⌋ 全额 + 下一场景分数权重），整数 floor 口径会误报超限。6 个新测试
  （绑定/空转/手算/α 语义/参数校验/面板透传）。实验待跑（挂 920 后）。
- [x] **微软 RD-Agent / RD-Agent(Q) 研读归档**（09-24，不复现不替换）：量化 R&D 自动化
  多 Agent 框架（NeurIPS 2025，Qlib 绑定）——与 AI97/东吴 MCTS 同题不同栈。**不复现理由**：
  口径防线缺失（无纸面/T+1/执行价收口、无正交化/DPP）、数据管道断链（Qlib bin 格式）、
  仅 Linux+Docker、"2x ARR"基准（Alpha158 CSI300）与我们口径不可比（第三方实测 IC 仅 0.0152）。
  **机制抽取 8 条**：① 逐日 IC_max≥0.99 数值去重（比 AST 去重多数值防线）+ ④ 连续失败换向/
  复杂度渐进 prompt 规则 + ⑦ 公式禁省略号 JSON 纪律 → **随 MCTS Phase 0 落地**；
  ② bandit 方向调度（8 维状态 Thompson 采样）记录待查；⑥ `fin_factor_report`
  自动 vs 手工转译对照列 **P2（挂 920 后 + 好机器 WSL2，产出须过纸面三判据 + 920 IC）**。
  笔记 `reports/docs/research_notes/微软RD-Agent_研读_量化RND自动化多Agent框架与机制抽取.md`。
- [x] **AI97 研报消融补齐（5 组）——随主结果归档作废（10-16 拍板）**：主实验已
  终判为**真实失效**（PBO=0.704、100k 步归因=样本外失效非未收敛，批次 2 归档），
  失效引擎的超参/结构消融无裁决意义，不再投入。5 组原始设计存档于此备查：
  因子复杂度约束（研报示 `[3,8]` 最高 18.71%、`[3,7]` 崩 −1.56%，极敏感）/
  LSTM vs Transformer / IC vs ICIR Pool / 奖励 6 版 / 因子数量扫描（10 个最优）。
- [ ] **AI39 剩余（单独立项，P1——本机工程可做，立项与否待由莉酱拍板）**：
  **① 主动权重空间结构性口径**（x=w−w_b，工程量中等）：改口径不改结论逻辑，
  与组合层已收口（QP 全家族关闭）耦合——先回答「重开何线需要它」再动工。
  低成本 3 件套已落地（09-18）。
  **② Barra 风险模型自建（10-16 定版 = **CNE6 简化版**；**第一期 10-16 +
  第二期 10-21 均已完成**，仅应用侧未跑）**
  **目标 = CNE6 9 一级骨架简化版**（10-16 由莉酱拍板：按 CNE6 来、CNE7 新增自建
  困难的不追——国有持股仅十大股东近似口径、拥挤度细节未公开，不值得为两个近似
  臂拉长战线；CNE7 增量降级为「了解即可」备注，不作工程目标）：
  - **✅ 第一期已完成（10-16，实际落地为 13 描述变量 → 11 一级风格）**：
    ① builder `scripts/builders/build_alla_barra_styles.py` → `panels_barra/`
    （lncap/beta/resid_vol/momentum/nl_size/liquidity/ep_ttm/btop/growth/mlev/
    dtoa/blev/dtop，全 13 变量 2026-10-15 落盘，coverage 0.566~0.903）；
    ② WLS 回归链路 `research/barra.py`（MAD 5× 去极值 → 市值加权均值/等权 sd
    标准化 → 逐日 WLS sqrt(市值) 权重 + 申万一级哑变量；pandas 3.0 兼容——
    `ndarray @ DataFrame` 混乘返回错索引 DataFrame 会静默炸 NaN，全程 .values）；
    ③ 验收台 `run_checkup` 落 `reports/barra_cne6/`（factor_returns / meta /
    factor_t / summary）：**全窗 2244 日、加权 R² 均值 0.217（与回归目标权重
    sqrt(市值) 同尺度；等权口径约 0.3）、日均样本 3318 只**；风格因子收益 t 值
    符合 A 股已知形态（liquidity −8.98 强负 / beta +4.57 / momentum +3.39 /
    dtop −3.11 / btop +2.08 / nl_size +2.79）。测试 `tests/test_barra.py`
    21 用例全绿（合成数据精确答案：WLS 恢复已知因子收益 / CAP_NATIVE 等权例外 /
    NW 半正定 / Σ=XFX'+D 手工答案 / shift 对齐脉冲）。
  - **✅ 第二期已完成（10-21）**：`bayes_shrink_fcov`（向「等相关」先验收缩
    因子相关矩阵——对角保留各方差、非对角拉向 ρ̄·√(F_ii·F_jj)，δ 由 Ledoit-Wolf
    解析解估并钳 [0.05,0.95]）+ `vol_regime_adjust`（VRA：近 60 日实现波动 /
    长期 EWMA(250) 预测，跨股中位数，钳 [0.5,2]；长期项用 250 而非 90——否则
    跳变会被长期均值自己吸收，VRA 失效）+ `assemble_sigma` **多风格化**（旧版
    把所有 `sty:*` 列赋同一个 Series，单风格测试掩盖了它，10-21 修 + 补用例）+
    `risk_bias_stat`（Bias = sqrt(mean((r_i/(σ_i·√h))²))）。
    `build_sigma_artifacts` 全窗跑：**40 抽样期 Bias 均值 1.079 / 中位数 1.055**
    （理想 1.0，Barra 业内典型 1.0~1.15）；分年 2020 = 1.000 / 2018 = 0.781
    （高估）/ 2024–2026 = 1.11~1.22（低估，近三年波动上行）；离群
    2024-01-15 bias = 2.61 = 小微盘流动性危机（极端事件不可预测，勿当 bug）。
    产物 `reports/barra_cne6/`（resid_panel / spec_vol / f_cov_last /
    sigma_summary / sigma_diag_last）；`tests/test_barra.py` 30 用例全绿。
  - **应用侧（未跑，真缺口）**：组合层接入点 = `optimize.solver.solve_portfolio
    (style_exposures=B, style_tolerance=0.3)`（B 已 zscore，tol 单位即 σ）；
    归因报告同样未做。**注意**：项目「QP 全家族关闭」（10-17 收口）意味着
    风格约束只有在重开组合优化线时才有意义——先回答「重开哪条线需要它」。
  - **因子清单（第一期实际 13 变量）**：Size(lncap/nl_size) / Volatility(beta/
    resid_vol) / Momentum / Liquidity / EarningsYld(ep_ttm) / Value(btop) /
    Growth / Leverage(mlev/dtoa/blev) / DivYld(dtop)；行业 = **申万一级**。
    Quality 与 Sentiment 未建（基本面盈利质量变量缺面板、可后补）。
  - **CNE7 新增因子（10-16 拍板：不追，了解即可）**：国有持股——数据源虽可行
    （`share_holder.parquet` 十大股东 holder_name 关键词识别 22.9 万条，1990→2026-10），
    但口径 = 十大股东 ≠ 全部股本、与 MSCI 原版有系统差；拥挤度公开细节不全。
    两项均为近似臂，不值得拉长战线，若日后想挖「国有属性/拥挤」再单独立项。
  - **原三步工程对照（全部落地）**：① 风格因子 builder（✅ 半天，量价类全自建 +
    财务类 `build_pit_panel` PIT 展开）；② 逐日截面 WLS 因子收益回归链路（✅，
    参照 DolphinDB CNE6 教程窗口参数：HBETA 252d / DASTD 252d / CMRA 365d /
    HSIGMA 252d / RSTR 252d+lag11+halfLife 126 / STOM/STOQ/STOA 21/63/252d /
    MLEV·DTOA·BLEV 365d / DTOP 365d）；③ 验收台（✅ 因子收益 t + R² 时序，
    VIF 与 ΔR² 增量验收挂天风0830 六维框架待接）。
  - **有什么用**：① 风险坐标系——组合优化器加风格偏离约束（≤0.3σ）防风格逆转
    亏损（天风实证 MDD −1.1~−2.7pp）；② 归因——把组合日收益分解到风格/行业/个股
    （区分「风格顺风盈利」与「真 alpha」——现无此能力，gbdt 非线性可能重新长出
    市值暴露，`signal_mktcap_spearman` 监控只有单变量近似）；③ 验收参照系——天风
    BetaAgent 六维风险因子验收 + 未来任何新因子的「增量解释力」标尺；④ HRP/
    risk_parity 重开的前提零件。
  **② 零件盘点（10-16，盘点详情见 `reports/docs/research_notes/天风0830_研读_*.md` §五）**：
  - **已有零件**（分布四层）：华泰五因子中性化协变量（`factor/preprocessing.py:750`
    `build_style_covariates`，size/mom/vol/turn+行业）；16 个 C 族基本面风格因子
    Barra 式构建 → `panels_neu/`（`scripts/builders/build_alla_style_factors.py`，
    缺 Beta/ResVol/Liquidity 等量价类）；QP 风格中性化约束 `style_exposures`
    （`optimize/solver.py`）；风险归因/分解 `optimize/risk.py`（α/β + Brinson +
    Euler MRC/CR + B'ΣB 风格方差贡献 + VaR/CVaR + 预算校验，归因侧基本可用）；
    Ledoit-Wolf 个股协方差（`optimize/solver.py:94`）。
  - **缺口三条（✅ 10-21 全部消除）**：① 定版风格因子体系（✅ 13 变量 →
    11 一级风格）；② 因子收益时序估计（✅ `run_factor_returns` 逐日 WLS，
    2244 日 × 44 因子）；③ 结构化协方差 Σ = XFX'+D（✅ NW 期限调整 +
    特征值裁剪 + 贝叶斯压缩 + VRA，Bias 1.079）。**剩余 = 应用侧**（组合层
    风格约束 0.3σ / 归因报告），且受「QP 全家族关闭」前提约束。
  - **衔接**：天风0830 BetaAgent 六维验收框架（ΔR² 增量 / VIF）需 Barra 参照系
    （本期①验收台即解决）；未立项过渡期可用现有风格协变量搭「土参照系」先跑
    （数字不可对账研报）；HRP/risk_parity 重开条件亦挂此（§一 集中化裁决）。

### 基本面因子权重过低的诊断与双频融合（09-22 研读+本机诊断完成）

**现象**（`reports/alla_daily_ortho/feature_importance_20260916.csv`）：gain Top5
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
3. **头部归因（D2）**：`explain_top_20260916.csv` Top20 选股前三驱动全为量价+市值
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

### 🥈 ETF 轮动「盲区 / 出场加速」探针组（2026-10-27 立项；赛后执行，判据已预注册）

> 2026-10-28 清理：`etf_contest_probe.py`、诊断脚本 `probe_etf_blindspot_diag.py` 与
> `reports/etf_contest5_opt/{entrylow,exit,blindspot,explore,explore2,volpre,freq}/`
> 已送回收站；结论保留在 `reports/etf_contest5/README.md` §13.10–§13.15。

> **背景与冻结纪律**：12-1 动量分子锚定 `P(t−21)`——入场时最近 21 日价格对排序不可见
> （本线实测：最近月信息作**排序**是负资产，20/60/6-1 动量全负，只配做减法），且持仓后
> ~3 周内新跌势要等分子解冻才进信号（分母通道立即衰减但慢）。当前实例 = 2026-10-26
> 信号通信 77.55%（入选时 mom20 已 −30%）。定版 v2 的 `pre10` 只挡了盲区**上半边**
> （20 日涨 >10% 不买），本探针补**下半边（入场闸门）与出场加速**两条路。原则：
> **入场慢（抗噪声）、出场快（抗钝化），一律不加回排序层**。赛内零改动（`FINAL`
> 不动、比赛口径不动）；本组只跑数，采纳裁决赛后做——剩余赛程 ~7 次周频调仓，
> 任何"优化"在这个样本量上的实现值都是噪声。真源：`TODO.md` §D1、
> `reports/etf_contest5/README.md` §13.9。

- **A 组（入场闸门，3 变体）**：`mom20 < −10/−15/−20%` 不入场（`pre10` 的镜像，
  三档单调协议照抄 v2）。**先验双边、如实报**：挡掉的可能是接飞刀、也可能是洗盘后
  买点；v2 已示强趋势年（2024/25）剔过热会少赚，剔深跌可能同样砍收益腿。
- **B 组（出场加速，5 变体）**：周频调仓不变，新增**日级持仓检查**，触发即离场，
  资金留现金至下个调仓日（不设再入场规则，避免新参数）：
  - B1 锚定价出场：收盘 < `entry_mult × P(t_entry−21)` 离场，`entry_mult` ∈
    {1.00, 0.95, 0.90}——机制 = 入场所依据的价位（锚定价）已失效，零/单参数；
  - B2 无跳过动量出场排名：持仓的 `mom12_0` 跌出 Top5 即撤（skip 只保护入场，
    出场用快信号，"入场慢/出场快"不对称的直接实现，1 变体）；
  - B3 均线对照：收盘 < MA60 离场（防 B1 恰似均线的对照组，1 变体）。
- **C 组（trailing stop，5 变体）**：从入场后最高点回撤 {5, 8, 10, 15, 20}% 离场。
  **鞭打审计必做**：止损后 20 日内收复锚定价的比例逐档必报；>40% 且中位代价
  >0.5pp 直接判负（动量止损的经典结局是震荡市被反复扇耳光，只看左尾会自欺）。
- **预注册验收判据（先于跑数锁定，口径同 `c3_final`：46 池、80 个滚动 2 月窗口、
  1+5bp、EXCLUDED_5 自动排除）**：
  1. **主判据（左尾）**：p10 改善 ≥1.0pp 且最差窗口改善 ≥3pp 且 <−10% 概率不升；
  2. **约束**：中位代价 ≤0.3pp；年化成本拖累增量必报（v2 教训：换手 36→45%
     对应成本 5.2→9.4%/年）；
  3. **单调性**：A / B1 / C 阈值族须三档单调；分年改善不得集中在单一年份；
  4. **多重检验预算**：全组 **13 变体封顶，跑前不得追加**；采纳门槛沿用本线基准
     ——ΔSharpe 须超 18 变体噪声带 σ≈0.045 的 2 倍以上，且左尾判据过线。
- **对照臂（不计变体预算，作参照不作裁决）**：B1/B3 两条在磐石 `champ_dd90`
  月频口径重跑，看 MDD/p10 改善为主（防御腿+波动目标已缓冲一轮，预期增量小，
  优先级低）。
- **实现范围**（另一台机器可先做，先写 1 变体冒烟对齐口径再全跑）：
  `etf_rotation_probe.run_variant` 加 `exit_rule` 参数（`anchor` / `mom12_0_rank` /
  `ma60` / `trail`）与日级持仓检查；`etf_contest_probe.py` 新增 `ENTRYLOW_SPECS` /
  `EXIT_SPECS`（照 `REV_SPECS` 模式，基线仍从 `etf_contest_spec.FINAL` 派生，
  不得改动 `FINAL` 本体）；无新数据依赖（`daily_etf.parquet` 现成）。证据目录
  `reports/etf_contest5_opt/entrylow/`、`reports/etf_contest5_opt/exit/`、
  磐石对照 `reports/etf_rotation_p2/exit/`。

```bash
# 实现后固化复跑命令（EXCLUDED_5 自动排除，与 c3_final 同口径）
python -m scripts.portfolio.etf_contest_probe --specs entrylow   # A 组
python -m scripts.portfolio.etf_contest_probe --specs exit       # B+C 组
```

- **复现边界（引用数字必须注明）**：① 周内离场成交 = 触发日收盘（与调仓同口径，
  略乐观、未计跳空）；② 离场资金留现金至下个调仓日，无再入场规则；③ 鞭打统计
  窗口固定 20 日；④ 80 窗口重叠，t 类统计仅作参考不作判据；⑤ 判负项同样回写
  本条目收口，不留悬案；⑥ 赛内不得部署任何通过项（含 −15% 熔断之外的临时规则）。

**终判（2026-10-27，预注册后当日跑完；全表 = `reports/etf_contest5/README.md` §13.10）**

- **盲区：结构真实、经济不成立**。诊断脚本 `scripts/oneoff/probe_etf_blindspot_diag.py`
  （证据 `reports/etf_contest5_opt/blindspot/`）：建仓后 21 日实际收益与信号
  （`mom12_1_crowdsoft`）同期变化量相关仅 **+0.12**（n=281），首月跌 >10% 的 37 笔
  均值 −14.7% 而信号只掉 7.8pp ⇒ 钝化是机械事实。但按入场时 mom20 分桶，深跌
  （−20~−10%，n=15）后续 fwd42 均值 **+5.36%** / 中位 +6.98%，其余两桶 +0.30% / −0.20%
  ⇒「接飞刀」在 ETF 上是反的；且 name-day 口径仅 10.2% 落在 mom20 < −10%，
  **整组 Top3 全在深跌桶的调仓日仅 2.5%**，A 组能动仓位本就很薄。
- **A 组（入场闸门 3 变体）：单调恶化**。c3_final 均值 +5.59%/中位 +3.76%/p10 −3.64%/
  最差 −12.34%/Sharpe 1.28 → low20 +5.48% → low15 +4.43%（p10 −6.95%）→ low10 +4.06%
  （中位 +1.92%、p10 −7.30%、最差 −14.60%、Sharpe 0.91）。方向与 `pre10` 相反：
  闸门越严越差，且 low10 出现只持 2 只的调仓日。
- **B/C 组（出场加速 10 变体）：无一过主判据**。关键设计缺陷——B1 锚定价
  `P(t_entry−21)`、B2 `mom12_0` Top5、B3 MA60 三条在建仓时同时成立，触发集中于
  **建仓后 ≤1 日**（80.3%/68.6%/54.4%/75.6%/76.2%），实质是「把冻结点上已不合格的
  标的当日剔掉 + 下周原价买回」的周内来回，成本拖累 9.4%→10.3~16.3%/年，收益被吃光；
  它们不是「出场快」而是「入场又慢又反复」。只有 trail 族是真持仓期出场（中位触发
  间隔 9/12/20/30/39 日）：trail10 均值 +6.18%、Sharpe 1.53（Δ +0.25 > 2σ）、
  MDD −25.4%→−22.5%，但 p10 −4.65%、最差 −14.39%，且五档 Sharpe
  1.48/1.28/1.53/1.22/1.29 无单调 ⇒ 按预注册判据判为噪声。
- **鞭打审计**（离场后 20 日）：trail 5/8/10/15/20% 收复前高率
  41.2%/26.7%/20.8%/12.5%/5.0%；trail10 离场后 20 日中位收益 **+2.05%**
  ⇒ 中位情形下卖出是亏的，保险买贵了。仅 trail5 触发「收复 >40%」档，其中位代价
  −1.59%（<0），不满足判负的合取条件，但仍不过主判据。
- **裁决：0/13 通过，全组判负，赛内零改动**（`FINAL` / 宇宙 / 成本 / 调仓日 / SOP 全不动）。
  当前 3 只持仓 mom20 为 −30/−30/−25%，正是本探针最想剔除的形态；实测剔除它们会让
  全部指标变差。实现：`etf_rotation_probe` 的 `ExitOverlay` / `build_exit_schedule` /
  `whip_audit` / `run_variant(exit_plan=)`，`etf_contest_probe` 的 `ENTRYLOW_SPECS` /
  `EXIT_SPECS`（均从 `FINAL` 派生）与 `--specs entrylow|exit`。证据
  `reports/etf_contest5_opt/{entrylow,exit,blindspot}/`（`contest_entrylow_summary.csv`、
  `contest_exit_summary.csv`、`contest_exit_exit_events.csv`、`contest_exit_windows.csv`）。
- **口径对齐已验**：baseline 复跑与既有 `reports/etf_contest5_opt/rev` 完全一致
  （+5.59%/+3.76%/−3.64%/−12.34%/72.5%/1.279）；「永不触发」的出场层与基线逐日
  `max|Δequity| = 0`。演进新规则需另立预算，不得在本组 13 变体上追加。

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
- [x] **Phase 2（zz1000 主实验）：全档 24 臂终判（2026-10-05）——负结果归档**
  `scripts/factors/run_portfolio_phase2.py`：三标签 GBDT 信号（质量门槛/回退）+ PPO
  银河滚动协议（双窗双门槛/多种子 Softmax 集成/全败回退）+ 银河口径 QP + EW/TopN
  基准 + 消融组。**修复链三连后终判**（env er 基准改可交易基准——0/4 全灭真凶系
  口径 bug 非 RL 无效；PPO 门槛 0.02/0.08→0.005/0.04 按 zz1000 重校，原值系研报
  科创50 口径；断点持久化 + 降级 best-of-attempts）：PPO full +1.07/+0.25/−3.17
  三年不一致，2025/2026 **全部消融臂反超 full**（六项 reward 分量无稳健净价值）、
  QP 仅 2024 正超额（+3.88pp）且 2025 solver 失败——**银河口径 RL/QP 组合优化在
  zz1000 未产生稳健超额**，与 AI97 互证 RL 路线在本项目数据域（日频
  zz1000/hs300）整体有效性不成立。证据 `reports/batch2385_结果速览.md` 全档判读节。

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
**前置工程项已清（09-24）**：`portfolio_env`/QP 臂接入 `data/tradability`
（`build_tradable_mask(execution_lag=0)` T 日状态口径——决策日收盘调仓自洽；
`make_env` 掩码 = 指数成分 ∩ 可交易 + 全 False 行回退；QP `valid` 同口径 +
信号 reindex 兜底）；hs300 ppo/qp/ew 全臂冒烟通过。测试
`tests/test_phase2_tradability.py`（lag 语义 + 掩码合成 + 回退）。

**Phase 3（全A 规模化）**：**不启动**——启动前提「Phase 2 验证 RL 超额机制」失败
（终判见上）。原设计意图（hs300→zz1000→全A 集中度自由度递增梯度画关系曲线）
随 RL 线归档作废；若未来重启需先解决「reward 分量不稳健」的机制问题。

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

## 六、建议推进顺序（2026-10-19 刷新）

> **2026-10-19 增量线指针**：横截面选股在「当前数据 × h1–h20 × long-only」格子
> 已收口；新立项 = `TODO.md` §六（A1 长 horizon / A2 残差目标 / B1 beta hedge /
> 指数增强 / B2 SDK 未用表）。A2 探针脚本 `scripts/evaluation/residual_ic_probe.py`
> 已落地并开跑，结果落 `reports/frontier_audit_2026-10-19/`。本表旧的 1–4 项
> 保留作历史，不再覆盖新线。
>
> 排序原则变化：长实验队列已全部收官（序 0 划线），本表剩项 = 资源墙后置线 +
> 定版落地工程（后者真源在 TODO §一）。

| 序 | 任务 | 说明 |
|---|---|---|
| 0 | ~~920 治本重跑 + 好机器长实验队列（批次 1–11）~~ **收官（09-24~10-17）**：AI97 / GFlowNet / stage2 / 批次 8 / 东吴 MCTS / mf10 T 扫描全部终判（§〇 状态表 + GOOD_MACHINE_TASKS 终态）；主模型定版 fundind+buffer(15/40)；批次 7 e2e 口径重跑 **10-17 收官**；批次 11 cheap_lever 920 复验**环境分歧后切换搁置**（最终裁决 = 批次 12 首项） | 定版切换生产链见 TODO §一 |
| 1 | ~~本机半天级穿插（§一 🥉）~~ **全部清账**（09-23~10-04：市值漂移监控/市场状态特征/llm_pool/观点注入/CVaR/member_blend12/标签三臂） | 报告与证据见 §一 |
| 2 | ~~研读批~~ **全部完成**（09-18~10-06，含 RD-Agent / 兴业0526 / 标签双论文 / 知乎双篇） | 报告均见 `reports/docs/research_notes/` |
| 3 | 华泰3128 全频段 | 依赖分钟特征扩容，资源墙后置 |
| 4 | 国金24 残余②：全A 分钟扩容 | 先跑吞吐探针估成本（含知乎华泰篇矿池工程化四件套路径指针），再决定是否投入 |
| 5 | ETF 轮动「盲区/出场加速」探针组（13 变体封顶，判据已预注册） | **裁决条件 = 比赛结束（~12 月中）**；实现+跑数可提前在另一台机器做（§一 🥈 条目），采纳与否赛后按预注册判据落判 |

**暂缓**：东方 DFQ / 华泰 DQN 择时（**RL 线已整体归档**，重启需机制级新依据）；
AlphaNet / GNN / GAN 系列（P2）。

**择时线专题（10-14/15 收口 + 业界调研）**：大盘仓位择时不立项（证据链 =
`reports/timing_probe/`：日自相关 +0.048 · 月度 0/7 显著 · 资金面 0/4 ·
规则 7/7 负；业界调研见 `reports/docs/research_notes/择时业界方案调研_2026-10-14.md`——
AQR「sin a little」+ JFE vol-managed 103 策略复核 8 显著全在动量类，与项目实测同向；
10-20 补读开源证券《行业时序择时框架》并入该笔记 §五——行业级 overlay 新样本，
半仓基准仓差虚高 + 适用性识别前视嫌疑，不立项，机制抽取两条备查）。

**❌ Regime 条件化 收口（10-15 当天立项当天收口；`reports/regime_probe/README.md`）**：
Phase 0 探针发现集成信号 IC 涨/跌期差异显著（0.127 vs 0.110，t=3.55），**但两道
kill 点全部触发**：① 定版 50 因子分状态 IC —— **没有任何族跌期更强**（量价 36 因子
diff +0.0004；负端全是低波动因子负 IC 减弱，非可用 alpha）⇒ 「切换信号」无对象；
② 个股门分池 IC（drift 门 Δ+0.0023 p=0.70 / 高波动门 Δ+0.0061 p=0.074 校正后 0.15）
⇒ 门控无优势。集成 diff(+0.0173) ≫ 单因子 mean(+0.0009)，差异源于集成层结构效应
（疑似跌期 IC 相关性上升=分散红利衰减），不可变现。**预算 3.5 天实际用 0.5 天，
未进引擎。** 沉淀：横截面信号在本市场基本全天候；探针脚本
`scripts/evaluation/regime_probe.py`（--self-check 固化）。择时族三条线
（overlay/波动率管理/regime）至此全部收口，证据链三方同向（AQR + JFE + 本项目）。

---

## 附：全库分类速查

| 目录 | 篇数 | 已完成 | 待推进 | 不细读 |
|---|---|---|---|---|
| 华泰人工智能（主目录） | 44 | 21/23 GP、14/16 CPCV、11 stacking、6 Boosting、AI39 核对、AI29 研读、AI19/22 PBO、AI43 研读、AI106 转读（10-20，公众号版不复现） | AI32/34/42/13/27/40/45 | ~19 |
| 因子挖掘 | 9 | 国金22、国泰君安 GP、国金24 主体+残余③④⑤、AI97 全量（负结果归档）、银河 0608 研读、东吴0623 全量（席位突破/组合无增量归档）、东方0407 归档、微软 RD-Agent(Q) 归档、兴业0526 综述速读归档、天风0830 BetaAgent 归档（10-16） | AI26、国金24 残余②（全A 分钟扩容资源墙） | — |
| 因子合成 | 9 | 申万 ML、多因子10 补齐+T 扫描出数（T=3 最优不显著）、国金19 归档 | 华泰3128 | AI28 |
| 文本挖掘 | 6 | 51/57/63/41 全部 | 华泰 LLM_FADT | — |
| 强化学习 | 6 | T2RL 研读、银河0706+华安226 研读、stage2 Phase 0/1/2 全档终判（**负结果归档 10-05**） | 东方 DFQ、DQN（随 RL 线归档暂缓） | — |
| 组合优化 | 1 | AI39 三处低成本对齐 | 主动空间结构性对齐 + Barra CNE5（单独立项） | — |
| 传统机器学习 | 7 | AI6/11/14/16、AI19/22 PBO、AI43/AI29 研读 | —（落地项见 §一 🥉） | 6 |
| 大模型投研 | 4 | 4 篇速读 | — | — |
| 探索 | 1 | — | AI42 | — |

> 注：华泰篇目在多个分类目录重复归档共 15 篇，全库实际不同报告约 60 篇。


---

<!-- ═══════════════════════ 册Ⅲ 好机器队列（原 GOOD_MACHINE_TASKS.md 全文快照，逐字保留） ═══════════════════════ -->

# GOOD_MACHINE_TASKS — 好机器大任务队列（2026-09-22 定稿；批次 1–11 终结，批次 12–13 = 新队列）
> 汇总全部需要好机器执行的大任务：批次顺序、完整命令、耗时估计、依赖与验收。
> 本机半天级穿插项不在本清单（见 TODO §一 🥉）。**约定：批次内任务可并行，
> 批次间按序；每批产物 CSV/报告拷回本机入库归档（reports/ 不进 git）；
> 全链成交口径已定版 open=T+1 可执行（正名产物），close=乐观上限对照（_close）。**
> **收官状态（10-17）**：批次 1–10 终判；批次 7 e2e 口径重跑收官；批次 11
> cheap_lever 920 复验**环境分歧后切换搁置**（最终裁决移交批次 12 首项）；
> **批次 12（10-17 立）= 新队列：v1w2 终裁 + 五项「挂 920 后」实验**。
> **批次 13（10-19 立）= A1/A2 增量方向 920 复验三臂（overlay / stack / h60）**。
> **批次 14（10-26 立，当日出数终结）= mf 干净单臂补测**：批次 12② 的
> +1.44pp 混了 `ortho→ortho_fundind` 变量；干净臂实测 **+0.49pp / Sharpe
> −0.010，不达 ≥+1pp 门槛 ⇒ 不切 v1-mf，市场状态特征线收口**
> （`reports/mf_fundind_check/`）。
> **执行收官（10-21，本工作机 YURIMIZUKI 一夜跑完）**：批次 12/13 除 ⑥
> （WSL2 环境阻塞）外全部落判——**批次 13 三臂全败 ⇒ A1/A2 线关闭**
> （882 正结果未迁移 920 数据态）；批次 12 ①④⑤负结果归档、⑦按 10-16
> 拍板作废不跑、②③重训出数（verdict 见批次节）。产物 README 真源：
> `reports/frontier_audit_920_reval/` `reports/cheap_lever_scan/` §四
> `reports/cvar_arm_920/` `reports/views_arm_920/`。
> 结果速览真源 `reports/batch2385_结果速览.md` + `reports/batch1_920_收尾对照表.md`
> + `reports/cheap_lever_scan/README.md`。命令与验收口径全文保留作复现存档。
> **资源标注**：每批标 🖥️CPU / 🎮GPU / 💾大内存。

| 批次 | 内容 | 终态（10-09） | 资源 |
|---|---|---|---|
| 1 | 920 治本重跑 + 定版臂 | ✅ **定版落定**：v1 面板污染事故→v2 完整口径（h1020⊕s0.5 17.87%/0.781）→fundind 快信号 18.68%/0.813→§⑯ buffer(15/40) 终版（18.07%/0.805/换手 19.1%）；**环境漂移更正：income 补齐后同配置 ~17.0–17.2%** | 🖥️💾 CPU 密集 + 大内存（全A 面板 float32 ~24GB + 滚动工作副本） |
| 2 | AI97 三臂 ×3 seed | ✅ **负结果归档**：PBO=0.704 / DSR 不显著；100k 步归因=真实失效非未收敛 | 🖥️ 纯 CPU |
| 3 | GFlowNet 正式实验 | ✅ **raw+im 定版形态**（0.117 > im 0.109 > raw 0.099，臂间差>种子噪声）；RRE 中性、预算饱和；114 因子入库 PBO=0.114 | 🖥️ CPU 可跑 |
| 4 | mf10 T 扫描全量 | ✅ 出数：两法均 T=3 最优、单调变差，NW t≤1.4 不显著——不构成升级依据 | 🖥️ CPU |
| 5 | stage2 Phase 2 zz1000 | ✅ **负结果归档**（全档 24 臂）：PPO 三年不一致、消融臂普遍反超 full、QP 仅 2024 正超额——RL 线整体不成立（与批次 2 互证） | 🖥️ CPU 多核 |
| 6 | 口径修复重训 E5/E4/P5 | ✅ E5 fundind **采纳为定版成分**（+0.81pp）；E4 池竞争+强制注入双证伪（+0.14pp 噪声级）；P5 两批 9 因子收口 | 🖥️💾 同批次 1 |
| 7 | e2e 族年化口径重跑 | ✅ **10-17 收官**：归档链 252 口径重出两报告（e2e 10.37%/0.666 vs 基准 8.94%；invest 11.91%/0.594 vs 指数 11.08%）；记录历史数（244 口径）以换算注记对齐（年化 +1~2pp、Sharpe ×1.016），差距主因环境漂移（幸存者修复/数据端点/特征池），如实记录 | 🖥️ CPU |
| 8 | 表格基础模型对照 | ✅ 三级收官：hs300（tabicl_w3m 头部内加权 11.97%/0.739 组件冠军）/ zz1000（tabicl 跨池最优 + gbdt+tabicl 对集成 20.03%/1.25 新发现）/ 全A（**ICL 优势不跨大截面，gbdt IC 4.4 倍反超，主模型基座维持 gbdt**） | 🎮 GPU（已执行） |
| 9 | 东吴 LLM-MCTS Phase 0 | ✅ all-A 正式跑（750 候选收编 14、PBO=0.039、OOS≥IS）→ 入库 llm_* → 席位 4/14 **唯一超门槛引擎** → 组合层 +0.16pp 噪声级**不采纳**——席位突破、组合无增量，归档（hs300 29 Seed 四臂协议未按原样执行，跨引擎对照由既有线结论替代，如实记录） | 🖥️ CPU + LLM API |
| 10 | 920 后收尾对照包 | ✅ horizon_mix/stack/fundamental_blend/buffer/T 窗/income 复核/alt 终审全出（§⑫–⑯）；member_blend12 12 成员复验因 h1020 已裁决采纳不再必要 | 🖥️ CPU |
| 11 | **cheap_lever 胜者 920 定版复验**（10-14 立项） | ⚠️ **10-17 复验环境分歧，切换搁置**：62 因子慢信号环境叠加臂 5/5 窗 +1.47pp 通过；本机补齐 P5 后（72 因子）仅 3/5 窗（2023+/2025+ 转负，全窗 +0.88pp）——近窗稳健性不成立，**维持 dingban**。字面迁移两环境均否、h1 减半操作成立。登记锚点 18.07 本机两态均不可复现（数据态年份差主导）。**最终裁决挂批次 12 = 好机器自家口径重跑**。产物 `reports/cheap_lever_scan/` | 🖥️ CPU |
| 12 | **新队列（10-17 立）**：① v1w2-cheap 终裁 + ② 市场状态特征 + ③ 隔夜标签臂 + ④ 观点注入三臂 + ⑤ CVaR 组合臂 + ⑥ RD-Agent P2（WSL2）+ ⑦ AI97 消融 5 组 | ✅ **10-21 落判（本机）**：①3/5 窗未达门槛 ⇒ **v1w2-cheap 转 superseded、链路优化线关闭，维持 dingban**；④三臂零增量（BL 跑输等权/复制臂结构性无效/两段微负，bp 观点 t=+14.2 本身正确）负结果归档；⑤CVaR QP 不升级（c=0.05 Sharpe +0.026 但换手 3.7×，buffer 帕累托占优）；⑥无 WSL2/Docker **环境阻塞留队**；⑦**按 10-16 拍板作废不跑**（RESEARCH_TODO 已记，失效引擎消融无裁决意义）；②**唯一正结果**：+1.44pp/5 窗为正（10-26 复核：该对照混了 `ortho→ortho_fundind` 变量，**补 `ortho_fundind+mf` 干净单臂后再判**，见本节 ⚠️）；③隔夜标签判负（−0.5~−1.3pp/换手升），标签工程线收口 | 🖥️💾 ①纯回测 <1h；②③需全A 重训（💾≥64GB）；⑤⑦ CPU；⑥ WSL2+LLM |
| 13 | **A1/A2 增量方向 920 定版复验（10-19 立）**：① A2 负向 overlay（walk-forward 因子选择版，w=0.40 冻结）② A2 二阶 GBDT 融合（α=0.5 冻结，特征池按 920 残差榜早年列重算）③ A1 h60 第二策略（h60 清单 + 920 基座对照） | ❌ **10-21 三臂全败 ⇒ A1/A2 线关闭**：①wf6 ΔSharpe **−0.030**（验收 ≥+0.015；固定集 +0.085=选择前视上限）；②stage2 残差 IC 0.074 过线但 fuse α=0.5 ΔSharpe **−0.041**（验收 ≥+0.08）不进融合层；③h60 配对差 t=0.33/0.78 不显著 + 组合 Sharpe 0.695 < 基线−0.05=0.771 + 换手未降，**「低换手第二策略」不立项**。882 正结果均未迁移（数据态换相：920 残差榜 top5 转正向质量因子；h60 清单 fundamental 6→3 席）。证据 `reports/frontier_audit_920_reval/` | 🖥️💾 ①纯回测 <1h；②残差首轮+训练 ~2h；③h60 清单可直用（选特征不依赖 pred）+ 训练 ~1h |
| 14 | **mf 干净单臂补测（10-26 立，批次 12② 收尾）**：`--preproc ortho_fundind --market-features --out-tag ortho920fundind_mf` + `buffer_arm_920 --variant defv`；**唯一变量 = 市场特征开关** | ❌ **10-26 终判：不达验收，mf 线收口**——defv 16.80%/0.728 vs 锚 16.31%/0.738 = 全窗 Δ **+0.49pp**（门槛 ≥+1pp）且 Sharpe −0.010；分窗年化 5/5 为正但 ΔSharpe 4/5 为负（+2.05pp 集中 2025+ 单窗）。**+1.44pp 确系「去掉 fundind」混杂（历史 +0.81pp），干净增量只有 +0.5pp——不切 v1-mf**，市场状态特征线关闭；产物 `reports/mf_fundind_check/` | 🖥️💾 全A 重训 + predict（💾≥64GB，好机器一夜） |

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

## 批次 7 —— ✅ 10-17 收官（本工作机）

- e2e 族报告对齐年化口径（`perf_stats` 年化 244→252 后 ~3% 系统性偏移）。
  归档链重出：`scripts/archive/e2e_backtest.py --real --top 50` +
  `scripts/archive/investment_report.py --real`（本机 scs 3.3.0 DLL 损坏曾令
  cvxpy 原生崩溃，换 3.3.1 修复后 rp 臂含入、个别调仓日空解降级等权留痕）。产物 `reports/e2e_backtest/` +
  `reports/investment_report/`（含 README 换算注记）。**如实记录**：重出数
  与记录历史数的差距远超口径换算（环境漂移主导——09-07 幸存者偏差修复
  −2pp + 数据端点 08-25→09-15 + 特征池演进），旧产物已删无法单放口径变更；
  口径对齐以换算注记完成，归档链数字不再作现行证据（现行 = 全A 主实验）。

## 批次 12：新队列（10-17 立——v1w2 终裁 + 五项「挂 920 后」实验 + AI97 消融）

> ✅ **10-21 落判（本工作机执行，verdict 汇总）**：
> ① **3/5 窗（2019+ +1.22 / 2021+ +0.50 / 2023+ −0.52 / 2025+ −0.36，
> 全窗 +0.77pp）< 门槛 ≥4/5 ⇒ v1w2-cheap 转 superseded、链路优化线终判
> 关闭，生产维持 dingban**（`baselines.yaml` 已改 status；锚点自检通过
> 16.31/0.738；`reports/cheap_lever_scan/` §四）。
> ④ **三臂零增量负结果归档**（BL 两档跑输等权对照 −0.18~−0.23 Sharpe 且
> 换手 +20pp；特征复制臂在生产 gbdt 配置 feature_fraction=1 下**结构性
> 零作用**（与对照逐位相同，`model/views.py` docstring 预告的前提不成立）；
> 两段模型 IC −0.006；bp 观点单因子 t=+14.2 本身正确——无增量在注入机制
> 不在观点错；`reports/views_arm_920/`）。
> ⑤ **CVaR 臂不升级**：c=0.05 Sharpe 0.764 vs buffer 0.738（+0.026）但换手
> 76.9% vs 20.6%（3.7×）且年化更低；c=0.03 过紧反伤；控换手版优势消失
> （0.647）。补丁后 105/105 全求解零降级。**buffer(15/40) 帕累托占优保持**
> （`reports/cvar_arm_920/`）。
> ⑥ **环境阻塞留队**：本机无 WSL2（未安装）、无 Docker、未装 rdagent——
> 按设计（好机器 WSL2 + Docker）不可执行，转待有 WSL2 的机器；DEEPSEEK
> key 已在 `.env` 就绪。
> ⑦ **作废不跑**（RESEARCH_TODO 10-16 拍板：主实验已终判真实失效，失效
> 引擎的超参/结构消融无裁决意义——本清单 10-17 立项时未同步该拍板，
> 10-21 回写同步）。
> ② ✅ **本批唯一正结果**：定版形态（defv buffer 15/40，同数据态配对）
> mf 17.74%/0.769/换手 21.0% vs 无-mf 锚 16.31%/0.738/20.6% = **+1.44pp
> / Sharpe +0.032，5/5 窗年化为正（+0.7~+2.3pp）、4/5 Sharpe 为正**——
> 过「>±1pp 才值得并入」线；单模型 h1 层无差（−0.04pp）⇒ 增量来自长
> horizon 分量。**建议采纳（生产重训默认带 --market-features），挂由莉酱
> 拍板**（国证A指 399317 缓存已补拉；`reports/alla_rolling_ortho920mf/`）。
> ③ ❌ **隔夜标签臂判负**：gbdt h1 主口径三行一致 −0.5~−1.34pp 且换手
> 升高（73.0% vs 69.0%），ridge 族大面积崩溃——隔夜段噪声占比高于全日段，
> 「标签与执行价同口径」直觉不成立；与 882 时代 tl 基座 lblco 判负双基座
> 互证，**标签工程线彻底收口**（`reports/alla_rolling_ortho920fundind_lblco/`）。
> 工程副产物：修复 `--exclude-features` 被 `_extra` 补位池绕过的管线 bug
> （19 测试全绿）。

> 来源：批次 11 复验环境分歧的最终裁决移交（①）+ RESEARCH_TODO §一 🥉 的
> 五项"挂 920 后"待跑实验（920 fundind pred 已随产物同步就位）+ AI97 研报
> 消融补齐。**优先级：① 先行（纯回测 <1h，裁决生产切换）；②–⑦ 并行铺开。**

### ① v1w2-cheap 终裁（首项，好机器/生产机自家口径）

```bash
# 1) 自家环境锚点参照（先出再复验）
python -m scripts.evaluation.buffer_arm_920 --pred-dir reports/alla_rolling_ortho920fundind \
    --out reports/cheap_lever_scan/_anchor_check_gm --variant defv --entry 0.15 --exit 0.40
# 2) 七臂 × 五窗复验（--anchor-ref 指上一步产物）
python -m scripts.evaluation.cheap_lever_revalidate_920 \
    --anchor-ref reports/cheap_lever_scan/_anchor_check_gm/buffer_arm.csv \
    --out reports/cheap_lever_scan/revalidate_920_gm.csv
```

**验收**：叠加臂 ≥4/5 窗一致且全窗 ≥+0.5pp ⇒ 由莉酱拍板切生产
（三处联动：schedule.yaml preset + 两个 product_glob + 两个下游脚本默认目录，
然后生产机 `install alla_daily_rank` 重注册）；不达 ⇒ v1w2-cheap 转
superseded，链路优化线终判关闭。本机两环境证据见 `reports/cheap_lever_scan/`
（62 因子 5/5 通过 vs 72 因子 3/5 撤回——慢信号构成敏感）。

### ② 市场状态特征实验（代码 09-23 已入库只欠跑）

```bash
python scripts/pipelines/rolling_grid_alla.py --preproc ortho --out-tag ortho920mf \
    --exclude-features limit_pos --tradable-labels --execution open --market-features
```

对照 = 既有 ortho920tl/fundind 同配置（不加 --market-features）；验收 =
主口径年化/Sharpe 变化 >±1pp 才值得并入（基线数字见 baselines）。

> ⚠️ **10-26 复核：上面那条臂不是单变量对照，别直接按 +1.44pp 切生产。**
> `--preproc ortho` ≠ 定版 `ortho_fundind`，故「mf 17.74% vs 锚 16.31%」的
> +1.44pp 里混着**去掉 fundind**一项（该项历史 +0.81pp）。**真正要上生产的是
> `ortho_fundind + mf`，一次都没跑过**。补跑（唯一变量 = 市场特征开关）：

```bash
python scripts/pipelines/rolling_grid_alla.py --preproc ortho_fundind \
    --out-tag ortho920fundind_mf --exclude-features limit_pos \
    --tradable-labels --execution open --market-features
python -m scripts.evaluation.buffer_arm_920 \
    --pred-dir reports/alla_rolling_ortho920fundind_mf \
    --out reports/mf_fundind_check --variant defv --entry 0.15 --exit 0.40
```

> 验收（不变）：全窗 ≥+1pp **且 ≥4/5 窗年化为正**，且须与
> `reports/alla_rolling_ortho920fundind`（无 mf，同数据态）配对，不得再借用
> 旧数据态的 tl 数字。顺带更正 mf README 第 3 点：生产出榜链是
> `alla_daily_rank.py`（每日自 `compute_features` + `fit`，不是用既有 pred），
> **必须接线**——10-26 已加 `--market-features`（默认关、须配 `--out-tag`、
> meta 记 `n_market_features`/`market_feature_last_date`）。
> 另：`update_data` 不维护指数日线，采用 mf 前须补指数缓存更新（代码已告警）。
>
> ❌ **10-26 终判（批次 14，本机出数）**：干净单臂 `ortho920fundind_mf` defv
> buffer = **16.80% / 0.728 / 换手 21.4%** vs 锚（fundind 无 mf、同数据态、
> 复测逐位复现 16.305/0.738）——**全窗 Δ +0.49pp < +1pp 门槛且 Sharpe −0.010**；
> 分窗 Δ年化 +0.31~+2.05pp（5/5 为正）但 ΔSharpe 仅 2025+ 为正（+0.061），
> 增量集中于近年单窗。**判定：不切 `v1-mf`，mf 线收口**；批次 12② 的
> +1.44pp 归因拆解 = 去掉 fundind ≈ +0.8pp + 干净 mf 增量 ≈ +0.5pp。
> `alla_daily_rank --market-features` 接线保留（默认关，零回归）。

### ③ 隔夜标签臂（需小工程接线，~半天）

训练标签改 close→next open 收益、预测目标/执行不变（与 `--execution open`
天然对齐；Label Alchemy Scenario 1 证据）。工程 = rolling_grid_alla 加
`--label-source overnight` 之类的标签推导分支（现只有 rank/gauss_rank mode
变换，收益源固定 close）。单臂对照 fundind 基线。

### ④ 观点注入三臂对照（`model/views.py` 已入库）

同一观点因子下：BL 优化层注入（`bl_views_from_factor`）vs 预测层注入
（`inject_by_feature_duplication` / `TwoStagePredictor`）vs 不注入；
**须披露观点因子单因子 IC**（观点可能是错的）。基座 = fundind pred。

### ⑤ CVaR 约束组合臂（`optimize/solver.py` 已入库带 6 测试）

`optimize_weights_qp` 的 CVaR_α 约束（RU 线性化，max_iter=200k + eps 1e-8）
vs Top10% 等权 vs buffer 形态，production 口径对拍；基座 = fundind pred。

### ⑥ RD-Agent `fin_factor_report` P2（挂 920 + 好机器 WSL2）

自动 vs 手工转译对照列；产出须过纸面三判据 + 920 IC（设计真源：
`reports/docs/research_notes/微软RD-Agent_研读_*.md`）。

### ⑦ AI97 研报消融补齐（5 组）

因子复杂度约束（研报示 [3,8] 最高 18.71%、[3,7] 崩 −1.56%，极敏感）/
LSTM vs Transformer / IC vs ICIR Pool / 奖励 6 版 / 因子数量扫描。
注意：批次 2 已判 AI97 负结果（PBO 0.704），本项是**研报对照完整性**，
非翻案——产出仅归档，不改负结论。

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

## 批次 13：A1/A2 增量方向 920 定版复验（10-19 立——三臂冻结迁移）

> ❌ **10-21 终判：三臂全败，A1/A2 线关闭**（执行机 = 本工作机，数据面
> D:/data income 补齐后数据态；主信号基座 = defv 面板现场构造并登记
> caliber `dingban_fundind`，锚 16.31%/0.738）：
> ① overlay walk-forward：wf6_w040 ΔSharpe **−0.030**、wf9 −0.026
> （验收 ≥+0.015）——**薄剔除规则不采纳**；固定集 +0.085 与 882 同构 =
> 选择前视上限；placebo −0.036 方向正确（机制真实、增量不存在）。
> ② stack：stage2 残差 IC **0.0742**/IR 0.97（≥0.02 大幅过线，比 882
> 0.0359 翻倍）但 fuse α=0.50 ΔSharpe **−0.041**（验收 ≥+0.08）——
> **不进融合层候选**；「残差 IC 强 ≠ 组合层有增量」= four_paths 机理的
> 又一独立证据（920 基座 0.821 更强，吸收了 stack 增量）。
> ③ h60：配对差 t=0.33/0.78（882: 3.05/7.82）不显著；组合 Sharpe 0.695
> < 基线−0.05=0.771（defv 同窗 0.821）；换手 17.4% ≈ 基线 97%（未显著
> 降）——**「低换手第二策略」不立项**。边界：882 的 selection_h60 实体
> 不在本机，按 920 面板现算（fundamental 6→3 席构成漂移，候选归因之一）。
> **数据态换相注记**：920 残差榜 top5 转为**正向**质量/利润率因子
> （ebit_margin +5.13 / admin_exp_ratio_ttm +4.71 / roic_ttm +3.52），
> 882 态的负向剔除故事不复存在——与 ① 为负自洽。正向质量因子留观
> （fundamental 家族池已有席位，不单独开线）。
> 证据真源 `reports/frontier_audit_920_reval/README.md`。

> 来源：`TODO.md` §六 A1/A2 第二轮（882 口径全过组合层判据，证据
> `reports/frontier_audit_2026-10-19/`）。**冻结迁移原则：882 上调出来的
> 超参（overlay w=0.40、stack α=0.5）与因子集原样带过去，920 只做验证
> 不做重调**——重调会把 920 变成新一轮 in-sample。

### ① A2 负向 overlay（纯回测，<1h；**以 walk-forward 版为准**）

> 10-19 第三轮修正：固定 neg6 的 +0.17 ΔSharpe 里约 7/8 是选择前视
> （`overlay_wf/` walk-forward 复核后仅 +0.02）；复验与采纳判定都用
> walk-forward 臂，固定集臂只作上限参照。

```bash
python -m scripts.evaluation.residual_overlay_probe \
    --signal <920 定版信号登记键或 pred_path> --weights 0.40 \
    --walkforward --wf-csv reports/frontier_audit_920_reval/residual_ic_h1_nonpv_s3.csv \
    --out reports/frontier_audit_920_reval/overlay_wf
```

（`--wf-csv` 指向 920 残差首轮产物；若 920 首轮未跑，先按 ② 步骤 1 补跑。）

**验收**：wf6_w040 ΔSharpe ≥ +0.015 且近年（对应 920 榜的 2023+ 段）逐年
为正、placebo ≤0 ⇒ 作为薄剔除规则评估进出榜链（预期量级 +0.4~1pp/年，
与 882 一致即可，不要求大幅超额）。

### ② A2 二阶 GBDT 融合（残差首轮 + 训练，~2h）

```bash
# 1) 920 残差首轮（若 920 pred 已有残差 CSV 则跳过）
python -m scripts.evaluation.residual_ic_probe --pred-dir <920-pred> --resume \
    --out reports/frontier_audit_920_reval
# 2) stack（特征池自动按 920 榜的早年列重算，无选择前视口径不变）
python -m scripts.evaluation.residual_stack_probe --pred-dir <920-pred> \
    --out reports/frontier_audit_920_reval/residual_stack
```

**验收**：stage2 OOS IC（残差）≥0.02 且 fuse α=0.50 ΔSharpe ≥ +0.08、
placebo ≤0 ⇒ 进融合层候选（与 h1020⊕s0.5 的融合优先级由数字定）。

### ③ A1 h60 第二策略（清单可直用 + 训练，~1h）

`selection_h60/y2023__h60.json` 直接迁移（选特征只依赖 h60 fwd IC，不依赖
pred 口径）；训练与对照切 920 基座：

```bash
python -m scripts.evaluation.retrain_freq_ic --horizon 60 \
    --sel-dir reports/frontier_audit_2026-10-19/selection_h60 --sel-year 2023 \
    --refits 60 --save-pred --oos-begin 20240102 --oos-end 20260901 \
    --out reports/frontier_audit_920_reval --out-tag h60_920
python -m scripts.evaluation.horizon_cross_eval \
    --preds m_h20=<920 h20 pred>,m_h60=reports/frontier_audit_920_reval/h60_920/pred_refit060.parquet \
    --horizons 20,60 --portfolio --baseline <920 定版信号> \
    --out reports/frontier_audit_920_reval/cross_eval
```

**验收**：同目标配对差仍显著（|t|≥2）且组合层 Sharpe ≥ 920 定版基线 −0.05
且换手 ≤60% 基线 ⇒ 「低换手第二策略」立项进定版登记表（后续再做 m_h60 ⊕
主信号融合与 h60 专属 buffer 网格）。

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
