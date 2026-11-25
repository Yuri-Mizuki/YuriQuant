# YuriQuant 待办

> 只写「做什么 + 结论 + 证据路径」。2026-11-25 前的完整流水已并入 `TODO_ARCHIVE.md` 册Ⅳ。

## 当前状态

- 全A 定版 `yqcore-breadth`：**21.81% / Sharpe 1.008 / MDD 27.3%**（27bp，2019-01 起）。
  集中档 k15 **30.13% / 1.144**、k8 **34.54% / 1.221** —— 高收益来自集中度，代价是 ~38% 回撤
  （`reports/yq_core/README.md`）。
  ⚠️ 三个数字口径不齐：breadth/k15 是 uniform（1e6 无整手），k8 是 native（1e5 + 整手）⇒
  不可相减。2026-11-25 起 k8 的 `exit_mult` 由 1.0 上调到 **5.0**（21 相位配对 +1.32pp、
  t 2.86），native 重算后是 **35.01% / 1.229 / MDD 42.6% / 换手 84.4%**；b1.0 的 34.54%
  是历史臂。`reports/yq_core/README.md` 正文表格**尚未按 5.0 重写**（数字以
  `report_native.json` 为准），见 `reports/k8_buffer_scan/README.md`。
- ETF 两档：`monthly` 默认、`balanced` 备选，`core` 已退役；对外一律引 5 相位均值
  （`reports/etf_contest5/README.md`）。
- `inc_pure`（董监高增持，H=20）：毛超额 +14.53%、扣费 +11.95%、t 3.23；必须开盘集合竞价成交
  （操作手册见 `TODO_ARCHIVE.md` ⑨·补·12，出单器 `scripts/live/inc_pure_live.py`）。
- 估值定仓位规则已交付，只作个人纪律件，参数冻结、不入生产。
- 研报 DL / 行业选因子 / 纯择时三线均已判负或收线，只剩 `sue_low`（行业）与 79 个 H=1
  单因子候选待验。

## P0

- **② 行业轮动补「合法基线」（阻塞）**：`baselines.yaml` 无行业基线，`sue_low +10.82pp` 只能
  线内比较；建「31 行业等权买入持有」+「行业 `mom12_1`」并落指纹。
  → `reports/industry_rotation/README.md`
- **③ `sue_low` 可信化三连**：行业原生 SUE 代理 / 45% 静态成分检验 / 倒 U 中间组；判据 =
  过 ② 的基线且净超额 ≥+2pp，否则降级为「风格」不立项。
- **⑨·补·5 前视修复残余（阻断级）**：P0-B 行业轮动主表 / `report_factors` 57 因子 /
  `industry_factor_lib` 全库扫描未重跑；P0-H pred 停在 737/970（续跑清单 `logs/stale_all.txt`，
  alpha 上限 4 进程）；P0-F 三个计划任务需手工 `task_scheduler install`。
  → `reports/industry_rotation/oos_validation.md`
- **⑨·补·13 `inc_pure` H=10 / H=20 纸面双轨 A/B 40 日**：单边 <30bp 切 H=10（净额高 5.67pp），
  >48bp 维持 H=20。→ `TODO_ARCHIVE.md` ⑨·补·13
- **`keep_system_awake` 本机无效（P0 运维）**：应用层 API 压不住待机；唯一真解是系统层
  `PlatformAoAcOverride=0`（需管理员 + 重启）。

## P1

- **⑬ yqcore 三档 Barra 风格归因（未启动）**：把「超额」拆成「因子 beta 的回报」与「真残差 α」。
  现状是**只控了市场一个因子**：breadth 相对全A月频等权 α +7.67%/年（t 3.02）看着过关，但它
  已测暴露是 resid_vol **z −22**、btop **z +24**、股息 z +9 ⇒ 这 7.67% 极可能是低波 + 价值
  溢价的打包价，控住风格后大概率显著缩水。k8/k15 同理，问题更尖锐：74% 收益能被微盘宽基
  解释（`microcap_ref`），剩下的 +5.85pp 是 alpha 还是「微盘内部的低波」没验过。
  **做法**：`research/barra.py` 的 13 个风格面板 → 月度风格组合收益 → 对三档月收益做
  多因子时序回归（规模 / 低波 / 价值 / 动量 / 流动性 + 市场），报 α 与 HAC t。判据 =
  控住风格后 α 仍 >0 且 **t > 2**，否则三档一律降级为「风格产品」、不许再叫 alpha。
  → `reports/breadth_alpha/README.md`（P1 原始读数）· `reports/style_exposure/README.md`（暴露快照）
- **④ 模型层唯一未启动线索：反转 / 低位类欠拟合**：`alpha191_046` / `alpha360_LOW7` 在生产
  `panels_neu` 上 resfac t 7.7 / 8.1；做法 = 残差空间加两族或 gbdt 加单调约束，
  对照 `v1w-ortho882` 走 `caliber`。
- **⑥ 研报表 1 的 `week30` 四配置**：日线重采样周线后重跑（12 因子里 4 个是周频）。
- **⑫ 库内去重残余**：12 个新 `pap_*` 缺 neu 面板 + ic 列、进不了定版；冗余索引仍为 950 快照，
  需重建纳入新 pap。→ `reports/library_governance/dedup_execution.md`
- **因子库全A指标批量回填（另一台机器）**：`formula` / `family` 已回填，`annual_return_ls_M` /
  `significant` / `panel_path` 仍待跑 `scripts/factors/backfill_library_metrics.py`。
  → `reports/library_governance/backfill_summary.json`
- **无线网卡 `Netwtw14` 7021 掉线**：≥11 次/天则 altdata 必缺表；只能在脚本内做表级重试 /
  断点续跑（任务计划器重启条件对此无效）。

## P2 / 墙后 / 可选

- **⑦ 纯动量 `mom_240_20`（+6.42% / t(IR) 1.42）是否并入生产**：先按 ② 补动量族基线再判；
  研报选的 `mom_180_20` 我们只有 +3.49%，属事后可见差异，不得当强弱证据。
- **⑧ 资源墙不变**：分钟扩容仍卡全 A 分钟特征 / 华泰3128 / e2e 分钟链；本批 P0 不依赖分钟数据。
- **纯择时研究线（活跃）**：待在有数据的机器按 T+1（含周一修正）重跑改写 `multisrc` /
  `hf_minagg` / `frontier` 三份 README；新信息源只剩情绪面板与期权历史；
  唯一采纳候选 = `rev_shell@000001` 独立腿。
- **赛后残余三件**：账户层仓位分配（待拍板）/ 模式 B 双轨影子（先只落 14:50 快照）/
  相位分散（周一三五三子账户）。
- **H 扫描 v2 后续**：全库画像已出（fast 170 / mid 20 / slow 920），79 个过双门槛候选全部
  H*=1，待独立样本外验证。→ `reports/factor_dict_probe/hsweep_best_H.csv`
- **`fd_ent30` 全A 版**：已入库（IC +0.0464 / t 25.6），下一步做组合层增量检验
  （与低波 / 流动性族并集）。
- **华泰 AI3128 全频段**：同一资源墙，墙后复评。
- **生产级执行**：实盘下单对接、实时行情驱动（当前日频盘后信号，滑点未经真实成交验证）。
- **文本挖掘合规化**：同花顺 / 巨潮爬虫依赖，生产化需商业源或明确「研究性补充」定位。
- **在线 dashboard**：报告均为静态 HTML，缺增量刷新 / 定时生成（`reports/yq_core/dashboard.html`、
  `reports/factor_board/`、`reports/strategies/`）。
- **监控链收敛**：`monitor_performance` 复活前提 = 全A 库 evals 补齐 + runner 行情源参数化。
- **`all_a_2018_2026` 纳入正式因子库监控**（目前轻量 registry，非阻塞）。
- **邮件链路 Skills 化（可选）**：`build_daily_email_body` 口径约束固化。
- **RD-Agent `fin_factor_report`**：本机无 WSL2/Docker，转待有环境的机器；key 已就绪。
- 华泰 **AI32/34 AlphaNet**：GP/GFlowNet 之外第三条路径对照。
- 华泰 **AI26 GP 在 CTA**：跨资产泛化验证。
- 华泰 **AI42 GNN 选股**：图结构关系建模。
- 华泰 **AI13 损失函数 / AI27 可解释性 / AI40 Qlib / AI45 cGAN / AI35 WGAN**：辅助参考。
- **AI39 残余**：① 主动权重口径立项（待拍板）；② Barra 应用侧（注意 QP 家族已全关）。
- **C1 模型异质性 / meta-labeling**：条件性，仅当 A/B 增量线出正结果才投入。
- **P6 DoubleEnsemble 式特征×样本重加权**：条件性，基本面贡献仍低才做。
- **optimize P3**：完整多期最优执行、风险预算非等权（册Ⅰ §四）。

## 已否（别再挖）

- ①·附 六条改法：降换手对快信号无效；holdout + BH-FDR 是「IC 先打 0.27 折」的最硬负结果
  （`reports/report_replica/six_fixes/README.md`）。
- ②·附 直接训行业模型：IC 天花板 0.0443，31×85 样本量是硬约束
  （`reports/report_replica/industry_direct/README.md`）。
- ②·附2 中性化审计：放宽不是解，0.033 产生在「集成」这一步；59 个假面板留 P2 清理
  （`reports/report_replica/p0a/README.md` §五）。
- ⑤ 行业→ETF 覆盖率：可投 24/31，7 行业至今无专属产品（`reports/etf_coverage/`）。
- ⑨ 四篇研报 57 因子：只有 `sue_q`（反向）显著，加因子提 IR 关闭
  （`reports/industry_rotation/report_factors.md`）。
- ⑨·附 / 补 / 补·2：唯一三口径都过的是 `sue_q`；解释改为「行业内龙头的盈利惊喜」。
- ⑨·补·3：拿单因子比研报合成值属标准错配；等权合成反而更差。
- ⑨·补·8 事件当滤网：六族全负；⑨·补·14 L3 事件链 / L4 行业密度全负，四层探完不再开第五层。
- ⑨·补·9/10/11/12 `inc_pure`：策略成立、无可调参数、必须开盘集合竞价买；
  是第二条 alpha 来源而非独立风险腿。
- ⑩ RAFT 式样本加权：负增量（`TODO_ARCHIVE.md` 册Ⅱ ⑩）。
- ⑪ EV 分母 / 研发加回：`bp_ev` +3.63pp 有效、研发加回与合成判负
  （`reports/factor_ev_rd/README.md`）。
- λ 维持 0.5、h20-only 不换、h40 不新增腿、池子探针不宜做沪深300 指增。
- 打板 / 涨停线：半路 + 次日接力双判负，触板排队不立项。
- ETF 侧：−15% 降档纪律取消、VWAP 判否、buffer 不采纳、换权重与单只上限判否、模式 B 仅备选。
- 多空：仅作研究/诊断口径（beta 归零），融券容量被政策关掉，不作实盘方案
  （`reports/ls_arm_probe_2026-11-24/`）。
- 方正「高频因子低频化」16 篇：91 行里已实现 6 个，且无公式、Fitness 不可横比
  （`reports/founder_hf_probe_2026-11-24/`）。
- 「高开要跑」：是执行纪律不是策略，主动套利扣费后净 0（`reports/gap_fade_probe_2026-11-24/`）。
- 分钟因子 A/B（im15）：G1 PASS ΔIC +0.0091 / G2 FAIL Δ年化 −2.7pp ⇒「信息有、经济无」，
  im25 与 fddict 已砍。
- 第三实验 fd 臂：`fdnat` 双 FAIL，与 im15 合并 ⇒「字典因子进月频模型」双路径判负闭环。
- 周频方向：三次证伪（半衰期假设不成立、成本墙、h5/h20 = 0.73），瓶颈在数据信息结构。
- AlphaSAGE 编码器、awesome-systematic-trading、中国因子十强、QuantsPlaybook：增量不可分辨或不立项。
- 纯择时已收口十一探针：申万 / 华福 / 多源合成 / 缝合 / 前沿源 / 量价四臂 / 三腿组合 / 降频 /
  vr / 基差 / ETF 申赎调样，全判负或收线。
