# AGENTS.md — 会话工作约定

> 适用于在本仓库工作的所有 AI 会话。真源分工（2026-10-27 整合后）：
> 活跃待办 = `TODO.md`（唯一）；已收口结论 = `TODO_ARCHIVE.md`（册Ⅰ 工程线 /
> 册Ⅱ 研读复现线 / 册Ⅲ 好机器队列，引用一律指「册 + 节名」不指行号）；
> 报告产物治理规则 = `reports/README.md` §1.6（落盘判定表）。
> 本文件只放跨会话的硬约定，不重复真源内容。

## 1. 报告 / 文档产出纪律（2026-09-24 起；2026-10-28 加校验）

- **先看模板**：新建实验 README 一律复制 `reports/docs/TEMPLATE_实验README.md`，
  四段固定为 `〇 结论先行 / 一 口径与判据 / 二 证据 / 三 复现与边界`；
  `docs/` 下的新文档先读 `reports/docs/README.md` §五 命名与四要素约定。
- **合并优先，不先建新文件**：写分析报告、调研笔记、核对结论之前，先查
  `reports/docs/`（`research_notes/`、`consistency_checks/`、`design/`）里
  有没有同主题的老文件可以**续写或并入**——同日/同主题系列优先合并成册
  （加分隔与导航，内容逐字保留，参见 `research_notes/另类数据调研三部曲_*.md`
  的合并形态）。确无合适载体才新建文件。
- **分类落盘**：新文件按 `reports/README.md` §1.6 落盘判定表选目录；
  `reports/` 根部只允许 README + 4 个接口文件；实验日志放该实验自己的目录，
  不丢根部。
- **删除/移动前**：快照清单（路径+md5 → 仓库外）→ 引用扫描 → 同步改引用
  → 回收站删除（`scripts/oneoff/_recycle_batch.py`），并把记录写进
  `reports/README.md` §四。

## 2. 收尾回写（每个任务完成时自动做，不等用户提醒）

- 结论回写 `TODO.md`（研读 → 对应条目与指针；实验 → 状态、结论、证据路径）；
  条目整段收口后移入 `TODO_ARCHIVE.md` 对应册。
- **新增能力/入口**（新包、新 runner、新数据通道、新因子族）必须同步
  `README.md` 的能力表 / 目录索引；**归档/退休**同理（入口表、scripts 索引、
  依赖它的测试一起处理）。
- **产物落盘后跑文档校验**（2026-10-28 起，与 `pytest` 同级）：
  ```bash
  python -m scripts.data_tools.docs_lint --check   # 违规 rc=1
  python -m scripts.data_tools.docs_lint --verbose # 看完整清单
  ```
  四条规则：① 每个 `reports/<dir>/` 都登记进 `reports/README.md`；
  ② 实验 README 四要素齐备（**只审新写或刚改的**，存量按 mtime 基线豁免）；
  ③ README 标题带日期；④ `reports/docs/` 无孤儿文档。
  违规提示会直接给出改哪一条，别用「加空标题」绕过。
- 回写完成后与代码一起分组提交推送。
