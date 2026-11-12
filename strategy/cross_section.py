"""横截面选股策略的正名模块（``TopFracLongOnly`` / ``BufferedTopFracLongOnly`` 等）。

为什么会有这个文件
------------------
2026-11-03 做过一次「``examples.py`` → ``cross_section.py``」的归位重构：把横截面
选股策略从"示例"里搬出来，因为 ``TopFracLongOnly`` / ``BufferedTopFracLongOnly``
是**生产策略**（``caliber.py`` 的全部基线读它们、出榜链 ``alla_daily_rank`` 也读），
挂在 ``examples`` 名下名不副实。

但那次提交**只改了调用方、新文件没进 git**（典型的 ``git commit -a`` 不带未 track
文件），于是 8 处 ``from strategy.cross_section import ...`` 全部指向不存在的模块。
干净 clone 上直接 ImportError 的不只是几个实验脚本，还包括 ``caliber.py``——
它是全项目评估口径的唯一真源，起不来等于所有基线没法复算。

本文件把这条导入路径补回来。**实现仍在 :mod:`strategy.examples`**（47 处调用方
在那里），这里只做显式再导出，两套路径同时可用。

迁移方向
--------
新代码写 ``strategy.cross_section``。等 ``examples`` 里的横截面策略真正迁完
（连同那段 tie-break 语义说明），本模块升级为定义处，``examples`` 反向再导出。
在那之前不要在两边都改实现——**单一实现、两个入口**。
"""
from __future__ import annotations

# ``pool_scope_probe`` 直接用这个确定性排序（并列按列序打破），是模块私有名但
# 被跨模块消费，故在兼容层显式透出，避免调用方各自重写一遍排序。
from strategy.examples import (
    BufferedTopFracLongOnly,
    BufferedTopKLongOnly,
    QuantileLongShort,
    SectorCappedBufferedTopKLongOnly,
    SectorCappedTopKLongOnly,
    TopFracLongOnly,
    TopKLongOnly,
    TopKLongShort,
    rank_desc,
    build_strategy,
)

__all__ = [
    "TopKLongShort",
    "TopKLongOnly",
    "SectorCappedBufferedTopKLongOnly",
    "SectorCappedTopKLongOnly",
    "TopFracLongOnly",
    "BufferedTopFracLongOnly",
    "BufferedTopKLongOnly",
    "QuantileLongShort",
    "build_strategy",
    "rank_desc",
]
