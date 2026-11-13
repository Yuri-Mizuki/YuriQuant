"""调仓粒度探针的形态等价性守卫（2026-11-13）。

``StaggeredTopKLongOnly`` / ``CohortTrancheTopKLongOnly`` 是「每周调一部分」的
两种实现，都是**有状态**策略。实现时踩过两次静默坑（持仓从 5.9 只缩到 3.7 只、
年化从 36.58% 掉到 33.5%），**回测照跑不报错**，只有逐位比对锚点才能发现。
本文件把「退化到父类」这条不变量钉死，防止以后改 chunk 逻辑时再犯。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluation.rebalance_granularity_probe import (
    CohortTrancheTopKLongOnly,
    StaggeredTopKLongOnly,
)
from strategy.examples import SectorCappedBufferedTopKLongOnly

K = 5


def _make(n: int = 40, days: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=days)
    cols = [f"{i:06d}" for i in range(n)]
    fac = pd.DataFrame(rng.normal(size=(days, n)), index=idx, columns=cols)
    labels = np.array(["A", "B", "C", "D"] * (n // 4), dtype=object)
    sect = pd.DataFrame(np.tile(labels, (days, 1)), index=idx, columns=cols)
    return fac, sect


def test_staggered_chunk_ge_k_equals_parent():
    """``chunk >= k`` 时每期必然换完整个差集 ⇒ 必须与父类逐期同持仓。

    这是最强的退化检验：父类是定版形态，任何偏差都说明 chunk 截断写错了
    （2026-11-13 实测踩坑：``min(len(ins), len(outs))`` 让 annual 从 36.58%
    掉到 33.52%，且持仓数悄悄缩到 3.7）。
    """
    fac, sect = _make()
    base = SectorCappedBufferedTopKLongOnly(sect, k=K, exit_mult=1.0,
                                            max_per_sector=1)
    arm = StaggeredTopKLongOnly(sect, k=K, exit_mult=1.0, max_per_sector=1,
                                chunk=K)
    for d in fac.index:
        wa = base.get_weights_at(d, fac.loc[d])
        wb = arm.get_weights_at(d, fac.loc[d])
        assert set(wa.index) == set(wb.index), d


def test_staggered_chunk1_swaps_at_most_one_each_side():
    """``chunk=1`` 时每期换入 ≤1 且换出 ≤1（部分调仓的定义）。"""
    fac, sect = _make()
    arm = StaggeredTopKLongOnly(sect, k=K, exit_mult=1.0, max_per_sector=1,
                                chunk=1)
    prev: set = set()
    for d in fac.index:
        w = arm.get_weights_at(d, fac.loc[d])
        cur = set(w.index)
        if prev:
            assert len(cur - prev) <= 1, (d, cur - prev)
            assert len(prev - cur) <= 1, (d, prev - cur)
        prev = cur


def test_staggered_holds_no_more_than_k():
    """行业约束会让持仓少于 k，但**任何时候不许超过 k**。"""
    fac, sect = _make()
    for chunk in (1, 2, K):
        arm = StaggeredTopKLongOnly(sect, k=K, exit_mult=1.0,
                                    max_per_sector=1, chunk=chunk)
        for d in fac.index:
            w = arm.get_weights_at(d, fac.loc[d])
            assert len(w) <= K, (chunk, d, len(w))


def test_tranche_weights_sum_to_one_and_first_period_full():
    """分层建仓：首期把 c 层建满（总仓位 1），之后每期权重和恒为 1。"""
    fac, sect = _make()
    for c in (2, 3):
        arm = CohortTrancheTopKLongOnly(sect, k=K, exit_mult=1.0,
                                        max_per_sector=1, cohorts=c)
        for i, d in enumerate(fac.index):
            w = arm.get_weights_at(d, fac.loc[d])
            assert abs(float(w.sum()) - 1.0) < 1e-9, (c, d, float(w.sum()))
            if i == 0:
                # 首期 c 层内容相同 ⇒ 持仓数 = 单层持仓数，不是 c 倍
                one = SectorCappedBufferedTopKLongOnly(
                    sect, k=K, exit_mult=1.0, max_per_sector=1)
                assert len(w) == len(one.get_weights_at(d, fac.loc[d]))


def test_tranche_rebuilds_one_layer_per_period():
    """轮转：第 t 期重建的层 = t % c，其余层名单不变。"""
    fac, sect = _make()
    c = 3
    arm = CohortTrancheTopKLongOnly(sect, k=K, exit_mult=1.0,
                                    max_per_sector=1, cohorts=c)
    for d in fac.index:
        arm.get_weights_at(d, fac.loc[d])
        assert arm._tick <= len(fac.index)
    assert arm._tick == len(fac.index) - 1
    assert len(arm._layers) == c
