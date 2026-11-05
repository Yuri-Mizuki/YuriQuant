"""截面池排除清单（``config/factor_pool.yaml``）的守护测试。

为什么值得单独测
----------------
排除清单是**静默生效**的：名字写错一个字符，那个因子就照旧进池，
而症状（假面板）只在跑完几小时回测后才看得见。所以这里锁三件事：

1. 文件能被解析，且没有重复项；
2. 每个被排除的名字**真的存在于 registry.csv**（防止拼写漂移）；
3. `pool_exclusions` 的缓存与合并语义正确。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.pipelines import rolling_grid_alla as rga

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path("E:/data/factor_library/all_a_2018_2026/registry.csv")


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个用例前后清池缓存，避免互相污染。"""
    rga._POOL_EXCLUDE = None
    yield
    rga._POOL_EXCLUDE = None


def test_pool_file_exists():
    assert (ROOT / "config" / "factor_pool.yaml").exists()


def test_pool_entries_are_unique_and_nonempty():
    names = rga.pool_exclusions()
    assert len(names) == 8, f"预期 8 个结构性无效因子，实得 {len(names)}：{sorted(names)}"
    assert all(n.strip() == n and n for n in names)


def test_every_pool_name_exists_in_registry_or_rejected():
    """拼错一个名字 = 排除静默失效。registry 不在本机时跳过（不误报红）。

    🚨 允许「不在 registry 但被入库闸拒过」：两者是不同机制、不同时点——
    入库闸在写 registry 之前拦（统计判据），排除清单在 select 之前拦（设计判定）。
    `alt_macro_*` 下次重建会被闸门拒收、从 registry 消失，但清单里仍要留着：
    一旦某个数据态重建时 registry 复用了旧快照，清单是最后一道防线。
    """
    if not REGISTRY.exists():
        pytest.skip("registry.csv 不在本机")
    reg = set(pd.read_csv(REGISTRY, usecols=["name"])["name"])
    rej = set()
    for p in REGISTRY.parent.glob("factor_stats_*_rejected.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rej.add(json.loads(line)["name"])
            except (ValueError, KeyError):
                continue
    missing = sorted(rga.pool_exclusions() - reg - rej)
    assert not missing, f"排除清单里有不认识的名字（拼错？）：{missing}"


def test_pool_matches_known_structural_fakes():
    """名单必须与 2026-11-05 的扫描结论一致（6 广播 + 2 常数）。"""
    expect = {
        "alt_macro_chn_surprise_str_5d", "alt_macro_surprise_neg_cnt_1d",
        "alt_macro_surprise_pos_cnt_1d", "alt_macro_surprise_str_5d",
        "alt_macro_surprise_net_20d", "alt_macro_imp3_cnt_20d",
        "alpha360_CLOSE0", "alpha360_VOLUME0",
    }
    assert rga.pool_exclusions() == expect


def test_pool_excludes_interaction_term_not_in_list():
    """反例：宏观 × 个股波动是真截面因子，**不能**被一起扫掉。"""
    assert "alt_macro_surprise_str5_x_vol20" not in rga.pool_exclusions()


def test_pool_is_cached():
    first = rga.pool_exclusions()
    assert rga.pool_exclusions() is first
