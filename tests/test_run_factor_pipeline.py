"""编排器只拼命令、不做计算：这里锁住阶段顺序与参数透传。"""
from scripts.factors.run_factor_pipeline import build_commands


def test_full_chain_order_and_args():
    cmds = build_commands(dataset="ds", root="/tmp/lib/ds",
                          names=["a", "b"],
                          builders=("build_fd_factors",),
                          builder_args="--real --begin 20250101")
    # 39048cc 起首段 extend（面板延长）、evaluate 后接 ic_fuse（fd/im 通道补 IC）
    assert [c[0] for c in cmds] == [
        "extend", "build:build_fd_factors", "evaluate", "ic_fuse",
        "board_data", "board"]
    assert cmds[1][1][-3:] == ["--real", "--begin", "20250101"]
    assert "--stale-only" in cmds[2][1]
    bd = cmds[4][1]
    assert "--resume" in bd and "--names" in bd and "a,b" in bd
    assert "--root" in bd and "/tmp/lib/ds" in bd


def test_ic_fuse_only_with_names():
    cmds = build_commands(dataset="ds", root="/tmp/lib/ds")
    assert "ic_fuse" not in [c[0] for c in cmds]


def test_steps_selection():
    cmds = build_commands(dataset="ds", root="/tmp/lib/ds", steps=("evaluate",))
    assert [c[0] for c in cmds] == ["evaluate"]
    assert "--dataset" in cmds[0][1] and "ds" in cmds[0][1]


def test_no_builder_no_build_step():
    cmds = build_commands(dataset="ds", root="/tmp/lib/ds")
    assert all(not c[0].startswith("build:") for c in cmds)
