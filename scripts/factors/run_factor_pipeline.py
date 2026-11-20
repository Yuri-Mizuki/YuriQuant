"""因子流水线编排：实现 → 评估 → 看板数据 → 看板页面（2026-11-19）。

为什么需要它
------------
四段各有入口，但此前没有编排器：builder 写完面板后要人记着手动跑
``backfill_library_metrics``（把 ingested 补成 evaluated）、再跑
``factor_board_data``（逐因子分位净值/IC），最后 ``factor_board`` 渲染。
任一步忘跑，registry 就停在 ingested、看板就停在上一版——"流水线统一"
只在代码层成立，运行层仍靠人记。

**本脚本只做编排，不含任何计算逻辑**：每一步都是 ``python -m <既有入口>``，
参数原样透传。四段的唯一实现不变，编排层出问题也不会污染数据。

用法::

    # 新增一批因子后（builder 已手动跑完）
    python -m scripts.factors.run_factor_pipeline \
        --names fd_tcc,fd_tcc_g,fd_rnbr --skip-build

    # 全库增量：只补 ingested、只算缺缓存的看板
    python -m scripts.factors.run_factor_pipeline --skip-build

    # 连 builder 一起跑（builder 参数原样透传）
    python -m scripts.factors.run_factor_pipeline \
        --builders build_fd_factors --builder-args "--real --begin 20250101"

    # 只看会执行什么（不落任何产物）
    python -m scripts.factors.run_factor_pipeline --names a,b --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ("build", "evaluate", "board_data", "board")


def build_commands(*, dataset: str, root: str, names: list[str] | None = None,
                   builders: tuple[str, ...] = (), builder_args: str = "",
                   steps: tuple[str, ...] = STAGES
                   ) -> list[tuple[str, list[str]]]:
    """按阶段顺序拼出子进程命令；纯函数，便于测试与 ``--dry-run``。"""
    py = sys.executable
    cmds: list[tuple[str, list[str]]] = []

    if "build" in steps:
        extra = shlex.split(builder_args, posix=False) if builder_args else []
        for mod in builders:
            name = mod if mod.startswith("scripts.") else f"scripts.factors.{mod}"
            cmds.append((f"build:{mod}", [py, "-m", name, *extra]))

    if "evaluate" in steps:
        # 只补 eval_status=ingested / 缺指标的行；已 evaluated 的自动跳过（幂等）
        cmds.append(("evaluate", [py, "-m",
                                  "scripts.factors.backfill_library_metrics",
                                  "--dataset", dataset]))

    if "board_data" in steps:
        argv = [py, "-m", "scripts.factors.factor_board_data",
                "--root", root, "--resume"]
        if names:
            argv += ["--names", ",".join(names)]
        cmds.append(("board_data", argv))

    if "board" in steps:
        cmds.append(("board", [py, "-m", "scripts.reporting.factor_board"]))

    return cmds


def main() -> None:
    from config import Config

    ap = argparse.ArgumentParser(description="因子流水线编排（实现→评估→看板）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--names", default=None,
                    help="逗号分隔的因子名（只影响 builder 之后的看板数据层）")
    ap.add_argument("--builders", default="",
                    help="逗号分隔的 builder 模块名（build_fd_factors 等）；空=跳过实现段")
    ap.add_argument("--builder-args", default="",
                    help="透传给每个 builder 的参数字符串（如 \"--real --begin 20250101\"）")
    ap.add_argument("--steps", default=",".join(STAGES),
                    help=f"要跑的阶段，逗号分隔，可选 {','.join(STAGES)}")
    ap.add_argument("--skip-build", action="store_true", help="等价于 --steps 去掉 build")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--keep-going", action="store_true",
                    help="某段失败后继续跑后续段（默认失败即停）")
    ap.add_argument("--log-file", default=None,
                    help="同时把编排日志与各段 stdout 写入文件（计划任务失败留痕）")
    args = ap.parse_args()

    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    bad = [s for s in steps if s not in STAGES]
    if bad:
        ap.error(f"未知阶段 {bad}；可选 {list(STAGES)}")
    if args.skip_build:
        steps = tuple(s for s in steps if s != "build")
    builders = tuple(b.strip() for b in args.builders.split(",") if b.strip())
    names = ([n.strip() for n in args.names.split(",") if n.strip()]
             if args.names else None)
    root = str(Path(Config.paths()["factor_library_root"]) / args.dataset)

    cmds = build_commands(dataset=args.dataset, root=root, names=names,
                          builders=builders, builder_args=args.builder_args,
                          steps=steps)
    if not cmds:
        print("没有要执行的阶段。")
        return

    logf = open(args.log_file, "a", encoding="utf-8") if args.log_file else None

    def say(msg: str) -> None:
        print(msg, flush=True)
        if logf is not None:
            logf.write(msg + "\n")
            logf.flush()

    say(f"流水线数据集 {args.dataset}｜阶段 {','.join(steps)}｜共 {len(cmds)} 步")
    failed: list[str] = []
    for label, argv in cmds:
        say(f"\n=== [{label}] {' '.join(argv)}")
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(argv, cwd=str(ROOT), stdout=logf,
                            stderr=subprocess.STDOUT if logf else None).returncode
        say(f"=== [{label}] rc={rc}（{time.time() - t0:.0f}s）")
        if rc != 0:
            failed.append(label)
            if not args.keep_going:
                say(f"中止：{label} 失败（rc={rc}）。")
                if logf is not None:
                    logf.close()
                sys.exit(rc)

    if failed:
        say(f"\n完成但有失败阶段：{failed}")
        if logf is not None:
            logf.close()
        sys.exit(1)
    say("\n流水线完成。")
    if logf is not None:
        logf.close()


if __name__ == "__main__":
    main()
