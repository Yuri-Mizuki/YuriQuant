"""事件/两融/资金流数据回补 + 因子构建编排器（串行，防止 SDK 单连接竞争）。

步骤：
  1. 等待 equity_restricted 全A拉满（轮询 parquet，不占 SDK 连接）
  2. 串行回补 margin_detail / long_hu_bang / block_trading（全A）
  3. 依次运行三个因子构建脚本并入 registry + ic_h*

用法（系统 Python 3.12，后台长任务）:
    python -m scripts.builders.run_evt_margin_build [--skip-pull] [--skip-build]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = sys.executable
FULL_CODES = 5810


def _cache() -> Path:
    from config import Config

    return Path(str(Config.cache()["root"]))
PULL_MOD = "scripts.builders.backfill_pledge_profit_alla"
BUILD_ORDER = [
    ("scripts.builders.build_alla_event_factors", "event/evt"),
    ("scripts.builders.build_alla_margin_factors", "margin"),
    ("scripts.builders.build_alla_moneyflow_factors", "moneyflow"),
]


def table_ready(table: str) -> bool:
    """parquet 已落盘且含 code 列即视为完成（backfill 同步返回即为拉完）。"""
    p = _cache() / f"{table}.parquet"
    if not p.exists():
        return False
    try:
        return "code" in pd.read_parquet(p, columns=["code"]).columns
    except Exception:
        return False


def run(cmd: list[str], tag: str) -> bool:
    print(f"\n########## RUN {tag}: {' '.join(cmd)} ##########", flush=True)
    r = subprocess.run([PY, "-m"] + cmd, cwd=ROOT)
    ok = r.returncode == 0
    print(f"########## DONE {tag}: {'OK' if ok else 'FAIL exit=%d' % r.returncode} "
          f"##########", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    ok = True
    if not args.skip_pull:
        for table in ("equity_restricted", "margin_detail",
                      "long_hu_bang", "block_trading"):
            if table_ready(table):
                print(f"[skip] {table} 已就绪", flush=True)
            else:
                print(f"[pull] 回补 {table}", flush=True)
                ok &= run([PULL_MOD, "--batch", "80", "--tables", table],
                          f"pull {table}")

    if not args.skip_build:
        for mod, tag in BUILD_ORDER:
            ok &= run([mod], f"build {tag}")

    print(f"\n========== ALL DONE overall_ok={ok} ==========", flush=True)


if __name__ == "__main__":
    main()