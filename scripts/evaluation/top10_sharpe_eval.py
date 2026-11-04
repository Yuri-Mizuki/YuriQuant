"""十因子增量检验（复用 jq_factor_probe 的评估函数）
=================================================

对 ``top10_sharpe_2026_11`` 数据集里的因子做四项体检，口径与
``scripts/evaluation/jq_factor_probe.py`` 完全一致（全 A、T+1 可交易掩码、
Rank IC + NW t、行业+lncap 残差 IC、全库秩相关去重、月度十分位多空双边千三）：

1. ``ic_h20`` / ``t_h20``    原始 Rank IC 与 Newey-West t
2. ``resid_si_ic`` / ``_t``  控行业 + 市值秩后的增量 IC（回答"是不是风格代理"）
3. ``dup_top1``              与 950 因子面板的最大截面秩相关（回答"是不是换皮"）
4. ``ls_ann`` / ``ls_pass``  月度十分位多空净超额（双边千三）与 5 窗通过数

用法::

    python -m scripts.evaluation.top10_sharpe_eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from scripts.common.alla_common import ds_root, load_base  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation.jq_factor_probe import (  # noqa: E402
    _cs_ic, _dup_batch, _dup_exact, _ic_stats, _rank_panel, _resid_ic_np,
    monthly_ls,
)

log = setup_logging("top10_sharpe_eval")

LIB_DATASET = "top10_sharpe_2026_11"


def main():
    ap = argparse.ArgumentParser(description="十因子增量检验")
    ap.add_argument("--dataset", default=LIB_DATASET)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--dup-dir", default=None, help="去重对照面板目录（默认生产库 panels）")
    ap.add_argument("--dup-stride", type=int, default=20)
    ap.add_argument("--no-dup", action="store_true")
    ap.add_argument("--out-dir", default="reports/top10_sharpe")
    args = ap.parse_args()

    base = load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    idx, cols = close.index, close.columns

    lib_dir = Path(str(Config.get()["factor_library"]["root"])) / args.dataset
    reg_p = lib_dir / "registry.csv"
    if not reg_p.exists():
        raise SystemExit(f"没有 registry：{reg_p}")
    reg = pd.read_csv(reg_p, encoding="utf-8-sig")
    reg = reg[reg["panel_path"].notna()]
    log.info("待检因子 %d 个（%s）", len(reg), args.dataset)

    h = args.horizon
    fwd = (close.pct_change(h, fill_method=None).shift(-h)
           .replace([np.inf, -np.inf], np.nan).astype(np.float32))
    valid = mask & fwd.notna()
    fwd_r = _rank_panel(fwd.where(valid))

    # 残差化协变量：lncap 截面秩（行业以去均值方式进入）
    size_ln = np.log(base["market_cap"].where(base["market_cap"] > 0))
    ind = base["cov"]["industry"]
    ind_codes, _ = pd.factorize(ind.to_numpy().ravel())
    ind_np = ind_codes.reshape(ind.shape)
    n_ind = int(ind_np.max()) + 1
    regs_np = [_rank_panel(size_ln).to_numpy(np.float32)]
    ok_np = valid.to_numpy()

    dup_dir = Path(args.dup_dir) if args.dup_dir else ds_root() / "panels"
    if not args.no_dup and not dup_dir.exists():
        log.warning("去重目录不存在，跳过：%s", dup_dir)
        args.no_dup = True

    rows, ranks = [], {}
    for _, r in reg.iterrows():
        name = str(r["name"])
        p = pd.read_parquet(str(r["panel_path"])).reindex(index=idx, columns=cols)
        p = p.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        cov = float(p.notna().to_numpy(dtype=float).mean())
        xr = _rank_panel(p)
        st = _ic_stats(_cs_ic(xr, fwd_r, valid))
        row = {"name": name, "coverage": round(cov, 3),
               f"ic_h{h}": round(st["ic"], 4), f"t_h{h}": round(st["nw_t"], 2),
               f"ir_h{h}": round(st["ir"], 3), f"win_h{h}": round(st["win"], 3),
               f"n_h{h}": st["n"]}
        rst = _ic_stats(_resid_ic_np(xr.to_numpy(np.float32), fwd_r.to_numpy(np.float32),
                                     ok_np, ind_np, regs_np, n_ind))
        row["resid_si_ic"] = round(rst["ic"], 4)
        row["resid_si_t"] = round(rst["nw_t"], 2)
        ls = monthly_ls(p, close, mask, sign=1)
        row.update({"ls_ann": round(ls.get("ls_ann", np.nan), 4),
                    "ls_sharpe": round(ls.get("ls_sharpe", np.nan), 2),
                    "ls_turn": round(ls.get("ls_turnover_m", np.nan), 3),
                    "ls_win_m": round(ls.get("ls_win_m", np.nan), 3),
                    "ls_pass": f"{ls.get('ls_windows_pass', 0)}/{ls.get('ls_windows_n', 0)}"})
        ranks[name] = xr
        rows.append(row)
        log.info("%-22s IC %.4f (t %.1f) | 残差IC %.4f (t %.1f) | 多空 %.1f%% 换手 %.2f",
                 name, row[f"ic_h{h}"], row[f"t_h{h}"], row["resid_si_ic"],
                 row["resid_si_t"], row["ls_ann"] * 100, row["ls_turn"])

    df = pd.DataFrame(rows)
    if not args.no_dup:
        log.info("全库去重（对照 %d 个面板）...", len(list(dup_dir.glob("*.parquet"))))
        top = _dup_batch(ranks, dup_dir, args.dup_stride)
        dup1, who = [], []
        for name in df["name"]:
            t = top.get(name) or []
            if not t:
                dup1.append(np.nan); who.append("")
                continue
            exact = _dup_exact(ranks[name], dup_dir, t[0][1])
            dup1.append(round(float(exact), 3)); who.append(t[0][1])
        df["dup_top1"] = dup1
        df["dup_with"] = who

        # 增量检验：控「行业 + 市值 + 库内最相似因子」后还剩多少 IC
        log.info("控最相似因子后的增量 IC ...")
        inc_ic, inc_t = [], []
        ok_np32 = ok_np
        for _, r in df.iterrows():
            mate = str(r.get("dup_with") or "")
            p = Path(dup_dir) / f"{mate}.parquet"
            if not p.exists():
                inc_ic.append(np.nan); inc_t.append(np.nan)
                continue
            mate_p = pd.read_parquet(p).reindex(index=idx, columns=cols)
            mate_r = _rank_panel(mate_p.replace([np.inf, -np.inf], np.nan)).to_numpy(np.float32)
            regs2 = [regs_np[0], mate_r]
            st2 = _ic_stats(_resid_ic_np(ranks[r["name"]].to_numpy(np.float32),
                                         fwd_r.to_numpy(np.float32),
                                         ok_np32, ind_np, regs2, n_ind))
            inc_ic.append(round(st2["ic"], 4)); inc_t.append(round(st2["nw_t"], 2))
        df["resid_dup_ic"] = inc_ic
        df["resid_dup_t"] = inc_t

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_p = out / "top10_eval.csv"
    df.to_csv(csv_p, index=False, encoding="utf-8-sig")
    log.info("写出 %s（%d 行）", csv_p, len(df))
    pd.set_option("display.width", 240)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
