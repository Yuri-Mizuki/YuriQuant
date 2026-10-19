"""残差目标探针：现有 920 因子池里还有没有「主模型漏掉」的正交信息（2026-10-19）。

## 定位

`TODO.md` §六 A2（2026-10-19 增量立项）。此前项目只做过**单因子残差 IC 诊断**
（`text_factor_probe.py` 的 `res=...` 列：把因子对主信号 rank 正交化后看 IC），
**没有把「残差」当成目标做一轮系统搜索**。本脚本补这一层：

- `raw`      ：因子对原始未来收益的 IC（与生产 IC 同口径，作参照）；
- `resfac`   ：因子先对主信号 rank 正交化，再对原始未来收益算 IC
               （= 「这个因子还有没有主信号之外的信息」）；
- `restgt`   ：未来收益先对主信号 rank 正交化，再算因子对残差收益的 IC
               （= 「这个因子能不能解释主模型漏掉的那部分收益」）；
- `both`     ：两边都正交化后的 IC（最干净的「增量信息」口径）。

三者都走**可交易掩码**（`_base/tradable_mask`，T+1 口径）与 Newey-West t
（重叠 horizon 时自动加滞后），并输出逐年方向稳定性。

## 判据（与 `four_paths/README.md` §三 对齐）

> 新因子族准入门槛 = **自身可交易 IR ≥ 0.3**，不是「低相关」。

残差口径下，`resfac` / `restgt` / `both` 的 |IC| 本来就该远小于 raw（主信号已
 span 掉大部分），所以本脚本**不直接拿 0.3 卡残差 IC**；它回答两个更基础的问题：

1. **还有没有**：`both` 的 NW t 在全池里是否还有 |t|>3 且逐年稳定的因子？
2. **够不够**：这些因子的 `both` IC 量级能否支撑一个二阶模型（经验上 |IC|≥0.01
   且 t≥3 才值得进 GBDT；|IC|<0.005 基本是噪声）。

## 口径与边界

- 主信号默认 `reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet`（**882 ortho**
  本机唯一 pred；920 定版 pred 在好机器，结论须在 920 复验）。
- 因子面板默认 `panels_neu`（生产池 950 张）；`--panels-dir panels` 可切 raw。
- 收益 = `close_adj.pct_change(h).shift(-h)`；IC = 截面 Spearman（`rank+corrwith`
  与 `stats.ic.calc_ic_series` 逐位一致，已用测试锁死）。
- 残差化 = 逐日一元 OLS（与 `text_factor_probe.residual_panel` 同一实现），
  **不重排残差秩**；`both` 口径下两边都先 rank 再残差化，最后对残差重排秩算
  Spearman（单调变换不改 Spearman）。
- 内存：逐因子读 → 算 → 释放；单因子峰值约 6 个 2107×5801 float32 矩阵，
  前台 ~1.3GB 限制下可跑（实测单因子 2~4s）。

## 用法

    # 快速冒烟（前 20 个因子）
    python -m scripts.evaluation.residual_ic_probe --max-factors 20

    # 全池 h=1（约 40~70 分钟，建议后台）
    python -m scripts.evaluation.residual_ic_probe --resume

    # 长 horizon 复核（h=20 / h=60）
    python -m scripts.evaluation.residual_ic_probe --horizon 20 --out-tag h20

    # 只跑指定因子
    python -m scripts.evaluation.residual_ic_probe --only accruals bp sue_q
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("residual_ic_probe")

DATASET = "all_a_2018_2026"


def resolve_ds_root(explicit: str | None) -> Path:
    """定位因子库数据集根：显式路径 > `Config.paths()` 单一真源。"""
    from config import Config

    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    root = Config.paths().get("factor_library_root")
    if root:
        cands.append(Path(root) / DATASET)
    for c in cands:
        if (c / "registry.csv").exists():
            log.info("因子库根 = %s", c)
            return c
    raise FileNotFoundError(
        f"因子库 root 不可用：{[str(c) for c in cands]}；"
        "检查 config/settings_local.yaml 的 factor_library.root")


def resid_rank(sig_rank: pd.DataFrame, main_rank: pd.DataFrame) -> pd.DataFrame:
    """逐日一元 OLS 残差；与 `text_factor_probe.residual_panel` 同一实现。

    独立实现是为了不在本脚本 import 另一个入口脚本（layering 守卫），
    测试 `test_residual_ic_probe.py` 对两者做逐位一致断言。
    """
    mm = main_rank.reindex(index=sig_rank.index, columns=sig_rank.columns)
    v = sig_rank.notna() & mm.notna()
    a = sig_rank.where(v)
    b = mm.where(v)
    ya = a.sub(a.mean(axis=1), axis=0)
    yb = b.sub(b.mean(axis=1), axis=0)
    var = (yb ** 2).mean(axis=1)
    cov = (ya * yb).mean(axis=1)
    beta = cov / var.replace(0.0, np.nan)
    return ya.sub(yb.mul(beta, axis=0)).where(v)


def _ic_stats(ic: pd.Series) -> dict:
    """IC 均值 / IR / 胜率 / Newey-West t / 有效天数。"""
    from stats.robust_stats import nw_tstat

    s = ic.dropna()
    if len(s) < 2:
        return dict(ic=float("nan"), ir=float("nan"), win=float("nan"),
                    nw_t=float("nan"), n=0)
    sd = float(s.std())
    t_nw, _se, _lag = nw_tstat(s.to_numpy(dtype=float))
    return dict(ic=float(s.mean()), ir=float(s.mean() / sd) if sd else float("nan"),
                win=float((s > 0).mean()), nw_t=float(t_nw), n=int(len(s)))


def _yearly(ic: pd.Series) -> dict:
    """逐年 IC 均值 + 正年数（方向稳定性）。"""
    out: dict = {}
    if ic.empty:
        return out
    for y, sub in ic.groupby(ic.index.year):
        out[f"ic_{y}"] = float(sub.mean())
    vals = [v for k, v in out.items() if k.startswith("ic_")]
    out["years_pos"] = int(sum(v > 0 for v in vals))
    out["years_n"] = int(len(vals))
    out["years_sign_consistent"] = int(max(out["years_pos"],
                                           out["years_n"] - out["years_pos"]))
    return out


def _load_panel(fp: Path, idx: pd.DatetimeIndex,
                cols: pd.Index) -> pd.DataFrame | None:
    """读一张因子面板并对齐到 (idx, cols)；非数值/空面板返回 None。"""
    raw = pd.read_parquet(fp)
    if not isinstance(raw.index, pd.DatetimeIndex):
        try:
            raw.index = pd.to_datetime(raw.index)
        except Exception:
            log.warning("索引非日期，跳过: %s", fp)
            return None
    if raw.columns.has_duplicates:
        raw = raw.loc[:, ~raw.columns.duplicated()]
    raw = raw.select_dtypes(include=[np.number])
    if raw.shape[1] == 0:
        return None
    x = raw.reindex(index=idx, columns=cols)
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return x


def one_factor(x: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
                main_rank: pd.DataFrame, main: pd.DataFrame) -> dict | None:
    """单因子四口径 IC。返回 None 表示覆盖率不足/全空。"""
    valid = x.notna() & fwd.notna() & mask
    n_valid = int(valid.to_numpy(dtype=bool).sum())
    if n_valid < 1000:
        return None
    cov = float(x.notna().to_numpy(dtype=float).mean())

    # raw：与生产 IC 同口径（valid 仅由因子/收益/掩码决定）
    xr_raw = x.where(valid).rank(axis=1)
    yr_raw = fwd.where(valid).rank(axis=1)
    ic_raw = xr_raw.corrwith(yr_raw, axis=1).astype(float)

    # 残差口径：因子与主信号都必须有值（v 是更严的子集）
    v = valid & main.notna()
    if int(v.to_numpy(dtype=bool).sum()) < 1000:
        return None
    xr = x.where(v).rank(axis=1)
    mr = main.where(v).rank(axis=1)
    yr = fwd.where(v).rank(axis=1)

    xres = resid_rank(xr, mr)
    xres_r = xres.where(v).rank(axis=1)
    fres = resid_rank(yr, mr)
    fres_r = fres.where(v).rank(axis=1)

    ic_resfac = xres_r.corrwith(yr, axis=1).astype(float)
    ic_restgt = xr.corrwith(fres_r, axis=1).astype(float)
    ic_both = xres_r.corrwith(fres_r, axis=1).astype(float)

    row: dict = {"cov": cov, "n_valid": n_valid,
                 "rank_corr_main": float(xr.corrwith(mr, axis=1).mean())}
    for tag, s in (("raw", ic_raw), ("resfac", ic_resfac),
                   ("restgt", ic_restgt), ("both", ic_both)):
        st = _ic_stats(s)
        row.update({f"{tag}_ic": st["ic"], f"{tag}_ir": st["ir"],
                    f"{tag}_win": st["win"], f"{tag}_nw_t": st["nw_t"],
                    f"{tag}_n": st["n"]})
        if tag == "both":
            row.update(_yearly(s))
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds-root", default=None, help="因子库数据集根（默认探测）")
    ap.add_argument("--panels-dir", default="panels_neu",
                    help="因子面板子目录：panels_neu（默认）/ panels（raw）")
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred",
                    help="主信号目录（默认 882 ortho）")
    ap.add_argument("--main-signal", default="ens_h1h5__h1.parquet")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out", default="reports/frontier_audit_2026-10-19")
    ap.add_argument("--out-tag", default="", help="输出文件后缀（如 h20）")
    ap.add_argument("--max-factors", type=int, default=None,
                    help="只跑前 N 个（配 --sort-by-ic 时取最强的 N 个；冒烟用）")
    ap.add_argument("--sort-by-ic", action="store_true",
                    help="按 registry 的 |ic_mean_h1| 降序（配 --max-factors 做首轮筛查）")
    ap.add_argument("--stride", type=int, default=1,
                    help="评估日抽样步长（>1 为快速筛查，IC 仍按抽样日算）")
    ap.add_argument("--only", nargs="+", default=None,
                    help="只跑指定因子名（不含 .parquet）")
    ap.add_argument("--sets", default=None,
                    help="逗号分隔 registry set 过滤（如 fundamental,evt,margin）")
    ap.add_argument("--min-coverage", type=float, default=0.05,
                    help="因子面板非 NaN 覆盖率下限（默认 0.05，挡死面板）")
    ap.add_argument("--resume", action="store_true",
                    help="跳过输出 CSV 中已完成的因子")
    ap.add_argument("--flush-every", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=50,
                    help="额外输出按 both_nw_t 排序的 top K")
    args = ap.parse_args(argv)

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.out_tag}" if args.out_tag else ""
    out_csv = dest / f"residual_ic_h{args.horizon}{suffix}.csv"
    top_csv = dest / f"top_residual_h{args.horizon}{suffix}.csv"
    meta_json = dest / f"meta_h{args.horizon}{suffix}.json"

    from scripts.common.alla_common import load_base

    ds = resolve_ds_root(args.ds_root)
    pdir = ds / args.panels_dir
    if not pdir.exists():
        raise FileNotFoundError(f"面板目录不存在：{pdir}")

    base = load_base()
    close = base["close"]
    mask_all = base["mask"].astype(bool)

    main = pd.read_parquet(Path(args.pred_dir) / args.main_signal)
    main = main.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    idx = main.index
    if args.stride > 1:
        idx = idx[:: args.stride]
        log.info("评估日抽样 stride=%d -> %d 日（快速筛查口径）",
                 args.stride, len(idx))
    cols = main.columns.intersection(close.columns)
    log.info("主信号 %s: %d 日 × %d 列 | OOS %s ~ %s",
             args.main_signal, len(idx), len(cols), idx[0].date(), idx[-1].date())

    fwd = (close.pct_change(args.horizon, fill_method=None)
           .shift(-args.horizon)
           .reindex(index=idx, columns=cols)
           .replace([np.inf, -np.inf], np.nan)
           .astype(np.float32))
    mask = mask_all.reindex(index=idx, columns=cols).fillna(False).astype(bool)
    main = main.reindex(index=idx, columns=cols)
    main_rank = main.rank(axis=1, pct=True).astype(np.float32)
    log.info("h=%d | 有效收益日 %d | 主信号覆盖 %.3f",
             args.horizon, int(fwd.notna().any(axis=1).sum()),
             float(main.notna().to_numpy(dtype=float).mean()))

    files = sorted(pdir.glob("*.parquet"))
    reg_p = ds / "registry.csv"
    reg = pd.read_csv(reg_p) if reg_p.exists() else None
    if args.only:
        wanted = set(args.only)
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            log.warning("指定因子不存在：%s", sorted(missing))
    else:
        # 只保留 registry 中登记过的（panels_neu 基本都登记；防目录里混入临时件）
        if reg is not None:
            registered = set(reg["name"].astype(str))
            extra = {p.stem for p in files} - registered
            if extra:
                log.info("未登记面板 %d 个将被跳过（示例 %s）",
                         len(extra), sorted(extra)[:5])
            files = [p for p in files if p.stem in registered]
            if args.sets:
                allowed = {s.strip() for s in args.sets.split(",") if s.strip()}
                set_by_name = dict(zip(reg["name"].astype(str),
                                       reg["set"].astype(str)))
                files = [p for p in files
                         if set_by_name.get(p.stem) in allowed]
                log.info("set 过滤 %s -> %d 个", sorted(allowed), len(files))
    if args.sort_by_ic and reg is not None and "ic_mean_h1" in reg.columns:
        strength = (reg.set_index("name")["ic_mean_h1"].abs()
                    .dropna().to_dict())
        files = sorted(files, key=lambda p: strength.get(p.stem, 0.0),
                       reverse=True)
    if args.max_factors:
        files = files[: args.max_factors]
    log.info("待评估 %d 个因子（%s）", len(files), pdir)

    done: set[str] = set()
    if args.resume and out_csv.exists():
        try:
            done = set(pd.read_csv(out_csv, usecols=["name"])["name"].astype(str))
            log.info("resume: 已完成 %d 个，跳过", len(done))
        except Exception:
            log.warning("resume 读取失败，从头开始：%s", out_csv)

    rows: list[dict] = []
    t0 = time.time()
    n_skip = 0
    for i, fp in enumerate(files, 1):
        name = fp.stem
        if name in done:
            continue
        x = _load_panel(fp, idx, cols)
        if x is None:
            n_skip += 1
            continue
        cov = float(x.notna().to_numpy(dtype=float).mean())
        if cov < args.min_coverage:
            n_skip += 1
            log.info("[%4d/%d] %-24s 跳过（cov=%.3f < %.2f）",
                     i, len(files), name, cov, args.min_coverage)
            del x
            continue
        row = one_factor(x, fwd, mask, main_rank, main)
        if row is None:
            n_skip += 1
            log.info("[%4d/%d] %-24s 跳过（有效样本不足）", i, len(files), name)
            del x
            continue
        row = {"name": name, **row}
        rows.append(row)
        log.info("[%4d/%d] %-24s cov=%.2f | raw=%+.4f(t=%.2f) | "
                 "resfac=%+.4f(t=%.2f) | restgt=%+.4f(t=%.2f) | "
                 "both=%+.4f(t=%.2f) | corr(m)=%+.3f | 正年 %d/%d",
                 i, len(files), name, row["cov"], row["raw_ic"],
                 row["raw_nw_t"], row["resfac_ic"], row["resfac_nw_t"],
                 row["restgt_ic"], row["restgt_nw_t"], row["both_ic"],
                 row["both_nw_t"], row["rank_corr_main"],
                 row.get("years_pos", 0), row.get("years_n", 0))
        del x
        gc.collect()

        if len(rows) % max(1, args.flush_every) == 0:
            _flush(rows, out_csv, done)

    if rows:
        _flush(rows, out_csv, done, final=True)
    if not out_csv.exists():
        log.warning("没有产出（全部跳过）")
        return

    df = pd.read_csv(out_csv)
    df = df.sort_values("both_nw_t", ascending=False)
    df.head(args.top_k).to_csv(top_csv, index=False, encoding="utf-8-sig")
    meta = {
        "horizon": args.horizon,
        "stride": args.stride,
        "n_eval_days": int(len(idx)),
        "sort_by_ic": bool(args.sort_by_ic),
        "panels_dir": args.panels_dir,
        "main_signal": args.main_signal,
        "pred_dir": args.pred_dir,
        "n_files": len(files),
        "n_evaluated": int(len(df)),
        "n_skipped": n_skip,
        "elapsed_s": round(time.time() - t0, 1),
        "out_csv": str(out_csv),
        "top_csv": str(top_csv),
        "caveat": ("主信号为 882 ortho（本机唯一 pred）；920 定版须在好机器用 "
                   "--pred-dir 复验。残差 IC 已对主信号 rank 正交化，"
                   "正值 = 主信号之外的增量方向，不等于可直接交易的 alpha。"),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    log.info("完成 %d 个（跳过 %d）%.0fs -> %s",
             len(df), n_skip, time.time() - t0, out_csv)
    log.info("Top 5 by both_nw_t:\n%s",
             df.head(5)[["name", "both_ic", "both_nw_t", "resfac_ic",
                         "restgt_ic", "rank_corr_main"]].to_string(index=False))


def _flush(rows: list[dict], out_csv: Path, done: set[str],
           final: bool = False) -> None:
    """增量落盘（先读旧文件再合并，防中途失败丢已完成行）。"""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        new = pd.concat([old, new], ignore_index=True)
        new = new.drop_duplicates(subset=["name"], keep="last")
    new.to_csv(out_csv, index=False, encoding="utf-8-sig")
    done.update(new["name"].astype(str))
    rows.clear()
    log.info("flush -> %s（累计 %d 行）", out_csv, len(new))


if __name__ == "__main__":
    main()
