"""alt 族（altf 20 因子）的 panels_neu / panels_neu_fundind 回填（family-aware）。

## 背景

altf 20 因子已进 registry + ic 缓存 + ``panels/``，但**未进** ``panels_neu``
（生产中性化库）。后果：09-21 的「面板存在性门槛」（``check_panels_existence``，
防 882→920 那次静默脱钩）在 select 入口**必然失败**，跑 rolling 只能开
``YURIQUANT_PANEL_CHECK=warn`` —— 等于把整条路径安全网关掉。

## 为什么不复用 ``build_alla_factor_neutralized``

该 builder 对全部因子统一走「MAD 去极值 → 行业+市值中性化 → zscore」。实测对
altf 会产出**坏面板**（2026-10-10 定位）：

- **零膨胀计数族**（news/insider：行内 0 占比中位 0.88~0.91）⇒ ``winsorize_mad``
  的 MAD=0 ⇒ 整截面被 clip 成常数 ⇒ 中性化残差 0 ⇒ zscore 全 NaN
  （cov=0.000，**静默死面板**）。对照：库内既有事件因子 ``lhb_count_20d`` /
  ``sue_express_20d`` 用 **NaN 表示"无事件"**（行内零占比 0.00）故安然无恙。
- **市场级广播族**（macro：行 nunique=1）⇒ 截面中性化恒 0 ⇒ zscore 把浮点
  残差放大成 std≈1 的**假面板**（零信息伪装成正常面板，正是工程记忆 §B5 记的陷阱）。

## 本 builder 的处理（按面板形态 data-driven 分派，不硬编码名单）

1. 纯广播（行 nunique≤1）⇒ 截面中性化**无定义**，改做**时序 expanding z**
   （min_periods=60，防前视；与 ``model/market_features.py`` 同型），保留为市场
   状态列。截面常数 ⇒ 树模型不会用它（零信息但不是噪声），满足存在性门槛。
2. 零膨胀（行内 0 占比中位 >0.5）⇒ **跳过 MAD**（``winsorize=None``），其余
   （行业+市值中性化 → zscore）照常。
3. 其余 ⇒ 标准三步。

对 ``panels_neu`` 与 ``panels_neu_fundind`` **同口径各写一份**（altf=事件族不在
``FUNDAMENTAL_FAMILY_SETS``，故 fundind 变体对 alt 与标准变体一致）。

用法::

    python -m scripts.builders.build_alla_altfactor_neu --diagnose   # 只打表
    python -m scripts.builders.build_alla_altfactor_neu             # 回填 altf 两库
    python -m scripts.builders.build_alla_altfactor_neu --only bonus_freq_3y

注：`--only` 可传**任意**因子名（不限于 altf）。生产中另有一个同类漏网者
`bonus_freq_3y`（基本面族，覆盖 0.211 < 标准 builder 的 MIN_COVERAGE=0.3 被
整体跳过，但它**在** ic 缓存里 ⇒ 同样卡存在性门槛），用
`--only bonus_freq_3y` 一并补齐。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = "all_a_2018_2026"
BASE_DIR = Path("reports") / "alla_rolling" / "_base"

ZERO_INFLATED_THRESH = 0.5   # 行内零占比中位 > 此值 ⇒ 判零膨胀（MAD 会失效）
BROADCAST_NUNIQ = 1          # 行 nunique 中位 <= 此值 ⇒ 判市场级广播
MIN_ROW_VALID = 5            # 统计行形态时要求的最小有效样本
EXPAND_MIN_PERIODS = 60      # 时序 z 的最小窗口（防前视）


def _classify(panel: pd.DataFrame) -> tuple[str, dict]:
    """按面板形态判定处理口径，返回 (mode, 诊断 dict)。"""
    nn = panel.notna()
    row_valid = nn.sum(axis=1)
    rows = row_valid >= MIN_ROW_VALID
    nu = panel.nunique(axis=1, dropna=True)
    zero_frac = ((panel == 0.0) & nn).sum(axis=1) / row_valid.replace(0, np.nan)
    nu_med = float(nu[rows].median()) if rows.any() else 0.0
    zf_med = float(zero_frac[rows].median()) if rows.any() else 0.0
    diag = {"row_nuniq_med": round(nu_med, 1), "zero_frac_med": round(zf_med, 3),
            "valid_rows": int(rows.sum())}
    if nu_med <= BROADCAST_NUNIQ:
        return "broadcast", diag
    if zf_med > ZERO_INFLATED_THRESH:
        return "zero_inflated", diag
    return "standard", diag


def _transform(panel: pd.DataFrame, mode: str, mc: pd.DataFrame, ind: pd.DataFrame,
               *, name: str, fundind: bool) -> pd.DataFrame:
    from factor.preprocessing import preprocess_factor

    common = mc.index.intersection(panel.index)
    p = panel.reindex(index=common)
    m = mc.reindex(index=common, columns=p.columns)
    g = ind.reindex(index=common, columns=p.columns)

    if mode == "broadcast":
        # 市场级广播：截面中性化无定义 → 时序 expanding z 后广播（防前视）
        col = p.bfill(axis=1).iloc[:, 0]
        mu = col.expanding(min_periods=EXPAND_MIN_PERIODS).mean()
        sd = col.expanding(min_periods=EXPAND_MIN_PERIODS).std()
        z = (col - mu) / sd.replace(0.0, np.nan)
        x = pd.DataFrame(np.repeat(z.to_numpy()[:, None], p.shape[1], axis=1),
                         index=p.index, columns=p.columns)
    else:
        winsorize = None if mode == "zero_inflated" else "mad"
        # fundind 变体：基本面/股东族只剥行业（与 build_alla_factor_neutralized_fundind 同口径）
        from scripts.pipelines.rolling_grid_alla import FUNDAMENTAL_FAMILY_SETS
        if fundind and name in FUNDAMENTAL_FAMILY_SETS:
            x = preprocess_factor(p, industry_panel=g, winsorize=winsorize)
        else:
            x = preprocess_factor(p, market_cap_panel=m, industry_panel=g,
                                  winsorize=winsorize)
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return x.clip(-10.0, 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="只处理指定因子")
    ap.add_argument("--diagnose", action="store_true", help="只打印判定表，不写盘")
    args = ap.parse_args()

    from config import Config
    ds = Path(str(Config.get()["factor_library"]["root"])) / DATASET
    base = ROOT / BASE_DIR
    mc = pd.read_parquet(base / "market_cap.parquet")
    ind = pd.read_parquet(base / "cov_industry.parquet")

    reg = pd.read_csv(ds / "registry.csv").set_index("name")
    names = (args.only if args.only else
             sorted(n for n in reg.index if reg["set"].get(n) == "altf"))
    print(f"alt 因子 {len(names)} 个；base market_cap {mc.shape}", flush=True)

    hdr = f"{'factor':34s} {'mode':14s} {'行nuniq':>7s} {'零占比':>6s} {'cov':>7s} 输出"
    print(hdr)
    print("-" * len(hdr))

    stats = []
    t0 = time.time()
    for i, n in enumerate(names, 1):
        src = ds / "panels" / f"{n}.parquet"
        if not src.exists():
            print(f"{n:34s} !! 无原始面板，跳过", flush=True)
            continue
        panel = pd.read_parquet(src)
        mode, diag = _classify(panel)
        x = _transform(panel, mode, mc, ind, name=n, fundind=False)
        cov = float(x.notna().mean().mean())
        if not args.diagnose:
            for sub, fin in (("panels_neu", False), ("panels_neu_fundind", True)):
                out = ds / sub
                out.mkdir(parents=True, exist_ok=True)
                arr = x if not fin else _transform(panel, mode, mc, ind,
                                                   name=n, fundind=True)
                arr.to_parquet(out / f"{n}.parquet")
        tag = "已写盘" if not args.diagnose else "(dry)"
        print(f"{n:34s} {mode:14s} {diag['row_nuniq_med']:7.1f} "
              f"{diag['zero_frac_med']:6.2f} {cov:7.4f} {tag}", flush=True)
        stats.append({"name": n, "mode": mode, "coverage": cov, **diag})

    if not args.diagnose:
        (ds / "factor_stats_altf_neu.jsonl").open("w", encoding="utf-8").write(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in stats) + "\n")
    print(f"\n完成 {len(stats)} 个（{time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
