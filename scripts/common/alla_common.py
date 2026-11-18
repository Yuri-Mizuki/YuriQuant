"""全A滚动实验的**共享底座**（2026-10-19 架构清理）。

背景：``scripts/pipelines/rolling_grid_alla.py`` 是实验入口（含 argparse /
``__main__``），却被 25 个脚本反向 import 其内部件（``load_base`` / ``res_metrics``
/ ``FeatureStore`` / ``_rank_average`` ……）——入口一改签名就断一片。
本模块把这些**无副作用**的共享件下沉到 ``scripts/common/``，入口与实验脚本都
只依赖这里；入口脚本自身不再被任何 live 脚本 import（``tests/test_layering.py``
新增自动守卫）。

边界：只放「读面板 + 纯计算 + 常量」，不放进 argparse / 训练 / 阶段编排。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = "all_a_2018_2026"
#: 主实验产物根（``_base`` / ``pred`` / ``selection`` 所在）；与 --preproc 输出
#: 目录无关——``_base`` 永远由主臂 ``stage_prep`` 落到这里。
ROLL_DIR = ROOT / "reports" / "alla_rolling"
#: 🚨 ``_base`` 是**数据不是报告**（中性化要用 market_cap / cov_industry / close），
#: 20+ 处 builder 与 evaluation 反向依赖它 ⇒ 2026-11-18 实体已迁到
#: ``E:/data/factor_library/all_a_2018_2026/base``，本路径只是一个 **junction**。
#: 读路径不变；但**别在清理 reports 时把它当普通目录删**——虽然
#: ``shutil.rmtree`` 对 junction 只删链接（已实测，目标内容不受影响），
#: 删了之后所有中性化/builder 仍会 `FileNotFoundError`。
BASE_DIR = ROLL_DIR / "_base"

#: 基本面慢变量族（A 族股东结构 + B 族财务报表）——用于 DPP 保留席位与
#: ``panels_neu_fundind`` 的「只剥行业」分支。常量随共享底座下沉。
HOLDER_SETS = {
    "holder_num_chg", "holder_num_yoy", "inst_holding",
    "top10_hhi", "top10_holding", "top1_holding", "top5_holding",
}
FUNDAMENTAL_SETS = {
    "ln_mktcap", "float_ratio",
    "ep_ttm", "bp", "sp_ttm", "pcf_ttm", "pcf", "fcf_yield", "netcf_yield", "peg",
    "div_yield", "div_yield_ttm", "div_payout_ratio",
    "roe_ttm", "roa_ttm", "gross_margin", "net_margin", "accruals", "roic_ttm",
    "fin_exp_ratio_ttm", "asset_turnover_ttm", "inv_turnover_ttm", "recv_turnover_ttm",
    "leverage", "rev_growth_yoy", "np_growth_yoy", "np_growth_sq_yoy", "np_growth_sq_qoq",
    "rev_growth_sq_yoy", "rev_growth_sq_qoq", "oppro_growth_sq_yoy", "oppro_growth_sq_qoq",
    "np_ded_growth_ttm_yoy", "cfo_growth_ttm_yoy",
    "goodwill_ratio", "pledge_ratio", "pledge_holder_ratio", "frozen_ratio",
    "profit_notice_chg", "profit_express_np_yoy", "profit_express_rev_yoy",
    "np_ded_ratio", "main_profit_ratio", "ebit_margin",
    "altman_zscore", "debt_to_ebitda", "interest_coverage",
    "np_rev_gap", "rev_recv_gap", "cfo_to_np",
    "margin_delta_ttm", "np_accel_sq", "roe_vol_pit",
    "risky_asset_ratio", "div_growth_yoy", "div_consecutive_years",
    "sue_q", "asset_growth_yoy", "capex_intensity", "spsr", "bonus_freq_3y",
    "ccc", "report_delay", "piotroski_f", "inv_rev_gap",
}
FUNDAMENTAL_FAMILY_SETS = HOLDER_SETS | FUNDAMENTAL_SETS


def ds_root() -> Path:
    """因子库数据集根（``Config.factor_library.root / DATASET``）。"""
    from config import Config

    return Path(str(Config.get()["factor_library"]["root"])) / DATASET


def load_base(base_dir: str | Path | None = None) -> dict:
    """读 ``_base``（复权价 / 市值 / 协变量 / 掩码 / 基准）。

    ``base_dir`` 默认 :data:`BASE_DIR`（主臂 ``reports/alla_rolling/_base``）；
    入口的 ``--preproc`` 只影响 pred/selection 输出，不影响 ``_base``。
    """
    base = Path(base_dir) if base_dir is not None else BASE_DIR
    need = ["close_adj", "market_cap", "tradable_mask", "bench_index", "bench_eqw"]
    missing = [n for n in need if not (base / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"基础面板缺失 {missing}，先跑 --stage prep")
    close_adj = pd.read_parquet(base / "close_adj.parquet")
    cov = {}
    for k in ("size", "industry", "mom", "vol", "turn"):
        p = base / f"cov_{k}.parquet"
        if p.exists():
            cov[k] = pd.read_parquet(p)
    return {
        "close": close_adj,
        "market_cap": pd.read_parquet(base / "market_cap.parquet"),
        "cov": cov,
        "mask": pd.read_parquet(base / "tradable_mask.parquet"),
        "bench_index": pd.read_parquet(base / "bench_index.parquet")["ret"],
        "bench_eqw": pd.read_parquet(base / "bench_eqw.parquet")["ret"],
    }


class FeatureStore:
    """因子面板 LRU 读取（float32，全A面板单个 ~50MB）。

    加载时统一做截面 zscore + ±10σ 剪裁（与因子库"入库即 zscore"及
    multiyear_oos 的加载口径对齐；全A原始公式值含爆炸量级——alpha191_017
    最大 ~3e38，直接进 Ridge 的稠密矩阵会 SVD 崩溃，2026-09-01 实证）。
    IC/去冗余均为秩相关，不受该单调变换影响。
    """

    def __init__(self, panels_dir: Path, cap: int = 130,
                 name_dir: dict[str, Path] | None = None):
        self.dir = panels_dir
        self.name_dir = name_dir or {}   # 逐因子目录覆盖（mixed 口径）
        self.cap = cap
        self._cache: dict[str, pd.DataFrame] = {}
        self._order: list[str] = []

    def get(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            if len(self._order) >= self.cap:
                evict = self._order.pop(0)
                self._cache.pop(evict, None)
            from factor.preprocessing import standardize_zscore
            src = self.name_dir.get(name, self.dir)
            p = pd.read_parquet(src / f"{name}.parquet")
            p = standardize_zscore(p).clip(-10.0, 10.0).astype(np.float32)
            self._cache[name] = p
            self._order.append(name)
        return self._cache[name]

    def get_many(self, names: list[str]) -> dict[str, pd.DataFrame]:
        return {n: self.get(n) for n in names}


# ---------------------------------------------------------------------------
# E5 fundind 口径的分层存储（2026-11-18）
# ---------------------------------------------------------------------------
#: 覆盖层：**只存与基准层内容不同**的因子（基本面/股东族「只剥行业」+ 27 个
#: 事件/资金流/浮点噪声因子），其余一律回落 :data:`FUNDIND_BASE_SUBDIR`。
#: 历史原因：建库脚本按因子集整批落盘，两个目录 950 个同名文件里 863 个逐字节
#: 相同 —— 48.55 GB 纯重复（2026-11-18 实测 md5 全量比对）。
FUNDIND_SUBDIR = "panels_neu_fundind"
FUNDIND_BASE_SUBDIR = "panels_neu"


def fundind_dirs(root: Path | None = None) -> tuple[Path, dict[str, Path]]:
    """E5 fundind 口径的 ``(基准目录, 逐因子覆盖映射)``。

    解析顺序：覆盖层存在该因子 ⇒ 用覆盖层；否则回落基准层。覆盖映射按目录实际
    内容生成（~1e2 个文件，毫秒级），不作静态清单维护。

    Raises:
        SystemExit: 基本面/股东族因子在基准层有面板、覆盖层却没有 —— 那意味着
            该因子会静默退化成「行业+市值全剥」，与 E5 口径不符。
    """
    ds = Path(root) if root is not None else ds_root()
    base, ov = ds / FUNDIND_BASE_SUBDIR, ds / FUNDIND_SUBDIR
    if not base.is_dir():
        raise FileNotFoundError(f"基准层缺失：{base}")
    name_dir = {p.stem: ov for p in ov.glob("*.parquet")}
    missing = sorted(
        n for n in FUNDAMENTAL_FAMILY_SETS
        if (base / f"{n}.parquet").exists() and n not in name_dir)
    if missing:
        raise SystemExit(
            f"E5 覆盖层缺 {len(missing)} 个基本面/股东族因子：{missing[:8]}。"
            f"这些因子会静默按「行业+市值全剥」取值（口径错误）。补救："
            f"python -m scripts.builders.build_alla_factor_neutralized_fundind "
            f"--only {' '.join(missing[:8])}")
    return base, name_dir


def fundind_store(**kwargs) -> FeatureStore:
    """E5 fundind 口径的 :class:`FeatureStore`（基准层 + 覆盖层）。"""
    base, name_dir = fundind_dirs()
    return FeatureStore(base, name_dir=name_dir, **kwargs)


def fundind_names(root: Path | None = None) -> list[str]:
    """E5 fundind 口径下的全部可用因子名（基准层 ∪ 覆盖层，去重有序）。"""
    base, name_dir = fundind_dirs(root)
    return sorted({p.stem for p in base.glob("*.parquet")} | set(name_dir))


def layer_panel_file(name: str, subdir: str,
                     root: Path | None = None) -> Path:
    """``<root>/<subdir>/<name>.parquet``；``panels_neu_fundind`` 走分层解析。

    单因子路径查询不做目录扫描（只 ``exists()``），可在 1e3 量级循环里调用；
    批量场景用 :func:`fundind_dirs` / :func:`fundind_names` 更省。
    """
    ds = Path(root) if root is not None else ds_root()
    p = ds / subdir / f"{name}.parquet"
    if subdir == FUNDIND_SUBDIR and not p.exists():
        return ds / FUNDIND_BASE_SUBDIR / f"{name}.parquet"
    return p


def existence_mask(feats: dict, close: pd.DataFrame,
                   test_days: pd.DatetimeIndex) -> pd.DataFrame:
    """股票在信号日的可用性掩码：当日有行情 且 >=1/4 特征非 NaN。

    全A滚动 IPO 场景的关键守卫（2026-09-01 quick 冒烟发现）：LightGBM 对
    全 NaN 特征行照样输出预测，未上市/新股会占满信号顶部（买入 1110 只
    幽灵股 → 组合近似空仓）。特征可用率掩码把预测限制在"当时真实存在、
    因子已有值"的股票上（PIT 正确）。
    """
    valid = close.loc[test_days].notna()
    if feats:
        avail = None
        for p in feats.values():
            a = p.loc[test_days].notna() if test_days.isin(p.index).all() \
                else p.reindex(index=test_days).notna()
            # 注意 bool+bool 在 pandas 2.x 下仍为 bool（0/1），须显式整型累加
            avail = a.astype(np.int16) if avail is None \
                else avail + a.astype(np.int16)
        valid = valid & (avail >= max(1, len(feats) // 4))
    return valid


def ann(dr: pd.Series) -> float:
    """几何年化（<20 个有效日返回 0，与实验脚本历史行为一致）。"""
    dr = dr.dropna()
    if len(dr) < 20:
        return 0.0
    from backtest.metrics import PERIODS_PER_YEAR
    return float((1 + dr).prod() ** (PERIODS_PER_YEAR / len(dr)) - 1)


def yearly_metrics(dr: pd.Series, bench: pd.Series) -> dict[int, dict]:
    from backtest.metrics import PERIODS_PER_YEAR
    out = {}
    for year, sub in dr.groupby(dr.index.year):
        sub = sub.dropna()
        if len(sub) < 20:
            continue
        b = bench.reindex(sub.index).fillna(0.0)
        ann_ret = ann(sub)
        b_ann = ann(b)
        sd = sub.std()
        eq = (1 + sub).cumprod()
        out[int(year)] = {
            "annual": ann_ret, "excess": ann_ret - b_ann, "bench_annual": b_ann,
            "sharpe": float(sub.mean() / sd * np.sqrt(PERIODS_PER_YEAR))
            if sd > 0 else 0.0,
            "max_dd": float(abs((eq / eq.cummax() - 1).min())),
        }
    return out


def yearly_rows(base_row: dict, dr: pd.Series, bench: pd.Series,
                turnover: pd.Series | None) -> list[dict]:
    """逐年指标行：所有度量列**重算**，绝不从整体行继承（2026-09-14 修 bug）。"""
    from backtest.metrics import PERIODS_PER_YEAR
    keep = ("run_id", "model", "ensemble", "horizon", "freq", "neut",
            "frac", "weight", "arm", "scheme")
    out = []
    for year, ym in yearly_metrics(dr, bench).items():
        idx_y = dr.index[dr.index.year == year]
        ex = (dr.reindex(idx_y) - bench.reindex(idx_y).fillna(0.0)).dropna()
        to_y = (turnover.reindex(idx_y).dropna() if turnover is not None
                else pd.Series(dtype=float))
        out.append({
            **{k: base_row[k] for k in keep if k in base_row},
            "year": year, **ym,
            "excess_idx": ym["excess"],
            "ir": (float(ex.mean() / ex.std() * np.sqrt(PERIODS_PER_YEAR))
                   if len(ex) > 20 and ex.std() > 0 else np.nan),
            "turnover": float(to_y.mean()) if len(to_y) else np.nan,
        })
    return out


def res_metrics(dr: pd.Series, bench: pd.Series,
                turnover: pd.Series | None) -> dict:
    """从日收益序列计算整体指标（首跑与断点续跑同口径）。"""
    from backtest.metrics import calc_all_metrics
    dr = dr.dropna()
    m = calc_all_metrics(dr, bench.reindex(dr.index).fillna(0.0), None)
    to = turnover.dropna() if turnover is not None else pd.Series(dtype=float)
    return {
        "annual": m.get("annual_return", 0.0),
        "excess": m.get("excess_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "sharpe_t_stat": m.get("sharpe_t_stat", np.nan),
        "years_to_prove": m.get("years_to_prove", np.nan),
        "excess_t_stat": m.get("excess_t_stat", np.nan),
        "ir": m.get("information_ratio", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "beta": m.get("beta", np.nan),
        "turnover": float(to.mean()) if len(to) else 0.0,
    }


def rank_average(panels: list[pd.DataFrame], min_panels: int = 2) -> pd.DataFrame:
    """逐日截面秩平均（与 calc_ic 的 Spearman 同口径的单调变换）。"""
    out = pd.DataFrame(np.nan, index=panels[0].index, columns=panels[0].columns)
    if not panels:
        return out
    codes = panels[0].columns
    for d in panels[0].index:
        ranks = []
        for p in panels:
            if d not in p.index:
                continue
            row = p.loc[d]
            r = row.rank(pct=True)
            r = r[~row.isna()]
            ranks.append(r)
        # 对齐当日所有面板的有效代码，取交集计数
        valid_codes = ranks[0].index if ranks else codes[:0]
        for r in ranks[1:]:
            valid_codes = valid_codes.intersection(r.index)
        if len(valid_codes) < 1 or len(ranks) < min_panels:
            continue
        sub = [r.reindex(valid_codes) for r in ranks]
        ok = pd.concat(sub, axis=1).dropna(axis=0)
        if len(ok) < 30:
            out.loc[d, valid_codes] = pd.concat(sub, axis=1).mean(axis=1)
            continue
        out.loc[d, ok.index] = ok.mean(axis=1).astype(np.float32)
    return out
