"""
因子库（Factor Library / Factor Zoo）
====================================

把挖掘/合成产出的因子**持久化**为一个可查询、可对比、可迭代的因子库。

解决的问题（用户需求）：
1. 每个因子都能快速查看**某一段时间**的回测结果 —— 通过预存 IC 序列与多套
   canonical 回测的净值/日收益，``evaluate_period`` 按时间段切片、重算指标，秒级返回。
2. 所有因子可以**统一对比**绩效指标、挑选好因子 —— ``compare`` 按任意指标/配置排序。
3. 复合因子**入库并参与下一轮迭代** —— 复合因子记录 ``parents`` 血缘；``load_library_features``
   把它们作为新特征喂给下一轮挖掘（``mine_factors --use-library``）。

持久化结构（root 默认 e:/data/factor_library/）：
    registry.csv              # 所有因子的元数据 + 各配置绩效指标
    panels/<slug>.parquet     # 预计算因子面板 (date × code)，已截面标准化
    evals/<slug>.parquet      # ic 序列 + 各 canonical 配置的 equity/dret（按日期索引）

按数据集分库根（dataset）：
    传入 dataset=<name> 时，上述三件套落在 root/<dataset>/ 子目录下，互不串扰。
    FactorLibrary.list_datasets() 可列出全部数据集；CLI 用 --dataset 指定，
    mine/synthesize 用 --library-dataset（不传则自动推导：真实→<指数>_<年>，mock→mock）。

canonical 回测配置（注册时一次性算好，之后查看/对比都不重算）：
    ls_M  : TopK 多空, k=30, 月调仓
    lo_M  : TopK 纯多, k=30, 月调仓
    ls_W  : TopK 多空, k=30, 周调仓
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import ShortCostModel
from backtest.engine import BacktestResult, VectorBacktest
from backtest.metrics import PERIODS_PER_YEAR, calc_all_metrics
from research.factor_analysis import calc_ic_decay, calc_ic_series, calc_ir, factor_autocorr
from strategy.examples import TopKLongOnly, TopKLongShort

log = logging.getLogger("factor_library")


# ===========================================================================
# canonical 回测配置
# ===========================================================================
class _Config:
    key: str          # 列后缀，如 "ls_M"
    label: str        # 人类可读
    strategy: type    # TopKLongShort / TopKLongOnly
    freq: str         # D / W / M
    k: int = 30

    def __init__(self, key, label, strategy, freq, k=30):
        self.key = key
        self.label = label
        self.strategy = strategy
        self.freq = freq
        self.k = k


CANONICAL_CONFIGS: list = [
    _Config("ls_M", "TopK多空·月", TopKLongShort, "M", 30),
    _Config("lo_M", "TopK纯多·月", TopKLongOnly, "M", 30),
    _Config("ls_W", "TopK多空·周", TopKLongShort, "W", 30),
]

_METRIC_COLS = ["annual_return", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "avg_turnover",
                "avg_margin_usage", "borrow_fee_drag_annual"]


def _slug(name: str) -> str:
    """把因子名（可能是长公式）映射为安全的文件名片段。"""
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def _write_parquet_robust(df: pd.DataFrame, path: Path, retries: int = 3) -> None:
    """健壮写盘：先删旧文件再写 + 失败重试。

    Windows 上 pyarrow 覆盖写已存在 parquet 偶发 PermissionError——外部服务
    （Defender 实时扫描 / 索引）对特定文件的瞬态锁（2026-08-05 实测：文件可
    删除但覆盖写被拒，且只发生在个别文件）。先 unlink 再 to_parquet 绕过
    "打开已存在文件"路径；仍失败则退避重试。
    """
    import time
    for attempt in range(retries):
        try:
            try:
                if path.exists():
                    path.unlink(missing_ok=True)
            except OSError:
                pass  # 沙箱 safe-delete / 回收站不可用时退回覆盖写
            df.to_parquet(path, compression="snappy")
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(1 + 2 * attempt)


def _coerce_date(d) -> pd.Timestamp | None:
    if d is None:
        return None
    if isinstance(d, (int, np.integer)):
        s = str(int(d))
        return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return pd.Timestamp(str(d))


def _sanitize_dataset(ds: str) -> str:
    """把数据集名映射为安全的子目录名。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(ds).strip())


# ===========================================================================
# 因子库
# ===========================================================================
class FactorLibrary:
    """因子持久化仓库。"""

    def __init__(self, root: str | Path | None = None, dataset: str | None = None):
        if root is None:
            try:
                from config import Config
                root = Config.get().get("factor_library", {}).get("root", "e:/data/factor_library/")
            except Exception:
                root = "e:/data/factor_library/"
        self.base_root = Path(root)
        self.dataset = dataset
        # 按数据集分库根：dataset 给定时落到 base_root/<dataset>/，否则用 legacy 默认库（向后兼容）
        self.root = self.base_root / _sanitize_dataset(dataset) if dataset else self.base_root
        self.panels_dir = self.root / "panels"
        self.evals_dir = self.root / "evals"
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.root / "registry.csv"

    @classmethod
    def list_datasets(cls, root: str | Path | None = None) -> list[str]:
        """列出 base_root 下所有含 registry.csv 的数据集名（按数据集分库根）。"""
        if root is None:
            try:
                from config import Config
                root = Config.get().get("factor_library", {}).get("root", "e:/data/factor_library/")
            except Exception:
                root = "e:/data/factor_library/"
        base = Path(root)
        if not base.exists():
            return []
        return sorted(d.name for d in base.iterdir()
                      if d.is_dir() and (d / "registry.csv").exists())

    # ---- registry IO ----
    def _load_registry(self) -> pd.DataFrame:
        if self._registry_path.exists():
            return pd.read_csv(self._registry_path, dtype={"parents": str})
        cols = [
            "name", "kind", "formula", "source", "dataset", "parents", "created_at",
            # 六维标签子集（对齐因子工程实践：家族/频率/成熟度；其余维可放 note）
            "family", "frequency", "maturity", "note",
            "n_dates", "n_codes", "ic_mean", "ic_std", "ic_ir", "t_stat", "t_stat_nw",
            "ic_win_rate", "ic_decay5", "autocorr", "significant",
            # 入库前冗余预检（check_dup）
            "dup_checked", "dup_corr_max", "dup_top", "resid_ic", "resid_t_nw",
        ] + [f"{m}_{c.key}" for c in CANONICAL_CONFIGS for m in _METRIC_COLS] + [
            "best_sharpe", "best_config", "panel_path", "eval_path",
        ]
        return pd.DataFrame(columns=cols)

    def _save_registry(self, df: pd.DataFrame) -> None:
        # 写临时文件再原子替换，避免 read-modify-write 并发入库时损坏 CSV
        tmp = self._registry_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(self._registry_path)

    # ---- 注册 ----
    def register(
        self,
        name: str,
        panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        kind: str = "raw",
        formula: str | None = None,
        parents: list[str] | None = None,
        source: str = "",
        ic_method: str = "spearman",
        short_costs: ShortCostModel | None = None,
        deleverage: bool = False,
        family: str = "",
        frequency: str = "",
        maturity: str = "experimental",
        note: str = "",
        check_dup: bool = False,
        dup_corr: float = 0.7,
        reject_dup: bool = False,
    ) -> dict:
        """注册一个因子：预计算 IC 序列 + 各 canonical 回测，落盘面板/评估/registry。

        Args:
            name: 因子唯一名（建议用公式字符串）。
            panel: date×code 因子面板（建议已截面标准化）。
            returns_panel: date×code 未来一期收益面板。
            kind: 'raw' | 'composite'。
            formula: 人类可读公式（默认=name）。
            parents: 复合因子的父因子名列表（血缘）。
            source: 来源标注（如 "mining:factor_mining_xxx.csv" / "synthesis:ic_weighted"）。
            ic_method: IC 计算方式。
            short_costs: 空头腿成本模型（默认 None=引擎默认：从配置读并启用借券费，
                         修正空头腿乐观偏差；传 ShortCostModel(borrow_rate=0) 关闭）。
            deleverage: 1 倍资金约束（总保证金需求 > 1 时降杠杆）。
            family: 因子家族标签（六维标签之一：动量/反转/波动率/价值/质量/成长/
                    情绪/流动性/拥挤度/技术/非线性组合/其他）。
            frequency: 信号频率（日频/周频/月频/日内…）。
            maturity: 成熟度状态（experimental / oos_verified / active / retired）。
            note: 备注（设计动机、差异化贡献说明等）。
            check_dup: 入库前冗余预检——与库内已有因子算截面相关性 + 对最相关
                       因子做正交残差 IC 检验（对齐因子工程实践：防止因子库
                       "一锅粥"、判断新因子是否只是旧因子的线性组合）。
            dup_corr: 相关性阈值（> 该值判定疑似冗余；默认 0.7，与业界一致）。
            reject_dup: True 时冗余直接抛 ValueError 拒绝入库；False 仅记 warning
                        （个人研究场景默认警告，保留变体对比的灵活性）。
        Returns:
            该因子的 registry 行（dict）。
        """
        slug = _slug(name)
        panel_path = self.panels_dir / f"{slug}.parquet"
        eval_path = self.evals_dir / f"{slug}.parquet"

        # 1) IC 序列
        ic = calc_ic_series(panel, returns_panel, method=ic_method)
        ic_mean = float(ic.mean()) if ic.notna().any() else float("nan")
        ic_std = float(ic.std()) if ic.notna().any() else float("nan")
        ic_valid = ic.dropna()
        n = len(ic_valid)
        ic_ir = calc_ir(ic) if n >= 2 else 0.0
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
        # Newey-West 自相关稳健显著性（IC 序列强自相关时 OLS t 会虚高；
        # significant 判定基于 NW t，2026-08-03 修复，避免累积伪显著）
        from stats.robust_stats import nw_tstat
        t_stat_nw, _se_nw, _lag = nw_tstat(ic_valid) if n >= 2 else (0.0, 0.0, 0)
        ic_win_rate = float((ic_valid > 0).mean()) if n else float("nan")
        significant = bool(abs(t_stat_nw) > 2.0)

        # 1b) IC 衰减 + 截面排名自相关（换手率代理）
        try:
            with np.errstate(all="ignore"):
                _decay = calc_ic_decay(panel, returns_panel, max_lag=10)
            ic_decay5 = float(_decay.get(5, float("nan")))
        except Exception:
            ic_decay5 = float("nan")
        try:
            autocorr = factor_autocorr(panel)
        except Exception:
            autocorr = float("nan")

        # 2) canonical 回测
        eval_cols: dict = {"ic": ic}
        metric_rows: dict = {}
        best_sharpe = -np.inf
        best_config = CANONICAL_CONFIGS[0].key
        for cfg in CANONICAL_CONFIGS:
            bt = VectorBacktest(cfg.strategy(k=cfg.k), rebalance_freq=cfg.freq,
                                short_costs=short_costs, deleverage=deleverage)
            # 库内收益口径为 shift(-1)（第 i 行 = i→i+1 收益，调仓日 t 权重赚
            # rp[t]=t→t+1，无前视）；canonical 回测只算绝对指标、不对齐指数
            # 算 beta/IR，故显式声明关闭引擎的 shift 指纹守卫。
            res = bt.run(panel, returns_panel, check_convention=False)
            dret = res.daily_returns
            eval_cols[f"dret_{cfg.key}"] = dret
            eval_cols[f"equity_{cfg.key}"] = res.equity_curve
            m = res.metrics()
            for mc in _METRIC_COLS:
                if mc in m:
                    metric_rows[f"{mc}_{cfg.key}"] = m[mc]
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_config = cfg.key

        # 3) 落盘（先删旧文件再写，绕过 Windows 覆盖写偶发锁）
        _write_parquet_robust(panel, panel_path)
        eval_df = pd.DataFrame(eval_cols).sort_index()
        _write_parquet_robust(eval_df, eval_path)

        # 4) registry 行
        dup_row = self._run_dup_check(name, panel, returns_panel, check_dup, dup_corr, reject_dup)
        row = {
            "name": name,
            "kind": kind,
            "formula": formula if formula is not None else name,
            "source": source,
            "dataset": self.dataset or "",
            "parents": "|".join(parents) if parents else "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "family": family,
            "frequency": frequency,
            "maturity": maturity,
            "note": note,
            "n_dates": panel.shape[0],
            "n_codes": panel.shape[1],
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "t_stat": t_stat,
            "t_stat_nw": t_stat_nw,
            "ic_win_rate": ic_win_rate,
            "ic_decay5": ic_decay5,
            "autocorr": autocorr,
            "significant": significant,
            **dup_row,
            **metric_rows,
            "best_sharpe": best_sharpe,
            "best_config": best_config,
            "panel_path": str(panel_path),
            "eval_path": str(eval_path),
        }
        reg = self._load_registry()
        reg = reg[reg["name"] != name]  # 覆盖同名
        new_df = pd.DataFrame([row])
        reg = new_df if reg.empty else pd.concat([reg, new_df], ignore_index=True)
        self._save_registry(reg)
        log.info("因子入库: %s (kind=%s, IC=%.4f, best_sharpe=%.3f @%s)",
                 name, kind, ic_mean, best_sharpe, best_config)
        return row

    # ---- 查询 ----
    def has(self, name: str) -> bool:
        reg = self._load_registry()
        return name in set(reg["name"])

    def list_all(self, kind: str | None = None, family: str | None = None,
                 maturity: str | None = None) -> pd.DataFrame:
        reg = self._load_registry()
        if kind is not None:
            reg = reg[reg["kind"] == kind]
        if family is not None:
            fam = reg.get("family", pd.Series("", index=reg.index)).fillna("")
            reg = reg[fam == family]
        if maturity is not None:
            mat = reg.get("maturity", pd.Series("", index=reg.index)).fillna("")
            reg = reg[mat == maturity]
        return reg

    def list_composites(self) -> pd.DataFrame:
        return self.list_all(kind="composite")

    def get_panel(self, name: str) -> pd.DataFrame | None:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return None
        p = Path(hit.iloc[0]["panel_path"])
        return pd.read_parquet(p) if p.exists() else None

    def _load_eval(self, name: str) -> pd.DataFrame | None:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return None
        p = Path(hit.iloc[0]["eval_path"])
        return pd.read_parquet(p) if p.exists() else None

    def load_library_features(self, kind: str | None = None) -> dict:
        """返回 {name: panel}，作为下一轮挖掘的特征集（迭代用）。"""
        reg = self.list_all(kind=kind)
        out: dict = {}
        for _, r in reg.iterrows():
            p = Path(r["panel_path"])
            if p.exists():
                out[r["name"]] = pd.read_parquet(p)
        return out

    def load_significant_features(self, exclude_model: bool = True) -> dict:
        """加载 significant 因子面板（2026-08-31 从 e2e_common 下沉）。

        Args:
            exclude_model: 排除 model:* 来源（模型预测回写因子，面板通常滞后，
                且与预测/回测工作流自身循环引用）。默认 True——这是 e2e
                预测日能到数据末端的关键（model:* 面板截至 2025-12-31）。
        """
        reg = self.list_all()
        sig = reg["significant"].fillna(False).astype(bool)
        mask = sig.copy()
        if exclude_model:
            mask &= ~reg["source"].fillna("").str.startswith("model:")
        sig_names = set(reg[mask]["name"])
        log.info("因子库: %d 个因子, significant %d 个（排除 model:* 后 %d）",
                 len(reg), int(sig.sum()), len(sig_names))

        all_feats = self.load_library_features()
        feats = {k: v for k, v in all_feats.items() if k in sig_names}
        if feats:
            sample = next(iter(feats.values()))
            log.info("加载面板 %d 个, 日期范围 %s ~ %s",
                     len(feats), sample.index[0].date(), sample.index[-1].date())
        return feats

    # ---- 时间段回测查看 ----
    def evaluate_period(self, name: str, start=None, end=None, config: str = "ls_M") -> dict:
        """切片查看某因子在 [start, end] 的回测绩效（秒级，不重算）。

        Returns:
            dict: 含 metrics、ic 统计量、子区间 series、可直接喂报告的 BacktestResult。
        """
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        if config not in [c.key for c in CANONICAL_CONFIGS]:
            raise ValueError(f"未知 config: {config}，可选 {[c.key for c in CANONICAL_CONFIGS]}")
        s, e = _coerce_date(start), _coerce_date(end)
        sub = eval_df.loc[s:e]

        dret = sub[f"dret_{config}"].dropna()
        ic_sub = sub["ic"].dropna()
        metrics = calc_all_metrics(dret)
        ic_mean = float(ic_sub.mean()) if len(ic_sub) else float("nan")
        ic_std = float(ic_sub.std()) if len(ic_sub) else float("nan")
        ic_win = float((ic_sub > 0).mean()) if len(ic_sub) else float("nan")
        ic_ir = calc_ir(ic_sub) if len(ic_sub) >= 2 else 0.0

        bt = BacktestResult(
            daily_returns=dret,
            weights_history=pd.DataFrame(index=dret.index),
            equity_curve=(1 + dret).cumprod(),
            turnover_series=pd.Series(dtype=float),
            cost_series=pd.Series(dtype=float),
            config={"strategy": config, "period": f"{s} ~ {e}"},
        )
        return {
            "name": name, "config": config,
            "start": str(s) if s is not None else None,
            "end": str(e) if e is not None else None,
            "metrics": metrics,
            "ic_mean": ic_mean, "ic_std": ic_std, "ic_win_rate": ic_win, "ic_ir": ic_ir,
            "n_days": len(dret),
            "ic_series": ic_sub,
            "equity_curve": bt.equity_curve,
            "daily_returns": dret,
            "backtest_result": bt,
        }

    def reconstruct_backtest(self, name: str, config: str = "ls_M") -> BacktestResult:
        """从存储的序列重建完整 BacktestResult（用于报告/绘图）。"""
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        dret = eval_df[f"dret_{config}"].dropna()
        return BacktestResult(
            daily_returns=dret,
            weights_history=pd.DataFrame(index=dret.index),
            equity_curve=eval_df[f"equity_{config}"].reindex(dret.index),
            turnover_series=pd.Series(dtype=float),
            cost_series=pd.Series(dtype=float),
            config={"strategy": config},
        )

    # ---- 统一对比 ----
    def compare(
        self,
        metric: str = "ir",
        config: str | None = None,
        ascending: bool = False,
        kind: str | None = None,
        topn: int | None = None,
    ) -> pd.DataFrame:
        """所有因子按统一指标排序，方便挑好因子。

        默认按 **IR（ic_ir，业界统一主轴）** 排序；IR 尺度无关、跨因子可比。
        metric 也支持 ic_mean / sharpe / annual_return / max_drawdown / calmar /
        avg_turnover 等（sharpe/return/calmar/sortino 默认取 best_<metric>）。

        Args:
            metric: 排序指标。
            config: 指定配置列（如 "ls_M"）；为 None 对 sharpe 等用 best_<metric>。
            kind: 仅比较某类（raw/composite）。
            topn: 仅返回前 N 行。
        """
        reg = self.list_all(kind=kind).copy()
        if reg.empty:
            return reg
        # IC 类指标无 config 后缀（单一口径，不随回测配置变，2026-08-05 修复：
        # 原实现把 ic_mean 也拼成 ic_mean_ls_M 导致 KeyError）
        _IC_COLS = {"ic_mean", "ic_std", "ic_ir", "t_stat", "t_stat_nw",
                    "ic_win_rate", "ic_decay5", "autocorr", "significant"}
        if metric in ("ir", "ic_ir"):
            col = "ic_ir"
        elif metric in _IC_COLS:
            col = metric
            if col not in reg.columns:
                raise ValueError(f"列不存在: {col}")
        elif config is not None:
            col = f"{metric}_{config}"
            if col not in reg.columns:
                raise ValueError(f"列不存在: {col}")
        elif metric in ("sharpe", "annual_return", "calmar", "sortino"):
            col = f"best_{metric}" if f"best_{metric}" in reg.columns else f"{metric}_{CANONICAL_CONFIGS[0].key}"
        else:
            col = f"{metric}_{CANONICAL_CONFIGS[0].key}"
        reg = reg.sort_values(col, ascending=ascending).reset_index(drop=True)
        reg["_sort_col"] = col
        if topn is not None:
            reg = reg.head(topn)
        return reg

    # ---- 删除 ----
    def delete(self, name: str) -> bool:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return False
        r = hit.iloc[0]
        for p in (r.get("panel_path"), r.get("eval_path")):
            if p and Path(p).exists():
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    # 文件可能被占用/沙箱回收站不可用：保留文件，仅从 registry 移除
                    log.warning("因子文件删除失败（保留文件）: %s", p)
        reg = reg[reg["name"] != name]
        self._save_registry(reg)
        log.info("已删除因子: %s", name)
        return True

    # ---- 血缘 ----
    def lineage(self, name: str) -> list:
        """返回该因子的父因子链（复合因子才有）。"""
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return []
        parents = hit.iloc[0].get("parents", "")
        return [p for p in str(parents).split("|") if p]

    # ---- 入库前冗余预检（对齐因子工程实践：防"因子库一锅粥"） ----
    def _run_dup_check(self, name, panel, returns_panel, check_dup, dup_corr, reject_dup) -> dict:
        """冗余预检：与库内已有因子算截面相关 + 对最相关因子做正交残差 IC。

        Returns:
            dict: dup_checked / dup_corr_max / dup_top / resid_ic / resid_t_nw。
            check_dup=False 时 dup_checked=False，其余为空。
        """
        empty = {"dup_checked": False, "dup_corr_max": "", "dup_top": "", "resid_ic": "", "resid_t_nw": ""}
        if not check_dup:
            return empty
        hits = []
        reg = self._load_registry()
        for _, r in reg.iterrows():
            if r["name"] == name:
                continue
            p = Path(str(r.get("panel_path", "")))
            if not p.exists():
                continue
            try:
                old = pd.read_parquet(p)
            except Exception:
                continue
            common_dates = panel.index.intersection(old.index)
            common_codes = panel.columns.intersection(old.columns)
            if len(common_dates) < 30 or len(common_codes) < 10:
                continue
            a = panel.loc[common_dates, common_codes]
            b = old.loc[common_dates, common_codes]
            valid = a.notna() & b.notna()
            ra = a.where(valid).rank(axis=1)
            rb = b.where(valid).rank(axis=1)
            corr = ra.corrwith(rb, axis=1, method="pearson").mean()
            if not np.isnan(corr):
                hits.append((float(corr), r["name"]))
        if not hits:
            return {**empty, "dup_checked": True}
        hits.sort(reverse=True)
        top_corr, top_name = hits[0]
        resid_ic, resid_t = self._residual_ic(panel, returns_panel, top_name)
        msg = (f"因子 {name} 冗余预检: 与 {top_name} 相关 {top_corr:.2f}"
               + (f"（>{dup_corr} 疑似冗余）" if top_corr > dup_corr else "")
               + f"；正交残差 IC={resid_ic:.4f} t_nw={resid_t:.2f}"
               + ("（增量信息不足）" if abs(resid_t) < 2.0 else "（含增量信息）"))
        if top_corr > dup_corr:
            log.warning(msg)
            if reject_dup:
                raise ValueError(f"入库被拒（reject_dup）: {msg}")
        else:
            log.info(msg)
        return {
            "dup_checked": True,
            "dup_corr_max": round(top_corr, 3),
            "dup_top": top_name,
            "resid_ic": round(resid_ic, 4),
            "resid_t_nw": round(resid_t, 2),
        }

    def _residual_ic(self, panel, returns_panel, top_name) -> tuple[float, float]:
        """对新因子做「对 top 因子逐日截面回归取残差」，算残差 IC 与 NW t。

        残差 IC 显著（|t|>2）→ 新因子相对库内最相似因子仍含增量信息；
        不显著 → 只是旧因子的（近似）线性组合，入库价值低。
        """
        old = self.get_panel(top_name)
        if old is None:
            return float("nan"), 0.0
        common_dates = panel.index.intersection(old.index).intersection(returns_panel.index)
        common_codes = panel.columns.intersection(old.columns).intersection(returns_panel.columns)
        resid_dates, resid_vals = [], []
        for d in common_dates:
            y = panel.loc[d, common_codes]
            x = old.loc[d, common_codes]
            r = returns_panel.loc[d, common_codes]
            ok = y.notna() & x.notna() & r.notna()
            if ok.sum() < 20:
                continue
            yv = y[ok].values.astype(float)
            xv = x[ok].values.astype(float)
            rv = r[ok].values.astype(float)
            X = np.column_stack([np.ones(len(xv)), xv])
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ coef
            if np.std(resid) < 1e-10:
                # 完全共线（残差为浮点噪声）：无增量信息 → IC=0（而非跳过/随机）
                resid_dates.append(d)
                resid_vals.append(0.0)
                continue
            if len(resid) < 5 or np.std(rv) == 0:
                continue
            ic_d = float(np.corrcoef(resid, rv)[0, 1])
            if not np.isnan(ic_d):
                resid_dates.append(d)
                resid_vals.append(ic_d)
        if len(resid_dates) < 20:
            return float("nan"), 0.0
        from stats.robust_stats import nw_tstat
        t_nw, _se, _lag = nw_tstat(pd.Series(resid_vals, index=resid_dates).values)
        return float(np.mean(resid_vals)), float(t_nw)

    # ---- 标签管理（六维标签子集：家族/频率/成熟度） ----
    def set_tag(self, name: str, family: str | None = None, frequency: str | None = None,
                maturity: str | None = None, note: str | None = None) -> bool:
        """给因子补打/更新标签（不重算任何指标）。"""
        reg = self._load_registry()
        hit = reg["name"] == name
        if not hit.any():
            return False
        for col, val in (("family", family), ("frequency", frequency),
                         ("maturity", maturity), ("note", note)):
            if val is not None:
                if col not in reg.columns:
                    reg[col] = ""
                reg[col] = reg[col].astype(object)  # 字符串写入避免 float64 dtype 冲突
                reg.loc[hit, col] = val
        self._save_registry(reg)
        log.info("已更新标签: %s", name)
        return True

    # ---- 生命周期监控（滚动 IC 漂移） ----
    def monitor(self, window: int = 60) -> pd.DataFrame:
        """库内因子生命周期监控：全期 vs 近期 IC 漂移（复用 evals 已存 IC 序列）。

        Returns:
            DataFrame(name/maturity/ic_mean_full/ic_mean_recent/ic_drift/
                      ic_ir_recent/ic_t_nw_recent/status)，warning 排前。
        """
        from stats.monitor import monitor_ic_series
        reg = self._load_registry()
        rows = []
        for _, r in reg.iterrows():
            p = Path(str(r.get("eval_path", "")))
            if not p.exists():
                continue
            try:
                ic = pd.read_parquet(p)["ic"].dropna()
            except Exception:
                continue
            if len(ic) < 20:
                continue
            m = monitor_ic_series(ic, window=window)
            rows.append({
                "name": r["name"],
                "kind": r.get("kind", ""),
                "maturity": str(r.get("maturity", "")),
                "family": str(r.get("family", "")),
                "ic_mean_full": m["ic_mean_full"],
                "ic_mean_recent": m["ic_mean_recent"],
                "ic_drift": m["ic_drift"],
                "ic_ir_recent": m["ic_ir_recent"],
                "ic_t_nw_recent": m["ic_t_nw_recent"],
                "n_days": m["n_days"],
                "status": m["status"],
            })
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows)
                .sort_values(["status", "ic_mean_recent"], ascending=[False, False])
                .reset_index(drop=True))

    # ---- 分市场状态检验（八维检验之一） ----
    def regime_analysis(self, name: str, market_returns: pd.Series | None = None,
                        n_tiles: int = 3) -> dict:
        """按市场状态（牛/熊/震荡，按市场收益分位）分段看因子 IC。

        Args:
            name: 因子名。
            market_returns: 市场日收益 Series（默认从日线缓存读等权市场）。
            n_tiles: 分段数（3=熊/震荡/牛）。
        Returns:
            dict: {段名: {ic_mean, ir, win_rate, n_days, market_ann}}。
        """
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        ic = eval_df["ic"].dropna()
        if len(ic) < 30:
            raise ValueError(f"IC 样本过少（{len(ic)} < 30），无法分市场状态")
        if market_returns is None:
            market_returns = self._default_market_returns(ic.index)
        if market_returns is None or market_returns.empty:
            raise RuntimeError("无法获取市场收益（需要日线缓存或显式传入 market_returns）")
        mr = market_returns.reindex(ic.index).dropna()
        ic = ic.reindex(mr.index).dropna()
        mr = mr.reindex(ic.index)
        if len(ic) < 30:
            raise ValueError("市场收益与 IC 对齐后样本过少")
        seg = pd.cut(mr.rank(pct=True), bins=n_tiles,
                     labels=["熊/弱市", "震荡市", "牛/强市"] if n_tiles == 3 else None)
        out = {}
        for label in seg.cat.categories:
            mask = (seg == label)
            s = ic[mask]
            m = mr[mask]
            out[str(label)] = {
                "ic_mean": float(s.mean()),
                "ir": float(calc_ir(s)) if len(s) >= 2 else 0.0,
                "win_rate": float((s > 0).mean()),
                "n_days": int(len(s)),
                "market_ann": float((1 + m.mean()) ** PERIODS_PER_YEAR - 1),
            }
        return out

    def _default_market_returns(self, dates: pd.Index) -> pd.Series | None:
        """从日线缓存构造等权市场日收益（次期口径，与 IC 对齐）。"""
        try:
            from data.cache import DataCache
            from data.cache_helpers import returns_from_cache
            from data.offline import OfflineDataSource
            cache = DataCache(OfflineDataSource())
            begin = int(str(dates.min().date()).replace("-", ""))
            end = int(str(dates.max().date()).replace("-", ""))
            returns = returns_from_cache(cache, begin, end)
            return returns.mean(axis=1)
        except Exception:
            return None

    # ---- 集合级多样性筛选（DPP，研报系列之二十四 §3.1） ----
    def select_diverse(self, names: list[str] | None = None, k: int | None = None,
                       method: str = "cross", sigma: float = 0.2,
                       quality_col: str | None = "ic_mean",
                       min_overlap_dates: int = 30,
                       min_overlap_codes: int = 10) -> dict:
        """对库内因子做 DPP 集合级多样性筛选（log-det 最大化，去冗余）。

        对比现有两两去重（check_dup / select_low_corr）：DPP 是集合级全局判据，
        不会因三角相关结构（A~B、B~C 高相关，A~C 独立）连锁误杀；结果与顺序无关。

        Args:
            names: 候选因子名（None=库内全部）；可按 source/family 预过滤后传入。
            k: 目标入选数（None=ceil(0.7 × n_pool)，对齐研报 800/1134≈0.7）。
            method: 相关口径，"cross"（逐日截面相关均值）或 "flat"（flatten）。
            sigma: 相似度核带宽（小→对高相关惩罚强，默认 0.2）。
            quality_col: registry 质量列（默认 ic_mean → 质量=|IC| 归一化）；
                         None=纯多样性（对齐研报 DPP 口径）。
            min_overlap_dates / min_overlap_codes: 面板公共样本下限。

        Returns:
            dict: selected(list[str]) / k / n_pool / 各 summary 指标（含质量保留率）。
        """
        from research.dpp_selection import corr_matrix, dpp_select
        reg = self._load_registry()
        if names is None:
            names = list(reg["name"])
        name_set = set(names)
        panels: dict[str, pd.DataFrame] = {}
        meta: dict[str, dict] = {}
        for _, r in reg[reg["name"].isin(name_set)].iterrows():
            p = Path(str(r.get("panel_path", "")))
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            if df.empty:
                continue
            panels[r["name"]] = df
            meta[r["name"]] = dict(r)
        if not panels:
            raise RuntimeError("无可用因子面板（请先入库）")
        corr = corr_matrix(panels, method=method,
                           min_overlap_dates=min_overlap_dates,
                           min_overlap_codes=min_overlap_codes)
        quality = None
        if quality_col is not None:
            ic = pd.Series({n: meta.get(n, {}).get(quality_col, float("nan"))
                            for n in corr.index}, dtype=float)
            ic = ic.fillna(0.0)
            from research.dpp_selection import quality_from_ic
            quality = quality_from_ic(ic)
        if k is None:
            k = int(np.ceil(0.7 * len(corr)))
        res = dpp_select(corr, k=k, quality=quality, sigma=sigma)
        # 质量保留率（入选子集 vs 候选池的 |IC| 均值）
        def _mean_abs(col: str) -> float:
            vals = [meta.get(n, {}).get(col) for n in res["selected"]]
            pool = [meta.get(n, {}).get(col) for n in corr.index]
            v = pd.Series(vals, dtype=float).abs().dropna()
            p = pd.Series(pool, dtype=float).abs().dropna()
            return float(v.mean()) if len(v) else float("nan"), \
                   float(p.mean()) if len(p) else float("nan")
        if quality_col is not None and quality_col in reg.columns:
            sel_m, pool_m = _mean_abs(quality_col)
            res["quality_mean_selected"] = sel_m
            res["quality_mean_pool"] = pool_m
        log.info("DPP 筛选: 池 %d → %d 因子, max|corr| %.3f→%.3f, mean|corr| %.3f→%.3f",
                 res["n_pool"], res["k"], res["max_abs_corr_pool"],
                 res["max_abs_corr_selected"], res["mean_abs_corr_pool"],
                 res["mean_abs_corr_selected"])
        return res
