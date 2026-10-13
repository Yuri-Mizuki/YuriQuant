"""配置加载：读取 settings.yaml，支持环境变量占位符替换 + 本机覆盖。

两层结构（解决「同一仓库、多台机器数据盘符不同」）：

1. ``config/settings.yaml``（**tracked，仓库默认值**）——只放**跨机通用**的缺省路径，
   绝对盘符写成 ``${YQ_DATA_ROOT:D:/data}`` 形式，便于整根切换。
2. ``config/settings_local.yaml``（**本机私有，已 gitignore**）——按**深合并**覆盖默认值，
   用于「本机某类数据在别的盘」这种逐键差异（例如 E 盘放 factor_library、
   D 盘放 parquet 缓存）。存在即生效，不存在则完全走默认值（零回归）。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_LOCAL_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings_local.yaml"

# Real-data credentials are kept in the repository-local .env file.  Loading
# it here keeps command-line runs and IDE runs consistent; existing process
# environment variables still take precedence.
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_ENV_PATTERN = re.compile(r"\$\{([A-Z_]+)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """递归替换 ${VAR:default} 占位符为环境变量值。"""
    if isinstance(value, str):

        def _repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)

        return _ENV_PATTERN.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def expand_env(value: Any) -> Any:
    """公开的占位符展开（供 settings.yaml 以外的 yaml 复用，如 config/schedule.yaml）。

    规则同 ``load_config``：把 ``${VAR:default}`` 换成环境变量值，未设置则用 default。
    这样「某台机器特有的绝对路径」只需登记一次默认值，换机器时用环境变量覆盖即可。
    """
    return _expand_env(value)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归深合并：dict 逐键递归，其余类型由 override 整体替换。

    不做 list 合并（列表语义上「整段替换」比逐项拼接更可预期）。
    """
    out = dict(base)
    for k, v in override.items():
        cur = out.get(k)
        if isinstance(v, dict) and isinstance(cur, dict):
            out[k] = _deep_merge(cur, v)
        else:
            out[k] = v
    return out


def load_config(path: Path | str | None = None) -> dict:
    """加载 yaml 配置并展开环境变量占位符。

    Args:
        path: 显式配置路径。传 None（默认）时读取 ``config/settings.yaml``，
            并叠加同目录下的 ``settings_local.yaml``（若存在）；**显式传 path
            时只读该文件、不叠加本机覆盖**（便于测试与「只想要纯默认值」的场景，
            传默认路径本身也算显式）。

    Returns:
        展开 ``${VAR:default}`` 后的配置字典。
    """
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if path is None and _LOCAL_CONFIG_PATH.exists():
        with open(_LOCAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
        if isinstance(local, dict) and local:
            raw = _deep_merge(raw or {}, local)
    return _expand_env(raw)


class Config:
    """单例式配置访问器。"""

    _data: dict | None = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> dict:
        cls._data = load_config(path)
        return cls._data

    @classmethod
    def get(cls) -> dict:
        if cls._data is None:
            cls.load()
        return cls._data  # type: ignore

    @classmethod
    def datasource(cls) -> dict:
        return cls.get()["datasource"]

    @classmethod
    def cache(cls) -> dict:
        return cls.get()["cache"]

    @classmethod
    def paths(cls) -> dict:
        """数据路径**单一真源**（多机适配，2026-10-13）：取数据盘路径一律走这里。

        背景：两台机器的数据盘符不同（一台 D:/data、一台 E:/data），且迁移只搬了
        一部分（本机 parquet 在 D:、factor_library 在 E:）。脚本里硬编码
        ``E:/data/...`` / ``D:/data/...`` 会让另一台机器直接 FileNotFoundError，
        而这类失败**不会在改配置时暴露**（只有真正跑到那行才炸）。

        路径来源（见 ``config/settings.yaml`` 文件头「路径约定」）：
        ``${YQ_DATA_ROOT:<默认根>}`` 占位符 → 环境变量或默认值 →
        再被 ``config/settings_local.yaml``（本机私有）逐键覆盖。

        Returns:
            {"cache_root": 行情/另类 parquet 缓存根（parquet 表直接拼文件名）,
             "factor_library_root": 因子库根（下面挂 <dataset>/ 数据集目录）,
             "textmining_model_dir": 本地 FinBERT 模型目录,
             "sdk_local_path": AmazingData SDK 本地缓存目录}

        新增消费方请走这里，不要在脚本里硬编码盘符——否则另一台机器上必断。
        """
        c = cls.get()
        ds = c.get("datasource") or {}
        amazing = (ds.get("amazing_data") or {}) if isinstance(ds, dict) else {}
        return {
            "cache_root": str(cls.cache().get("root") or ""),
            "factor_library_root": str((c.get("factor_library") or {}).get("root") or ""),
            "textmining_model_dir": str((c.get("textmining") or {}).get("bert_model_dir") or ""),
            "sdk_local_path": str(amazing.get("sdk_local_path") or ""),
        }

    @classmethod
    def universe(cls) -> dict:
        return cls.get()["universe"]

    @classmethod
    def fetch(cls) -> dict:
        return cls.get()["fetch"]

    @classmethod
    def discipline(cls) -> dict:
        """L2 段落契约：冻结日历（train/valid/test 边界单一真源）。

        Returns:
            {"begin", "train_end", "valid_end"}（8 位整型日期）；
            test 段 = valid_end 之后，只允许最终验证与上线后监控。
        """
        d = cls.get().get("discipline") or {}
        return {
            "begin": int(d.get("begin", 20220101)),
            "train_end": int(d.get("train_end", 20231231)),
            "valid_end": int(d.get("valid_end", 20241231)),
        }

    @classmethod
    def costs(cls) -> dict:
        """交易成本（全项目**唯一真源**，缺省值与 settings.yaml 的 `costs` 段一致）。

        2026-09-10 收敛：此前费率分散在 `backtest` 段与 `model_portfolio` 段两处，
        差 2~3 倍、互相不可比。现在引擎缺省费率（backtest/engine.py）与生产管线
        （scripts/run_model_portfolio.default_costs）都经由本方法取值。

        Returns:
            {commission_rate, commission_min, stamp_duty, slippage_bp,
             short_borrow_rate, short_margin_ratio,
             impact_model, impact_eta, impact_buy_multiplier, impact_max}

        新增消费方请走这里，不要在别处硬编码费率字面量——否则 config 改动后会漂移。
        """
        c = cls.get().get("costs") or {}
        return {
            "commission_rate": float(c.get("commission_rate", 0.0003)),
            "commission_min": float(c.get("commission_min", 5.0)),
            "stamp_duty": float(c.get("stamp_duty", 0.001)),
            "slippage_bp": float(c.get("slippage_bp", 10.0)),
            "short_borrow_rate": float(c.get("short_borrow_rate", 0.08)),
            "short_margin_ratio": float(c.get("short_margin_ratio", 1.0)),
            # 逐股冲击成本模型（P0-1，2026-10-13）：默认 none（用固定 slippage_bp）
            "impact_model": str(c.get("impact_model", "none")),
            "impact_eta": float(c.get("impact_eta", 0.4)),
            "impact_buy_multiplier": float(c.get("impact_buy_multiplier", 1.5)),
            "impact_max": float(c.get("impact_max", 0.10)),
        }

    @classmethod
    def benchmarks(cls) -> dict:
        """对照基准指数（各报告/策略的"与谁比"**唯一真源**）。

        2026-09-11 收敛：此前指数代码散落在 scripts 里（`rolling_grid_alla.BENCH_INDEX`
        = 000001.SH、`jq_style_report.BENCH_INDEX` = 399317.SZ、
        `run_etf_rotation.BENCH` = 000300.SH），加上 `backtest.benchmark` 与
        `model_portfolio.benchmark`，共 5 处字面量、改一个要满仓找。
        ⚠️ **三者取值本就不同**（用途不同，非同一概念的重复），故这里按**用途**分键，
        不强行合并成一个值——合并会静默改变某条链路的对照基准。

        Returns:
            {"default": 引擎缺省对照（backtest.benchmark）,
             "all_a": 全A组合对照（上证指数，rolling_grid_alla / model_portfolio）,
             "etf": ETF 轮动对照（沪深300）,
             "report_a_share": 全A报告图表对照（国证A指，中证全指行情缺失的替代）,
             "report_a_share_label": 上者的展示名}

        新增消费方请走这里，不要在别处硬编码指数字面量。
        """
        b = cls.get().get("benchmarks") or {}
        bt = cls.get().get("backtest") or {}
        mp = cls.get().get("model_portfolio") or {}
        default = str(b.get("default") or bt.get("benchmark") or "000300.SH")
        return {
            "default": default,
            "all_a": str(b.get("all_a") or mp.get("benchmark") or "000001.SH"),
            "etf": str(b.get("etf") or "000300.SH"),
            "report_a_share": str(b.get("report_a_share") or "399317.SZ"),
            "report_a_share_label": str(
                b.get("report_a_share_label") or "国证A指(399317)≈中证全指"),
        }

    @classmethod
    def bert_model_dir(cls) -> str:
        """本地 FinBERT 模型目录（把 ``//`` 归一化为 ``/``，供 transformers 直接消费）。

        6 处 textmining 脚本此前各自 ``try: Config... except: 回落硬编码盘符`` ——
        在另一台机器上会**静默指向不存在的目录**。统一走这里；未配置时直接报错，
        而不是猜一个盘符。
        """
        d = str(cls.paths().get("textmining_model_dir") or "").replace("//", "/")
        if not d:
            raise RuntimeError(
                "textmining.bert_model_dir 未配置：请检查 config/settings.yaml")
        return d

    @classmethod
    def monitoring(cls) -> dict:
        """生产化监控阈值（monitoring/ 包单一真源，缺省值与 settings.yaml 一致）。"""
        m = cls.get().get("monitoring") or {}
        return {
            "window": int(m.get("window", 60)),
            "window_long": int(m.get("window_long", 252)),
            "max_stale_days": int(m.get("max_stale_days", 7)),
            "min_coverage": float(m.get("min_coverage", 0.5)),
            "warn_ic_retention": float(m.get("warn_ic_retention", 0.5)),
            "min_monotonicity": float(m.get("min_monotonicity", 0.5)),
            "min_t_nw_recent": float(m.get("min_t_nw_recent", 1.0)),
            "ledger_root": str(m.get("ledger_root", "reports/monitoring")),
        }
