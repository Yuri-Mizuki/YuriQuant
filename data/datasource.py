"""
数据源抽象层
============

定义统一的 DataSource 接口，所有具体数据源（AmazingData / CSV / ...）
实现该接口。上层 cache / universe / factor / backtest 只依赖此接口，
切换数据源时只需改 config + 新增一个实现类，不动业务代码。

数据约定
--------
- 所有行情返回 multi-index DataFrame：(date, code) 索引，列统一为
  open / high / low / close / volume / amount。
- 所有日期统一为 pandas.Timestamp（内部用 int8 日期到 SDK 层转换）。
- 代码格式统一 "XXXXXX.SH/SZ/BJ"（6位+交易所后缀）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd


def _to_datetime_safe(s) -> pd.Series:
    """日期列安全解析。

    int 型 YYYYMMDD（如 20240105）必须按 ``%Y%m%d`` 解析——``pd.to_datetime``
    会把整数当纳秒时间戳，得到 1970 年的垃圾时间戳（正是 calendar 那类 PIT
    bug 的根因）。其余 dtype（str / datetime）走自动解析。
    """
    s = pd.Series(s) if not isinstance(s, pd.Series) else s
    if pd.api.types.is_integer_dtype(s):
        return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


# 手册 4.1.7「数据周期 Period」支持的分钟档位（Period.minN）
VALID_MINUTE_PERIODS = frozenset({1, 3, 5, 10, 15, 30, 60, 120})


def validate_minute_period(period: int) -> int:
    """校验分钟档位，非法抛 ValueError。返回原值（便于链式调用）。"""
    if period not in VALID_MINUTE_PERIODS:
        raise ValueError(
            f"period 必须是 {sorted(VALID_MINUTE_PERIODS)} 之一（对应手册 "
            f"Period.minN），got {period}"
        )
    return period


class DataSource(ABC):
    """数据源抽象基类。"""

    # ---- 交易日历 ----
    @abstractmethod
    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        """返回交易日列表（8位整型 YYYYMMDD，升序）。"""

    # ---- 代码表 ----
    @abstractmethod
    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        """获取每日最新代码表。"""

    @abstractmethod
    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        """指数成分股。
        返回列: con_code, in_date, out_date。"""

    # ---- 行情 ----
    @abstractmethod
    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        """日K线。返回 multi-index (date, code) DataFrame，
        列: open/high/low/close/volume/amount。"""

    @abstractmethod
    def get_minute_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
        period: int = 5,
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        """分钟K线（AmazingData 手册 3.5.4.2 query_kline + 4.1.7 Period）。

        period: 分钟数，取 {1,3,5,10,15,30,60,120}，对应手册 ``Period.minN``。
        begin_time/end_time: 可选，限定日内时段（时分，如 930 / 1500）。

        返回 multi-index (kline_time, code) DataFrame，
        列: open/high/low/close/volume/amount。kline_time 为完整 datetime
        （含时分，如 2026-01-05 09:35），跨交易日按天对齐。
        """

    # ---- 复权因子 ----
    @abstractmethod
    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        """单次复权因子（相邻两次除权除息之间的调整系数，不能直接乘到原始价格上
        得到跨区间可比的复权价）。index=date, columns=code, values=adj_factor。"""

    @abstractmethod
    def get_backward_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        """累积后复权因子。index=date, columns=code, values=backward_factor。
        用于把原始价格转成后复权价: adjusted_price = raw_price * backward_factor。"""

    # ---- 基础信息 ----
    @abstractmethod
    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        """每日最新证券信息。index=code, 列含 symbol/status/pre_close/
        high_limited/low_limited/price_tick。"""

    # ---- 历史涨跌停/停牌/ST ----
    @abstractmethod
    def get_history_stock_status(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        """按日历史证券状态（涨跌停价、停牌、ST、除权除息标记）。

        返回长表，列: date, code, pre_close, high_limited, low_limited,
        is_st, is_suspended, is_ex_dividend, is_ex_rights。"""

    # ---- 行业分类 ----
    @abstractmethod
    def get_industry_classification(self, level: int = 1) -> pd.DataFrame:
        """行业分类，point-in-time（个股在某个行业指数下的纳入/剔除区间）。

        level: 1/2/3 对应申万一级/二级/三级行业。
        返回长表，列: code, industry_code, industry_name, level, in_date, out_date。"""

    # ---- 股本结构（用于构建市值面板）----
    @abstractmethod
    def get_equity_structure(self, code_list: Iterable[str]) -> pd.DataFrame:
        """历史股本变动事件表（稀疏，一次变动一行，不是每日行情）。

        返回长表，列: code, change_date, tot_share, float_share（单位：万股）。"""

    # ---- 分红（用于股息率/股利支付率类因子）----
    @abstractmethod
    def get_dividend(self, code_list: Iterable[str]) -> pd.DataFrame:
        """上市公司分红实施数据（稀疏事件表）。

        返回长表，列: code, ann_date, record_date, ex_date, payout_date,
        report_period, cash_per_share_pre_tax（每股派息税前元）, bonus_rate,
        base_share（基准股本万股）。"""

    # ---- 十大股东（用于机构持仓/股东集中度类因子）----
    @abstractmethod
    def get_share_holder(self, code_list: Iterable[str]) -> pd.DataFrame:
        """十大股东明细（稀疏事件表，每期最多 10 行/股）。

        返回长表，列: code, ann_date, holder_end_date, holder_name,
        holder_pct（持股比例%）, holder_quantity（持股数股）, float_quantity。"""

    # ---- 股东户数（用于股东数时序类因子）----
    @abstractmethod
    def get_holder_num(self, code_list: Iterable[str]) -> pd.DataFrame:
        """股东户数（稀疏事件表，每披露期一行）。

        返回长表，列: code, ann_date, holder_end_date, holder_num（A股户数）。"""

    # ---- 财务报表（利润表/资产负债表/现金流量表）----
    @abstractmethod
    def get_balance_sheet(
        self, code_list: Iterable[str], begin_date: int | None = None, end_date: int | None = None
    ) -> pd.DataFrame:
        """资产负债表（稀疏事件表，按报告期一行）。

        point-in-time 关键字段为 ann_date（公告日）。返回长表，列:
        code, ann_date, report_period, statement_type, report_type, + 各财务字段。
        """

    @abstractmethod
    def get_cash_flow(
        self, code_list: Iterable[str], begin_date: int | None = None, end_date: int | None = None
    ) -> pd.DataFrame:
        """现金流量表。schema 同 get_balance_sheet。"""

    @abstractmethod
    def get_income(
        self, code_list: Iterable[str], begin_date: int | None = None, end_date: int | None = None
    ) -> pd.DataFrame:
        """利润表。schema 同 get_balance_sheet。"""


# ---------------------------------------------------------------------------
# AmazingData 实现
# ---------------------------------------------------------------------------
class AmazingDataSource(DataSource):
    """银河证券 AmazingData SDK 封装。

    SDK 首次使用需 `pip install AmazingData-*.whl`（见开发手册 3.3）。
    实例化时自动 login，之后复用同一连接。
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._local_path = cfg.get("sdk_local_path", "e://data//sdk_cache//")
        self._ad = None       # 模块句柄
        self._base = None     # BaseData 实例
        self._market = None   # MarketData 实例
        self._info = None     # InfoData 实例
        self._login(cfg)

    # ---- 登录 ----
    def _login(self, cfg: dict) -> None:
        import AmazingData as ad  # type: ignore

        required = ("username", "password", "host", "port")
        missing = [key for key in required if not cfg.get(key)]
        if missing:
            raise ValueError(
                "AmazingData credentials are missing: "
                f"{', '.join(missing)}. Set AMAZINGDATA_* environment variables "
                "or provide them in the repository .env file."
            )

        self._ad = ad
        try:
            # The SDK calls exit(0) itself when authentication fails.  Convert
            # that silent successful-looking process exit into an actionable
            # application error.
            ad.login(
                username=cfg["username"],
                password=cfg["password"],
                host=cfg["host"],
                port=int(cfg["port"]),
            )
        except SystemExit as exc:
            raise ConnectionError(
                "AmazingData login failed. Verify AMAZINGDATA_HOST/PORT, "
                "credentials, and network access."
            ) from exc
        self._base = ad.BaseData()
        calendar = self._base.get_calendar()
        self._market = ad.MarketData(calendar)
        self._info = ad.InfoData()

    # ---- 交易日历 ----
    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        cal = self._base.get_calendar()
        if end is None:
            return [d for d in cal if d >= begin]
        return [d for d in cal if begin <= d <= end]

    # ---- 代码表 ----
    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        return self._base.get_code_list(security_type=security_type)

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        raw = self._info.get_index_constituent([index_code], is_local=False)
        df = raw[index_code]
        return df[["CON_CODE", "INDATE", "OUTDATE", "INDEX_NAME"]].rename(
            columns={"CON_CODE": "con_code", "INDATE": "in_date", "OUTDATE": "out_date"}
        )

    # ---- 日K线 ----
    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        codes = list(code_list)
        kline_dict = self._market.query_kline(
            codes,
            begin_date=begin_date,
            end_date=end_date,
            period=self._ad.constant.Period.day.value,
        )
        frames = []
        for code, df in kline_dict.items():
            # SDK 返回的 df 索引是普通位置索引，真正的交易日期在 kline_time 列里
            # （不是 index），必须显式取出来重建索引，否则整数位置索引会被
            # 误当成日期使用，下游所有 point-in-time / 日期对齐逻辑都会失效。
            sub = df[["kline_time", "open", "high", "low", "close", "volume", "amount"]].copy()
            sub = sub.rename(columns={"kline_time": "date"})
            sub.insert(1, "code", code)
            frames.append(sub)
        if not frames:
            return pd.DataFrame(
                columns=["date", "code", "open", "high", "low", "close", "volume", "amount"]
            )
        out = pd.concat(frames, ignore_index=True)
        out = out.set_index(["date", "code"])
        return out.sort_index()

    # ---- 分钟K线 ----
    def get_minute_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
        period: int = 5,
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        codes = list(code_list)
        validate_minute_period(period)
        period_value = getattr(self._ad.constant.Period, f"min{period}").value
        kwargs: dict = dict(begin_date=begin_date, end_date=end_date, period=period_value)
        if begin_time is not None:
            kwargs["begin_time"] = begin_time
        if end_time is not None:
            kwargs["end_time"] = end_time
        kline_dict = self._market.query_kline(codes, **kwargs)
        frames = []
        for code, df in kline_dict.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            sub = df[["kline_time", "open", "high", "low", "close", "volume", "amount"]].copy()
            sub.insert(1, "code", code)
            frames.append(sub)
        if not frames:
            return pd.DataFrame(
                columns=["kline_time", "code", "open", "high", "low", "close", "volume", "amount"]
            )
        out = pd.concat(frames, ignore_index=True)
        # kline_time 含日内时分，是完整 datetime（不是 int YYYYMMDD），
        # 走自动解析即可；个别脏值置 NaT 后丢弃。
        out["kline_time"] = pd.to_datetime(out["kline_time"], errors="coerce")
        out = out.dropna(subset=["kline_time"])
        out = out.set_index(["kline_time", "code"])
        return out.sort_index()

    # ---- 复权因子 ----
    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        # SDK 内部以 pickle 格式缓存到 local_path，首次 is_local=False 从服务端拉取并落地，
        # 之后 is_local=True 时直接读本地 pickle；两套缓存（SDK pickle / 自建 parquet）共存互补。
        return self._base.get_adj_factor(
            list(code_list), local_path=self._local_path, is_local=False
        )

    def get_backward_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        return self._base.get_backward_factor(
            list(code_list), local_path=self._local_path, is_local=False
        )

    # ---- 基础信息 ----
    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        return self._base.get_code_info(security_type=security_type)

    # ---- 历史涨跌停/停牌/ST ----
    def get_history_stock_status(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        raw = self._info.get_history_stock_status(
            list(code_list), local_path=self._local_path, is_local=False,
            begin_date=begin_date, end_date=end_date,
        )
        # AmazingData 1.0 returns a {code: DataFrame} mapping here (despite
        # the manual describing a DataFrame).  Later SDK versions may return
        # the table directly, so accept both wire formats.
        if isinstance(raw, dict):
            raw = pd.concat(raw.values(), ignore_index=True) if raw else pd.DataFrame()
        if raw.empty:
            return pd.DataFrame(
                columns=["date", "code", "pre_close", "high_limited", "low_limited",
                         "is_st", "is_suspended", "is_ex_dividend", "is_ex_rights"]
            )
        out = raw.rename(columns={
            "MARKET_CODE": "code",
            "TRADE_DATE": "date",
            "PRECLOSE": "pre_close",
            "HIGH_LIMITED": "high_limited",
            "LOW_LIMITED": "low_limited",
            "IS_ST_SEC": "is_st",
            "IS_SUSP_SEC": "is_suspended",
            "IS_WD_SEC": "is_ex_dividend",
            "IS_XR_SEC": "is_ex_rights",
        })
        out["date"] = _to_datetime_safe(out["date"])
        for col in ("is_st", "is_suspended", "is_ex_dividend", "is_ex_rights"):
            if col in out.columns:
                out[col] = out[col].astype(str) == "1"
        keep = ["date", "code", "pre_close", "high_limited", "low_limited",
                "is_st", "is_suspended", "is_ex_dividend", "is_ex_rights"]
        return out[[c for c in keep if c in out.columns]]

    # ---- 行业分类 ----
    def get_industry_classification(self, level: int = 1) -> pd.DataFrame:
        base_info = self._info.get_industry_base_info(
            local_path=self._local_path, is_local=False
        )
        if base_info.empty:
            return pd.DataFrame(
                columns=["code", "industry_code", "industry_name", "level", "in_date", "out_date"]
            )
        level_info = base_info[base_info["LEVEL_TYPE"] == level]
        if level_info.empty:
            return pd.DataFrame(
                columns=["code", "industry_code", "industry_name", "level", "in_date", "out_date"]
            )
        level_codes = level_info["INDEX_CODE"].tolist()
        name_col = f"LEVEL{level}_NAME"
        name_map = level_info.set_index("INDEX_CODE")[name_col].to_dict()

        constituent = self._info.get_industry_constituent(level_codes, is_local=False)
        frames = []
        for industry_code, df in constituent.items():
            sub = df[["CON_CODE", "INDATE", "OUTDATE"]].copy()
            sub["industry_code"] = industry_code
            sub["industry_name"] = name_map.get(industry_code)
            frames.append(sub)
        if not frames:
            return pd.DataFrame(
                columns=["code", "industry_code", "industry_name", "level", "in_date", "out_date"]
            )
        out = pd.concat(frames, ignore_index=True).rename(
            columns={"CON_CODE": "code", "INDATE": "in_date", "OUTDATE": "out_date"}
        )
        out["level"] = level
        return out[["code", "industry_code", "industry_name", "level", "in_date", "out_date"]]

    # ---- 股本结构 ----
    def get_equity_structure(self, code_list: Iterable[str]) -> pd.DataFrame:
        raw = self._info.get_equity_structure(
            list(code_list), local_path=self._local_path, is_local=False
        )
        if raw.empty:
            return pd.DataFrame(columns=["code", "change_date", "tot_share", "float_share"])
        valid = raw[raw["IS_VALID"].astype(str) == "1"] if "IS_VALID" in raw.columns else raw
        out = valid.rename(columns={
            "MARKET_CODE": "code",
            "CHANGE_DATE": "change_date",
            "TOT_SHARE": "tot_share",
            "FLOAT_SHARE": "float_share",
        })
        out["change_date"] = _to_datetime_safe(out["change_date"])
        keep = ["code", "change_date", "tot_share", "float_share"]
        return out[[c for c in keep if c in out.columns]]

    # ---- 分红 ----
    def get_dividend(self, code_list: Iterable[str]) -> pd.DataFrame:
        raw = self._info.get_dividend(
            list(code_list), local_path=self._local_path, is_local=False
        )
        empty_cols = ["code", "ann_date", "record_date", "ex_date", "payout_date",
                      "report_period", "cash_per_share_pre_tax", "bonus_rate", "base_share"]
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return pd.DataFrame(columns=empty_cols)
        if isinstance(raw, dict):
            raw = pd.concat(raw.values(), ignore_index=True) if raw else pd.DataFrame()
        if raw.empty:
            return pd.DataFrame(columns=empty_cols)
        out = raw.rename(columns={
            "MARKET_CODE": "code",
            "DATE_DVD_ANN": "ann_date",
            "DATE_EQY_RECORD": "record_date",
            "DATE_EX": "ex_date",
            "DATE_DVD_PAYOUT": "payout_date",
            "REPORT_PERIOD": "report_period",
            "DVD_PER_SHARE_PRE_TAX_CASH": "cash_per_share_pre_tax",
            "DIV_BONUSRATE": "bonus_rate",
            "DIV_BASESHARE": "base_share",
        })
        for col in ("ann_date", "record_date", "ex_date", "payout_date"):
            if col in out.columns:
                out[col] = _to_datetime_safe(out[col])
        if "report_period" in out.columns:
            out["report_period"] = _to_datetime_safe(out["report_period"])
        for col in ("cash_per_share_pre_tax", "bonus_rate", "base_share"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out[[c for c in empty_cols if c in out.columns]]

    # ---- 十大股东 ----
    def get_share_holder(self, code_list: Iterable[str]) -> pd.DataFrame:
        raw = self._info.get_share_holder(
            list(code_list), local_path=self._local_path, is_local=False
        )
        empty_cols = ["code", "ann_date", "holder_end_date", "holder_name",
                      "holder_pct", "holder_quantity", "float_quantity"]
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return pd.DataFrame(columns=empty_cols)
        if isinstance(raw, dict):
            raw = pd.concat(raw.values(), ignore_index=True) if raw else pd.DataFrame()
        if raw.empty:
            return pd.DataFrame(columns=empty_cols)
        out = raw.rename(columns={
            "MARKET_CODE": "code",
            "ANN_DATE": "ann_date",
            "HOLDER_ENDDATE": "holder_end_date",
            "HOLDER_NAME": "holder_name",
            "HOLDER_PCT": "holder_pct",
            "HOLDER_QUANTITY": "holder_quantity",
            "FLOAT_QTY": "float_quantity",
        })
        for col in ("ann_date", "holder_end_date"):
            if col in out.columns:
                out[col] = _to_datetime_safe(out[col])
        for col in ("holder_pct", "holder_quantity", "float_quantity"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out[[c for c in empty_cols if c in out.columns]]

    # ---- 股东户数 ----
    def get_holder_num(self, code_list: Iterable[str]) -> pd.DataFrame:
        raw = self._info.get_holder_num(
            list(code_list), local_path=self._local_path, is_local=False
        )
        empty_cols = ["code", "ann_date", "holder_end_date", "holder_num"]
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return pd.DataFrame(columns=empty_cols)
        if isinstance(raw, dict):
            raw = pd.concat(raw.values(), ignore_index=True) if raw else pd.DataFrame()
        if raw.empty:
            return pd.DataFrame(columns=empty_cols)
        out = raw.rename(columns={
            "MARKET_CODE": "code",
            "ANN_DT": "ann_date",
            "HOLDER_ENDDATE": "holder_end_date",
            "HOLDER_NUM": "holder_num",
        })
        for col in ("ann_date", "holder_end_date"):
            if col in out.columns:
                out[col] = _to_datetime_safe(out[col])
        if "holder_num" in out.columns:
            out["holder_num"] = pd.to_numeric(out["holder_num"], errors="coerce")
        return out[[c for c in empty_cols if c in out.columns]]

    # ---- 财务报表 ----
    # 字符串/元数据列：归一化时排除，不作为数值字段保留。
    _FINANCIAL_META_COLS = {
        "MARKET_CODE", "SECURITY_NAME", "CURRENCY_CODE", "COMMENTS",
        "_code", "code", "ann_date", "actual_ann_date", "report_period",
        "statement_type", "report_type",
    }

    def _normalize_financial(self, raw) -> pd.DataFrame:
        """把 SDK 的 {code: DataFrame} 财务报表归一化为长表。

        - 统一字段名：MARKET_CODE→code, ANN_DATE→ann_date,
          ACTUAL_ANN_DATE→actual_ann_date, REPORTING_PERIOD→report_period,
          STATEMENT_TYPE→statement_type, REPORT_TYPE→report_type。
        - 公告日取 actual_ann_date，缺失回退 ann_date（PIT 发布日）。
        - 其余列强转 numeric（非数值变 NaN）。
        """
        empty_cols = ["code", "ann_date", "report_period", "statement_type", "report_type"]
        if isinstance(raw, dict):
            frames = []
            for code, df in raw.items():
                if df is None or (hasattr(df, "empty") and df.empty):
                    continue
                sub = df.copy()
                sub["_code"] = code
                frames.append(sub)
            if not frames:
                return pd.DataFrame(columns=empty_cols)
            concat = pd.concat(frames, ignore_index=True)
        elif isinstance(raw, pd.DataFrame) and not raw.empty:
            concat = raw.copy()
        else:
            return pd.DataFrame(columns=empty_cols)

        rename = {
            "MARKET_CODE": "code",
            "ANN_DATE": "ann_date",
            "ACTUAL_ANN_DATE": "actual_ann_date",
            "REPORTING_PERIOD": "report_period",
            "STATEMENT_TYPE": "statement_type",
            "REPORT_TYPE": "report_type",
        }
        out = concat.rename(columns=rename)
        if "code" not in out.columns:
            out["code"] = out.get("_code")

        for col in ("ann_date", "actual_ann_date", "report_period"):
            if col in out.columns:
                out[col] = _to_datetime_safe(out[col])
        # PIT 发布日：优先 actual_ann_date，回退 ann_date
        if "actual_ann_date" in out.columns:
            out["ann_date"] = out["actual_ann_date"].fillna(out.get("ann_date"))

        keep = [c for c in ("code", "ann_date", "report_period",
                            "statement_type", "report_type") if c in out.columns]
        numeric = [c for c in out.columns if c not in self._FINANCIAL_META_COLS]
        for c in numeric:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        # 同一 (code, ann_date) 多条记录（不同 statement_type）时保留最新报告期
        out = out.sort_values(["code", "ann_date", "report_period"])
        out = out.drop_duplicates(subset=["code", "ann_date"], keep="last")
        return out[keep + numeric].reset_index(drop=True)

    def get_balance_sheet(self, code_list: Iterable[str],
                          begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        raw = self._info.get_balance_sheet(
            list(code_list), local_path=self._local_path, is_local=False
        )
        return self._normalize_financial(raw)

    def get_cash_flow(self, code_list: Iterable[str],
                      begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        raw = self._info.get_cash_flow(
            list(code_list), local_path=self._local_path, is_local=False
        )
        return self._normalize_financial(raw)

    def get_income(self, code_list: Iterable[str],
                   begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        raw = self._info.get_income(
            list(code_list), local_path=self._local_path, is_local=False
        )
        return self._normalize_financial(raw)


# ---------------------------------------------------------------------------
# CSV 备用数据源
# ---------------------------------------------------------------------------
class CSVDataSource(DataSource):
    """本地 CSV 目录数据源，目录结构: root/{table}/{code}.csv
    用于离线开发 / 无 SDK 凭证时 / 切换到第三方数据。"""

    def __init__(self, cfg: dict):
        self._root = Path(cfg["root"]) if cfg.get("root") else Path("data/csv")
        self._root.mkdir(parents=True, exist_ok=True)

    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        # 简化：从任意 K 线文件的日期列推断
        return []

    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        d = self._root / "daily"
        if not d.exists():
            return []
        return [f.stem for f in d.glob("*.csv")]

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        p = self._root / f"index_constituent_{index_code}.csv"
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame(columns=["con_code", "in_date", "out_date"])

    def get_daily_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        frames = []
        for code in code_list:
            p = self._root / "daily" / f"{code}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, parse_dates=["date"])
            df["code"] = code
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).set_index(["date", "code"])
        return out.sort_index()

    def get_minute_kline(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
        period: int = 5,
        begin_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        # CSV 备用源暂无分钟数据，返回空表（列结构与真实实现一致）
        return pd.DataFrame(
            columns=["kline_time", "code", "open", "high", "low", "close", "volume", "amount"]
        )

    def get_adj_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def get_backward_factor(self, code_list: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame()

    def get_code_info(self, security_type: str = "EXTRA_STOCK_A") -> pd.DataFrame:
        return pd.DataFrame()

    def get_history_stock_status(
        self,
        code_list: Iterable[str],
        begin_date: int,
        end_date: int,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["date", "code", "pre_close", "high_limited", "low_limited",
                     "is_st", "is_suspended", "is_ex_dividend", "is_ex_rights"]
        )

    def get_industry_classification(self, level: int = 1) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["code", "industry_code", "industry_name", "level", "in_date", "out_date"]
        )

    def get_equity_structure(self, code_list: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "change_date", "tot_share", "float_share"])

    def get_dividend(self, code_list: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "ann_date", "record_date", "ex_date",
                                     "payout_date", "report_period",
                                     "cash_per_share_pre_tax", "bonus_rate", "base_share"])

    def get_share_holder(self, code_list: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "ann_date", "holder_end_date", "holder_name",
                                     "holder_pct", "holder_quantity", "float_quantity"])

    def get_holder_num(self, code_list: Iterable[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "ann_date", "holder_end_date", "holder_num"])

    def get_balance_sheet(self, code_list: Iterable[str],
                          begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "ann_date", "report_period", "statement_type", "report_type"])

    def get_cash_flow(self, code_list: Iterable[str],
                      begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "ann_date", "report_period", "statement_type", "report_type"])

    def get_income(self, code_list: Iterable[str],
                   begin_date: int | None = None, end_date: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "ann_date", "report_period", "statement_type", "report_type"])


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_datasource(cfg: dict | None = None) -> DataSource:
    """根据 config 创建数据源实例。"""
    from config import Config
    if cfg is None:
        cfg = Config.datasource()

    ds_type = cfg["type"]
    if ds_type == "amazing_data":
        return AmazingDataSource(cfg["amazing_data"])
    elif ds_type == "csv":
        return CSVDataSource(cfg["csv"])
    else:
        raise ValueError(f"未知数据源类型: {ds_type}")
