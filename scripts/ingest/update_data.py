"""
每日数据更新脚本
================

用法:
    python -m scripts.ingest.update_data              # 更新默认池（config.universe.default）日K线
    python -m scripts.ingest.update_data --pool zz1000  # 更新中证1000（按池分文件 daily_zz1000.parquet）
    python -m scripts.ingest.update_data --pool all_a   # 更新全A池（daily_all_a.parquet）
    python -m scripts.ingest.update_data --index 000905.SH  # 指定指数（pool 由映射推导）
    python -m scripts.ingest.update_data --begin 20230101   # 指定起始日
    python -m scripts.ingest.update_data --minute 5         # 拉取 5 分钟K线
    python -m scripts.ingest.update_data --minute 1,5,15    # 拉取多档分钟K线
    python -m scripts.ingest.update_data --no-minute        # 跳过分钟K线（即使配置了）

说明:
    - 增量更新：只拉本地缺失的日期段。
    - 首次运行会从 config.fetch.begin_date 开始全量拉取。
    - 缓存按池分文件：daily_{pool}.parquet / min{period}_{pool}.parquet，
      多池并存互不干扰（2026-08-26 池隔离扩展）。
    - 分钟K线默认按 config.minute.periods 拉取（默认 [5]），可用 --minute/--no-minute 覆盖。
    - 无 SDK 凭证时自动回退到 CSV 数据源（离线开发模式）。
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

from config import Config  # noqa: E402
from data.cache import DataCache  # noqa: E402
from data.datasource import create_datasource  # noqa: E402
from data.universe import Universe  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("update_data")

def _parse_minute_arg(raw: str | None) -> list[int] | None:
    """解析 --minute "1,5,15" → [1, 5, 15]；None 表示未指定（走配置）。

    档位合法性统一走 data.datasource.validate_minute_period（单一来源）。
    """
    if raw is None:
        return None
    from data.datasource import validate_minute_period
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [validate_minute_period(int(p)) for p in parts]


# ---- 状态表（涨跌停/停牌/ST/除权标记）拉取：子进程隔离 + 硬超时 ----
# 2026-09-16 事故：SDK 该接口当晚 hang 住，17:58 启动的那轮在此步骤挂了 40+ 分钟，
# 进程最终被杀、当日排名未能产出（18:43 改走 --skip-update 才在 7 分钟内出榜）。
# 根因是 **hang 而非 raise**：cache._batched_status_fetch 的 `for attempt in (1,2,3)`
# 只在抛异常时生效，而 datasource 层直调 SDK、全链路无 timeout，挂住后没有可中断点。
# 对策：整步交给独立子进程（scripts.ingest.fetch_status_batched，每批即落盘），
# 主进程用 subprocess timeout 兜底——超时 kill 得掉，且已拉批次不丢（断点续拉）。


def _status_incremental_begin(begin: int, target_date: int) -> int:
    """状态表增量起点 = 本地已覆盖的最后日期 + 1 天；无缓存时回退 begin。

    只读 parquet 索引（``columns=[]``，不读数值列）——85MB 表约 0.1s。
    返回值 > target_date 表示本地已覆盖到目标日，调用方应跳过拉取。
    """
    p = Path(str(Config.cache()["root"])) / "history_stock_status.parquet"
    if not p.exists():
        return begin
    try:
        idx = pd.read_parquet(p, columns=[]).index
        if len(idx) == 0:
            return begin
        last = int(pd.Timestamp(idx.get_level_values("date").max()).strftime("%Y%m%d"))
    except Exception:  # noqa: BLE001 - 索引不可读时保守走全量补拉
        return begin
    return int((pd.Timestamp(str(last)) + pd.Timedelta(days=1)).strftime("%Y%m%d"))


def fetch_status_table(begin: int, target_date: int, cfg) -> None:
    """拉取历史涨跌停/停牌/ST 状态表（子进程 + 硬超时，失败不阻断主流程）。

    非关键路径：失败仅使下游降级——可交易掩码漏拒（方向保守，不错杀）、
    状态特征当日缺失、信号日可交易改用日线行情推断涨跌停。
    """
    st_cfg = cfg.get("fetch", {}).get("status_table", {}) or {}
    if not st_cfg.get("enabled", True):
        log.info("状态表拉取已禁用（fetch.status_table.enabled=false），跳过")
        return
    st_begin = _status_incremental_begin(begin, target_date)
    if st_begin > target_date:
        log.info("状态表已覆盖至 %s，跳过拉取", target_date)
        return
    timeout_s = int(st_cfg.get("timeout_s", 300))
    batch = int(st_cfg.get("batch", 200))
    log.info("拉取历史涨跌停/停牌状态: %s -> %s（子进程，预算 %ds，%d 只/批）",
             st_begin, target_date, timeout_s, batch)
    cmd = [sys.executable, "-m", "scripts.ingest.fetch_status_batched",
           "--begin", str(st_begin), "--end", str(target_date), "--batch", str(batch)]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.warning("状态表拉取超过 %ds 预算，子进程已终止（已落盘批次保留）。"
                    "下游降级：掩码漏拒、信号日可交易用日线推断涨跌停。"
                    "重跑本脚本或 `python -m scripts.ingest.fetch_status_batched` 可断点续拉",
                    timeout_s)
        return
    except Exception as exc:  # noqa: BLE001 - 非关键步骤，任何异常都不阻断主流程
        log.warning("状态表拉取异常（保留旧缓存，不影响核心行情数据）: %s", exc)
        return
    if proc.returncode != 0:
        log.warning("状态表拉取子进程 exit=%d（保留旧缓存，下游降级）", proc.returncode)
    else:
        log.info("状态表拉取完成")

# 股票池 -> 指数代码 / 池名（2026-08-26 池隔离扩展）
_POOL_INDEX = {"hs300": "000300.SH", "zz500": "000905.SH", "zz1000": "000852.SH"}
_VALID_POOLS = (*_POOL_INDEX.keys(), "all_a")

def check_pool_consistency(cache: DataCache, index_code: str, pool: str) -> None:
    """拉取前校验：meta 中 daily/min5 的 pool 口径与本次拉取是否一致。

    2026-08-26 后缓存按池分文件（daily_{pool}.parquet）；本校验确认
    目标池文件与本次拉取池一致，防止历史教训重演（daily 曾混入 2179 只
    ZZ1000 + 指数 000905.SH）。不匹配时给出显式告警。
    """
    import json
    meta_path = cache.root / "_meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return

    for base in ("daily", "min5"):
        table = f"{base}_{pool}"
        info = meta.get(table, {})
        cur_pool = info.get("pool", "")
        if cur_pool and cur_pool != pool:
            log.warning(
                "[池一致性] %s 已有数据且 meta.pool=%s，本次拉取 pool=%s——"
                "文件名不一致，可能产生重复/混合数据",
                table, cur_pool, pool,
            )

def main():
    parser = argparse.ArgumentParser(description="YuriQuant 数据更新")
    parser.add_argument("--index", default=None, help="指数代码，默认取 config.universe.index_code")
    parser.add_argument("--begin", type=int, default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", type=int, default=None, help="结束日期 YYYYMMDD，默认至今")
    parser.add_argument("--minute", default=None, help="拉取分钟K线档位，逗号分隔，如 5 或 1,5,15")
    parser.add_argument("--no-minute", action="store_true", help="跳过分钟K线拉取")
    parser.add_argument("--pool", default=None,
                        help=f"股票池: {' | '.join(_VALID_POOLS)}"
                             "（默认取 config.universe.default）")
    parser.add_argument("--allow-intraday", action="store_true",
                        help="允许把拉取终点设为今天（盘中数据不完整，默认回退上一交易日）")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = Config.get()
    fetch_cfg = cfg["fetch"]
    begin = args.begin or fetch_cfg["begin_date"]
    end = args.end or fetch_cfg.get("end_date")

    # 2. 创建数据源 + 缓存
    ds = create_datasource()
    cache = DataCache(ds)

    # 3. 确定股票池（--pool 优先，其次 config；指数代码按池映射，--index 覆盖）
    pool = args.pool or cfg["universe"].get("default", "hs300")
    if pool not in _VALID_POOLS:
        raise SystemExit(f"未知股票池: {pool}（可选 {' | '.join(_VALID_POOLS)}）")
    index_code = args.index or _POOL_INDEX.get(pool) or cfg["universe"]["index_code"]
    uni = Universe(cache)
    check_pool_consistency(cache, index_code, pool)
    log.info("股票池: %s (index=%s)", pool, index_code)

    log.info("获取 %s 成分股 ...", index_code)
    # PIT 口径（2026-08-13 统一）：拉取 begin~end 区间【历史在册成分并集】，
    # 而非 end 时点成分——否则历史期被调出/退市的股票永远缺数据（幸存者偏差）。
    cal = cache.get_calendar(begin, end)
    if not cal:
        log.warning("交易日历为空，请检查数据源配置。")
        return
    # 盘中守卫：target=今天且未到收盘确认时刻 → 回退上一交易日（防半拉日永久入缓存）
    from scripts.common.cli_common import complete_day_target
    target_date, rolled_back = complete_day_target(
        cal, end if end else cal[-1], allow_intraday=args.allow_intraday)
    if rolled_back:
        log.info("盘中守卫: 拉取终点回退到 %s（今天数据未完整，--allow-intraday 可跳过守卫）",
                 target_date)
    if pool == "all_a":
        # 全 A：优先用本地缓存的全 A 日线清单（增量维护）；首次拉取时
        # daily_all_a.parquet 尚不存在 → 回退数据源安全主档取全 A 代码
        codes = uni.get_all_a(target_date)
        if not codes:
            log.info("本地无 daily_all_a 缓存，从数据源 get_code_list 获取全A清单 ...")
            codes = ds.get_code_list("EXTRA_STOCK_A")
        log.info("全A池代码: %d 只", len(codes))
    else:
        from data.cache_helpers import pit_universe_codes
        codes = pit_universe_codes(uni, index_code, begin, target_date)
        log.info("历史成分并集池: %d 只（%s~%s 期间在册，含调出/退市）",
                 len(codes), begin, target_date)

    # 4. 增量拉取日K线（按池落盘 daily_{pool}.parquet）
    log.info("增量拉取日K线: %s -> %s", begin, target_date)
    kline = cache.get_daily_kline(codes, begin, target_date, pool=pool)
    if len(kline) == 0:
        raise SystemExit("日K线拉取为空（代码清单或数据源可能不可用），中止后续步骤")
    log.info("日K线行数: %d, 代码数: %d", len(kline),
             kline.index.get_level_values("code").nunique())
    if pool == "all_a":
        log.info("提示: 因子面板不会自动延伸——需要新日期进实验时重算 "
                 "scripts/builders/build_alla_alpha_panels.py --workers 6（全量 ~1h）"
                 "及基本面/股东/质押/构造型构建脚本")

    # 4.5 增量拉取分钟K线（日内研究，按池落盘 min{period}_{pool}.parquet）
    minute_periods = []
    if args.minute:
        minute_periods = _parse_minute_arg(args.minute)
    elif not args.no_minute:
        minute_periods = list(cfg.get("minute", {}).get("periods", [5]))
    minute_rows: dict[str, int] = {}
    for period in minute_periods:
        log.info("增量拉取 %d 分钟K线: %s -> %s", period, begin, target_date)
        mk = cache.get_minute_kline(codes, begin, target_date, period=period, pool=pool)
        n_codes = mk.index.get_level_values("code").nunique() if len(mk) else 0
        minute_rows[f"min{period}_rows"] = len(mk)
        log.info("%d 分钟K线行数: %d, 代码数: %d", period, len(mk), n_codes)

    # 5. 复权因子（单次复权因子 + 累积后复权因子）
    # upto=target_date：本地宽表已覆盖目标日则短路、不访问数据源。SDK 的复权
    # 因子接口没有日期参数（每次回源都是全量拉取并覆写 h5，实测约 6 分钟），
    # 同日重跑（补跑/重试）直接复用本地，避免重复付出这份代价。
    log.info("拉取复权因子(目标 %s，本地已覆盖则跳过) ...", target_date)
    adj = cache.get_adj_factor(codes, upto=target_date)
    log.info("单次复权因子行数: %d, 列数: %d", len(adj), adj.shape[1])
    backward = cache.get_backward_factor(codes, upto=target_date)
    log.info("后复权因子行数: %d, 列数: %d", len(backward), backward.shape[1])

    # 6. 历史涨跌停/停牌/ST 状态（非关键：失败仅降级过滤能力，不阻断后续）
    # 走子进程 + 硬超时：SDK 该接口会 hang 而非抛异常，进程内重试对 hang 无效
    # （2026-09-16 事故，详见 fetch_status_table 文档）。
    fetch_status_table(begin, target_date, cfg)

    # 7. 行业分类（因子行业中性化用）
    industry_level = int(cfg.get("preprocessing", {}).get("industry_level", 1))
    log.info("拉取行业分类 (level=%d) ...", industry_level)
    industry = cache.get_industry_classification(industry_level)
    log.info("行业分类行数: %d", len(industry))

    # 8. 股本结构（市值中性化用）
    log.info("拉取股本结构 ...")
    equity = cache.get_equity_structure(codes)
    log.info("股本结构行数: %d", len(equity))

    # 9. 财务三表（基本面因子 / 因子挖掘用）
    log.info("拉取利润表 ...")
    income = cache.get_income(codes)
    log.info("利润表行数: %d", len(income))

    log.info("拉取资产负债表 ...")
    balance = cache.get_balance_sheet(codes)
    log.info("资产负债表行数: %d", len(balance))

    log.info("拉取现金流量表 ...")
    cashflow = cache.get_cash_flow(codes)
    log.info("现金流量表行数: %d", len(cashflow))

    # 9b. 分红 / 十大股东 / 股东户数（股息率/股东类因子用，2026-08-04）
    log.info("拉取分红数据 ...")
    dividend = cache.get_dividend(codes)
    log.info("分红行数: %d", len(dividend))

    log.info("拉取十大股东 ...")
    share_holder = cache.get_share_holder(codes)
    log.info("十大股东行数: %d", len(share_holder))

    log.info("拉取股东户数 ...")
    holder_num = cache.get_holder_num(codes)
    log.info("股东户数行数: %d", len(holder_num))

    # 10. 实验记录（数据版本指纹 + 拉数摘要）
    try:
        from research.experiments import record_experiment
        fingerprint = cache.get_fingerprint()
        metrics = {"kline_rows": len(kline), "adj_cols": adj.shape[1],
                   "status_rows": len(status), "income_rows": len(income)}
        metrics.update(minute_rows)
        record_experiment(
            kind="data_update",
            command=" ".join(sys.argv),
            params={"index": index_code, "pool": pool, "begin": begin, "end": target_date,
                    "n_codes": len(codes), "minute_periods": minute_periods},
            data_fingerprint=fingerprint,
            result_path=str(cache.root),
            metrics=metrics,
            note="数据更新完成",
        )
        log.info("数据指纹: %s", fingerprint)
    except Exception as e:
        log.warning("实验记录写入失败（不影响数据更新）: %s", e)

    log.info("数据更新完成。缓存目录: %s", cache.root)

if __name__ == "__main__":
    main()