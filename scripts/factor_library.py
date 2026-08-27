"""
因子库 CLI
=========

管理持久化的因子库：列出、统一对比、查看某时间段回测、导出 Excel 对比报告、删除、迭代特征。

子命令:
    list                      列出所有因子（--kind raw|composite --family --maturity）
    compare                   统一指标排名（默认按 IR；--metric ic_mean/sharpe/annual_return/max_drawdown/calmar/avg_turnover --config ls_M --top N）
    view NAME                 查看某因子全期或指定时间段回测（--start --end --config --regime 分市场状态）
    set-tag NAME              补打/更新标签（--family --frequency --maturity --note）
    monitor                   生命周期监控：全期 vs 近期 IC 漂移，warning 因子排前
    report                    导出 Excel 对比报告（--names a,b,c 或 --all --config --out）
    features                  列出可作为下一轮挖掘特征（迭代）的因子
    delete NAME               删除一个因子（带确认）

示例:
    python scripts/factor_library.py datasets
    python scripts/factor_library.py list --dataset hs300_2025 --family 反转
    python scripts/factor_library.py set-tag "ts_delta(amount,20)" --family 反转 --frequency 日频 --dataset hs300_2025
    python scripts/factor_library.py monitor --dataset hs300_2025 --window 60
    python scripts/factor_library.py compare --dataset hs300_2025 --metric sharpe --top 15
    python scripts/factor_library.py view "ts_delta(amount,20)" --dataset hs300_2025 --start 20250101 --end 20250601 --regime
    python scripts/factor_library.py report --dataset hs300_2025 --all --config ls_M --out reports/lib_report.xlsx
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("factor_library_cli")


def _print_df(df: pd.DataFrame, cols=None, max_rows=40):
    with pd.option_context("display.max_rows", max_rows, "display.width", 240,
                           "display.float_format", lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)):
        print(df[cols] if cols else df)


def cmd_list(args):
    lib = _lib()
    df = lib.list_all(kind=args.kind, family=args.family, maturity=args.maturity)
    if df.empty:
        print("因子库为空。先运行 `python scripts/mine_factors.py --save-library` 或合成 --save-library。")
        return
    cols = ["name", "kind", "family", "maturity", "ic_mean", "t_stat", "significant",
            "best_sharpe", "best_config", "created_at"]
    _print_df(df, [c for c in cols if c in df.columns])


def cmd_compare(args):
    lib = _lib()
    df = lib.compare(metric=args.metric, config=args.config, ascending=args.ascending, kind=args.kind, topn=args.top)
    if df.empty:
        print("因子库为空。")
        return
    show_cols = ["name", "kind", "ic_mean", "t_stat", "significant",
                 "sharpe_ls_M", "annual_return_ls_M", "max_drawdown_ls_M", "calmar_ls_M",
                 "sharpe_lo_M", "sharpe_ls_W", "best_config"]
    _print_df(df, [c for c in show_cols if c in df.columns])
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        log.info("对比结果已保存: %s", out)


def cmd_view(args):
    lib = _lib()
    if not lib.has(args.name):
        print(f"因子不存在: {args.name}")
        print("可用因子:", [n for n in lib.list_all()['name'].tolist()][:10], "...")
        return
    info = lib.evaluate_period(args.name, start=args.start, end=args.end, config=args.config)
    m = info["metrics"]
    print(f"\n===== 因子 {args.name} @ {info['config']} =====")
    print(f"时间段: {info['start']} ~ {info['end']}  (交易日 {info['n_days']})")
    print(f"IC均值={info['ic_mean']:.4f}  IC_IR={info['ic_ir']:.3f}  IC胜率={info['ic_win_rate']:.2%}")
    print(f"年化={m['annual_return']:.2%}  夏普={m['sharpe']:.3f}  索提诺={m['sortino']:.3f}")
    print(f"最大回撤={m['max_drawdown']:.2%}  卡玛={m['calmar']:.3f}  胜率={m['win_rate']:.2%}")
    if args.regime:
        try:
            mr = None
            if getattr(args, "market_returns", None):
                mc = pd.read_csv(args.market_returns)
                mr = pd.Series(mc.iloc[:, 1].values, index=pd.to_datetime(mc.iloc[:, 0]))
            regs = lib.regime_analysis(args.name, market_returns=mr)
            print("\n----- 分市场状态检验（八维检验之一）-----")
            print(f"{'市场状态':<10}{'IC均值':>10}{'IR':>8}{'IC胜率':>10}{'天数':>6}{'市场年化':>12}")
            for label, v in regs.items():
                print(f"{label:<10}{v['ic_mean']:>10.4f}{v['ir']:>8.2f}{v['win_rate']:>10.2%}"
                      f"{v['n_days']:>6}{v['market_ann']:>12.1%}")
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"分市场检验跳过: {e}")
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": info["daily_returns"].index,
                      "daily_return": info["daily_returns"].values,
                      "equity": info["equity_curve"].reindex(info["daily_returns"].index).values,
                      "ic": info["ic_series"].reindex(info["daily_returns"].index).values}).to_csv(out, index=False)
        log.info("时间段回测已保存: %s", out)


def cmd_report(args):
    from research.xlsx_report import generate_excel_report

    lib = _lib()
    reg = lib.list_all()
    if reg.empty:
        print("因子库为空。")
        return
    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    else:
        names = reg["name"].tolist()
    results = {}
    summaries = {}
    for name in names:
        try:
            res = lib.reconstruct_backtest(name, config=args.config)
            ev = lib._load_eval(name)
            ic = ev["ic"].dropna() if ev is not None else pd.Series(dtype=float)
            summaries[name] = {
                "ic_series": ic,
                "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
                "ir": 0.0,
                "ic_win_rate": float((ic > 0).mean()) if len(ic) else float("nan"),
                "ic_decay": {},
                "layer_returns": {},
            }
            results[name] = res
        except Exception as e:
            log.warning("跳过 %s: %s", name, e)
    if not results:
        print("没有可生成报告的因子。")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_excel_report(results, summaries, output_path=out)
    log.info("Excel 对比报告已生成: %s", out)
    if not args.no_html:
        from research.html_report import generate_html_report
        html_path = Path(args.html_out)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        generate_html_report(
            results, summaries, output_path=html_path,
            title=f"因子库报告 — {lib.dataset or 'legacy'}",
            meta=f"config={args.config} · 共 {len(results)} 个因子 · 点击表头排序 / 标签页切换因子",
        )
        log.info("HTML 交互报告已生成: %s", html_path)


def cmd_features(args):
    lib = _lib()
    feats = lib.load_library_features(kind=args.kind)
    if not feats:
        print("库内暂无可用特征。先运行 mine/synthesize --save-library。")
        return
    print(f"可作为下一轮挖掘特征的因子（共 {len(feats)} 个）:")
    for name in feats:
        print(f"  - {name}")


def cmd_set_tag(args):
    lib = _lib()
    if not lib.has(args.name):
        print(f"因子不存在: {args.name}")
        return
    ok = lib.set_tag(args.name, family=args.family, frequency=args.frequency,
                     maturity=args.maturity, note=args.note)
    print(f"已更新标签: {args.name}" if ok else "更新失败")
    r = lib.list_all()
    hit = r[r["name"] == args.name].iloc[0]
    print(f"  family={hit.get('family', '')}  frequency={hit.get('frequency', '')}"
          f"  maturity={hit.get('maturity', '')}  note={hit.get('note', '')}")


def cmd_monitor(args):
    lib = _lib()
    df = lib.monitor(window=args.window)
    if df.empty:
        print("库内暂无因子可监控（或 IC 样本不足 20 日）。")
        return
    cols = ["name", "family", "maturity", "ic_mean_full", "ic_mean_recent",
            "ic_drift", "ic_ir_recent", "ic_t_nw_recent", "n_days", "status"]
    _print_df(df, cols)
    n_warn = int((df["status"] == "warning").sum())
    print(f"\nwarning 因子 {n_warn}/{len(df)} —— 近期 IC 均值已跌破全期一半，建议评估是否衰减/降权（退役=set-tag --maturity retired，保留不删）")


def cmd_delete(args):
    lib = _lib()
    if not lib.has(args.name):
        print(f"因子不存在: {args.name}")
        return
    if not args.force:
        print(f"确认删除 {args.name} ? 加 --force 强制执行。")
        return
    lib.delete(args.name)
    print(f"已删除: {args.name}")


def _lib() -> "FactorLibrary":
    from research.factor_library import FactorLibrary
    return FactorLibrary(
        root=args.root if getattr(args, "root", None) else None,
        dataset=getattr(args, "dataset", None),
    )


def cmd_datasets(args):
    from research.factor_library import FactorLibrary
    ds = FactorLibrary.list_datasets(root=args.root)
    if not ds:
        print("尚未创建任何数据集库。运行 mine/synthesize --library-dataset <name> 创建。")
        return
    print("可用数据集（按数据集分库根）:")
    for name in ds:
        lib = FactorLibrary(dataset=name, root=args.root)
        n = len(lib.list_all())
        print(f"  {name:24s} 因子数={n}")


def main():
    global args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=None, help="因子库根目录（默认读 settings）")
    common.add_argument("--dataset", default=None,
                        help="数据集名（按数据集分库根；如 hs300_2025 / mock）。不填=legacy 默认库")

    parser = argparse.ArgumentParser(description="YuriQuant 因子库管理")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("datasets", parents=[common], help="列出所有数据集（库根）")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("list", parents=[common], help="列出因子")
    p.add_argument("--kind", choices=["raw", "composite"], default=None)
    p.add_argument("--family", default=None, help="按因子家族过滤（如 反转/动量/波动率）")
    p.add_argument("--maturity", default=None, help="按成熟度过滤（experimental/oos_verified/active/retired）")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("compare", parents=[common], help="统一指标排名")
    p.add_argument("--metric", default="ir", help="排序指标（默认 ir=信息比率，业界统一主轴；也支持 ic_mean/sharpe/annual_return/max_drawdown/calmar/avg_turnover）")
    p.add_argument("--config", default=None, help="指定配置列（如 ls_M）；默认用 best_<metric>")
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--kind", choices=["raw", "composite"], default=None)
    p.add_argument("--top", type=int, default=None)
    p.add_argument("--csv", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("view", parents=[common], help="查看某因子时间段回测")
    p.add_argument("name")
    p.add_argument("--start", default=None, help="YYYYMMDD 或 YYYY-MM-DD")
    p.add_argument("--end", default=None)
    p.add_argument("--config", default="ls_M")
    p.add_argument("--regime", action="store_true", help="附加分市场状态检验（牛/熊/震荡三段 IC）")
    p.add_argument("--market-returns", default=None, help="可选：市场日收益 CSV（date,ret）；缺省从日线缓存读等权市场")
    p.add_argument("--csv", default=None)
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("set-tag", parents=[common], help="补打/更新因子标签（家族/频率/成熟度/备注）")
    p.add_argument("name")
    p.add_argument("--family", default=None, help="因子家族：动量/反转/波动率/价值/质量/成长/情绪/流动性/拥挤度/技术/非线性组合/其他")
    p.add_argument("--frequency", default=None, help="信号频率：日内/日频/周频/月频/季频")
    p.add_argument("--maturity", default=None, help="成熟度：experimental/oos_verified/active/retired")
    p.add_argument("--note", default=None, help="备注（设计动机/差异化贡献）")
    p.set_defaults(func=cmd_set_tag)

    p = sub.add_parser("monitor", parents=[common], help="生命周期监控：全期 vs 近期 IC 漂移")
    p.add_argument("--window", type=int, default=60, help="近期窗口交易日数（默认 60）")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("report", parents=[common], help="导出 Excel + HTML 对比报告")
    p.add_argument("--names", default=None, help="逗号分隔；不填则全部")
    p.add_argument("--config", default="ls_M")
    p.add_argument("--out", default="reports/factor_library_report.xlsx")
    p.add_argument("--html-out", default="reports/factor_library_report.html",
                   help="交互式 HTML 报告输出路径（--no-html 跳过）")
    p.add_argument("--no-html", action="store_true", help="只导出 Excel，不生成 HTML")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("features", parents=[common], help="列出可迭代特征")
    p.add_argument("--kind", choices=["raw", "composite"], default=None)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("delete", parents=[common], help="删除因子")
    p.add_argument("name")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
