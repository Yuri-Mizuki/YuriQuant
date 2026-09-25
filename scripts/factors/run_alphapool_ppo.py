"""
华泰 AI97《大模型+强化学习因子挖掘》—— AlphaPool + MaskablePPO 训练入口
========================================================================

链路（对齐研报「方法」节）：

1. 构建 6 字段量价面板（open/high/low/close/volume/vwap）与未来收益面板；
2. **大模型构造基础池**（``--llm``）—— 让 ``AlphaPool.best_obj`` 从 0 抬起来，
   这样研报奖励里「位于失败缓存 / 效果无法入池」两档才有非零信号；
3. MaskablePPO 训练（LSTM / Transformer 共享网络，见 ``factor.rl.alphapool_nets``）；
4. 训练中每 ``--llm-every`` 步做一次**去弱留强**（剔除 ``--drop-rl-n`` 个 RL 因子
   再注入新因子）；
5. 训练后在**测试段**评估池内因子与等权/池权重复合因子的 OOS IC。

用法::

    P=D:/python/Python312/python.exe
    # 离线小预算冒烟（合成面板，不碰数据）
    $P -u scripts/factors/run_alphapool_ppo.py --panel mock --llm template \\
        --llm-init 12 --total-timesteps 1024 --n-steps 256 --batch-size 64

    # 真实面板（需 AmazingData；离线缓存模式加 --offline）
    $P -u scripts/factors/run_alphapool_ppo.py --panel real --offline \\
        --llm template --llm-init 20 --policy lstm --total-timesteps 10000

    # 真实大模型（DeepSeek，2026-09-16 已联网验证）
    export DEEPSEEK_API_KEY=sk-xxx
    $P -u scripts/factors/run_alphapool_ppo.py --panel mock --llm openai \\
        --llm-model deepseek-flash --llm-max-tokens 16000 \\
        --llm-init 8 --total-timesteps 1024

**口径披露（必读）**：

- ``vwap``：研报的 6 字段含 VWAP，项目日线面板没有该字段，用 ``amount / volume``
  构造（日频 VWAP 的标准定义）。这是**替代口径**，不是研报原始字段；脚本会打印
  ``vwap/close`` 中位数做量纲自检（正常应在 1 附近）。
- ``--horizon``：研报超参表**未给出**因子评估所用的收益期限，默认 10 个交易日，
  引用结果时必须同时注明该值。
- ``--llm-every`` / ``--drop-rl-n`` 取自研报（3000 / 5）；``--llm-init`` 与
  ``--llm-new`` 在研报中**没有对应超参**（研报只说"构造基础池"与"定期注入新因子"），
  属自拟参数，不得当成研报口径引用。
- 大模型路径默认 ``template``（离线确定性模板），它**不是真实大模型**：
  ``--llm openai`` 才是真实调用，需 ``DEEPSEEK_API_KEY``。
- ⚠️ **推理模型的 ``max_tokens`` 必须覆盖思维链**：``deepseek-flash`` / ``deepseek-v4-pro``
  的 ``content`` 与 ``reasoning_content`` **共享** ``max_tokens``。实测给 32 / 2000 / 3000
  时思维链把预算吃光 → ``finish_reason="length"`` 且 ``content`` 为**空字符串**
  （HTTP 200、无异常，极易误判成 key 或模型名问题）。默认 16000 实测可用
  （reasoning ~6.3k + 正文 ~0.2k）。
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from factor.rl.alphapool_env import (
    FIELDS,
    MAX_EXPR_LENGTH,
    WINDOWS,
    AlphaPool,
    portfolio_ic,
)
from factor.rl.alphapool_gym import AlphaPoolGymEnv
from factor.rl.alphapool_nets import (
    REPORT_D_MODEL,
    REPORT_DROPOUT,
    REPORT_N_LAYERS,
    make_extractor_kwargs,
)
from factor.rl.llm_pool import (
    PoolUpdateReport,
    make_proposer,
    refresh_pool,
    seed_pool,
)

# 放在导入之后：夹在导入中间会让所有 factor.* 导入触发 E402，
# 而这条 filter 只影响后续运行时计算，与导入顺序无关。
warnings.filterwarnings("ignore", message="invalid value encountered in reduce")


# ---------------------------------------------------------------------------
# 面板
# ---------------------------------------------------------------------------
def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """未来 ``horizon`` 日收益（t 日对齐）。``horizon=1`` 即因子库 canonical 的
    ``pct_change().shift(-1)``，故 h>1 用 ``pct_change(h).shift(-h)`` 是同一约定的推广。

    只在**传入的切片内**计算 —— 切片尾部 ``horizon`` 行必为 NaN，从而天然阻止
    收益跨过 train/test 边界（否则训练段末尾会吃到测试段的收益）。
    """
    return close.pct_change(horizon).shift(-horizon)


def attach_vwap(panel: dict[str, pd.DataFrame], *, backward=None,
                verbose: bool = True) -> None:
    """给面板补 ``vwap``（= 成交额 / 成交量）并做量纲自检。就地修改 ``panel``。

    **复权口径（2026-09-16 修复，真实数据暴露）**：``data.cache_helpers.build_panel``
    只对 OHLC 乘了后复权因子，``amount`` / ``volume`` 保持原始口径。因此
    ``amount/volume`` 得到的是**原始价** VWAP，必须同样乘 ``backward`` 才能与
    后复权 ``close`` 可比 —— 否则 ``vwap/close`` 会等于 ``1/backward``，
    在真实数据上中位数仅 0.285（跨股票范围 0.004~1.0），把一条纯量纲错误
    伪装成"因子信号"喂进 RL 动作空间。

    Args:
        panel: 含 ``amount`` / ``volume`` 的面板；若已含 ``vwap`` 则直接返回。
        backward: 后复权因子面板（index×code，与 close 对齐）。为 None 时
            **不做复权**（保持旧行为，仅适用于 mock / 未复权面板）。
        verbose: 打印量纲自检结果。

    自检对照（修复后必须成立）：
    - ``backward`` 给出时：``vwap / close`` 中位数应≈1（两者同为后复权口径）；
    - ``backward`` 为 None 时：打印告警，提示该面板若为后复权则 vwap 不可用。
    """
    if "vwap" in panel:
        return
    vwap = panel["amount"] / panel["volume"].replace(0, np.nan)
    if backward is not None:
        bf = backward.reindex(index=vwap.index, columns=vwap.columns).ffill()
        vwap = vwap * bf
    panel["vwap"] = vwap
    if not verbose or "close" not in panel:
        return
    ratio = (panel["vwap"] / panel["close"]).stack()
    med = float(ratio.median()) if len(ratio) else float("nan")
    if backward is not None and abs(med - 1.0) < 0.1:
        print(f"[面板] vwap := backward × (amount/volume)；"
              f"vwap/close 中位数 {med:.4f}（口径自检通过）")
    elif backward is not None:
        print(f"[面板] ⚠️ vwap/close 中位数 {med:.4f} 仍偏离 1 —— 复权因子可能未对齐，"
              f"该 vwap 不可信")
    else:
        print(f"[面板] ⚠️ vwap := amount/volume（**未复权**）；vwap/close 中位数 {med:.4f}。"
              f"若 close 为后复权口径，此 vwap 与 close 不可比（中位数会等于 1/backward），"
              f"必须先调用方传入 backward 做复权。")


def build_mock_panel(n_days: int = 400, n_codes: int = 40, seed: int = 0
                     ) -> dict[str, pd.DataFrame]:
    """合成面板：收益含 20 日反转信号，供离线冒烟（不碰数据层）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = 0.25 * rets[t - 1] + rng.normal(0, 0.02, n_codes)
    close = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), idx, codes)
    open_ = close * (1 + rng.normal(0, 0.005, close.shape))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.006, close.shape))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.006, close.shape))
    volume = pd.DataFrame(rng.lognormal(13, 0.6, close.shape), idx, codes)
    amount = close * volume
    return {"open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "amount": amount, "vwap": amount / volume}


def build_real_panel(begin: int, end: int, cache_root: str | None = None,
                     sdk_cache: str | None = None, offline: bool = False):
    """真实 HS300 面板（历史成分并集池 + membership mask，消除幸存者偏差）。

    与 ``data.cache_helpers.build_panel`` 同源 —— 与
    ``scripts/factors/run_gflownet_phase1.build_real_panel`` 走的是同一个统一数据管道。
    """
    from data.cache_helpers import build_panel as _build_panel

    cfg = {"universe": {"index_code": "000300.SH", "adjust": "backward"}}
    panel, _returns = _build_panel(
        cfg, begin, end, cache_root=cache_root, sdk_cache=sdk_cache,
        offline=offline, include_market_cap=True, retry=True,
    )
    mask = panel["mask"]
    print(f"[面板] membership mask 每日在册中位数 "
          f"{int(mask.sum(axis=1).median())} 只（{mask.shape}）")

    # 复权因子：amount/volume 是原始口径，而 close 已被 build_panel 后复权 →
    # vwap 必须同样乘该因子才与 close 可比（2026-09-16 真实数据暴露的 bug）。
    # 复用 build_panel 已缓存的 backward 面板，不再重复开数据源。
    backward = None
    try:
        from data.cache import DataCache
        from data.cache_helpers import load_backward_factor
        if offline:
            # --offline 时绝不开真实 SDK（登录+接口 hang 曾卡死主流程 2.5h，09-24 修）
            from data.offline import OfflineDataSource
            ds = OfflineDataSource()
        else:
            from data.datasource import create_datasource
            from config import Config
            ds = create_datasource(Config.datasource())
        cache = DataCache(ds, cache_root=cache_root) if cache_root else DataCache(ds)
        backward = load_backward_factor(cache, list(panel["close"].columns))
    except Exception as exc:                        # noqa: BLE001 —— 复权因子缺失不应中断主流程
        print(f"[面板] ⚠️ 读取后复权因子失败（{type(exc).__name__}: {exc}）→ "
              f"vwap 将保持未复权口径，与 close 不可比，请检查缓存")
    panel["_backward"] = backward if backward is not None else pd.DataFrame()
    return panel


# ---------------------------------------------------------------------------
# 大模型定期更新回调
# ---------------------------------------------------------------------------
def _llm_callback_class():
    """延迟导入 sb3（避免无 sb3 环境下 import 即失败）。"""
    from stable_baselines3.common.callbacks import BaseCallback

    class LLMRefreshCallback(BaseCallback):
        """每 ``every_n_steps`` 步做一次「去弱留强」（研报 ``llm_every_n_steps``）。

        ``pool`` 由外部直接传入，**不走环境属性**：sb3 会把环境包进 ``Monitor``，
        ``model.env.envs[0]`` 拿到的是包装层而不是 ``AlphaPoolGymEnv``，从那里取
        ``.pool`` 会 AttributeError。
        """

        def __init__(self, pool: AlphaPool, proposer, *, every_n_steps: int,
                     n_new: int, drop_rl_n: int, features, origin: str = "llm",
                     verbose: int = 0):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.proposer = proposer
            self.every = int(every_n_steps)
            self.n_new = int(n_new)
            self.drop_rl_n = int(drop_rl_n)
            self.features = list(features)
            self.origin = origin
            self.history: list[dict] = []
            self._next = self.every

        def _on_step(self) -> bool:
            if self.proposer is None or self.every <= 0:
                return True
            if self.model.num_timesteps < self._next:
                return True
            self._next += self.every
            rep = refresh_pool(self.pool, self.proposer,
                               n_new=self.n_new, drop_rl_n=self.drop_rl_n,
                               features=self.features, origin=self.origin,
                               verbose=False)
            self.history.append({"step": int(self.model.num_timesteps),
                                 **rep.as_dict()})
            print(f"[llm] step {self.model.num_timesteps}: {rep.summary()}")
            return True

    return LLMRefreshCallback


def make_arm_proposer(args):
    """按 ``--arm`` 返回该臂的提案器（``None`` 表示无提案器）。

    三臂语义（2026-09-16，对齐研报 + 自造对照）：

    ==========  ====  ================  ==========================
    臂         初始池  LLM 知识          对应关系
    ==========  ====  ================  ==========================
    ``llm``     有     有（--llm）      研报「RL + 大模型」臂
    ``none``    无     无               研报「RL alone」臂
    ``random``  有     无（均匀采样）    自造：分离「池非空」与「有知识」
    ==========  ====  ================  ==========================

    ``random`` 臂的动机：研报只对比「有 LLM」vs「没 LLM」，但这两者**同时**差在
    两件事上 —— 初始池是否非空、以及提案是否含选股知识。``random`` 臂有初始池
    但纯随机无知识，故：

    - ``random`` − ``none`` ≈ **「初始池非空」的纯效应**
    - ``llm`` − ``random`` ≈ **「LLM 知识」的纯效应**

    注意 ``--arm random`` 时 :class:`UniformRandomProposer` **仍受与 LLM 相同的
    三/四道语义校验约束**（``check_report_formula`` 在 ``propose()`` 内部过滤），
    这是刻意的：对照臂该控制的是"知识来源"，不该放宽合法性标准。
    """
    from factor.rl.llm_pool import UniformRandomProposer

    arm = getattr(args, "arm", "llm")
    if arm == "none":
        return None
    if arm == "random":
        return UniformRandomProposer(
            seed=args.seed if args.random_seed is None else args.random_seed)
    # arm == "llm"
    prop = make_proposer(
        args.llm, model=args.llm_model, base_url=args.llm_base_url,
        max_tokens=args.llm_max_tokens, timeout=args.llm_timeout,
        rounds=args.llm_rounds)
    if prop is None:
        # --arm llm 配 --llm none 会静默退化成「和 --arm none 一模一样」，
        # 实验日志里却仍写着 arm=llm —— 是典型的静默降级，直接拒绝。
        raise SystemExit(
            "--arm llm 需要 --llm {template,openai}；"
            "若想跑「RL alone」请用 --arm none（--llm none 与之等价但语义不清）")
    return prop


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate_pool(pool: AlphaPool, panel: dict[str, pd.DataFrame],
                  rets: pd.DataFrame, label: str) -> dict:
    """在给定段上评估池内单因子与复合因子（池权重）的 IC。"""
    from factor.formula import formula_builder
    from stats.ic import calc_ic_series

    rows = []
    panels = []
    for i, f in enumerate(pool.formulas):
        try:
            fp = formula_builder(f, features=pool.features)(panel)
        except Exception as exc:                       # noqa: BLE001
            rows.append({"formula": f, "ic_mean": float("nan"),
                         "ic_ir": 0.0, "error": str(exc)[:60]})
            panels.append(None)
            continue
        ic = calc_ic_series(fp, rets).dropna()
        rows.append({
            "formula": f,
            "origin": pool.origins[i] if i < len(pool.origins) else "?",
            "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
            "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 and ic.std() > 0 else 0.0,
            "ic_abs_mean": float(ic.abs().mean()) if len(ic) else float("nan"),
        })
        panels.append(fp)

    out: dict = {"segment": label, "n_factors": len(pool.formulas),
                 "factors": rows}
    valid = [p for p in panels if p is not None]
    if valid and len(pool.weights) == len(panels):
        try:
            out["composite_ic"] = float(portfolio_ic(valid, pool.weights, rets))
        except Exception as exc:                       # noqa: BLE001
            out["composite_ic_error"] = str(exc)[:80]
        try:
            # 逐日复合 IC 序列（PBO/DSR 的 T×N 输入；与 portfolio_ic 同口径：
            # 逐日截面标准化后加权）
            import numpy as np
            from factor.rl.alphapool_env import _panels_to_np, _standardize_panel, _ic_np
            arr = _panels_to_np(valid)
            w = np.asarray(pool.weights, dtype=float)
            w = w / (np.abs(w).sum() + 1e-12)
            r = rets.reindex_like(valid[0]).to_numpy(dtype=np.float64)
            f0 = np.zeros(arr.shape[:2], dtype=np.float64)
            for k in range(arr.shape[2]):
                f0 += w[k] * _standardize_panel(arr[:, :, k])
            ic_daily = _ic_np(f0, r)
            ic_daily = [float(x) for x in ic_daily if np.isfinite(x)]
            out["composite_daily_ic"] = ic_daily
        except Exception as exc:                       # noqa: BLE001 —— 序列缺省不致命
            out["composite_daily_ic_error"] = str(exc)[:80]
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    从 ``main()`` 抽出（2026-09-16）以便单测直接断言参数语义
    （如 ``--arm`` 的闭集校验），无需真正跑训练。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="mock", choices=["mock", "real"])
    ap.add_argument("--offline", action="store_true", help="真实面板走本地缓存")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--sdk-cache", default=None)
    ap.add_argument("--train-begin", type=int, default=20190101)
    ap.add_argument("--train-end", type=int, default=20241231)
    ap.add_argument("--test-begin", type=int, default=20250101)
    ap.add_argument("--test-end", type=int, default=20260716)
    ap.add_argument("--horizon", type=int, default=10,
                    help="因子评估的收益期限（研报未给出，默认 10 交易日，须随结果注明）")

    # AlphaPool（研报超参）
    ap.add_argument("--capacity", type=int, default=10)
    ap.add_argument("--metric", default="ic", choices=["ic", "icir"])
    ap.add_argument("--corr-threshold", type=float, default=0.7)

    # RL（研报超参）
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lstm", "transformer"])
    ap.add_argument("--d-model", type=int, default=REPORT_D_MODEL)
    ap.add_argument("--n-layers", type=int, default=REPORT_N_LAYERS)
    ap.add_argument("--dropout", type=float, default=REPORT_DROPOUT)
    ap.add_argument("--total-timesteps", type=int, default=10000)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)

    # 实验臂（2026-09-16 新增，研报两臂 + 自造第三臂）
    ap.add_argument("--arm", default="llm", choices=["llm", "none", "random"],
                    help="实验臂。研报只有两臂："
                         "llm=RL+大模型（初始池由 LLM 构造 + 每 llm_every 步去弱留强）；"
                         "none=RL alone（空池起步、全程无 LLM，对应研报对照组）。"
                         "random=自造第三臂：初始池/刷新都用**均匀随机采样**，"
                         "用于把『LLM 的知识』与『初始池非空』两个效应拆开"
                         "（random 有初始池但无知识；none 两者皆无；llm 两者皆有）")
    ap.add_argument("--random-seed", type=int, default=None,
                    help="[--arm random] 均匀采样器 seed，默认 = --seed")

    # 大模型（研报超参 + 自拟参数）
    ap.add_argument("--llm", default="template",
                    choices=["none", "template", "openai"],
                    help="template=离线确定性模板兜底（非真实大模型）；"
                         "openai=真实接口（需 DEEPSEEK_API_KEY）")
    ap.add_argument("--llm-init", type=int, default=20,
                    help="初始池候选数【自拟参数，研报无对应超参】")
    ap.add_argument("--llm-every", type=int, default=3000,
                    help="研报 llm_every_n_steps = 3000")
    ap.add_argument("--drop-rl-n", type=int, default=5,
                    help="研报 drop_rl_n = 5")
    ap.add_argument("--llm-new", type=int, default=None,
                    help="每次刷新注入的因子数（默认 = --drop-rl-n，自拟参数）")
    # --llm openai 的连接参数（2026-09-16 已联网验证）
    ap.add_argument("--llm-model", default="deepseek-flash",
                    help="[openai] 模型名；本项目验证过 deepseek-flash")
    ap.add_argument("--llm-base-url", default="https://api.deepseek.com",
                    help="[openai] OpenAI 兼容端点根（DeepSeek 根路径即可，无需 /v1）")
    ap.add_argument("--llm-max-tokens", type=int, default=16000,
                    help="[openai] 单次回复 token 上限，**含思维链**。推理模型"
                         "（deepseek-flash/v4-pro）给太小会把预算全花在 reasoning 上"
                         "→ 返回空 content。实测 16000 可用")
    ap.add_argument("--llm-timeout", type=float, default=300.0,
                    help="[openai] 单次请求超时（秒）；推理模型单次 15~50s")
    ap.add_argument("--llm-rounds", type=int, default=2,
                    help="[openai] 最多请求轮数（凑不满候选就再要一轮）")
    ap.add_argument("--out-prefix", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()

    t0 = time.time()
    # 输出目录**必须含臂名与 seed**，否则三臂同参跑会互相覆盖（2026-09-16 修）。
    # 臂名取 --arm（llm/none/random），LLM 具体实现（template/openai）附在后面，
    # 便于一眼看出「llm 臂用的是模板兜底还是真接口」。
    _arm_tag = args.arm if args.arm != "llm" else f"llm_{args.llm}"
    prefix = args.out_prefix or (
        f"reports/alphapool_ppo_{args.panel}_{args.policy}_{_arm_tag}_seed{args.seed}")
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. 面板 ----
    if args.panel == "mock":
        panel = build_mock_panel()
        close = panel["close"]
    else:
        panel = build_real_panel(args.train_begin, args.test_end,
                                 cache_root=args.cache_root,
                                 sdk_cache=args.sdk_cache, offline=args.offline)
        close = panel["close_m"]
        _bf = panel.pop("_backward", None)
        attach_vwap(panel, backward=_bf if (_bf is not None and len(_bf)) else None)
    features = list(FIELDS)
    missing = [f for f in features if f not in panel]
    if missing:
        raise SystemExit(f"面板缺少研报 6 字段中的 {missing}")

    # ---- 2. 训练 / 测试切片（收益只在切片内计算，杜绝跨界泄漏）----
    # mock 面板是合成日期序列，与真实默认区间无关 → 按 75/25 切；
    # real 面板用显式日期参数（这是唯一有意义的切法）。
    if args.panel == "mock":
        cut = int(len(close) * 0.75)
        tr_idx, te_idx = close.index[:cut], close.index[cut:]
        print(f"[数据] mock 面板按 75/25 切分：训练段截止 {tr_idx[-1].date()}，"
              f"测试段自 {te_idx[0].date()} 起")
    else:
        tr_end = pd.Timestamp(str(args.train_end))
        te_beg = pd.Timestamp(str(args.test_begin))
        tr_idx = close.index[close.index <= tr_end]
        te_idx = close.index[close.index >= te_beg]
    if len(tr_idx) == 0 or len(te_idx) == 0:
        raise SystemExit(f"训练段 {len(tr_idx)} 日 / 测试段 {len(te_idx)} 日为空，检查日期参数")
    panel_tr = {k: v.loc[tr_idx] for k, v in panel.items()}
    panel_te = {k: v.loc[te_idx] for k, v in panel.items()}
    rets_tr = forward_returns(panel["close"].loc[tr_idx], args.horizon)
    rets_te = forward_returns(panel["close"].loc[te_idx], args.horizon)
    print(f"[数据] 面板={close.shape}；训练 {tr_idx[0].date()}~{tr_idx[-1].date()}"
          f"（{len(tr_idx)} 日）/ 测试 {te_idx[0].date()}~{te_idx[-1].date()}"
          f"（{len(te_idx)} 日）；horizon={args.horizon} 日")

    # ---- 3. 构造基础池（按实验臂选择提案器） ----
    # 三臂共用一个代码路径，唯一差异就是这里的 proposer 是否为 None、
    # 以及它是「有知识的」（LLM）还是「无知识的」（均匀采样）。
    proposer = make_arm_proposer(args)
    # 池内 origin 标记：LLM 臂沿用研报语义 "llm"；random 臂标 "random"，
    # 避免日志里 random 臂被读成 LLM 臂（两者池内因子数量级相同，极易误读）。
    seed_origin = "random" if args.arm == "random" else "llm"
    pool = AlphaPool(panel_tr, rets_tr, features, capacity=args.capacity,
                     metric=args.metric, corr_threshold=args.corr_threshold,
                     seed=args.seed)
    seed_rep = PoolUpdateReport(stage="seed")
    if proposer is not None:
        t_llm = time.time()
        candidates = proposer.propose(args.llm_init,
                                      existing=(), avoid=())
        print(f"[llm] arm={args.arm} {proposer.name} 提案 {len(candidates)} 条，"
              f"耗时 {time.time() - t_llm:.1f}s")
        seed_rep = seed_pool(pool, candidates, features=features,
                             origin=seed_origin, verbose=False)
        print(f"[llm] 校验/入池：提案 {seed_rep.proposed} → 入池 "
              f"{seed_rep.accepted}；status={seed_rep.status_counts}")
        print(f"[llm] best_obj {seed_rep.best_obj_before:.4f} → "
              f"{seed_rep.best_obj_after:.4f}（{seed_rep.best_obj_delta:+.4f}）")
        if seed_rep.accepted == 0:
            print("[llm] ⚠️ 一条都没入池 —— 检查大模型回复是否为空、"
                  "或公式是否全部违反研报约束（看上面的 rejected 原因）")
    else:
        print(f"[llm] arm={args.arm}：不构造初始池（空池起步，全程无 LLM）")
    print(f"[pool] 起始池内 {len(pool.formulas)} 因子 {pool.origin_counts()}，"
          f"best_obj={pool.best_obj:.4f}")

    # ---- 4. 训练 ----
    env = AlphaPoolGymEnv(panel_tr, rets_tr, features=features,
                          capacity=args.capacity, metric=args.metric,
                          corr_threshold=args.corr_threshold, seed=args.seed)
    env.pool = pool                              # 让环境复用已经喂好的初始池
    policy_kwargs = make_extractor_kwargs(
        args.policy, n_layers=args.n_layers, d_model=args.d_model,
        dropout=args.dropout)

    from sb3_contrib import MaskablePPO

    cb_cls = _llm_callback_class()
    callback = cb_cls(pool, proposer, every_n_steps=args.llm_every,
                      n_new=args.llm_new or args.drop_rl_n,
                      drop_rl_n=args.drop_rl_n, features=features,
                      origin=seed_origin)
    model = MaskablePPO(
        "MlpPolicy", env, learning_rate=args.learning_rate,
        n_steps=args.n_steps, batch_size=args.batch_size, gamma=args.gamma,
        ent_coef=args.ent_coef, policy_kwargs=policy_kwargs or None,
        seed=args.seed, verbose=0,
    )
    print(f"[rl] MaskablePPO policy={args.policy} n_actions={env.n_actions} "
          f"total_timesteps={args.total_timesteps}")
    model.learn(total_timesteps=args.total_timesteps, callback=callback,
                progress_bar=False)

    # ---- 5. 评估 ----
    train_eval = evaluate_pool(pool, panel_tr, rets_tr, "train")
    oos_eval = evaluate_pool(pool, panel_te, rets_te, "test")
    print(f"[pool] 结束池内 {len(pool.formulas)} 因子 {pool.origin_counts()}，"
          f"best_obj={pool.best_obj:.4f}")
    print(f"[eval] 训练段复合 IC {train_eval.get('composite_ic', float('nan')):.4f}；"
          f"测试段复合 IC {oos_eval.get('composite_ic', float('nan')):.4f}")
    print(f"[eval] 测试段单因子 |IC| 均值 "
          f"{np.nanmean([r.get('ic_abs_mean', np.nan) for r in oos_eval['factors']]):.4f}")

    result = {
        "args": vars(args),
        "n_actions": int(env.n_actions),
        "max_expr_length": MAX_EXPR_LENGTH,
        "windows": list(WINDOWS),
        "seed_report": seed_rep.as_dict(),
        "llm_history": callback.history,
        "pool_stats": pool.stats(),
        "evaluation": {"train": train_eval, "test": oos_eval},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(f"{prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"[done] 耗时 {result['elapsed_sec']}s → {prefix}.json")


if __name__ == "__main__":
    main()
