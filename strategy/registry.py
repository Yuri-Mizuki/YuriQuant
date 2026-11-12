"""定版策略族注册表 —— `yqcore`（YQ 核心）。

为什么需要这个文件
------------------
2026-11-10 的集中度研究把「持仓宽度」从隐含假设变成了显式结论：

- **882 纯量价（`ens_h1h5`）是头部型 alpha**：毛收益在**前 0.5%（≈25 只）见峰 28.4%**，
  放到 50% 只剩 20.0%，头部溢价 +8.4pp ⇒ **集中不损失收益**（TopK15~35 全在 21.9~24.9%）。
- **920 定版（`dingban_fundind`，h1020 ⊕ 慢信号 s0.5）是广度型 alpha**：前 2% 反而比 5%
  更低（22.5 vs 24.1），**要放到 5%（≈250 只）才见峰** ⇒ 一集中就丢（25 只掉 3.5~5.6pp）。

两者不是「新旧版本」，是**同一目标下两种 alpha 形状**，适用资金规模不同。故登记为
**一个策略族的多个档位**（tier），而不是互相取代的版本：

==============================  =================  ============  ================  ============
档位                            信号               持仓          适用资金          一句话
==============================  =================  ============  ================  ============
``yqcore-breadth``（广度档）    dingban_fundind   ~628 只       ≥ 200 万          回撤低 7pp
``yqcore-focus``（集中档）      gbdt_mid13        8~25 只       10~200 万         收益高 7pp
``yqcore-solo``（独苗档）       gbdt_mid13        ≤8 只（实 ~6） 10~50 万          行业去重，+7pp
==============================  =================  ============  ================  ============

**`solo` 是 focus 的资源约束变体**，不是第四个方向：信号相同，只在「每个申万一级行业
最多留一只」上更严。10 万整手口径下同实际持仓数（≈5 只）年化 31.0% → **38.3%**、
Sharpe 1.08 → 1.23，且过了随机行业标签安慰剂（10 次最高 33.8%，真实 38.3%）。
它是目前**唯一一条 2024-2026 样本外年化高于 2019-2023 的线**（39.4% vs 33.6%）。
但**回撤没降**（36.4%），要低回撤只能按 `gross_exposure` 线性降仓。证据
``reports/focus_micro/README.md``。

**2026-11-10 当天把集中档的信号换了**（`ens_h1h5` → `gbdt_mid13`）。原因不是「换个数字好看的」，
而是 `ens_h1h5` **只用了 h1+h5 两个短周期**、把 h20（长周期选择器）漏在外面：在
K=6..30 的 11 个格子上，`ens_h1h5` 的年化均值 22.1%，而任何**含 h20** 的组合都在
28~30%。机制与「h1 只负责剔除涨过头的、h20 才负责选股」一致。四个 horizon 的权重
按 2:1:1:2（h1/h20 各 1/3、两个中期合计 1/3）—— 因为它同时满足「需要中期信息」与
「不必在 h5/h10 之间选边」（两者差 1.5pp 但无机制，视为噪声）。证据
``reports/focus_upgrade/README.md``。旧信号的数字留在登记表的 ``history`` 里。

命名
----
族名**刻意不带版本号**（所以不叫 `v3-core`）：版本是基线的事（`v1w-ortho882` → `v2-dingban`
→ …），由 `config/baselines.yaml` 的 `baseline_id` 与报告里的数据态指纹承担。产品名一旦
跟着版本走，每次基线换代都得改名，所有指针随之失效。档位词同理取**机制**（breadth / focus
描述 alpha 的宽度形状）而非尺寸（wide / narrow 描述持仓数量），因为资金规模和持仓数都会变。

口径纪律
--------
两条档位**同窗口、同执行层、同费率**（prod 2019 起 / T+1 open / 固定 10bp / 1e6），
故**可以互相比较**；但：
- 边界止 **2026-09-07**（882 的 pred 面板末日；920 重建版到 11-04，不对齐会让 2026 混进
  36 个交易日、虚低 1.4~1.6pp）⇒ 与 `v2-dingban` 登记的 1876 日（止 09-23）**差 12 个交易日，
  数字不可逐位对比**。
- `focus` 档的 K 曲线是**锯齿**（K=8 30.9 / K=15 28.8 / K=25 30.9），12 个格子极差 4.2pp；
  **且 K 之间的差异统计上不显著**：配对 bootstrap（同月度对齐，vs K=15）K=5 +2.32pp
  CI[−4.13,+9.06]、K=6 +2.72pp CI[−2.93,+8.51]、K=8 +1.94pp CI[−3.05,+6.96]、
  K=20 −0.82pp CI[−3.15,+1.53] ⇒ **九个区间全含 0**。故实盘按 **28~29%** 预期，
  **K 一律按整手约束与单票风险定，不许按年化挑**（更不许说「某 K 收益更高」）。
  另：均值最高的两条臂 IS/OOS 跨格秩相关 **−0.71 / −0.66**（p<0.02）。
  证据 ``reports/focus_holding/README.md`` §2.2。
- `focus` 档用 882 信号 ⇒ **不是 `v2-dingban` 的替代品**，是另一条线；判「是否推翻定版」
  仍只认 `baselines.yaml` 里 `status: production` 的 `v2-dingban`。

参数来源（2026-11-12 消双真源）
--------------------------------
档位的**数字**（``signal_key`` / 策略类 / 构造参数 / 所需面板）全部从
``config/baselines.yaml`` 经 :func:`tier_wiring` 读；本文件只留**代码该管的东西**：
有哪些档、怎么描述（label / notes）、资金边界（容量结论）。改口径只改登记表一处。

三道闸防悄悄漂移（都在 import 期跑）：
- 登记表写了构造器不认的参数名 → ``ValueError``（不静默忽略）；
- ``_REQUIRED_PARAMS`` 里的必填键缺失 → ``ValueError``（不落回构造器缺省值，
  否则 ``BufferedTopKLongOnly`` 的缺省 k=45 会把口径变成另一条策略）；
- 登记了 ``family: yqcore`` 但代码没定义 → :func:`assert_no_orphan_registration` 报错。

用法
----
    from strategy.registry import build, describe, list_tiers

    sig_key, strat = build("yqcore-focus")            # 默认 K=15 + buf2（20 万以上）
    sig_key, strat = build("yqcore-focus", k=8)       # 10 万：整手实投 92%，K=15 只有 82%
    ind = pd.read_parquet(...)                        # 行业面板
    sig_key, strat = build("yqcore-solo", industry=ind)   # 10 万，实际 ~6 只
    print(describe("yqcore-focus"))
"""
from __future__ import annotations

import inspect
import unicodedata
from dataclasses import dataclass, field

from config import get_baseline, load_baselines
from strategy.cross_section import (  # noqa: F401
    BufferedTopFracLongOnly,
    BufferedTopKLongOnly,
    SectorCappedBufferedTopKLongOnly,
)


@dataclass(frozen=True)
class TierSpec:
    """一个档位的完整体征（信号 + 构造器 + 适用边界）。"""

    name: str
    label: str
    signal_key: str
    strategy: str                    # 类名，运行时由 ``_FACTORIES`` 解析
    params: dict = field(default_factory=dict)
    min_capital: float = 0.0
    max_capital: float = float("inf")
    notes: str = ""
    #: 本机代理信号键（登记口径的 pred 不在本机时用它复算，例如 `dingban_fundind`
    #: 的登记路径 `reports/frontier_audit_920_reval/` 在本机已缺失）。None = 与
    #: ``signal_key`` 同源。**用它会换数据态**，报数时必须标注。
    signal_key_local: str | None = None
    #: 构造器需要的**外部面板**（如 ``industry`` 行业截面）。None = 只需信号。
    #: 有值时必须由调用方经 ``build(..., <panel>=df)`` 注入，注册表本身不读数据
    #: （保持 strategy 层不依赖数据层，见 tests/test_layering.py）。
    panel: str | None = None
    #: 构造器里接收该面板的**参数名**（``panel="industry"`` → ``sectors=``）。
    #: None = 与 ``panel`` 同名。
    panel_arg: str | None = None

    @property
    def runnable_signal(self) -> str:
        """本机实际能读到的信号键。"""
        return self.signal_key_local or self.signal_key

    def build(self, **overrides):
        if self.panel is not None:
            arg = self.panel_arg or self.panel
            if arg not in overrides:
                if self.panel in overrides:
                    val = overrides.pop(self.panel)      # 先 pop 再展开，否则两个键都在
                    overrides = {**overrides, arg: val}
                else:
                    raise TypeError(
                        f"{self.name} 需要外部面板 {self.panel!r}，须显式注入："
                        f"build(\"{self.name}\", {self.panel}=df)")
        params = {**self.params, **overrides}
        cls = _FACTORIES[self.strategy]
        return cls(**params)


_FACTORIES = {
    "BufferedTopKLongOnly": BufferedTopKLongOnly,
    "BufferedTopFracLongOnly": BufferedTopFracLongOnly,
    "SectorCappedBufferedTopKLongOnly": SectorCappedBufferedTopKLongOnly,
}

FAMILY = "yqcore"
FAMILY_LABEL = "YQ 核心（breadth / focus / solo 三档）"

#: ``execution`` 段里**不是构造参数**的键（执行方式与基准，属口径不属策略）。
_EXEC_NON_PARAM_KEYS = frozenset({"strategy", "rebalance_freq", "benchmark"})

#: 登记表按「缓冲带」语义命名（``buffer_entry``），构造器按 ``frac_*`` 命名。
#: 别名只映射**方向**，值仍从登记表读，不构成第二份数字。
_PARAM_ALIASES = {"buffer_entry": "frac_entry", "buffer_exit": "frac_exit"}

#: 每档**必须**在登记表里显式给出的构造参数名。
#: 只钉「有没有登记」，不钉「值是多少」——缺了就 import 期报错，避免悄悄落回
#: 构造器缺省值（``BufferedTopKLongOnly`` 的缺省 k=45 会把口径变成另一条策略）。
_REQUIRED_PARAMS = {
    "yqcore-breadth": ("frac_entry", "frac_exit"),
    "yqcore-focus": ("k", "exit_mult"),
    "yqcore-solo": ("k", "exit_mult", "max_per_sector", "min_sectors"),
}


def _strategy_params(tier_id: str, execution: dict, strategy: str) -> dict:
    """把登记表 ``execution`` 段解析成构造器 kwargs（含别名与未知键校验）。"""
    cls = _FACTORIES[strategy]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    params: dict = {}
    unknown: list[str] = []
    for key, val in execution.items():
        if key in _EXEC_NON_PARAM_KEYS:
            continue
        name = _PARAM_ALIASES.get(key, key)
        if name in accepted:
            params[name] = val
        else:
            unknown.append(f"execution.{key}")
    if unknown:
        raise ValueError(
            f"{tier_id}: 登记表含 {cls.__name__} 不接受的构造参数 {unknown}"
            "（若属执行口径，请加进 registry._EXEC_NON_PARAM_KEYS）")
    missing = [n for n in _REQUIRED_PARAMS[tier_id] if n not in params]
    if missing:
        raise ValueError(f"{tier_id}: 登记表 execution 段缺必填参数 {missing}")
    return params


def tier_wiring(tier_id: str, registry: dict | None = None) -> dict:
    """取一档的**数字与构造信息**：信号键 / 策略类 / 构造参数 / 所需面板。

    这是本模块与 ``config/baselines.yaml`` 的**唯一接线点**——登记表改口径，
    这里跟着变，代码不动。档位的「标签 / 长注释 / 资金边界」仍留在代码中：
    它们是产品说明与容量结论，不是登记数字。

    Args:
        tier_id: 登记表里的 ``id``（= 档位名）。
        registry: 显式登记表字典（测试注入用）；None = 读 ``config/baselines.yaml``。
    """
    b = get_baseline(tier_id, registry)
    if b.get("family") != FAMILY:
        raise ValueError(f"{tier_id}: 登记表 family={b.get('family')!r}，不是 {FAMILY!r}")
    execution = b.get("execution") or {}
    strategy = execution.get("strategy")
    if not strategy:
        raise ValueError(f"{tier_id}: 登记表缺 execution.strategy")
    return {
        "signal_key": b["signal_key"],
        "signal_key_local": b.get("signal_key_local"),
        "strategy": strategy,
        "params": _strategy_params(tier_id, execution, strategy),
        "panel": b.get("requires_panel"),
    }


TIERS: dict[str, TierSpec] = {
    "yqcore-breadth": TierSpec(
        name="yqcore-breadth",
        label="广度档（Breadth）· 920 定版信号 + buffer(15/40)",
        **tier_wiring("yqcore-breadth"),
        min_capital=1.5e7,
        notes=(
            "alpha 分布在 ~5% 宽度上，持仓 620+ 只时收益/风险最优。"
            "⚠️ 登记口径的 pred 本机已缺失（`frontier_audit_920_reval/`），"
            "复算走 `signal_key_local` 的**重建代理** ⇒ 换数据态，数字不等同 v2 登记值。"
            "🔑 **容量**：单票权重 0.14%、ADV 中位 0.50 亿 ⇒ 容量中位 **34.5 亿**（p10 9.7 亿）"
            "⇒ 本档是唯一能装大资金的档。代价是收益：2000 万整手 19.84%、Sharpe 1.02；"
            "而回撤只有 24.6%（三档最低）⇒ 低波偏好者选它。"
        ),
    ),
    "yqcore-focus": TierSpec(
        name="yqcore-focus",
        label="集中档（Focus）· 882 四周期融合信号 + TopK + buffer(×2)",
        **tier_wiring("yqcore-focus"),
        min_capital=6.0e6,
        max_capital=1.5e7,
        notes=(
            "alpha 集中在最头部，不必摊宽。信号 = gbdt h1:h5:h10:h20 = 2:1:1:2 秩平均"
            "（复合信号，caliber 现场构造）。2026-11-10 从 `ens_h1h5` 升级：只用 h1+h5 时"
            " K 曲线均值 22.1%，含 h20 的组合 28~30% ⇒ 补上长周期选择器"
            "（旧数字见 baselines 的 history）。"
            "**K 按资金配，不按年化配**：配对 bootstrap 证 K∈[5,25] 年化无显著差异，"
            "真正确定的是整手约束——10 万本金下 K=15 只能建 82% 仓位（等效 23.7%），"
            "K=8 能建 92%（等效 28.6%）⇒ 10 万用 k=8，20 万+ 用默认 15，50 万+ 可到 25。"
            "权重维持等权：9 种加权方案 × 5 个 K 全在 ±2pp 且不显著，"
            "把权重压给信号头部（softmax 给第一名 40%）反而亏 2.7~14.9pp。"
            "⚠️ 成交额分位 0.186 ⇒ 组合偏小盘，K 越小尾部风险越集中"
            "（k=5 时一只 −30% 就是组合 −6%）。证据 reports/focus_holding/README.md。"
            "🔑 **产品线角色（2026-11-10 重定）**：小资金位已被 `yqcore-solo` 取代"
            "（同资金下 solo 收益/Sharpe 全面更高）。本档现在是**容量中间台阶**："
            "单票权重 6.6%、容量中位 3900 万（p10 1500 万），承接 solo 装不下的 600~1500 万。"
            "它在 600 万以下仍可跑，只是不再是最优选择。"
        ),
    ),
    "yqcore-solo": TierSpec(
        name="yqcore-solo",
        label="独苗档（Solo）· 882 四周期信号 + TopK8 + 每申万一级 ≤1 只",
        **tier_wiring("yqcore-solo"),
        panel_arg="sectors",
        min_capital=5.0e4,
        max_capital=6.0e6,
        notes=(
            "focus 档 + **行业去重**（每个申万一级行业只留最强的一只）。"
            "2026-11-10 实测（10 万整手，**同参数对照** k=8 b=1.0）：年化 29.08%→35.57%、"
            "Sharpe 1.075→1.203（+6.48pp / +0.128）。"
            "🚨 **但这两个数字是 phase=0 单点值**：21 相位逐相位配对差均值只有 **+1.39pp**"
            "（+6.48pp 是 21 个里的最大值，7 个相位为负），年度轴 **+1.66pp**（t=1.72，正 6/8 年，"
            "其中 2025 一年 +6.86pp，扣掉只剩 +0.91pp/年），跨相位安慰剂 **90 分位**"
            "（真实 24.22% vs 随机 22.69%±0.85）⇒ **增益为真，但量级是 1.5pp 级不是 6pp 级**。"
            "引用 +6.48pp **必须**注明 phase=0 单点。证据 `rtl_rebalance_timing/VARIANTS.md`。"
            "机制（**2026-11-10 已按实现修正**）：信号在 TopK 内会扎堆同一行业"
            "（同一波行情里同行业股票一起得高分；实测 8 只平均只有 5.87 个行业、"
            "92.4% 的期数有重复）。`fill_to_k=False` ⇒ **被行业挡掉的票直接踢掉、不往下补位**"
            "（旧文案「换成其他行业最强股」是错的，策略不做该替换）；"
            "「踢掉 vs 往下补第 9~16 名」逐期配对 −0.022pp/月、t=−0.03 ⇒ 换不换无质量差异。"
            "🔑 去重真正的优点是**行业权重上限**：最大单行业权重从均值 32.1%（最坏 75%）"
            "压到恒 17.1% —— 风险约束，不是 alpha。证据 `rtl_rebalance_timing/SECTOR_DEDUP.md`。"
            "🚨 **实际持仓少于 k**：TopK8 内常有 2~3 只同行业，实测平均 **5.9 只**、"
            "分布 3只×4/4只×8/5只×24/6只×24/7只×25/8只×8（93 期），最少 3 只。"
            "实投率 98.3%（利润再投入）/ 96.3%（利润取走）、8 年全正、最差年 +8.1%、"
            "滚动 IS(19-23) 33.6% / OOS(24-26) 39.4%（⚠️ 时段效应，非稳健性证据）。"
            "`min_sectors=3` 是**行业数硬下限**：在 k=8 上 8 年 93 期**从未触发**（持仓数≡行业数，"
            "最少 3 个），故所有数字与不设时逐位相同；它的价值是尾部保险 —— k=6 b1.5 上确实触发"
            "1 期（2 只→3 只），年化 +0.18pp、最差年 +10.9%→+12.1%。"
            "⚠️ 回撤 36.4%（与 focus 档同量级）—— 想要低回撤只能降仓：gross_exposure=0.75 "
            "给 28.2% 回撤 / 28.4% 年化，Sharpe 不变（线性缩放）。"
            "❌ **不要每年重选参数**：滚动重估（每年按 IS Sharpe 重选 k/buf）跑输固定 k8 1.0pp"
            "（30.02% vs 31.00%），且回撤一点没降 ⇒ 固定参数本身就是对参数不确定性的防御。"
            "🔑 **容量（产品线分界的主依据）**：单票权重中位 16.4%、持仓 ADV 中位 0.26 亿"
            "（p10 0.10 亿）⇒ 单票 ≤ADV 10% 口径下组合容量中位 **1600 万 / p10 600 万**。"
            "所以本档的资金区间是 **5 万~600 万**（p10 容量为界），不是靠收益定的 ——"
            "收益上本档在各资金档都最高（100 万 36.31%、500 万 36.36%），"
            "再往上换档换的是「装得下」，不是「赚更多」。"
            "证据 reports/focus_micro/README.md、reports/yq_core/README.md。"
        ),
    ),
}


def assert_no_orphan_registration(registry: dict | None = None) -> None:
    """反向校验：登记表里不能有代码不认识的 ``family: yqcore`` 条目。

    正向（登记 → 代码）由 :func:`tier_wiring` 保证；反向没有它会静默漏：新增一档
    只写进 yaml、忘了在 ``TIERS`` 里给标签与资金边界，产品线就会少一档而无人报错。

    Args:
        registry: 显式登记表字典（测试注入用）；None = 读 ``config/baselines.yaml``。
    """
    reg = registry if registry is not None else load_baselines()
    fam = {b.get("id") for b in (reg.get("baselines") or [])
           if b.get("family") == FAMILY}
    orphans = sorted(str(x) for x in fam - set(TIERS))
    if orphans:
        raise ValueError(
            f"登记表里有 {FAMILY} 族条目未在 registry.TIERS 定义（缺标签/资金边界）：{orphans}")


assert_no_orphan_registration()


def get_tier(name: str) -> TierSpec:
    if name not in TIERS:
        raise KeyError(f"未知档位 {name!r}，可选 {sorted(TIERS)}")
    return TIERS[name]


def build(name: str, **overrides) -> tuple[str, object]:
    """按档位名构造策略。返回 ``(signal_key, strategy_instance)``。

    ``signal_key`` 是**本机可读**的那个键（见 :attr:`TierSpec.runnable_signal`）。
    """
    spec = get_tier(name)
    return spec.runnable_signal, spec.build(**overrides)


def _width(text: str) -> int:
    """显示宽度：东亚全角字符占 2 列。

    ``str.ljust`` 按码位补空格，表头含中文时会与数据行错位，故自己算宽度。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def list_tiers() -> str:
    cols = (16, 20, 16, 0)          # 最后一列（适用资金）不补空格
    lines = [f"{FAMILY} — {FAMILY_LABEL}", ""]
    head = "".join(_pad(v, w) for v, w in zip(("档位", "信号", "持仓", "适用资金"), cols))
    lines += [head, "-" * _width(head)]
    for t in TIERS.values():
        hold = ("~610 只" if t.strategy == "BufferedTopFracLongOnly"
                else f"≤{t.params.get('k', '?')} 只（实 5.9）"
                if t.strategy == "SectorCappedBufferedTopKLongOnly"
                else f"{t.params.get('k', '?')} 只")
        cap = (f"{t.min_capital/1e4:.0f} 万以上" if t.max_capital == float("inf")
               else f"{t.min_capital/1e4:.0f}~{t.max_capital/1e4:.0f} 万")
        sig = t.signal_key + ("*" if t.signal_key_local else "")
        lines.append("".join(_pad(v, w)
                             for v, w in zip((t.name, sig, hold, cap), cols)))
    lines.append("\n* = 本机用重建代理复算（换数据态，非登记指纹）")
    lines.append("资金区间按**容量**划分（单票 ≤ ADV 的 10%）；收益上 solo 各档都最高，")
    lines.append("往上一档换的是「装得下」，不是「赚更多」。")
    return "\n".join(lines)


def describe(name: str) -> str:
    t = get_tier(name)
    return (f"[{t.name}] {t.label}\n"
            f"  信号: {t.signal_key} | 构造: {t.strategy}({t.params})\n"
            f"  {t.notes}")


__all__ = ["FAMILY", "FAMILY_LABEL", "TierSpec", "TIERS",
           "assert_no_orphan_registration", "tier_wiring",
           "get_tier", "build", "list_tiers", "describe"]
