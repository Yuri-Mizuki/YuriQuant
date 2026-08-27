"""
AlphaPool：P0 最小闭环环境（对齐华泰 AI97《大模型+强化学习因子挖掘》）
======================================================================

研报方法还原（Method 节）：

- **Token 化**：逆波兰式（RPN）序列。``BEG``/``SEP`` 为起止符，动作空间 = 6 字段
  （open/high/low/close/volume/vwap）+ 13 常数 + 21 算子。**滚动算子的窗口是
  算子内建参数**（``Mean($close,20)`` 的 20 属于算子定义，Yu et al. 2023 的
  RPN 表达式即 ``[close, Mean_20]`` 形态），因此每个 (滚动算子, 窗口) 组合为
  一个独立 token。表达式最大长度 ``MAX_EXPR_LENGTH=15`` token。
- **MDP**：状态 = 已生成 token 序列；动作 = 下一步 token；终止 = 输出 ``SEP``
  或超长（研报：超长给 0 奖励并评估）。
- **AlphaPool 评估**：``MSE Pool`` 用**因子组合 IC** 作评估指标（``MeanStd Pool``
  用 ICIR，P0 只实现 MSE Pool）。每入池一个因子，用 Adam 优化因子组合权重
  使组合 IC 最大，奖励 = 组合最新评估指标 ``new_obj``。
- **奖励（4 档，研报图表 11）**：
  - 因子无效（RPN 不完整 / 求值异常）→ **-1**
  - 空值 / 异常值 / 超长度限制 → **0**
  - 位于失败缓存 / 效果无法入池 → 当前池最优评估指标 ``best_obj``
  - 成功入池 → 因子池更新后最新评估指标 ``new_obj``

设计说明：
- 算子用项目 ``factor.operators`` 注册表（P0 取研报 21 算子映射，字段统一小写）。
- 求值走 ``factor.formula.formula_builder``（GP 前缀语法 + 求值缓存），与既有
  GFlowNet / GP 挖掘同一链路。
- 本文件只实现「环境 + 池 + 奖励」核心逻辑，**不依赖 sb3**：纯 numpy/pandas，
  便于单元测试与 mock 验证；gym 封装与训练在 ``alphapool_gym.py``。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from factor.operators import op_registry

# ---------------------------------------------------------------------------
# Token 空间（研报图表 8/9/10）
# ---------------------------------------------------------------------------

FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "vwap")
CONSTANTS: tuple[float, ...] = (-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01,
                                0.5, 1.0, 2.0, 5.0, 10.0, 30.0)

# 研报 21 算子 -> 项目注册表名
REPORT_OPERATORS: dict[str, str] = {
    # 一元
    "Abs": "abs", "Log": "log",
    # 二元
    "Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div",
    "Greater": "greater", "Less": "less",
    # 滚动（窗口内建）
    "Ref": "ts_ref", "Mean": "ts_mean", "Sum": "ts_sum", "Std": "ts_std",
    "Var": "ts_var", "Max": "ts_max", "Min": "ts_min",
    "Med": "ts_median", "Mad": "ts_avedev", "Delta": "ts_delta",
    "WMA": "ts_wma", "EMA": "ts_ema",
    # 配对滚动（窗口内建）
    "Cov": "ts_cov", "Corr": "ts_corr",
}

# 滚动算子窗口候选（研报常数表与常见取值）
WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 60)

BEG, SEP = 0, 1
FIRST_ACTION = 2                # 0/1 保留给 BEG/SEP

MAX_EXPR_LENGTH = 15            # 研报：表达式最大长度（token，含 BEG/SEP）

# 不带窗口的算子
_NON_WINDOW_REPORT = {"Abs", "Log", "Add", "Sub", "Mul", "Div", "Greater", "Less"}

# 算子 token：非滚动算子 1 个 token；滚动算子 每窗口 1 个 token
# (kind, op_report_name, window|None)


def _build_op_tokens() -> list[tuple[str, str, Optional[int]]]:
    ops: list[tuple[str, str, Optional[int]]] = []
    for name in REPORT_OPERATORS:
        if name in _NON_WINDOW_REPORT:
            ops.append(("op", name, None))
        else:
            for w in WINDOWS:
                ops.append(("op", name, w))
    return ops


OP_TOKENS: list[tuple[str, str, Optional[int]]] = _build_op_tokens()


def _build_token_space() -> dict[int, tuple[str, object]]:
    """token_id -> (kind, value)。布局：[BEG][SEP][op...][field...][const...]"""
    out: dict[int, tuple[str, object]] = {BEG: ("beg", None), SEP: ("sep", None)}
    i = FIRST_ACTION
    for _kind, name, win in OP_TOKENS:
        out[i] = ("op", (name, win))
        i += 1
    for f in FIELDS:
        out[i] = ("field", f)
        i += 1
    for c in CONSTANTS:
        out[i] = ("const", c)
        i += 1
    return out


TOKEN_SPACE: dict[int, tuple[str, object]] = _build_token_space()
N_ACTIONS = len(TOKEN_SPACE)
KIND_IDS: dict[str, list[int]] = {}
for _tid, (_tkind, _tval) in TOKEN_SPACE.items():
    KIND_IDS.setdefault(_tkind, []).append(_tid)
OP_IDS = KIND_IDS["op"]
FIELD_IDS = KIND_IDS["field"]
CONST_IDS = KIND_IDS["const"]


def token_count() -> int:
    return N_ACTIONS


def op_name_win(tid: int) -> tuple[str, Optional[int]]:
    return TOKEN_SPACE[tid][1]          # ("Mean", 20) 或 ("Abs", None)


def op_is_rolling(name: str) -> bool:
    return name not in _NON_WINDOW_REPORT


def op_arity(name: str) -> int:
    reg = op_registry()
    return reg[REPORT_OPERATORS[name]].arity


def _fmt_const(c: float) -> str:
    return str(int(c)) if float(c).is_integer() else repr(round(c, 6))


# ---------------------------------------------------------------------------
# RPN 后缀解析：token 序列 -> 项目前缀公式字符串（GP 风格，兼容 formula.py）
# ---------------------------------------------------------------------------

class RPNParser:
    """后缀 RPN 求值。

    - field / const → 压栈（操作数）。
    - op → 弹出 ``arity`` 个操作数；滚动算子窗口内建在 token 定义中。
    - 结束时栈内必须有且仅有一个完整表达式。
    """

    def __init__(self):
        self.reg = op_registry()

    def parse(self, tokens: list[int]) -> Optional[str]:
        if len(tokens) < 3 or tokens[0] != BEG or tokens[-1] != SEP:
            return None
        stack: list[str] = []
        for tid in tokens[1:-1]:
            kind, value = TOKEN_SPACE[tid]
            if kind == "field":
                stack.append(str(value))
            elif kind == "const":
                stack.append(_fmt_const(float(value)))
            elif kind == "op":
                name, win = value
                reg_name = REPORT_OPERATORS[name]
                arity = op_arity(name)
                if len(stack) < arity:
                    return None
                args = stack[-arity:]           # 栈底→栈顶 = 左→右
                del stack[-arity:]
                head = f"{reg_name}_{win}" if win is not None else reg_name
                stack.append(f"{head}({','.join(args)})")
            else:
                return None
        if len(stack) != 1:
            return None
        return stack[0]


def _virtual_stack_depth(tokens: list[int]) -> int:
    """轻量模拟：计算当前 token 序列（不含 BEG/SEP）后缀求值的栈深。

    field/const 压栈 +1；算子弹 arity 压 1。若弹栈不足返回 -1（已非法）。
    """
    depth = 0
    for tid in tokens:
        if tid in (BEG, SEP):
            continue
        kind, value = TOKEN_SPACE[tid]
        if kind == "field" or kind == "const":
            depth += 1
        elif kind == "op":
            name, _ = value
            depth -= op_arity(name)
            if depth < 0:
                return -1                       # 已非法
            depth += 1
    return depth


def legal_mask_for_tokens(tokens: list[int]) -> np.ndarray:
    """返回 bool 掩码（length=N_ACTIONS），表示下一步可选的 token。

    精确 RPN 栈掩码（研报 MaskablePPO 掩码思路——动态忽略非法动作）：
    - field / const：始终可选（压栈），除非达长度上限。
    - op：需要栈深 >= arity（否则弹栈不足，非法）。
    - SEP：**仅当栈深 == 1**（栈内恰一个完整表达式）才可终止。
    - 长度 >= MAX_EXPR_LENGTH-2 时若栈深==1 强制只能 SEP；栈不完整则允许 SEP
      收尾（超长给 0 奖励，研报「超出长度限制」）。
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    body = [t for t in tokens if t not in (BEG, SEP)]
    depth = _virtual_stack_depth(body)
    if depth < 0:
        return mask                             # 已非法，无合法动作

    remaining = (MAX_EXPR_LENGTH - 2) - len(body)   # 还能放几个 body token
    if remaining <= 0:
        mask[SEP] = True                        # 必须收尾（超长给 0）
        return mask

    mask[FIELD_IDS] = True
    mask[CONST_IDS] = True
    for op_id in OP_IDS:
        name, _ = op_name_win(op_id)
        if depth >= op_arity(name):
            mask[op_id] = True
    if depth == 1:
        mask[SEP] = True
    return mask


# ---------------------------------------------------------------------------
# 单因子 / 组合 IC 评估（MSE Pool 用 IC，MeanStd Pool 用 ICIR）
# ---------------------------------------------------------------------------

def _ic_np(f: np.ndarray, r: np.ndarray, min_cnt: int = 5) -> np.ndarray:
    """纯 numpy 逐日截面 Pearson IC（全向量化，无 Python 循环、无 np.nan*）。

    f/r: (T, N) float64 数组（含 NaN）。
    Returns: (T,) IC 数组，无效日为 NaN。
    """
    mask = ~(np.isnan(f) | np.isnan(r))
    cnt = mask.sum(axis=1)
    valid = cnt >= min_cnt
    T = f.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        cnt_safe = np.maximum(cnt, 1)[:, None]
        fa = np.where(mask, f, 0.0)
        ra = np.where(mask, r, 0.0)
        fa_mean = fa.sum(axis=1, keepdims=True) / cnt_safe
        ra_mean = ra.sum(axis=1, keepdims=True) / cnt_safe
        fa_c = np.where(mask, fa - fa_mean, 0.0)
        ra_c = np.where(mask, ra - ra_mean, 0.0)
        num = (fa_c * ra_c).sum(axis=1)
        den = np.sqrt((fa_c * fa_c).sum(axis=1)
                      * (ra_c * ra_c).sum(axis=1))
        ic = np.full(T, np.nan)
        np.divide(num, den, out=ic, where=den != 0)
    ic = np.where(valid, ic, np.nan)
    return ic


def panel_ic_series(factor: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """逐日截面 Pearson IC（因子 t 对齐收益面板同日起始；纯 numpy 安全实现）。"""
    f = factor.to_numpy(dtype=np.float64)
    r = rets.reindex_like(factor).to_numpy(dtype=np.float64)
    ic = _ic_np(f, r)
    return pd.Series(ic, index=factor.index)


def eval_formula(panel: dict[str, pd.DataFrame], rets: pd.DataFrame,
                 formula: str, features: list[str]) -> tuple[Optional[pd.DataFrame], float]:
    """求值公式并计算单因子 |IC| 均值。

    Returns:
        (factor_panel, mean_abs_ic)。factor_panel=None 表示求值异常（无效）。
        mean_abs_ic=0 表示空值/异常（面板全 NaN）。
    """
    from factor.formula import formula_builder
    try:
        fp = formula_builder(formula, features=features)(panel)
    except Exception:
        return None, 0.0
    if fp is None or fp.empty:
        return None, 0.0
    # Greater/Less 返回 bool 面板 → 转 float（研报图表10 二元逻辑算子）
    if fp.dtypes.eq("bool").any():
        fp = fp.astype(float)
    ic = panel_ic_series(fp, rets)
    if ic.isna().all():
        return fp, 0.0
    return fp, float(ic.abs().mean())


def _panels_to_np(factors: list[pd.DataFrame]) -> np.ndarray:
    """因子面板列表 -> (T, N, K) numpy 数组（含 NaN）。

    显式对齐 index/columns（formula_builder 各算子可能返回不同列序），
    避免 np.stack 在形状/顺序不一致时的 C 层问题。
    """
    if not factors:
        return np.zeros((0, 0, 0))
    cols = list(factors[0].columns)
    idx = factors[0].index
    arrs = []
    for p in factors:
        q = p.reindex(index=idx, columns=cols)
        arrs.append(np.ascontiguousarray(q.to_numpy(dtype=np.float64)))
    return np.stack(arrs, axis=2)


def _standardize_panel(a: np.ndarray) -> np.ndarray:
    """逐行截面标准化（NaN 保留；单值行置 0）。"""
    m = ~np.isnan(a)
    cnt = m.sum(axis=1, keepdims=True)
    mu = np.where(m, a, 0.0).sum(axis=1, keepdims=True) / np.maximum(cnt, 1)
    c = np.where(m, a - mu, 0.0)
    sd = np.sqrt((c * c).sum(axis=1, keepdims=True) / np.maximum(cnt, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(m, c / np.where(sd == 0, np.nan, sd), 0.0)
    z = np.where(np.isnan(z), 0.0, z)
    return z


def portfolio_ic_np(panels_arr: np.ndarray, weights: np.ndarray,
                    rets_arr: np.ndarray) -> float:
    """组合 IC（纯 numpy）：因子先逐日截面标准化，再加权组合，与收益算 IC。"""
    w = np.asarray(weights, dtype=float)
    w = w / (np.abs(w).sum() + 1e-12)
    T, N, K = panels_arr.shape
    f0 = np.zeros((T, N), dtype=np.float64)
    for k in range(K):
        f0 += w[k] * _standardize_panel(panels_arr[:, :, k])
    ic = _ic_np(f0, rets_arr)
    valid = ic[np.isfinite(ic)]
    m = float(valid.mean()) if len(valid) else 0.0
    return m if np.isfinite(m) else 0.0


def portfolio_icir_np(panels_arr: np.ndarray, weights: np.ndarray,
                      rets_arr: np.ndarray) -> float:
    """组合 ICIR（纯 numpy）。"""
    w = np.asarray(weights, dtype=float)
    w = w / (np.abs(w).sum() + 1e-12)
    T, N, K = panels_arr.shape
    f0 = np.zeros((T, N), dtype=np.float64)
    for k in range(K):
        f0 += w[k] * _standardize_panel(panels_arr[:, :, k])
    ic = _ic_np(f0, rets_arr)
    ic = ic[np.isfinite(ic)]
    if len(ic) < 2:
        return 0.0
    return float(ic.mean() / (ic.std() + 1e-12))


def portfolio_ic(factors: list[pd.DataFrame], weights: np.ndarray,
                 rets: pd.DataFrame) -> float:
    """组合 IC：加权因子面板（逐日截面标准化后线性组合）与收益的 IC 均值。"""
    if not factors:
        return 0.0
    arr = _panels_to_np(factors)
    r = rets.reindex_like(factors[0]).to_numpy(dtype=np.float64)
    return portfolio_ic_np(arr, weights, r)


def portfolio_icir(factors: list[pd.DataFrame], weights: np.ndarray,
                   rets: pd.DataFrame) -> float:
    """组合 ICIR：IC 均值 / IC 标准差（MeanStd Pool 的目标函数）。"""
    if not factors:
        return 0.0
    arr = _panels_to_np(factors)
    r = rets.reindex_like(factors[0]).to_numpy(dtype=np.float64)
    return portfolio_icir_np(arr, weights, r)


# ---------------------------------------------------------------------------
# AlphaPool：因子池 + 权重优化 + 4 档奖励
# ---------------------------------------------------------------------------

class AlphaPool:
    """研报 AlphaPool 环境核心（P0：MSE Pool，metric="ic"）。

    入池规则（对齐研报 + 低相关约束）：
    - 因子与池内全部因子逐日截面相关均值 <= corr_threshold（默认 0.7）方可入池；
    - 池未满直接入；池满则替换单因子 |IC| 最差者（池容量 capacity=10）。
    权重优化：Adam 梯度上升最大化组合 IC（研报 lr=1e-3, max_steps=2000, tol=200）。

    奖励（研报图表 11）：
    - invalid（求值异常）→ -1
    - empty（空值/全 NaN）→ 0
    - fail_cache 命中 / 无法入池 → best_obj
    - 成功入池 → new_obj（组合最新 IC）
    """

    def __init__(self, panel: dict[str, pd.DataFrame], rets: pd.DataFrame,
                 features: list[str], capacity: int = 10,
                 metric: str = "ic", corr_threshold: float = 0.7,
                 lr: float = 1e-3, max_steps: int = 2000,
                 tolerance: int = 200, seed: int = 0):
        self.panel = panel
        self.rets = rets
        self.features = list(features)
        self.capacity = capacity
        self.metric = metric.lower()
        self.corr_threshold = corr_threshold
        self.lr = lr
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.rng = np.random.default_rng(seed)

        self.formulas: list[str] = []           # 池内公式
        self.factor_panels: list[pd.DataFrame] = []
        self.weights: np.ndarray = np.array([], dtype=float)
        # 池最优组合评估指标。研报由 LLM 初始池保证非零；P0 空池起始 0.0，
        # 避免 -inf 毒化奖励（所有 no_pool 都返回 0 而非负无穷）
        self.best_obj = 0.0
        self.fail_cache: set[str] = set()       # 失败缓存（无效公式）
        self.empty_cache: set[str] = set()      # 空值/异常公式
        # 已评估单因子 |IC| 缓存（避免重复求值，训练提速）
        self._ic_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 目标函数（纯 numpy 路径，规避 pandas 热循环 C 层问题）
    # ------------------------------------------------------------------
    def _panels_arr(self) -> np.ndarray:
        """池内因子面板 -> (T, N, K) numpy 数组。"""
        return _panels_to_np(self.factor_panels)

    def _objective_np(self, panels_arr: np.ndarray, w: np.ndarray) -> float:
        r = self.rets.reindex_like(self.factor_panels[0]).to_numpy(dtype=np.float64)
        if self.metric == "ic":
            return portfolio_ic_np(panels_arr, w, r)
        return portfolio_icir_np(panels_arr, w, r)

    def _objective(self, panels: list[pd.DataFrame], w: np.ndarray) -> float:
        if self.metric == "ic":
            return portfolio_ic(panels, w, self.rets)
        return portfolio_icir(panels, w, self.rets)

    def optimize_weights(self, panels: list[pd.DataFrame]) -> np.ndarray:
        """组合权重：均值-方差闭式解 w ∝ Σ⁻¹μ（最大化组合 IC 的解析解）。

        研报 MSE Pool 用 Adam 迭代优化组合 IC；本项目用闭式解替代（无循环、
        无数值梯度，规避 Windows numpy 偶发 C 层崩溃且更快）：
          - μ_k = 因子 k 逐日 IC 均值；
          - Σ   = 因子逐日 IC 协方差（Ledoit 收缩 50% 到对角，保证可逆）；
          - w   = Σ⁻¹μ，L2 归一化。
        这是「最大化组合 IC」的解析最优（在逐日 IC 高斯近似下），等价于
        研报 Adam 的收敛目标。
        """
        n = len(panels)
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.array([1.0], dtype=float)
        # 逐日 IC 矩阵 (T, K)
        r = self.rets.reindex_like(panels[0]).to_numpy(dtype=np.float64)
        ics = []
        for p in panels:
            ic = _ic_np(p.to_numpy(dtype=np.float64), r)
            ics.append(ic)
        ic_mat = np.stack(ics, axis=1)              # (T, K)
        # 只保留所有因子都有有效 IC 的行
        valid = np.isfinite(ic_mat).all(axis=1)
        if valid.sum() < 5:
            return np.ones(n, dtype=float) / n
        X = ic_mat[valid]
        mu = X.mean(axis=0)
        Xc = X - mu
        cov = (Xc.T @ Xc) / (len(X) - 1)
        # Ledoit 收缩：0.5 * cov + 0.5 * trace/cov 对角（保证正定）
        lam = 0.5
        shrink = cov * (1 - lam) + lam * np.eye(n) * np.trace(cov) / n
        try:
            inv = np.linalg.inv(shrink)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(shrink)
        w = inv @ mu
        denom = np.sqrt((w * w).sum()) + 1e-12
        w = w / denom
        if not np.isfinite(w).all():
            return np.ones(n, dtype=float) / n
        return w

    # ------------------------------------------------------------------
    # 入池判定
    # ------------------------------------------------------------------
    def _cross_section_corr(self, a: pd.DataFrame, b: pd.DataFrame) -> float:
        """两因子逐日截面 rank 相关均值（去 NaN；纯 numpy）。"""
        aa = np.ascontiguousarray(a.to_numpy(dtype=float))
        bb = np.ascontiguousarray(b.to_numpy(dtype=float))
        if aa.shape != bb.shape:
            return 0.0
        mask = ~(np.isnan(aa) | np.isnan(bb))
        cnt = mask.sum(axis=1)
        corrs = []
        for t in range(len(aa)):
            if cnt[t] < 5:
                continue
            fa_t = np.where(mask[t], aa[t], 0.0)   # rank 前先填 0（掩码剔除）
            fb_t = np.where(mask[t], bb[t], 0.0)
            ra = np.argsort(np.argsort(fa_t)).astype(float)
            rb = np.argsort(np.argsort(fb_t)).astype(float)
            # 只保留有效位置的 rank（无效位置填 0 后中心化；无 nan 运算）
            ra = np.where(mask[t], ra, 0.0)
            rb = np.where(mask[t], rb, 0.0)
            mcnt = mask[t].sum()
            ra = ra - ra.sum() / mcnt
            rb = rb - rb.sum() / mcnt
            ra = np.where(mask[t], ra, 0.0)
            rb = np.where(mask[t], rb, 0.0)
            den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
            if den == 0:
                continue
            corrs.append(float((ra * rb).sum() / den))
        return float(np.mean(corrs)) if corrs else 0.0

    def _corr_with_pool(self, fp: pd.DataFrame) -> list[float]:
        return [self._cross_section_corr(fp, p) for p in self.factor_panels]

    def _factor_ic(self, formula: str) -> Optional[float]:
        """单因子 |IC|（带缓存）。"""
        if formula in self._ic_cache:
            return self._ic_cache[formula]
        fp, mean_abs_ic = eval_formula(self.panel, self.rets, formula,
                                       self.features)
        self._ic_cache[formula] = mean_abs_ic
        return mean_abs_ic

    def _add_factor(self, formula: str, fp: pd.DataFrame) -> float:
        """入池并重优化权重，返回新组合评估指标 new_obj。"""
        if len(self.formulas) >= self.capacity:
            ic_scores = [self._factor_ic(f) for f in self.formulas]
            worst_idx = int(np.argmin(ic_scores))
            self.formulas.pop(worst_idx)
            self.factor_panels.pop(worst_idx)
        self.formulas.append(formula)
        self.factor_panels.append(fp)
        self.weights = self.optimize_weights(self.factor_panels)
        new_obj = abs(self._objective(self.factor_panels, self.weights))
        self.best_obj = max(self.best_obj, new_obj)
        return new_obj

    # ------------------------------------------------------------------
    # 外部评估接口：返回 (状态, 奖励)
    # ------------------------------------------------------------------
    def evaluate(self, formula: str, fp: Optional[pd.DataFrame] = None
                 ) -> tuple[str, float]:
        """评估一个因子，返回 (status, reward)。

        status ∈ {"invalid", "empty", "fail_cache", "no_pool", "pooled"}。
        """
        if formula in self.fail_cache:
            return "fail_cache", self.best_obj
        if formula in self.empty_cache:
            return "empty", 0.0

        if fp is None:
            fp, mean_abs_ic = eval_formula(self.panel, self.rets, formula,
                                           self.features)
        else:
            mean_abs_ic = self._factor_ic(formula)
        if fp is None:
            self.fail_cache.add(formula)
            return "invalid", -1.0
        if mean_abs_ic <= 1e-9:
            self.empty_cache.add(formula)
            return "empty", 0.0

        corrs = self._corr_with_pool(fp)
        if any(c > self.corr_threshold for c in corrs):
            return "no_pool", self.best_obj
        if len(self.formulas) >= self.capacity:
            ic_scores = [self._factor_ic(f) for f in self.formulas]
            if mean_abs_ic <= min(ic_scores):
                return "no_pool", self.best_obj
        new_obj = self._add_factor(formula, fp)
        return "pooled", new_obj

    def stats(self) -> dict:
        return {
            "n_factors": len(self.formulas),
            "best_obj": self.best_obj,
            "formulas": list(self.formulas),
            "weights": list(self.weights),
            "fail_cache_size": len(self.fail_cache),
        }
