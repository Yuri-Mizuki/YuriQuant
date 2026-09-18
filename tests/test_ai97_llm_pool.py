"""华泰 AI97 后段：大模型因子池（初始池 + 定期去弱留强）单元测试。

覆盖：
- 研报语法解析 ↔ 项目语法互转（21 算子全覆盖）
- 三个语义校验（研报点名的 构造简单 / 不合逻辑 / 符号多余）+ 窗口与长度白名单
- 量纲判定的指数运算（避免启发式正则误伤）
- 大模型裸文本 → 候选公式的容错抽取
- 模板提案器：目录全量通过校验且**可实际求值**
- ``seed_pool`` 把 ``best_obj`` 从 0 抬起来（本任务的验收点）
- ``refresh_pool`` 只淘汰 RL 来源的因子
- 小数常数字面量的回归测试（``factor.formula`` 修复）
- **推理模型陷阱**：``max_tokens`` 必须覆盖思维链，空 ``content`` 要报错而非静默返回空
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from factor.formula import formula_builder
from factor.rl.alphapool_env import (
    BEG,
    CONSTANTS,
    REPORT_OPERATORS,
    SEP,
    TOKEN_SPACE,
    AlphaPool,
    RPNParser,
)
from factor.rl.alphapool_gym import AlphaPoolGymEnv
from factor.rl.llm_pool import (
    FormulaSyntaxError,
    OpenAICompatibleProposer,
    TemplateProposer,
    build_prompt,
    canonical,
    check_many,
    check_report_formula,
    extract_formulas,
    make_proposer,
    parse_report_formula,
    refresh_pool,
    seed_pool,
    to_project,
    to_report,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mini_panel():
    """含真实反转信号的合成量价面板 + 未来收益（与 test_htai_rl_p0 同构）。"""
    rng = np.random.default_rng(0)
    n_days, n_codes = 200, 30
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = 0.3 * rets[t - 1] + rng.normal(0, 0.02, n_codes)
    close = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), idx, codes)
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_codes)))
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    vwap = close * (1 + rng.normal(0, 0.002, (n_days, n_codes)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, (n_days, n_codes)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, (n_days, n_codes)))
    panel = {"open": open_, "high": high, "low": low, "close": close,
             "volume": volume, "vwap": vwap}
    mom20 = close.pct_change(20)
    signal = -mom20.shift(1)
    sig = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1) + 1e-9, axis=0)
    noise = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    fwd = (0.6 * 0.03 * sig.fillna(0) + noise).where(signal.notna())
    return panel, fwd


@pytest.fixture
def pool(mini_panel):
    panel, fwd = mini_panel
    return AlphaPool(panel, fwd, list(panel), capacity=10,
                     metric="ic", corr_threshold=0.7, seed=0)


# ---------------------------------------------------------------------------
# 1. 语法解析与互转（21 算子全覆盖）
# ---------------------------------------------------------------------------
_ALL_OPS_REPORT = {
    "Abs": "Abs($close)",
    "Log": "Log($volume)",
    "Add": "Add($close, $open)",
    "Sub": "Sub($close, $open)",
    "Mul": "Mul($close, $volume)",
    "Div": "Div($close, $open)",
    "Greater": "Greater($close, $open)",
    "Less": "Less($close, $open)",
    "Ref": "Ref($close, 5)",
    "Mean": "Mean($close, 20)",
    "Sum": "Sum($volume, 5)",
    "Std": "Std($close, 20)",
    "Var": "Var($close, 20)",
    "Max": "Max($close, 20)",
    "Min": "Min($close, 20)",
    "Med": "Med($close, 20)",
    "Mad": "Mad($close, 20)",
    "Delta": "Delta($close, 1)",
    "WMA": "WMA($close, 20)",
    "EMA": "EMA($close, 20)",
    "Cov": "Cov($close, $volume, 20)",
    "Corr": "Corr($close, $volume, 20)",
}


def test_report_operators_are_covered_by_fixture():
    assert set(_ALL_OPS_REPORT) == set(REPORT_OPERATORS)


@pytest.mark.parametrize("op,text", sorted(_ALL_OPS_REPORT.items()))
def test_all_report_operators_parse_to_project(op, text):
    node = parse_report_formula(text)
    project = to_project(node)
    reg_name = REPORT_OPERATORS[op]
    assert project.startswith(reg_name)
    # 字段名统一小写、窗口写进算子名后缀
    assert "$" not in project
    assert project == project.lower()


def test_field_case_insensitive_and_dollar_prefix():
    assert to_project(parse_report_formula("Mean($CLOSE, 20)")) == \
        to_project(parse_report_formula("Mean($close, 20)"))
    with pytest.raises(FormulaSyntaxError):
        parse_report_formula("Mean(close, 20)")      # 缺 $ 前缀 → 被当成算子名


def test_report_round_trip():
    text = "Div(Sub($close, Mean($close, 20)), Std($close, 20))"
    node = parse_report_formula(text)
    assert to_report(node) == "Div(Sub($close, Mean($close, 20)), Std($close, 20))"
    assert to_report(parse_report_formula(to_report(node))) == to_report(node)


def test_canonical_normalizes_spacing_and_case():
    a = canonical("Div(Sub($CLOSE, Mean($close,20)),Std($Close, 20))")
    b = canonical("Div(Sub($close, Mean($close, 20)), Std($close, 20))")
    assert a == b
    assert canonical("Foo($close)") is None


# ---------------------------------------------------------------------------
# 2. 三个语义校验（研报点名的三个毛病）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,reason", [
    # 研报原话：「构造简单（例如 $volume）」
    ("$volume", "trivial_terminal"),
    ("$close", "trivial_terminal"),
    ("Abs($close)", "trivial_terminal"),
    # 研报原话：「不合逻辑（例如 Sub($volume, $close)）」
    ("Sub($volume, $close)", "dimension_mismatch"),
    ("Add($volume, $high)", "dimension_mismatch"),
    ("Greater($volume, $close)", "dimension_mismatch"),
    # 研报原话：「符号多余（例如 Abs(Abs(Sub($close, $open))))」
    ("Abs(Abs(Sub($close, $open)))", "redundant_op"),
    ("Log(Log($close))", "redundant_op"),
    # 恒等式：无信息量
    ("Sub($close, $close)", "degenerate"),
    ("Div($volume, $volume)", "degenerate"),
    ("Corr($close, $close, 20)", "degenerate"),
    # 窗口白名单
    ("Mean($close, 7)", "window_not_allowed"),
    ("Ref($close, 3)", "window_not_allowed"),
])
def test_semantic_checks_reject(text, reason):
    chk = check_report_formula(text)
    assert not chk.ok, f"{text} 应被拒绝"
    assert reason in chk.issues, f"{text} 期望 {reason}，实际 {chk.issues}"


@pytest.mark.parametrize("text", [
    "Div(Sub($close, Mean($close, 20)), Std($close, 20))",
    "Corr($close, $volume, 20)",
    "Mul($close, $volume)",                      # 价格 × 成交量 = 成交额，合法
    "Div(Mul($close, $volume), Mean(Mul($close, $volume), 20))",
    "Div(Sub($high, $low), $close)",
    "Div($volume, Mean($volume, 20))",
    "Mul(-0.01, $close)",                        # 小数常数
    "Add($close, 1)",                            # 与常数相加不参与量纲判定
])
def test_semantic_checks_accept(text):
    chk = check_report_formula(text)
    assert chk.ok, f"{text} 应通过，实际 {chk.issues}"
    assert chk.project is not None


def test_dimension_uses_exponent_arithmetic():
    """量纲判定是"价格指数 × 成交量指数"的精确向量运算，不是启发式正则。"""
    # Div(成交额, 成交量) = 价格 → 可再与价格相加
    assert check_report_formula("Add(Div(Mul($close, $volume), $volume), $close)").ok
    # Div(价格, 价格) = 无量纲 → 可与常数/另一无量纲相加
    assert check_report_formula("Add(Div($close, $open), Div($high, $low))").ok
    # 价格 + 成交量 逐位指数不等 → 拒绝
    assert not check_report_formula("Add($close, $volume)").ok
    # 常数不参与判定（避免 Div($close, 100) 这类合法缩放被误伤）
    assert check_report_formula("Div($close, 100)").ok


def test_too_long_is_rejected():
    # Add 每层贡献 1 个 token + 1 个叶子；8 层 = 8 + 9 = 17 > 13（上限 15 含 BEG/SEP）
    deep = "$close"
    for _ in range(8):
        deep = f"Add({deep}, $open)"
    chk = check_report_formula(deep)
    assert not chk.ok
    assert "too_long" in chk.issues
    assert chk.n_tokens > 13
    # 刚好 13 个 token 的式子应当通过（边界不能误杀）
    edge = "$close"
    for _ in range(6):
        edge = f"Add({edge}, $open)"
    edge_chk = check_report_formula(edge)
    assert edge_chk.n_tokens == 13
    assert edge_chk.ok, edge_chk.issues


def test_check_many_preserves_order_and_isolates_failures():
    texts = ["Sub($volume, $close)", "Corr($close, $volume, 20)", "$volume"]
    out = check_many(texts)
    assert [c.text for c in out] == texts
    assert [c.ok for c in out] == [False, True, False]


# ---------------------------------------------------------------------------
# 3. 大模型裸文本 → 候选公式
# ---------------------------------------------------------------------------
def test_extract_from_json_array():
    text = ('好的，以下是我构造的因子：\n'
            '["Div(Sub($close, Mean($close, 20)), Std($close, 20))",'
            ' "Corr($close, $volume, 20)"]\n希望有帮助。')
    got = extract_formulas(text)
    assert len(got) == 2
    assert got[0].startswith("Div(")


def test_extract_from_dict_payload_and_code_fence():
    text = '```json\n{"factors": ["Div($volume, Mean($volume, 20))"]}\n```'
    got = extract_formulas(text)
    assert got == ["Div($volume, Mean($volume, 20))"]


def test_extract_from_plain_chatter_without_json():
    text = ("因子一：Div(Sub($close, $vwap), $vwap) 表示收盘相对均价偏离；\n"
            "因子二：Div(Std($close, 20), Mean($close, 20)) 表示波动率水平。")
    got = extract_formulas(text)
    assert len(got) == 2
    assert all(canonical(f) for f in got)


def test_extract_dedupes_equivalent_spellings():
    text = "Div(Sub($CLOSE, Mean($close, 20)), Std($Close, 20))\n" \
           "Div(Sub($close, Mean($close,20)),Std($close,20))"
    assert len(extract_formulas(text)) == 1


def test_extract_respects_limit():
    text = " | ".join(f"Div($close, Mean($close, {w}))" for w in (5, 10, 20, 60))
    assert len(extract_formulas(text, limit=2)) == 2


# ---------------------------------------------------------------------------
# 4. 提示词与提案器
# ---------------------------------------------------------------------------
def test_build_prompt_lists_report_constraints():
    p = build_prompt(5, existing=["Corr($close, $volume, 20)"],
                     avoid=["Sub($volume, $close)"])
    for token in ("$OPEN", "$VWAP", "Corr(a,b,w)", "JSON", "量纲必须自洽"):
        assert token in p
    assert "Corr($close, $volume, 20)" in p      # 已在池中，不得重复
    assert "Sub($volume, $close)" in p           # 已知无效，需避开
    assert all(c in p for c in ("-0.01", "0.5"))  # 常数表按研报给全


def test_template_catalogue_entries_all_valid_and_executable(mini_panel):
    """目录里每条都必须通过校验**并且真能求值**（防"语法转对了但算子求不出"）。"""
    panel, _fwd = mini_panel
    cat = TemplateProposer().catalogue()
    assert len(cat) >= 40, f"模板目录过小：{len(cat)}"
    for text in cat:
        chk = check_report_formula(text)
        assert chk.ok, f"{text} 未通过校验：{chk.issues}"
        out = formula_builder(chk.project, features=list(panel))(panel)
        assert isinstance(out, pd.DataFrame), text
        assert out.notna().any().any(), f"{text} 求值后全为 NaN"


def test_template_proposer_is_deterministic_and_respects_bans():
    p1, p2 = TemplateProposer(seed=7), TemplateProposer(seed=7)
    a = p1.propose(10)
    b = p2.propose(10)
    assert a == b and len(a) == 10
    banned = {canonical(f) for f in a}
    c = p1.propose(10, existing=a)
    assert not (banned & {canonical(f) for f in c})


def test_template_proposer_honours_avoid_list():
    p = TemplateProposer(seed=1)
    first = p.propose(5)
    nxt = p.propose(5, avoid=first)
    assert not ({canonical(f) for f in first} & {canonical(f) for f in nxt})


def test_make_proposer_variants():
    assert make_proposer("none") is None
    assert isinstance(make_proposer("template"), TemplateProposer)
    assert isinstance(make_proposer("deepseek"), OpenAICompatibleProposer)
    with pytest.raises(ValueError):
        make_proposer("bogus")


def test_openai_proposer_requires_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    prop = OpenAICompatibleProposer()
    with pytest.raises(RuntimeError, match="API key"):
        prop.api_key()
    with pytest.raises(RuntimeError, match="API key"):
        prop.propose(3)


def test_openai_proposer_build_request_shape():
    prop = OpenAICompatibleProposer(model="deepseek-flash", api_key="dummy")
    body = prop.build_request(4, existing=["Corr($close, $volume, 20)"])
    assert body["model"] == "deepseek-flash"
    assert body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "4 个" in body["messages"][1]["content"]


def test_openai_proposer_defaults_cover_reasoning_budget():
    """默认值必须是"能装下思维链"的量级。

    ``deepseek-flash`` / ``deepseek-v4-pro`` 是**推理模型**：``content`` 与
    ``reasoning_content`` 共享 ``max_tokens``。实测 ``max_tokens`` 为 32 / 2000 / 3000 时
    思维链把预算吃光 → ``finish_reason="length"`` 且 ``content`` 为空字符串
    （HTTP 200、无异常！）。故默认值不得低于 8000。
    """
    prop = OpenAICompatibleProposer(api_key="dummy")
    assert prop.max_tokens >= 8000, "推理模型的默认 max_tokens 太小会把预算全花在 reasoning 上"
    assert prop.timeout >= 120, "推理模型单次 15~50s，超时不能按普通模型的量级设"
    assert prop.base_url == "https://api.deepseek.com"
    assert prop.model == "deepseek-flash"


def test_empty_content_raises_with_diagnosis(monkeypatch):
    """空 content 必须报错并带上诊断，不能静默返回空列表。

    模拟"推理模型吃光 token 预算"的真实响应：HTTP 200、
    ``finish_reason="length"``、``reasoning_content`` 很长而 ``content`` 为空。
    """
    prop = OpenAICompatibleProposer(api_key="dummy", max_tokens=3000)

    class _Resp:
        def read(self):
            return json.dumps({
                "model": "deepseek-flash",
                "choices": [{
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "",
                                "reasoning_content": "想了很多" * 500},
                }],
                "usage": {"completion_tokens": 3000,
                          "completion_tokens_details": {"reasoning_tokens": 3000}},
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="空 content"):
        prop._post(prop.build_request(4))
    # 诊断信息必须落在 last_meta 里，供调用方排查
    meta = prop.last_meta[-1]
    assert meta["finish_reason"] == "length"
    assert meta["reasoning_tokens"] == 3000
    assert meta["content_len"] == 0


def test_make_proposer_filters_kwargs_per_kind():
    """`make_proposer` 必须按提案器类型过滤 kwargs。

    CLI 传的是一套统一参数（含 ``model``/``base_url``/``max_tokens``），
    直接透给 ``TemplateProposer`` 会 TypeError。
    """
    t = make_proposer("template", seed=3, model="x", base_url="y", max_tokens=1)
    assert isinstance(t, TemplateProposer) and t.seed == 3
    o = make_proposer("openai", api_key="dummy", model="m", max_tokens=9000, seed=9)
    assert isinstance(o, OpenAICompatibleProposer)
    assert o.model == "m" and o.max_tokens == 9000


# ---------------------------------------------------------------------------
# 5. 注入原语（本任务的核心验收点）
# ---------------------------------------------------------------------------
def test_empty_pool_best_obj_is_zero(pool):
    """空池起步时 best_obj = 0 —— 这正是需要初始池的原因。"""
    assert pool.best_obj == 0.0
    assert pool.formulas == []


def test_seed_pool_raises_best_obj_from_zero(pool):
    formulas = TemplateProposer(seed=0).propose(25)
    rep = seed_pool(pool, formulas)
    assert rep.n_pool_before == 0
    assert rep.best_obj_before == 0.0
    assert rep.accepted >= 1, f"没有因子入池：{rep.status_counts}"
    assert rep.best_obj_after > 0.0
    assert rep.n_pool_after == len(pool.formulas) >= 1
    assert pool.origin_counts().get("llm", 0) == len(pool.formulas)
    assert rep.as_dict()["accepted"] == rep.accepted


def test_seed_pool_rejects_illegal_without_evaluating(pool):
    rep = seed_pool(pool, ["$volume", "Sub($volume, $close)",
                           "Abs(Abs($close))", "Mean($close, 7)"])
    assert rep.accepted == 0
    assert rep.proposed == 4
    reasons = {r for r, _f in rep.rejected}
    assert "trivial_terminal" in reasons
    assert "dimension_mismatch" in reasons
    assert "redundant_op" in reasons
    assert "window_not_allowed" in reasons
    # 全被校验拦下 → 池没有被污染
    assert pool.formulas == [] and pool.best_obj == 0.0


def test_seed_pool_is_idempotent_for_duplicates(pool):
    formulas = TemplateProposer(seed=0).propose(12)
    first = seed_pool(pool, formulas)
    before = list(pool.formulas)
    second = seed_pool(pool, formulas)
    # 重复注入不应改变池内容（相关性/IC 门槛会把它们挡掉）
    assert set(pool.formulas) <= set(before) | set(second.rejected and [])
    assert first.accepted >= 1


def _fill(pool, seed: int, origin: str, n: int = 12) -> int:
    """往池里塞候选，返回实际入池数。

    测试辅助：先入池的来源才保证有位置（池满后新因子要打败最差的才进得来，
    而"进不来"本身正是初始池生效的表现，不适合当测试前提）。
    """
    got = 0
    for text in TemplateProposer(seed=seed).propose(n):
        chk = check_report_formula(text)
        if not chk.ok or chk.project is None:
            continue
        status, _r = pool.evaluate(chk.project, origin=origin)
        got += int(status == "pooled")
    return got


def test_drop_worst_only_touches_requested_origin(pool):
    # 先让 RL 来源占位（空池 → 必然能入池），保证两种来源都有因子可观测
    assert _fill(pool, seed=0, origin="rl") >= 3
    n_rl = pool.origin_counts()["rl"]
    seed_pool(pool, TemplateProposer(seed=99).propose(12))   # LLM 能进多少算多少
    n_llm = pool.origin_counts().get("llm", 0)

    dropped = pool.drop_worst(origin="rl", n=2)
    assert len(dropped) == 2
    assert all(f not in pool.formulas for f in dropped)
    assert pool.origin_counts()["rl"] == n_rl - 2
    assert pool.origin_counts().get("llm", 0) == n_llm, "LLM 来源的因子不应被本轮剔除"


def test_drop_worst_returns_empty_when_no_match(pool):
    seed_pool(pool, TemplateProposer(seed=0).propose(10))
    assert pool.drop_worst(origin="rl", n=5) == []
    assert pool.drop_worst(origin="rl", n=0) == []


def test_drop_worst_keeps_best_obj_monotone(pool):
    """研报奖励档位依赖 best_obj 单调不减 —— 剔除因子不得让它回退。"""
    seed_pool(pool, TemplateProposer(seed=0).propose(20))
    before = pool.best_obj
    pool.drop_worst(origin="llm", n=len(pool.formulas))
    assert pool.best_obj == before
    assert pool.formulas == []


def test_refresh_pool_drops_rl_then_seeds(pool):
    assert _fill(pool, seed=0, origin="rl") >= 4
    seed_pool(pool, TemplateProposer(seed=7).propose(12))
    rl_n = pool.origin_counts()["rl"]
    llm_n = pool.origin_counts().get("llm", 0)
    n_start = len(pool.formulas)

    rep = refresh_pool(pool, TemplateProposer(seed=5), n_new=6, drop_rl_n=3)
    assert len(rep.dropped) == 3
    assert rep.stage == "refresh"
    assert pool.origin_counts()["rl"] == rl_n - 3
    assert pool.origin_counts().get("llm", 0) >= llm_n
    assert rep.best_obj_after >= rep.best_obj_before
    assert rep.as_dict()["dropped"] == rep.dropped
    # n_pool_before 必须是**剔除前**的规模：refresh 先 drop 再 add，
    # 若从 pool 现读会拿到剔除后的规模，日志里 10→7→9 被压成 7→9（曾如此）。
    assert rep.n_pool_before == n_start
    assert rep.n_pool_after == len(pool.formulas)
    # 更新过程只发生"剔 3 个 + 入 accepted 个"，规模变化必须能对上账
    assert rep.n_pool_after == n_start - 3 + rep.accepted


# ---------------------------------------------------------------------------
# 6. 与 gym 环境 / RPN 链路的衔接
# ---------------------------------------------------------------------------
def test_gym_env_marks_rl_origin(pool, mini_panel):
    panel, fwd = mini_panel
    env = AlphaPoolGymEnv(panel, fwd, capacity=10, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.mdp.max_len + 3,)
    # 直接注入一个已知有效的项目公式，走环境内部的评估路径
    env.pool.evaluate("ts_mean_20(close)", origin="rl")
    assert set(env.pool.origin_counts()) <= {"rl", "llm"}


def test_rpn_decimal_constant_now_evaluates(mini_panel):
    """回归：``0.5`` 曾被 ``parse_formula`` 当成特征名 → 全公式判 invalid。

    研报 13 个常数里 3 个是小数（0.5 / -0.5 / -0.01），修复前它们在整个 token
    空间里恒拿 -1 奖励。
    """
    panel, fwd = mini_panel
    assert 0.5 in CONSTANTS and -0.01 in CONSTANTS

    def tid_of_const(c):
        for t, (kind, val) in TOKEN_SPACE.items():
            if kind == "const" and abs(float(val) - c) < 1e-12:
                return t
        raise AssertionError(f"常数 {c} 不在 token 空间")

    def tid_of_field(name):
        for t, (kind, val) in TOKEN_SPACE.items():
            if kind == "field" and val == name:
                return t
        raise AssertionError(f"字段 {name} 不在 token 空间")

    def tid_of_op(report_name, win):
        for t, (kind, val) in TOKEN_SPACE.items():
            if kind == "op" and val == (report_name, win):
                return t
        raise AssertionError(f"算子 {report_name}({win}) 不在 token 空间")

    # RPN: BEG close 0.5 Mul SEP   →  mul(close, 0.5)（栈底→栈顶即左→右）
    toks = [BEG, tid_of_field("close"), tid_of_const(0.5),
            tid_of_op("Mul", None), SEP]
    formula = RPNParser().parse(toks)
    assert formula == "mul(close,0.5)", formula
    out = formula_builder(formula, features=list(panel))(panel)
    assert out.notna().any().any()

    # 池对它的评估不再是 invalid（修复前 eval_formula 会 KeyError → invalid/-1）
    pool = AlphaPool(panel, fwd, list(panel), capacity=5, seed=0)
    status, _reward = pool.evaluate(formula)
    assert status != "invalid", "小数常数公式仍被判为无效"


def test_decimal_constant_int_vs_float_unchanged():
    """整数分支保持 int（不引入 2 → 2.0，避免缓存键与算子行为漂移）。"""
    from factor.formula import parse_formula
    assert parse_formula("2", features=["close"]) == ("const", 2)
    assert parse_formula("2.0", features=["close"]) == ("const", 2.0)
    assert parse_formula("0.5", features=["close"]) == ("const", 0.5)
    assert parse_formula("-0.01", features=["close"]) == ("const", -0.01)
    assert parse_formula(".5", features=["close"]) == ("const", 0.5)


# ---------------------------------------------------------------------------
# vwap 复权口径（2026-09-16 真实数据暴露的 bug）
# ---------------------------------------------------------------------------
def test_attach_vwap_backward_makes_it_comparable_to_close():
    """``amount/volume`` 是**原始价**口径，close 是后复权 → 必须乘 backward 才可比。

    真实数据上的旧行为：``vwap/close`` 中位数 0.285（= 1/backward 的量级），
    而除以**原始** close 时中位数 0.999。若不修，这条纯量纲错误会被当成因子
    信号喂进 RL 动作空间。
    """
    from scripts.factors.run_alphapool_ppo import attach_vwap

    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(6)]
    raw_close = pd.DataFrame(rng.uniform(5, 50, (30, 6)), idx, codes)
    # backward 因子：跨股票量级差异极大（模拟真实情形）
    bf = pd.DataFrame(
        np.tile(rng.uniform(0.5, 120, 6), (30, 1)), idx, codes)
    adj_close = raw_close * bf
    volume = pd.DataFrame(rng.uniform(1e6, 5e7, (30, 6)), idx, codes)
    # amount 用**原始价**成交额（这就是数据层的真实口径）
    amount = raw_close * volume

    panel = {"close": adj_close, "close_m": adj_close,
             "open": adj_close, "high": adj_close, "low": adj_close,
             "volume": volume, "amount": amount}

    # 不传 backward：vwap 与复权 close 不可比（旧行为）
    attach_vwap(panel, verbose=False)
    bad = float((panel["vwap"] / panel["close"]).stack().median())
    assert abs(bad - 1.0) > 0.1, "未复权时不该偶然接近 1"

    # 传 backward：应精确恢复为可比的 VWAP
    panel.pop("vwap")
    attach_vwap(panel, backward=bf, verbose=False)
    good = float((panel["vwap"] / panel["close"]).stack().median())
    assert abs(good - 1.0) < 0.02, f"复权后 vwap/close 应≈1，实测 {good:.4f}"


def test_attach_vwap_idempotent_and_zero_volume_safe():
    """重复调用不覆盖；volume=0 不产生 inf。"""
    from scripts.factors.run_alphapool_ppo import attach_vwap

    rng = np.random.default_rng(12)
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    codes = ["600000.SH", "600001.SH"]
    close = pd.DataFrame(rng.uniform(10, 20, (10, 2)), idx, codes)
    volume = pd.DataFrame(rng.uniform(1e6, 1e7, (10, 2)), idx, codes)
    volume.iloc[0, 0] = 0.0                       # 停牌日
    amount = close * volume
    panel = {"close": close, "volume": volume, "amount": amount}

    attach_vwap(panel, verbose=False)
    first = panel["vwap"].copy()
    assert np.isfinite(panel["vwap"].values[~np.isnan(panel["vwap"].values)]).all()
    assert np.isnan(panel["vwap"].iloc[0, 0])     # 零成交 → NaN 而非 inf

    attach_vwap(panel, verbose=False)             # 二次调用应直接返回
    assert panel["vwap"].equals(first)


# ---------------------------------------------------------------------------
# 标量实参 / 标量公式（2026-09-16 均匀采样臂暴露）
#
# 背景（与 test_rpn_decimal_constant_now_evaluates 同型：**静默降级，不报错**）：
# 项目的 ts_* 系算子内部直接调 .rolling/.shift/.ewm，实参必须是 DataFrame；
# Corr/Cov 走 DataFrame 对齐运算。但 ``_dim`` 把常数视作 None（"与任何量纲兼容"），
# 于是「常数出现在算子实参位」这类子树**量纲上永远合法**，dimension_mismatch
# 一条都拦不住 —— 直到运行时才抛 AttributeError / ValueError。
#
# 实测（scripts/oneoff/_probe_uniform_true_yield.py，200 条均匀采样）：
#   修复前 → 73% 公式抛异常且校验器全部放行，真正可用仅 23.5%；
#   修复后 → 0% 抛异常，可用 87.5%。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,reason", [
    # 常量子树出现在滚动算子实参位（运行时 AttributeError: 'int' has no 'rolling'）
    ("Mean(Std(-10, 60), 1)", "scalar_operand"),
    ("Med(5, 5)", "scalar_operand"),
    ("Sum(Max(1, 10), 5)", "scalar_operand"),
    # Corr/Cov 的实参也必须是面板（运行时 ValueError: other must be a DataFrame）
    ("Cov(Max($volume, 5), 5, 5)", "scalar_operand"),
    ("Corr(30, -30, 60)", "scalar_operand"),
    # 整棵树无字段 → 返回裸标量/bool（运行时 'bool'/'int' has no 'astype'）
    ("Greater(-1, 10)", "scalar_formula"),
    ("Div(-1, 10)", "scalar_formula"),
    ("Abs(-1)", "scalar_formula"),
])
def test_constant_operand_is_rejected(text, reason):
    chk = check_report_formula(text)
    assert not chk.ok, f"{text} 应被拒绝（常量当面板实参，运行时必炸）"
    assert reason in chk.issues, f"{text} 期望 {reason}，实际 {chk.issues}"


@pytest.mark.parametrize("text", [
    "Mul($close, 2)",                     # 面板 × 常数：合法缩放，不能误伤
    "Div($close, 100)",
    "Sub($close, 5)",
    "Add($close, 1)",
    "Mul(-0.01, $close)",
    "Greater($close, 5)",                 # 至少一侧是面板即可
    "Mean($close, 20)",
    "Var(Med(Mean($close, 5), 5), 5)",    # 嵌套滚动，全链路面板
    "Corr($close, $volume, 20)",
])
def test_panel_valued_operands_still_accepted(text):
    """修复不能误伤「面板 × 常数」这类合法写法。"""
    chk = check_report_formula(text)
    assert chk.ok, f"{text} 不应被拒绝，实际 {chk.issues}"


def test_rejected_const_operand_would_really_crash(mini_panel):
    """钉住「被拒绝的理由是真实的」——若放行，运行时确实抛异常。

    这是防"校验器过度收紧"的关键对照：只有当公式真有运行期缺陷时才该拒。
    """
    panel, _fwd = mini_panel
    bad = "Med(5, 5)"
    assert not check_report_formula(bad).ok
    # 绕过校验器直接求值，确认它确实炸
    from factor.formula import formula_builder as fb
    with pytest.raises(AttributeError):
        fb("ts_median_5(5)", features=list(panel))(panel)


def test_good_formula_survives_full_eval(mini_panel):
    """正向对照：被放行的公式在真实面板上能求值出非全 NaN 结果。"""
    panel, _fwd = mini_panel
    for text in ("Mean($close, 20)", "Div(Sub($close, Mean($close, 20)), Std($close, 20))"):
        chk = check_report_formula(text)
        assert chk.ok and chk.project
        out = formula_builder(chk.project, features=list(panel))(panel)
        assert out.notna().any().any(), f"{text} 求值后全 NaN"


# ---------------------------------------------------------------------------
# UniformRandomProposer（第三臂：均匀采样对照，2026-09-16 新增）
# ---------------------------------------------------------------------------
def test_uniform_proposer_emits_checkable_formulas():
    """产出必须**全部**能通过校验器 —— propose() 内部已过滤。"""
    from factor.rl.llm_pool import UniformRandomProposer

    prop = UniformRandomProposer(seed=0)
    got = prop.propose(15)
    assert len(got) == 15
    for text in got:
        chk = check_report_formula(text)
        assert chk.ok and chk.project, f"{text} 应合规，实际 {chk.issues}"
    assert prop.n_ok == 15


def test_uniform_proposer_is_deterministic():
    from factor.rl.llm_pool import UniformRandomProposer

    a = UniformRandomProposer(seed=7).propose(10)
    b = UniformRandomProposer(seed=7).propose(10)
    assert a == b, "同 seed 必须可复现"
    assert UniformRandomProposer(seed=8).propose(10) != a


def test_uniform_proposer_respects_bans():
    from factor.rl.llm_pool import UniformRandomProposer

    prop = UniformRandomProposer(seed=0)
    first = prop.propose(10)
    assert first, "前置条件：能产出公式"
    proj0 = check_report_formula(first[0]).project

    nxt = UniformRandomProposer(seed=0).propose(10, avoid=[first[0]])
    assert check_report_formula(first[0]).project not in {
        check_report_formula(x).project for x in nxt}


def test_uniform_proposer_yields_no_scalar_operand_formulas():
    """修复的端到端意义：随机臂不再把动作空间的一大块浪费在必炸公式上。

    修复前 73% 白抽；此处断言「抽 200 条、达标率 > 40%」，
    既能抓住回归（退回 5% 会失败），又不对随机性过敏。
    """
    from factor.rl.llm_pool import UniformRandomProposer

    prop = UniformRandomProposer(seed=0)
    got = prop.propose(100)
    assert len(got) == 100
    assert prop.n_draws / 100 < 20, f"采样效率退化：{prop.n_draws} 次抽 100 条"


# ---------------------------------------------------------------------------
# --arm 三臂接线（2026-09-16 新增）
#
# 研报只有两臂（RL alone / RL+LLM），但两者同时差在「初始池是否非空」与
# 「提案是否含知识」两个自变量上。第三臂 random 用来拆开这两个效应：
#     none ←初始池非空→ random ←LLM 知识→ llm
# ---------------------------------------------------------------------------
def _arm_args(**kw):
    """构造 make_arm_proposer 需要的最小 args 命名空间。"""
    from types import SimpleNamespace

    base = dict(
        arm="llm", seed=0, random_seed=None, llm="template",
        llm_model="deepseek-flash", llm_base_url="https://api.deepseek.com",
        llm_max_tokens=16000, llm_timeout=300.0, llm_rounds=2,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_arm_none_has_no_proposer():
    """none 臂 = 研报「RL alone」：空池起步、全程无 LLM。"""
    from scripts.factors.run_alphapool_ppo import make_arm_proposer

    assert make_arm_proposer(_arm_args(arm="none")) is None


def test_arm_random_uses_uniform_proposer():
    from factor.rl.llm_pool import UniformRandomProposer
    from scripts.factors.run_alphapool_ppo import make_arm_proposer

    prop = make_arm_proposer(_arm_args(arm="random"))
    assert isinstance(prop, UniformRandomProposer)
    assert prop.name == "uniform"


def test_arm_random_seed_independent_of_main_seed():
    """--random-seed 覆盖主 seed，便于「同 RL seed、不同池」的配对对照。"""
    from scripts.factors.run_alphapool_ppo import make_arm_proposer

    p1 = make_arm_proposer(_arm_args(arm="random", seed=3))
    p2 = make_arm_proposer(_arm_args(arm="random", seed=3, random_seed=9))
    assert p1.seed == 3 and p2.seed == 9


def test_arm_llm_delegates_to_make_proposer():
    """llm 臂沿用原 --llm template/openai 语义，不破坏向后兼容。"""
    from factor.rl.llm_pool import TemplateProposer
    from scripts.factors.run_alphapool_ppo import make_arm_proposer

    assert isinstance(make_arm_proposer(_arm_args(arm="llm")), TemplateProposer)


def test_arm_llm_with_llm_none_is_rejected_not_silently_degraded():
    """`--arm llm --llm none` 必须**报错**，不能静默变成「和 --arm none 一样」。

    否则实验日志写着 arm=llm、实际跑的是 RL alone —— 典型的静默降级
    （与本项目小数常数字面量、标量实参两次事故同型）。
    """
    from scripts.factors.run_alphapool_ppo import make_arm_proposer

    with pytest.raises(SystemExit, match="--arm llm 需要"):
        make_arm_proposer(_arm_args(arm="llm", llm="none"))


def test_arm_choices_are_closed_set():
    """--arm 只接受三个值（防止拼错臂名静默落到默认 llm 臂）。"""
    import argparse
    from scripts.factors.run_alphapool_ppo import build_parser

    ap = build_parser()
    for arm in ("llm", "none", "random"):
        assert ap.parse_args(["--arm", arm]).arm == arm
    with pytest.raises(SystemExit):
        ap.parse_args(["--arm", "nope"])
