"""守卫：tracked 代码里不得出现「只对某台机器成立」的绝对盘符路径。

背景（2026-10-13）
------------------
同一仓库在两台机器上部署，数据盘不同（一台 ``D:/data``、一台 ``E:/data``），且
2026-09-25 的迁移只搬了一部分（本机行情缓存在 D:、因子库/模型仍在 E:）。硬编码
``E:/data/...`` 的代码在另一台机器上**改配置时不会暴露**，只在真正跑到那行时
FileNotFoundError —— 或更糟，静默读到另一个盘上的**旧副本**。

已实证的两次静默错读：
1. ``config/schedule.yaml`` 的 altdata ``product_glob`` 指向 ``E:/data/parquet``
   旧副本，而任务实际写 ``Config.cache().root``（本机 D:）⇒ doctor 的新鲜度
   核对长期看错文件；
2. ``scripts/textmining/*`` 的 FinBERT 目录缺省值写死 ``D:/data/models``，而本机
   模型只在 E: ⇒ 6 处脚本在本机可跑、换机器即断。

统一做法
--------
- 数据路径 → ``config.Config.paths()`` / ``Config.bert_model_dir()``；
- 机器特有绝对路径（解释器 / 仓库根 / 数据根）→ ``${VAR:默认值}`` 占位符
  （见 ``config/settings.yaml`` 文件头「路径约定」与 ``config/schedule.yaml``）。

允许清单
--------
每条例外都必须写明理由；新增例外 = 一次显式决策，而不是随手加一个路径。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 引号紧跟盘符 ⇒ 只命中**字符串字面量**，不误伤注释/文档里的举例
#: （如 "原为 e://data//" 这种历史说明）。
LITERAL = re.compile(r"""['"][A-Za-z]:[\\/]""")
#: ``${VAR:默认值}`` 占位符：其默认值本就是某台机器的取值，属于有意为之。
ENV_PLACEHOLDER = re.compile(r"\$\{[^}]*\}")

SUFFIXES = (".py", ".yaml", ".yml", ".toml", ".bat", ".ps1", ".cmd")

#: 路径前缀 / 精确路径 → 理由
ALLOWED: dict[str, str] = {
    "tests/": "测试夹具：故意构造 Windows 绝对路径，验证路径归一化与任务 XML 生成，"
              "不读真实盘",
}


def _tracked(suffixes: tuple[str, ...]) -> list[str]:
    """只审计 git 跟踪的文件（自动跳过 _tmp/ reports/ scripts/oneoff/ 等）。"""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8")
    if out.returncode != 0:
        pytest.skip("不在 git 工作区内，跳过")
    return [f for f in out.stdout.split("\0") if f and f.endswith(suffixes)]


def _probe_text(path: Path, text: str) -> str:
    """非 .py：去掉注释行与 ${...} 占位符，避免误报历史说明与占位符默认值。"""
    if path.suffix == ".py":
        return text
    body = "\n".join(line.split("#")[0] for line in text.splitlines())
    return ENV_PLACEHOLDER.sub("", body)


def test_no_hardcoded_drive_paths() -> None:
    offenders: list[str] = []
    for rel in _tracked(SUFFIXES):
        if any(rel.startswith(p) or rel == p for p in ALLOWED):
            continue
        path = ROOT / rel
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(_probe_text(path, text).splitlines(), 1):
            if LITERAL.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()[:90]}")

    assert not offenders, (
        "发现硬编码盘符路径（换机器后会 FileNotFoundError 或静默读旧副本）：\n  "
        + "\n  ".join(offenders)
        + "\n\n改用 config.Config.paths() / Config.bert_model_dir()，或写成 "
          "${VAR:默认值} 占位符。确有例外请加进本文件 ALLOWED 并写明理由。"
    )
