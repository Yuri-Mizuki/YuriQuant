"""跨脚本复用的公共库（CLI 骨架 / 全A共享件 / 组合编排 / 原子 IO）。**不是入口**。

2026-09-11 自 `scripts/` 根目录按功能归类迁入。
2026-10-19：HS300 时代的 ``e2e_common`` 编排已移至 ``scripts/archive/``；
新增 ``alla_common.py`` 作为全A实验共享底座。
调用方式：`python -m scripts.common.<script>`（或 `python scripts/common/<script>.py`）。
"""
