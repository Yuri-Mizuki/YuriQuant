"""一次性迁移：把 legacy 默认库里混在一起的 29 个因子，按来源分到数据集子库。

- 真实因子（source=mining:hs300_2025_std）→ 数据集 hs300_2025
- mock 因子（mining:mine_factors / gp:mine_factors / synthesis:*）→ 数据集 mock

迁移完成后清空 legacy 默认库（数据已复制到数据集子库，无丢失）。
"""
import sys
sys.path.insert(0, "E:/YuriQuant")

from config import Config
from research.factor_library import FactorLibrary
from scripts.mine_factors import build_real_panel, gen_mock_panel_with_signal

cfg = Config.get()
legacy = FactorLibrary()
reg = legacy.list_all()
print("legacy 因子数:", len(reg))

# ---- 真实因子 -> hs300_2025 ----
real_src = "mining:hs300_2025_std"
real_rows = reg[reg["source"] == real_src]
mock_rows = reg[reg["source"] != real_src]
print(f"真实因子 {len(real_rows)} 个 -> hs300_2025；mock 派生因子 {len(mock_rows)} 个 -> mock")

real_panel, real_ret = build_real_panel(cfg, 20250101, 20251231)
real_lib = FactorLibrary(dataset="hs300_2025")
for _, r in real_rows.iterrows():
    name = r["name"]
    fp = legacy.get_panel(name)
    if fp is None:
        print("  skip (无面板):", name)
        continue
    real_lib.register(name, fp, real_ret, kind="raw", formula=name, source=real_src)
print("  hs300_2025 入库:", len(real_lib.list_all()))

# ---- mock 因子 -> mock ----
mock_panel = gen_mock_panel_with_signal()
mock_ret = mock_panel["close"].pct_change().shift(-1)
mock_lib = FactorLibrary(dataset="mock")
for _, r in mock_rows.iterrows():
    name = r["name"]
    fp = legacy.get_panel(name)
    if fp is None:
        print("  skip (无面板):", name)
        continue
    kind = r["kind"] if "kind" in r and r["kind"] in ("raw", "composite") else "raw"
    parents = [p for p in str(r.get("parents", "")).split("|") if p]
    mock_lib.register(name, fp, mock_ret, kind=kind, formula=r.get("formula", name),
                      parents=parents, source=r["source"])
print("  mock 入库:", len(mock_lib.list_all()))

# ---- 清空 legacy 默认库 ----
print("清空 legacy 默认库 ...")
for name in legacy.list_all()["name"].tolist():
    legacy.delete(name)
print("legacy 剩余:", len(legacy.list_all()))
print("datasets:", FactorLibrary.list_datasets())
