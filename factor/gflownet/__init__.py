"""GFlowNet 因子挖掘（Phase 0 最小闭环）。

研报参照：国金证券《Alpha掘金系列之二十二：基于GFlowNet的低相关性量价因子挖掘策略》
(2026-04-10)。Phase 0 目标 = 验证 TB 训练闭环：能采样出多样（batch 内相关性 < 0.2）
且 IC 非零的因子，并与 PPO-RL 对照展示「模式崩溃 vs 多样性」。

模块划分：
- expr.py   表达式树构建 / 简化 / canonical 字符串（奖励缓存 key，兼容 formula.py）
- env.py    因子构造 MDP（状态 / 动作空间 / 合法掩码 / 状态编码）
- reward.py rank IC 奖励（含缓存）
- tb.py     Trajectory Balance 训练器
- ppo.py    简化 PPO 对照（Actor-Critic + clipped objective + 熵奖励）
"""
