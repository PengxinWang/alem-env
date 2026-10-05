# Alem 随机策略 Demo

使用官方 Alem 环境及其 masked-random 策略：每个智能体独立、均匀地从当前合法动作中采样。无需 LLM、模型权重、API Key、GPU 或桌面窗口。

## 运行

在仓库根目录执行（当前 `.venv` 已安装好）：

```bash
.venv/bin/python -u scripts/random_visual_demo.py --steps 100 --seed 0 --coord easy
```

默认保留官方世界大小、9 层世界、3 个智能体与 10,000 步 episode 上限；demo 最多运行 100 步，若 episode 提前结束则停止，不自动 reset。策略不训练、不规划，也不会有意协作。CPU 首次运行需要等待 JAX 编译，日志会标出所处阶段。

输出目录 `outputs/random_demo/`：

- `replay.gif`：三个智能体各自视角并排，每帧对应一个环境步，120ms/帧，循环播放。
- `initial.png`、`final.png`：初始和最终画面。
- `trace.jsonl`：每步的动作、奖励、位置、生命值和终止状态。
- `summary.json`：种子、步数、累计奖励、成就、后端和源代码版本。

这是短程环境连通性与可视化验证，不是论文训练或排行榜分数复现。`team_return_sum` 是三个智能体原始累计奖励的和，不是归一化排行榜分数。达到 demo 步数限制时不代表完整 episode 已结束。

## 重新安装

上游：https://github.com/alem-world/alem-env

本次版本：`14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r configs/dependencies/requirements-demo.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

只看终端统计可以直接运行官方例子：

```bash
JAX_PLATFORMS=cpu .venv/bin/python -u scripts/random_rl_agent.py --players 3 --coord easy --steps 100 --seed 0
```

更换种子、步数或难度时建议指定新的输出目录，以保留已有结果：

```bash
.venv/bin/python -u scripts/random_visual_demo.py --steps 300 --seed 1 --coord medium --output outputs/random_demo_medium
```
