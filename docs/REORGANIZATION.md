# 目录整理记录

上游基线：`alem-world/alem-env@14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`。

| 原位置 | 新位置 |
| --- | --- |
| `alem/` | `envs/alem/`（Python 导入名不变） |
| `baselines/*.py` | `algorithms/rl/` |
| `baselines/llm/` | `algorithms/llm/` |
| `examples/random_rl_agent.py` | `algorithms/random/agent.py`，启动入口为 `scripts/random_rl_agent.py` |
| `baselines/config/` | `configs/rl/` |
| `baselines/llm/config/` | `configs/llm/` |
| `requirements*` | `configs/dependencies/` |
| `examples/` 其他入口 | `scripts/` |
| 根目录指南文档 | `docs/` |
| `images/llm_*.gif` | `docs/media/` |

移除了 logo 素材和 `sample_agents_playing.gif`，保留环境运行必需贴图。上游 Git 历史仍保留这些文件，删除只减少当前工作树的体积。`outputs/` 与虚拟环境不会提交。

已更新 setuptools 包发现和配置文件打包、Hydra 配置定位、脚本、文档链接、Docker COPY/入口和 GitHub CI。上游 PyPI 发布 job 限制在官方仓库执行。

验证结果：

- 环境工厂测试：3 个通过。
- LLM ASCII map、language wrapper、prompt builder：168 个测试，165 个通过、3 个跳过。
- 全部 7 份 Hydra YAML 可加载；LLM 评估入口 `--cfg job` 成功。
- 相同 seed=0 / Easy / 3 agents / 100 步：整理前后 `trace.jsonl` 完全一致，奖励和成就一致；GIF 有 101 帧。
- Ruff lint / format、Python 编译、shell 语法检查通过。
- wheel 构建成功，核对环境包、贴图、随机策略、RL/LLM 配置均包含在包内。
- 文档本地 Markdown 链接检查通过。

本次没有运行 GPU 训练、真实 LLM API 调用或 Docker 镜像构建。
