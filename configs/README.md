# 配置

- `rl/`：MARL 算法的 Hydra 配置。
- `llm/`：LLM 评估的 Hydra 配置。
- `dependencies/`：基础 CPU/GPU 依赖与随机 demo 的锁定版本。

算法入口使用相对其源文件的绝对定位方式加载这里的配置，不需要复制配置到工作目录。随机 demo 的种子、步数、难度使用脚本命令行参数配置。
