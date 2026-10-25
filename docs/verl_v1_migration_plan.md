# Search-R1 向最新版 verl 架构迁移执行方案

## 1. 文档目的

本文档用于指导本仓库从历史版 verl/自定义 Search-R1 PPO 实现，逐步迁移到最新版 verl 的统一模型引擎、`TrainingWorker`、原生 Rollout Server、Checkpoint Engine、TransferQueue 和 V1 Trainer 架构。

当前阶段以“代码逻辑闭合、架构可学习、后续可实验”为目标，不以吞吐、显存占用和训练收敛为首轮验收条件。算法语义正确性（mask、log probability、参数隔离和权重版本）仍属于必须满足的要求。

## 2. 基线与边界

### 2.1 上游基线

- 上游仓库：`G:\Project\verl`
- 分支：`main`
- 基线提交：`dcb222b3c5f64c8280088856fb7f2c1c7c5706ec`
- 上游入口：`verl/trainer/main_ppo.py`、`verl/trainer/main_ppo_v0.py`
- 统一 Worker：`verl/workers/engine_workers.py`

迁移期间不得无记录地改变上游基线。更新上游后，应在本文档“变更记录”中记录新提交和重新评估结果。

### 2.2 必须保留的 Search-R1 能力

- 搜索与推理交错的多轮生成语义。
- 检索服务、查询构造、搜索状态和停止条件。
- Search-R1 数据集、prompt 模板和 trajectory 结构。
- reward、advantage、验证与指标逻辑。
- `search_r1/llm_agent` 中仍有价值的领域逻辑。
- FSDP/FSDP2；Megatron 保留架构入口。

### 2.3 暂不作为首轮目标

- 性能达到上游基准。
- 多节点容错和生产稳定性。
- 完全异步 PPO 的最佳 staleness 参数。
- Router Replay（除非引入 MoE 实验）。
- Distillation（除非引入 Teacher）。
- VeOmni、TorchTitan 等后端的真实运行验证。

## 3. 当前状态

本仓库仍以 `RayPPOTrainer + DataProto + 自定义 LLMGenerationManager` 为主。已经存在统一 `EngineRegistry` 和新版 Actor/Critic 原型，但 Actor、Rollout、Ref、数据协议和 Trainer 尚未形成最新版的完整闭环。

当前已加入一个过渡版 `verl/workers/engine_workers.py`，并让 `use_legacy_worker_impl=disable` 可以选择融合角色。该实现用于验证组合边界，不是最终实现；进入阶段 1 后应由最新版 `TrainingWorker` 设计替换，不继续形成第三套长期架构。

## 4. 目标架构

```text
SearchR1AgentLoop / AgentLoopManager
                 │
                 ▼
           TransferQueue
                 │
                 ▼
             V1 Trainer
                 │ TensorDict
       ┌─────────┼──────────┐
       ▼         ▼          ▼
    Actor      Critic      Ref
       │         │          │
       └──── TrainingWorker ┘
                 │
          EngineRegistry
       ┌─────────┼──────────┐
       ▼         ▼          ▼
     FSDP2    Megatron   other engines
       │
       ▼
  CheckpointEngine ──────► Native Rollout Server
                            (vLLM/SGLang)
```

Search-R1 应尽量作为 verl 基础设施上的扩展层存在，而不是持续 fork 大量 verl 内核：

```text
search_r1/
  agent_loop/      # 搜索状态机和工具调用
  rewards/         # Search-R1 reward
  datasets/        # 样本与轨迹转换
  trainers/        # 必要的薄扩展
  config/          # Search-R1 配置
```

## 5. 数据语义清单

迁移前必须为以下字段建立唯一语义，禁止仅以“名称相同”判断兼容：

| 语义 | 当前常见字段 | 目标表示 | 必查条件 |
|---|---|---|---|
| prompt token | `prompts` | TensorDict/Nested Tensor | 左右 padding 与真实长度 |
| response token | `responses` | TensorDict/Nested Tensor | 搜索/tool token 是否计入 |
| 完整序列 | `input_ids` | jagged/nested input | prompt-response 边界 |
| 有效 token | `attention_mask` | offsets/mask | padding 不进入 loss |
| response mask | `response_mask` | 显式 tensor | 工具返回和环境 token 规则 |
| 行为策略概率 | `old_log_probs` | TensorDict tensor | 与 response token 一一对齐 |
| 参考策略概率 | `ref_log_prob` | TensorDict tensor | Ref 参数不得被 Actor 更新 |
| token reward | `token_level_rewards` | TensorDict tensor | shape 与 mask 一致 |
| advantage/return | `advantages`/`returns` | TensorDict tensor | 截断及 bootstrap 规则 |
| 权重版本 | 隐式 | `global_steps`/version | rollout 不误用未来权重 |

阶段 2 必须建立一份可执行的字段契约测试。

## 6. 分阶段执行计划

### 阶段 0：基线冻结与差异清单

目标：让后续修改可追踪。

任务：

- 记录当前仓库关键入口和 Search-R1 定制点。
- 对比上游的 `engine_workers.py`、配置 dataclass、Engine API、Trainer 和 Agent Loop。
- 标注可直接移植、需要适配和不迁移的文件。
- 为当前过渡实现补充结构测试，作为替换参照。

验收：

- 差异表覆盖所有直接依赖。
- 无未分类的上游导入。
- 当前代码通过 compile/AST 检查。

### 阶段 1：迁移 TrainingWorker 与统一 Engine 配置

目标：Actor、Critic、Ref 都通过同一种训练执行抽象创建。

主要来源：

- 上游 `verl/workers/engine_workers.py`
- 上游 `verl/workers/config`
- 上游 `verl/workers/engine`

任务：

- 迁移 `TrainingWorker`、`TrainingWorkerConfig` 及必要配置类。
- 对齐 `BaseEngine` 的训练、推理、offload、参数导出和 checkpoint API。
- Actor、Critic、Ref 通过 `model_type + engine.strategy` 选择后端。
- 用最新版组合替换当前过渡版 `roles.ActorWorker` 组合。
- 暂时保留旧 Trainer 边界适配器。

验收：

- `main_ppo` 不再按 FSDP/Megatron 选择 Actor/Critic Worker 类。
- Actor/Ref/Critic 的 Engine 可通过配置解析构建。
- Worker API 结构测试覆盖 train/infer/save/load/to。

### 阶段 2：建立 DataProto 到 TensorDict 的迁移边界

目标：Worker 内部统一使用最新版 TensorDict；Search-R1 算法外层可暂时保留 DataProto。

任务：

- 新增单一适配层，禁止在各 Worker 中散落转换代码。
- 建立第 5 节字段契约和 shape/mask 断言。
- 对齐 Nested Tensor、remove-padding、动态 batch 和多模态 metadata。
- 逐步将 reward、advantage 和指标函数改为 TensorDict 原生接口。

验收：

- padding 与无 padding 往返后有效 token 完全一致。
- old/ref log probability 与 response token 无偏移。
- 变长、多轮、含工具返回的合成样本测试通过。

### 阶段 3：迁移最新版 ActorRolloutRefWorker

目标：统一组合 Actor、Rollout 和可选 Ref。

任务：

- Actor/Ref 使用 `TrainingWorker`。
- Rollout 使用 `BaseRollout`/ServerAdapter。
- 对齐 dispatch/collect 的真实并行拓扑。
- 融合 Ref 时只初始化一次 WorkerGroup，并验证参数隔离。
- 移除过渡版 Worker 和重复 legacy 路由。

验收：

- `actor`、`rollout`、`ref`、`actor_rollout`、`actor_rollout_ref` 角色均可完成配置构建。
- Trainer 能把 Actor 和 Ref RPC 路由到融合 WorkerGroup。
- 不再依赖 backend-specific ActorRolloutRefWorker。

### 阶段 4：接入 CheckpointEngine

目标：把训练端到 Rollout 的权重同步变成稳定抽象。

任务：

- 先迁移 `naive` backend。
- 同步接口携带 `global_steps`。
- 明确首次完整同步与后续同步行为。
- 为 LoRA full-weight/adapter/merge 三种路径保留扩展点。
- 后续再接入分桶和异步传输 backend。

验收：

- 同一权重版本不可重复、乱序应用。
- Actor 更新后导出的参数名与 Rollout 接收端匹配。
- 无 Rollout 时 checkpoint save/load 行为不受影响。

### 阶段 5：Search-R1 生成逻辑迁移到 Agent Loop

目标：保留搜索算法，将推理执行切换为 Native Rollout Server。

任务：

- 将 `LLMGenerationManager` 拆分为纯搜索状态机和推理客户端。
- 新建 `SearchR1AgentLoop`。
- 每个请求独立维护对话、搜索结果、步数和终止原因。
- 通过 vLLM/SGLang Server Client 生成，不直接控制 SPMD Worker。
- 保留验证路径和可复现随机种子。

验收：

- 合成 Rollout Client 下，多轮搜索状态转换测试通过。
- 并发请求之间状态不泄漏。
- 轨迹可无损转换成阶段 2 的训练字段契约。

### 阶段 6：接入 V1 Trainer 与 TransferQueue

目标：对齐最新版主流程，但先以逻辑同步方式运行。

任务：

- 迁移 V1 Trainer 的最小闭环。
- Agent Loop 输出写入 TransferQueue。
- Trainer 消费 TensorDict batch。
- 初始设置为“生成一批、消费一批、更新一次”。
- 逻辑稳定后再允许生成与训练重叠。

验收：

- 无 GPU 的 fake producer/consumer 测试通过。
- 队列关闭、异常传播和 backpressure 行为明确。
- 同步模式下的数据顺序和权重版本可预测。

### 阶段 7：清理、文档化与实验准备

任务：

- 删除过渡实现、无效 legacy 开关和重复配置。
- 形成最小 FSDP2、Megatron、同步和异步示例配置。
- 输出实验验证矩阵和已知限制。
- 更新 README/启动脚本。

## 7. 验证策略

### 7.1 每阶段必须执行

- Python compile/AST 检查。
- 模块导入检查。
- Hydra/OmegaConf 配置组合与 dataclass 转换。
- 不依赖 GPU 的构造、mock 和字段契约测试。
- 针对修改文件的 lint/type 检查（在项目工具可用时）。

### 7.2 获得 GPU 环境后执行

1. 单卡小模型初始化和一次 forward。
2. 两卡 FSDP/FSDP2 Actor、Ref、Critic forward。
3. 单个 PPO mini-batch backward/update。
4. Actor 更新前后参数与 log probability 变化。
5. Ref 参数保持不变。
6. Actor 到 Rollout 权重同步与版本检查。
7. 一条 Search-R1 多轮轨迹端到端生成。
8. checkpoint 保存、恢复及继续更新。
9. Megatron 多卡冒烟测试。
10. 异步模式 staleness、吞吐和收敛验证。

## 8. 必须优先防止的静默错误

- response mask 偏移或把 padding/tool observation 计入 loss。
- `old_log_probs`、`ref_log_prob` 与 token 不对应。
- Ref 与 Actor 意外共享可训练参数或 optimizer。
- Rollout 使用错误 global step 的权重。
- Megatron 非输出 PP rank 返回伪有效结果。
- mini/micro batch 的全局与每 DP rank 语义混淆。
- dynamic batching 后样本顺序无法还原。
- async queue 中旧策略样本没有版本标记。

这些问题必须通过断言和测试阻断，不能留给“后续调效果”。

## 9. 实施规则

- 每次只推进一个阶段，不跨阶段混合大面积修改。
- 每阶段先更新本文档中的依赖和决策，再修改代码。
- 优先移植上游抽象；Search-R1 特性放在扩展层。
- 不为暂时跑通复制 backend-specific 业务逻辑。
- 上游更新时先做差异审计，不直接覆盖本仓库。
- 每阶段完成后记录修改文件、检查命令、结果和遗留项。

## 10. 状态表

| 阶段 | 状态 | 说明 |
|---|---|---|
| 0. 基线与差异清单 | 已完成 | 见 `docs/verl_v1_dependency_audit.md` |
| 1. TrainingWorker | 源码完成，待 GPU | Actor/Ref 已通过 TrainingWorker 构建；Critic 已走同一 EngineRegistry。缺少 GPU，不能构建真实后端。 |
| 2. TensorDict 边界 | 源码完成，待 GPU | 单一 DataProto↔TensorDict adapter、response/log-prob 对齐断言和无 GPU 合成契约已覆盖。 |
| 3. 最新复合 Worker | 源码完成，待 GPU | Actor/Ref 由 TrainingWorker 构建；Actor 更新恢复 PPO epoch/minibatch 语义，RPC 仍保持 DataProto 兼容。 |
| 4. CheckpointEngine | 源码完成，待 GPU | `naive` registry、权重版本检查和复合 Worker 同步路径已接入。 |
| 5. SearchR1AgentLoop | 完成（源码） | 既有 `SearchR1AgentLoop` 是请求级 async server；V1 同步桥接保留其输出接入点。 |
| 6. V1/TransferQueue | 源码完成，待服务/GPU | 新增 `verl.trainer.ppo.v1.PPOTrainerSync` 与可注入 `TransferQueueAdapter`；fake producer/consumer 已通过，真实 TQ 服务和异步重叠待实验。 |
| 7. 清理与实验准备 | 完成（文档） | 保留 `use_legacy_worker_impl` 作为回滚开关；GPU 验证矩阵和已知限制已记录。 |

## 11. 变更记录

- 2026-10-25：创建执行方案；上游基线固定为 `dcb222b3c5f64c8280088856fb7f2c1c7c5706ec`。
- 2026-10-25：完成阶段 0；新增 `docs/verl_v1_dependency_audit.md`，确认阶段 1 的最小迁移边界。
- 2026-10-25：阶段 1 启动；新增兼容 `TrainingWorkerConfig`，尚未切换运行时 Worker。
- 2026-10-25：阶段 2 启动；新增 DataProto/TensorDict 双向适配与 token-logprob 对齐断言。
- 2026-10-25：阶段 4 启动；新增 `CheckpointEngineRegistry` 和 `naive` 版本化权重传输，并接入过渡复合 Worker。
- 2026-10-25：审计确认已有 SearchR1AgentLoop（请求级 async rollout）；阶段 5 将在 V1/TransferQueue 接入时完成收口。
- 2026-10-25：阶段 1 增加最小 `TrainingWorker`；阶段 4 将 `checkpoint_engine.naive` 写入 PPO 配置。
- 2026-10-25：阶段 3 启动；ActorRolloutRefWorker 的 Actor/Ref 改由 TrainingWorker 构建，保留 DataProto RPC adapter，并新增版本化 `update_weights`。
- 2026-10-25：恢复统一 Actor Worker 的 PPO epoch/minibatch 更新语义，学习率调度改为每次 PPO update 一次；新增最小 V1 同步 Trainer 与 TransferQueue adapter，并以 fake queue 完成无 GPU producer/consumer 契约验证。
- 2026-10-25：本地补齐 ray、tensordict、TransferQueue 及兼容版本 transformers；统一 Worker、V1 Trainer 和全部改动模块通过 Python compile/import。真实 Ray+GPU Engine 构建、真实 TransferQueue 服务及异步吞吐验证转入后续实验矩阵。
