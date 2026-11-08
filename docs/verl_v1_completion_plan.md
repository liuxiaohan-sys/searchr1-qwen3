# Search-R1 对齐最新版 verl 的迁移收口计划

## 1. 目标与约束

本计划用于收口 2026-10-25 提交 `2d84638` 引入的 V1 迁移原型，使 Search-R1
逐步具备最新版 verl 的统一 Trainer、TrainingWorker、TensorDict/TransferQueue、
Replay Buffer、Checkpoint Engine 和 Rollout Server 生命周期。

迁移必须满足以下约束：

- 永久保留 `trainer.use_legacy_worker_impl` 回退路径，直到用户明确要求移除。
- `auto` 继续保持当前稳定语义；新实现必须由显式配置启用。
- 每个阶段单独提交、单独验证，失败时只需切回 legacy 配置，不回滚用户数据。
- Search-R1 搜索状态机、reward、数据字段和多轮轨迹语义不得被上游代码覆盖。
- 未通过真实 GPU 验证的能力必须标记为实验性，不得写成“迁移完成”。

## 2. 当前事实基线

### 2.1 本仓库

- 当前提交基线：`2d84638ff7fd98c39a305a664a27a5da4769b719`。
- 主入口仍直接运行 V0 `RayPPOTrainer`。
- 默认 `trainer.use_legacy_worker_impl=auto`，实际使用 backend-specific legacy Worker。
- 迁移开始时，`verl/workers/engine_workers.py` 仅为 phase-1 兼容实现；本轮已补齐 V1 所需的
  TensorDict train/infer、PPO mini-batch/epoch、loss 注入、checkpoint 和权重同步接口。
- `verl/trainer/ppo/v1/trainer_sync.py` 只是未接入主入口的单 batch 原型。
- Trainer 主链路仍使用 `DataProto`；TensorDict 只存在于 Worker 边界。
- Checkpoint Engine 仅实现进程内 `naive` 原型。
- TransferQueue 只有可注入 adapter，没有生产者、Replay Buffer 和消费闭环。

### 2.2 上游目标能力

上游参考仓库为本地 `G:\Project\verl`，本轮冻结提交为
`b17218ba58a6fbcdfb0de32772e3c2a79f88529d`。更新该基线前必须重新执行差异审计；
禁止用“最新版”代替 commit SHA。

目标能力包括：

- `trainer.use_v1` 和 `trainer.v1.trainer_mode` 入口选择。
- `sync`、`colocate_async`、`separate_async` 共享 Trainer 基类和 step 语义。
- Actor、Ref、Critic 使用通用 `TrainingWorker`；Rollout 保留复合协调层。
- TensorDict + TransferQueue KV metadata 作为 V1 主数据通道。
- Replay Buffer、样本版本、staleness、refill 和恢复语义。
- CheckpointEngineManager 管理 Actor 到 Rollout 的权重传输和 replicas 生命周期。
- Agent Loop、Reward Loop 与 Rollout Server 的流式协作。

## 3. 配置与回退契约

迁移期间同时保留两组正交开关：

```yaml
trainer:
  use_v1: false                 # Trainer 控制流：false=V0，true=V1
  use_legacy_worker_impl: auto  # Worker 实现：auto/enable=legacy，disable=unified
  v1:
    trainer_mode: sync          # sync/colocate_async/separate_async
```

合法组合及预期：

| `use_v1` | `use_legacy_worker_impl` | 预期用途 |
|---|---|---|
| `false` | `auto`/`enable` | 当前稳定回退路径，行为不得改变 |
| `false` | `disable` | V0 Trainer + unified Worker，迁移期兼容验证 |
| `true` | `disable` | 最终目标路径 |
| `true` | `auto`/`enable` | 不支持；配置校验应尽早报错 |

所有新示例先显式设置开关，不改变旧脚本的隐式行为。

## 4. 分阶段实施

### M0：基线和测试护栏

任务：

- 记录上游 commit、文件清单和 Search-R1 本地差异。
- 修正旧迁移文档中把原型标为“完成”的状态。
- 增加配置组合测试、模块导入测试和 legacy 路由测试。

验收：

- `main_ppo` 的 legacy 路由测试通过。
- 四种配置组合均得到确定结果。
- Python compile/import 不依赖 GPU。

### M1：接入真正的 V1 入口和配置

任务：

- 增加 `trainer.use_v1` 与 `trainer.v1.*` 配置。
- 把当前 V0 `TaskRunner` 移到兼容入口，保持原逻辑不变。
- V1 入口通过 trainer registry 选择 `sync/colocate_async/separate_async`。
- 在完整 V1 Trainer 未落地前，`use_v1=true` 必须清晰报出实验阶段错误，不能静默回退。

验收：

- 默认命令仍进入 V0。
- `use_v1=true` 确实进入 V1 factory。
- V1 + legacy Worker 的非法组合在启动前失败。

### M2：统一 Critic/Ref/Actor TrainingWorker

任务：

- 将 `CriticConfig` 转换为 `TrainingWorkerConfig(model_type="value_model")`。
- Critic 使用 `reset/set_loss_fn/infer_batch/train_batch` 通用接口。
- Actor/Ref 删除配置字段 `pop()` 等有副作用的兼容操作。
- 保留原 `roles.CriticWorker` 与 backend-specific Worker 作为 legacy 路径。

验收：

- unified 路径不导入 backend-specific Actor/Critic Worker。
- Actor、Ref、Critic 的配置可在 CPU 上完成构造前校验。
- value loss、PPO loss 的注入各有单元测试。

### M3：TensorDict 原生训练契约

任务：

- 定义唯一的训练字段 schema 和 KV metadata schema。
- V1 Trainer、TrainingWorker、loss 函数使用 TensorDict。
- `DataProto` adapter 只允许存在于 V0/unified 兼容边界。
- 覆盖 padding、remove-padding、变长、多轮、工具 observation 和多模态 metadata。

验收：

- V1 Trainer 中不出现 `DataProto`。
- response、mask、old/ref log-prob、reward、advantage shape 契约测试通过。

### M4：TransferQueue 与 Replay Buffer 同步闭环

任务：

- 初始化/关闭 TransferQueue，并管理 partition 生命周期。
- Agent Loop 输出 KV batch；Trainer 通过 metadata 采样。
- 实现同步 Replay Buffer、失败组处理、step 后清理和 checkpoint 恢复。
- 首轮只支持 `trainer_mode=sync`，不做训练/生成重叠。

验收：

- fake queue 和真实本地 queue 均能完成 produce/sample/train/clear。
- 权重版本、样本版本和恢复后的 pending 状态可验证。

### M5：完整同步 V1 Trainer

任务：

- 迁移统一 `PPOTrainer` 基类和 `PPOTrainerSync` hooks。
- 接通 rollout、reward、old/ref log-prob、advantage、critic、actor、metrics、validation、checkpoint。
- 接入 Search-R1 Agent Loop，不改变搜索轨迹语义。

验收：

- CPU mock 完成一个完整 PPO step。
- GPU 小模型完成 rollout → reward → actor/critic update → weight sync。
- 同一输入下 legacy 与 V1 的 mask/reward/advantage 对齐。

### M6：CheckpointEngineManager 和 Rollout replicas

任务：

- 用 manager 管理 prepare/topology/update/sleep/resume。
- 保留 `naive`，再按环境增加 NCCL/HCCL/NIXL 等 backend。
- 支持首次完整同步、版本拒绝、LoRA adapter/full-weight 路径。

验收：

- 旧权重和乱序权重被拒绝。
- colocated 与 separate rollout 都能收到正确版本。
- checkpoint save/load 不依赖 rollout 是否在线。

### M7：Reward Loop 与 Agent Loop 流式化

任务：

- 引入 RewardLoopManager 和独立 resource pool。
- Search-R1 搜索状态机作为 Agent Loop 插件，不直接控制训练 Worker。
- rollout、搜索工具和 reward 可流式重叠，并传播异常/取消。

验收：

- 并发请求不共享搜索状态。
- reward worker 失败不会产生可训练的半成品 batch。
- validation 与 train 使用一致的轨迹解释规则。

### M8：异步训练模式

任务：

- 实现 `colocate_async`，再实现 `separate_async`。
- 增加 warmup、staleness `drop/wait`、refill、off-policy metrics。
- 最后评估动态 GPU 切换，不将其作为首轮正确性前提。

验收：

- staleness 上界有测试保证。
- resume 后 pending/running prompts 可恢复或重发。
- sync 与 async 的 step、checkpoint、指标语义有文档化差异。

### M9：验证、示例与清理

任务：

- 增加 FSDP/FSDP2、Megatron、sync、两种 async 示例。
- 更新启动脚本和故障排查文档。
- 删除已经没有调用者的过渡 adapter；保留 legacy 回退路径。

验收：

- CI 覆盖配置、CPU mock 和可用的 GPU 冒烟测试。
- legacy 与 V1 均有可复制启动命令。
- 所有未验证能力明确列入限制，不以 TODO 代替状态。

## 5. 每阶段统一验证清单

1. `python -m compileall` 覆盖改动模块。
2. Hydra/OmegaConf compose 与 dataclass 转换。
3. 无 GPU import 和 CPU mock。
4. legacy 默认路径回归。
5. Tensor shape、mask、样本顺序和权重版本断言。
6. 获得 GPU 后执行单卡、双卡 FSDP2，再执行 Megatron 冒烟。

## 6. 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| M0 | 源码完成，待验证 | 已固定上游 SHA，并增加配置/路由护栏测试 |
| M1 | 源码完成，待 GPU | 已接入 `trainer.use_v1`、三种 mode factory、独立 V1 TaskRunner 与提前配置校验 |
| M2 | 源码完成，待 GPU | unified 路径直接注册 TrainingWorker(value_model)，V0 通过显式 TensorDict bridge 调用；legacy 路径不变 |
| M3 | 源码完成，待 GPU | V1 Trainer/Worker/Queue 使用 TensorDict；DataProto 仅保留在 V0 adapter 与既有 AgentLoop API 边界 |
| M4 | 源码完成，待真实服务/GPU | KVBatchMeta、Replay Buffer、TensorDict producer/consumer、版本门禁和成功后 clear 已闭合 |
| M5 | 源码完成，待 GPU | V1 TaskRunner 已接真实 Ray Worker/数据集/Agent Loop/Queue，并由 TensorDict PPO transaction 驱动同步 fit |
| M6 | 源码完成，待 GPU backend | Manager 支持 naive 与注册 transport 的 actor publish/rollout commit、版本单调性及 replica abort/wake；具体 NCCL/NIXL/HCCL 由 registry 插件提供 |
| M7 | 源码完成，待服务验证 | AgentLoopWorkerTQ 逐请求后台生成并流式发布；RewardLoopManager 支持 colocated/worker handle |
| M8 | 源码完成，待服务/GPU | colocate/separate async 已注册，支持后台 rollout、warmup、parameter sync cadence 与 drop/wait staleness |
| M9 | 源码完成，待部署验证 | 已增加 sync/colocate_async/separate_async 示例和独立验证矩阵；真实 GPU/服务项明确列出 |

## 7. 变更日志

- 2026-11-08：创建迁移收口计划；确认保留 `use_legacy_worker_impl` 回退路径，
  并将此前“源码完成”的原型状态重新分类为“部分原型/未接入”。
- 2026-11-08：冻结上游基线 `b17218ba58a6fbcdfb0de32772e3c2a79f88529d`；
  增加 V1/V0 正交配置、V1 trainer factory 和路由测试。默认仍为 V0/legacy；尚未接通的
  V1 生命周期会明确失败，不会静默回退。
- 2026-11-08：M2 unified Critic 改为直接使用 `TrainingWorker(model_type="value_model")`；
  Trainer 负责 CriticConfig 转换、value loss 注入及 V0 DataProto bridge，legacy CriticWorker 保持不变。
- 2026-11-08：M3 增加统一 TensorDict 训练契约，覆盖 response 对齐字段、mask、input/attention
  shape 和 global_steps/weight_version；V0 adapter 与 TransferQueue consumer 共用校验。
- 2026-11-08：M4 引入可测试的 TransferQueue ReplayStore、KVBatchMeta 和同步 ReplayBuffer；
  支持 terminal group、按权重版本排序、future-version 拒绝、超时、materialization 与清理。
- 2026-11-08：TransferQueueAdapter 增加原生 TensorDict put/get，PPOTrainerSync 消费原生
  TensorDict 并保留 V0 DataProto wrapper；Queue→Worker→权重版本同步测试通过。
- 2026-11-08：同步 Trainer 仅在 actor update 与权重同步成功后清理 KV batch；失败时保留数据供恢复。
- 2026-11-08：增加 AgentLoopTQProducer，把 Search-R1 AgentLoopOutput 转成版本化 TensorDict，
  并按 pending/running/finished/failure prompt tag 模型写入 TransferQueue。
- 2026-11-08：M6 增加 CheckpointEngineManager，统一 naive 权重更新、版本提交等待和
  replica sleep/wake/abort；为后续 NCCL/NIXL/HCCL topology 保留稳定 Trainer API。
- 2026-11-08：M5 新增共享 PPOTrainer 基类和完整同步 TensorDict transaction，串联 Replay
  Buffer、old/ref log-prob、critic value、reward、advantage/return、critic/actor update、权重同步和成功清理。
- 2026-11-08：统一 ActorRolloutRefWorker RPC 增加 TensorDict 原生分支；DataProto padding/mini-batch
  循环仅保留在 V0 compatibility branch，V1 直接调用 TrainingWorker infer/train。
- 2026-11-08：V0 TaskRunner 抽出只负责资源构造的 TaskRunnerCore；V1 TaskRunner 复用资源构造后，
  由 AgentLoopManagerTQ、ReplayBuffer 和 PPOTrainerSync 驱动真实同步 fit，不调用 V0 fit。
- 2026-11-08：M7 增加 RewardLoopManager，统一 colocated reward callable 与独立 worker handle
  调度；PPOTrainer 优先从 reward loop 获取 token-level rewards。
- 2026-11-08：M8 注册 colocate_async/separate_async，共享后台 rollout executor；增加
  ReplayBufferAsync 的 max_off_policy_threshold 与 drop/wait 策略。
- 2026-11-08：V1 PPOTrainer 增加 actor/critic checkpoint、latest marker、恢复步数和 validation
  callback；TaskRunner 按 save/test frequency 调用 V1 hooks。
- 2026-11-08：CheckpointEngineManager 完成注册 transport 的 actor publish/rollout commit 生命周期；
  非 naive backend 必须在 registry 注册，缺失 consumer API 会在更新前明确失败。
- 2026-11-08：M9 增加三种 V1 mode 启动脚本和 `docs/verl_v1_validation_matrix.md`，
  区分已自动验证与必须在目标 GPU/Ray/TransferQueue/检索环境执行的部署门禁。
- 2026-11-08：M7 将兼容的整批发布替换为 AgentLoopWorkerTQ；每条轨迹独立后台执行，
  完成即写入 TransferQueue，并用 running/finished/failure terminal tag 管理生命周期。
- 2026-11-08：收口审计补齐真实训练语义：TrainingWorker 按 PPO epoch/mini-batch 更新；V1 复用
  GAE/GRPO/RLOO 等既有 advantage estimator；RewardLoopManager 接入主入口；异步 warmup、
  parameter_sync_step 与实际 weight_version 生效。
- 2026-11-08：TransferQueue 配置正式纳入 Hydra defaults，V1 启动时显式启用；新增跨平台参考配置
  生成器并同步 `_generated_ppo_trainer.yaml`。`separate_async` 禁止使用仅支持同进程的 `naive`
  权重传输，必须选择已注册的 standalone transport。
- 2026-11-08：CheckpointEngineManager 增加非 naive transport 的 prepare/build_topology/
  init_process_group/send/receive/finalize 闭环；具体 NCCL/NIXL/HCCL 实现仍由目标环境插件提供，
  没有对应依赖或插件时会在启动/首次同步时明确失败。
