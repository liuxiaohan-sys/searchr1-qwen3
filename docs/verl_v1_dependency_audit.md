# verl V1 迁移：阶段 0 依赖审计

日期：2026-10-25  
上游基线：`G:\Project\verl` @ `dcb222b3c5f64c8280088856fb7f2c1c7c5706ec`

## 结论

当前仓库不能直接替换为上游最新版 `engine_workers.py`。其运行时依赖新版 `TrainingWorker`、TensorDict 数据协议、CheckpointEngine、扩展配置模型和 V1 Trainer；这些依赖在当前仓库均未形成完整闭环。

迁移入口应是“先引入 TrainingWorker 与最小配置”，而不是继续修改当前早期 `ActorRolloutRefWorker`。当前的 `verl/workers/engine_workers.py` 是阶段 0 的结构性参照，必须在阶段 3 被替换。

## 核心差异

| 上游能力 | 上游位置 | 当前状态 | 迁移阶段 | 决策 |
|---|---|---|---|---|
| `TrainingWorker` | `workers/engine_workers.py` | 缺失 | 1 | 必须移植 |
| `TrainingWorkerConfig` | `workers/config` | 已新增兼容 dataclass | 1 | 后续随 Worker 补齐字段 |
| Engine 的统一训练 API | `TrainingWorker.train_mini_batch/infer_batch` | 仅角色 Worker API | 1 | 必须适配 |
| TensorDict/Nested Tensor | Worker/Trainer 边界 | 当前以 `DataProto` 为主 | 2 | 增加单一 adapter |
| 最新复合 Worker | `ActorRolloutRefWorker` | 早期兼容版 | 3 | 完整替换 |
| CheckpointEngine | `checkpoint_engine` | 缺失 | 4 | 先实现 `naive` |
| Native rollout server | 新版 rollout/agent loop | 当前有 SPMD 兼容路径 | 5 | 重接 Search-R1 Agent Loop |
| TransferQueue + V1 Trainer | `trainer/ppo/v1` | 缺失 | 6 | 最后迁移 |

## 上游 Worker API 与当前 API

上游 `TrainingWorker` 的核心接口是：

```text
reset, to, train_batch, train_mini_batch, infer_batch,
save_checkpoint, load_checkpoint
```

当前早期组合 Worker 的核心接口是：

```text
init_model, compute_log_prob, compute_ref_log_prob, update_actor,
sleep, wake_up, save_checkpoint, load_checkpoint
```

迁移策略：阶段 1 提供前者；阶段 2 在 Trainer 边界增加数据适配；阶段 3 由新版组合 Worker 对外保留后者所需要的 PPO RPC 名称。

## 当前仓库的保留点

以下代码不能被上游直接覆盖，须在迁移后通过扩展点重新接入：

| 位置 | 保留原因 | 目标归属 |
|---|---|---|
| `search_r1/llm_agent/generation.py` | 搜索与推理交错的领域流程 | `SearchR1AgentLoop` |
| `search_r1/llm_agent/async_agent_loop.py` | 异步 Agent 行为参考 | Agent Loop/TransferQueue 适配 |
| `search_r1/search/*` | 检索服务和请求协议 | 工具客户端层 |
| `verl/trainer/ppo/ray_trainer.py` 中 Search-R1 导入 | 当前训练入口定制 | 阶段 6 前保留 adapter |

## 阶段 1 的最小依赖包

必须审计并按需移植：

```text
verl/workers/engine_workers.py: TrainingWorker 部分
verl/workers/config/{checkpoint,distillation,disaggregation}.py
verl/workers/config/{actor,critic,engine,model,optimizer}.py 的字段差异
verl/workers/engine/* 的 BaseEngine API 差异
verl/utils/tensordict_utils.py 的必要 helper
```

不在阶段 1 直接移植：

```text
verl/checkpoint_engine/*
verl/trainer/ppo/v1/*
TransferQueue
Native server 生命周期改造
```

## V1 运行时依赖

已将上游固定版本 `TransferQueue==0.1.8` 加入 `requirements.txt` 和
`setup.py`。V1 Trainer 的运行验证需同时具备 Ray、TensorDict、TransferQueue
及 GPU/rollout 后端；本地仅在这些依赖均完成安装后执行导入和结构测试。

## 阶段 1 前置检查

- [x] 上游基线提交已固定。
- [x] 关键 Worker、配置和 Trainer 文件已定位。
- [x] Search-R1 定制目录已定位。
- [x] 当前过渡 Worker 与上游最终 Worker 的 API 差异已确认。
- [x] 为 `TrainingWorkerConfig` 选择兼容 dataclass 方案；首版仅包含当前 Engine 所需字段。
- [x] 建立当前/上游 Engine 公共 API 的逐方法差异表：当前缺少 `grad` offload 与 mode kwargs，阶段 1 的兼容 Worker 暂不传递它们。
- [x] 确认当前环境没有安装 `tensordict`；代码库已使用该协议，运行时验证须在实验环境完成。

## 阶段 0 验收

阶段 0 的代码和文档审计已完成。进入阶段 1 前，必须先完成本文件“阶段 1 前置检查”中的后三项；它们将决定第一批代码迁移的精确范围。
