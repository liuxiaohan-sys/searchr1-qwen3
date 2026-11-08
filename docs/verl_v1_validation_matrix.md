# verl V1 验证矩阵

## 已自动验证

| 范围 | 验证 |
|---|---|
| V0/legacy 回退 | 配置路由单元测试 |
| TensorDict schema | token、mask、log-prob、reward、advantage、version 契约测试 |
| TransferQueue | fake backend producer/consumer/clear/recovery 测试 |
| Replay Buffer | terminal group、顺序、future/stale version、drop/wait 测试 |
| 同步 PPO step | reward、old/ref log-prob、value、advantage、actor/critic update 测试 |
| Checkpoint | actor/critic save/load marker 与 transport lifecycle 测试 |
| 异步模式 | mode registry、后台 rollout drain、staleness 测试 |

## 部署前必须验证

以下检查依赖真实模型、GPU、Ray 集群、TransferQueue 服务或检索服务，不能由 CPU mock 证明：

1. 单卡小模型完成一个 sync PPO step。
2. 双卡 FSDP2 Actor/Ref/Critic forward、backward 和 checkpoint resume。
3. Search-R1 多轮检索轨迹的 response mask 与 legacy 路径逐 token 对齐。
4. vLLM/SGLang 权重更新后 rollout log-prob 使用正确版本。
5. `colocate_async` 和 `separate_async` 在压力下满足 staleness 上界。
6. Megatron 多卡 smoke test。
7. 注册的 NCCL/NIXL/HCCL backend 在目标环境完成 topology 和传输验证。

`separate_async` 的启动命令必须额外指定 standalone checkpoint backend，例如
`actor_rollout_ref.rollout.checkpoint_engine.backend=nccl`，并确保该 backend 已通过
`custom_backend_module` 或环境包注册。`naive` 只适用于 actor/rollout 同进程路径，配置校验会拒绝
把它用于 `separate_async`。

## 本轮自动化结果

- V1 专项 CPU 测试：30 passed。
- V1 专项加协议与 Worker 配置回归：70 passed。
- 改动 Python 文件：`ruff check` 通过，`compileall` 通过。
- Hydra 三种 trainer mode 均可完成配置展开；`separate_async + naive` 按设计提前失败。

## 回退

任何 V1 实验均可用以下配置切回旧路径：

```yaml
trainer:
  use_v1: false
  use_legacy_worker_impl: auto
```

回退不会读取 V1 Replay Buffer；切换前应等待或清理对应 TransferQueue partition。
