#!/usr/bin/env bash
set -euo pipefail

python3 -m verl.trainer.main_ppo \
  trainer.use_v1=true \
  trainer.use_legacy_worker_impl=disable \
  trainer.v1.trainer_mode=separate_async \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.v1.sampler.max_off_policy_threshold=8 \
  trainer.v1.sampler.max_off_policy_strategy=drop \
  actor_rollout_ref.hybrid_engine=false \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=search_r1/llm_agent/search_r1_agent.yaml \
  "$@"
