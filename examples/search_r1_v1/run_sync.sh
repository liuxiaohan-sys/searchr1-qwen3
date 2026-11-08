#!/usr/bin/env bash
set -euo pipefail

# Supply model/data/retriever overrides after this script's arguments.
python3 -m verl.trainer.main_ppo \
  trainer.use_v1=true \
  trainer.use_legacy_worker_impl=disable \
  trainer.v1.trainer_mode=sync \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.agent.default_agent_loop=search_r1_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=search_r1/llm_agent/search_r1_agent.yaml \
  "$@"
