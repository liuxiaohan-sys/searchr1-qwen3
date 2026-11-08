"""Cross-platform equivalent of generate_trainer_config.sh."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    ("ppo_trainer", "_generated_ppo_trainer.yaml", []),
    ("ppo_megatron_trainer", "_generated_ppo_megatron_trainer.yaml", ["--config-name=ppo_megatron_trainer.yaml"]),
)


def main() -> None:
    for config_name, output_name, extra_args in SPECS:
        command = [sys.executable, str(ROOT / "scripts" / "print_cfg.py"), "--cfg", "job", *extra_args]
        rendered = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
        marker = "actor_rollout_ref:"
        if marker not in rendered:
            raise RuntimeError(f"Hydra output for {config_name} has no {marker!r} section")
        body = marker + rendered.split(marker, 1)[1]
        header = (
            "# This reference configration yaml is automatically generated via "
            "'scripts/generate_trainer_config.sh' or 'scripts/generate_trainer_config.py'\n"
            f"# It flattens verl/trainer/config/{config_name}.yaml into a single file.\n"
            "# Do not modify this file directly.\n"
            "# The file is usually only for reference and never used.\n\n"
        )
        (ROOT / "verl" / "trainer" / "config" / output_name).write_text(header + body, encoding="utf-8")
        print(f"Generated: verl/trainer/config/{output_name}")


if __name__ == "__main__":
    main()
