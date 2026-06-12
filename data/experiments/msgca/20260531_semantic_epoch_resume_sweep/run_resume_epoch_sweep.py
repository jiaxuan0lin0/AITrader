from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

CODE_DIR = Path("/data/jiaxuanLin/AItrader/code")
sys.path.insert(0, str(CODE_DIR))

from model.msgca.config import write_resolved_config
from model.msgca.run_systematic_ablations import (
    CODE_ROOT,
    PYTHON_BIN,
    build_variant_config,
    flatten_summary,
    variant_definitions,
    write_matrix_summary,
    write_run_summary,
)


ROOT = Path("/data/jiaxuanLin/AItrader/data/experiments/msgca/20260531_semantic_epoch_resume_sweep")
RUN_ROOT = ROOT / "runs"
CONFIG_ROOT = ROOT / "configs"
LOG_ROOT = ROOT / "logs"
BASE_CONFIG = CODE_ROOT / "model/msgca/config.yaml"
VARIANT_NAME = "gpt_final_upgrade_h48_proto2_topk005_softgate_train20260520"
TARGET_EPOCHS = (5, 8, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run semantic final MSGCA epoch sweep with checkpoint resume.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    variant = _variant()
    rows: list[dict[str, object]] = []
    previous_checkpoint: Path | None = None
    for target_epoch in TARGET_EPOCHS:
        job_id = f"20260531_semantic_resume_softgate_e{target_epoch}"
        run_dir = RUN_ROOT / f"{job_id}_{VARIANT_NAME}"
        config_dir = CONFIG_ROOT / job_id
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{VARIANT_NAME}.yaml"
        config = build_variant_config(BASE_CONFIG, variant, run_dir, target_epoch)
        write_resolved_config(config, config_path)
        meta = deepcopy(variant)
        meta["target_epoch"] = target_epoch
        if previous_checkpoint is not None:
            meta["resume_checkpoint"] = str(previous_checkpoint)
        config_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        train_cmd = [
            PYTHON_BIN,
            "-m",
            "model.msgca.train",
            "--config",
            str(config_path),
            "--final-validate",
        ]
        if previous_checkpoint is not None:
            train_cmd.extend(["--resume-checkpoint", str(previous_checkpoint)])
        eval_cmd = [
            PYTHON_BIN,
            "-m",
            "model.msgca.evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(run_dir / "checkpoints/msgca_latest.pt"),
            "--split",
            "holdout",
            "--output-prefix",
            "holdout",
        ]
        print(f"target_epoch={target_epoch}", flush=True)
        print("train_cmd=" + " ".join(train_cmd), flush=True)
        print("eval_cmd=" + " ".join(eval_cmd), flush=True)
        if args.prepare_only:
            previous_checkpoint = run_dir / "checkpoints/msgca_latest.pt"
            continue

        subprocess.run(train_cmd, check=True, cwd=str(CODE_ROOT))
        subprocess.run(eval_cmd, check=True, cwd=str(CODE_ROOT))
        checkpoint = run_dir / "checkpoints/msgca_latest.pt"
        write_run_summary(run_dir, config_path, checkpoint)
        summary_path = run_dir / "run_summary.json"
        row = flatten_summary(f"{VARIANT_NAME}_e{target_epoch}", json.loads(summary_path.read_text(encoding="utf-8")))
        row["status"] = "ok"
        rows.append(row)
        write_matrix_summary(ROOT / "semantic_resume_epoch_summary.csv", rows)
        previous_checkpoint = checkpoint
    return 0


def _variant() -> dict[str, object]:
    for variant in variant_definitions():
        if variant["name"] == VARIANT_NAME:
            return deepcopy(variant)
    raise KeyError(f"Unknown variant: {VARIANT_NAME}")


if __name__ == "__main__":
    raise SystemExit(main())
