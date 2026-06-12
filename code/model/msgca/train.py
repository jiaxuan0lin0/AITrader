from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path
from typing import Sequence

from model.msgca.config import load_config, write_resolved_config
from model.msgca.trainer import train_msgca


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MSGCA model.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for local debugging.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-only", action="store_true", help="Use only the configured train split and skip validation outputs.")
    parser.add_argument("--final-validate", action="store_true", help="Force final validation even if config.train.final_validate is false.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Continue training from an existing MSGCA checkpoint.")
    parser.add_argument("--resume-weights-only", action="store_true", help="Load checkpoint weights but start the configured schedule from epoch 0.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable()
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.train_only:
        config.train.final_validate = False
        config.train.validate_each_epoch = False
    if args.final_validate:
        config.train.final_validate = True
    config.ensure_output_dirs()
    write_resolved_config(config, config.paths.output_root / "config.resolved.yaml")
    result = train_msgca(
        config,
        limit=args.limit,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
        resume_weights_only=args.resume_weights_only,
    )
    print(f"checkpoint={result['checkpoint_path']}")
    if result.get("validation_predictions_path") is not None:
        print(f"validation_predictions={result['validation_predictions_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
