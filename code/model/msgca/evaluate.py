from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from model.msgca.config import load_config
from model.msgca.inference import build_model_from_layout, evaluate_checkpoint, predict_dataset
from model.msgca.metrics import write_evaluation_outputs

__all__ = ["build_model_from_layout", "evaluate_checkpoint", "predict_dataset"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MSGCA checkpoint and write prediction artifacts.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "valid", "val", "holdout", "all"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-prefix", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is None and args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = load_config(overrides=checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {})
    else:
        config = load_config(args.config)
    config.ensure_output_dirs()
    predictions = evaluate_checkpoint(config, args.checkpoint, split=args.split, limit=args.limit)
    paths = write_evaluation_outputs(predictions, config.paths.output_root, prefix=args.output_prefix)
    print(f"predictions={paths['predictions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
