from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent


def _resolve_project_path(value: str | os.PathLike[str] | None, default: Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser() if value is not None else default
    if not path.is_absolute():
        path = (base or PROJECT_ROOT) / path
    return path.resolve()


_PROJECT_ROOT_RAW = Path(os.environ.get("AITRADER_ROOT", CODE_ROOT.parent)).expanduser()
if _PROJECT_ROOT_RAW.is_absolute():
    PROJECT_ROOT = _PROJECT_ROOT_RAW.resolve()
else:
    PROJECT_ROOT = (CODE_ROOT.parent / _PROJECT_ROOT_RAW).resolve()
DATA_ROOT = _resolve_project_path(os.environ.get("AITRADER_DATA_ROOT"), PROJECT_ROOT / "data")
EXPERIMENTS_ROOT = _resolve_project_path(os.environ.get("AITRADER_EXPERIMENTS_ROOT"), DATA_ROOT / "experiments")

RAW_MARKET_DATA_DIR = _resolve_project_path(os.environ.get("AITRADER_RAW_DATA_DIR"), DATA_ROOT / "raw_market_data")
DATASETS_ROOT = _resolve_project_path(os.environ.get("AITRADER_DATASETS_ROOT"), DATA_ROOT / "datasets")
MODELS_ROOT = _resolve_project_path(os.environ.get("AITRADER_MODELS_ROOT"), DATA_ROOT / "models")
LOG_DIR = _resolve_project_path(os.environ.get("AITRADER_LOG_DIR"), DATA_ROOT / "logs")
RUNTIME_DIR = _resolve_project_path(os.environ.get("AITRADER_RUNTIME_DIR"), DATA_ROOT / "runtime")
SECRETS_DIR = _resolve_project_path(os.environ.get("AITRADER_SECRETS_DIR"), DATA_ROOT / "secrets")
