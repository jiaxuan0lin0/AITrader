"""MSGCA model framework for the deep-learning trading assignment.

Keep package-level imports lazy so data/factor utilities can import
``model.msgca.feature_set`` without requiring model-training dependencies.
"""

__all__ = [
    "FactorAwareEncoder",
    "MSGCA",
    "MSGCAConfig",
    "MSGCAOutput",
    "StrongFactorMLP",
    "build_model_from_layout",
    "evaluate_checkpoint",
    "load_config",
    "predict_dataset",
    "train_msgca",
]


def __getattr__(name: str):
    if name in {"MSGCAConfig", "load_config"}:
        from model.msgca.config import MSGCAConfig, load_config

        values = {"MSGCAConfig": MSGCAConfig, "load_config": load_config}
    elif name in {"FactorAwareEncoder", "MSGCA", "MSGCAOutput", "StrongFactorMLP"}:
        from model.msgca.modules import FactorAwareEncoder, MSGCA, MSGCAOutput, StrongFactorMLP

        values = {
            "FactorAwareEncoder": FactorAwareEncoder,
            "MSGCA": MSGCA,
            "MSGCAOutput": MSGCAOutput,
            "StrongFactorMLP": StrongFactorMLP,
        }
    elif name in {"build_model_from_layout", "evaluate_checkpoint", "predict_dataset"}:
        from model.msgca.inference import build_model_from_layout, evaluate_checkpoint, predict_dataset

        values = {
            "build_model_from_layout": build_model_from_layout,
            "evaluate_checkpoint": evaluate_checkpoint,
            "predict_dataset": predict_dataset,
        }
    elif name == "train_msgca":
        from model.msgca.trainer import train_msgca

        values = {"train_msgca": train_msgca}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals().update(values)
    return values[name]
