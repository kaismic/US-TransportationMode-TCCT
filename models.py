from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def suggest_model(
    trial,
    families: list[str],
    random_seed: int,
    n_jobs: int,
):
    family = trial.suggest_categorical("family", list(families))
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "random_forest_n_estimators", 100, 600, step=100
            ),
            max_depth=trial.suggest_categorical(
                "random_forest_max_depth", [None, 12, 20, 30, 40]
            ),
            min_samples_leaf=trial.suggest_int(
                "random_forest_min_samples_leaf", 1, 10
            ),
            max_features=trial.suggest_categorical(
                "random_forest_max_features", ["sqrt", "log2", 0.5]
            ),
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=n_jobs,
        )
    if family == "mlp":
        hidden_name = trial.suggest_categorical(
            "mlp_hidden_layers", ["64", "128", "128_64", "256_128"]
        )
        hidden_layers = tuple(int(value) for value in hidden_name.split("_"))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        alpha=trial.suggest_float("mlp_alpha", 1e-6, 1e-2, log=True),
                        learning_rate_init=trial.suggest_float(
                            "mlp_learning_rate_init", 1e-4, 1e-2, log=True
                        ),
                        batch_size=trial.suggest_categorical(
                            "mlp_batch_size", [64, 128, 256]
                        ),
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model family: {family}")


def _family_param(params: dict[str, Any], family: str, name: str) -> Any:
    namespaced = f"{family}_{name}"
    if namespaced in params:
        return params[namespaced]
    return params[name]


def model_from_params(
    params: dict[str, Any],
    random_seed: int,
    n_jobs: int,
):
    family = str(params["family"])
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(_family_param(params, family, "n_estimators")),
            max_depth=_family_param(params, family, "max_depth"),
            min_samples_leaf=int(_family_param(params, family, "min_samples_leaf")),
            max_features=_family_param(params, family, "max_features"),
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=n_jobs,
        )
    if family == "mlp":
        hidden_layers = tuple(
            int(value)
            for value in str(_family_param(params, family, "hidden_layers")).split("_")
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        alpha=float(_family_param(params, family, "alpha")),
                        learning_rate_init=float(
                            _family_param(params, family, "learning_rate_init")
                        ),
                        batch_size=int(_family_param(params, family, "batch_size")),
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model family: {family}")


def fit_with_sample_weight(model, x, y, sample_weight) -> None:
    if isinstance(model, Pipeline):
        model.fit(x, y, classifier__sample_weight=sample_weight)
    else:
        model.fit(x, y, sample_weight=sample_weight)
