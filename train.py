from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from dataset_reuse import materialize_compatible_transformed
from model_config import ModelConfig
from models import fit_with_sample_weight, model_from_params, suggest_model
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

import utils


METADATA_COLUMNS = ["transport_mode", "group_id"]


def _label_order(labels: dict[str, int]) -> tuple[list[int], list[str]]:
    ordered = sorted(labels.items(), key=lambda item: item[1])
    return [value for _, value in ordered], [name for name, _ in ordered]


def _class_limited_splits(y: np.ndarray, desired: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(0, min(desired, int(counts.min())))


def _group_limited_splits(y: np.ndarray, groups: np.ndarray, desired: int) -> int:
    frame = pd.DataFrame({"label": y, "group": groups}).drop_duplicates()
    group_counts = frame.groupby("label")["group"].nunique()
    if group_counts.empty:
        return 0
    return max(0, min(desired, int(group_counts.min()), len(set(groups))))


def validation_splits(
    y: np.ndarray,
    groups: np.ndarray,
    desired_splits: int,
    random_seed: int,
):
    grouped_splits = _group_limited_splits(y, groups, desired_splits)
    if grouped_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=grouped_splits,
            shuffle=True,
            random_state=random_seed,
        )
        return splitter.split(np.zeros(len(y)), y, groups)

    stratified_splits = _class_limited_splits(y, desired_splits)
    if stratified_splits < 2:
        raise ValueError("Not enough samples per class for validation")
    splitter = StratifiedKFold(
        n_splits=stratified_splits,
        shuffle=True,
        random_state=random_seed,
    )
    return splitter.split(np.zeros(len(y)), y)


def grouped_holdout_indices(
    y: np.ndarray,
    groups: np.ndarray,
    splits: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_splits = _group_limited_splits(y, groups, splits)
    if grouped_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=grouped_splits,
            shuffle=True,
            random_state=random_seed,
        )
        train_indices, test_indices = next(
            splitter.split(np.zeros(len(y)), y, groups)
        )
        return train_indices, test_indices

    stratified_splits = _class_limited_splits(y, splits)
    if stratified_splits < 2:
        raise ValueError("Not enough samples per class for holdout validation")
    splitter = StratifiedKFold(
        n_splits=stratified_splits,
        shuffle=True,
        random_state=random_seed,
    )
    train_indices, test_indices = next(splitter.split(np.zeros(len(y)), y))
    return train_indices, test_indices


def metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: dict[str, int],
) -> dict[str, Any]:
    label_indices, label_names = _label_order(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_indices,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=label_indices
        ).tolist(),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "rows": int(len(y_true)),
    }


class Train:
    config: ModelConfig
    df: pd.DataFrame

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_yaml()
        self.config.prepare_run_directory()

    def create_dataframe(self) -> None:
        """
        Create a DataFrame by reading all transformed CSV files and concatenating them.
        """

        if not materialize_compatible_transformed(self.config):
            from preprocess import Preprocess

            Preprocess(self.config).transform_files()
        input_files = list(self.config.transformed_data_path.rglob("*.csv"))
        if not input_files:
            raise FileNotFoundError(
                "No transformed data is available for this configuration. "
                "Run preprocess.py --all first."
            )
        print(f"Total files to read: {len(input_files)}")

        frames = []
        for file in tqdm(input_files, desc="Reading transformed files"):
            df = pd.read_csv(file)
            mode = utils.get_transport_mode_from_path(file)
            df["transport_mode"] = self.config.transport_modes[mode]
            df["group_id"] = utils.get_user_id_from_path(file)
            frames.append(df)

        result_df = pd.concat(frames, ignore_index=True)
        print(f"DataFrame shape before dropping rows with NaN: {result_df.shape}")

        result_df = result_df.dropna(subset=self.config.sensor_features_in_order)

        print(f"Final DataFrame shape: {result_df.shape}")

        self.df = result_df

    def create_model(self, classifier, n_features):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        options = {id(classifier): {"zipmap": False, "output_class_labels": True}}
        input_shape = [("features", FloatTensorType([1, n_features]))]
        return convert_sklearn(classifier, initial_types=input_shape, options=options)

    def train_and_save_model(
        self,
        trials: int | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        Tune configured model families with Optuna, evaluate on a holdout split,
        and export the best final model trained on all transformed data.
        """

        import optuna
        from skl2onnx.helpers.onnx_helper import save_onnx_model

        self.create_dataframe()

        self.config.models_path.mkdir(parents=True, exist_ok=True)
        self.config.reports_path.mkdir(parents=True, exist_ok=True)

        feature_columns = self.config.sensor_features_in_order
        x = self.df[feature_columns].to_numpy(dtype=np.float32)
        y = self.df["transport_mode"].to_numpy(dtype=np.int64)
        groups = self.df["group_id"].astype(str).to_numpy()

        train_indices, holdout_indices = grouped_holdout_indices(
            y,
            groups,
            self.config.holdout_splits,
            self.config.random_seed,
        )
        x_train = x[train_indices]
        y_train = y[train_indices]
        groups_train = groups[train_indices]

        def objective(trial) -> float:
            fold_scores: list[float] = []
            for fold, (fold_train, fold_valid) in enumerate(
                validation_splits(
                    y_train,
                    groups_train,
                    self.config.cross_validation_folds,
                    self.config.random_seed,
                )
            ):
                model = suggest_model(
                    trial,
                    self.config.model_families,
                    self.config.random_seed + fold,
                    self.config.n_jobs,
                )
                weights = compute_sample_weight("balanced", y_train[fold_train])
                fit_with_sample_weight(
                    model,
                    x_train[fold_train],
                    y_train[fold_train],
                    weights,
                )
                predictions = model.predict(x_train[fold_valid])
                fold_scores.append(
                    float(f1_score(y_train[fold_valid], predictions, average="macro"))
                )
                trial.report(float(np.mean(fold_scores)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        study.optimize(
            objective,
            n_trials=trials or self.config.trials,
            timeout=timeout_seconds
            if timeout_seconds is not None
            else self.config.timeout_seconds,
        )

        best_params = dict(study.best_trial.params)
        evaluation_model = model_from_params(
            best_params,
            self.config.random_seed,
            self.config.n_jobs,
        )
        evaluation_weights = compute_sample_weight("balanced", y[train_indices])
        fit_with_sample_weight(
            evaluation_model,
            x[train_indices],
            y[train_indices],
            evaluation_weights,
        )
        holdout_predictions = evaluation_model.predict(x[holdout_indices])
        report = metrics_report(
            y[holdout_indices],
            holdout_predictions,
            self.config.transport_modes,
        )
        report["best_cross_validation_macro_f1"] = float(study.best_value)
        report["best_params"] = best_params
        report["feature_names"] = feature_columns
        report["holdout_groups"] = sorted(set(groups[holdout_indices]))
        report["training_groups"] = int(len(set(groups[train_indices])))
        report["rows"] = int(len(self.df))

        final_model = model_from_params(
            best_params,
            self.config.random_seed,
            self.config.n_jobs,
        )
        all_weights = compute_sample_weight("balanced", y)
        fit_with_sample_weight(final_model, x, y, all_weights)

        onnx_model = self.create_model(final_model, len(feature_columns))
        model_path = self.config.models_path / "best_model.onnx"
        save_onnx_model(onnx_model, model_path)
        (self.config.reports_path / "training-summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        study.trials_dataframe().to_csv(
            self.config.reports_path / "optuna-trials.csv", index=False
        )

        print(f"Best model trained and saved to {model_path}")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    Train().train_and_save_model(
        trials=args.trials,
        timeout_seconds=args.timeout_seconds,
    )
