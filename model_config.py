import constants
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml


@dataclass
class ModelConfig:
    transport_modes: Dict[str, int] = field(default_factory=dict)
    sensors: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    window_size_seconds: int = 10
    window_next_step_seconds: int = 5
    schema_version: int = 1
    raw_data_file_name: str = constants.RAW_DATA_FILE_NAME
    raw_data_dir: str = "raw_data"
    runs_dir: str = constants.RUNS_DIR
    model_families: List[str] = field(
        default_factory=lambda: ["random_forest", "mlp"]
    )
    random_seed: int = 42
    holdout_splits: int = 5
    cross_validation_folds: int = 5
    trials: int = 50
    timeout_seconds: int | None = None
    n_jobs: int = -1

    id: str = field(init=False, default='')
    raw_data_path: Path = field(init=False, default_factory=Path)
    raw_data_extracted_path: Path = field(init=False, default_factory=Path)
    cleaned_data_path: Path = field(init=False, default_factory=Path)
    transformed_data_path: Path = field(init=False, default_factory=Path)
    models_path: Path = field(init=False, default_factory=Path)
    reports_path: Path = field(init=False, default_factory=Path)
    run_path: Path = field(init=False, default_factory=Path)
    sensor_features_in_order: List[str] = field(init=False, default_factory=list)

    @classmethod
    def from_yaml(cls, path: str = "model.config.yaml") -> "ModelConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "labels" in data or "dataset" in data or "training" in data:
            dataset = data.get("dataset", {})
            features = data.get("features", {})
            training = data.get("training", {})
            return cls(
                transport_modes={
                    str(key).lower(): int(value)
                    for key, value in data["labels"].items()
                },
                sensors=list(features["sensors"]),
                features=list(features["aggregations"]),
                window_size_seconds=int(features["window_seconds"]),
                window_next_step_seconds=int(features["step_seconds"]),
                schema_version=int(data.get("schema_version", 1)),
                raw_data_file_name=str(
                    dataset.get("raw_data_file_name", constants.RAW_DATA_FILE_NAME)
                ),
                raw_data_dir=str(dataset.get("raw_data_dir", "raw_data")),
                runs_dir=str(dataset.get("runs_dir", constants.RUNS_DIR)),
                model_families=list(
                    training.get("model_families", ["random_forest", "mlp"])
                ),
                random_seed=int(training.get("random_seed", 42)),
                holdout_splits=int(training.get("holdout_splits", 5)),
                cross_validation_folds=int(
                    training.get("cross_validation_folds", 5)
                ),
                trials=int(training.get("trials", 50)),
                timeout_seconds=training.get("timeout_seconds"),
                n_jobs=int(training.get("n_jobs", -1)),
            )

        return cls(
            transport_modes=data["transport_modes"],
            sensors=data["sensors"],
            features=data["features"],
            window_size_seconds=data["window_size_seconds"],
            window_next_step_seconds=data["window_next_step_seconds"],
        )

    def __post_init__(self) -> None:
        self.id = self.generate_config_id()
        self.raw_data_path = constants.DATA_PATH / self.raw_data_file_name
        self.raw_data_extracted_path = constants.DATA_PATH / self.raw_data_dir
        self.run_path = constants.DATA_PATH / self.runs_dir / self.id
        self.cleaned_data_path = self.run_path / constants.CLEANED_DATA_DIR
        self.transformed_data_path = self.run_path / constants.TRANSFORMED_DATA_DIR
        self.models_path = self.run_path / constants.MODELS_DIR
        self.reports_path = self.run_path / constants.REPORTS_DIR
        self.sensor_features_in_order = [
            f"{sensor}#{feature}"
            for sensor in self.sensors
            for feature in self.features
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": {
                "raw_data_dir": self.raw_data_dir,
                "raw_data_file_name": self.raw_data_file_name,
                "runs_dir": self.runs_dir,
            },
            "features": {
                "aggregations": list(self.features),
                "sensors": list(self.sensors),
                "step_seconds": self.window_next_step_seconds,
                "window_seconds": self.window_size_seconds,
            },
            "labels": dict(self.transport_modes),
            "schema_version": self.schema_version,
            "training": {
                "cross_validation_folds": self.cross_validation_folds,
                "holdout_splits": self.holdout_splits,
                "model_families": list(self.model_families),
                "n_jobs": self.n_jobs,
                "random_seed": self.random_seed,
                "timeout_seconds": self.timeout_seconds,
                "trials": self.trials,
            },
        }

    def generate_config_id(self) -> str:
        hash_input = "".join(_stringified_values(self.to_dict()))
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def prepare_run_directory(self) -> Path:
        self.run_path.mkdir(parents=True, exist_ok=True)
        config_path = self.run_path / "model.config.yaml"
        canonical_yaml = yaml.safe_dump(
            self.to_dict(),
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=False,
        )
        if config_path.exists():
            if config_path.read_text(encoding="utf-8") != canonical_yaml:
                raise ValueError(
                    f"Configuration hash collision or modified config copy: {config_path}"
                )
        else:
            config_path.write_text(canonical_yaml, encoding="utf-8")
        return self.run_path


def _stringified_values(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _stringified_values(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _stringified_values(item)
    elif value is None:
        yield "None"
    else:
        yield str(value)


if __name__ == '__main__':
    config = ModelConfig.from_yaml()
