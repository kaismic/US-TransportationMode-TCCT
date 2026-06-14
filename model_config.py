from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List
from pathlib import Path
import constants
import hashlib
import yaml

@dataclass
class ModelConfig:
    # Fields loaded from YAML
    transport_modes: Dict[str, int] = field(default_factory=dict)
    sensors: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    window_size_seconds: int = 10
    window_next_step_seconds: int = 5

    # Derived fields — excluded from __init__, populated in __post_init__
    id: str = field(init=False, default='')
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
        return cls(
            transport_modes=data["transport_modes"],
            sensors=data["sensors"],
            features=data["features"],
            window_size_seconds=data["window_size_seconds"],
            window_next_step_seconds=data["window_next_step_seconds"],
        )

    def __post_init__(self) -> None:
        self.id = self.generate_config_id()
        self.run_path = constants.DATA_PATH / constants.RUNS_DIR / self.id
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
            "features": list(self.features),
            "sensors": list(self.sensors),
            "transport_modes": dict(self.transport_modes),
            "window_next_step_seconds": self.window_next_step_seconds,
            "window_size_seconds": self.window_size_seconds,
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
