import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import constants
from model_config import ModelConfig


class ModelConfigTest(unittest.TestCase):
    def test_hash_uses_values_in_recursive_key_order(self) -> None:
        config = ModelConfig(
            transport_modes={"train": 2, "bus": 0, "car": 1},
            sensors=["accelerometer", "gyroscope"],
            features=["mean", "std"],
            window_size_seconds=10,
            window_next_step_seconds=5,
        )
        values = (
            "raw_data"
            "raw_data.tar.gz"
            "runs"
            "mean"
            "std"
            "accelerometer"
            "gyroscope"
            "5"
            "10"
            "0"
            "1"
            "2"
            "1"
            "5"
            "5"
            "random_forest"
            "mlp"
            "-1"
            "42"
            "None"
            "50"
        )

        self.assertEqual(
            config.id,
            hashlib.sha256(values.encode("utf-8")).hexdigest(),
        )

    def test_prepare_run_directory_writes_canonical_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_path = Path(temporary) / "data"
            with patch.object(constants, "DATA_PATH", data_path):
                config = ModelConfig(
                    transport_modes={"train": 2, "bus": 0, "car": 1},
                    sensors=["accelerometer", "gyroscope"],
                    features=["mean", "std"],
                    window_size_seconds=10,
                    window_next_step_seconds=5,
                )
                run_path = config.prepare_run_directory()

            saved = yaml.safe_load(
                (run_path / "model.config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(run_path.name, config.id)
            self.assertEqual(saved, config.to_dict())
            self.assertEqual(list(saved), sorted(saved))
            self.assertTrue(
                all(
                    list(value) == sorted(value)
                    for value in saved.values()
                    if isinstance(value, dict)
                )
            )

    def test_from_yaml_loads_nested_pipeline_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "model.config.yaml"
            config_path.write_text(
                """
dataset:
  raw_data_dir: extracted
  raw_data_file_name: raw.tar.gz
  runs_dir: experiments
features:
  aggregations:
    - mean
  sensors:
    - accelerometer
  step_seconds: 3
  window_seconds: 9
labels:
  bus: 0
schema_version: 1
training:
  cross_validation_folds: 2
  holdout_splits: 2
  model_families:
    - random_forest
  n_jobs: 1
  random_seed: 7
  timeout_seconds: 60
  trials: 4
""",
                encoding="utf-8",
            )

            config = ModelConfig.from_yaml(str(config_path))

        self.assertEqual(config.raw_data_dir, "extracted")
        self.assertEqual(config.raw_data_file_name, "raw.tar.gz")
        self.assertEqual(config.runs_dir, "experiments")
        self.assertEqual(config.sensors, ["accelerometer"])
        self.assertEqual(config.features, ["mean"])
        self.assertEqual(config.window_next_step_seconds, 3)
        self.assertEqual(config.window_size_seconds, 9)
        self.assertEqual(config.transport_modes, {"bus": 0})
        self.assertEqual(config.model_families, ["random_forest"])
        self.assertEqual(config.random_seed, 7)
        self.assertEqual(config.timeout_seconds, 60)


if __name__ == "__main__":
    unittest.main()
