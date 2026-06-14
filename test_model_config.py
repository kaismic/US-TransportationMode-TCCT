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
        values = "meanstdaccelerometergyroscope012510"

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
            self.assertEqual(
                list(saved["transport_modes"]),
                sorted(saved["transport_modes"]),
            )


if __name__ == "__main__":
    unittest.main()
