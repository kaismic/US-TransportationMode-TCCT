import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import constants
from dataset_reuse import (
    materialize_compatible_cleaned,
    materialize_compatible_transformed,
)
from model_config import ModelConfig


def _config(
    *,
    modes: dict[str, int],
    sensors: list[str],
    features: list[str],
    window_size: int = 10,
) -> ModelConfig:
    return ModelConfig(
        transport_modes=modes,
        sensors=sensors,
        features=features,
        window_size_seconds=window_size,
        window_next_step_seconds=5,
    )


class DatasetReuseTest(unittest.TestCase):
    def test_filters_cleaned_data_from_saved_superset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(constants, "DATA_PATH", Path(temporary) / "data"):
                saved = _config(
                    modes={"bus": 0, "car": 1},
                    sensors=["accelerometer", "gyroscope"],
                    features=["mean", "std"],
                )
                saved.prepare_run_directory()
                source_file = (
                    saved.cleaned_data_path / "nested" / "sensorfile_1_bus_1.csv"
                )
                source_file.parent.mkdir(parents=True)
                with source_file.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["0", "accelerometer", "1", "2", "3"])
                    writer.writerow(["0", "gyroscope", "4", "5", "6"])

                current = _config(
                    modes={"bus": 0},
                    sensors=["accelerometer"],
                    features=["mean"],
                )
                current.prepare_run_directory()

                self.assertTrue(materialize_compatible_cleaned(current))
                output_file = (
                    current.cleaned_data_path / "nested" / "sensorfile_1_bus_1.csv"
                )
                with output_file.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
                provenance = json.loads(
                    (current.cleaned_data_path / "reuse.json").read_text(
                        encoding="utf-8"
                    )
                )

            self.assertEqual(rows, [["0", "accelerometer", "1", "2", "3"]])
            self.assertEqual(provenance["reused_from"], saved.id)

    def test_selects_feature_columns_from_transformed_superset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(constants, "DATA_PATH", Path(temporary) / "data"):
                saved = _config(
                    modes={"bus": 0, "car": 1},
                    sensors=["accelerometer", "gyroscope"],
                    features=["mean", "std"],
                )
                saved.prepare_run_directory()
                source_file = saved.transformed_data_path / "sensorfile_1_bus_1.csv"
                source_file.parent.mkdir(parents=True)
                with source_file.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(saved.sensor_features_in_order)
                    writer.writerow(["1", "2", "3", "4"])

                current = _config(
                    modes={"bus": 0},
                    sensors=["accelerometer", "gyroscope"],
                    features=["mean"],
                )
                current.prepare_run_directory()

                self.assertTrue(materialize_compatible_transformed(current))
                output_file = current.transformed_data_path / source_file.name
                with output_file.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))

            self.assertEqual(
                rows,
                [["accelerometer#mean", "gyroscope#mean"], ["1", "3"]],
            )

    def test_rejects_transformed_data_with_different_window_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(constants, "DATA_PATH", Path(temporary) / "data"):
                saved = _config(
                    modes={"bus": 0},
                    sensors=["accelerometer"],
                    features=["mean"],
                )
                saved.prepare_run_directory()
                source_file = saved.transformed_data_path / "sensorfile_1_bus_1.csv"
                source_file.parent.mkdir(parents=True)
                source_file.write_text("accelerometer#mean\n1\n", encoding="utf-8")

                current = _config(
                    modes={"bus": 0},
                    sensors=["accelerometer"],
                    features=["mean"],
                    window_size=20,
                )
                current.prepare_run_directory()

                self.assertFalse(materialize_compatible_transformed(current))


if __name__ == "__main__":
    unittest.main()
