from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Callable

import constants
import utils
import yaml
from model_config import ModelConfig


def _mapping_subset(required: dict[str, int], available: dict[str, int]) -> bool:
    return all(available.get(key) == value for key, value in required.items())


def _candidate_runs(
    config: ModelConfig,
    compatible: Callable[[ModelConfig], bool],
    directory_name: str,
) -> list[tuple[int, str, Path, ModelConfig]]:
    candidates = []
    for config_path in sorted(config.run_path.parent.glob("*/model.config.yaml")):
        run_dir = config_path.parent
        if run_dir == config.run_path:
            continue
        try:
            saved = ModelConfig.from_yaml(str(config_path))
        except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
            continue
        source_dir = run_dir / directory_name
        if compatible(saved) and list(source_dir.rglob("*.csv")):
            excess = (
                len(saved.transport_modes) - len(config.transport_modes)
                + len(saved.sensors) - len(config.sensors)
                + len(saved.features) - len(config.features)
            )
            candidates.append((excess, run_dir.name, run_dir, saved))
    return candidates


def _write_provenance(target: Path, source_run: Path, stage: str) -> None:
    (target / "reuse.json").write_text(
        json.dumps(
            {"reused_from": source_run.name, "stage": stage},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def materialize_compatible_cleaned(config: ModelConfig) -> bool:
    target = config.cleaned_data_path
    if list(target.rglob("*.csv")):
        return True
    if target.exists() and any(target.iterdir()):
        return False

    def compatible(saved: ModelConfig) -> bool:
        return _mapping_subset(config.transport_modes, saved.transport_modes) and set(
            config.sensors
        ).issubset(saved.sensors)

    candidates = _candidate_runs(config, compatible, constants.CLEANED_DATA_DIR)
    if not candidates:
        return False
    _, _, source_run, _ = min(candidates, key=lambda candidate: candidate[:2])
    source = source_run / constants.CLEANED_DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    written_files = 0
    try:
        for source_file in sorted(source.rglob("*.csv")):
            if (
                utils.get_transport_mode_from_path(source_file)
                not in config.transport_modes
            ):
                continue
            output_file = target / source_file.relative_to(source)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            wrote_row = False
            with (
                source_file.open("r", encoding="utf-8", newline="") as input_stream,
                output_file.open("w", encoding="utf-8", newline="") as output_stream,
            ):
                reader = csv.reader(input_stream)
                writer = csv.writer(output_stream)
                for row in reader:
                    if len(row) > constants.RawDataFieldLocation.SENSOR_TYPE and row[
                        constants.RawDataFieldLocation.SENSOR_TYPE
                    ] in config.sensors:
                        writer.writerow(row)
                        wrote_row = True
            if not wrote_row:
                output_file.unlink(missing_ok=True)
            else:
                written_files += 1
        if written_files == 0:
            shutil.rmtree(target, ignore_errors=True)
            return False
        _write_provenance(target, source_run, "cleaned_data")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    print(f"Reused cleaned data from {source_run}")
    return True


def materialize_compatible_transformed(config: ModelConfig) -> bool:
    target = config.transformed_data_path
    if list(target.rglob("*.csv")):
        return True
    if target.exists() and any(target.iterdir()):
        return False

    def compatible(saved: ModelConfig) -> bool:
        return (
            _mapping_subset(config.transport_modes, saved.transport_modes)
            and set(config.sensors) == set(saved.sensors)
            and set(config.features).issubset(saved.features)
            and config.window_size_seconds == saved.window_size_seconds
            and config.window_next_step_seconds == saved.window_next_step_seconds
        )

    candidates = _candidate_runs(config, compatible, constants.TRANSFORMED_DATA_DIR)
    if not candidates:
        return False
    _, _, source_run, _ = min(candidates, key=lambda candidate: candidate[:2])
    source = source_run / constants.TRANSFORMED_DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    required_columns = config.sensor_features_in_order
    written_files = 0
    try:
        for source_file in sorted(source.rglob("*.csv")):
            if (
                utils.get_transport_mode_from_path(source_file)
                not in config.transport_modes
            ):
                continue
            output_file = target / source_file.relative_to(source)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with (
                source_file.open("r", encoding="utf-8", newline="") as input_stream,
                output_file.open("w", encoding="utf-8", newline="") as output_stream,
            ):
                reader = csv.DictReader(input_stream)
                if reader.fieldnames is None or not set(required_columns).issubset(
                    reader.fieldnames
                ):
                    raise ValueError(f"Missing required feature columns in {source_file}")
                writer = csv.DictWriter(output_stream, fieldnames=required_columns)
                writer.writeheader()
                for row in reader:
                    writer.writerow({column: row[column] for column in required_columns})
            written_files += 1
        if written_files == 0:
            shutil.rmtree(target, ignore_errors=True)
            return False
        _write_provenance(target, source_run, "transformed_data")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    print(f"Reused transformed data from {source_run}")
    return True
