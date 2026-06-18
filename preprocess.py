import argparse
import re
from model_config import ModelConfig
from dataset_reuse import (
    materialize_compatible_cleaned,
    materialize_compatible_transformed,
)

import pandas as pd
import constants
import os
import csv
from tqdm import tqdm

import utils

class Preprocess:
    config: ModelConfig

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_yaml()
        self.config.prepare_run_directory()

    def extract_raw_data(self) -> None:
        """
        Extract raw data from the downloaded tar.gz file.
        """
        raw_data_path = self.config.raw_data_path

        if not os.path.isfile(raw_data_path):
            print(f"Raw data file not found at {raw_data_path}. Please download it first.")
            return

        print(f"Extracting raw data from {raw_data_path}...")

        import tarfile

        with tarfile.open(raw_data_path, "r:gz") as tar:
            tar.extractall(path=constants.DATA_PATH)

        print(f"Extraction complete. Extracted files are located at {self.config.raw_data_extracted_path}.")

    def clean_file(self, input_path: os.PathLike, output_path: os.PathLike) -> None:
        """
        Clean a raw data file by:
        - Removing rows where the second column (sensor type) does not end with a valid sensor type
        - Removing rows where the first column (time) is empty or does not contain a valid timestamp
        - Stripping "android.sensor." prefix from the sensor type and keeping only the sensor name
        - Removing any non-alphanumeric characters from the time and sensor type fields
        """

        if (os.path.getsize(input_path) == 0):
            return

        has_output: bool = False
        with open(input_path, "r", encoding='utf-8', errors='replace') as input_file, \
            open(output_path, "w", encoding="utf-8", newline='') as output_file:

            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            for row in reader:
                if len(row) == 0:
                    continue
                else:
                    row = [re.sub(r'[^a-zA-Z0-9._-]', '', val) for val in row]
                    remove: bool = False
                    time_str: str = row[constants.RawDataFieldLocation.TIME]
                    sensor_type_str: str = row[constants.RawDataFieldLocation.SENSOR_TYPE]
                    if not time_str:
                        remove = True
                    elif not any(sensor_type_str.endswith('.' + sensor) for sensor in self.config.sensors):
                        remove = True
                    time_values = re.findall(r'\d+', time_str)
                    if time_values and not remove:
                        row[constants.RawDataFieldLocation.TIME] = time_values[0]
                        row[constants.RawDataFieldLocation.SENSOR_TYPE] = sensor_type_str.split('.')[-1]
                        writer.writerow(row)
                        has_output = True
        if not has_output:
            os.remove(output_path)

    def clean_files(self) -> None:
        """
        - Clean raw data files in the extracted folder and save cleaned versions in a new folder.
        - Select files to process based on transport modes
        """

        if materialize_compatible_cleaned(self.config):
            return
        os.makedirs(self.config.cleaned_data_path, exist_ok=True)

        processing_files = list(self.config.raw_data_extracted_path.rglob("*.csv"))
        print(f"Total files to process: {len(processing_files)}")

        iter =  tqdm(processing_files, desc="Cleaning files")
        for input_path in processing_files:
            iter.update()
            print(iter)
            transport_mode = utils.get_transport_mode_from_path(input_path)
            if transport_mode.lower() not in self.config.transport_modes:
                continue

            output_file_path = self.config.cleaned_data_path / input_path.relative_to(self.config.raw_data_extracted_path)
            os.makedirs(output_file_path.parent, exist_ok=True)

            self.clean_file(input_path, output_file_path)

        print(f"Cleaning complete. Cleaned files are located at {self.config.cleaned_data_path}.")

    def transform_file(self, input_path: os.PathLike, output_path: os.PathLike) -> None:
        """
        Transform a cleaned data file by:
        - Segmenting the data into overlapping windows of a specified size and step.
        - For each window, compute features (mean, std, min, max) for each sensor type.
        - For vector sensors, compute t he magnitude of the vector and use that for feature computation instead of individual components.
        - Save the transformed features in a new CSV file with a header indicating the sensor and feature type.
        """

        has_output: bool = False

        with open(output_path, "w", encoding="utf-8", newline='') as output_file:
            writer = csv.writer(output_file)

            window_size_ms = int(self.config.window_size_seconds * 1000)
            window_next_step_ms = int(self.config.window_next_step_seconds * 1000)

            df = pd.read_csv(
                input_path,
                names=['time', 'sensor_type', 'val1', 'val2', 'val3', 'val4', 'val5']
            )

            curr_window_start: int = 0
            last_time: int = df['time'].iloc[-1]

            writer.writerow(self.config.sensor_features_in_order)

            while True:
                curr_window_end = curr_window_start + window_size_ms
                if curr_window_end > last_time:
                    break
                window: pd.DataFrame = df[(df['time'] >= curr_window_start) & (df['time'] < curr_window_end)]
                curr_window_start += window_next_step_ms
                all_sensors_in_window = window['sensor_type'].unique()
                # If not all sensors are present in the window, skip it. This is a design choice to ensure that we only compute features for windows that have data from all sensors, which may be important for certain machine learning models that expect a complete feature set. However, this also means that we will lose data from windows that are missing one or more sensors, which could potentially reduce the amount of training data available. Depending on the specific use case and the importance of having a complete feature set versus having more training data, this decision may need to be revisited.
                if (all_sensors_in_window.size < len(self.config.sensors)):
                    continue
                transformed_features = {}
                for sensor in self.config.sensors:
                    transformed_values: pd.Series = pd.Series()
                    sensor_window: pd.DataFrame = window[window['sensor_type'] == sensor]
                    transformed_values = (sensor_window[['val1', 'val2', 'val3']].pow(2).sum(axis=1)).pow(0.5)
                    transformed_features[f"{sensor}#mean"] = transformed_values.mean()
                    transformed_features[f"{sensor}#std"] = transformed_values.std()
                    transformed_features[f"{sensor}#min"] = transformed_values.min()
                    transformed_features[f"{sensor}#max"] = transformed_values.max()
                writer.writerow([transformed_features[sensor_feature] for sensor_feature in self.config.sensor_features_in_order])
                has_output = True

        if not has_output:
            os.remove(output_path)

    def transform_files(self) -> None:
        """
        Transform cleaned data files into a format suitable for machine learning models.
        """
        if materialize_compatible_transformed(self.config):
            return
        materialize_compatible_cleaned(self.config)
        os.makedirs(self.config.transformed_data_path, exist_ok=True)

        processing_files = list(self.config.cleaned_data_path.rglob("*.csv"))
        print(f"Total files to process for transformation: {len(processing_files)}")


        iter =  tqdm(processing_files, desc="Transforming files")
        for input_path in processing_files:
            iter.update()
            print(iter)
            output_file_path = self.config.transformed_data_path / input_path.relative_to(self.config.cleaned_data_path)
            os.makedirs(output_file_path.parent, exist_ok=True)
            self.transform_file(input_path, output_file_path)

        print(f"Transformation complete. Transformed files are located at {self.config.transformed_data_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all preprocessing steps (extract, clean, transform)"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract raw data files"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean extracted raw data files"
    )
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Transform cleaned data files"
    )
    args = parser.parse_args()

    p = Preprocess()

    if args.extract:
        p.extract_raw_data()
    if args.all and not materialize_compatible_transformed(p.config):
        if not materialize_compatible_cleaned(p.config):
            p.extract_raw_data()
            p.clean_files()
        p.transform_files()
    if args.clean:
        p.clean_files()
    if args.transform:
        p.transform_files()
