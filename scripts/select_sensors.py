"""
Select some stations based on requirements. This should be done before
processing raw data.
"""
import os
from pathlib import Path
import argparse

import pandas as pd

from scripts.pems_helper import load_pems_file, calculate_sensor_distance


def select_sensors(meta_file, output_file, min_num_lanes=None, station_types=None, use_data_file=False, data_file=None):
    """
    Pick up a subset of stations based on conditions, e.g., the number of lanes, the type of the station

    Parameters
    ----------
    meta_file: str
        The full path of meta file that contains the information about stations.

    min_num_lanes: int, default=None
        The minimum number of lances that each selected station must at least have

    station_types: list, default=None
        The list of types of selected stations

    use_data_file: bool, default=False
        If True, then use some data file to help filter sensors.

    data_file: str, default=None
        The data file to be used.

    Returns
    -------
    station_df: DataFrame for selected stations
    """

    data = pd.read_csv(str(meta_file), sep='\t')

    # the last 5 columns are [Name, User_ID_1, User_ID_2, User_ID_3, User_ID_4]
    data = data.drop(data.columns[-5:], axis=1)

    if min_num_lanes is not None:
        data = data[data['Lanes'] >= min_num_lanes]

    if station_types is not None:
        assert type(station_types) is list, f"The station_types {station_types} has type {type(station_types)}. Expected 'list'"
        data = data[data['Type'].isin(station_types)]

    # TODO: some stations have 0s for all measurements (broken sensors?)
    # use data to find the sensors without data
    if use_data_file:
        if not data_file:
            data_root = Path(meta_file).parent
            for f in os.listdir(str(data_root)):
                if "station_5min" not in f:
                    continue
                data_file = str(data_root / f)
                break

        if data_file:
            pems = load_pems_file(data_file)
            # the NaN values mostly result from the malfunction sensors
            # since the missing values are consecutive
            # so we just find the sensors with NaN values and then
            # remove all records from these sensors
            # TODO: we should consider keeping sensors that only have limited number of missing values
            indices_with_nan = pems.isnull().any(axis=1)
            stations_with_nan = pems['Station'].loc[indices_with_nan == True]
            stations_with_nan_set = set(stations_with_nan.values)

            stations_wt_nan = set(pems['Station'].values).difference(stations_with_nan_set)

            selected_stations_IDs = data['ID'].values
            # remove these stations from the selected_stations_IDs
            selected_stations_IDs = set(selected_stations_IDs).intersection(stations_wt_nan)

            # keep rows from the filtered/selected stations only
            data = data[data['ID'].isin(selected_stations_IDs)]
        else:
            print(f"Warning: There is NO data file.")

    # TODO: remove sensors that are too close to each other

    data.to_csv(output_file, index=False)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", type=str, default="data_raw/d07/d07_text_meta_2019_11_09.txt",
                        help="The file that contains sensor meta information")
    parser.add_argument("--output_file", type=str, default="data_raw/d07/d07_meta_selected.csv",
                        help="The file to save the selected sensor")
    parser.add_argument("--overwrite", type=bool, default=False,
                        help="Redo and overwrite the existing file if it is set to True.")
    args = parser.parse_args()
    if not os.path.isfile(args.output_file) or args.overwrite:
        select_sensors(args.meta_file, args.output_file, min_num_lanes=3, station_types=['ML'], use_data_file=True)

    calculate_sensor_distance(args.output_file, save=True)
