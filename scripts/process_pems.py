"""
For each district, select data for certain sensors and merge the data
into a single file. Then get the final data with shape (number_of_time, number_of_node, number_of_channel)
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.pems_helper import load_adj_matrix_etc, load_pems_file


def select_and_merge_district(district_root: str, output_file: str) -> None:
    """
    Select data under the district root based on selected sensors, and merge the data together.

    Parameters
    ----------
    district_root: str
        The path of the district that contains the data and sensor information

    output_file: str
        The output file.

    Returns
    -------
    None
    """
    adj_matrix_file = None
    data_files = []
    for f in os.listdir(district_root):
        if "station_5min" in f:
            data_files.append(os.path.join(district_root, f))
        elif "adj_mat" in f:
            adj_matrix_file = os.path.join(district_root, f)

    sensor_ids, _, _ = load_adj_matrix_etc(adj_matrix_file)

    sensor_set = set(sensor_ids)
    pems_list = []
    for data_file in tqdm(data_files):
        pems = load_pems_file(data_file)
        pems = pems[pems['Station'].isin(sensor_set)]
        pems_list.append(pems)

    pems = pd.concat(pems_list, ignore_index=True)

    pems.to_csv(output_file, index=False)


def get_final_data(merged_data_file: str, output_file: str, adj_matrix_file: str):
    """
    from the merged data, get the final data with shape (number_of_time, number_of_node, number_of_channel)

    Parameters
    ----------
    merged_data_file: str
        The merged data file which contains a DataFrame whose columns are ['Timestamp', 'Station', etc]

    output_file: str
        The output file

    adj_matrix_file:
        The file that contains the sensor IDs, sensor ID to index, and adj matrix.

    Returns
    -------
    None
    """

    unit_time_interval = pd.Timedelta('5 min')

    date_parser = lambda x: datetime.strptime(x, "%m/%d/%Y %H:%M:%S")
    # load the merged data
    # columns: 'Timestamp', 'Station', etc
    df = pd.read_csv(merged_data_file, parse_dates=['Timestamp'], date_parser=date_parser)
    times = df['Timestamp'].values
    start_time = min(times)
    end_time = max(times)
    time_diff = end_time - start_time
    num_times = time_diff // unit_time_interval + 1  # each day should be 288 for 5 minutes interval
    # print(f"The number of time stamps is: {num_times}")

    num_channels = len(df.columns) - 2

    sensor_ids, sensor_id_to_idx, _ = load_adj_matrix_etc(adj_matrix_file)

    data = np.zeros((num_times, len(sensor_ids), num_channels))
    mask = np.ones_like(data)

    for _, row in tqdm(df.iterrows()):
        time_slot = (row['Timestamp'] - start_time) // unit_time_interval
        sensor_idx = sensor_id_to_idx[str(row['Station'])]
        data[time_slot, sensor_idx, :] = row[2:].values

        # TODO: use a better way to check possible missing value
        for i, val in enumerate(row[2:].values):
            if not val or val == 0.0:
                mask[time_slot, sensor_idx, i] = 0

    np.savez(output_file, data=data, mask=mask, start_time=start_time, end_time=end_time)


if __name__ == "__main__":
    # TODO: for each district
    district_root = "data_raw/d07"
    merged_file = os.path.join(district_root, "selected_merged.csv")
    if not os.path.isfile(merged_file):
        select_and_merge_district(district_root, merged_file)
    final_data_file = os.path.join(district_root, "data.npz")
    adj_matrix_etc_file = os.path.join(district_root, "d07_adj_matrix_etc.npz")
    if not os.path.isfile(final_data_file):
        get_final_data(merged_file, final_data_file, adj_matrix_etc_file)
