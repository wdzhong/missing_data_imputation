import os
from datetime import datetime
import pdb
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.pems_helper import load_adj_matrix_etc, load_pems_file

random.seed(0)


def add_missing(mask: np.ndarray, missing_rate: int):
    """
    Add randomly missing to mask (set to 0)

    Parameters
    ----------
    mask: np.ndarray
        The original mask

    missing_rate: int
        The missing rate (percentage) to add to the mask.

    Returns
    -------
    mask with added missing: np.ndarray
        Have the same shape with input mask.
    """
    shape_backup = mask.shape
    mask = mask.flatten()
    # TODO: Deal with the existing missing in the mask
    missing_index = random.sample(range(len(mask)), int(len(mask) * missing_rate / 100))
    mask[missing_index] = 0
    return mask.reshape(shape_backup)


def generate_graph_seq2seq_io_data(data, mask, x_offsets, y_offsets, missing_rate: int, times):
    """
    Generate samples from data

    Parameters
    ----------
    data: np.array
        The array contains data, whose shape is (num_times, num_nodes, num_channels).

    mask: np.array
        It has the same shape with data.
        mask[i, j, k] == 0 means data[i, j, k] is missing.

    x_offsets: np.array
        The time interval offset for x, assuming the current time point is 0.
        e.g., if history length is 12, it is [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]

    y_offsets: np.array
        The time interval offset for prediction.
        e.g., if prediction length is 12, it is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

    missing_rate: int
        The missing rate to add to the mask

    times: List[numpy.datetime]
        A list of numpy.datetime that corresponding to each of the num_times of the data.
        The length of times should be num_times which is the same as data.shape[0].

    Returns
    -------
    x: (num_samples, input_length, num_nodes, input_dim)
    y: (num_samples, output_length, num_nodes, output_dim)
    mask_x: has the same shape with x
    mask_y: has the same shape with y
    times_x: (num_samples, input_length), the timestamp for each sample
    times_y: (num_samples, output_length)
    """

    num_times, _, _ = data.shape

    x, y = [], []
    mask_x = []
    mask_y = []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_times - abs(max(y_offsets)))  # Exclusive
    times_x = []
    times_y = []
    for t in tqdm(range(min_t, max_t)):
        x_t = data[t + x_offsets, ...]  # (12, 200, 4)
        y_t = data[t + y_offsets, ...]  #
        mask_x_t = mask[t + x_offsets, ...]
        mask_y_t = mask[t + y_offsets, ...]
        # add randomly missing to mask
        mask_x_t = add_missing(mask_x_t, missing_rate)
        mask_y_t = add_missing(mask_y_t, missing_rate)

        x.append(x_t)
        y.append(y_t)

        mask_x.append(mask_x_t)
        mask_y.append(mask_y_t)

        times_x.append(times[t + x_offsets])
        times_y.append(times[t + y_offsets])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    mask_x = np.stack(mask_x, axis=0)
    mask_y = np.stack(mask_y, axis=0)
    times_x = np.stack(times_x, axis=0)
    times_y = np.stack(times_y, axis=0)

    return x, y, mask_x, mask_y, times_x, times_y


def process_daily_file(df, num_sensors, sensor_id_to_idx):
    """

    Parameters
    ----------
    df
    num_sensors
    sensor_id_to_idx

    Returns
    -------
    data: (num_times, num_sensors, num_channels)
    mask: (num_times, num_sensors, num_channels)
    times: num_times
    """
    unit_time_interval = pd.Timedelta('5 min')
    df['Timestamp'] = df['Timestamp'].apply(lambda t: datetime.strptime(t, "%m/%d/%Y %H:%M:%S"))
    times = df['Timestamp'].values
    times = list(set(times))
    # times = [datetime.strptime(t, "%m/%d/%Y %H:%M:%S") for t in times]
    times = sorted(times)

    start_time = times[0]
    end_time = times[-1]
    time_diff = end_time - start_time
    num_times = time_diff // unit_time_interval + 1

    # They should be 288 for a day
    if num_times != len(times):
        print(f"there might be some times missing.")
        pdb.set_trace()

    num_channels = len(df.columns) - 2

    data = np.zeros((num_times, num_sensors, num_channels))
    mask = np.zeros_like(data)

    print(f"Shape of DataFrame: {df.shape}")
    for _, row in tqdm(df.iterrows()):
        time_slot = (row['Timestamp'] - start_time) // unit_time_interval
        sensor_idx = sensor_id_to_idx[str(row['Station'])]
        data[time_slot, sensor_idx, :] = row[2:].values
        mask[time_slot, sensor_idx, :] = 1

        # TODO: use a better way to check possible missing value
        for i, val in enumerate(row[2:].values):
            if not val or val == 0.0:
                mask[time_slot, sensor_idx, i] = 2
                print(f"find missing! {row}")

    return data, mask, np.array(times)


def generate_train_val_test_from_raw(district_root: str, output_dir: str,
                                     input_length: int, predict_length: int,
                                     missing_rate: int):
    adj_matrix_file = None
    data_files = []
    for f in os.listdir(district_root):
        if "station_5min" in f:
            data_files.append(os.path.join(district_root, f))
        elif "adj_mat" in f:
            adj_matrix_file = os.path.join(district_root, f)

    sensor_ids, sensor_id_to_idx, _ = load_adj_matrix_etc(adj_matrix_file)

    # 0 is the latest observed sample
    x_offsets = np.arange(-input_length + 1, 1, 1)

    # Predict the next one hour
    y_offsets = np.arange(1, predict_length + 1, 1)

    all_x = []
    all_y = []
    all_mask_x = []
    all_mask_y = []
    all_times_x = []
    all_times_y = []

    sensor_set = set(sensor_ids)
    for data_file in tqdm(sorted(data_files)):
        pems = load_pems_file(data_file)
        pems = pems[pems['Station'].isin(sensor_set)]

        data, mask, times = process_daily_file(pems, len(sensor_ids), sensor_id_to_idx)

        # x: (num_samples, input_length, num_nodes, input_dim)
        # y: (num_samples, output_length, num_nodes, output_dim)
        # mask_x: has the same shape with x
        # mask_y: has the same shape with y
        # times_x: (num_samples, input_length), the timestamp for each sample
        # times_y: (num_samples, output_length)
        x, y, mask_x, mask_y, times_x, times_y = generate_graph_seq2seq_io_data(data, mask, x_offsets, y_offsets,
                                                                                missing_rate, times)

        all_x.append(x)
        all_y.append(y)
        all_mask_x.append(mask_x)
        all_mask_y.append(mask_y)
        all_times_x.append(times_x)
        all_times_y.append(times_y)

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    mask_x = np.concatenate(all_mask_x, axis=0)
    mask_y = np.concatenate(all_mask_y, axis=0)
    times_x = np.concatenate(all_times_x, axis=0)
    times_y = np.concatenate(all_times_y, axis=0)

    # x shape: (x, 12, 200, 4), y shape: (x, 12, 200, 4)
    print(f"x shape: {x.shape}, y shape: {y.shape}, mask_x shape: {mask_x.shape}, times_x shape: {times_x.shape}")

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * 0.7)
    num_test = round(num_samples * 0.2)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[: num_train], y[: num_train]
    mask_x_train, mask_y_train = mask_x[: num_train], mask_y[: num_train]
    times_x_train, times_y_train = times_x[: num_train], times_y[: num_train]

    # val
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    mask_x_val, mask_y_val = mask_x[num_train: num_train + num_val], mask_y[num_train: num_train + num_val]
    times_x_val, times_y_val = times_x[num_train: num_train + num_val], times_y[num_train: num_train + num_val]

    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    mask_x_test, mask_y_test = mask_x[-num_test:], mask_y[-num_test:]
    times_x_test, times_y_test = times_x[-num_test:], times_y[-num_test:]

    # Normalization
    mean = np.mean(x_train, axis=0, keepdims=True)  # use np.nanmean() in case of missing
    std = np.std(x_train, axis=0, keepdims=True) + 1e-6

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    postfix = f"_{input_length}_{predict_length}_{missing_rate}.npz"

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _mask_x, _mask_y = locals()["mask_x_" + cat], locals()["mask_y_" + cat]
        _times_x, _times_y = locals()["times_x_" + cat], locals()["times_y_" + cat]
        # train x: (x, 12, 200, 4), y: (x, 12, 200, 4)
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, cat + postfix),
            x=_x,
            y=_y,
            mask_x=_mask_x,
            mask_y=_mask_y,
            # x_offsets=x_offsets[:, None],
            # y_offsets=y_offsets[:, None],
            times_x=_times_x,
            times_y=_times_y
        )


if __name__ == "__main__":
    # TODO: for each district
    district_root = "data_raw/d07"
    output_dir = "data/d07"

    input_lengths = [12]
    predict_lengths = [12]
    missing_rates = [int(10 * i) for i in range(4, 5)]
    for input_length in input_lengths:
        for predict_length in predict_lengths:
            for missing_rate in missing_rates:
                generate_train_val_test_from_raw(district_root, output_dir,
                                                 input_length, predict_length, missing_rate)
