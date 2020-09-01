"""
Generate more graphs based on historical data pattern and etc.

calculate the average travel features for each 5 minutes slot (totally 288 slots) for each station,

divide the time slots (288) into several groups. Within each group,
calculate the distance matrix and normalized adjacency matrix among
all sensors.
"""
import os
from datetime import datetime
import pdb

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

from scripts.pems_helper import load_adj_matrix_etc, load_pems_file


def historical_average(district_root, output_file: str):
    """
    Calculate the historical average value for each node during each time slot.

    Parameters
    ----------
    district_root: str
        The folder that contains all raw data files

    output_file: str
        The file to save the results.

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

    sensor_ids, sensor_id_to_idx, _ = load_adj_matrix_etc(adj_matrix_file)
    sensor_set = set(sensor_ids)

    # unit_time_interval = pd.Timedelta('5 min')
    total_slots = 24 * 60 // 5

    print(f"Loading data...")
    data = {}  # {sensor: {time: [[features]]}}
    for data_file in tqdm(sorted(data_files)):
        df = load_pems_file(data_file)
        if sensor_set:
            df = df[df['Station'].isin(sensor_set)]

        df['Timestamp'] = df['Timestamp'].apply(lambda t: datetime.strptime(t, "%m/%d/%Y %H:%M:%S"))

        for _, row in tqdm(df.iterrows()):
            station = str(row['Station'])
            if station not in data:
                data[station] = [[] for _ in range(total_slots)]
            time_slot = row['Timestamp'].hour * 12 + row['Timestamp'].minute // 5
            time_slot = int(time_slot)
            data[station][time_slot].append(row[2:].values)

    print(f"Calculate the average")
    # get the average
    res = [[] for _ in range(len(data.keys()))]
    for sensor, sensor_data in tqdm(data.items()):
        sensor_idx = sensor_id_to_idx[sensor]
        res[sensor_idx] = [0] * total_slots
        for t, d in enumerate(sensor_data):
            res[sensor_idx][t] = np.mean(d, axis=0)

    res = np.array(res)  # shape: (num_of_sensors, num_of_times, num_of_features),
    # where num_of_times here is 288 = 24 * 60 / 5

    np.savez(output_file, data=res)


def calculate_distance_matrix(data):
    """
    Calculate the distance matrix among each pair of rows in the data. Each row of the data stands for values
    from one sensor.

    Parameters
    ----------
    data: numpy.array
        The first dimension is number of sensors.

    Returns
    -------
    distance matrix: numpy.array, with shape (num_sensors, num_sensors)
    """
    num_sensors = data.shape[0]
    data = np.reshape(data, (num_sensors, -1))

    '''
    The distance metric to use. If a string, the distance function can be
    ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’,
    ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’,
    ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
    ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    '''
    dist = cdist(data, data, metric='euclidean')

    return dist


def get_adj_matrix_for_different_time(historical_file, normalized_k=0.1):
    """
    Get adjacency matrix for each of the time blocks.

    Parameters
    ----------
    historical_file: str
        The file that contains the mean values of each feature for each sensor at each time slot (within a day)

    normalized_k: float, default is 0.1
        entries that become lower than normalized_k after normalization are set to zero for sparsity.

    Returns
    -------
    None
    """
    # TODO: any better way to split the time? e.g., split so that the distance
    #  among intervals are maximum?
    time_blocks = [range(6), range(6, 12), range(12, 18), range(18, 24)]
    npz = np.load(historical_file, allow_pickle=True)
    data = npz['data']  # shape: (num_of_sensors, num_of_times, num_of_features)

    for idx, block in enumerate(time_blocks):
        dist = calculate_distance_matrix(data[:, block, :])
        adj_matrix = np.exp(-np.square(dist / np.std(dist)))
        adj_matrix[adj_matrix < normalized_k] = 0
        np.fill_diagonal(adj_matrix, 0)  # Don't forget this

        out_file = os.path.join(os.path.dirname(historical_file), f"adj_matrix_time_{idx}.npz")
        np.savez(out_file, adj_matrix=adj_matrix)


if __name__ == "__main__":
    root = "data_raw/d07"
    normalized_k = 0.1

    historical_average_file = os.path.join(root, "historical_average.npz")
    if not os.path.isfile(historical_average_file):
        historical_average(root, historical_average_file)
    get_adj_matrix_for_different_time(historical_average_file, normalized_k)
