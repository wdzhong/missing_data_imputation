import os
import argparse
import random

import numpy as np
import pandas as pd

random.seed(0)
np.random.seed(0)


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    Generate adjacency matrix from sensor meta info.

    Parameters
    ----------
    distance_df: pd.DataFrame
        data frame with three columns: [from, to, distance].

    sensor_ids: List
        list of sensor ids.

    normalized_k: float, default is 0.1
        entries that become lower than normalized_k after normalization are set to zero for sparsity.

    Returns
    -------
    sensor_ids: List
    sensor_id_to_idx: dict
    adj_matrix: 2D array
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    # Builds sensor id to index map.
    sensor_id_to_idx = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_idx[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_idx or row[1] not in sensor_id_to_idx:
            continue
        dist_mx[sensor_id_to_idx[row[0]], sensor_id_to_idx[row[1]]] = row[2]

        # TODO: not necessary symmetric
        dist_mx[sensor_id_to_idx[row[1]], sensor_id_to_idx[row[0]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_matrix = np.exp(-np.square(dist_mx / std))

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_matrix[adj_matrix < normalized_k] = 0
    np.fill_diagonal(adj_matrix, 0)

    return sensor_ids, sensor_id_to_idx, adj_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default="",
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data_raw/d07/d07_meta_selected_dist_matrix.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_adj_filename', type=str, default='data_raw/d07/d07_adj_matrix_etc.npz',
                        help='Path of the output file.')
    args = parser.parse_args()

    sensor_ids = None
    if args.sensor_ids_filename and os.path.isfile(args.sensor_ids_filename):
        with open(args.sensor_ids_filename) as f:
            sensor_ids = f.read().strip().split(',')

    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    if not sensor_ids:
        sensor_ids = distance_df['from'].values

        sensor_ids = set(sensor_ids)
        # TODO: choose a subset first
        sensor_ids = random.sample(list(sensor_ids), 200)
        sensor_ids = sorted(sensor_ids)

    normalized_k = args.normalized_k
    _, sensor_id_to_idx, adj_matrix = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)

    np.savez(args.output_adj_filename, sensor_ids=sensor_ids, sensor_id_to_idx=sensor_id_to_idx, adj_matrix=adj_matrix)
