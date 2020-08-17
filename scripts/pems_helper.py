from copy import deepcopy
import os

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm


# 'Samples' and '% Observed' are always 0
columns = ['Timestamp', 'Station', 'District', 'Freeway #', 'Direction', 'Lane Type', 'Station Length', 'Samples',
           '% Observed', 'Total Flow', 'Ave Occupancy', 'Avg Speed']

# 'Lane N Samples' and 'Lane N Observed' are always 0
lane_columns = ['Lane N Samples', 'Lane N Flow', 'Lane N Avg Occ', 'Lane N Avg Speed', 'Lane N Observed']


def get_all_columns(num_cols: int):
    """
    Get/Generate the column names

    Parameters
    ----------
    num_cols: int
        The number of columns of the loaded file

    Returns
    -------
    column names: List[str]
        The name for all columns

    number of lanes: int
        The maximum number of lanes that the data file can hold

    number of columns per lane: int
        The number of columns for each lane in the file
    """
    global columns, lane_columns
    cols = deepcopy(columns)
    num_lanes = (num_cols - len(columns)) // len(lane_columns)

    assert num_lanes * len(lane_columns) + len(columns) == num_cols, "The number of columns is NOT expected"

    for lane in range(1, num_lanes + 1):
        cols.extend([col.replace('N', str(lane)) for col in lane_columns])

    return cols, num_lanes, len(lane_columns)


def load_pems_file(filename: str, num_lanes_to_keep=3) -> pd.DataFrame:
    """
    Load raw PeMS data file, drop some columns, and only keep time, ID, and measures.

    Parameters
    ----------
    filename: str
        The full path of the pems data file.

    num_lanes_to_keep: int, default is 3
        Only keep the measurements for the first few lanes.

    Returns
    -------
    DataFrame: the DataFrame that contains the loaded data.
    """
    data = pd.read_csv(filename, header=None)

    # get the real column names
    cols = data.columns.values
    all_columns, num_lanes, count_per_lane = get_all_columns(len(cols))
    data.columns = all_columns

    # pdb.set_trace()

    cols_to_drop = ['District', 'Freeway #', 'Direction', 'Lane Type', 'Station Length', 'Samples', '% Observed',
                    'Total Flow', 'Ave Occupancy']
    # only keep columns from limited number of lanes
    # only keep the ave speed column (the 4th column in each lane group) from each of the remaining lanes
    # TODO: choose other measurements
    for idx, col in enumerate(data.columns[len(cols) - num_lanes * count_per_lane:]):
        if idx % count_per_lane == 3 and idx < num_lanes_to_keep * count_per_lane:
            continue
        cols_to_drop.append(col)

    data = data.drop(cols_to_drop, axis=1)

    return data


def calculate_sensor_distance(meta_file, save=False, output_file=None):
    """
    Calculate the distance between any two sensors.

    Parameters
    ----------
    meta_file: str
        The full path of the file that contains the meta information of the sensors.

    save: bool, default is False
        If True, then save the distance into a file

    output_file: str, default is None
        The default output file to store the result

    Returns
    -------
    2D array: [[from, to, distance]]
    """
    data = pd.read_csv(meta_file)
    sensor_geo = {}
    for _, row in data.iterrows():
        sensor_geo[row['ID']] = (row['Latitude'], row['Longitude'])

    sensors = data['ID'].values
    dist = []
    for i in tqdm(range(len(sensors))):
        geo_one = sensor_geo[sensors[i]]
        for j in range(i + 1, len(sensors)):
            # TODO: other ways to calculate the distance
            # the distance might not be symmetric if using the driving distance
            cur_dist = geodesic(geo_one, sensor_geo[sensors[j]]).meters
            cur_dist = round(cur_dist, 1)
            dist.append([sensors[i], sensors[j], cur_dist])
            # dist.append([sensors[j], sensors[i], cur_dist])

    if save:
        if not output_file:
            output_file = str(meta_file).split('.')[0] + "_dist_matrix.csv"

        pd.DataFrame(dist, columns=['from', 'to', 'dist']).to_csv(output_file, index=False)

    return dist


def load_adj_matrix_etc(filename: str):
    """
    Load (filtered) sensors, including sensor IDs, map from sensor
    ID to index, and normalized adjacency matrix of these sensors,

    Parameters
    -----------
    filename: str
        The path of the npz file.

    Returns
    --------
    sensor IDs: list
    sensor ID to index: dictionary, {sensor ID: index}
    normalized adjacency matrix: 2D numpy array, shape (num_of_sensors, num_of_sensors)
    """
    if not os.path.isfile(filename):
        print(f"Adjacency matrix file {filename} does NOT exist.")
        exit(0)

    data = np.load(filename, allow_pickle=True)
    sensor_ids = data['sensor_ids']
    # https://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez
    sensor_id_to_idx = data['sensor_id_to_idx'][()]
    adj_matrix = data['adj_matrix']
    return sensor_ids, sensor_id_to_idx, adj_matrix
