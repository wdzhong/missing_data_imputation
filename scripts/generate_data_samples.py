import argparse
import os
import random

import numpy as np

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


def generate_graph_seq2seq_io_data(data, mask, x_offsets, y_offsets, missing_rate: int):
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

    Returns
    -------
    x: (num_samples, input_length, num_nodes, input_dim)
    y: (num_samples, output_length, num_nodes, output_dim)
    mask_x: has the same shape with x
    mask_y: has the same shape with y
    """

    num_times, num_nodes, num_channels = data.shape

    x, y = [], []
    mask_x = []
    mask_y = []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_times - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
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
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    mask_x = np.stack(mask_x, axis=0)
    mask_y = np.stack(mask_y, axis=0)
    return x, y, mask_x, mask_y


def generate_train_val_test(source_data_filename: str, output_dir: str,
                            input_length: int, predict_length: int, missing_rate: int) -> None:
    """
    Generate train, val, and test data.

    Parameters
    ----------
    source_data_filename: str
        The full path of the npz file that contains data and mask.

    output_dir: str
        The folder to hold the results.

    input_length: int
        The (time) length of input data

    predict_length: int
        The (time) length of predict

    missing_rate: int
        The missing rate (percentage)

    Returns
    -------
    None
    """
    npz = np.load(source_data_filename)
    data = npz['data']
    mask = npz['mask']

    # 0 is the latest observed sample
    x_offsets = np.arange(-input_length + 1, 1, 1)

    # Predict the next one hour
    y_offsets = np.arange(1, predict_length + 1, 1)

    # TODO: scale??

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, mask_x, mask_y = generate_graph_seq2seq_io_data(data, mask, x_offsets, y_offsets, missing_rate)

    # x shape: (553, 12, 200, 4), y shape: (553, 12, 200, 4)
    # print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * 0.7)
    num_test = round(num_samples * 0.2)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[: num_train], y[: num_train]
    mask_x_train, mask_y_train = mask_x[: num_train], mask_y[: num_train]

    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    mask_x_val, mask_y_val = (
        mask_x[num_train: num_train + num_val],
        mask_y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    mask_x_test, mask_y_test = mask_x[-num_test:], mask_y[-num_test:]

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    postfix = f"_{input_length}_{predict_length}_{missing_rate}.npz"

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _mask_x, _mask_y = locals()["mask_x_" + cat], locals()["mask_y_" + cat]
        # train x: (387, 12, 200, 4), y: (387, 12, 200, 4)
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, cat + postfix),
            x=_x,
            y=_y,
            mask_x=_mask_x,
            mask_y=_mask_y,
            x_offsets=x_offsets[:, None],
            y_offsets=y_offsets[:, None],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/d07",
                        help="Output directory.")
    parser.add_argument("--source_data_filename", type=str, default="data_raw/d07/data.npz",
                        help="Raw traffic readings.")
    args = parser.parse_args()
    input_lengths = [12]
    predict_lengths = [12]
    missing_rates = [int(10 * i) for i in range(4, 5)]
    for input_length in input_lengths:
        for predict_length in predict_lengths:
            for missing_rate in missing_rates:
                generate_train_val_test(args.source_data_filename, args.output_dir,
                                        input_length, predict_length, missing_rate)
