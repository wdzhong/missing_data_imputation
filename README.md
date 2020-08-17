# missing data imputation

## Goal

Use the sparse GPS data collected from NFTA buses to impute the traffic condition for the time points when no GPS data is available.

## Requirements

- Python 3.x
- PyTorch


## Plans
- [ ] Data process
  - [ ] PeMS, used for validation; refer to `References 2` (which uses the same data as [DCRNN](https://github.com/liyaguang/DCRNN)) for more details
    - [ ] Pick up the stations and get their info
      - We can use the selected sensors from DCRNN directly.
    - [ ] Download data
    - [ ] Transfer data to required format
  - [ ] NFTA (Refer to this [repository](https://github.com/wdzhong/NFTA-process-data) for details)
    - [ ] Pick up the road segments and find a way to save them
    - [ ] Map raw GPS data to corresponding road segments and get the traffic condition from GPS
    - [ ] Transfer data to required format
- [ ] Implement existing missing data imputation methods
  - [ ] Tensor decomposition
  - [ ] Spatial-temporal based methods
  - [ ] etc.
- [ ] Develop new deep learning based model
- [ ] Integration with Nittec
  - [ ] Clarify the accepted data format
  - [ ] Data uploading process
  - [ ] Data updating frequency


## Data Process

### PEMS

Some raw PEMS data can be found [here](https://www.dropbox.com/sh/wfb3coid21in0km/AAA3T19RYjSYK1iVP6PTbyana?dl=0). Download them, unzip, and put under folder `data_raw/d[xx]/`, where `xx` is the district ID in two digits.

**Run the following commands under the root directory of this repository.**

1. Select sensors/stations based on some rules, and calculate the distance between each pair of sensors

    > $ python -m scripts.select_sensors

1. Generate the distance matrix among selected sensors, where elements smaller than threshold are set to 0.

    > $ python -m scripts.generate_adj_matrix

1. For each district, select data based on selected sensors and merge them together

    > $ python -m scripts.process_pems

1. Generate samples

    > $ python -m scripts.generate_data_samples --source_data_filename=data_raw/d07/data.npz --output_dir=data/d07

    The train, val, and test files will have the following format
    ```
    x: (number of samples, input length, number of nodes, number of traffic measurements)
    y: (number of samples, prediction length, number of nodes, number of traffic measurements)
    ```


## References
1. KDD'14 Travel Time Estimation of a Path using Sparse Trajectories
2. AAAI'20 GMAN a graph multi-attention network for traffic prediction ([GitHub](https://github.com/zhengchuanpan/GMAN))
