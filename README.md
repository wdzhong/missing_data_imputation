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

---
#### Existing Downlaoded Data

Some raw PEMS data can be found [here](https://www.dropbox.com/sh/wfb3coid21in0km/AAA3T19RYjSYK1iVP6PTbyana?dl=0). Download them, unzip, and put under folder `data_raw/d[xx]/`, where `xx` is the district ID in two digits.

---
#### Steps to download more raw data and sensor metadata from [official website](http://pems.dot.ca.gov)

   1. Register if not yet (it might take some time for the new account to be approved) and sign in
   2. Download by following these steps
      1. Click on **Data Clearinghouse** at the bottom left of the homepage
      2. To download **data**,
          1. on the top of the page, in the dropdown list of
             - `Type`: select **Station 5-Minute**
             - `District`: select target district, e.g., **District 7**
          2. Click **Submit** button
          3. In the table below the **Submit** button, click on the cell in the year and Month table
          4. Download data from the **Available Files** table
      3. To downlaod **metadata file**, choose **Station Metadata** in the `Type` dropdown list, and then select the desired `District`.
   3. Put data and meta file under folder `data_raw/d[xx]/`, where `xx` is the district ID in two digits.


---
#### Steps to process PEMS data and generate samples

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
    mask_x: has the same shape with x; mask_x[idx] == 0 means data missing at certain point.
    mask_y: has the same shape with y.
    ```

    The names of `train`, `val`, and `test` data files are in format
    `{mode}_{input length}_{predict length}_{missing rate}.npz`, where
     `mode` is train, val or test, `input length` is the length of input sequence in terms of time interval,
     `predict length` is the length of prediction sequence, and `missing rate` is the missing rate in data samples.

## References
1. KDD'14 Travel Time Estimation of a Path using Sparse Trajectories
2. AAAI'20 GMAN a graph multi-attention network for traffic prediction ([GitHub](https://github.com/zhengchuanpan/GMAN))
