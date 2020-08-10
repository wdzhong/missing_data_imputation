# missing data imputation

## Goal

Use the sparse GPS data collected from NFTA buses to impute the traffic condition for the time points when no GPS data is available.

## Requirements

- Python 3.x
- PyTorch


## Plans
- [ ] Data process
  - [ ] PeMS, used for validation; refer to `References 2` for more details
    - [ ] Pick up the stations and get their info
    - [ ] Download data
    - [ ] Transfer data to required format
  - [ ] NFTA
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

## References
1. KDD'14 Travel Time Estimation of a Path using Sparse Trajectories
2. AAAI'20 GMAN a graph multi-attention network for traffic prediction ([GitHub](https://github.com/zhengchuanpan/GMAN))
