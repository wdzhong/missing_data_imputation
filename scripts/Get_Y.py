#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import numpy as np
import pandas as pd
from pems_helper import load_adj_matrix_etc, load_pems_file
def get_Y():
    data = pd.read_csv(str("d07_text_meta_2019_11_09.txt"), sep='\t')
    sensor_ids, sensor_id_to_idx, _ = load_adj_matrix_etc("d07_adj_matrix_etc.npz")
    Length = data["Length"]
    Lanes = data["Lanes"]
    Y = []
    meta_id = list(map(int, data["ID"]))
    sensor_ids = list(map(int, sensor_ids))
    for i in range(len(meta_id)):
        for j in range(len(sensor_ids)):
            if meta_id[i] == sensor_ids[j]:
                Y.append([Length[i], Lanes[i]])
    return Y
   


# In[ ]:




