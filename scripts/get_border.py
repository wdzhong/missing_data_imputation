#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from pems_helper import load_adj_matrix_etc, load_pems_file
sensor_ids, sensor_id_to_idx, _ = load_adj_matrix_etc("d07_adj_matrix_etc.npz")

# Find the border of the selected sensor according to Latitude and Longtitude.
def border(sensor_ids):
    data = pd.read_csv(str("d07_text_meta_2019_11_09.txt"), sep='\t')
    data = data.drop(data.columns[-5:], axis=1)
    la = list(data['Latitude'])    #Latitude in meta data
    long = list(data['Longitude']) #Longitude in meta data
    meta_id = list(data["ID"])     #sensor_ids in meta data
    select_La = []
    select_Lo = []
    for j in range(len(meta_id)):
        for i in range(len(sensor_ids)):
            print (int(sensor_ids[i]),int(meta_id[j]))
            if int(sensor_ids[i]) == meta_id[j]:
                select_La.append(la[j])
                select_Lo.append(long[j])
                print ("this is la", select_La)
                print ("this is lO", select_Lo)
                break

    print (len(select_La))
    print (len(select_Lo))
    #print (select_La)
    #print (select_Lo)
    print ("Max Latitude",max(select_La))
    print ("Min Latitude",min(select_La))
    print ("Max Longitude",max(select_Lo))
    print ("Min Longitude",min(select_Lo))
    return (max(select_La), min(select_La), max(select_Lo), min(select_Lo))
print (border(sensor_ids))


# In[ ]:




