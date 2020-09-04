#!/usr/bin/env python
# coding: utf-8

# In[100]:


import os
import numpy as np
import pandas as pd
from pems_helper import load_adj_matrix_etc, load_pems_file
def get_selected_coordinate():     #Generate the .csv file with 3 columns('sensor_id', 'Latitude', 'Longitude') 
    sensor_ids, sensor_id_to_idx, _ = load_adj_matrix_etc("d07_adj_matrix_etc.npz")
    data = pd.read_csv(str("d07_text_meta_2019_11_09.txt"), sep='\t')
    la = list(data['Latitude'])    #Latitude in meta data
    long = list(data['Longitude']) #Longitude in meta data
    meta_id = list(data["ID"])
    n = len(sensor_ids)
    cvs = [[0 for i in range(3)] for j in range(n)]
    for i in range(n):             #iterate the meta data to get the Latitude and Longitude.
        for j in range(len(meta_id)):
            if int(sensor_ids[i]) == meta_id[j]:
                cvs[i][0] = meta_id[j]
                cvs[i][1] = la[j]
                cvs[i][2] = long[j]
                
    name = ['sensor_id', 'Latitude', 'Longitude']
    re = pd.DataFrame(columns=name, data=cvs)
    re.to_csv("select_cor.csv" ,encoding="utf-8")
get_selected_coordinate()

def get_grid(max_la, min_la, max_long, min_long, n):
    grid = [[[] for i in range(n)] for j in range(n)]
    df = pd.read_csv("select_cor.csv")
    
    horizon = abs(max_long-min_long)
    vertical = abs(max_la-min_la)
    grid_h = horizon/n    #Get the horizontal length of a single grid.
    grid_v = vertical/n   #Get the vertical length of a single grid.
    
    for i in range(n):
        for j in range(n):
            for index, row in df.iterrows():
                ids = row['sensor_id']
                la = row['Latitude']
                long = row['Longitude']
                #If the sensor is in this single grad, add it into the grid matrix.
                if la<(i+1)*grid_v+min_la and la>i*grid_v+min_la and long<grid_h*(j+1)+min_long and long>grid_h*j+min_long:
                    grid[i][j].append(ids)
    print (grid)
    return grid, n
grid, N = get_grid(34.374508, 33.788737, -117.72793, -118.984481, 3)


# In[115]:


def Get_X(n, N, grid): # n is the nth tensor of the train.npz, N is the level of grid
    data = np.load('data.npz')
    data = data["data"]
    sensor_ids, sensor_id_to_idx, _ = load_adj_matrix_etc("d07_adj_matrix_etc.npz")
    list_X = []
    X = []
    tem_grid = [[[] for i in range(N)] for j in range(N)]
    for l in range(12):
        for i in range(N): #replace the entry to speed first and then get the mean of each single grid into list_X
            for j in range(N):
                for k in range(len(grid[i][j])):
                    tem_grid[i][j].append(data[n+l][sensor_id_to_idx[str(int(grid[i][j][k]))]][0])
                list_X.append(np.mean(tem_grid[i][j]))
        
        list_X = []
        X.append(list_X)
    return X

grid_replace(2, N, grid)     

#def gird():
    


# In[ ]:




