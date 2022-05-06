from spectogram import spec_to_image
import numpy as np 
import pandas as pd

def extract_data(df,i,batch_size,path):
    x1 = np.zeros((batch_size,165,626,3))
    x2 = np.zeros((batch_size,165,626,3))
    x3 = np.zeros((batch_size,165,626,3))
    y = np.zeros((batch_size,165,626,3))
    labels = np.zeros((batch_size,1))

    for j in range(batch_size):
        af = df.iloc[i+j]["X"].split(',')
        x1[j] = spec_to_image(path+af[0]+".flac")
        x2[j] = spec_to_image(path+af[1]+".flac")
        x3[j] = spec_to_image(path+af[2]+".flac")
        y[j] = spec_to_image(path+df.iloc[i+j]["Y"]+".flac")
        labels[j] = df.iloc[i+j]["label"]

    x = [x1,x2,x3]
  
    return x,y,labels,i