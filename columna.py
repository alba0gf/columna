# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
import copy
from datetime import timedelta, datetime
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing
import numpy as np
import os
from pathlib import Path
import pydicom
from pydicom import dcmread
import pytest
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
from skimage import measure, morphology, segmentation
from skimage.transform import resize
from time import time, sleep
#for masking
from skimage.measure import label,regionprops
from skimage.segmentation import clear_border

# loading libraries 
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Print shapes of data
print('train_df shape:', train_df.shape)
print('test_df shape:', test_df.shape)
# Show first few rows of training data
train_df.head()
test_df.head()

# % Patients with fracture and without fracture
# There are more patients without fractures than with fractures
plt.figure(figsize=(10,7))
ax = sns.countplot(data=train_df, x='patient_overall')
plt.show()

# Which fraction of fractures have each vertebrae? Countplot of each vertebrae

# C7 and C2 have the highest number of fractures
# C3 and C4 have the lowest number of fractures
data_count = pd.melt(train_df, id_vars=['StudyInstanceUID'],value_vars=['C1','C2','C3','C4','C5','C6','C7'], var_name='vertebrae', value_name='fracture') #uppivot vertebrae columns to a single column
data_count.head() 

plt.figure(figsize=(10,7), facecolor='white')
ax = sns.countplot(data=data_count, x='vertebrae', hue='fracture', palette='Set2') # hue as value, vertebrae as variable
for container in ax.containers:
    ax.bar_label(container)
ax.set_title('Count of fractures per vertebrae')
plt.show()
# Percentage of fractures per vertebrae. Percentage of the total fractures that each vertebrae contributes to
#C1              146   10.110803
#C2              285   19.736842
#C3               73    5.055402
#C4              108    7.479224
#C5              162   11.218837
grouped_df = data_count[["vertebrae","fracture"]].groupby('vertebrae').sum()
grouped_df.head()
total_fractures = grouped_df['fracture'].sum()
grouped_df['percentage'] = grouped_df['fracture']/total_fractures * 100

grouped_df.head()

# Correlation heatmap of vertebrae fractures
# There is more correlation of fractures in contiguous vertebrae like C1 and C2, C5 and C6. 
plt.figure(figsize=(10,7))
corr = train_df[['C1','C2','C3','C4','C5','C6','C7']].corr() # get correlation dataframe
ax = sns.heatmap(corr, cbar=True,)
ax.set_title('Correlation heatmap of vertebrae fractures')
plt.show()


# See one dicom image
ds = pydicom.dcmread('data/1.2.826.0.1.3680043.10001_train_images/1.dcm')
ds
plt.imshow(ds.pixel_array)
plt.show()

# List all dicom images for one patient
file_names = os.listdir('/Users/albag/Documents/IA/columna/data/1.2.826.0.1.3680043.10001_train_images')
print(file_names)

#Visualize all slices from list
patient_path = ('/Users/albag/Documents/IA/columna/data/1.2.826.0.1.3680043.10001_train_images')

for file_name in file_names:
   file_path = os.path.join(patient_path, file_name) #dcmread needs file path, not name of the file, thats why you need to create paths for each file name with os.path.join
   ds = pydicom.dcmread(file_path)
   plt.imshow(ds.pixel_array)
   plt.show

   



