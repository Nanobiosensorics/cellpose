import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from cellpose.data_loader import CellDataset, split_dataset
from cellpose import models

use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')

import os
import pandas as pd
import numpy as np
import re
import random

def get_image_ids(path):
    pattern = r'^(\d+[_-]\d+(?:[_-]\d+)?)'
    if os.path.isdir(path):
        ids = []
        for filename in os.listdir(path):
            match = re.match(pattern, filename)
            if match is not None:
                ids.append(match.group(1))
        ids = sorted(list(set(ids)))
    else:
        split = path.split(os.sep)
        if len(split) <= 1:
            split = path.split('/')
        parts, name = split[:-1], split[-1]
        name = re.match(pattern, name)
        if name is None:
            raise Exception('Experiment id not found in filename!')
        ids = [name.group(1)]
        path = os.path.join(*parts)
        if os.sep == '/':
            path = '/' + path
    return path, ids

def get_tif(path, idx):
    files = [name for name in os.listdir(path) if name.endswith('5.tif') and idx in name]
    return None if len(files) == 0 else files[0]

def get_smlm_file(path, idx):
    files = [name for name in os.listdir(path) if name.endswith('.txt') and idx in name]
    return None if len(files) == 0 else files[0]

def get_smlm_aligned_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and 'ClusterData' in name]
    return None if len(files) == 0 else files[0]

def get_srrf_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and 'segmResultsPRED' in name]
    return None if len(files) == 0 else files[0]

def get_esrrf_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and name.endswith('_esrrf.tif')]
    return None if len(files) == 0 else files[0]

def get_seg_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and 'seg.npy' in name]
    return None if len(files) == 0 else files[0]

def get_raw_srrf_file(path, idx):
    files = [name for name in os.listdir(path) if idx in name and name.endswith('.ome.tif')]
    return None if len(files) == 0 else files[0]

def get_sample(path, idx):
    if os.path.isfile(path):
        path = os.path.join(*path.split('/')[:-1])
    return { 
            'img' : os.path.join(path, f) if (f := get_tif(path, idx)) is not None else None, 
            'smlm': os.path.join(path, f) if (f := get_smlm_file(path, idx)) is not None else None, 
            'smlm_aligned': os.path.join(path, f) if (f := get_smlm_aligned_file(path, idx)) is not None else None, 
            'srrf': os.path.join(path, f) if (f := get_srrf_file(path, idx)) is not None else None,
            'esrrf':os.path.join(path, f) if (f := get_esrrf_file(path, idx)) is not None else None,
            'raw-srrf': os.path.join(path, f) if (f := get_raw_srrf_file(path, idx)) is not None else None,
            'seg': os.path.join(path, f) if (f := get_seg_file(path, idx)) is not None else None
            }


# In[3]:


data_path = '/home/balint/projects/nc_data/240229-SRRF-SMLM-data-WF-and-TIRF-imaging/1.1.1-STORM-PooledPlasma-ch640/'

diam_mean=50
initial_model = "scratch" #@param ['cyto','nuclei','tissuenet','livecell','cyto2','CP','CPx','TN1','TN2','TN3','LC1','LC2','LC3','LC4','scratch']
model_dir = os.path.join(data_path, 'results')
model_name = f"ev_segment_{diam_mean}.pt" #@param {type:"string"}
n_epochs =  1000 #@param {type:"number"}
learning_rate = 0.1 
weight_decay = 0.0001
batch_size = 16
chan = 1
chan2 = 2

data_path, ids = get_image_ids(data_path)


# In[4]:


paths = []

for idx in ids:
    sample = get_sample(data_path, idx)
    paths.append(sample['esrrf'])


# In[5]:


val_percent = 0.2

random.shuffle(paths)

n_val = int(len(paths) * val_percent)
n_train = len(paths) - n_val
train_paths, test_paths = paths[:n_train], paths[n_train:]


# In[6]:


# set channels
channels = [chan, chan2]
# train_paths, test_paths = split_dataset(paths, 0.2)
train_ds = CellDataset(paths=train_paths, generate_flows=True, mask_filter='_masks', channels=channels)
test_ds = CellDataset(paths=test_paths, generate_flows=True, mask_filter='_masks', channels=channels)


# In[8]:


# start logger (to see training across epochs)
logger = io.logger_setup()

# DEFINE CELLPOSE MODEL (without size model)
model = models.CellposeModel(gpu=use_GPU, model_type=None, diam_mean=diam_mean)


new_model_path = model.train(train_ds, test_dataset=test_ds,
                             save_path=model_dir, 
                              n_epochs=n_epochs,
                              learning_rate=learning_rate, 
                              weight_decay=weight_decay, 
                              nimg_per_epoch=5,
                              model_name=model_name,
                              batch_size=batch_size,
                              patience=100
                              )

# diameter of labels in training images
diam_labels = model.diam_labels.copy()


# In[ ]:




