#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from data_loader import CellDataset, split_dataset
from cellpose import models

use_GPU = core.use_gpu()
yn = ['NO', 'YES']
print(f'>>> GPU activated? {yn[use_GPU]}')


# In[ ]:

initial_model = "cyto" #@param ['cyto','nuclei','tissuenet','livecell','cyto2','CP','CPx','TN1','TN2','TN3','LC1','LC2','LC3','LC4','scratch']
model_dir = "./"
model_name = "single_cell_params_test.pt" #@param {type:"string"}
n_epochs =  100#@param {type:"number"}
learning_rate = 0.001 
weight_decay = 0.0001
chan = 0
chan2 = 0

# In[ ]:

data_paths = []
with open("./file_paths.txt", "r") as fp:
    data_paths = fp.read().strip().split('\n')


# In[ ]:

train_paths, test_paths = split_dataset(data_paths, 0.25)
train_ds = CellDataset(paths=train_paths, generate_flows=True)
test_ds = CellDataset(paths=test_paths, generate_flows=True)


# In[ ]:

# start logger (to see training across epochs)
logger = io.logger_setup()

# DEFINE CELLPOSE MODEL (without size model)
model = models.CellposeModel(gpu=use_GPU, diam_mean=45)

# set channels
channels = [chan, chan2]

new_model_path = model.train(train_ds, test_ds, save_path=model_dir, 
                              n_epochs=n_epochs,
                              learning_rate=learning_rate, 
                              weight_decay=weight_decay, 
                              nimg_per_epoch=5,
                              model_name=model_name,
                              batch_size=8
                              )

# diameter of labels in training images
diam_labels = model.diam_labels.copy()
