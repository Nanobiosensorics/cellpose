import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
from data_loader import CellDataset, split_dataset
from cellpose import models
import argparse


def run(src_path, model_name, model_dir='./', initial_model = "cyto", 
        n_epochs=100, learning_rate=0.01, batch_size=8, patience=20):
        
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')
    
    weight_decay = 0.0001

    data_paths = []
    with open(src_path, "r") as fp:
        data_paths = fp.read().strip().split('\n')

    train_paths, test_paths = split_dataset(data_paths, 0.2)
    train_ds = CellDataset(paths=train_paths, generate_flows=True)
    test_ds = CellDataset(paths=test_paths, generate_flows=True)

    logger = io.logger_setup()

    model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

    model.train(train_ds, test_ds, save_path=model_dir, 
                                n_epochs=n_epochs,
                                learning_rate=learning_rate, 
                                weight_decay=weight_decay, 
                                nimg_per_epoch=5,
                                model_name=model_name,
                                batch_size=batch_size,
                                patience=patience
                                )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cellpose trainer')
    parser.add_argument('-p', '--src_path', type=str, required=True)
    parser.add_argument('-mn', '--model_name', type=str, required=True)
    parser.add_argument('-mp', '--model_path', type=str, default='./')
    parser.add_argument('-i', '--initial_model', type=str, default='cyto')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-pt', '--patience', type=int, default=20)
    args = parser.parse_args()
    run(args.src_path, args.model_name, args.model_path, args.initial_model,
        args.epochs, args.learning_rate, args.batch_size, args.patience)
