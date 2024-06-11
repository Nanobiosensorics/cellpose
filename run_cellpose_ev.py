import os, random
import numpy as np
from cellpose import core, io, models
from cellpose.data_loader import CellDataset
from cellpose import models
from load import get_image_ids, get_sample
import argparse

def run(data_path, model_name, initial_model = "scratch", 
        n_epochs=100, learning_rate=0.1, batch_size=8, patience=20, val_percent=.2, diam_mean=30,
        channels=[1, 2]):


    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    weight_decay = 0.0001
    model_dir = os.path.join(data_path, 'results')

    data_path, ids = get_image_ids(data_path)

    paths = []

    for idx in ids:
        sample = get_sample(data_path, idx)
        paths.append(sample['esrrf'])

    random.shuffle(paths)

    n_val = int(len(paths) * val_percent)
    n_train = len(paths) - n_val
    train_paths, test_paths = paths[:n_train], paths[n_train:]

    train_ds = CellDataset(paths=train_paths, generate_flows=True, mask_filter='_masks', channels=channels)
    test_ds = CellDataset(paths=test_paths, generate_flows=True, mask_filter='_masks', channels=channels)

    model = models.CellposeModel(gpu=use_GPU, model_type=None if initial_model == 'scratch' else initial_model, diam_mean=diam_mean)

    model.train(train_ds, test_dataset=test_ds,
                save_path=model_dir, 
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
    parser.add_argument('-i', '--initial_model', type=str, default='cyto')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-pt', '--patience', type=int, default=20)
    parser.add_argument('-c', '--channels', nargs='+', type=int, default=[0,0])
    parser.add_argument('-v', '--val_percent', type=float, default=.2)
    parser.add_argument('-dm', '--diam_mean', type=int, default=30)  # Added argument for diam_mean
    args = parser.parse_args()
    
    run(
        data_path=args.src_path,
        model_name=args.model_name,
        initial_model=args.initial_model,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
        val_percent=args.val_percent,
        diam_mean=args.diam_mean,
        channels=args.channels
    )