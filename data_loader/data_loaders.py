from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import CellDataset

class CellDataLoader(BaseDataLoader):
    """
    Cell data loading demo using BaseDataLoader
    """
    def __init__(self, dataset, batch_size, train=True, shuffle=True, validation_split=0.0, num_workers=1):
        if type(dataset) == str:
            self.dataset = CellDataset(data_dir=self.dataset, train=train)
        else:
            self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)