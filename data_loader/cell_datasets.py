from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from cellpose import dynamics, plot, transforms, io, utils
import torchvision.transforms as T
# from pycocotools.coco import COCO


class CellDataset(Dataset):
    def __init__(self, dir, train=True, channels=[0,0], mask_filter='_seg.npy', imf=None, look_one_level_down=False):
        super().__init__()
        self.ids = []
        self.train = train
        self.channels = channels
        self.seg_list = None
        self.diam_mean = 30
        self.scale_range = 1.0
        self.rescale = True
        self.unet = False
        self.image_names = io.get_image_files(dir, mask_filter, imf=imf, look_one_level_down=look_one_level_down)
        self.label_names, self.flow_names = io.get_label_files(self.image_names, mask_filter, imf=imf)
        self.ids = list(range(len(self.image_names)))
        
    def set_train_params(self, diam_mean=30, scale_range=1.0, rescale=True, unet=False):
        self.diam_mean=diam_mean
        self.scale_range = scale_range
        self.rescale = rescale
        self.unet = unet

    def load_raw(self, img_id):
        if self.train:
            # image
            image = self.get_image(img_id)
            # label
            target = self.get_target(img_id) if self.train else {}
            image, target = self.transform(image, target, raw=True)
            return image, target
        else:
            image = self.get_image(img_id)
            image, pre_info = self.transform(image, raw=True)
            return image, pre_info

    def __getitem__(self, img_id):
        if self.train:
            # image
            image = self.get_image(img_id)
            # label
            target = self.get_target(img_id) if self.train else {}
            image, target = self.transform(image, target)
            return image, target
        else:
            image = self.get_image(img_id)
            image, pre_info = self.transform(image)
            return image, pre_info

    def __len__(self):
        return len(self.ids)

    def get_image(self, img_id):
        image = io.imread(self.image_names[img_id])
        return [ image ]
    
    def get_image_files(self):
        return self.image_names
    
    def get_label_files(self):
        return self.label_names, self.flow_names

    @staticmethod
    def convert_to_xyxy(box):
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box # new_box format: (xmin, ymin, xmax, ymax)

    def get_target(self, img_id):
        # return target.shape: [4, Ly, Lx]
        # target[0] is masks, target[1] is cell_probability, target[2] is flow Y, target[3] is flow X.Q
        if self.flow_names != None:
            masks = io.imread(self.flow_names[img_id])
        else:
            masks = io.imread(self.label_names[img_id])
            masks = self.mask_convert(masks)

        # mask to flows, flows.shape: list of [4 x Ly x Lx] arrays
        flows = dynamics.labels_to_flows([masks], files=[self.image_names[img_id]])
        target = flows[0]
        return [target]

    def transform(self, img, label=None, raw=False):
        from time import time
        start = time()
        # dataset argument
        # step1: reshape and normalize data
        tr, ts, rn = transforms.reshape_and_normalize_data(img, channels=self.channels, normalize=True)
        if raw:
            return tr[0], label[0]
        # step2: random rotate and resize
        if self.train and label is not None:
            rsc = utils.diameters(label[0][0])[0] / self.diam_mean if self.rescale else 1.0
            img, label, _ = transforms.random_rotate_and_resize(tr, [label[0][1:]], scale_range=self.scale_range, rescale=[rsc], unet=self.unet)
            img, label = map(torch.from_numpy, [img, label])
            return torch.squeeze(img), torch.squeeze(label)
        else:
            # eval transform
            img, *pre_info  = transforms.pad_image_ND(img)
            img = torch.from_numpy(img)
            return torch.squeeze(img), pre_info

    def mask_convert(self, masks):
        if masks.ndim == 3:
            return masks
        else:
            return masks[np.newaxis, :, :]
