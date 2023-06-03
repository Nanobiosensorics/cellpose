from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from cellpose import dynamics, plot, transforms, io, utils
import torchvision.transforms as T
# from pycocotools.coco import COCO


class CellDataset(Dataset):
    def __init__(self, data_dir, channels=[0,0], load_seg=False, train=True, mask_filter='_seg.npy'):
        super().__init__()
        self.ids = []
        self.channels = channels
        self.train = train
        self.data_dir = data_dir
        self.seg_list = None
        self.diam_mean = 30
        self.scale_range = 1.0
        self.rescale = True
        self.unet = False
        
        self.imgs = [f[:-4] for f in os.listdir(data_dir) if f.endswith(('png', 'jpg', 'tif'))]
        self.img_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('png', 'jpg', 'tif'))]
        self.masks_list = [os.path.join(data_dir, f + mask_filter) for f in self.imgs]
        self.flows_list = [os.path.join(data_dir, f + '_flows.tif') for f in self.imgs]
        self.ids = list(range(len(self.img_list)))
        print(self.img_list, self.masks_list, self.flows_list, sep='\n')
        
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
        image = io.imread(self.img_list[img_id])
        return [ image ]

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
        if os.path.exists(self.flows_list[img_id]):
            masks = io.imread(self.flows_list[img_id])
        else:
            masks = io.imread(self.masks_list[img_id])
            masks = self.mask_convert(masks)

        # mask to flows, flows.shape: list of [4 x Ly x Lx] arrays
        flows = dynamics.labels_to_flows([masks], files=[self.img_list[img_id]])
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
