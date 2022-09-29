import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable

import glob

PLANT_IDX_RANGE = [5729, 9999]
OFFSET = PLANT_IDX_RANGE[0]
TARGET_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

def base_path_name(x):
    return os.path.basename(os.path.dirname(x))

class DataLoader_iNaturalist(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable,
    ) -> None:

        self.root = root 
        self.transform = transform

        assert os.path.isdir(self.root), "Provided directory of images does not exist"

        paths = sorted(glob.glob(root + '/*Plantae*/*'))

        self.hierarchy_map = {base_path_name(path).split('_')[-1]:int(base_path_name(path).split('_')[0]) - OFFSET for path in paths}
            
        self.index = {
                i:[paths[i], self.hierarchy_map[base_path_name(paths[i]).split('_')[-1]]] for i in range(len(paths))
                }
                
        for idx in range(PLANT_IDX_RANGE[0], PLANT_IDX_RANGE[1] + 1):
            assert idx-OFFSET in self.index, "Index %d not present in root" %idx

    def __len__(self) -> None:
        return len(self.index)

    def __getitem__(self, idx: int):
        """
        Input: idx (int): Index; loader handles offset

        Returns: tuple: (image, target)
        """

        fname, target = self.index[idx]
        img = Image.open(fname)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self) -> int:
        return len(self.index)
