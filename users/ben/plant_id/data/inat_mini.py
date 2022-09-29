"""iNat DataModule."""

import argparse
import os
import glob
from pathlib import Path
from PIL import Image

from plant_id.data.base_data_module import BaseDataModule, load_and_print_info
import plant_id.metadata.inat as metadata
from plant_id.stems.image import iNatStem
from plant_id.data.util import BaseDataset


class INAT_MINI(BaseDataModule):
    """iNat-mini DataModule."""

    def __init__(self, split: str, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.transform = iNatStem()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.augment = self.args.get("augment_data", "true").lower() == "true"
        self.root = self.args.get("root", "/")
        
        self.setup('train')
        self.setup('val')
#        self.data_test = self.setup('test')

    def setup(self, split: str):
        if split == 'train':
            data_dir = metadata.MINI_DATA_DIRNAME
        elif split == 'val':
            data_dir = metadata.VAL_DATA_DIRNAME
        elif split == 'test':
            data_dir = metadata.TEST_DATA_DIRNAME
        else:
            ValueError("Split must be 'train', 'val' or 'test'")

        assert os.path.isdir(data_dir), f"Provided directory of images {data_dir} does not exist"

        paths = sorted(glob.glob(data_dir + '/*Plantae*/*'))
        self.hierarchy_map = {self.base_path_name(path).split('_')[-1]:int(self.base_path_name(path).split('_')[0]) - metadata.OFFSET for path in paths}
    
        indices = {
            i:[paths[i], self.hierarchy_map[self.base_path_name(paths[i]).split('_')[-1]]] for i in range(len(paths))
            }
        for idx in range(metadata.PLANT_IDX_RANGE[0], metadata.PLANT_IDX_RANGE[1] + 1):
            assert idx - metadata.OFFSET in self.index, "Index %d not present in root" %idx
        
        if split == 'train':
            self.data_train = indices
        elif split == 'val':
            self.data_val = indices
        elif split == 'test':
            pass
        else:
            ValueError("Split must be 'train', 'val' or 'test'")    
            
    def __len__(self) -> None:
        return len(self.train_index) + len(self.val_index)


    def __getitem__(self, idx: int):
        """
        Inputs: - idx (int): Index; loader handles offset
                - split: get item from train/val/test

        Returns: tuple: (image, target)
        """

        fname, target = self.index[idx]
        img = self.transform(Image.open(fname))

        return img, target
       
    @staticmethod
    def base_path_name(x):
        return os.path.basename(os.path.dirname(x))
    
    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

if __name__ == "__main__":
    load_and_print_info(INAT_MINI)
