"""Metadata for iNat."""

from pathlib import Path

import plant_id.metadata.shared as shared


MINI_DATA_DIRNAME = shared.DATA_DIRNAME + '/data_2021_mini/2021_train_mini/'
MID_DATA_DIRNAME = shared.DATA_DIRNAME + '/data_2021_full/2021_train/'
VAL_DATA_DIRNAME = shared.DATA_DIRNAME + '/data_validation/'
TEST_DATA_DIRNAME = shared.DATA_DIRNAME + '/data_validation/'

PLANT_IDX_RANGE = [5729, 9999]
OFFSET = PLANT_IDX_RANGE[0]
NUM_PLANT_CLASSES = PLANT_IDX_RANGE[1] - PLANT_IDX_RANGE[0]

DIMS = (3, 224, 224)
OUTPUT_DIMS = (1,)

MINI_TRAIN_SIZE = 500_000
MID_TRAIN_SIZE = 2_686_843
VAL_SIZE = 100_000
TEST_SIZE = 500_000
