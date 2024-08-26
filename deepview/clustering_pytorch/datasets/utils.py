

import numpy as np
from enum import Enum

class Batch(Enum):
    inputs = 0
    part_score_targets = 1
    part_score_weights = 2
    locref_targets = 3
    locref_mask = 4
    pairwise_targets = 5
    pairwise_mask = 6
    data_item = 7
