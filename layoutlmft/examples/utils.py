from typing import Optional
import os
import random
import numpy as np
import torch


def seed_everything(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = 42

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed
