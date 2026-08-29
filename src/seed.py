import torch     
import random   
import numpy as np

def set_global_seed(seed:int = 42)->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed(seed)

