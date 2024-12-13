import torch as tc
import numpy as np

def eye(*args, **kwargs):
    return tc.from_numpy(np.eye(*args, **kwargs))