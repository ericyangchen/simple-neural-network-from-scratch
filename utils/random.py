import random
import numpy as np

def set_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)