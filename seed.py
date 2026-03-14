'''
Author Ankit Yadav
Date: 2025-04-03
Comment: This script sets the seed for reproducibility in PyTorch and NumPy.
'''
import random
import numpy as np
import torch


# TODO 
# Set the seed for the diffuser model but less priority



# Setting the seed for reproducibility
Seed = 8
random.seed(Seed)
np.random.seed(Seed)
torch.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#######################################################################################################################
