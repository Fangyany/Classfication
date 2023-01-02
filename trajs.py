import os

import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number
import torch
from torch.utils.data import Sampler, DataLoader
from utils import Logger, load_pretrain
import matplotlib.pyplot as plt
from lanegcn import get_model
from fyy1 import abc

b = abc()
# print('b =', b)




config, Dataset, collate_fn, net, loss, post_process, opt = get_model()

def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)

dataset = Dataset("train_split", config, train=True)
train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,   # True: At each epoch, reorder the data
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,   # The next 36 were thrown away
        drop_last=True,
    )

data = {}
for i, data in enumerate(train_loader):
    data = dict(data)
    break


print(data.keys())

print(len(data['trajs'][0]))
print(len(data['traj1'][0]))

