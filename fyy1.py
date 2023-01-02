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
from lanegcn import get_model
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt

b = 1
def abc():
    print(b)
    return b

config, Dataset, collate_fn, net, loss, post_process, opt = get_model()

def worker_init_fn(pid):
    np_seed = int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)

dataset = Dataset('./dataset/train_mini/data', config, train=True)
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



# avl = ArgoverseForecastingLoader('./dataset/train_mini/data')
# print(avl[0])

split = np.load('./dataset/preprocess/train_crs_dist6_angle90.p', allow_pickle=True)

print(len(split))  # split = 110(train_mini = 110)

# print(split[1])

print(split[0].keys())   # split.keys = (['idx', 'city', 'feats', 'ctrs', 'orig', 'theta', 'rot', 'gt_preds', 'has_preds', 'graph'])

# print(split[9]['feats'].size)  # split[i]['feats'].size = 300-3000




for i, data in enumerate(train_loader):
    data =dict(data) 
    # print('i =', i,'len = ', len(data['city']))
    # break
# print('batch =', i)

# batch_size = 32
# data_keys = (['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']) 
# data['graph']_keys = (['ctrs', 'num_nodes', 'feats', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'left', 'right'])
# len(feats) = 32, data['feats'].size = [12,20,3], actor_num = 12





# len(data['graph']) = 32

#############################splite[0]##########################
print(data['graph'][0]['ctrs'].size(), data['graph'][0]['ctrs'])
# torch.Size([1431, 2])
# graph_ctrs = (ctrln[:-1] + ctrln[1:]) / 2
# graph_feats = ctrln[1:] - ctrln[:-1]

ctrs0 = data['graph'][0]['ctrs']
ctrs1 = data['graph'][1]['ctrs']
x0 = ctrs0[:,0]
y0 = ctrs0[:,1]
x1 = ctrs1[:,0]
y1 = ctrs1[:,1]
plt.scatter(x0, y0)
# plt.scatter(x1, y1)

# print(data['graph'][0]['num_nodes'])


# print(data['ctrs'][0].size())
# data_ctrs = feat[-1, :2] ----traj_destination
# data_feats = feat[1:, :2] - feat[:-1, :2]

ctrs = data['ctrs'][0]
feat = data['feats'][0]
print(data['feats'][0].size())   # [12, 20, 3]
# print(data['feats'][0][0], len(data['feats'][0][0]))

# print(feat[0][-1, :2])   # ctrs is the last row of feat

# print(feat[0][:, :2])
# a = feat[0][:,:2].clone()
# a[19] = ctrs[0]

for j in range(len(ctrs)):
    a = feat[j][:,:2].clone()
    a[19] = ctrs[j]
    for i in range(18):
        a[18-i] = a[19-i] - feat[j][19-i,:2]
        plt.scatter(a[:,0], a[:,1])
# print(a)
plt.show()

