from lanegcn import ActorNet, actor_gather
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

config, Dataset, collate_fn, net, loss, post_process, opt = get_model()

# def worker_init_fn(pid):
#     np_seed = int(pid)
#     np.random.seed(np_seed)
#     random_seed = np.random.randint(2 ** 32 - 1)
#     random.seed(random_seed)

# dataset = Dataset('./dataset/train_mini/data', config, train=True)
# train_loader = DataLoader(
#         dataset,
#         batch_size=config["batch_size"],
#         num_workers=config["workers"],
#         shuffle=False,   # True: At each epoch, reorder the data
#         collate_fn=collate_fn,
#         pin_memory=True,
#         worker_init_fn=worker_init_fn,   # The next 36 were thrown away
#         drop_last=True,
#     )


# for i, data in enumerate(train_loader):
#     data =dict(data) 


avl = ArgoverseForecastingLoader('./dataset/train_mini/data')


actor_net = ActorNet(config)
# construct actor feature
actors, actor_idcs = actor_gather(data["feats"])
actor_ctrs = data["ctrs"]
actors = actor_net(actors)

final_h = encoder(traj_rel)
















class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores