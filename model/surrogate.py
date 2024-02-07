import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from model.operations import *

class DAG(nn.Module):
    def __init__(self, d_model, num_op, steps):
        super().__init__()
        self.d_model = d_model
        self.num_op = num_op
        self.steps = steps
        self.matrices = nn.Parameter(torch.randn(num_op, d_model, d_model))
        nn.init.kaiming_uniform_(self.matrices)
        self.init = nn.Parameter(torch.randn(d_model))
        # self.linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        stdv = 1.0 / math.sqrt(d_model)
        for weight in self.init:
            weight.data.uniform_(-stdv, stdv)

    def forward(self, archs):
        bs = archs.size(0)
        states = [self.init.unsqueeze(0).expand(bs, self.d_model)]
        offset = 0
        for i in range(self.steps):
            s = sum(torch.bmm(self.matrices[archs[:, offset + j]], h.unsqueeze(-1)).squeeze() for j, h in enumerate(states)) / len(states)
            offset += len(states)
            # s = self.linear(s)
            states.append(s)

        return states

class Surrogate(nn.Module):
    def __init__(self, d_model, num_op, steps, num_classes=25):
        super(Surrogate, self).__init__()
        self.d_model = d_model
        self.dag = DAG(d_model, num_op, steps)
        self.attention = Attention(d_model, num_head=1)
        self.task_emb = nn.Parameter(torch.Tensor(num_classes, d_model))
        self.bias = nn.Parameter(torch.Tensor(d_model))
        stdv = 1.0 / math.sqrt(d_model)
        for weight in self.task_emb:
            weight.data.uniform_(-stdv, stdv)
        for weight in self.bias:
            weight.data.uniform_(-stdv, stdv)
        self.out_mlp = nn.Sequential(nn.Linear(2*d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))
        self.layernorm = nn.LayerNorm(d_model)
        # self.layernorm1 = nn.LayerNorm(d_model)

    def forward(self, archs, tasks):
        graph = self.dag(archs)[-1]
        bs, numtask = tasks.size()
        task_emb = self.task_emb.unsqueeze(0).expand(bs, numtask, self.d_model) * tasks.unsqueeze(-1) + self.bias.unsqueeze(0).expand(bs, numtask, self.d_model)
        task = torch.sum(self.attention(task_emb) * tasks.unsqueeze(-1), dim=1)
        cat = torch.cat((self.layernorm(graph), self.layernorm(task)), dim=-1)
        return self.out_mlp(cat)


from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MetaData(Dataset):
    def __init__(self, archs, tasks, gains):
        self.archs = archs
        self.tasks = tasks
        self.gains = gains

    def __len__(self):
        return len(self.archs)

    def __getitem__(self, idx):
        return self.archs[idx], self.tasks[idx], self.gains[idx]


def meta_collate(batch):
    arch = [item[0] for item in batch]
    tasks = [item[1] for item in batch]
    gains = [item[2] for item in batch]

    return torch.LongTensor(arch), torch.LongTensor(np.stack(tasks, axis=0)), torch.Tensor(np.stack(gains, axis=0))