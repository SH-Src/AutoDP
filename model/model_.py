import torch
import torch.nn.functional as F
from model.operations import *
from torch.autograd import Variable

class Cell(nn.Module):
    def __init__(self, d_model, genotype, steps):
        super(Cell, self).__init__()
        self._steps = steps
        self._ops = nn.ModuleList()
        for index in genotype:
            op = OPS[index](d_model)
            self._ops += [op]

    def forward(self, s0):
        states = [s0]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return states


class Network(nn.Module):
    def __init__(self, input_dim=76, d_model=256, steps=2, genotype=None, num_classes=25):
        super(Network, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.cell = Cell(d_model, genotype, steps)
        self.pooler = MaxPoolLayer()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, length):
        x = self.embedding(x)
        x = self.cell(x)[-1]
        x = self.pooler(x, length)
        x = self.classifier(x)
        return torch.sigmoid(x)