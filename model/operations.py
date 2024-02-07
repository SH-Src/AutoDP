import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    0: lambda d_model: Identity(d_model),
    1: lambda d_model: Zero(d_model),
    2: lambda d_model: FFN(d_model),
    3: lambda d_model: RNN(d_model),
    4: lambda d_model: Attention(d_model)
}

class Identity(nn.Module):
  def __init__(self, d_model):
      super(Identity, self).__init__()
  def forward(self, x):
      return x

class FFN(nn.Module):

  def __init__(self, d_model):
      super(FFN, self).__init__()
      self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
      self.layer_norm = nn.LayerNorm(d_model)
      self.dropout = nn.Dropout(0.1)

  def forward(self, x):
      x = self.layer_norm(x + self.dropout(self.ffn(x)))
      return x

class Zero(nn.Module):
    def __init__(self, d_model):
        super(Zero, self).__init__()

    def forward(self, x):
        return torch.mul(x, 0)

class RNN(nn.Module):
    def __init__(self, d_model):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
    def forward(self, x):
        rnn_input = x
        rnn_output, _ = self.rnn(rnn_input)
        return rnn_output


class Attention(nn.Module):
    def __init__(self, in_feature, num_head=4, dropout=0.1):
        super(Attention, self).__init__()
        self.in_feature = in_feature
        self.num_head = num_head
        self.size_per_head = in_feature // num_head
        self.out_dim = num_head * self.size_per_head
        assert self.size_per_head * num_head == in_feature
        self.q_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.k_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.v_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.fc = nn.Linear(in_feature, in_feature, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, x):
        q, k, v = x, x, x
        batch_size = q.size(0)
        res = q
        query = self.q_linear(q)
        key = self.k_linear(k)
        value = self.v_linear(v)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        scale = np.sqrt(self.size_per_head)
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / scale

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.in_feature)
        x = self.fc(x)
        x = self.dropout(x)
        x += res
        x = self.layer_norm(x)
        return x


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths=None):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if mask_or_lengths is not None:
            if len(mask_or_lengths.size()) == 1:
                mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(
                    1))
            else:
                mask = mask_or_lengths
            inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = inputs.max(1)[0]
        return max_pooled