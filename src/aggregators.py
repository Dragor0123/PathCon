import torch
import torch.nn as nn
from abc import abstractmethod


class Aggregator(nn.Module):
    def __init__(self, batch_size: int, input_dim: int, output_dim: int, act, self_included: bool):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included

    def forward(self, self_vectors, neighbor_vectors, masks):
        # self_vectors: (batch_size, -1, input_dim)
        # neighbor_vectors: (batch_size, -1, 2, n_neighbor, input_dim)
        # masks: (batch_size, -1, 2, n_neighbor, 1)entity_vectors = tor
        entity_vectors = torch.mean(neighbor_vectors * masks, dim=-2)
        outputs = self._call(self_vectors, entity_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: (batch_size, -1, input_dim)
        # entity_vectors: (batch_size, -1, 2, input_dim)
        pass


class MeanAggregator:
    pass


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size: int, input_dim: int, output_dim: int,
                 act=lambda x:x, self_included: bool=True):
        super(ConcatAggregator, self).__init__()

        multiplier = 3 if self_included else 2
        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    # 이 부분 공부해보자 - 이해가 잘 안됨
    def _call(self, self_vectors, entity_vectors):
        output = entity_vectors.view([-1, self.input_dim * 2])  # (-1, input_dim * 2)
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # (-1, input_dim)
            output = torch.cat([self_vectors, output], dim=-1)      # (-1, input_dim * 3)
        output = self.layer(output)     # (-1, output_dim)
        output = output.view([self.batch_size, -1, self.output_dim])    # (batch_size, -1, output_dim)

        return self.act(output)
