import torch.nn as nn
import torch
from typing import Tuple


class MultiModalActionNet(nn.Module):
    def __init__(self, num_components: int, latent_dimension: int,
                 action_dimension: int, train_categorical_weights: bool):
        super(MultiModalActionNet, self).__init__()

        self.means = nn.ModuleList([nn.Linear(latent_dimension, action_dimension)
                                    for _ in range(num_components)])

        self.num_components = num_components
        self.train_categorical_weights = train_categorical_weights
        if self.train_categorical_weights:
            self.weight_logits = nn.Linear(latent_dimension, num_components)
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.weight_logits = None
            self.softmax = None

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means = torch.stack([mean(tensor) for mean in self.means])
        means = means.transpose(1, 0)  # switch batch and component index

        if self.train_categorical_weights:
            weight_logits = self.weight_logits(tensor)
            weights = self.softmax(weight_logits)
        else:
            weights = torch.ones(size=(*tensor.shape[:-1], self.num_components))/self.num_components
        return weights, means
