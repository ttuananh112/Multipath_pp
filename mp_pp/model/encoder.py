import torch
from torch import nn

from omegaconf import DictConfig
from model.blocks import BaseModule, MLP, StackMCG


class ObjectEncoder(BaseModule):
    def __init__(
            self,
            configs: DictConfig
    ):
        """
        Use to encode
            - light
            - map
            - agent
            - other objects
        :param configs:
        """
        super().__init__(configs)
        self._mlp = MLP(configs)

    def forward(
            self,
            inputs: torch.Tensor
    ):
        return self._mlp(inputs)


class DynamicEncoder(BaseModule):
    """
    Encode agent and others dynamic object
    """
    def __init__(
            self,
            configs: DictConfig
    ):
        super().__init__(configs)
        # input encoder
        self._encoder = ObjectEncoder(configs)
        # get historical information
        self._lstm = nn.LSTM(
            input_size=self._dim,
            hidden_size=self._dim,
            batch_first=True
        )
        # synthetic historical data
        self._mcg = StackMCG(configs)

        # transform concatenated data
        self._mlp = nn.Sequential(
            nn.Linear(self._dim * 2, self._dim),
            nn.ReLU(),
            nn.BatchNorm1d(self._dim)
        )

    def forward(
            self,
            inputs: torch.Tensor
    ):
        """
        Concatenate
            - output of lstm
            - context of MCG
        :param inputs: shape (batch, history_step, dimension)
        :return: (batch, dimension)
        """
        batch, history_step, dimension = inputs.shape
        enc = self._encoder(inputs)

        # encode sequence historical data using lstm
        output, _ = self._lstm(enc)
        last_state = output[:, -1, :]  # (batch, dim)

        # encode general context using MCG
        _, context = self._mcg(enc, torch.ones((batch, dimension), device=self._device))  # (batch, dim)

        # concatenate information
        concat = torch.concat([last_state, context], dim=-1)  # (batch, dim*2)
        # transform data
        output = self._mlp(concat)

        return output
