import torch
from torch import nn

from abc import abstractmethod
from omegaconf import DictConfig
from copy import deepcopy
from typing import Union
from model.utils import get_device


class BaseModule(nn.Module):
    """
    Base module for block
    """

    def __init__(
            self,
            configs: DictConfig
    ):
        super().__init__()
        self._configs = configs
        self._dim = configs.model.data.dim
        self._device = get_device(configs.model.train.device)

    @abstractmethod
    def forward(self, **kwargs):
        pass


class MLP(BaseModule):
    """
    Multi layer perceptron
    FC -> ACT -> BN
    """

    def __init__(
            self,
            configs: DictConfig,
            inp_dim: int = None,
            out_dim: int = None,
            is_bn: bool = True,
            act: Union[str, None] = "tanh"
    ):
        super().__init__(configs)
        self._inp_dim = inp_dim if inp_dim is not None else self._dim
        self._out_dim = out_dim if out_dim is not None else self._dim
        self._is_bn = is_bn
        self._drop_rate = configs.model.mlp.drop_rate

        # linear func
        self._fc = nn.Linear(self._inp_dim, self._out_dim)
        # activation
        self._act = None
        if act == "relu":
            self._act = nn.ReLU()
        elif act == "leakyrelu":
            self._act = nn.LeakyReLU(negative_slope=0.1)
        elif act == "tanh":
            self._act = nn.Tanh()
        elif act == "softmax":
            # applied softmax by K anchors (dim=-2)
            self._act = nn.Softmax(dim=-2)
        # batch norm
        self._bn = nn.BatchNorm1d(self._out_dim, momentum=0.5)
        # drop-out
        self._drop = nn.Dropout(self._drop_rate)

    def block(self, inputs):
        x = self._fc(inputs)
        if self._act is not None:
            x = self._act(x)
        if self._is_bn:
            shape = x.shape
            x = x.view(-1, self._out_dim)
            x = self._bn(x)
            x = x.view(shape)
        x = self._drop(x)
        return x

    def forward(self, inputs):
        return self.block(inputs)


class MultiContextGating(BaseModule):
    """
    Multi context gating
    """

    def __init__(
            self,
            configs: DictConfig
    ):
        super().__init__(configs)

        self._mlp = MLP(configs)
        self._pool = lambda x: torch.max(x, dim=1)[0]

    def forward(
            self,
            inputs: torch.Tensor,
            context: torch.Tensor
    ):
        """
        :param inputs: shape (batch, num_object, dimension)
        :param context: shape (batch, dimension)
        :return:
            - input_fuse: (batch, num_object, dimension)
            - context_fuse: (batch, dimension)
        """
        inp_trans = self._mlp(inputs)
        batch, num, dim = inp_trans.shape
        ctx_trans = self._mlp(context)
        ctx_trans = ctx_trans.view(batch, 1, dim)

        # fuse input and context
        inp_fuse = inp_trans * ctx_trans
        # get new context by pooling fuse
        ctx_fuse = self._pool(inp_fuse)

        return inp_fuse, ctx_fuse


class StackMCG(BaseModule):
    """
    Stack of Multi context gating
    Do running mean for input
    """

    def __init__(
            self,
            configs: DictConfig,
            dim: int = None
    ):
        super().__init__(configs)

        if dim is not None:
            configs = deepcopy(configs)
            configs.model.data.dim = dim

        self._repeat = configs.model.mcg.repeat
        self._mcg = MultiContextGating(configs)

    def forward(
            self,
            inputs: torch.Tensor,
            context: torch.Tensor
    ):
        """
        Running average skip connection
        :param inputs: shape (batch, num_object, dimension)
        :param context: shape (batch, dimension)
        :return:
            - input_fuse: (batch, num_object, dimension)
            - context_fuse: (batch, dimension)
        """
        batch, num_object, dimension = inputs.shape
        # mean container
        inp_mean_container = torch.empty((batch, 0, num_object, dimension), device=self._device)
        ctx_mean_container = torch.empty((batch, 0, dimension), device=self._device)
        # expand dimension to concatenate with container
        inp_fuse_ext = inputs.view(batch, 1, num_object, dimension)
        ctx_fuse_ext = context.view(batch, 1, dimension)
        # add to container
        inp_mean_container = torch.concat([inp_mean_container, inp_fuse_ext], dim=1)
        ctx_mean_container = torch.concat([ctx_mean_container, ctx_fuse_ext], dim=1)

        for _ in range(self._repeat):
            inp_mean = torch.mean(inp_mean_container, dim=1)
            ctx_mean = torch.mean(ctx_mean_container, dim=1)

            # get output from multi-context-gating
            inputs, context = self._mcg(inp_mean, ctx_mean)

            # expand dimension to concatenate with container
            inp_fuse_ext = inputs.view(batch, 1, num_object, dimension)
            ctx_fuse_ext = context.view(batch, 1, dimension)
            # add to container
            inp_mean_container = torch.concat([inp_mean_container, inp_fuse_ext], dim=1)
            ctx_mean_container = torch.concat([ctx_mean_container, ctx_fuse_ext], dim=1)

        inp_mean = torch.mean(inp_mean_container, dim=1)
        ctx_mean = torch.mean(ctx_mean_container, dim=1)
        return inp_mean, ctx_mean


class Reshape(nn.Module):
    """
    Reshape block
    """

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
