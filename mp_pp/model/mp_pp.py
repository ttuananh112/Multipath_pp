from abc import ABC
from typing import Dict, Tuple
from omegaconf import DictConfig

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from model.encoder import ObjectEncoder, DynamicEncoder
from model.blocks import StackMCG, MLP, Reshape
from model.utils import get_pseudo_prediction, get_device


class MP(pl.LightningModule, ABC):
    def __init__(
            self,
            configs: DictConfig
    ):
        super().__init__()
        self._configs = configs
        # --- data params ---
        self._dim = configs.model.data.dim
        self._num_info = configs.model.data.num_info
        self._history_steps = configs.model.data.history_steps
        self._future_steps = configs.model.data.future_steps
        self._num_waypoints = configs.model.data.num_waypoints
        self._num_anchor = configs.model.anchor.num_anchor
        self._device = configs.model.train.device

        self._is_val_sanity_check = True

        # --- train params ---
        self._batch_size = self._configs.model.train.batch_size
        # lr_scheduler
        self._lower_lr = self._configs.model.train.lower_lr
        self._upper_lr = self._configs.model.train.upper_lr
        self._step_size_up = self._configs.model.train.step_size_up
        self._gamma = self._configs.model.train.gamma
        # ratio of loss func
        self._alpha = self._configs.model.train.alpha

        # --- input encoder ---
        self._light_encoder = ObjectEncoder(configs)
        self._map_encoder = ObjectEncoder(configs)
        self._dynamics_encoder = DynamicEncoder(configs)
        # --- intermediate encoder ---
        self._mcg_map = StackMCG(configs)
        self._mcg_interaction = StackMCG(configs)
        self._mcg_predictor = StackMCG(configs, dim=self._dim * self._num_info)
        # --- learnable anchors ---
        self._anchors = torch.nn.Parameter(
            torch.rand(self._num_anchor, self._dim * self._num_info),
            requires_grad=True
        )
        # --- output heads ---
        self._classification = nn.Sequential(
            MLP(
                configs,
                inp_dim=self._dim * self._num_info,
                out_dim=1,  # each anchor has 1 prob score
                is_bn=False,
                act="softmax"
            ),
            Reshape(-1, self._num_anchor)
        )

        self._regression = nn.Sequential(
            MLP(
                configs,
                inp_dim=self._dim * self._num_info,
                out_dim=2 * self._future_steps,  # x, y
                is_bn=False,
                act=None
            ),
            Reshape(-1, self._num_anchor, self._future_steps, 2)
        )

        # --- loss func ---
        self._classification_loss = nn.NLLLoss()
        # self._regression_loss = nn.GaussianNLLLoss()
        self._regression_loss = nn.MSELoss()

        # allocate device
        self.to(get_device(self._device))

        # --- logging ---
        self.loggings = {
            "all": self.__reset_log(),
            "avg": self.__reset_log(),
        }

    @staticmethod
    def __reset_log():
        return {
            "train": {
                "loss": list(),
                "cls_loss": list(),
                "reg_loss": list(),
            },
            "val": {
                "loss": list(),
                "cls_loss": list(),
                "reg_loss": list(),
            }
        }

    def forward(
            self,
            x: Dict
    ):
        """
        :param x: input data should be in dict
        {
            "traffic_light": tensor with shape (batch, dimension)
            "map": tensor with shape (batch, num_waypoints, dimension)
            "agent": tensor with shape (batch, history_step, dimension)
            "others": tensor with shape (batch, num_objects, history_step, dimension)
        }
        :return:
            - classification_head: (batch, num_anchor)
            - regression_head: (batch, num_anchor, 4)  # mu_x, mu_y, sig_x, sig_y
        """
        if not set(x.keys()).issuperset({"traffic_light", "map", "agent", "others"}):
            raise "Input should have keys:\n-\"traffic_light\"\n-\"map\"\n-\"agent\"\n-\"others\""

        light_enc = self._light_encoder(x["traffic_light"])  # (batch, dim)
        map_enc = self._map_encoder(x["map"])  # (batch, num_waypoints, dim)
        agent_enc = self._dynamics_encoder(x["agent"])  # (batch, dim)

        batch, num_objects, history_step, dimension = x["others"].shape
        # (batch * num_objects, dim)
        others_enc = self._dynamics_encoder(
            x["others"].view(batch * num_objects, history_step, dimension)
        )
        # (batch, num_objects, dim)
        others_enc = others_enc.view(
            batch, num_objects, dimension
        )

        _, map_context = self._mcg_map(map_enc, agent_enc)  # (batch, dim)
        _, interaction_context = self._mcg_interaction(others_enc, agent_enc)  # (batch, dim)

        # concatenate all information
        # (batch, dim * 4)
        gathered_info = torch.concat([
            light_enc,
            map_context,
            agent_enc,
            interaction_context
        ], dim=-1)

        # repeat weight of anchors in batch dimension for computing in batch
        # (batch, num_anchor, dim * 4)
        repeat_anchor = self._anchors.repeat(batch, 1, 1)
        # (batch, num_anchor, dim * 4)
        predictor_heads, _ = self._mcg_predictor(repeat_anchor, gathered_info)

        # (batch, num_anchor)
        # corresponding to probability of each anchor
        classification_head = self._classification(predictor_heads)

        # simple version of regression
        # (batch, num_anchor, future_step, 2)
        # 2: mu_x, mu_y
        regression_head = self._regression(predictor_heads)

        return classification_head, regression_head

    def configure_optimizers(self):
        # cyclical learning rate scheduler
        optimizer = torch.optim.SGD(self.parameters(), lr=self._upper_lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self._lower_lr, max_lr=self._upper_lr,
            step_size_up=self._step_size_up,
            gamma=self._gamma,
            mode="exp_range"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }

    def get_loss(
            self,
            data_batch: Tuple
    ) -> Dict:
        """
        :param data_batch: List[(x, y)],
            - x: List[dict]
            {
                "traffic_light": tensor with shape (batch, dimension)
                "map": tensor with shape (batch, num_waypoints, dimension)
                "agent": tensor with shape (batch, history_step, dimension)
                "others": tensor with shape (batch, num_objects, history_step, dimension)
            }
            and should be pre-processed

            - y: tensor with shape (batch, future_steps, 2)

        :return:
            Dict:
                - loss
                - cls_loss
                - reg_loss
        """
        x, y = data_batch
        # allocate data into device
        for key in x.keys():
            x[key] = x[key].to(self._device)
        y = y.to(self._device)

        # cls: (batch, num_anchor)
        # reg: (batch, num_anchor, future_steps, 2)  # mu_x, mu_y
        cls, reg = self.forward(x)

        # get pseudo label from prediction
        pseudo_gt_idx, y_hat = get_pseudo_prediction(y, reg, self._device)

        # loss
        cls_loss = self._alpha * self._classification_loss(torch.log(cls), pseudo_gt_idx)
        reg_loss = (1 - self._alpha) * self._regression_loss(y_hat, y.float())
        loss = cls_loss + reg_loss

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss
        }

    def training_step(
            self,
            train_batch,
            batch_idx
    ):
        """
        Training step
        :param train_batch:
        :param batch_idx:
        :return:
        """
        losses = self.get_loss(data_batch=train_batch)

        self.loggings["all"]["train"]["loss"].append(losses["loss"].item())
        self.loggings["all"]["train"]["cls_loss"].append(losses["cls_loss"].item())
        self.loggings["all"]["train"]["reg_loss"].append(losses["reg_loss"].item())
        self.log("train_loss", losses["loss"])

        return losses["loss"]

    def validation_step(self, val_batch, batch_idx):
        """
        Validation step
        :param val_batch:
        :param batch_idx:
        :return:
        """
        losses = self.get_loss(data_batch=val_batch)

        self.loggings["all"]["val"]["loss"].append(losses["loss"].item())
        self.loggings["all"]["val"]["cls_loss"].append(losses["cls_loss"].item())
        self.loggings["all"]["val"]["reg_loss"].append(losses["reg_loss"].item())
        self.log("val_loss", losses["loss"])

        return losses["loss"]

    def on_validation_epoch_end(self) -> None:
        """
        Gather logging information after each end of validation epoch
        :return:
        """
        # skip validation sanity check
        if self._is_val_sanity_check:
            self._is_val_sanity_check = False
            return

        # get mean value in epoch
        for phase in ["train", "val"]:
            for score in ["loss", "cls_loss", "reg_loss"]:
                self.loggings["avg"][phase][score].append(
                    np.mean(self.loggings["all"][phase][score])
                )
        # reset data in all steps
        self.loggings["all"] = self.__reset_log()

    def on_train_epoch_start(self) -> None:
        """
        Get learning before each start of training epoch
        :return:
        """
        # get learning rate
        lr = self.lr_schedulers().get_last_lr()
        if "hyper_params" not in self.loggings["avg"]:
            self.loggings["avg"]["hyper_params"] = {"lr": [lr]}
        else:
            self.loggings["avg"]["hyper_params"]["lr"] += [lr]
