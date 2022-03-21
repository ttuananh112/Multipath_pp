import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from constants import ROOT_DIR
from model.mp_pp import MP
from dataset.carla import CarlaDataset
from dataset.data_processor import collate_fn

from common.visualize import viz_loggings
from common.io import write_config
from common.seed import set_seed


@hydra.main(config_path=f"{ROOT_DIR}/configs", config_name="config")
def train(configs: DictConfig):
    """
    Training pipeline
    :param configs: configuration
    :return:
    """
    print("--- CONFIGS ---")
    print(OmegaConf.to_yaml(configs))
    print("--- TRAINING ---")
    data_folder = configs.dataset.folder_path
    print("Dataset:", data_folder)
    print("----------------")

    # set random seed
    set_seed(number=configs.model.random.seed_number)

    # --- model ---
    model = MP(configs)
    # --- dataset ---
    dataset = CarlaDataset(configs, data_folder)

    # --- data loader ---
    train_ratio = configs.model.train.train_ratio
    len_train = int(len(dataset) * train_ratio)
    len_val = len(dataset) - len_train
    train_data, val_data = random_split(dataset, (len_train, len_val))

    batch_size = configs.model.train.batch_size
    num_workers = configs.model.train.max_workers
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers
    )

    # --- train ---
    epochs = configs.model.train.epochs
    num_gpus = configs.model.train.num_gpus
    precision = configs.model.train.precision
    save_path = os.path.join(ROOT_DIR, configs.model.train.save_path)
    # create save_folder if not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # checkpoint model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=save_path,
        filename="mp-pp_{epoch:02d}_{val_loss:.5f}",
        save_top_k=3,
        mode="min",
    )
    # training
    trainer = pl.Trainer(
        gpus=num_gpus, precision=precision, max_epochs=epochs,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    # --- visualize curves ---
    viz_loggings(
        logging=model.loggings["avg"],
        save_path=save_path
    )
    # --- log file config ---
    write_config(
        config=OmegaConf.to_container(configs, resolve=True),
        save_path=save_path
    )


if __name__ == "__main__":
    train()
