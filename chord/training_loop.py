import gc
import os
from pathlib import Path
from typing import Any, List

import gin  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from chord.model.hcqt import HarmonicVQT
from chord.stone_loss import CrossPowerSpectralDensityLoss
from chord.utils.gin import get_save_dict
import tensorflow as tf  # type: ignore
import torch
from tensorflow.keras.utils import Progbar  # type: ignore
from ssl_framework.dataloader import get_datasets

from chord.callbacks import (
    EarlyStoppingCallback,
    NaNLoopCallback,
    add_audio_tensorboard,
    add_losses_tensorboard,
    get_writer,
    restart_from_checkpoint,
    save_fn,
)
from chord.stone import (
    Stone,
)
from chord.utils.gin import parse_gin
from chord.utils.parallel import (
    cleanup,
    set_gpus
)
from chord.utils.scheduler import (
    get_learning_rate_scheduler,
    get_weights_decay_scheduler,
)
from chord.utils.training import clip_gradients, get_optimizer, update_optimizer


def create_save_dir(save_dir: str, name: str, circle_type: int) -> str:
    save_dir = os.path.join(save_dir, "models", str(circle_type), name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print("PARAMETERS used for SAVING:")
    print("\t save_model_dir: {}".format(save_dir))
    print("\t exp_name: {}".format(name))
    return save_dir


class ModelCustomWrapper:
    def __init__(
        self,
        learning_rate: float,
        device: str,
        n_steps: int,
        n_epochs: int,
        circle_type: int,
    ) -> None:

        self.device = torch.device(device)
        self.lr = learning_rate
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.current_epoch = 0
        self.circle_type = circle_type

        # MODELS
        self.stone = Stone(HarmonicVQT(), device=self.device).to(device)

        # LOSS
        self.loss_fn = CrossPowerSpectralDensityLoss(self.circle_type, self.device).cuda(self.device)
        # self.loss_fn = CrossEntropyLoss().cuda(self.device)

        # OPTIMIZER
        self.optimizer = get_optimizer(self.stone)
        self.scaler = torch.cuda.amp.GradScaler()  # type: ignore

        # TRAINING STEPS
        self.step = (
            lambda self, batch: self.loss_fn(self.stone(batch))
            )

        # SCHEDULES
        self.lr_schedule = get_learning_rate_scheduler(
            self.lr, self.n_epochs, self.n_steps
        )
        self.wd_schedule = get_weights_decay_scheduler(self.n_epochs, self.n_steps)

    def training_step(
        self, batch: Any, current_global_step: int, current_epoch: int
    ) -> Any:
        self.current_global_step = current_global_step
        self.current_epoch = current_epoch
        lr_step = self.lr_schedule[self.current_global_step]
        wd_step = self.wd_schedule[self.current_global_step]

        # update weight decay and learning rate according to their schedule
        self.optimizer = update_optimizer(self.optimizer, lr_step, wd_step)
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # POWERFUL
            loss = self.step(self, batch)
        self.scaler.scale(loss["loss"]).backward()

        # unscale the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer)
        self.stone = clip_gradients(self.stone)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # removing data
        torch.cuda.empty_cache()
        del batch

        return loss["loss_to_print"]

    def validation_step(self, batch: Any) -> Any:
        with torch.no_grad():
            loss = self.step(self, batch)
        return loss["loss_to_print"]


def do_one_iter(
    model: ModelCustomWrapper,
    ds_train_iter: Any,
    ds_val_iter: Any,
    epoch: int,
    n_steps: int,
    val_steps: int,
    progress_bar: Any,
    writer: Any,
) -> float:
    # --- TRAINING ---
    model.stone.train()
    for i in range(n_steps):
        current_global_step = i + epoch * n_steps
        train_batch = next(ds_train_iter)
        train_loss = model.training_step(
            batch=train_batch,
            current_epoch=epoch,
            current_global_step=current_global_step,
        )
        progress_bar.update(
                i, [
                    ("train_total", train_loss["loss_total"].item()), 
                    # ("train_pos", train_loss["loss_pos"].item()), 
                    ("train_equi", train_loss["loss_equi"].item()), 
                    # ("train_distribution", train_loss["loss_distribution"].item()), 
                    # ("train_mode", train_loss["loss_mode"].item())
                    ]
                )
        # if i % 50 == 0:
        #     add_audio_tensorboard(writer, train_batch, "train", epoch)

    # --- VAL ---
    model.stone.eval()

    for _ in range(val_steps):
        val_batch = next(ds_val_iter)
        val_loss = model.validation_step(batch=val_batch)
        progress_bar.update(
                n_steps, [
                    ("val_total", val_loss["loss_total"].item()), 
                    # ("val_pos", val_loss["loss_pos"].item()), 
                    ("val_equi", val_loss["loss_equi"].item()), 
                    # ("val_distribution", val_loss["loss_distribution"].item()), 
                    # ("val_mode", val_loss["loss_mode"].item())
                    ]
                )

    val_loss_ckpt = add_losses_tensorboard(writer, progress_bar, epoch)

    # Clean memeory
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    torch.cuda.empty_cache()
    plt.close()
    _ = gc.collect()
    return val_loss_ckpt

@gin.configurable
def main_loop(
    n_epochs: int,
    n_steps: int,
    val_steps: int,
    learning_rate: float,
    gin_file: str,
    save_dir: str,
    name: str,
    circle_type: int,
    save_epochs: List = [25, 50, 75, 100, 150, 250, 500],
    ) -> None:
    device = set_gpus()
    save_dict = get_save_dict()
    save_dict["gin_info"] = parse_gin(gin_file)
    print(" ------- CREATING model -----------")
    model = ModelCustomWrapper(
        learning_rate=learning_rate,
        device=device,
        n_steps=n_steps,
        n_epochs=n_epochs,
        circle_type=circle_type,
    )
    # DATALOADERS
    print(" ------- CREATING datasets -----------")
    ds_train, ds_val = get_datasets(
        device=device,
        val_steps=val_steps,
        sr=16000,
        duration=4
    )
    save_dict["audio"] = {
        "dur": int(ds_train.duration),
        "sr": ds_train.sr,
    }
    save_dir = create_save_dir(save_dir, name, circle_type)
    writer = get_writer(save_dir)
    model, save_dict = restart_from_checkpoint(model, save_dict, save_dir)
    early_stopping = EarlyStoppingCallback(save_dir, best_score=save_dict["val_loss"])
    nan_loop = NaNLoopCallback(save_dir)
    ds_train_iter = iter(ds_train)
    ds_val_iter = iter(ds_val)
    epoch, val_loss_ckpt = [save_dict["epoch"], save_dict["val_loss"]]
    
    while epoch < n_epochs:
        print("\nepoch {}/{}".format(epoch + 1, n_epochs))
        # Training loop
        val_loss_ckpt = do_one_iter(
            model=model,
            ds_train_iter=ds_train_iter,
            ds_val_iter=ds_val_iter,
            epoch=epoch,
            n_steps=n_steps,
            val_steps=val_steps,
            progress_bar=Progbar(n_steps + 1),
            writer=writer,
        )
        model, ds_train, save_dict, epoch = nan_loop(
            False, model, save_dict, ds_train, n_steps, epoch
        )
        # epoch is updated inside nan_loop
        save_dict["epoch"], save_dict["val_loss"] = [epoch, val_loss_ckpt]
        early_stopping(val_loss_ckpt, model, save_dict, epoch)
        save_fn(save_dict, model, os.path.join(save_dir, "last_iter.pt"))
        if epoch in save_epochs:
            save_fn(
                save_dict, model, os.path.join(save_dir, "epoch_{}.pt".format(epoch))
            )
    writer.close()
    cleanup()
    return
