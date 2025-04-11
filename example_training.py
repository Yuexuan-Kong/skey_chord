from typing import Any, List

import torch
from chord.model.hcqt import HarmonicVQT
from chord.stone import Stone
from chord.stone_loss import CrossPowerSpectralDensityLoss

from chord.utils.scheduler import (
    get_learning_rate_scheduler,
    get_weights_decay_scheduler,
)
from chord.utils.training import clip_gradients, get_optimizer, update_optimizer

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
    # ds_train_iter: Any,
    # ds_val_iter: Any,
    supervised_val_dataloder: Any,  # new
    synth_dataloader_iter: Any,  # new
    epoch: int,
    n_steps: int,
    synth_ratio: float,  # new
    val_steps: int,
    # progress_bar: Any,
    # writer: Any,
) -> float:
    # --- TRAINING ---
    import random
    model.stone.train()
    for i in range(n_steps):
        current_global_step = i + epoch * n_steps
        if random.random() < synth_ratio:
            # Use synth data
            train_batch = next(synth_dataloader_iter)
        else:
            continue  # Use supervised data, here we just skip the training
            train_batch = next(ds_train_iter)
        x = train_batch["x"]
        y_root = train_batch["y_root"]
        y_quality = train_batch["y_quality"]
        continue  # Compute training loss, etc., can be done in the model custom wrapper
        
    # --- VAL ---
    # SSL validation skipped
    model.stone.eval()

    # Supervised validation
    for data in supervised_val_dataloder:
        x = train_batch["x"]
        y_root = train_batch["y_root"]
        y_quality = train_batch["y_quality"]
        continue  # Compute validation loss, root and quality acc. etc., can be done in the model custom wrapper


    # val_loss_ckpt = add_losses_tensorboard(writer, progress_bar, epoch)

    # # Clean memeory
    # tf.keras.backend.clear_session()
    # tf.compat.v1.reset_default_graph()
    # torch.cuda.empty_cache()
    # plt.close()
    # _ = gc.collect()
    # return val_loss_ckpt

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

    device = "cpu"
    epoch = 0
    from chord.dataloader.supervised_data import get_dataloader as get_supervised_val_dataloader
    from chord.dataloader.synth_data import get_synth_dataloader

    print(" ------- CREATING model -----------")
    model = ModelCustomWrapper(
        learning_rate=learning_rate,
        device=device,
        n_steps=n_steps,
        n_epochs=n_epochs,
        circle_type=circle_type,
    )
    # DATALOADERS
    # Here are the new dataloders for supervised validation and synth data
    # The original code (ssl dataloader) are not included here
    print(" ------- CREATING datasets -----------")
    supervised_val_dataloder = get_supervised_val_dataloader(
        audio_folder="../datasets/Schubert_Winterreise_Dataset_v2-1/01_RawData/audio_wav/",  # To actual audio_wav path
        label_folder="../datasets/Schubert_Winterreise_Dataset_v2-1/02_Annotations/ann_audio_chord/",  # To actual annotation path
        dataset_name="swd", # Don't change
        batch_size=16,  # Batch size
        num_workers=0,  # Number of workers
        label_fps=6.25,  # Label frames per second, with hop size = 0.01s, downsample * 16 gives fps = 1 / (0.01 * 16) = 6.25
        sr=22050,  # Resample on the fly or pre-process
        seg_length=10.0,  # Should not matter
        seg_shift_length=10.0,  # Should not matter
    )

    synth_dataloader = get_synth_dataloader(
        label_fps=6.25,  # Label frames per second, with hop size = 0.01s, downsample * 16 gives fps = 1 / (0.01 * 16) = 6.25
        sr=16000,  # Sample rate - the same as SSL data
        seg_length=4,  # Length of the segment
        batch_size=16,  # Batch size
        num_workers=0  # Number of workers
    )
    
    while epoch < n_epochs:
        print("\nepoch {}/{}".format(epoch + 1, n_epochs))
        # Training loop
        val_loss_ckpt = do_one_iter(
            model=model,
            # ds_train_iter=ds_train_iter,
            # ds_val_iter=ds_val_iter,
            supervised_val_dataloder=supervised_val_dataloder,  # new
            synth_dataloader_iter=iter(synth_dataloader),  # new
            epoch=epoch,
            n_steps=n_steps,
            synth_ratio=0.1,  # new, Ratio of synth data,
                              # use ratio instead of steps so everything related to n_steps don't need to change
                              # Can refactor if needed
            val_steps=val_steps,  # Not used here
                                  # The supervised val always iter all data
            # progress_bar=Progbar(n_steps + 1),
            # writer=writer,
        )
        epoch += 1
    # ------------- BELOW KEEP THE SAME AS THE ORIGINAL CODE -------------
    #     model, ds_train, save_dict, epoch = nan_loop(
    #         False, model, save_dict, ds_train, n_steps, epoch
    #     )
    #     # epoch is updated inside nan_loop
    #     save_dict["epoch"], save_dict["val_loss"] = [epoch, val_loss_ckpt]
    #     early_stopping(val_loss_ckpt, model, save_dict, epoch)
    #     save_fn(save_dict, model, os.path.join(save_dir, "last_iter.pt"))
    #     if epoch in save_epochs:
    #         save_fn(
    #             save_dict, model, os.path.join(save_dir, "epoch_{}.pt".format(epoch))
    #         )
    # writer.close()
    # cleanup()
    return

if __name__ == "__main__":
    main_loop(
        n_epochs=1,
        n_steps=100,
        val_steps=100,
        learning_rate=0.001,
        gin_file="",
        save_dir="",
        name="test",
        circle_type=7,
        save_epochs=[25, 50, 75, 100, 150, 250, 500],
    )