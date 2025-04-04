from typing import Any, Tuple, Union, Generator
import torch
import gin  # type: ignore
import numpy as np
import tensorflow as tf
from chord.dataloader.main import dataloader_tf_2segments
from chord.dataloader.main_supervised import dataloader_tf_supervised


@gin.configurable
class Dataloader(torch.utils.data.IterableDataset):  # type: ignore
    def __init__(
        self,
        split_set: str,
        path: str,
        device: str,
        dataset_type: str,
        test_percent: float,
        buffer_size: int,
        batch_size: int,
        sr: int,
        duration: int,
        step_percent: float,
    ):
        assert dataset_type == "mix" or dataset_type == "unsupervised" or dataset_type == "supervised"
        super(Dataloader, self).__init__()
        self.split_set = split_set
        self.device = torch.device(device)
        self.test_percent = test_percent
        self.batch_size = batch_size
        self.device = device
        self.duration = duration
        self.sr = sr
        self.step_percent = step_percent
        self.buffer_size = buffer_size
        self.dataset_type = dataset_type
        if self.dataset_type == "mix":
            self.tf_dataloader_unsupervised = dataloader_tf_2segments(
                    path=path,
                    split_set=split_set,
                    test_percent=test_percent,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    sr=sr,
                    duration=duration,
                    step_percent=step_percent,
                    )
            self.tf_dataloader_supervised = dataloader_tf_supervised(
                    path="/data/nfs/analysis/interns/ykong/stone/files/gs_annotations.json",
                    split_set=split_set,
                    test_percent=test_percent,
                    batch_size=batch_size,
                    buffer_size=int(buffer_size),
                    sr=sr,
                    duration=duration,
                    step_percent=step_percent,
                    )
            datasets = [self.tf_dataloader_unsupervised, self.tf_dataloader_supervised]
            if self.split_set == "train":
                order = tf.constant([0, 0, 1], dtype=tf.int64)
                choice_dataset = tf.data.Dataset.from_tensor_slices(order).repeat()
                self.ds_mix = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)
            else:
                order = tf.constant([0, 0, 1], dtype=tf.int64)
                choice_dataset = tf.data.Dataset.from_tensor_slices(order).repeat()
                self.ds_mix = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)
        elif self.dataset_type == "supervised":
            self.ds_mix = dataloader_tf_supervised(
                    path="/data/nfs/analysis/interns/ykong/stone/files/gs_annotations.json",
                    split_set=split_set,
                    test_percent=test_percent,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    sr=sr,
                    duration=duration,
                    step_percent=step_percent,
                    )
        else:
            self.ds_mix = dataloader_tf_2segments(
                    path=path,
                    split_set=split_set,
                    test_percent=test_percent,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    sr=sr,
                    duration=duration,
                    step_percent=step_percent,
                    )

    def __iter__(self) -> Generator[tf.Tensor, None, None]:
        for batch in self.ds_mix.as_numpy_iterator():
            audio = torch.from_numpy(np.copy(batch["segments"]))
            key_signature = np.copy(batch["key_signature"]).tolist(),
            keymode = np.copy(batch["keymode"]).tolist(),

            output = {
                "audio": audio.to(self.device, non_blocking=True),
                "key_signature": key_signature,
                "keymode" : keymode
                }
            yield output


@gin.configurable  # type: ignore
def get_datasets(
        device: str,
        path:str,
        dataset_type_train: str,
        dataset_type_test: str,
        test_percent: int,
        batch_size: int,
        sr: int,
        duration: int,
        step_percent: float,
        buffer_size: int,
        val_steps: int,
        run_val: bool=True,
) -> Tuple[Any, Any]:
    print("--- val_steps ---- ", val_steps)
    ds_train = Dataloader(
        split_set="train", 
        device=device, 
        path=path, 
        dataset_type=dataset_type_train,
        test_percent=test_percent, 
        batch_size=batch_size, 
        sr=sr, 
        duration=duration, 
        step_percent=step_percent, 
        buffer_size=buffer_size,
    )
    ds_test = Dataloader(
        split_set="test",
        device=device,
        path=path,
        dataset_type=dataset_type_test,
        test_percent=test_percent, 
        batch_size=batch_size, 
        sr=sr,
        duration=duration, 
        step_percent=step_percent, 
        buffer_size=buffer_size,
    )
    ds_test.ds_mix = (
        ds_test.ds_mix.take(val_steps)
        .cache("/tmp/validation_{}_set_cache_{}".format(dataset_type_test, device))
        .repeat()
    )

    if run_val:
        print("--- Runing the ds_test to cache in memory before training ---")
        _ = [None for _ in ds_test.ds_mix.take((val_steps * 2) + 1)]
        print("--- FINISHED ---")

    return ds_train, ds_test
