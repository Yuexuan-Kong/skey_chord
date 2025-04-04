import argparse
import glob
import math
import os
from typing import Any, Dict, Iterator, List, Tuple
import logging
import gin  # type: ignore

from chord.utils.gin import update_gin, gin_info_list_to_dict

import numpy as np
import numpy.typing as npt
import torch
import torch.nn
import torchaudio
from chord.model.chromanet import ChromaNet
from chord.model.hcqt import HarmonicVQT, CropCQT
import tensorflow as tf
import tensorflow_io as tfio
from einops import rearrange
from torch import Tensor
from torch.utils.data import IterableDataset
from tqdm import tqdm  # type: ignore

from chord.stone import Stone
from chord.utils.parallel import set_cpu, set_gpus


def yield_ids(song_path: str) -> Iterator[Dict[str, Any]]:
    for idx in np.random.permutation(len(song_path)):
        yield {"idx": idx, "song_path": song_path[idx]}


def load_checkpoint(ckpt_path: str, gin_file: str, save_dict: Dict[str, Any]) -> Any:
    # load checkpoint
    logging.info("Loading checkpoint from: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    logging.info("Loading checkpoint done!")
    logging.info("GIN INFO FROM CHECKPOINT: {}".format(gin_file))
    for i in ckpt["gin_info"]:
        logging.info("\t {}".format(i))
    ckpt["gin_info"] = gin_info_list_to_dict(ckpt["gin_info"])
    return ckpt


# @gin.configurable  # type: ignore
@tf.function(experimental_relax_shapes=True)  # type: ignore
def load_audio_demucs(
    data: Dict[str, Any],
    sr: int,
    source: str,
    mono: bool = True,
    do_norm: bool = True,
) -> Dict[str, Any]:
    """
    Load separated stems (by demucs) from path.
    """
    def py_load_separated_audio(
        song_path: tf.string,
        sr: tf.int32,
    ) -> Tuple[npt.NDArray[np.float32], int, float]:
        try:
            x, sr_in = torchaudio.load(
                song_path.numpy().decode("utf-8"),
                channels_first=False,
            )
        except:
            sr_in = sr
            x = torch.zeros(1, 2)
            print(song_path)
        x = torch.mean(x, dim=1).unsqueeze(1)
        if sr_in != sr:
            x = torchaudio.transforms.Resample(sr_in, sr.numpy(), dtype=x.dtype)(x.T).T
        if mono:
            x = torch.mean(x, dim=1).unsqueeze(-1)

        return x.numpy()

    x = tf.py_function(
        py_load_separated_audio,
        [data["song_path"], sr],
        (tf.float32),
    )

    if do_norm:
        # normalize audios according to the max of the track of the source
        x = tf.where(
            tf.reduce_max(tf.abs(x), keepdims=True) != 0,
            x=tf.divide(x, tf.reduce_max(tf.abs(x), keepdims=True)),
            y=x,
        )

    return {"audio": x}


class RelativePitchProfile(IterableDataset):  # type: ignore
    def __init__(
        self,
        device: str,
        song_path: List,
        source: str,
        sr: int,
        mono: bool=True,
        do_norm: bool = True,
    ) -> None:
        """
        Args:
            path (List):
                Paths to audios.
            sr (int):
                sample rate.
            mono (bool):
                Self-explanatory.
            model_duration_seconds (float):
                Maximum audio load duration in seconds.
        """
        self.song_path = song_path
        self.sr = sr
        self.device = device

        taxonomy = {
            "song_path": tf.TensorSpec(shape=(), dtype=tf.string),
            "idx": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        self.tf_dataloader = (
                tf.data.Dataset.from_generator(
                    yield_ids,
                    output_signature=taxonomy,
                    args=[self.song_path],
                    )
                .map(lambda x: (load_audio_demucs(x, sr, source,mono, do_norm), x["song_path"]))
                .filter(lambda x, y: tf.reduce_max(tf.abs(x["audio"])) != 0)
                )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get audio and accompanying tags.
        ---
        Returns:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                Waveform:
                    In train setting: shape is [batch_size, model_input_dur, num_channels]
                    In test setting: shape is [batch_size, track_num_samples, num_channels]
                Tag:
                    Shape is [batch_size, num_tags]
        """
        for data, song_path in self.tf_dataloader.as_numpy_iterator():
            yield (torch.Tensor(data["audio"]).to(self.device), song_path.decode())


def get_embedding(
        sr: int,
        dur: int,
        hcqt: HarmonicVQT,
        chromanet: ChromaNet, 
        crop_fn: CropCQT,
        batch: torch.Tensor, 
        song_path: str,
        overlap: float=False, 
        average: bool=True, 
    ) -> Tensor:
    load_len = batch.shape[-2]
    if overlap:
        new_batch = tf.signal.frame(tf.transpose(batch), sr * dur, math.floor(sr * dur*(1-overlap))).numpy()
        new_batch = rearrange(
                new_batch,
                "c b f -> b c f"
                )
        new_batch = torch.from_numpy(new_batch)
    else:
        # num_batches = math.floor(load_len / (sr * dur))
        # s = int(sr * dur)
        # if num_batches != 0:
        #     proc_duration = int(sr * dur * num_batches)
        #     new_batch = batch
        #     new_batch = new_batch[:proc_duration, :]
        #     new_batch = rearrange(
        #         new_batch,
        #         "(b s) c -> b c s",
        #         b=num_batches,
        #         s=s,
        #     )
        # else: 
        new_batch = batch.permute(1, 0).unsqueeze(dim=0)
    with torch.no_grad():
        after_crop = crop_fn(hcqt(new_batch), torch.zeros(len(new_batch, )))
        try:
            yh = chromanet(after_crop)
            return yh if not average else torch.mean(yh, dim=0).argmax()
        except:
            print(after_crop.shape)
            return torch.Tensor([-1])            


def main(ckpt_path: str, audio_path: str, extension: str, overlap:float, average:bool) -> None:
    device = set_cpu()
    ckpt = load_checkpoint(ckpt_path, "", {})
    dur, sr = [ckpt["audio"]["dur"], ckpt["audio"]["sr"]]
    num_frames = dur*sr
    model_name = "-".join(ckpt_path.split("/")[-5:-1])
    hcqt = HarmonicVQT(
        harmonics=eval(ckpt["gin_info"]["HarmonicVQT.harmonics"]),
        fmin=float(ckpt["gin_info"]["HarmonicVQT.fmin"]), 
        n_bins=int(ckpt["gin_info"]["HarmonicVQT.n_bins"]), 
    ).to(device)
    chromanet = ChromaNet(
        n_bins = int(ckpt["gin_info"]["Stone.n_bins"]),
        n_harmonics=len(eval(ckpt["gin_info"]["HarmonicVQT.harmonics"])), 
        out_channels=eval(ckpt["gin_info"]["Stone.out_channels"]),
        kernels=eval(ckpt["gin_info"]["Stone.kernels"]),
        temperature=float(ckpt["gin_info"]["Stone.temperature"])
    ).to(device)
    hcqt_state_dict = {
        k.replace("hcqt.", ""): v
        for k, v in ckpt["stone"].items()
        if "hcqt" in k
    }
    ckpt["stone"] = {
        k.replace("chromanet"+".", ""): v
        for k, v in ckpt["stone"].items()
        if "chromanet" in k
    }
    hcqt.load_state_dict(hcqt_state_dict)
    hcqt.eval()
    chromanet.load_state_dict(ckpt["stone"])
    chromanet.eval()
    crop_fn = CropCQT(int(ckpt["gin_info"]["Stone.n_bins"]))
    print(
        f"\n\n Computing the relative pitch profile class for {audio_path} using the model {ckpt_path} \n"
    )
    ds = RelativePitchProfile(device, glob.glob("{}/**/*.{}".format(audio_path, extension), recursive=True), num_frames, sr)
    pbar = tqdm(iter(ds))
    results = {
        song_path: get_embedding(sr, dur, hcqt, chromanet, crop_fn, test_batch, song_path, overlap, average).cpu().numpy()
        for test_batch, song_path in pbar
    }

    path_results = os.path.join(audio_path, "relative_pitch_profile", model_name) 
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    np.savez(os.path.join(path_results, "values_final.npz"), **results)


def input_args() -> None:
    parser = argparse.ArgumentParser(
        description="Relative pitch class profile", add_help=True
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the base model checkpoint",
    )
    parser.add_argument("audio_path", help="Path to the audios", type=str)
    parser.add_argument(
        "-e", "--extension", type=str, default="wav", help="audio format extension"
    )
    parser.add_argument(
        "-o", "--overlap", type=float, default=False, help="the ratio of overlap portion of the moving window"
    )
    parser.add_argument(
        "-a", "--average", default=True, action="store_true", help="if calculating the average over time axis or not"
    )
    parser.add_argument(
        "-p", "--model_part", type=str, default="acc", help="accompaniment or vocal for chromanet"
            )

    args = parser.parse_args()

    main(
        os.path.abspath(args.checkpoint_path),
        os.path.abspath(args.audio_path),
        args.extension,
        args.overlap,
        args.average,
    )


if __name__ == "__main__":
    input_args()
