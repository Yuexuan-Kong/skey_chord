from typing import Any, Dict, Generator, Tuple, Iterator, Union
import gin  # type: ignore
import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore
import tensorflow_io as tfio  # type: ignore
import torch
import torchaudio  # type: ignore
import json
import random
from einops import rearrange

@tf.function(experimental_relax_shapes=True)  # type: ignore
def song_path_from_ID(
    data: Dict[str, Any], 
) -> Dict[str, Any]:
    """
    Extract song path from a ID identifier from deezer database 
    """

    def py_separate_song_path_ID(ID: tf.Tensor) -> str:
        ID = ID.numpy().decode("utf-8")
        return "/data/nfs/analysis/interns/ykong/data/giantsteps-mtg-key-dataset/audio/{}.LOFI.mp3".format(ID)

    return {
        "ID": data["ID"],
        "keymode": data["keymode"],
        "mode": data["mode"],
        "key_signature": data["key_signature"],
        "song_path": tf.py_function(
            py_separate_song_path_ID, [data["ID"]], (tf.string)
        ),
    }


def split_set_fn(
    data: Dict[str, Any],
    split_set: str,
    test_percent: float,
    split_key: str = "ID",
    factor: int = 10,
) -> Any:
    assert split_set in ["train", "test"]
    if split_set == "test":
        output = (
            tf.strings.to_hash_bucket(data[split_key], 100 * factor)
            < test_percent * factor
        )
    else:
        output = (
            tf.strings.to_hash_bucket(data[split_key], 100 * factor)
            >= test_percent * factor
            and
            tf.strings.to_hash_bucket(data[split_key], 100 * factor)
            <= test_percent * 10 * factor
        )
    return output


@tf.function(reduce_retracing=True)  # type: ignore
def load_audio(
    data: dict[str, Any],
    sr: int,
    do_norm: bool = True,
) -> dict[str, Any]:
    def py_load_audio(
        song_path: tf.string, sr: tf.int32
    ) -> tuple[npt.NDArray[np.float64], int]:
        x, sr_in = torchaudio.load(
            song_path.numpy().decode("utf-8"),
            frame_offset = sr*10*torch.rand(1),
            channels_first=False,
        )
        x = torch.mean(x, dim=1).unsqueeze(-1)
        if sr_in != sr:
            x = torchaudio.transforms.Resample(sr_in, sr.numpy(), dtype=x.dtype)(x.T).T
        return x.numpy()

    audio = tf.py_function(
        py_load_audio,
        [data["song_path"], sr],
        (tf.float32),
    )

    if do_norm:
        audio = tf.where(
            tf.reduce_max(tf.abs(audio), keepdims=True) != 0,
            x=tf.divide(audio, tf.reduce_max(tf.abs(audio), keepdims=True)),
            y=audio,
        )
    return {"audio": audio, "duration": float(len(audio)) / float(sr), "key_signature": data["key_signature"], "keymode": data["keymode"], "mode": data["mode"]}


def read_json_file(
    path_json: str, 
) -> Iterator[Dict[str, Any]]:
    with open(path_json, 'r') as json_file:
        data = json.load(json_file)
    for i in data:
        yield {"ID": i["ID"], "key_signature": i["key_signature"], "keymode": i["MANUAL_KEY"], "mode": i["mode"]}


def yield_data(path: str, split_set: str, test_percent: float) -> tf.data.Dataset:
    taxonomy = {
        "ID": tf.TensorSpec(shape=(), dtype=tf.string),
        "key_signature": tf.TensorSpec(shape=(), dtype=tf.string),
        "keymode": tf.TensorSpec(shape=(), dtype=tf.string),
        "mode": tf.TensorSpec(shape=(), dtype=tf.string),
    }
    ds = (
        tf.data.Dataset.from_generator(
            read_json_file, output_signature=taxonomy, args=[path]
        )
        .filter(lambda data: split_set_fn(data, split_set, test_percent))
    )
    return ds

def chunk_audio(
    audio: npt.NDArray[np.float32],
    duration: float,
    sr: int,
    step_percent: float,
) -> tf.Tensor:
    def seconds_to_samples(x: float, sr: int) -> tf.Tensor:
        return tf.cast(x * tf.cast(sr, tf.float32), tf.int32)

    x = tf.signal.frame(
        tf.transpose(audio, [1, 0]),
        frame_length=seconds_to_samples(duration, sr),
        frame_step=seconds_to_samples(duration * step_percent, sr),
    ) # x of shape (source, num_segments, duration)
    x = tf.transpose(x, [1, 2, 0])

    return x


def select_no_silence_frames(
    audio: tf.Tensor,
    segments: tf.Tensor,
    dbs_threshold: float = -7.5,
) -> tf.Tensor:
    rms_ref = lambda x: tf.math.sqrt(tf.reduce_mean(tf.math.pow(x, 2)))
    rms_segments = lambda x: tf.math.sqrt(
        tf.reduce_mean(tf.math.pow(x, 2), axis=[-1, -2])
    )
    py_mask_audio_fn = lambda x, mask: tf.boolean_mask(x, mask)
    # non "silence" segments
    dbs = 10 * tf.math.log(rms_segments(segments) / rms_ref(audio))  # type: ignore
    # "silence" = segments with rms values below dbs_threshold
    mask = tf.math.greater(dbs, dbs_threshold)
    indices = tf.py_function(
        py_mask_audio_fn, [tf.range(tf.shape(segments)[0]), mask], (tf.int32)
    )
    segments = tf.py_function(py_mask_audio_fn, [segments, mask], (tf.float32))
    indices_tmp = tf.random.shuffle(tf.range(tf.shape(indices)[0]))
    segments = tf.gather(segments, indices_tmp, axis=0)
    indices = tf.gather(indices, indices_tmp, axis=0)
    return segments


@gin.configurable  # type: ignore
def preproces_for_views(
    data: Dict[str, Any],
    duration: float,
    sr: int,
    step_percent: float,
) -> Dict[str, Any]:
    output = {}
    # AUDIO SEGMENTS
    # load twice more segments from major than minor
    if data["mode"] == "major":
        output["segments"] = chunk_audio(
            data["audio"], duration, sr, step_percent/2
        )
    else:
        output["segments"] = chunk_audio(
            data["audio"], duration, sr, step_percent
        )
    # output["segments"], length = select_no_silence_frames(
    #     data["audio"], output["segments"]
    # )
    output["segments"] = select_no_silence_frames(
        data["audio"], output["segments"]
    )
    output["key_signature"] = tf.repeat(data["key_signature"], repeats=tf.shape(output["segments"])[0])
    output["keymode"] = tf.repeat(data["keymode"], repeats=tf.shape(output["segments"])[0])
    return output


@gin.configurable  # type: ignore
def dataloader_tf_supervised(
    path: str,
    split_set: str,
    sr: int,
    duration: float,
    test_percent: float,
    batch_size: int,
    buffer_size: int,
    step_percent: float,
):
    """
    duration: defines the selected segment durations
    context_dur: defines the area duration from which we can pool positive candidates given the true one
    far_from_dur: defines the area duration from which we cannot pool positive candidates given the true one, a.k.a avoid close ones
    """
    assert batch_size > 0
    assert buffer_size is not None and buffer_size > 0

    print("--- Loading two segments from an audio ---")

    ds = (
        yield_data(path, split_set, test_percent)
        .shuffle(299, reshuffle_each_iteration=True)
        .map(song_path_from_ID)
        .map(
            lambda data: load_audio(data, sr),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda data: preproces_for_views(
                data, duration, sr, step_percent
            )
        )
        .unbatch()
        .shuffle(buffer_size, reshuffle_each_iteration=True)
    )
    ds = (
        ds.repeat()
        .batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        # .prefetch(tf.data.AUTOTUNE)
    )
    return ds

