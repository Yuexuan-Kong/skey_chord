from typing import Any, Dict, Generator, Tuple, Iterator
import gin  # type: ignore
import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore
import tensorflow_io as tfio  # type: ignore
import torch
import torchaudio  # type: ignore
import json

@tf.function(experimental_relax_shapes=True)  # type: ignore
def song_path_from_md5(
    data: Dict[str, Any], 
) -> Dict[str, Any]:
    """
    Extract song path from a md5 identifier from deezer database 
    """
    def py_separate_song_path_md5(md5: tf.Tensor) -> str:
        song_path = ""
        try:
            md5 = md5.numpy().decode("utf-8")
            song_path = "/data/music/output/{}/{}/{}/{}/{}.mp3".format(
                "mp3_128", md5[0], md5[1], md5[2], md5
            )
        except:
            pass
        return song_path

    return {
        "md5": data["md5"],
        # "PRODUCT_TRACK_ID": data["PRODUCT_TRACK_ID"],
        "PRODUCT_TRACK_ID": data["PRODUCT_TRACK_ID"],
        "song_path": tf.py_function(
            py_separate_song_path_md5, [data["md5"]], (tf.string)
        ),
    }

def whole_song_path_from_md5(
    data: Dict[str, Any], quality: str = "mp3_128"
) -> Dict[str, Any]:
    def py_song_path_from_md5(md5: tf.Tensor, quality: tf.Tensor) -> str:
        song_path = ""
        try:
            md5 = md5.numpy().decode("utf-8")
            quality = quality.numpy().decode("utf-8")
            song_path = "/data/music/output/{}/{}/{}/{}/{}.mp3".format(
                quality, md5[0], md5[1], md5[2], md5
            )
        except:
            pass
        return song_path

    return {
        "song_path": tf.py_function(
            py_song_path_from_md5, [data["md5"], quality], (tf.string)
        )
    }

def split_set_fn(
    data: Dict[str, Any],
    split_set: str,
    test_percent: float,
    split_key: str = "PRODUCT_TRACK_ID",
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
        )
    return output


@tf.function(reduce_retracing=True)  # type: ignore
def load_audio(
    data: Dict[str, Any],
    sr: int,
    do_norm: bool = True,
) -> Dict[str, Any]:

    def py_load_audio(
        song_path: tf.string, sr: tf.int32
    ) -> Tuple[npt.NDArray[np.float64], int]:
        try:
            x, sr_in = torchaudio.load(
                song_path.numpy().decode("utf-8"),
                channels_first=False,
            )
            x = torch.mean(x, dim=1).unsqueeze(-1)
        except:
            sr_in = sr
            x = torch.zeros(2*sr, 2)

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
    return {"audio": audio, "duration": float(len(audio)) / float(sr)}

def add_item(
    data: Dict[str, Any], add_key: str, add_value: Any
) -> Dict[str, Any]:
    data[add_key] = add_value 
    return data


def read_json_file(
    path_json: str, 
) -> Iterator[Dict[str, Any]]:
    with open(path_json, 'r') as json_file:
        data = json.load(json_file)
    for i in data:
        yield {"PRODUCT_TRACK_ID": i["PRODUCT_TRACK_ID"], "md5": i["md5"]}


def yield_data(path: str, split_set: str, test_percent: float) -> tf.data.Dataset:
    taxonomy = {
        "PRODUCT_TRACK_ID": tf.TensorSpec(shape=(), dtype=tf.string),
        "md5": tf.TensorSpec(shape=(), dtype=tf.string),
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
) -> Tuple[tf.Tensor, tf.Tensor]:
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
    return (segments, indices)


@gin.configurable  # type: ignore
def preproces_for_views(
    data: Dict[str, Any],
    duration: float,
    context_dur: float,
    far_from_dur: float,
    sr: int,
    step_percent: float,
) -> Dict[str, Any]:
    output = {}
    # AUDIO SEGMENTS
    output["segments"] = chunk_audio(
        data["audio"], duration, sr, step_percent
    )
    output["segments"], output["indices"] = select_no_silence_frames(
        data["audio"], output["segments"]
    )
    # difference of indices from which we can extract segments
    output["allowed"] = tf.cast(
        tf.floor(context_dur / (duration * step_percent)), tf.int32
    )
    # difference of indices from which we cannot extract segments
    not_allowed = tf.cast(tf.round(far_from_dur / (duration * step_percent)), tf.int32)
    if not_allowed < 1:
        not_allowed = tf.convert_to_tensor([1], dtype=tf.int32)
    output["not_allowed"] = not_allowed
    return output


@gin.configurable  # type: ignore
def views_fn(
    data: Dict[str, tf.Tensor], n_views: int, n_segments_per_track: int
) -> Dict[str, tf.Tensor]:
    # utils functions
    def py_get_mask_fn(
        segments: tf.Tensor,
        indices: tf.Tensor,
        allowed: tf.Tensor,
        not_allowed: tf.Tensor,
        n_views: int,
    ) -> Dict:
        segments = segments.numpy()
        allowed = allowed.numpy()
        not_allowed = not_allowed.numpy()
        indices = indices.numpy()
        data = []
        for i in range(min(n_segments_per_track, len(indices))):
            v = indices[i]
            # segments allow to use
            m1 = (indices <= v - allowed).astype(np.int32)
            m2 = (indices >= v + allowed).astype(np.int32)
            # segments not allow to use
            m3 = (indices > v - not_allowed).astype(np.int32)
            m4 = (indices < v + not_allowed).astype(np.int32)
            m = np.nonzero((m1 + m2) * (m3 + m4 - 1))[0]
            # -1 because one anchor is already a view
            if len(m) >= n_views - 1:
                s = m[np.random.permutation(len(m))][: n_views - 1]
                data.append(
                        tf.concat([segments[i:i+1, :, :], segments[s, :, :]], axis=0)
                )
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        # return data 
        return tf.squeeze(tf.transpose(data, [0, 2, 1, 3]))

    # find the right pair within the allow context
    data = tf.py_function(
        py_get_mask_fn,
        [
            data["segments"],
            data["indices"],
            data["allowed"],
            data["not_allowed"],
            n_views,
        ],
        (tf.float32),
        # (tf.float32, tf.int32),
    )
    output = {"segments": data}
    output["key_signature"] = tf.repeat("-1", repeats=tf.shape(output["segments"])[0])
    output["keymode"] = tf.repeat("-1", repeats=tf.shape(output["segments"])[0])
    return output


@gin.configurable  # type: ignore
def dataloader_tf_2segments(
    path: str,
    split_set: str,
    sr: int,
    duration: float,
    test_percent: float,
    batch_size: int,
    buffer_size: int,
    step_percent: float,
    context_dur: float=5,
    far_from_dur: float=90,
    n_segments_per_track: int = 10,
    n_views: int = 2,
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
        # .shuffle(499, reshuffle_each_iteration=True)
        .map(whole_song_path_from_md5)
        .filter(lambda x: x["song_path"] != "")
        .map(
            lambda data: load_audio(data, sr),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .filter(
            lambda data: data["duration"] > context_dur * 10
            and tf.reduce_max(tf.abs(data["audio"])) != 0
        )
        .map(
            lambda data: preproces_for_views(
                data, duration, context_dur, far_from_dur, sr, step_percent
            )
        )
        .filter(
            lambda data: data["segments"].shape[0] != 0 
        )
        .map(
            lambda data: views_fn(data, n_views, n_segments_per_track),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .unbatch()
        .shuffle(buffer_size, reshuffle_each_iteration=True)
    )
    print("Number of views: {}".format(n_views))
    ds = (
        ds.repeat()
        .batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        # .prefetch(tf.data.AUTOTUNE)
    )
    return ds

@gin.configurable
class Dataloader(torch.utils.data.IterableDataset):  # type: ignore
    def __init__(
        self,
        split_set: str,
        path: str,
        device: str,
        test_percent: int,
        buffer_size: int,
        batch_size: int,
        sr: int,
        duration: int,
        step_percent: float,
        views: str,
    ):
        super(Dataloader, self).__init__()
        assert views == "segments" or "sources"
        self.split_set = split_set
        self.device = torch.device(device)
        self.test_percent = test_percent
        self.batch_size = batch_size
        self.device = device
        self.duration = duration
        self.sr = sr
        self.step_percent = step_percent
        self.buffer_size = buffer_size
        self.tf_dataloader = dataloader_tf_2segments(
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
        for batch in self.tf_dataloader.as_numpy_iterator():
            batch = torch.from_numpy(np.copy(batch))
            yield batch.to(self.device, non_blocking=True)
