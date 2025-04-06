## Supervised data
# Schubert Winterreise (SWD), classical
# Beethoven String Quartet (BSQD), classical -> Tuning freq?
# Beatles, pop
# RWCPop, pop

import os
import numpy as np
import pandas as pd
import torch.utils.data as Data
import torchaudio
import mir_eval

read_label_csv_kwargs = {
    "swd": {"sep": ";", "header": 0},
    "bsqd": {"sep": ";", "header": None, "index_col": False, "names": ['start', 'end', 'shorthand', 'extended', 'majmin_inv', 'majmin']},
    "beatles": {"sep": " ", "header": None, "names": ['start', 'end', 'shorthand']},
}

class ChordSingleTrackDataset(Data.Dataset):
    def __init__(
        self,
        audio_path: str,
        label_path: str,
        dataset_name: str,
        label_fps: int = 10,

        sr: int = 22050,
        seg_length: float = 10.0,
        seg_shift_length: float = 10.0,
    ) -> None:
        """
        Dataset for a single track, concatenated later into a single dataset

        Parameters
        ----------
        audio_path : str
            Path to the input audio file
        label_path : str
            Path to the label file
        dataset_name : str
            Name of the dataset

        seg_length : float, optional
            Length of the segment in seconds, by default 10, <0 means load the whole audio
        seg_shift_length : float, optional
            Shift length of the segment in seconds, by default 10
        """
        super().__init__()

        self.audio_path = audio_path
        self.label_path = label_path
        self.label_fps = label_fps
        self.sr = sr

        self.seg_length = seg_length
        self.seg_shift_length = seg_shift_length

        self.seg_frames = int(seg_length * sr) if seg_length > 0.0 else -1
        self.seg_shift_frames = int(seg_shift_length * seg_shift_length)

        # Load the label file
        self.label_df: pd.DataFrame = pd.read_csv(label_path, **read_label_csv_kwargs[dataset_name])
        self.label_df = self.label_df.astype({"start":"float32", "end":"float32"})

        self.audio, _ = torchaudio.load(audio_path, normalize=True)
        self.audio = self.audio.mean(dim=0, keepdim=True)  # Convert to mono

        if seg_length > 0.0:
            # Compute how many segments are in the audio file
            self.length = (self.audio.shape[-1] - self.seg_frames) // self.seg_shift_frames
        else:
            self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.seg_length > 0:
            start_frame = index * self.seg_shift_frames
            x = self.audio[..., start_frame:start_frame+self.seg_frames]
            time_frame = np.arange(start_frame / self.sr, (start_frame + self.seg_frames) / self.sr, 1 / self.label_fps)
        else:
            x = self.audio
            time_frame = np.arange(0, len(x) / self.sr, 1 / self.label_fps)

        root_labels = -1 * np.ones(len(time_frame), dtype=np.int64)
        quality_labels = -1 * np.ones(len(time_frame), dtype=np.int64)
        bass_labels = -1 * np.ones(len(time_frame), dtype=np.int64)
        for _, row in self.label_df.iterrows():
            if "majmin" in row:
                # For SWD and BSQD, majmin is already in the format we need
                root, quality, bass = self.encode_chord(row["majmin"])
            elif "shorthand" in row:
                # For Beatles and RWCPop, we need to convert the shorthand to majmin
                root, quality, bass = self.encode_chord(row["shorthand"])

            root_labels[(time_frame - 0.5 / self.label_fps > row["start"]) & (time_frame + 0.5 / self.label_fps < row["end"])] = root
            quality_labels[(time_frame - 0.5 / self.label_fps > row["start"]) & (time_frame + 0.5 / self.label_fps < row["end"])] = quality
            bass_labels[(time_frame - 0.5 / self.label_fps > row["start"]) & (time_frame + 0.5 / self.label_fps < row["end"])] = bass
        return dict(x=x, y_root=root_labels, y_quality=quality_labels, y_bass=bass_labels)

    def encode_chord(self, chord: str) -> int:
        """
        Encode a chord to an index based on mir_eval.chord.encode
        Here the chord str is supposed to already be the "majmin" format
        
        Parameters
        ---------
        chord : str
            The chord to encode.

        Returns
        ---------
        int
            The index for root note of the chord, 0 for C, 1 for C#, ..., 11 for B
        int
            The index for quality of the chord, 0 for major, 1 for minor
        int
            The index for bass note (relateive to root) of the chord
        """     
        root, bitmap, bass = mir_eval.chord.encode(chord)
        # for majmin chord, bitmap is sufficient to determine the chord
        if bitmap[4] == 1: # maj
            quality = 0
        elif bitmap[3] == 1: # min
            quality = 1
        else:
            root, quality, bass = -1, -1, -1
        return root, quality, bass


def get_dataloader(
    audio_folder: str,
    label_folder: str,
    dataset_name: str,

    label_fps: int = 10,
    sr: int = 22050,
    seg_length: float = 10.0,
    seg_shift_length: float = 10.0,

    batch_size: int = 1,
    num_workers: int = 0,
):
    # Get the list of audio files for each dataset
    if dataset_name == "swd" or dataset_name == "bsqd":
        filename_list = [filename.replace(".wav", "") for filename in os.listdir(audio_folder) if filename.endswith(".wav")]
        label_suffix = ".csv"
    elif dataset_name == "beatles":
        filename_list = []
        for album in os.listdir(audio_folder):
            album_dir = os.path.join(audio_folder, album)
            if os.path.isdir(album_dir):
                for filename in os.listdir(album_dir):
                    if filename.endswith(".wav"):
                        filename_list.append(os.path.join(album, filename.replace(".wav", "")))
        label_suffix = ".lab"
    elif dataset_name == "rwcpop":
        raise NotImplementedError("RWCPop dataset is not implemented yet.")
        filename_list = [filename.split(".")[0] for filename in os.listdir(audio_folder) if filename.endswith(".wav")]
        label_suffix = ".lab"

    dataset_list = []
    for filename in filename_list:
        audio_path = os.path.join(audio_folder, filename + ".wav")
        label_path = os.path.join(label_folder, filename + label_suffix)
        if not os.path.exists(label_path):
            print(f"Label file {label_path} does not exist. Skipping.")
            continue
        dataset = ChordSingleTrackDataset(
            audio_path=audio_path,
            label_path=label_path,
            dataset_name=dataset_name,
            label_fps=label_fps,
            sr=sr,
            seg_length=seg_length,
            seg_shift_length=seg_shift_length,
        )
        dataset_list.append(dataset)
    dataset = Data.ConcatDataset(dataset_list)
    return Data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)