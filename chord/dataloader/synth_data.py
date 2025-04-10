## Synthesized data
# Use sinusoid waveforms to generate chors

import random
import numpy as np
import torch.utils.data as Data

def generate_sinusoid_chord(pitches=[69], duration=1, Fs=4000, amplitude_max=0.5):
    """Generate synthetic sound of chord using sinusoids

    Notebook: C5/C5S1_Chords.ipynb

    Args:
        pitches (list): List of pitches (MIDI note numbers) (Default value = [69])
        duration (float): Duration (seconds) (Default value = 1)
        Fs (scalar): Sampling rate (Default value = 4000)
        amplitude_max (float): Amplitude (Default value = 0.5)

    Returns:
        x (np.ndarray): Synthesized signal
    """
    N = int(duration * Fs)
    t = np.arange(0, N) / Fs
    x = np.zeros(N)
    for p in pitches:
        omega = 2 ** ((p - 69) / 12) * 440
        x = x + np.sin(2 * np.pi * omega * t)
    x = amplitude_max * x / np.max(x)
    return x

def infinite_chord_generator(duration, sr, label_fps, worker_id=0, num_workers=1):
    base_pitch = 60  # C4
    len_label = int(duration * label_fps)

    seed = 42 + worker_id
    random.seed(seed)
    while True:
        # Replace this with your real synthesis logic
        shift = random.randint(0, 11)  # 0–23 major/minor chords
        quality = random.randint(0, 1)  # 0 for major, 1 for minor

        root = base_pitch + shift
        third = root + (4 if quality == 0 else 3)  # Major or minor third
        fifth = root + 7  # Perfect fifth

        # random octave_shift
        root = root + 12 * random.randint(-1, 1)
        third = third + 12 * random.randint(-1, 1)
        fifth = fifth + 12 * random.randint(-1, 1)

        waveform = generate_sinusoid_chord([root, third, fifth], duration=duration, Fs=sr)
        yield dict(x=waveform[np.newaxis], y_root=shift * np.ones(len_label, dtype=np.int64), y_quality=quality* np.ones(len_label, dtype=np.int64))

class ChordSynthDataset(Data.IterableDataset):
    def __init__(self, duration=4, sr=22050, label_fps=10):
        super(ChordSynthDataset).__init__()

        self.duration = duration
        self.sr = sr
        self.label_fps = label_fps
        self.generater_fn = infinite_chord_generator


    def __iter__(self):
        worker_info = Data.get_worker_info()
        if worker_info is None:
            # Single-process
            return self.generater_fn(self.duration, self.sr, self.label_fps)
        else:
            # Multi-process
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            return self.generater_fn(self.duration, self.sr, self.label_fps, worker_id=worker_id, num_workers=num_workers)
        return 
    
def get_synth_dataloader(
    label_fps: int = 10,
    sr: int = 22050,
    seg_length: float = 10.0,

    batch_size: int = 1,
    num_workers: int = 0,
):
    dataset = ChordSynthDataset(duration=seg_length, sr=sr, label_fps=label_fps)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader
