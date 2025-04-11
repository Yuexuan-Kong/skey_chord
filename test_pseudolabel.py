import torch
import torchaudio

from chord.model.hcqt import HarmonicVQT, CropCQT
from chord.pseudo_label import compute_quality_pseudo_label, compute_root_pseudo_label
from chord.dataloader.supervised_data import ChordSingleTrackDataset, get_dataloader

def test_quality_single_track_swd():
    dataset = ChordSingleTrackDataset(
        audio_path="../datasets/Schubert_Winterreise_Dataset_v2-1/01_RawData/audio_wav/Schubert_D911-01_AL98.wav",
        label_path="../datasets/Schubert_Winterreise_Dataset_v2-1/02_Annotations/ann_audio_chord/Schubert_D911-01_AL98.csv",
        dataset_name="swd",
        label_fps=10,
        sr=22050,
        seg_length=10.0,
        seg_shift_length=10.0,
    )
    
    cqt_fn = HarmonicVQT(sr=16000, hop_length=1600)
    crop_cqt = CropCQT(84)
    resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)

    for i in range(len(dataset)):
        data = dataset[i]

        x = data["x"].unsqueeze(0)
        x = resampler(x)
        y_root = torch.from_numpy(data["y_root"]).unsqueeze(0)

        cqt = cqt_fn(x)[..., :-1]
        cqt = crop_cqt(cqt, torch.zeros(len(x), ))
        
        cqt = cqt.squeeze(dim=1)
        assert cqt.shape[:-1] == (1, 84)

        pseudo_label = compute_quality_pseudo_label(cqt, y_root)
        pseudo_label = pseudo_label.squeeze(dim=0).numpy()
        y_quality = data["y_quality"]
        acc = ((pseudo_label[y_quality != -1] == y_quality[y_quality != -1]).sum() / len(y_quality[y_quality != -1] ))
        print(f"Accuracy: {acc}")
        assert acc > 0.5, "Accuracy should be greater than 0.5"


def test_root_single_track_swd():
    dataset = ChordSingleTrackDataset(
        audio_path="../datasets/Schubert_Winterreise_Dataset_v2-1/01_RawData/audio_wav/Schubert_D911-01_AL98.wav",
        label_path="../datasets/Schubert_Winterreise_Dataset_v2-1/02_Annotations/ann_audio_chord/Schubert_D911-01_AL98.csv",
        dataset_name="swd",
        label_fps=10,
        sr=22050,
        seg_length=10.0,
        seg_shift_length=10.0,
    )
    
    cqt_fn = HarmonicVQT(sr=16000, hop_length=1600)
    crop_cqt = CropCQT(84)
    resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)

    for i in range(len(dataset)):
        data = dataset[i]

        x = data["x"].unsqueeze(0)
        x = resampler(x)

        cqt = cqt_fn(x)[..., :-1]
        cqt = crop_cqt(cqt, torch.zeros(len(x), ))
        
        cqt = cqt.squeeze(dim=1)
        assert cqt.shape[:-1] == (1, 84)

        pseudo_label = compute_root_pseudo_label(cqt)
        pseudo_label = pseudo_label.squeeze(dim=0).numpy()
        y_root = torch.from_numpy(data["y_root"])

        acc = ((pseudo_label[y_root != -1] == y_root[y_root != -1]).sum() / len(y_root[y_root != -1] ))
        print(f"Accuracy: {acc}")
        assert acc > 0.1, "Accuracy should be greater than 0.1"