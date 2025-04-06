import pytest
import torch

from chord.dataloader.supervised_data import ChordSingleTrackDataset, get_dataloader

def test_single_track_swd():
    dataset = ChordSingleTrackDataset(
        audio_path="../datasets/Schubert_Winterreise_Dataset_v2-1/01_RawData/audio_wav/Schubert_D911-01_AL98.wav",
        label_path="../datasets/Schubert_Winterreise_Dataset_v2-1/02_Annotations/ann_audio_chord/Schubert_D911-01_AL98.csv",
        dataset_name="swd",
        label_fps=10,
        sr=22050,
        seg_length=10.0,
        seg_shift_length=10.0,
    )
    assert len(dataset) > 0

    data = dataset[0]
    assert data["x"].shape == (1, 220500)
    assert data["y_root"].shape == (100, )
    assert data["y_quality"].shape == (100, )
    assert data["y_bass"].shape == (100, )

def test_dataloader_swd():
    dataloder = get_dataloader(
        audio_folder="../datasets/Schubert_Winterreise_Dataset_v2-1/01_RawData/audio_wav/",
        label_folder="../datasets/Schubert_Winterreise_Dataset_v2-1/02_Annotations/ann_audio_chord/",
        dataset_name="swd",
        batch_size=4,
        num_workers=0,
        label_fps=10,
        sr=22050,
        seg_length=10.0,
        seg_shift_length=10.0,
    )
    assert len(dataloder) > 0
    for data in dataloder:
        assert data["x"].shape == (4, 1, 220500)
        assert data["y_root"].shape == (4, 100)
        assert data["y_quality"].shape == (4, 100)
        assert data["y_bass"].shape == (4, 100)
        break

def test_dataloader_bsqd():
    dataloder = get_dataloader(
        audio_folder="../datasets/Beethoven_String_Quartet_Dataset_v0/audio_wav_22050_mono/",
        label_folder="../datasets/Beethoven_String_Quartet_Dataset_v0/ann_audio_chord/",
        dataset_name="bsqd",
        batch_size=4,
        num_workers=0,
        label_fps=10,
        sr=22050,
        seg_length=10.0,
        seg_shift_length=10.0,
    )
    assert len(dataloder) > 0
    for data in dataloder:
        assert data["x"].shape == (4, 1, 220500)
        assert data["y_root"].shape == (4, 100)
        assert data["y_quality"].shape == (4, 100)
        assert data["y_bass"].shape == (4, 100)
        break

def test_dataloader_beatles():
    dataloder = get_dataloader(
        audio_folder="../datasets/ChordRec_Pop/Isophonics/database_orig/wav/The Beatles",
        label_folder="../datasets/ChordRec_Pop/Isophonics/database_orig/chordlab/The Beatles",
        dataset_name="beatles",
        batch_size=4,
        num_workers=0,
        label_fps=10,
        sr=22050,
        seg_length=10.0,
        seg_shift_length=10.0,
    )
    assert len(dataloder) > 0
    for data in dataloder:
        assert data["x"].shape == (4, 1, 220500)
        assert data["y_root"].shape == (4, 100)
        assert data["y_quality"].shape == (4, 100)
        assert data["y_bass"].shape == (4, 100)
        break