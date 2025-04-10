from typing import Any

import torch

from chord.model.hcqt import HarmonicVQT
from chord.stone import Stone

def test_stone():
    """
    Test the Stone model
    """
    duration = 4
    sr = 16000
    hop_length = 160
    bs = 8

    downsample = 16
    output_frames = duration * sr // hop_length // downsample  # = 25

    # Create a random input tensor
    train_batch = torch.randn(1, bs, sr * duration, 1) # 1 b t c for SSL

    # Initialize the Stone model
    stone = Stone(HarmonicVQT(hop_length=hop_length), device="cpu")

    # Forward pass
    y, difference = stone(train_batch)
    assert y.shape == (bs * 2, 12, output_frames)
    assert difference.shape == (bs, )
