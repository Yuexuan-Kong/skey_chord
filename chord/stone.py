from typing import Any, List, Tuple, Union, Dict

import gin  # type: ignore
import torch
from torch import Tensor
from einops import rearrange

from chord.model.chromanet import OctavePool, ChromaNetSeq2Seq
from chord.model.hcqt import HarmonicVQT, CropCQT


@gin.configurable
class Stone(torch.nn.Module):
    def __init__(
        self,
        hcqt: HarmonicVQT,
        # out_channels: List[int],
        # kernels: List[int],
        # temperature: float,
        # n_bins: int,
        device: str,
    ) -> None:
        super().__init__()
        self.hcqt = hcqt
        self.device = device
        self.n_harmonics = len(self.hcqt.harmonics)
        # self.n_bins = n_bins
        # self.bins_before_crop = hcqt.n_bins
        # self.out_channels = out_channels
        # self.kernels = kernels
        self.chromanet = ChromaNetSeq2Seq()
        self.octave_pool = OctavePool(12)
        self.keymode_to_num = {
                 'B minor': 0,
                 'C minor': 1,
                 'C# minor': 2,
                 'D minor': 3,
                 'D# minor': 4,
                 'E minor': 5,
                 'F minor': 6,
                 'F# minor': 7,
                 'G minor': 8,
                 'G# minor': 9,
                 'A minor': 10,
                 'A# minor': 11,
                 'D major': 12,
                 'D# major': 13,
                 'E major': 14,
                 'F major': 15,
                 'F# major': 16,
                 'G major': 17,
                 'G# major': 18,
                 'A major': 19,
                 'A# major': 20,
                 'B major': 21,
                 'C major': 22,
                 'C# major': 23
                 }

    def forward(self, 
                x: Tensor,
                ) -> Tuple[Tensor, Tensor]:
        audio = x

        # supervised
        if audio.shape[2] == 1:
            batch = audio.shape[0]
            audio = audio.permute(0, 2, 1)
            
            # change string annotation to int
            keymode = [self.keymode_to_num[i.decode()] for i in x["keymode"][0]]

            # difference of cropping (positive means pitch down)
            difference = torch.randint(1, 12, (len(audio), ))
            crop_transpose = CropCQT(84)
            vocal_hcqt = self.hcqt(audio)
            transpose_hcqt = crop_transpose(vocal_hcqt, difference)
            vocal_hcqt = vocal_hcqt[:, :, :84, :]
            stack_input = torch.cat((vocal_hcqt, transpose_hcqt), dim=0)

            # calculate output of chromanet
            y = self.chromanet(stack_input)
            # ground truth of y from annotations
            y_gt = torch.zeros((batch, 24)).to(self.device)
            for i, idx in enumerate(keymode):
                y_gt[i, idx] = 1
            y = torch.cat((y_gt, y), dim=0)
        
        # self-supervised
        else:
            audio = rearrange(audio, "1 b t c -> b c t")
            batch = audio.shape[0]
            hcqt = self.hcqt(audio)

            # calculate parameters for cropping CQT
            to_transpose = torch.randint(1, 12, (len(audio), ))
            original = torch.randint(1, 13, (len(audio), ))
            transpose = (to_transpose+original) % 12
            difference = transpose - original
            crop_fn = CropCQT(84)

            # crop CQT
            stack_original = crop_fn(hcqt, original)
            
            # mean_hcqt = self.octave_pool(torch.mean(stack_original, dim=3).unsqueeze(axis=3)).squeeze()
            # mean_hcqt = (mean_hcqt[:batch] + mean_hcqt[batch:])/2
            # max_hcqt = torch.argmax((mean_hcqt[:batch] + mean_hcqt[batch:])/2, dim=2).squeeze()

            source_transpose = crop_fn(hcqt, transpose)
            stack_input = torch.cat((stack_original, source_transpose), dim=0) # torch.Size([384, 1, 84, 646])
            stack_input = rearrange(stack_input, "b 1 f t -> b f t")

            y = self.chromanet(stack_input) # (b, 12, t)

        return (y, difference)
