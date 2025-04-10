from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import gin
import torch.nn as nn
from einops import rearrange, repeat



def update_static_parameter(module: Any, key: str, value: torch.Tensor) -> Any:
    state_dict = module.state_dict()
    state_dict[key] = value
    module.load_state_dict(state_dict)
    return module

log_clap: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.clamp(
    torch.log(x), min=-100
)

class Z_transformation(torch.nn.Module):
    def __init__(self,
                 circle_type: int,
                 device: str) -> None:
        super().__init__()
        self.omega = circle_type/12
        self.alpha = torch.exp(1j * 2 * torch.pi * self.omega * torch.arange(12))
        self.device = device

    def forward(self,
                y: torch.Tensor
                ) -> torch.Tensor:
        """
        Complex Z tranformation for loss calculation, project the 12 probability bins to a single point on a disk of r=1. 
        """
        z = torch.matmul(torch.complex(y, 0*y), self.alpha.to(y.device))
        return z


@gin.configurable
class CrossPowerSpectralDensityLoss(nn.Module):
    """
    Differentialble distance on the circle of fifths.
    """
    def __init__(self, 
                 circle_type: int,
                 device: str,
                 # weights: List,
             ) -> None:
        super(CrossPowerSpectralDensityLoss, self).__init__()
        self.z_transformation = Z_transformation(circle_type, device)
        # self.weights = weights


    def forward(
        self, 
        y: torch.Tensor
    ) -> Dict[str, Union[int, float, Dict]]:
        y, difference  = y
        batch_size = int(y.shape[0]/2)

        # calculate m, value for mode, vertical summation
        # channel_1, channel_2 = torch.split(y, 12, dim=1) # [n_views*batch+equivariant, 12]
        # m1 = torch.sum(channel_1, dim=1) # sum of mode per data point in the batch

        # for 3 views
        # m1_source1 = m1[:batch_size]
        # m1_source2 = m1[batch_size:2*batch_size]
        # m1_equivariant = m1[2*batch_size:]

        # horizontal summation
        # y = torch.add(channel_1, channel_2)


        # distribution loss: balance distribution of major and minor modes predictions
        # sum_1 = torch.sum(channel_1[:2*batch_size])
        # sum_2 = torch.sum(channel_2[:2*batch_size])
        # loss_distribution = (sum_1/(2*batch_size) - 0.5)*(0.5 - sum_2/(2*batch_size)) # maximum is 0.25

        # label the data by the cqt bin energy comparaison
        # with torch.no_grad():
        #     key_signature = ((y[:batch_size] + y[batch_size:2*batch_size])/2).argmax(axis=1)
        #     root_bin = (key_signature + 3)%12
        #     majorminor = torch.gather(mean_hcqt, 1, key_signature.unsqueeze(axis=1)) > torch.gather(mean_hcqt, 1, root_bin.unsqueeze(axis=1))
        #     feat_eng_mode = majorminor.int().squeeze()
            

        # loss calculation
        # loss_pos = (1 - z1 * z2.conj()).abs().pow(2).mean()
        # z transformation, for 2 views
        t = y.shape[-1]
        y = rearrange(y, "b f t -> (b t) f") # (256*7, 12)
        difference = repeat(difference, 'b -> b t', t=t)
        difference = rearrange(difference, "b t -> (b t)") # (256*7, 12)
        z = self.z_transformation(y)
        z1 = z[:batch_size*t, ...]
        z2 = z[batch_size*t:, ...]
        loss_equivariant = (torch.exp(1j * 2 * torch.pi * self.z_transformation.omega * difference.to(self.z_transformation.device)) - z1 * z2.conj()).abs().pow(2).mean()

        # loss_mode = (
        #         (-m1_source1*log_clap(m1_equivariant)-(1-m1_source1)*log_clap(1-m1_equivariant)).mean() + 
        #         0.5*(-m1_source2*log_clap(m1_equivariant)-(1-m1_source2)*log_clap(1-m1_equivariant)).mean() + 
        #         0.5*(-m1_equivariant*log_clap(m1_source2)-(1-m1_equivariant)*log_clap(1-m1_source2)).mean() + 
        #         (-m1_source1*log_clap(m1_source2)-(1-m1_source1)*log_clap(1-m1_source2)).mean()
        #         )

        # loss_mode = (
        #         (-feat_eng_mode*log_clap(m1_equivariant)-(1-feat_eng_mode)*log_clap(1-m1_equivariant)).mean() + 
        #         (-feat_eng_mode*log_clap(m1_source2)-(1-feat_eng_mode)*log_clap(1-m1_source2)).mean() + 
        #         (-feat_eng_mode*log_clap(m1_source1)-(1-feat_eng_mode)*log_clap(1-m1_source1)).mean()
        #         )

        # with torch.no_grad():
        #     key_signature = ((y[:batch_size] + y[batch_size:2*batch_size])/2).argmax(axis=1)
        #     root_bin_1 = (key_signature + 3)%12
        #     root_bin_2 = (key_signature - 3)%12
        #     majorminor = (torch.gather(mean_hcqt, 1, key_signature.unsqueeze(axis=1)) > torch.gather(mean_hcqt, 1, root_bin_2.unsqueeze(axis=1))) * ((torch.gather(mean_hcqt, 1, key_signature.unsqueeze(axis=1)) > torch.gather(mean_hcqt, 1, root_bin_1.unsqueeze(axis=1))))
        #     feat_eng_mode = majorminor.int().squeeze()
        #     
        #
        # # loss calculation
        # loss_pos = (1 - z1 * z2.conj()).abs().pow(2).mean()
        # loss_equivariant_1 = (torch.exp(1j * 2 * torch.pi * self.z_transformation.omega * difference.to(self.z_transformation.device)) - z1 * z3.conj()).abs().pow(2).mean()
        # loss_equivariant_2 = (torch.exp(1j * 2 * torch.pi * self.z_transformation.omega * difference.to(self.z_transformation.device)) - z2 * z3.conj()).abs().pow(2).mean()
        # loss_key = loss_pos + loss_equivariant_1 + loss_equivariant_2
        #
        # # loss_mode = (
        # #         (-m1_source1*log_clap(m1_equivariant)-(1-m1_source1)*log_clap(1-m1_equivariant)).mean() + 
        # #         0.5*(-m1_source2*log_clap(m1_equivariant)-(1-m1_source2)*log_clap(1-m1_equivariant)).mean() + 
        # #         0.5*(-m1_equivariant*log_clap(m1_source2)-(1-m1_equivariant)*log_clap(1-m1_source2)).mean() + 
        # #         (-m1_source1*log_clap(m1_source2)-(1-m1_source1)*log_clap(1-m1_source2)).mean()
        # #         )
        #
        # loss_mode = (
        #         (feat_eng_mode*(-feat_eng_mode*log_clap(m1_equivariant)-(1-feat_eng_mode)*log_clap(1-m1_equivariant))).mean() + 
        #         (feat_eng_mode*(-feat_eng_mode*log_clap(m1_source2)-(1-feat_eng_mode)*log_clap(1-m1_source2))).mean() + 
        #         (feat_eng_mode*(-feat_eng_mode*log_clap(m1_source1)-(1-feat_eng_mode)*log_clap(1-m1_source1))).mean()
        #         )
        # 
        # w_key, w_mode, w_distribution = self.weights

        # loss = w_key*loss_key + w_mode*1.5*loss_mode  + w_distribution*20*loss_distribution

        return {"loss": loss_equivariant, 
                "loss_to_print": {
                    # "loss_pos": loss_pos, 
                    "loss_equi": loss_equivariant, 
                    # "loss_mode": loss_mode, 
                    # "loss_distribution": loss_distribution, 
                    "loss_total":loss_equivariant
                    }
                }
