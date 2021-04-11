from torch import Tensor
import torch.nn as nn

from typing import Tuple, List, Optional, Callable

from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class PanopticExtraFPNBlock(LastLevelMaxPool):
    def __init__(self, featmap_names: List[str], in_channels: int, out_channels: int, conv_dims: int,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(PanopticExtraFPNBlock, self).__init__()

        self.featmap_names = featmap_names

        featmap_ids = [int(i) for i in featmap_names]

        self.predictor = nn.Conv2d(conv_dims, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.blocks = {}

        highest_resolution_featmap_id = featmap_ids[0]

        for featmap_name, featmap_id in zip(featmap_names, featmap_ids):
            ops = []
            num_upsampling = featmap_id - highest_resolution_featmap_id
            num_convs = max(1, num_upsampling)
            require_upsampling = num_upsampling > 0

            for j in range(num_convs):
                conv = nn.Conv2d(
                    in_channels if j == 0 else conv_dims,
                    conv_dims,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=norm_layer is None,
                )
                ops.append(conv)

                if norm_layer is not None:
                    ops.append(norm_layer(conv_dims))

                relu = nn.ReLU()
                ops.append(relu)

                if require_upsampling:
                    upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    ops.append(upsampling)

            self.blocks[featmap_id] = nn.Sequential(*ops)
            self.add_module('block{}'.format(featmap_id), self.blocks[featmap_id])

    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        # TODO: TorchScript support
        results, names = super(PanopticExtraFPNBlock, self).forward(results, x, names)

        keys = [int(k) for k in names if k in self.featmap_names]
        values = [v for k, v in zip(names, results) if k in self.featmap_names]

        out = self.blocks[keys[0]](values[0])

        for k, v in zip(keys[1:], values[1:]):
            out = out + self.blocks[k](v)

        results.append(self.predictor(out))
        names.append('panoptic_feature')

        return results, names
