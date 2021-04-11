from typing import Dict, List, Tuple, Optional

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign


class DensePoseHead(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int,
            num_convs: int = 8, kernel_size: Tuple[int] = (3, 3)
    ):
        super(DensePoseHead, self).__init__()

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        ops = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding
            )
            ops.append(conv)
            ops.append(nn.ReLU())

        self.convs = nn.Sequential(*ops)

    def forward(self, x: Tensor) -> Tensor:
        return self.convs(x)


class DensePosePredictor(nn.Module):
    def __init__(self, in_channels: int, num_segmentations: int, num_patches: int, kernel_size: Tuple[int, int],
                 scale_factor: int):
        super().__init__()

        self.scale_factor = scale_factor

        padding = (kernel_size[0] // 2 - 1, kernel_size[1] // 2 - 1)
        self.coarse_seg_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_segmentations,
            kernel_size=kernel_size,
            stride=(2, 2),
            padding=padding
        )
        self.fine_seg_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_patches + 1,
            kernel_size=kernel_size,
            stride=(2, 2),
            padding=padding
        )
        self.u_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_patches + 1,
            kernel_size=kernel_size,
            stride=(2, 2),
            padding=padding
        )
        self.v_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_patches + 1,
            kernel_size=kernel_size,
            stride=(2, 2),
            padding=padding
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple, Tuple, Tuple]:
        coarse_seg = F.interpolate(
            self.coarse_seg_conv(x),
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )
        fine_seg = F.interpolate(
            self.fine_seg_conv(x),
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )
        u = F.interpolate(
            self.u_conv(x),
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )
        v = F.interpolate(
            self.v_conv(x),
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False
        )

        return coarse_seg, fine_seg, u, v


class DensePoseRoIHeads(RoIHeads):
    def __init__(self,
                 box_roi_pool: MultiScaleRoIAlign,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool: MultiScaleRoIAlign = None,
                 mask_head=None,
                 mask_predictor=None,
                 # Keypoint
                 keypoint_roi_pool: MultiScaleRoIAlign = None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 # DensePose
                 densepose_roi_pool: MultiScaleRoIAlign = None,
                 densepose_head: DensePoseHead = None,
                 densepose_predictor: DensePosePredictor = None,
                 ):
        super(DensePoseRoIHeads, self).__init__(
            box_roi_pool, box_head, box_predictor,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
            positive_fraction, bbox_reg_weights, score_thresh,
            nms_thresh, detections_per_img, mask_roi_pool,
            mask_head, mask_predictor, keypoint_roi_pool,
            keypoint_head, keypoint_predictor,
        )

        self.densepose_roi_pool = densepose_roi_pool
        self.densepose_head = densepose_head
        self.densepose_predictor = densepose_predictor

    def forward(self,
                features: Dict[str, Tensor],
                proposals: List[Tensor],
                image_shapes: List[Tuple[int, int]],
                targets: Optional[List[Dict[str, Tensor]]] = None
                ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        if targets is not None:
            for t in targets:
                # TODO: Assert DensePose training data are correct.
                pass

        results, losses = super(DensePoseRoIHeads, self).forward(features, proposals, image_shapes, targets)

        densepose_features = self.densepose_roi_pool(
            features,
            [result['boxes'] for result in results],
            image_shapes
        )
        densepose_features = self.densepose_head(densepose_features)
        coarse_segs_batch, fine_segs_batch, us_batch, vs_batch = self.densepose_predictor(densepose_features)

        # TODO: Loss calculation

        box_per_image = [result['boxes'].shape[0] for result in results]
        coarse_segs_batch = coarse_segs_batch.split(box_per_image, dim=0)
        fine_segs_batch = fine_segs_batch.split(box_per_image, dim=0)
        us_batch = us_batch.split(box_per_image, dim=0)
        vs_batch = vs_batch.split(box_per_image, dim=0)

        for result, coarse_segs, fine_segs, us, vs in zip(
                results, coarse_segs_batch, fine_segs_batch, us_batch, vs_batch
        ):
            result['coarse_segs'] = coarse_segs
            result['fine_segs'] = fine_segs
            result['us'] = us
            result['vs'] = vs

        return results, losses
