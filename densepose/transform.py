from typing import List, Optional, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import expand_masks, expand_boxes
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def paste_mask_in_image(mask: Tensor, box: Tensor, im_h: int, im_w: int) -> Tensor:
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, -1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)[0].argmax(dim=0)

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
    ]
    return im_mask


def paste_masks_in_image(masks, boxes, img_shape, padding=1):
    # type: (Tensor, Tensor, Tuple[int, int], int) -> Tensor
    masks, scale = expand_masks(masks, padding=padding)
    boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
    im_h, im_w = img_shape

    res = [
        paste_mask_in_image(m, b, im_h, im_w)
        for m, b in zip(masks, boxes)
    ]
    if len(res) > 0:
        ret = torch.stack(res, dim=0)
    else:
        ret = masks.new_empty((0, im_h, im_w))
    return ret


class DensePoseRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(DensePoseRCNNTransform, self).__init__(min_size, max_size, image_mean, image_std)

    def forward(self,
                images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None
                ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        # TODO: DensePose target data transformation
        return super(DensePoseRCNNTransform, self).forward(images, targets)

    def postprocess(self,
                    result: List[Dict[str, Tensor]],
                    image_shapes: List[Tuple[int, int]],
                    original_image_sizes: List[Tuple[int, int]]
                    ) -> List[Dict[str, Tensor]]:
        result = super(DensePoseRCNNTransform, self).postprocess(result, image_shapes, original_image_sizes)

        if self.training:
            return result

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            if "coarse_segs" in pred:
                coarse_segs = pred["coarse_segs"]
                coarse_segs = paste_masks_in_image(coarse_segs, boxes, o_im_s)
                result[i]["coarse_segs"] = coarse_segs
            if "fine_segs" in pred:
                fine_segs = pred["fine_segs"]
                fine_segs = paste_masks_in_image(fine_segs, boxes, o_im_s)
                result[i]["fine_segs"] = fine_segs
            if "u" in pred:
                # TODO: Implement
                pass
            if "v" in pred:
                # TODO: Implement
                pass

        return result
