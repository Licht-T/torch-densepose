# torch-densepose
[DensePose](https://arxiv.org/abs/1802.00434) w/ [Panoptic-FPN](https://arxiv.org/abs/1901.02446) implementation by PyTorch and Torchvision.

## Install
```bash
python setup.py install
```

## Example
```python
import torch
import PIL.Image
import PIL.ImageDraw
import numpy as np
from densepose.model import DensePose


model = DensePose()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

img = PIL.Image.open('./data/chi.jpg')
img_array = np.array(img, dtype=np.float32).transpose((2, 0, 1))
img_tensor = torch.from_numpy(img_array).unsqueeze(0)

results = model(img_tensor.to(device))

boxes = results[0]['boxes'].to('cpu')
scores = results[0]['scores'].to('cpu')
coarse_segs = results[0]['coarse_segs'].to('cpu')
fine_segs = results[0]['fine_segs'].to('cpu')

draw = PIL.ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

img.save('./data/out_box.jpg')

seg_img_array = np.zeros(img_array.shape[1:], dtype=np.uint8)
for coarse_seg, fine_seg in zip(coarse_segs, fine_segs):
    coarse_seg = coarse_seg.numpy().astype(np.uint8)
    fine_seg = fine_seg.numpy().astype(np.uint8)
    seg = 10 * fine_seg * coarse_seg

    cond = seg_img_array == 0
    seg_img_array[cond] = seg_img_array[cond] + seg[cond]

seg_img = PIL.Image.fromarray(seg_img_array)
seg_img.save('./data/out_seg.jpg')
```
![output_box](https://raw.githubusercontent.com/Licht-T/torch-densepose/master/data/out_box.jpg)
![output_seg](https://raw.githubusercontent.com/Licht-T/torch-densepose/master/data/out_seg.jpg)


## TODO
* [x] Person detection
* [x] DensePose human body parts segmentation
* [ ] DensePose human body UV map estimation
* [ ] Loss calculation
