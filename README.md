# talking-head-metrics
1. PNSR, SSIM, LPIPS, CPBD, FID
2. [SyncNet](https://github.com/joonson/syncnet_python)
3. F-LMD, M-LMD

## Requirements
- python >= 3.8
- numpy==1.24.4
- opencv-python
- torch
- torchvision
- scikit-image==0.21.0
- lpips==0.1.4
- cpbd==1.0.7
- pytorch_fid_wrapper==0.0.4
- ffmpeg

when install cpbd lib, you have to fix compute.py
---
from scipy import ndimage -> from skimage.io import imread

