# talking-head-metrics
1. PNSR, SSIM, LPIPS, CPBD, FID
2. [SyncNet](https://github.com/joonson/syncnet_python)
3. F-LMD, M-LMD

## Requirements
- python>=3.8
- numpy>=1.18.1
- opencv-contrib-python
- torch>=1.4.0
- torchvision>=0.5.0
- python_speech_features
- scipy>=1.2.1
- scenedetect==0.5.1
- scikit-image==0.21.0
- lpips==0.1.4
- cpbd==1.0.7
- pytorch_fid_wrapper==0.0.4
- ffmpeg

when install cpbd lib, you have to fix compute.py
---
from scipy import ndimage -> from skimage.io import imread

