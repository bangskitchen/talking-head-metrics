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

---
When install cpbd lib, you have to fix %home%/anaconda3/envs/%env_name%/lib/python3.8/site-packages/cpbd/compute.py
```
ImportError: cannot import name 'imread' from 'scipy.ndimage' (%home%/anaconda3/envs/%env_name%/lib/python3.8/site-packages/scipy/ndimage/__init__.py)
14 from scipy.ndimage import imread
-> from skimage.io import imread
```
