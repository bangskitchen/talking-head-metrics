import os, sys
import argparse

from PIL import Image
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cpbd

import torch
import lpips
import pytorch_fid_wrapper as pyfid

def psnr_score(real_images: np.ndarray, fake_images: np.ndarray):
    score = [peak_signal_noise_ratio(real_images[i], fake_images[i], data_range=255) for i in range(len(real_images))]
    return sum(score) / len(score)

def ssim_score(real_images: np.ndarray, fake_images: np.ndarray):
    score = [structural_similarity(real_images[i], fake_images[i], data_range=255, multichannel=True, channel_axis=2) for i in range(len(real_images))]
    return sum(score) / len(score)

def lpips_score(real_images: torch.Tensor, fake_images: torch.Tensor, device="cpu"):
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    score = [float(loss_fn_alex(real_images[i].to(device), fake_images[i].to(device)).reshape(-1)[0]) for i in range(len(real_images))]
    return sum(score) / len(score)

def fid_score(real_images: torch.Tensor, fake_images: torch.Tensor, device="cpu"):
    pyfid.set_config(device=device)
    real_m, real_s = pyfid.get_stats(real_images)
    return pyfid.fid(fake_images, real_m=real_m, real_s=real_s)

def cpbd_score(fake_images: np.ndarray):
    PIL_gray_images = [Image.fromarray(cv2.merge(list(cv2.split(image)[::-1]))).convert("L") for image in fake_images]
    score = [cpbd.compute(np.array(image)) for image in PIL_gray_images]
    return sum(score) / len(score)

def syncnet_score():
    ...

def lmd_score():
    ...


def get_frames(video_path):
    frames_numpy, frames_torch = list()

    # read video
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frames_numpy.append(frame)
        R, G, B = cv2.split(frame)
        frame = cv2.merge([B, G, R]).astype(np.float32) / 255.
        frame = torch.from_numpy(frame).permute(2, 0, 1) * 2 - 1
        frames_torch.append(frame)

    return np.array(frames_numpy), torch.stack(frames_torch)

if __name__ == "__main__":
    print("main:", __file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--real_video", type=str, required=True, action="store", help="Path to the real video.mp4")
    parser.add_argument("-f", "--fake_video", type=str, required=True, action="store", help="Path to the fake video.mp4")
    parser.add_argument("-d", "--device", type=str, default="0", action="store", help="Device num to run the metric")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    real_frames_numpy, real_frames_torch = get_frames(args.real_video)
    fake_frames_numpy, fake_frames_torch = get_frames(args.fake_video)

    print("PSNR:", psnr_score(real_frames_numpy, fake_frames_numpy))
    print("SSIM:", ssim_score(real_frames_numpy, fake_frames_numpy))
    print("LPIPS:", lpips_score(real_frames_torch, fake_frames_torch, device=device))
    print("FID:", fid_score(real_frames_torch, fake_frames_torch, device=device))