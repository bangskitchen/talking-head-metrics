import os, sys
import argparse

from PIL import Image
import numpy as np
import cv2

# PSNR, SSIM, CPBD
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cpbd

# LPIPS, FID
import torch
import lpips
import pytorch_fid_wrapper as pyfid

# SyncNet
import time
from scipy import signal
from scipy.io import wavfile
import python_speech_features
from SyncNetModel import *

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

def cpbd_score(images: np.ndarray):
    PIL_gray_images = [Image.fromarray(cv2.merge(list(cv2.split(image)[::-1]))).convert("L") for image in images]
    score = [cpbd.compute(np.array(image)) for image in PIL_gray_images]
    return sum(score) / len(score)

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []

    for i in range(0, len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]))

    return dists

def syncnet_score(video_path, audio_path, fps=25):
    # parameters
    batch_size = 20
    vshift = 15

    # initialize syncnet
    __S__ = S(num_layers_in_fc_layers=1024).cuda()
    loaded_state = torch.load("data/syncnet_v2.model", map_location=lambda storage, loc: storage)
    self_state = __S__.state_dict()
    for name, param in loaded_state.items():
        self_state[name].copy_(param)
    __S__.eval()

    # load video
    frames, _ = get_frames(video_path)
    im = np.expand_dims(frames, axis=0)
    im = np.transpose(im, (0, 4, 1, 2, 3))
    imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

    # load audio
    sample_rate, audio = wavfile.read(audio_path)
    if (float(len(audio)) / sample_rate) != (float(len(frames)) / fps) :
        print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio)) / sample_rate, float(len(frames)) / fps))
        min_length = min(len(frames) * (sample_rate // fps), len(audio))
        if (min_length == len(audio)):
            min_length = (len(audio) // 640) * 640
        audio = audio[:min_length]

    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc])
    cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
    cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

    # calculate
    lastframe = min_length - 5
    im_feat = []
    cc_feat = []

    tS = time.time()
    for i in range(0, lastframe, batch_size):
        
        im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
        im_in = torch.cat(im_batch,0)
        im_out  = __S__.forward_lip(im_in.cuda())
        im_feat.append(im_out.data.cpu())

        cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in range(i, min(lastframe, i + batch_size))]
        cc_in = torch.cat(cc_batch,0)
        cc_out  = __S__.forward_aud(cc_in.cuda())
        cc_feat.append(cc_out.data.cpu())

    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)
        
    print('Compute time %.3f sec.' % (time.time() - tS))

    dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)

    minval, minidx = torch.min(mdist, 0)

    offset = vshift - minidx
    conf   = torch.median(mdist) - minval

    fdist   = np.stack([dist[minidx].numpy() for dist in dists])
    fconf   = torch.median(mdist).numpy() - fdist
    fconfm  = signal.medfilt(fconf, kernel_size=9)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print('Framewise conf: ')
    print(fconfm)
    print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

    dists_npy = np.array([ dist.numpy() for dist in dists ])
    return offset.numpy(), conf.numpy(), dists_npy

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