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
from torchvision.transforms import functional as trans_fn
from torchvision.transforms.functional import InterpolationMode
from scipy.io import wavfile
import python_speech_features
from util.SyncNetModel import *

# LMD code by SadTalker
from util.extract_kp_videos_safe import KeypointExtractor


def get_frames(video_path, res=256):
    frames_numpy, frames_torch = list(), list()

    # read video
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if (res != 256):
            PIL_image = convert_numpy_to_PIL([frame])[0]
            img = trans_fn.resize(PIL_image, 224, interpolation=InterpolationMode.LANCZOS)
            img = trans_fn.center_crop(img, 224)
            frame = convert_PIL_to_numpy([img])[0]

        frames_numpy.append(frame)
        R, G, B = cv2.split(frame)
        frame = cv2.merge([B, G, R]).astype(np.float32) / 255.
        frame = torch.from_numpy(frame).permute(2, 0, 1) * 2 - 1
        frames_torch.append(frame)

    return np.array(frames_numpy), torch.stack(frames_torch)

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


# it needs ffmpeg offset control.
def syncnet_score(video_path, audio_path, fps=25):
    def calc_pdist(feat1, feat2, vshift=10):
        win_size = vshift * 2 + 1
        feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
        dists = []

        for i in range(0, len(feat1)):
            dists.append(torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]))

        return dists
    
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
    frames, _ = get_frames(video_path, res=224)
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
    lastframe = (min_length // 640) - 5
    im_feat = []
    cc_feat = []

    for i in range(0, lastframe, batch_size):
        im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
        im_in = torch.cat(im_batch, 0)
        im_out  = __S__.forward_lip(im_in.cuda())
        im_feat.append(im_out.data.cpu())

        cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in range(i, min(lastframe, i + batch_size))]
        cc_in = torch.cat(cc_batch,0)
        cc_out  = __S__.forward_aud(cc_in.cuda())
        cc_feat.append(cc_out.data.cpu())

    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)

    dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)

    minval, minidx = torch.min(mdist, 0)

    offset = vshift - minidx
    conf   = torch.median(mdist) - minval

    dists_npy = np.array([ dist.numpy() for dist in dists ])
    return offset, minval, conf, dists_npy

def lmd_score(real_video_path, fake_video_path):
    real_frames, _ = get_frames(real_video_path)
    fake_frames, _ = get_frames(fake_video_path)

    device = torch.device(0)
    torch.cuda.set_device(device)
    preprocesser = KeypointExtractor(device)

    real_lm = preprocesser.extract_keypoint(list(real_frames))
    fake_lm = preprocesser.extract_keypoint(list(fake_frames))

    FLMD_score = 0
    MLMD_score = 0
    T, P, _  = real_lm.shape  
    for i in range(T):
        for j in range(P):
            FLMD_score += torch.norm(torch.tensor(real_lm[i, j, :] - fake_lm[i, j, :]))

            if (j >= 48):
                MLMD_score += torch.norm(torch.tensor(real_lm[i, j, :] - fake_lm[i, j, :]))
    
    return FLMD_score / (T * P), MLMD_score / (T * P)

def convert_PIL_to_numpy(images):
    return np.array([cv2.merge(list(cv2.split(np.uint8(image))[::-1])) for image in images])

def convert_numpy_to_PIL(images):
    return [Image.fromarray(cv2.merge(list(cv2.split(image)[::-1]))) for image in images]


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
    print("real CPBD:", cpbd_score(real_frames_numpy))
    print("fake CPBD:", cpbd_score(fake_frames_numpy))