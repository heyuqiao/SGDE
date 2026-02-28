import os
import sys
import time

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torch_ema import ExponentialMovingAverage

import cv2
import numpy as np
from scipy import io as sio
from tqdm import tqdm

from csi.config import get_cfg
from csi.engine import default_argument_parser, default_setup
from csi.data import CSITrainDataset, LoadVal, LoadTSATestMeas, shift_back_batch, generate_mask_3d, generate_mask_3d_shift, gen_meas_torch
from csi.architectures import DERNN_LNLT
from csi.utils.schedulers import get_cosine_schedule_with_warmup
from csi.losses import CharbonnierLoss, TVLoss
from csi.metrics import torch_psnr, torch_ssim, sam
from csi.utils.utils import checkpoint

def video_gen(img, mode=0, FPS=5, output_file="./hyer_video.avi"):
    """
        mode = 0 灰色视频， 1 连续视频， 2跳波段视频
    """
    height, width, length = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if mode == 0:
        out = cv2.VideoWriter(output_file, fourcc, FPS, (width, height), isColor=False)
        for channel in range(length):
            frame = img[:, :, channel]
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            out.write(frame_normalized)

    if mode == 1:
        out = cv2.VideoWriter(output_file, fourcc, FPS, (width, height), isColor=True)
        for f in range(length):
            if f + 2 >= length:
                break
            b_ch = img[:, :, f]
            g_ch = img[:, :, f + 1]  # G
            r_ch = img[:, :, f + 2]  # R
            rgb = np.stack([b_ch, g_ch, r_ch], axis=-1)  # cv2用BGR顺序，所以这里先放b
            rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            out.write(rgb)

    if mode == 2:
        if mode == 1:
            out = cv2.VideoWriter(output_file, fourcc, FPS, (width, height), isColor=True)
            for f in range(length // 3):
                b_ch = img[:, :, f]
                g_ch = img[:, :, f + 9]  # G
                r_ch = img[:, :, f + 18]  # R
                rgb = np.stack([b_ch, g_ch, r_ch], axis=-1)  # cv2用BGR顺序，所以这里先放b
                rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                out.write(rgb)

    out.release()
    cv2.destroyAllWindows()


def ss_tv_loss(input_t):
    input_t = input_t.squeeze(0).permute(1, 2, 0)
    temp1 = torch.cat((input_t[1:, :, :], input_t[-1, :, :].unsqueeze(0)), 0)
    temp2 = torch.cat((input_t[:, 1:, :], input_t[:, -1, :].unsqueeze(1)), 1)
    temp3 = torch.cat((input_t[:, :, 1:], input_t[:, :, -1].unsqueeze(2)), 2)
    temp1_, temp2_, temp3_ = temp1 - input_t, temp2 - input_t, temp3 - input_t
    tv = torch.abs(temp1_) + torch.abs(temp2_) + torch.abs(temp3_)
    #tv2 = (temp1_)**2 + (temp2_)**2 + (temp3_)**2
    return tv.mean() #- 0.5*tv2.mean()

args = default_argument_parser().parse_args()
args.config_file = "configs/dernn_lnlt_5stg_real.yaml"
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SGDE
mask_256 = torch.from_numpy(sio.loadmat('/dataset/imaging/TSA_real_data/mask.mat')['mask'])

mask = torch.zeros((660, 660 + 2*(28 - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, 28)
grid = F.affine_grid(torch.tensor([[[ 9.9959350e-01, -8.8151508e-05,  1.5117968e-03],
                                    [ 8.8151508e-05,  9.9959350e-01, -1.2136354e-03]]]), size=mask_256.unsqueeze(0).unsqueeze(0).size(), align_corners=False)
mask_256 = F.grid_sample(mask_256.unsqueeze(0).unsqueeze(0), grid, align_corners=False)
mask_256.squeeze_()

for i in range(28):
    mask_3d[:, i*2:i*2+660, i] = mask_256
mask_test = mask_3d.to(device)
mask_test = mask_test.permute(2, 0, 1)
# SGDE


test_meas = LoadTSATestMeas(cfg.DATASETS.TEST.PATH).to(device)

model = eval(cfg.MODEL.DENOISER.TYPE)(cfg).to(device)
ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

if cfg.PRETRAINED_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
    save_state = torch.load(cfg.PRETRAINED_CKPT_PATH, map_location=device)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])

def test(test_meas, name="test_a"):
    model.eval()
    model_out = []
    data = {}
    data['Y'] = test_meas / test_meas.max() * 0.8  # 难道是模拟暗场？

    B, _, _ = test_meas.shape
    data['mask'] = mask_test.unsqueeze(0).tile((B, 1, 1, 1))
    data['H'] = shift_back_batch(test_meas, step=cfg.DATASETS.STEP, nC=cfg.DATASETS.WAVE_LENS)
        
    with torch.no_grad():
        with ema.average_parameters():
            model_out = model(data)

    
    for i in range(B):
        out_plot = F.interpolate(model_out[i:i+1, :, :, :], size=(128, 128))
        if name == "TSA": out_plot = torch.flip(out_plot, dims=(2, 3))
       
        
    model_out = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)

    pred = np.transpose(model_out, (0, 2, 3, 1)).astype(np.float32)
    print(ss_tv_loss(torch.from_numpy(pred)[:1].permute(0, 3, 1, 2)))

    model.train()

    return model_out


def main():
    test_out = test(test_meas, "TSA")
    for i in range(test_out.shape[0]):
        video_gen(test_out[i], mode=0, FPS=5, output_file="./hyer_video_cal" + str(i) + ".avi")
    sio.savemat("./results/dernn_lnlt_5stg_real.mat", {"pred": test_out})
    

if __name__ == "__main__":
    main()