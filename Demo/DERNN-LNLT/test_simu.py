import os
import sys
import time

# add python path of PadleDetection to sys.path
import matplotlib.pyplot

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
import matplotlib.pyplot as plt
from csi.config import get_cfg
from csi.engine import default_argument_parser, default_setup
from csi.data import CSITrainDataset, LoadVal, LoadTSATestMeas, shift_back_batch, generate_mask_3d, generate_mask_3d_shift, gen_meas_torch
from csi.architectures import DERNN_LNLT
from csi.utils.schedulers import get_cosine_schedule_with_warmup
from csi.losses import CharbonnierLoss, TVLoss
from csi.metrics import torch_psnr, torch_ssim, sam
from csi.utils.utils import checkpoint
import matplotlib.pyplot as plt
import math
#
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

def apply_affine_transform(img_tensor, angle=0, translate=(0, 0), scale_x=1.0, scale_y=1.0):
    """
    对输入图像张量执行仿射变换
    参数：
        img_tensor: Tensor, 形状 [1, C, H, W]
        angle: float, 旋转角度（度）
        translate: tuple(x, y)，平移（像素）
        scale_x, scale_y: 缩放比例
    返回：
        out: 变换后的图像张量
        theta: 仿射矩阵 (1, 2, 3)
    """
    B, C, H, W = img_tensor.shape
    theta_deg = angle * math.pi / 180.0
    cos_a = math.cos(theta_deg)
    sin_a = math.sin(theta_deg)

    # 平移转换到[-1,1]归一化坐标系
    tx = 2 * translate[0] / W
    ty = 2 * translate[1] / H

    # 仿射矩阵
    theta = torch.tensor([[
        [cos_a * scale_x, -sin_a * scale_y, tx],
        [sin_a * scale_x,  cos_a * scale_y, ty]
    ]], dtype=img_tensor.dtype, device=img_tensor.device)

    # 生成采样网格并采样
    grid = F.affine_grid(theta, size=img_tensor.size(), align_corners=False)
    out = F.grid_sample(img_tensor, grid, align_corners=False)

    return out, theta
#

def ss_tv_loss(input_t):
    input_t = input_t.squeeze(0).permute(1, 2, 0)
    temp1 = torch.cat((input_t[1:, :, :], input_t[-1, :, :].unsqueeze(0)), 0)
    temp2 = torch.cat((input_t[:, 1:, :], input_t[:, -1, :].unsqueeze(1)), 1)
    temp3 = torch.cat((input_t[:, :, 1:], input_t[:, :, -1].unsqueeze(2)), 2)
    temp1_, temp2_, temp3_ = temp1 - input_t, temp2 - input_t, temp3 - input_t
    tv = torch.abs(temp1_) + torch.abs(temp2_) + torch.abs(temp3_)
    #tv2 = (temp1_)**2 + (temp2_)**2 + (temp3_)**2
    return tv.mean() #- 0.5*tv2.mean()

#
def get_shift_blur_kernel(dx, dy, sigma, kernel_size=5, device='cuda:0'):
    """
    构造一个卷积核，支持亚像素偏移(dx, dy) + 模糊(sigma)
    dx, dy: 位移，float，可为负
    sigma: 高斯模糊标准差
    kernel_size: 核大小, 必须为奇数
    """
    # 创建坐标网格
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')

    # 先平移坐标实现亚像素位移
    xx = xx - dx
    yy = yy - dy

    # 高斯模糊
    if sigma > 1e-2:
        # 高斯模糊
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    else:
        x0 = torch.floor((kernel_size - 1) / 2 - dx)
        y0 = torch.floor((kernel_size - 1) / 2 - dy)
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = ((kernel_size - 1) / 2 - dx) - x0
        wx0 = 1 - wx1
        wy1 = ((kernel_size - 1) / 2 - dy) - y0
        wy0 = 1 - wy1

        kernel = torch.zeros((kernel_size, kernel_size), device=device)

        for ix, wx in zip([x0, x1], [wx0, wx1]):
            for iy, wy in zip([y0, y1], [wy0, wy1]):
                if 0 <= ix < kernel_size and 0 <= iy < kernel_size:
                    kernel[int(iy), int(ix)] = wx * wy

    kernel /= kernel.sum()  # 归一化
    return kernel
#

args = default_argument_parser().parse_args()
args.config_file = "configs/dernn_lnlt_5stg_simu.yaml"
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mask = generate_mask_3d_shift(mask_path=cfg.DATASETS.VAL.MASK_PATH).to(device)



val_datas = LoadVal(cfg.DATASETS.VAL.PATH)

model = eval(cfg.MODEL.DENOISER.TYPE)(cfg).to(device)

ema = ExponentialMovingAverage(model.parameters(), decay=cfg.MODEL.EMA.DECAY)

if cfg.PRETRAINED_CKPT_PATH:
    print(f"===> Loading Checkpoint from {cfg.PRETRAINED_CKPT_PATH}")
    save_state = torch.load(cfg.PRETRAINED_CKPT_PATH, map_location=device)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])



def eval():
    psnr_list, ssim_list, sam_list = [], [], []
    val_H = []
    val_Y = []
    val_gt = []
    for val_label in val_datas['hsi']:
        val_label = torch.from_numpy(val_label).permute(2, 0, 1).to(device).float()

        # SGDE
        # mask_ = torch.from_numpy(sio.loadmat('/pythonCode/LCTC-main/Data/KAIST_Dataset/mask/mask256.mat')['mask']).cuda()
        mask_ = mask[0, :256, :256]
        mask_3d = torch.zeros((256, 310, 28))
        mask_256_new, matrix = apply_affine_transform(mask_.unsqueeze(0).unsqueeze(0),
                                                      angle=0.4, translate=(0.2, 0.0), scale_x=1.000, scale_y=1.000)
        mask_256_new.squeeze_()
        for i in range(28):
            mask_3d[:, i * 2:i * 2 + 256, i] = mask_256_new
        mask_ = mask_3d.to(device).permute(2, 0, 1)
        video_gen(val_label.permute(1, 2, 0).cpu().numpy())
        # SGDE

        YH = gen_meas_torch(val_label, mask_, step=cfg.DATASETS.STEP, wave_len=cfg.DATASETS.WAVE_LENS, mask_type=cfg.DATASETS.MASK_TYPE)
        val_H.append(YH['H'].to(device))
        val_Y.append(YH['Y'].to(device))
        val_gt.append(val_label)
    val_gt = torch.stack(val_gt)
    val_H = torch.stack(val_H)
    val_Y = torch.stack(val_Y)
    data = {}
    data['hsi'] = val_gt
    data['H'] = val_H
    B, _, _, _ = val_H.shape
    data['mask'] = mask.unsqueeze(0).tile((B, 1, 1, 1))
    data['Y'] = val_Y
    model.eval()
    begin = time.time()
    with torch.no_grad():
        with ema.average_parameters():
            out = model(data)  # 输出
            model_out = out

    for i in range(len(model_out)):
        psnr_val = torch_psnr(model_out[i, :, :, :], val_gt[i, :, :, :])
        ssim_val = torch_ssim(model_out[i, :, :, :], val_gt[i, :, :, :])
        sam_val = sam(model_out[i, :, :, :].permute(1, 2, 0).cpu().numpy(), val_gt[i, :, :, :].permute(1, 2, 0).cpu().numpy())
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val)

    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    print(ss_tv_loss(torch.from_numpy(pred)[:1].permute(0, 3, 1, 2)))
    truth = np.transpose(val_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    sam_mean = np.mean(np.asarray(sam_list))

    end = time.time()
    print("psnr:" + str(np.asarray(psnr_list)))
    print("ssim:" + str(np.asarray(ssim_list)))
    print("sam:" + str(np.asarray(sam_list)))
    print('===> testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, time: {:.2f}'
                .format(psnr_mean, ssim_mean, sam_mean, (end - begin)))
    model.train()
    return pred, truth, psnr_list, ssim_list, sam_list, psnr_mean, ssim_mean, sam_mean


def main():
    import time
    start = time.perf_counter()
    (pred, truth, psnr_all, ssim_all, sam_all, psnr_mean, ssim_mean, sam_mean) = eval()
    end = time.perf_counter()
    print(f"运行时间: {end - start:.4f} 秒")
    # sio.savemat("./results/dernn_lnlt_9stg_star_simu.mat", {"pred": pred, "truth" : truth})
    for i in range(pred.shape[0]):
        video_gen(pred[i], mode=0, FPS=5, output_file="./hyer_video_cal" + str(i) + ".avi")
    sio.savemat("./results/缩放0.004倍.mat", {"pred": pred})


if __name__ == "__main__":
    main()