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
import math
import cv2
import numpy as np
from scipy import io as sio
from tqdm import tqdm

from csi.config import get_cfg
from csi.engine import default_argument_parser, default_setup
from csi.data import CSITrainDataset, LoadVal, LoadTSATestMeas, shift_back_batch, generate_mask_3d, \
    generate_mask_3d_shift, gen_meas_torch
from csi.architectures import DERNN_LNLT
from csi.utils.schedulers import get_cosine_schedule_with_warmup
from csi.losses import CharbonnierLoss, TVLoss
from csi.metrics import torch_psnr, torch_ssim, sam
from csi.utils.utils import checkpoint

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def A(data, Phi):
    return torch.sum(data * Phi, 2)

def shift(inputs, step=2):
    [h, w, nC] = inputs.shape
    output = torch.zeros((h, w+(nC - 1)*step, nC)).to(device)
    for i in range(nC):
        output[:, i*step : i*step + w, i] = inputs[:, :, i]
    del inputs
    return output

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input

def fcn(num_input_channels=6, num_output_channels=1, num_hidden=10):
    model = nn.Sequential(
        nn.Linear(num_input_channels, num_hidden, bias=True),
        nn.ReLU(),
        nn.Linear(num_hidden, num_output_channels),
        nn.Tanh()
    )
    return model


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


args = default_argument_parser().parse_args()
args.config_file = "configs/dernn_lnlt_5stg_real.yaml"
cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.freeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 原本的Mask是660x660的，但是这里是660x714的，观察了一下，居然是移动的，没有黑边？很奇怪啊
# mask_test = generate_mask_3d_shift(mask_path=cfg.DATASETS.TEST.MASK_PATH).to(device)  # 28x660x714

# hyq
mask_256 = torch.from_numpy(sio.loadmat('/dataset/imaging/TSA_real_data/mask.mat')['mask'])

mask = torch.zeros((660, 660 + 2 * (28 - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, 28)
# grid = F.affine_grid(torch.tensor([[[9.9983e-01, 1.2426e-04, 1.4998e-03],
#                                     [-1.2426e-04, 9.9983e-01, -1.1969e-03]]]),
#                      size=mask_256.unsqueeze(0).unsqueeze(0).size(), align_corners=False)
# mask_256 = F.grid_sample(mask_256.unsqueeze(0).unsqueeze(0), grid, align_corners=False)
mask_256.squeeze_()

for i in range(28):
    mask_3d[:, i * 2:i * 2 + 660, i] = mask_256
mask_test = mask_3d.to(device)
mask_test = mask_test.permute(2, 0, 1)
# hyq

# 是同时测了5个
test_meas = LoadTSATestMeas(cfg.DATASETS.TEST.PATH).to(device)

model = eval(cfg.MODEL.DENOISER.TYPE)(cfg).to(device)
# 影子参数？通过 EMA 平滑参数更新，帮助模型在训练中更稳定地收敛，最终提升性能。
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

    # hyq
    net_input_kernel = get_noise(6, 'noise', (1, 1)).type(torch.cuda.FloatTensor)
    net_input_kernel.squeeze_()
    net_kernel = fcn(6, 6)
    net_kernel = net_kernel.type(torch.cuda.FloatTensor)
    net_kernel.train()
    kernel_params = list(net_kernel.parameters())
    optimizer = torch.optim.Adam([
        {'params': kernel_params, 'lr': 1e-3},  # 仿射参数回归网络（FCN）更新慢 10 倍
    ])

    Phi_T = mask_test.permute(1, 2, 0)[:, 0: mask_test.permute(1, 2, 0).shape[0], 0].unsqueeze(0).unsqueeze(0)
    mask_new = torch.zeros((mask_test.permute(1, 2, 0).shape[0],  mask_test.permute(1, 2, 0).shape[1]))
    Phi_new = torch.unsqueeze(mask_new, 2).repeat(1, 1, mask_test.permute(1, 2, 0).shape[2]).to(device)
    cols = torch.arange(660, device=device) + torch.arange(28, device=device)[:,None] * 2  # (28, 256)
    cols = cols.unsqueeze(0)
    rows = torch.arange(660, device=device)[:, None, None]
    channels = torch.arange(28, device=device)[None, :, None]  # (1, 28, 1)
    loss_l1 = torch.nn.L1Loss().to(device)
    # hyq

    for idx in range(100):
        with torch.no_grad():
            with ema.average_parameters():
                model_out = model(data)

        for i in range(1):
            out_k = net_kernel(net_input_kernel).float().contiguous()
            out_k = out_k.view(1, 6)  # 保证二维
            max_translation = 2.0
            h, w = Phi_T.shape[2], Phi_T.shape[3]
            tx = out_k[:, 4] * 2 * max_translation / (w - 1)
            ty = out_k[:, 5] * 2 * max_translation / (h - 1)
            max_angle = 1 * math.pi / 180
            angle = out_k[:, 2] * max_angle
            scale = 1.0 + out_k[:, 3] * 0.004
            theta = torch.zeros(1, 2, 3, device=out_k.device, dtype=torch.float32)
            theta[:, 0, 0] = torch.cos(angle) * scale
            theta[:, 0, 1] = -torch.sin(angle) * scale
            theta[:, 0, 2] = tx
            theta[:, 1, 0] = torch.sin(angle) * scale
            theta[:, 1, 1] = torch.cos(angle) * scale
            theta[:, 1, 2] = ty
            grid = F.affine_grid(theta, size=Phi_T.size(), align_corners=False)
            Phi_ = F.grid_sample(Phi_T, grid, align_corners=False)
            Phi_ = Phi_.squeeze()
            Phi_expanded = Phi_[:, :, None].to(device)
            Phi_new_updated = Phi_new.clone()
            Phi_new_updated[rows, cols, channels] = Phi_expanded.permute(0, 2, 1)

            pred_meas = A(shift(model_out[0].permute(1, 2, 0), 2).to(device), Phi_new_updated.to(device))
            loss = 0.07 * loss_l1(data['Y'][0], pred_meas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(theta)
        data['mask'] = Phi_new_updated.permute(2, 0, 1).unsqueeze(0).tile((B, 1, 1, 1))
        data['H'] = shift_back_batch(Phi_new_updated.permute(2, 0, 1), step=cfg.DATASETS.STEP, nC=cfg.DATASETS.WAVE_LENS)



    for i in range(B):
        out_plot = F.interpolate(model_out[i:i + 1, :, :, :], size=(128, 128))
        if name == "TSA": out_plot = torch.flip(out_plot, dims=(2, 3))

    model_out = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()

    return model_out


def main():
    test_out = test(test_meas, "TSA")
    for i in range(test_out.shape[0]):
        video_gen(test_out[i], mode=0, FPS=5, output_file="./hyer_video_cal" + str(i) + ".avi")
    sio.savemat("./results/dernn_lnlt_5stg_real.mat", {"pred": test_out})


if __name__ == "__main__":
    main()