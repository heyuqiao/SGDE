import torch
import os
import argparse
from utils import dataparallel
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import cv2
from torch.nn import functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='/dataset/imaging/TSA_real_data/Measurements/', type=str,help='path of data')
parser.add_argument('--mask_path', default='/dataset/imaging/TSA_real_data/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=660, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=5, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--pretrained_model_path", default='/pythonCode/MIDET-main/checkpoints/real/real.pth', type=str)
opt = parser.parse_args()
print(opt)

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

def prepare_data(path, file_num):
    HR_HSI = np.zeros((((660,714,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,idx] = data['meas_real']
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def load_mask(path,size=660):
    ## load mask
    data = sio.loadmat(path)
    mask = data['mask']

    # SGDE
    mask = torch.Tensor(mask)
    grid = F.affine_grid(torch.tensor([[[9.9959350e-01, -8.8151508e-05, 1.5117968e-03],
                                        [8.8151508e-05, 9.9959350e-01, -1.2136354e-03]]]),
                         size=mask.unsqueeze(0).unsqueeze(0).size(), align_corners=False)
    mask = F.grid_sample(mask.unsqueeze(0).unsqueeze(0), grid, align_corners=False)
    mask.squeeze_()
    mask = mask.cpu().numpy()
    # SGDE

    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)

HR_HSI = prepare_data(opt.data_path, 5)
mask_3d_shift, mask_3d_shift_s = load_mask(opt.mask_path)


model = torch.load(opt.pretrained_model_path)
model = model.eval()
model = dataparallel(model, 1)
psnr_total = 0
k = 0
for j in range(5):
    with torch.no_grad():
        meas = HR_HSI[:,:,j]
        meas = meas / meas.max() * 0.8
        meas = torch.FloatTensor(meas)
        # meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
        input = meas.unsqueeze(0)
        input = Variable(input)
        input = input.cuda()
        mask_3d_shift = mask_3d_shift.cuda()
        mask_3d_shift_s = mask_3d_shift_s.cuda()
        mask=mask_3d_shift,mask_3d_shift_s
        out = model(input, mask)
        result = out
        result = result.clamp(min=0., max=1.)
        video_gen(result[0, :, :, :].permute(1, 2, 0).cpu().numpy(), output_file="./hyer_video" + str(j) + ".avi")
        x = 100
    k = k + 1

    save_path='/pythonCode/MIDET-main/result/'
    if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
        os.makedirs(save_path)
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    save_file = save_path + f'{j}.mat'
    sio.savemat(save_file, {'res':res})
