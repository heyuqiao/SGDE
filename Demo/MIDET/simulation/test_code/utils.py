import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from torch.nn import functional as F
import math
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

def generate_shift_masks(mask_path, batch_size, device):
    mask = sio.loadmat(mask_path + '/mask_3d_shift.mat')
    mask_3d_shift = mask['mask_3d_shift']
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).to(torch.float32).to(device)

    #
    # kernel = get_shift_blur_kernel(dx=torch.tensor(0), dy=torch.tensor(0), sigma=torch.tensor(0.56),
    #                                kernel_size=5)
    # kernel = kernel.unsqueeze(0).unsqueeze(0)
    # kernel = kernel.repeat(28, 1, 1, 1)
    # Phi_batch = F.conv2d(input=Phi_batch, weight=kernel, padding=2, groups=28)
    #

    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    return Phi_batch,Phi_s_batch

def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']

    #


    #

    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def LoadTraining(path, debug=False):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    for i in range(len(scene_list) if not debug else 5):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            img = img.astype(np.float32)
            imgs.append(img)
            print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data





def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename



def shuffle_crop(train_data, batch_size, crop_size=256, augment=True):
    if augment:
        flag = random.randint(0, 1)
        if flag:
            index = np.random.choice(range(len(train_data)), batch_size)
            processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
            for i in range(batch_size):
                h, w, _ = train_data[index[i]].shape
                x_index = np.random.randint(0, h - crop_size)
                y_index = np.random.randint(0, w - crop_size)
                processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
            for i in range(gt_batch.shape[0]):
                gt_batch[i] = augment_1(gt_batch[i])
        else:
            gt_batch = []
            processed_data = np.zeros((4, 128, 128, 28), dtype=np.float32)
            for i in range(batch_size):
                sample_list = np.random.randint(0, len(train_data), 4)
                for j in range(4):
                    h, w, _ = train_data[sample_list[j]].shape
                    x_index = np.random.randint(0, h-crop_size//2)
                    y_index = np.random.randint(0, w-crop_size//2)
                    processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
                generated_sample = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))  # [4,28,128,128]
                gt_batch.append(augment_2(generated_sample))
            gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 28), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))

    return gt_batch


def augment_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def augment_2(generate_gt):
    c, h, w = generate_gt.shape[1],256,256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c,h,w)
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

def shift(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs,step=2):
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

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

def gen_meas_torch(data_batch, mask3d_batch):
    [batch_size, nC, H, W] = data_batch.shape

    # hyq
    mask_ = mask3d_batch[0, 0, :256, :256]
    mask_256_new, matrix = apply_affine_transform(mask_.unsqueeze(0).unsqueeze(0),
                                                  angle=0.2, translate=(0.5, 0.0), scale_x=1.002, scale_y=1.002)
    mask3d_batch = mask_256_new.expand([batch_size, nC, H, W]).cuda().float()  # [10,28,256,256]
    # hyq

    # mask3d_batch = (mask3d_batch[0, :, :, :]).expand([batch_size, nC, H, W]).cuda().float()  # [10,28,256,256]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    return meas


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def init_mask(mask_path, mask_type, batch_size, device="cuda"):
    if mask_type == 'Phi':
        Phi_batch,_ = generate_shift_masks(mask_path, batch_size, device)
        input_mask = Phi_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch = generate_masks(mask_path, batch_size)
        input_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size,device)
        input_mask = (input_batch, Phi_s_batch)
    return Phi_batch,input_mask

def init_meas(gt, phi, input_setting):
    if input_setting == 'Y':
        input_meas = gen_meas_torch(gt, phi)
    return input_meas


def checkpoint(model, ema, optimizer, scheduler,  epoch, model_path, logger):
    save_dict = {}
    save_dict['model'] = model.state_dict()
    save_dict['ema'] = ema.state_dict()
    save_dict['optimizer'] = optimizer.state_dict()
    save_dict['scheduler'] = scheduler.state_dict()
    save_dict['epoch'] = epoch
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(save_dict, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))



def seed_everything(
    seed = 3407,
    deterministic = False, 
):
    """Set random seed.
    Args:
        seed (int): Seed to be used, default seed 3407, from the paper
        Torch. manual_seed (3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision[J]. arXiv preprint arXiv:2109.08203, 2021.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False