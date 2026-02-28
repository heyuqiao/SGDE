from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
from pytorch_msssim import ssim


def calculate_ssim(data, recon, border=0):
    if not data.shape == recon.shape:
        raise ValueError('Data size must have the same dimensions!')
    if not data.dtype == recon.dtype:
        data, recon = data.float(), recon.float()

    h, w = data.shape[:2]
    data = data[border:h - border, border:w - border]
    recon = recon[border:h - border, border:w - border]
    if data.ndim == 2:
        return ssim_(data, recon)
    elif data.ndim == 3:
        return ssim(torch.unsqueeze(data, 0).permute(3, 0, 1, 2), torch.unsqueeze(recon, 0).permute(3, 0, 1, 2), data_range=1).data

def sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    #
    return sam_deg


def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC
#

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Intialize mask
Phi_batch_test,input_mask = init_mask(
    opt.mask_path, 
    opt.input_mask, 
    10, 
    device=device)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# model
model = model_generator(opt, device)
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

if opt.pretrained_model_path:
    # print(f"===> Loading Checkpoint from {opt.pretrained_model_path}")
    # save_state = torch.load(opt.pretrained_model_path, map_location=device)
    # state_dict = save_state['model']
    # state_ema = save_state['ema']
    # print(state_ema['collected_params'])
    # keys = []
    # new_ema = []
    # for k,v in state_dict.items():    
    #     if k.startswith('stage_model.1.r') or k.startswith('stage_model.0.r'):       
    #         continue    
    #     keys.append(k)
    # new_dict = {k:state_dict[k] for k in keys} 
    #model.load_state_dict(new_dict)
    # print(f"===> Loading Checkpoint from {opt.pretrained_model_path}")
    save_state = torch.load(opt.pretrained_model_path, map_location=device)
    model.load_state_dict(save_state['model'])
    ema.load_state_dict(save_state['ema'])

def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.to(torch.float32).to(device)
    input_meas = init_meas(test_gt, Phi_batch_test, opt.input_setting)
    model.eval()

    #

    with torch.no_grad():
        with ema.average_parameters():
            model_out = model(input_meas, input_mask)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()
    return pred, truth

def main():
    import time
    start = time.perf_counter()
    pred, truth = test(model)
    end = time.perf_counter()
    print(f"运行时间: {end - start:.4f} 秒")

    psnr_list, ssim_list, sam_list = [], [], []
    for i in range(len(pred)):
        psnr_val = torch_psnr(torch.from_numpy(pred[i, :, :, :]).permute(2, 0, 1), torch.from_numpy(truth[i, :, :, :]).permute(2, 0, 1))
        ssim_val = calculate_ssim(torch.from_numpy(truth[i, :, :, :]), torch.from_numpy(pred[i, :, :, :]))
        sam_val = sam(truth[i, :, :, :], pred[i, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val)
    print("psnr:" + str(np.asarray(psnr_list)))
    print("ssim:" + str(np.asarray(ssim_list)))
    print("sam:" + str(np.asarray(sam_list)))

    # name = opt.outf + 'Test_result.mat'
    name = opt.outf + '复杂变换.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})
    
    

if __name__ == '__main__':
    main()