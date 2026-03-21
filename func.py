import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


def count_model_params(model, name="Model"):
    """统计模型的总参数和可训练参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{name} - Total Parameters: {total_params:,}")
    print(f"{name} - Trainable Parameters: {trainable_params:,}\n")
    return total_params, trainable_params

def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

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
    return sam_deg


def apply_affine_transform(img_tensor, angle=0, translate=(0, 0), scale_x=1.0, scale_y=1.0):
    import torch.nn.functional as F
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


def zoom_hsi(hsi_tensor, x_scale=1.1, y_scale=0.9):
    """
    处理高光谱图像：缩放 -> 中心截取 -> 补零至原尺寸
    :param hsi_tensor: 输入tensor，形状为[1, 28, 256, 256]
    :param x_scale: x轴（宽）缩放比例（扩大10%为1.1）
    :param y_scale: y轴（高）缩放比例（缩小10%为0.9）
    :return: 处理后的tensor，形状保持[1, 28, 256, 256]
    """
    # 获取原尺寸
    batch, bands, h, w = hsi_tensor.shape  # h=256, w=256

    # 计算缩放后的尺寸
    new_h = int(h * y_scale)
    new_w = int(w * x_scale)

    # 缩放操作（使用双线性插值，保持通道维度在前）
    # 注意：F.interpolate要求输入为4D tensor，且需要指定空间维度
    scaled = F.interpolate(
        hsi_tensor,
        size=(new_h, new_w),  # (高, 宽)
        mode='bilinear',  # 双线性插值适合连续数据
        align_corners=False  # 不强制对齐角点
    )

    # 计算中心截取区域的坐标（若缩放后尺寸小于原尺寸，直接居中；若大于则截取中心部分）
    start_h = max(0, (new_h - h) // 2)
    start_w = max(0, (new_w - w) // 2)
    end_h = start_h + h
    end_w = start_w + w

    # 处理超出边界的情况（当缩放后尺寸小于原尺寸时）
    # 创建输出tensor并初始化为0
    output = torch.zeros_like(hsi_tensor)

    # 计算实际有效的截取区域（防止越界）
    crop_h = min(end_h, new_h) - start_h
    crop_w = min(end_w, new_w) - start_w

    # 将缩放后图像的中心区域复制到输出tensor（不足部分保持0）
    output[:, :, :crop_h, :crop_w] = scaled[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

    return output


def fcn(num_input_channels=6, num_output_channels=1, num_hidden=10):
    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden,bias=True))
    # model.add(nn.ReLU6())
    model.add(nn.ReLU())
    model.add(nn.Linear(num_hidden, num_output_channels))
    model.add(nn.Tanh())
    return model


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


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


def ss_tv_loss(input_t):
    input_t = input_t.squeeze(0).permute(1, 2, 0)
    temp1 = torch.cat((input_t[1:, :, :], input_t[-1, :, :].unsqueeze(0)), 0)
    temp2 = torch.cat((input_t[:, 1:, :], input_t[:, -1, :].unsqueeze(1)), 1)
    temp3 = torch.cat((input_t[:, :, 1:], input_t[:, :, -1].unsqueeze(2)), 2)
    temp1_, temp2_, temp3_ = temp1 - input_t, temp2 - input_t, temp3 - input_t
    tv = torch.abs(temp1_) + torch.abs(temp2_) + torch.abs(temp3_)
    return tv.mean() #- 0.5*tv2.mean()


def A(data, Phi):
    return torch.sum(data * Phi, 2)


def At(meas, Phi):
    meas = torch.unsqueeze(meas, 2).repeat(1, 1, Phi.shape[2])
    return meas * Phi


# def shift(inputs, step):
#     [h, w, nC] = inputs.shape
#     output = torch.zeros((h, w+(nC - 1)*step, nC)).to(device)
#     for i in range(nC):
#         output[:, i*step : i*step + w, i] = inputs[:, :, i]
#     del inputs
#     return output

def shift(inputs, step):
    """
    inputs: [H, W, C]
    return: [H, W + (C-1)*step, C]
    """
    H, W, C = inputs.shape
    out_W = W + (C - 1) * step
    # [H, C, W]
    x = inputs.permute(0, 2, 1).contiguous()
    # 输出: [H, C, out_W]
    output = inputs.new_zeros((H, C, out_W))
    # 每个通道对应的写入位置
    # idx shape: [1, C, W]
    idx = torch.arange(W, device=inputs.device).view(1, 1, W) + \
          torch.arange(C, device=inputs.device).view(1, C, 1) * step
    # 扩展到 batch 维度 H
    idx = idx.expand(H, C, W)
    # 一次性并行写入
    output.scatter_(dim=2, index=idx, src=x)
    # 变回 [H, out_W, C]
    return output.permute(0, 2, 1).contiguous()


def shift_back(inputs, step):
    [h, w, nC] = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = torch.roll(inputs[:, :, i], (-1)*step*i, dims=1)
    output = inputs[:, 0 : w - step*(nC - 1), :]
    return output


def get_input(tensize, const=10.0):
    inp = torch.rand(tensize)/const
    inp = torch.autograd.Variable(inp, requires_grad=True).to(device)
    inp = torch.nn.Parameter(inp)
    return inp


def calculate_tv(x):
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    ib = torch.arange(1, N[2]+1)
    ib[-1] = N[2]-1

    x1 = x[:,ir,:] - x
    x2 = x[idx,:,:] - x
    x3 = x[:,:,ib] - x
    tv = (x1)**2 + (x2)**2 + (x3)**2
    return torch.mean(torch.sum(tv, 2))


def calculate_psnr_tensor(data, recon):
    mse = torch.mean((recon - data)**2)
    if mse == 0:
        return 100
    Pixel_max = 1.
    return 20 * torch.log10(Pixel_max / torch.sqrt(mse))


def ssim_(data, recon):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    data = data.astype(np.float64)
    recon = recon.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(data, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(recon, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(data ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(recon ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(data * recon, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim_index(im1, im2):
    if im1.ndim == 2:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)
    elif im1.ndim == 3:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out


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


def calculate_psnr(data, recon):
    mse = torch.mean((recon - data)**2)
    if mse == 0:
        return 100
    Pixel_max = 1.0
    return 20 * torch.log10(Pixel_max / torch.sqrt(mse))


def cssim(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    SSIM = ssim_index(img, img_clean)
    return SSIM


def cpsnr(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    PSNR = peak_signal_noise_ratio(img, img_clean, data_range=255)
    return PSNR