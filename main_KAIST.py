import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from func import *
from numpy import *
import scipy.io as sio
from optimization import ADMM_Iter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)
import time

# -----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='scene01', help="scene01-10")
parser.add_argument('--method', default='SK-LCTC', help="LCTC, SK-LCTC or MK-LCTC")
parser.add_argument('--unit_size', default=1, help="encoding unit size of mask")

# The following are the parameters of Our SGDE
parser.add_argument('--max_translation', default=2.0, help="Range of single core calibration")
parser.add_argument('--max_angle', default=0.4, help="Range of single core calibration")
parser.add_argument('--max_scale', default=0.004, help="Range of single core calibration")
parser.add_argument('--kernel_num', default=8, help='It is for MK-LCTC')
parser.add_argument('--add_order', default='forward', help="forward or reverse, it is for MK-LCTC")
parser.add_argument('--switch_iters', default=1000, help="Switching from SK to MK, it is for MK-LCTC")

# The following are the affine transformation parameters for Mask,
# with the aim of introducing controllable perturbations
parser.add_argument('--translate_x', default=0.0, help="Affine params")
parser.add_argument('--translate_y', default=0, help="Affine params")
parser.add_argument('--angle', default=0, help="Affine params")
parser.add_argument('--scale_x', default=1.000, help="Affine params")
parser.add_argument('--scale_y', default=1.000, help="Affine params")

# The following are the original parameters of LCTC
parser.add_argument('--iter_num', default=1, help="Maximum number of iterations")
parser.add_argument('--lambda_', default=1, help="Facotr of the LCTC regularization")
parser.add_argument('--LR_iter', default=6000, help="Training epochs of CTC networks")
parser.add_argument('--R_iter', default=1000, help="Reduced Training epochs of CTC networks")
parser.add_argument('--lambda_R', default=0.07, help="Factor of TV/SSTV regularization in CTC")
parser.add_argument('--ip_BI', default=4, help="The number of channel of input")
parser.add_argument('--step', default=2, help="step for spectral shifting")

args = parser.parse_args()
# ----------------------- Data Configuration -----------------------#
dataset_dir = './Data/KAIST/GT'
data_name = args.scene
results_dir = './Results/' + data_name + '/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
matfile = dataset_dir + '/' + data_name + '.mat'
data_truth = torch.from_numpy(sio.loadmat(matfile)['img'])
mask_256 = torch.from_numpy(sio.loadmat('./Data/KAIST/mask/mask256.mat')['mask'])

truth_tensor = data_truth.permute(2, 0, 1).unsqueeze(0).to(device)
_, nC, h, w = truth_tensor.shape
data_truth_shift = torch.zeros((h, w + args.step * (nC - 1), nC)).to(device)
for i in range(nC):
    data_truth_shift[:, i * args.step:i * args.step + h, i] = data_truth[:, :, i]

# ----------------------- Mask Configuration -----------------------#
mask = torch.zeros((h, w + args.step * (nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
for i in range(nC):
    mask_3d[:, i * args.step:i * args.step + h, i] = mask_256
Phi = mask_3d.to(device)

mask_new = torch.zeros((h, w + args.step * (nC - 1)))
mask_3d_new = torch.unsqueeze(mask_new, 2).repeat(1, 1, nC)
mask_256_new, matrix = apply_affine_transform(mask_256.unsqueeze(0).unsqueeze(0),
                                              angle=args.angle,
                                              translate=(args.translate_x, args.translate_y),
                                              scale_x=args.scale_x,
                                              scale_y=args.scale_y)
print("GT affine params: " + str(matrix))
mask_256_new.squeeze_()
for i in range(nC):
    mask_3d_new[:, i * args.step:i * args.step + h, i] = mask_256_new
Phi_new = mask_3d_new.to(device)
meas = torch.sum(Phi_new * data_truth_shift, 2)

# -------------------------- Optimization --------------------------#
start_time = time.time()
x_rec = ADMM_Iter(meas.to(device), Phi.to(device), truth_tensor, args)
end_time = time.time()
print(f"runtime: {end_time - start_time:.2f} s")
sio.savemat(results_dir + '/{}.mat'.format(data_name), {'img': x_rec.cpu().numpy()})
if os.path.exists('./Results/model_weights.pth'):
    os.remove('./Results/model_weights.pth')
