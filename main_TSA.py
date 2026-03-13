import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from func import *
from numpy import *
import scipy.io as sio
from optimization import ADMM_Iter
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random.seed(5)

# -----------------------Opti. Configuration -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--meas_path', default='./Data/TSA/meas/scene01.mat')
parser.add_argument('--mask_path', default='./Data/TSA/mask/mask.mat')
parser.add_argument('--results_dir', default='./Results/')
parser.add_argument('--method', default='SK-TC', help="LCTC, SK-LCTC or MK-LCTC")
parser.add_argument('--unit_size', default=1, help="encoding unit size of mask")

# The following are the parameters of Our SGDE
parser.add_argument('--max_translation', default=3.0, help="Range of single core calibration")
parser.add_argument('--max_angle', default=0.8, help="Range of single core calibration")
parser.add_argument('--max_scale', default=0.008, help="Range of single core calibration")
parser.add_argument('--kernel_num', default=8, help='It is for MK-LCTC')
parser.add_argument('--add_order', default='forward', help="forward or reverse, it is for MK-LCTC")
parser.add_argument('--switch_iters', default=1500, help="Switching from SK to MK, it is for MK-LCTC")

# The following are the original parameters of LCTC
parser.add_argument('--iter_num', default=1, help="Maximum number of iterations")
parser.add_argument('--lambda_', default=1, help="Facotr of the LCTC regularization")
parser.add_argument('--LR_iter', default=4000, help="Training epochs of CTC networks")
parser.add_argument('--R_iter', default=1000, help="Reduced Training epochs of CTC networks")
parser.add_argument('--lambda_R', default=0.07, help="Factor of TV/SSTV regularization in CTC")
parser.add_argument('--ip_BI', default=4, help="The number of channel of input")
parser.add_argument('--step', default=2, help="step for spectral shifting")

args = parser.parse_args()
# ----------------------- Data Configuration -----------------------#
dataset_dir = './Data/TSA/meas/'
meas = torch.from_numpy(sio.loadmat(args.meas_path)['meas_real'])
meas = (meas - torch.min(meas)) / (torch.max(meas) - torch.min(meas))
mask_660 = torch.from_numpy(sio.loadmat(args.mask_path)['mask'])
mask_660 = torch.from_numpy(sio.loadmat("Demo/Estimated TSA mask/mask.mat")['mask'])

min_s = torch.min(mask_660)
max_s = torch.max(mask_660)
mask_660 = (mask_660 - torch.min(mask_660)) / (torch.max(mask_660) - torch.min(mask_660))


if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
data_truth = torch.zeros((660, 660, 28))
truth_tensor = data_truth.permute(2, 0, 1).unsqueeze(0).to(device)
_, nC, h, w = truth_tensor.shape

# ----------------------- Mask Configuration -----------------------#
mask = torch.zeros((h, w + args.step * (nC - 1)))
mask_3d = torch.unsqueeze(mask, 2).repeat(1, 1, nC)
for i in range(nC):
    mask_3d[:, i * args.step:i * args.step + h, i] = mask_660
Phi = mask_3d.to(device)
# -------------------------- Optimization --------------------------#
x_rec = ADMM_Iter(meas.to(device), Phi.to(device), truth_tensor, args)
sio.savemat(args.results_dir + '/test.mat', {'img': x_rec.cpu().numpy()})
if os.path.exists('./Results/model_weights.pth'):
    os.remove('./Results/model_weights.pth')
