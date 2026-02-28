# SGDE: Self-supervised Geometry Degradation Estimation Framework for Coded Aperture Compressive Spectral Imaging

This repo is the implementation of paper "SGDE: Self-supervised Geometry Degradation Estimation Framework for Coded Aperture Compressive Spectral Imaging"
<img src="./figure/SGDE_Framework.png">

## Overview



## Directory Structure
During the capture process, to prevent the accumulation of disturbances, we recalibrated the mask after taking several shots; therefore, there are a total of three masks. 


```
SGDE/   
├── Data/                       # hyperspectral datasets (KAIST, TSA and our self-build)
│     ├── KAIST/                # KAIST is a simulated dataset, hence it has ground truth (GT)
│     ├── Our_1/                # Our self-build Scene 01-02
│     ├── Our_2/                # Our self-build Scene 03-05
│     ├── Our_3/                # Our self-build Scene 06
│     ├── TSA/                  # TSA Scene 01-05
│     └── Estimated TSA mask/   # The TSA Mask estimated by our code. 
├── figure                      # figure of README.md
├── modeal/                     # Model definition for CASSI reconstruction
├── Results/                    # Output directory for reconstructed spectral images
├── func.py                     # Utilities for forward model and pre/post-processing
├── main_KAIST.py               # 
├── main_Our.py                 # 
├── main_TSA.py                 #
├── optimization.py             # 
└── README.md                   # Project documentation and usage instructions
```
If the reconstruction results appear too dark or exhibit flickering, you can crop out the image borders.
```
data = sio.loadmat("Result/test.mat")
img = data['img'][15:-15, 15:-15, :]
```


## Getting Started
### 1 Clone the Repository
Begin by cloning the repository to your local machine:

```
git clone 
cd SGDE
```

### 2 prepare the environment
```
conda create -n sgde python=3.10
conda activate sgde
pip install -r requirements.txt
```

### 3 Run SGDE
```
cd LCTC_4FastLineScan
python main_TSA.py
```

### 4 Migrate the affine parameters into your code



## Acknowledgement
This repository is partly built on [LCTC](https://github.com/YurongChen1998/LCTC), [SelfDeblur](https://github.com/csdwren/SelfDeblur) and [BD_noise_robust_kernel_estimation](https://github.com/csleemooo/BD_noise_robust_kernel_estimation). We appreciate the authors for creating these brilliant works and sharing code with the community.

## Contact
For questions, please contact:

* 何宇桥 (Yuqiao He) 
* Hunan University
* Email: heyuqiao@hnu.edu.cn


## Citation
If you find our code useful, please star ⭐ this repository and consider citing:
