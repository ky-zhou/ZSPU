# [ICME 2022] (Oral) ZSPU: "Zero-Shot" Point Cloud Upsampling

### News
Self-training code released. PU and PS xyz data released. PU mesh data released.

### Introduction

This repository is for our submitted [paper](https://arxiv.org/pdf/2106.13765.pdf) for ICME 2022. The code is modified from [PU-GAN](https://github.com/liruihui/PU-GAN), [3PU](https://github.com/yifita/3PU) and [PU-Net](https://github.com/yulequan/PU-Net). 

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under TF1.15 (lower version should also work) with CUDA 10.0 and Python 3.6 on Ubuntu 18.04.

For compiling TF operators, please check `tf_xxx_compile.sh` or `tf_xxx_compile_abi.sh` under each op subfolder in `code/tf_ops` folder. Note that you need to update `nvcc`, `cuda` and `tensoflow include library` if necessary. 

### Note
When running the code, if you have `undefined symbol: _ZTIN10tensorflow8OpKernelE` error, you need to compile the TF operators. If you have already added the `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` but still have ` cannot find -ltensorflow_framework` error. Please use 'locate tensorflow_framework
' to locate the tensorflow_framework library and make sure this path is in `$TF_LIB`.

### Usage

1. Compile the TF operators
   Follow the above information to compile the TF operators. 
   
2. Train the model:
    Test point clouds `x` are provided in folder `data/[dataset]`.
    Run:
   ```shell
   python main.py --phase train --data_file x --use_data y
   ```
   Here, `x` and `y` are the filename (without extension) for the point cloud and the index for dataset (0 for Data PU, 1 for Data Princeton, 2 for KITTI), respectively.

3. The prediction will be conducted automatically after the model is trained.
   You will see the input and output results in the folder `data/[dataset]/input` and `data/[dataset]/output`.
   
4. We will release all mesh files after the review period.

### Evaluation code
We provide the code to calculate the uniform metric in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) and  [PU-Net](https://github.com/yulequan/PU-Net) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   ./evaluation ps/m0.off ps/m0.xyz
   ```
in where, the second argument is the mesh, and the third one is the predicted points.

Then, use `evaluate.py` to gain the quantitative results:
    ```
    python evaluate.py --pred ./evaluation_code/[dataset]/ --gt ./evaluation_code/[dataset]/gt/
    ```

## Citation
Please consider citing this paper with the following bibtex if you are interested in this work:

    @inproceedings{zhou2022zspu,
        author    = {Zhou, Kaiyue and Dong, Ming and Arslanturk, Suzan},
        title     = {"Zero-Shot" Point Cloud Upsampling},
        booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
        year      = {2022}
    }
    

