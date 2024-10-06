# Semantic Segmentation with zero shot

## Installation

2080 ti, ubuntu 22.04 lts
```sh
conda create -n ae-zero-shot python=3.9

# pytorch==2.0.1, cuda 11.8 or 11.4
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

----------------------------------------------------------------
# error message
 /home/user/anaconda3/envs/ae-zero-shot/lib/python3.9/site-packages/torch/nn/modules/conv.py:459: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggered internally at /opt/conda/conda-bld/pytorch_1682343997789/work/aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
   return F.conv2d(input, weight, bias, self.stride,
----------------------------------------------------------------
# solved
conda install cudatoolkit=11.4

# if numpy version>=2
pip uninstall numpy
pip install "numpy<2"

pip install tensorboard
pip install matplotlib
pip install scikit-learn
pip install torchsummary
pip install tqdm
pip install torcheval
pip install lpips scikit-image
```

---

4090, ubuntu 22.04 lts
```sh
# conda version 24.5.0, Anaconda3-2024.06-1-Linux-x86_64.sh
conda create -n ae-zero-shot python=3.9

# CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# if numpy version>=2
pip uninstall numpy
pip install "numpy<2"

pip install tensorboard
pip install matplotlib
pip install scikit-learn
pip install torchsummary
pip install tqdm
pip install torcheval
pip install lpips scikit-image
pip install seaborn
pip install timm==0.3.2 tensorboardX six

# libmr
sudo apt install gcc g++
pip install cython
pip installl libmr
```

## AutoEncoder Architecture

```sh
$ python train.py -a

# output
[cuda] latent space shape: torch.Size([1, 128, 12, 12]), output shape: torch.Size([1, 1, 200, 200])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 200, 200]             640
            Conv2d-2         [-1, 64, 100, 100]          36,928
              ReLU-3         [-1, 64, 100, 100]               0
         MaxPool2d-4           [-1, 64, 50, 50]               0
      EncoderLayer-5           [-1, 64, 50, 50]               0
            Conv2d-6          [-1, 128, 50, 50]          73,856
            Conv2d-7          [-1, 128, 25, 25]         147,584
              ReLU-8          [-1, 128, 25, 25]               0
         MaxPool2d-9          [-1, 128, 12, 12]               0
     EncoderLayer-10          [-1, 128, 12, 12]               0
          Encoder-11          [-1, 128, 12, 12]               0
  ConvTranspose2d-12          [-1, 128, 24, 24]         262,272
             ReLU-13          [-1, 128, 24, 24]               0
     DecoderLayer-14          [-1, 128, 24, 24]               0
  ConvTranspose2d-15          [-1, 128, 49, 49]         262,272
             ReLU-16          [-1, 128, 49, 49]               0
     DecoderLayer-17          [-1, 128, 49, 49]               0
  ConvTranspose2d-18         [-1, 64, 100, 100]         204,864
             ReLU-19         [-1, 64, 100, 100]               0
     DecoderLayer-20         [-1, 64, 100, 100]               0
  ConvTranspose2d-21          [-1, 1, 200, 200]           1,025
          Sigmoid-22          [-1, 1, 200, 200]               0
          Decoder-23          [-1, 1, 200, 200]               0
================================================================
Total params: 989,441
Trainable params: 989,441
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.15
Forward/backward pass size (MB): 60.11
Params size (MB): 3.77
Estimated Total Size (MB): 64.03
----------------------------------------------------------------
```

## Folder tree

```sh
├── data
│   ├── test_data
│   └── train_data
├── model
│   ├── ae.py
│   ├── __init__.py
├── saved
│   └── models
└── utils
    ├── loader.py
    ├── openmax.py
├── inference.ipynb
├── README.md
├── requirements.txt
├── test.py
├── train.py
```