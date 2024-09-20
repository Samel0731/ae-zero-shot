# Semantic Segmentation with zero shot

## Installation

2080 ti, ubuntu 22.04 lts
```
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
```

---

4090, ubuntu 22.04 lts
```
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
```

## AutoEncoder Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 100, 100]             640
              ReLU-2         [-1, 64, 100, 100]               0
         MaxPool2d-3           [-1, 64, 50, 50]               0
      EncoderLayer-4           [-1, 64, 50, 50]               0
            Conv2d-5          [-1, 128, 50, 50]          73,856
              ReLU-6          [-1, 128, 50, 50]               0
         MaxPool2d-7          [-1, 128, 25, 25]               0
      EncoderLayer-8          [-1, 128, 25, 25]               0
            Conv2d-9          [-1, 256, 25, 25]         295,168
             ReLU-10          [-1, 256, 25, 25]               0
        MaxPool2d-11          [-1, 256, 12, 12]               0
     EncoderLayer-12          [-1, 256, 12, 12]               0
           Conv2d-13          [-1, 512, 12, 12]       1,180,160
             ReLU-14          [-1, 512, 12, 12]               0
        MaxPool2d-15            [-1, 512, 6, 6]               0
     EncoderLayer-16            [-1, 512, 6, 6]               0
          Encoder-17            [-1, 512, 6, 6]               0
  ConvTranspose2d-18          [-1, 256, 12, 12]       2,097,408
             ReLU-19          [-1, 256, 12, 12]               0
     DecoderLayer-20          [-1, 256, 12, 12]               0
  ConvTranspose2d-21          [-1, 128, 24, 24]         524,416
             ReLU-22          [-1, 128, 24, 24]               0
     DecoderLayer-23          [-1, 128, 24, 24]               0
  ConvTranspose2d-24          [-1, 128, 49, 49]         262,272
             ReLU-25          [-1, 128, 49, 49]               0
     DecoderLayer-26          [-1, 128, 49, 49]               0
  ConvTranspose2d-27         [-1, 64, 100, 100]         204,864
             ReLU-28         [-1, 64, 100, 100]               0
     DecoderLayer-29         [-1, 64, 100, 100]               0
  ConvTranspose2d-30          [-1, 1, 200, 200]           1,025
          Sigmoid-31          [-1, 1, 200, 200]               0
          Decoder-32          [-1, 1, 200, 200]               0
================================================================
Total params: 4,639,809
Trainable params: 4,639,809
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.15
Forward/backward pass size (MB): 47.99
Params size (MB): 17.70
Estimated Total Size (MB): 65.84
----------------------------------------------------------------
```