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
pip install lpips scikit-image
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
pip install lpips scikit-image
```

## AutoEncoder Architecture

```
$ python train.py -a
```