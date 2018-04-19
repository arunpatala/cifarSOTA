# ShakeDrop

This repository contains the implementation of the paper [ShakeDrop regularization](https://arxiv.org/abs/1802.02375).
The code is based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) and [PyramidNet](https://github.com/jhkim89/PyramidNet).

## Usage

1. Install [Torch](http://torch.ch) and [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) and put on all files of this repository into "fb.resnet.torch/models".
2. Change the learning rate schedule in the file train.lua: "decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0" to "decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0".
3. Train the network, by running main.lua as below:
To train additive PyramidNet-110 (alpha=270) on CIFAR-10 dataset:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -dataset cifar10 -nEpochs 300 -netType pyramidnet -batchSize 128 -LR 0.5 -shareGradInput true -nGPU 4 -nThreads 8
```
To train additive PyramidNet-110 (alpha=270) on CIFAR-100 dataset:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua -dataset cifar100 -nEpochs 300 -netType pyramidnet -batchSize 128 -LR 0.5 -shareGradInput true -nGPU 4 -nThreads 8
```

The "ShakeDrop.lua" is a implementation of pixel level ShakeDrop with memory efficiency.
In the paper, PyramidNet-110 (alpha=270) with pixel level ShakeDrop achieved 15.78% error on CIFAR-100.
