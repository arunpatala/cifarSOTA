# efficient_densenet_pytorch
A PyTorch implementation of DenseNets, optimized to save GPU memory.

## Recent updates
1. **Now works on PyTorch 0.3.x!**
1. `models/densenet_efficient_multi_gpu.py` is now depricated. `models/densenet_efficient.py` can handle both single and multi-GPU setups.

## Motivation
While DenseNets are fairly easy to implement in deep learning frameworks, most
implmementations (such as the [original](https://github.com/liuzhuang13/DenseNet)) tend to be memory-hungry.
In particular, the number of intermediate feature maps generated by batch normalization and concatenation operations
grows quadratically with network depth.
*It is worth emphasizing that this is not a property inherent to DenseNets, but rather to the implementation.*

This implementation uses a new strategy to reduce the memory consumption of DenseNets.
We assign all intermediate feature maps to two shared memory allocations,
which are utilized by every Batch Norm and concatenation operation.
Because the data in these allocations are temporary, we re-populate the outputs during back-propagation.
This adds 15-20% of time overhead for training, but **reduces feature map consumption from quadratic to linear.**

For more details, please see the [technical report](https://arxiv.org/pdf/1707.06990.pdf).

![Diagram of implementation](https://raw.github.com/gpleiss/efficient_densenet_pytorch/master/images/forward.png)

## Requirements
- PyTorch 0.3.x
- CUDA

## Usage

**In your existing project:**
There are two files in the `models` folder.
 - `models/densenet.py` is a "naive" implementation, based off the [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py) and
[project killer](https://github.com/felixgwu/img_classification_pk_pytorch/blob/master/models/densenet.py) implementations.
 - `models/densenet_efficient.py` is the new efficient implementation. (Code is still a little ugly. We're working on cleaning it up!)
Copy either one of those files into your project!

**Options:**
- All options are described in [the docstrings of the model files](https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet_efficient.py#L189)
- The depth is controlled by `block_config` option
- If you want to use the model for ImageNet, set `small_inputs=False`. For CIFAR or SVHN, set `small_inputs=True`.

**Running the demo:**

The only extra package you need to install is [python-fire](https://github.com/google/python-fire):
```sh
pip install fire
```

- single GPU:

```sh
CUDA_VISIBLE_DEVICES=0,1,2 python demo.py --efficient True --data <path_to_folder_with_cifar10> --save <path_to_save_dir>
```

Options:
- `--depth` (int) - depth of the network (number of convolution layers) (default 40)
- `--growth_rate` (int) - number of features added per DenseNet layer (default 12)
- `--n_epochs` (int) - number of epochs for training (default 300)
- `--batch_size` (int) - size of minibatch (default 256)
- `--seed` (int) - manually set the random seed (default None)

## Performance

A comparison of the two implementations (each is a DenseNet-BC with 100 layers, batch size 64, tested on a NVIDIA Pascal Titan-X):

| Implementation | Memory cosumption (GB/GPU) | Speed (sec/mini batch) |
|----------------|------------------------|------------------------|
| Naive          |  2.863  | 0.165                  |
| Efficient      |  1.605  | 0.207                  |
| Efficient (multi-GPU)      |  0.985  | -                  |


## Other efficient implementations
- [LuaTorch](https://github.com/liuzhuang13/DenseNet/tree/master/models) (by Gao Huang)
- [MxNet](https://github.com/taineleau/efficient_densenet_mxnet) (by Danlu Chen)
- [Caffe](https://github.com/Tongcheng/DN_CaffeScript) (by Tongcheng Li)

## Reference

```
@article{pleiss2017memory,
  title={Memory-Efficient Implementation of DenseNets},
  author={Pleiss, Geoff and Chen, Danlu and Huang, Gao and Li, Tongcheng and van der Maaten, Laurens and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:1707.06990},
  year={2017}
}
```