<p align="center"><img width="40%" src="http://i.imgur.com/EMM7Qn2.pngg" /></p>

# Image Classification Project Killer in PyTorch
This repo is designed for those who want to start their experiments two days before the deadline and kill the project in the last 6 hours. :new_moon_with_face:
Inspired by [fb.torch.resnet](https://github.com/facebook/fb.resnet.torch),
it provides fast experiment setup and attempts to maximize the number of projects killed within the given time.
Please feel free to submit issues or pull requests if you want to contribute.

## Usage
Both Python 2.7 and 3.5 are supported; however, it was mainly tested on Python 3.
Use `python main.py -h` to show all arguments.

### Training
Train a ResNet-56 on CIFAR-10 with data augmentation using GPU0:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --data cifar10+ --arch resnet --depth 56 --save save/cifar10+-resnet-56 --epochs 164
```
Train a ResNet-110 on CIFAR-100 without data augmentation using GPU0 and GPU2:
```sh
CUDA_VISIBLE_DEVICES=0,2 python main.py --data cifar100 --arch resnet --depth 110 --save save/cifar100-resnet-110 --epochs 164
```

See *scripts/cifar10.sh* and *scripts/cifar100.sh* for more training examples.
### Evaluation
```sh
python main.py --resume save/resnet-56/model_best.pth.tar --evaluate test --data cifar10+
```

### Adding your custom model
You can write your own model in a *.py* file and put it into *models* folder. All you need it to provide a `createModel(arg1, arg2, **kwarg)` function that returns the model which is an instance of *nn.Module*. Then you'll be able to use your model by setting `--arch your_model_name` (assuming that your model is in a the file *models/your_model_name*).

### Show Training & Validation Results
#### Python script
```sh
getbest.py save/* FOLDER_1 FOLDER_2
```
In short, this script reads the *scores.tsv* in the saving folders and display the best validation errors of them.

#### Using Tensorboard
```sh
tensorboard --logdir save --port PORT
```

## Features

### Experiment Setup & Logging
- Ask before overwriting existing experiments, and move the old one to /tmp instead of overwriting
- Saving training/validation loss, errors, and learning rate of each epoch to a TSV file
- Automatically copying all source code to saving directory to prevent accidental deleteion of codes. This is inspired by [SGAN code](https://github.com/xunhuang1995/SGAN/tree/master/mnist).
- [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) support using [tensorboard\_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
- One script to show all experiment results
- Display training time
- Holding out testing set and using validation set for hyperparameter tuning experiments
- GPU support
- Adding *save* & *data* folders to .gitignore to prevent commiting the datasets and trained models
- Result table
- Python 2.7 & 3.5 support


### Models (See *models* folder for details)
- [ ] AlexNet ([paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks))
- [ ] VGGNet ([paper](https://arxiv.org/abs/1409.1556))
- [ ] SqueezeNet ([paper](https://arxiv.org/abs/1602.07360)) ([code](https://github.com/DeepScale/SqueezeNet))
- [x] ResNet ([paper](https://arxiv.org/abs/1512.03385)) ([code](https://github.com/facebook/fb.resnet.torch))
- [x] ResNet with Stochastic Depth ([paper](https://arxiv.org/abs/1603.09382)) ([code](https://github.com/yueatsprograms/Stochastic_Depth))
- [ ] Pre-ResNet ([paper](https://arxiv.org/abs/1603.05027)) ([code](https://github.com/facebook/fb.resnet.torch))
- [ ] Pre-ResNet with Stochastic Depth
- [ ] Wide ResNet ([paper](https://arxiv.org/abs/1605.07146)) ([code](https://github.com/szagoruyko/wide-residual-networks))
- [ ] DenseNet ([paper](https://arxiv.org/abs/1608.06993)) ([code](https://github.com/liuzhuang13/DenseNet)) (Our implementation is buggy now, we encourage you to put [Andreas Veit's implementation](https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py) or [Brandon Amos's implementation](https://github.com/bamos/densenet.pytorch/blob/master/densenet.py) into *models* folder and add a `createModel` function to it)
- [ ] PyramidalNet ([paper](https://arxiv.org/abs/1610.02915))([code](https://github.com/jhkim89/PyramidNet))
- [ ] PyramidalNet with Separated Stochastic Depth ([paper](https://arxiv.org/abs/1612.01230))([code](https://github.com/AkTgWrNsKnKPP/PyramidNet_with_Stochastic_Depth))
- [ ] ResNeXt ([paper](https://arxiv.org/abs/1611.05431)) ([code](https://github.com/facebookresearch/ResNeXt))
- [ ] MSDNet ([paper](https://arxiv.org/abs/1703.09844)) ([code](https://github.com/gaohuang/MSDNet))
- [ ] Steerable CNN ([paper](https://arxiv.org/abs/1612.08498))

### Datasets
#### CIFAR
Last 5000 samples in the original training set is used for validation. Each pixel is in [0, 1]. Based on experiments results, normalizing the data to zero mean and unit standard deviation seems to be redundant.
- CIFAR-10
- CIFAR-10+ (Horizontal flip and random cropping with padding 4)
- CIFAR-100
- CIFAR-100+ (Horizontal flip and random cropping with padding 4)

### Todo List
- [ ] More learning rate decay strategies (currently only dropping at 1/2 and 3/4 of the epochs)
- [ ] CPU support
- [ ] SVHN-small (without extra training data)
- [ ] SVHN
- [ ] MNIST
- [ ] ImageNet
- [ ] Comparing tensorboard\_logger v.s. pycrayon
- [ ] Adding acknowledgement
- [ ] Custom models & criterions tutorial
- [ ] Custom train & test functions tutorial
- [ ] Custom datasets tutorial
- [ ] Custom initialization
- [ ] Adding an example project killing scenario
- [ ] Adding license
- [ ] Pretrained models
- [ ] Iteration mode (Counting iterations instead of epochs)
- [ ] Pep8 check

## Results
### Test Error Rate (in percentage) **with** validation set
The number of parameters are calculated based on CIFAR-10 model.
ResNets were training with 164 epochs (the same as the default setting in fb.resnet.torch) and DenseNets were trained 300 epochs.
Both are using batch\_size=64.

| Model                                   | Parameters | CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ |
|-----------------------------------------| -----------|----------|-----------|-----------|------------|
| ResNet-56                               | 0.86M      |          | 6.82      |           |            |
| ResNet-110                              | 1.73M      |          |           |           |            |
| ResNet-110 with Stochastic Depth        | 1.73M      |          | 5.25      |           | 24.2       |
| DenseNet-BC-100 (k=12)                  | 0.8M       |          | 5.34      |           |            |
| DenseNet-BC-190 (k=40)                  | 25.6M      |          |           |           |            |
| Your model                              |            |          |           |           |            |

### Top1 Testing Error Rate (in percentage)
Coming soon...

## File Descriptions
- *main.py*: main script to train or evaluate models
- *train.py*: training and evaluation part of the code
- *config*: storing configuration of datasets (and maybe other things in the future)
- *utils.pypy*: useful functions
- *getbest.py*: display the best validation error of each saving folder
- *dataloader.py*: defines *getDataloaders* function which is used to load datasets
- *models*: a folder storing all network models. Each script in it should contain a *createModel(\*\*kwargs)* function that takes the arguments and return a model (subclass of nn.Module) for training
- *scripts*: a folder storing example training commands in UNIX shell scripts

## References

## Acknowledgement
This code is based on the ImageNet training script provided in [PyTorch examples](https://github.com/pytorch/examples/blob/master/imagenet/main.py).

The author is not familiar with licensing. Please contact me there is there are any problems with it.
