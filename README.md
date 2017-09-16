# Sensorimotor Object Recognition
Torch code for "Deep Affordance-grounded Sensorimotor Object Recognition" [link](http://openaccess.thecvf.com/content_cvpr_2017/papers/Thermos_Deep_Affordance-Grounded_Sensorimotor_CVPR_2017_paper.pdf)

What's in the repo so far:
- The baseline object appearance model.
- The GTM slow multi-level fusion model that fuses object appearance with the corresponding accumulated 3D flow magnitude.

Coming soon:
- GST late fusion model (convnet + LSTM).


<img src="http://sor3d.vcl.iti.gr/wp-content/uploads/2017/07/gtm3.png" width="800">

Slow multi-level fusion is the (d) model.

Bibtex:
```
@InProceedings{Thermos_2017_CVPR,
author = {Thermos, Spyridon and Papadopoulos, Georgios Th. and Daras, Petros and Potamianos, Gerasimos},
title = {Deep Affordance-Grounded Sensorimotor Object Recognition},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```



## Prerequisites
- [Torch](http://torch.ch/)
- CuDNN (5.1.10)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)



## Experiments

### SOR3D

The dataset and the data preprocessing are available [here](http://sor3d.vcl.iti.gr/)

The data are organised in train, validation and test set. We just use a simple convention: SubFolderName == ClassName. So, for example: if you have classes {bottle, knife}, bottle images go into the folder train/bottle and knife images go into train/knife.

Object sample:

<img src="http://sor3d.vcl.iti.gr/wp-content/uploads/2017/03/4.png" width="200">

Accumulated 3D flow magnitude (affordance) samples:

<img src="http://sor3d.vcl.iti.gr/wp-content/uploads/2017/07/magn.png" width="400">


### Pretrained model

We use [VGG ILSVRC-2014 16-layer](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) backed with loadcaffe (thanks [szagoruyko](https://github.com/szagoruyko)) as base model.


### Baseline

The training scripts have several options which can be listed by running the script with the flag --help

To train the baseline (appearance) model, simply run the train_baseline.lua script. By default 1-GPU VGG-16 with CuDNN backend.

`th train_baseline.lua -appearanceSamplesPath [path to object data/train/]`

To test the baseline model, run the test_baseline.lua script. Add the path to the directory that your trained model is saved.

`qlua test_baseline.lua -appearanceSamplesPath [path to object data/test/] -loadNet [path to saved model directory]`

we use 'xlua' in order to visualize the confusion matrix.


### Slow Multi-level Fusion (SML)

The training scripts have several options which can be listed by running the script with the flag --help

To train the SML model, simply run the train_sml.lua script. By default 1-GPU VGG-16 with CuDNN backend.

`th train_sml.lua -appearanceSamplesPath [path to object data/train/] -affordanceSamplesPath [path to hand data/train/]`

To test the SML model, run the test_sml.lua script. Add the path to the directory that your trained model is saved.

`qlua test_sml.lua -appearanceSamplesPath [path to object data/test/] -affordanceSamplesPath [path to hand data/train/] -loadNet [path to saved model directory]`

we use 'xlua' in order to visualize the confusion matrix.

