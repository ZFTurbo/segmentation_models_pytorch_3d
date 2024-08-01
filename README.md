# Segmentation Models Pytroch 3D

Python library with Neural Networks for Volume (3D) Segmentation based on PyTorch.

This library is based on famous [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) library for images. Most of the documentation can be used directly from there. 

## Installation

* Type 1: `pip install segmentation-models-pytorch-3d`
* Type 2: Copy `segmentation_models_pytorch_3d` folder from this repository in your project folder.

## Quick start

Segmentation model is just a PyTorch nn.Module, which can be created as easy as:

```python
import segmentation_models_pytorch_3d as smp
import torch

model = smp.Unet(
    encoder_name="efficientnet-b0", # choose encoder, e.g. resnet34
    in_channels=1,                  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)

# Shape of input (B, C, H, W, D). B - batch size, C - channels, H - height, W - width, D - depth
res = model(torch.randn(4, 1, 64, 64, 64)) 
```

## Models

### Architectures

 - Unet [[paper](https://arxiv.org/abs/1505.04597)] [[docs](https://smp.readthedocs.io/en/latest/models.html#unet)]
 - Unet++ [[paper](https://arxiv.org/pdf/1807.10165.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#id2)]
 - MAnet [[paper](https://ieeexplore.ieee.org/abstract/document/9201310)] [[docs](https://smp.readthedocs.io/en/latest/models.html#manet)]
 - Linknet [[paper](https://arxiv.org/abs/1707.03718)] [[docs](https://smp.readthedocs.io/en/latest/models.html#linknet)]
 - FPN [[paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)] [[docs](https://smp.readthedocs.io/en/latest/models.html#fpn)]
 - PSPNet [[paper](https://arxiv.org/abs/1612.01105)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pspnet)]
 - PAN [[paper](https://arxiv.org/abs/1805.10180)] [[docs](https://smp.readthedocs.io/en/latest/models.html#pan)]
 - DeepLabV3 [[paper](https://arxiv.org/abs/1706.05587)] [[docs](https://smp.readthedocs.io/en/latest/models.html#deeplabv3)]

### Encoders

The following is a list of supported encoders in the SMP. Select the appropriate family of encoders and click to expand the table and select a specific encoder and its pre-trained weights (`encoder_name` and `encoder_weights` parameters).

<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnet18                        |imagenet / ssl / swsl           |11M                             |
|resnet34                        |imagenet                        |21M                             |
|resnet50                        |imagenet / ssl / swsl           |23M                             |
|resnet101                       |imagenet                        |42M                             |
|resnet152                       |imagenet                        |58M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">ResNeXt</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|resnext50_32x4d                 |imagenet / ssl / swsl           |22M                             |
|resnext101_32x4d                |ssl / swsl                      |42M                             |
|resnext101_32x8d                |imagenet / instagram / ssl / swsl|86M                         |
|resnext101_32x16d               |instagram / ssl / swsl          |191M                            |
|resnext101_32x32d               |instagram                       |466M                            |
|resnext101_32x48d               |instagram                       |826M                            |

</div>
</details>

<details>
<summary style="margin-left: 25px;">SE-Net</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|senet154                        |imagenet                        |113M                            |
|se_resnet50                     |imagenet                        |26M                             |
|se_resnet101                    |imagenet                        |47M                             |
|se_resnet152                    |imagenet                        |64M                             |
|se_resnext50_32x4d              |imagenet                        |25M                             |
|se_resnext101_32x4d             |imagenet                        |46M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">DenseNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|densenet121                     |imagenet                        |6M                              |
|densenet169                     |imagenet                        |12M                             |
|densenet201                     |imagenet                        |18M                             |
|densenet161                     |imagenet                        |26M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">EfficientNet</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|efficientnet-b0                 |imagenet                        |4M                              |
|efficientnet-b1                 |imagenet                        |6M                              |
|efficientnet-b2                 |imagenet                        |7M                              |
|efficientnet-b3                 |imagenet                        |10M                             |
|efficientnet-b4                 |imagenet                        |17M                             |
|efficientnet-b5                 |imagenet                        |28M                             |
|efficientnet-b6                 |imagenet                        |40M                             |
|efficientnet-b7                 |imagenet                        |63M                             |
</div>
</details>

<details>
<summary style="margin-left: 25px;">DPN</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|dpn68                           |imagenet                        |11M                             |
|dpn68b                          |imagenet+5k                     |11M                             |
|dpn92                           |imagenet+5k                     |34M                             |
|dpn98                           |imagenet                        |58M                             |
|dpn107                          |imagenet+5k                     |84M                             |
|dpn131                          |imagenet                        |76M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|vgg11                           |imagenet                        |9M                              |
|vgg11_bn                        |imagenet                        |9M                              |
|vgg13                           |imagenet                        |9M                              |
|vgg13_bn                        |imagenet                        |9M                              |
|vgg16                           |imagenet                        |14M                             |
|vgg16_bn                        |imagenet                        |14M                             |
|vgg19                           |imagenet                        |20M                             |
|vgg19_bn                        |imagenet                        |20M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">Mix Vision Transformer</summary>
<div style="margin-left: 25px;">

Backbone from SegFormer pretrained on Imagenet! Can be used with other decoders from package, you can combine Mix Vision Transformer with Unet, FPN and others!

Limitations:  

   - encoder is **not** supported by Linknet, Unet++
   - encoder is supported by FPN only for encoder **depth = 5**

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|mit_b0                          |imagenet                        |3M                              |
|mit_b1                          |imagenet                        |13M                             |
|mit_b2                          |imagenet                        |24M                             |
|mit_b3                          |imagenet                        |44M                             |
|mit_b4                          |imagenet                        |60M                             |
|mit_b5                          |imagenet                        |81M                             |

</div>
</details>

<details>
<summary style="margin-left: 25px;">MobileOne</summary>
<div style="margin-left: 25px;">

Apple's "sub-one-ms" Backbone pretrained on Imagenet! Can be used with all decoders.

Note: In the official github repo the s0 variant has additional num_conv_branches, leading to more params than s1.

|Encoder                         |Weights                         |Params, M                       |
|--------------------------------|:------------------------------:|:------------------------------:|
|mobileone_s0                    |imagenet                        |4.6M                              |
|mobileone_s1                    |imagenet                        |4.0M                              |
|mobileone_s2                    |imagenet                        |6.5M                              |
|mobileone_s3                    |imagenet                        |8.8M                              |
|mobileone_s4                    |imagenet                        |13.6M                             |

</div>
</details>

### Timm 3D encoders

We now support encoders from [timm_3d](https://github.com/ZFTurbo/timm_3d) library. Full list available [here](https://github.com/ZFTurbo/timm_3d/blob/main/docs/models_list.md). To use them add `tu-` before encoder name.
Example:

```python
encoder_name = 'tu-maxvit_base_tf_224.in21k'
model = smp.Unet(
    encoder_name=encoder_name,
    encoder_weights=None,
    in_channels=3,
    classes=1,
)
```

## Notes for 3D version

### Input size

Recommended input size for backbones can be calculated as: `K = pow(N, 2/3)`. 
Where N - is size for input image for the same model in 2D variant.

For example for N = 224, K = 32. For N = 512, K = 64.

### Strides

Typical strides for 2D case is 2 for H and W. It applied `depth` times (in almost all cases 5 times). So input image reduced from (224, 224) to (7, 7) on final layers. For 3D case because of very massive input, it's sometimes useful to control strides for every dimension independently. For this you can use input variable `strides`, which default values is: `strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))`. Example:

Let's say you have input data of size: (224, 128, 12). You can use strides like that:
((2, 2, 2), (4, 2, 1), (2, 2, 2), (2, 2, 1), (1, 2, 3)). Output shape for these strides will be: (7, 4, 1)
```python
import segmentation_models_pytorch_3d as smp
import torch

model = smp.Unet(
    encoder_name="resnet50",        
    in_channels=1,                  
    strides=((2, 2, 2), (4, 2, 1), (2, 2, 2), (2, 2, 1), (1, 2, 3)),
    classes=3, 
)

res = model(torch.randn(4, 1, 224, 128, 12)) 
```

**Note**: Strides currently supported by `resnet`-family and `densenet` models with `Unet` decoder only.

### Related repositories

 * [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - original segmentation 2D repo
 * [segmentation_models_3D](https://github.com/ZFTurbo/segmentation_models_3D) - segmentation models in 3D for keras/tensorflow
 * [timm_3d](https://github.com/ZFTurbo/timm_3d) - classification models in 3D for pytorch 
 * [volumentations](https://github.com/ZFTurbo/volumentations) - 3D augmentations

## Citation

If you find this code useful, please cite it as:
```
@article{solovyev20223d,
  title={3D convolutional neural networks for stalled brain capillary detection},
  author={Solovyev, Roman and Kalinin, Alexandr A and Gabruseva, Tatiana},
  journal={Computers in Biology and Medicine},
  volume={141},
  pages={105089},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2021.105089}
}
```

## To Do List
* Support for strides for all encoders
