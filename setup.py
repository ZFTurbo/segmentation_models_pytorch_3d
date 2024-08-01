try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='segmentation_models_pytorch_3d',
    version='1.0.2',
    author='Roman Sol (ZFTurbo)',
    packages=[
        'segmentation_models_pytorch_3d',
        'segmentation_models_pytorch_3d/losses',
        'segmentation_models_pytorch_3d/datasets',
        'segmentation_models_pytorch_3d/base',
        'segmentation_models_pytorch_3d/encoders',
        'segmentation_models_pytorch_3d/decoders',
        'segmentation_models_pytorch_3d/utils',
        'segmentation_models_pytorch_3d/metrics',
        'segmentation_models_pytorch_3d/decoders/linknet',
        'segmentation_models_pytorch_3d/decoders/unet',
        'segmentation_models_pytorch_3d/decoders/deeplabv3',
        'segmentation_models_pytorch_3d/decoders/pan',
        'segmentation_models_pytorch_3d/decoders/pspnet',
        'segmentation_models_pytorch_3d/decoders/fpn',
        'segmentation_models_pytorch_3d/decoders/unetplusplus',
        'segmentation_models_pytorch_3d/decoders/manet',
    ],
    url='https://github.com/ZFTurbo/segmentation_models_pytorch_3d',
    description='Set of models for segmentation of 3D volumes using PyTorch.',
    long_description='3D variants of popular models for segmentation like FPN, Unet, Linknet etc using Pytorch module.'
                     'Automatic conversion of 2D imagenet weights to 3D variant',
    install_requires=[
        'torch',
        'torchvision>=0.5.0',
        "pretrainedmodels==0.7.4",
        "efficientnet-pytorch==0.7.1",
        "timm==0.9.7",
        "timm-3d==1.0.1",
        "tqdm",
        "pillow",
        "six",
    ],
)
