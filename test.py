# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

import torch

if __name__ == '__main__':
    import segmentation_models_pytorch_3d as smp

    if 1:
        print('Test Unet...')
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')

    if 1:
        print('Test FPN + mobileone_s0 + imagenet weights...')
        model = smp.FPN(
            encoder_name="mobileone_s0",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')


    if 1:
        print('Test UnetPlusPlus...')
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')

    if 1:
        print('Test MAnet...')
        model = smp.MAnet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')

    if 1:
        print('Test Linknet...')
        model = smp.Linknet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')

    if 1:
        print('Test PSPNet...')
        model = smp.PSPNet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')

    if 1:
        print('Test PAN...')
        model = smp.PAN(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 256, 256, 256))
        print(f'Unpooled shape: {o.shape}')

    if 1:
        print('Test DeepLabV3...')
        model = smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')

    if 0:
        print('Test DeepLabV3Plus...')
        # Doesn't work. Something with shapes. Need to fix later
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 128, 128))
        print(f'Unpooled shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.resnet import resnet_encoders

        print('Test all Resnet encoders + Non default strides...')
        for encoder_name in list(resnet_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=4,
                classes=1,
                strides=((1, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
            )
            o = model(torch.randn(2, 4, 10, 128, 128))
            print(f'Result shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.densenet import densenet_encoders

        print('Test all Densenet encoders...')
        for encoder_name in list(densenet_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            o = model(torch.randn(2, 3, 128, 128, 128))
            print(f'Result shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.efficientnet import efficient_net_encoders

        print('Test all EfficientNet encoders...')
        for encoder_name in list(efficient_net_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=4,
                classes=1,
            )
            o = model(torch.randn(2, 4, 128, 128, 128))
            print(f'Result shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.vgg import vgg_encoders

        print('Test all VGG encoders...')
        for encoder_name in list(vgg_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=4,
                classes=1,
            )
            o = model(torch.randn(2, 4, 128, 128, 128))
            print(f'Result shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.dpn import dpn_encoders

        print('Test all DPN encoders...')
        for encoder_name in list(dpn_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=4,
                classes=1,
            )
            o = model(torch.randn(2, 4, 128, 128, 128))
            print(f'Result shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.mix_transformer import mix_transformer_encoders

        print('Test all MixTransformer encoders...')
        for encoder_name in list(mix_transformer_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            o = model(torch.randn(2, 3, 128, 128, 128))
            print(f'Result shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.mobileone import mobileone_encoders

        print('Test all Mobileone encoders...')
        for encoder_name in list(mobileone_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=2,
            )
            o = model(torch.randn(4, 3, 64, 64, 64))
            print(f'Result shape: {o.shape}')

    if 1:
        from segmentation_models_pytorch_3d.encoders.densenet import densenet_encoders

        print('Test Densenet non-default strides ...')
        for encoder_name in list(densenet_encoders.keys())[:1]:
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                strides=((2, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            )
            o = model(torch.randn(2, 3, 32, 64, 128))
            print(f'Result shape: {o.shape}')

    if 0:
        # Doesn't work need to find a workaround with paddings in EffNet
        from segmentation_models_pytorch_3d.encoders.efficientnet import efficient_net_encoders

        print('Test EfficientNet non-default strides ...')
        for encoder_name in list(efficient_net_encoders.keys()):
            print('Go for {}'.format(encoder_name))
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                strides=((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            )
            o = model(torch.randn(2, 3, 32, 64, 128))
            print(f'Result shape: {o.shape}')

    if 1:
        encoder_name = 'tu-maxvit_base_tf_224.in21k'
        print('Test Timm 3d model: {}...'.format(encoder_name))
        print('Go for {}'.format(encoder_name))
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        o = model(torch.randn(2, 3, 128, 64, 64))
        print(f'Result shape: {o.shape}')