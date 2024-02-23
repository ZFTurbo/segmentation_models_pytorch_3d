# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

import torch

def convert_2d_weights_to_3d(state_dict, verbose=False):
    layers = list(state_dict.keys())
    for layer in layers:
        if (
            'conv' in layer
            or 'downsample' in layer
            or '_se_expand' in layer
            or '_se_reduce' in layer
            or 'patch_embed' in layer
            or 'attn.sr.weight' in layer
        ):
            if len(state_dict[layer].shape) == 4:
                shape_init = state_dict[layer].shape
                state_dict[layer] = torch.stack([state_dict[layer]]*state_dict[layer].shape[-1], dim=-1)
                state_dict[layer] /= state_dict[layer].shape[-1]
                if verbose:
                    print("Convert layer weights: {}. Shape: {} -> {}".format(layer, shape_init, state_dict[layer].shape))
    return state_dict