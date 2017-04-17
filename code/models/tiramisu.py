from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input
from layers.tiramisu_layers import *


"""
Paper: https://arxiv.org/abs/1611.09326
Implementation based on the Theano / Lasagne code from the original paper
Adapted code from: https://github.com/SimJeg/FC-DenseNet
"""


def build_tiramisu(img_shape=(None, None, 3), n_classes=8, weight_decay=1e-4, nb_filter=48, compression=0, dropout=0.2, freeze_layers_from=None):

    # Parameters for the Tiramisu Network
    n_layers_block = [4] * 11  # Dense layers per dense block
    growth_rate = 12  # Growth rate of dense blocks, k in DenseNet paper
    compression = 1 - compression  # Compression factor applied in Transition Down (only in case of OOM problems)

    model = tiramisu(img_shape, n_classes, n_layers_block, growth_rate, weight_decay, nb_filter, compression, dropout, freeze_layers_from)

    print("   Total Layers: ", len(model.layers))

    return model


def tiramisu(img_shape, n_classes, n_layers_block, growth_rate, weight_decay, nb_filter, compression, dropout, freeze_layers_from):
    """
    Create a Keras Model to represent the Tiramisu Netwrk of Dense56 layers. This Network will be configured by the TiramisuLayers Class().
    :param img_shape: shape of the input image
    :param n_classes: number of classes of the segmentation, including background
    :param n_layers_block: layesr per each DenseBlock from upsampling and downsampling path. (IMPORTANT: Must be a even nunber)
    :param growth_rate: grow_rate
    :param weight_decay: weight_decay (L2 norm penalization)
    :param nb_filter: number of filters (kernels)
    :param compression: compression factor
    :param dropout: dropout (used over the conv2D layers)
    :param freeze_layers_from: (not implemented yet). Only works with freeze_layers_from = FALSE
    
    :return: Tiramisu Model for Keras
    """

    # Creta the TiramisuLayers object to create some global parameters (grow_rate, border_mode, weight_decay ...)
    tiramisu_layers = TiramisuLayers(img_shape, n_classes, growth_rate, weight_decay)


    skip_connection = list() # Skip Connections
    net = {}

    # Number of layers per block must be odd
    assert (len(n_layers_block) - 1) % 2 == 0

    # Transition index
    transition_index = int(np.floor(len(n_layers_block) / 2))

    # Layers per block: DownsamplingPath, TransitionUp/Down and UpsamplingPath
    down_layers_block = n_layers_block[:transition_index]
    transition_layers_block = n_layers_block[transition_index]
    up_layers_block = n_layers_block[transition_index + 1:]

    assert len(down_layers_block) == len(up_layers_block)

    # Ensure input shape can be handled by the network architecture (Theano or TensorFlow)
    if dim_ordering == 'th':
        input_rows = img_shape[1]
        input_cols = img_shape[2]
    else:
        input_rows = img_shape[0]
        input_cols = img_shape[1]

    num_transitions = len(down_layers_block)
    multiple = 2 ** num_transitions
    if input_rows is not None:
        assert (input_rows / (multiple)) % 2 == 0
    if input_cols is not None:
        assert (input_cols / (multiple)) % 2 == 0

    # Initial Convolution
    net['input'] = Input(shape=img_shape)
    x = Convolution2D(nb_filter, 3, 3,
                      init='he_uniform',
                      border_mode=tiramisu_layers.border_mode,
                      name="initial_conv2D",
                      W_regularizer=l2(tiramisu_layers.weight_decay))(net['input'])

    net['init_conv'] = x

    # Downsampling Path: Dense blocks + TransitionDown
    for block_idx, n_layers_block in enumerate(down_layers_block):
        # Dense Block
        x, nb_filter = tiramisu_layers.DenseBlock(x, nb_layers=n_layers_block, nb_filter=nb_filter,
                                                  dropout_rate=dropout,
                                                  stack_input=True,
                                                  block_id='down_db{}'.format(block_idx)
                                                  )

        feature_name = 'db_{}'.format(block_idx)
        net[feature_name] = x
        skip_connection.append(x)

        nb_filter = int(compression * nb_filter)

        # Transition Down
        x = tiramisu_layers.TransitionDown(x, nb_filter, dropout_rate=dropout, id='down_td{}'.format(block_idx))
        feature_name = 'td_{}'.format(block_idx)
        net[feature_name] = x

    # Reverse Skip Connection List
    skip_connection = skip_connection[::-1]

    # Last DenseBlock does not have a Transition Down and does not stack the input
    x, nb_filter = tiramisu_layers.DenseBlock(x, nb_layers=transition_layers_block, nb_filter=nb_filter,
                                              dropout_rate=dropout,
                                              stack_input=True,
                                              block_id='transition'
                                              )

    feature_name = 'db_{}'.format(transition_index)
    net[feature_name] = x

    # Upsampling Path: TransitionUp + DenseBlock
    keep_filters = growth_rate * transition_layers_block  # Number of filters for the first transposed convolution
    for block_idx, n_layers_block in enumerate(up_layers_block):
        skip = skip_connection[block_idx]

        # Transition Up
        x_up = tiramisu_layers.TransitionUp(x, skip, keep_filters, id='up_tu{}'.format(block_idx))

        feature_name = 'tu_{}'.format(block_idx)
        net[feature_name] = x_up
        keep_filters = growth_rate * n_layers_block

        # Dense Block
        x, _ = tiramisu_layers.DenseBlock(x_up, n_layers_block, nb_filter=0,
                                          dropout_rate=dropout,
                                          stack_input=True,
                                          block_id='up_db{}'.format(block_idx)
                                          )

        feature_name = 'up_db_{}'.format(block_idx)
        net[feature_name] = x

    # Add last TransitionUp + DenseBlock features and compute the scores class for each pixel using a 1x1 Conv2D
    net['output_features'] = x
    net['pixel_score'] = Convolution2D(n_classes, 1, 1,
                                       init='he_uniform',
                                       border_mode='same',
                                       W_regularizer=l2(weight_decay),
                                       b_regularizer=l2(weight_decay),
                                       name='class_score')(net['output_features'])

    # Softmax
    net['softmax'] = NdSoftmax(name='softmax')(net['pixel_score'])

    # Model
    model = Model(input=[net['input']], output=[net['softmax']])

    # Freeze some layers
    if freeze_layers_from is not None:
        raise NotImplementedError

    return model

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    print(' > Building Tiramisu')
    model = build_tiramisu(input_shape, n_classes=11, weight_decay=1e-4, dropout=0.4)
    print(' > Compiling Tiramisu')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.summary()
