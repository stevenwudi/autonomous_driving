# imports
# import keras.backend as K
# from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from keras.layers.core import Reshape, Permute
from layers.segnet_layers import *

dim_ordering = K.image_dim_ordering()

"""
SegNET Model.
Paper: http://arxiv.org/pdf/1511.00561v2.pdf
Adapted Code: From: https://github.com/imlab-uiip/keras-segnet
"""


def build_segnet(img_shape=(None, None, 3), n_classes=11, w_decay=0.,
               freeze_layers_from=None, path_weights=None, basic=True):

    # Regularization warning
    if w_decay > 0.:
        print ("Regularizing the weights: " + str(w_decay))

    # Get base model
    if basic:
        model = SegNet_basic(input_shape=img_shape, n_classes=n_classes, w_decay=w_decay)
    else:
        # model = SegNet_vgg(input_shape=img_shape, n_classes=n_classes, w_decay=w_decay)
        raise NotImplementedError

    print ("   Total Layers: ", len(model.layers))

    # if load_pretrained: # TODO: From Caffe
    #     # Rename last layer to not load pretrained weights
    #     model.layers[-1].name += '_new'
    #     # model.load_weights('weights/SSD300.hdf5', by_name=True)
    #     model.load_weights('weights/segnet.hdf5', by_name=True)

    return model

def SegNet_basic(input_shape, n_classes, w_decay):
    """
    Create the SegNET Model for the semgmentation approach. Encoding and Decoding Layers.
    The SegNET uses a AutoEncoder Class() in order to create the Encoding and Decoding part for the network.
    :param input_shape: input shape of the images
    :param n_classes: number of classes of the dataset
    :param w_decay: weight_decay
    
    :return: SegNet Basic Model 
    """

    # K.set_image_dim_ordering('tf')
    inputs = Input(input_shape)

    layers = AutoEncoder_Layers(n_classes=n_classes, w_decay=w_decay, border_mode='valid')

    encoding_layers = layers.EncodingLayers(inputs)
    decoding_layers = layers.DecodingLayers(encoding_layers)

    x = Activation('relu')(decoding_layers)
    # x = CropLayer2D(inputs, name='score')(x)
    softmax_segnet = NdSoftmax()(x)
    # softmax_segnet = Activation('softmax')(x)

    model = Model(input=inputs, output=softmax_segnet)

    return model

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    print (' > Building')
    model = build_segnet(input_shape, 11)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()