from keras.layers import merge
from keras.layers.core import Activation, Dropout
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from layers.deconv import Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from layers.ourlayers import *

# Initializations for the Deconv2D
from initializations.initializations import bilinear_init

# from lasagne.layers import (
#     NonlinearityLayer, Conv2DLayer, DropoutLayer, Pool2DLayer, ConcatLayer, Deconv2DLayer,
#     DimshuffleLayer, ReshapeLayer, get_output, BatchNormLayer)
#
# from lasagne.nonlinearities import linear, softmax
# from lasagne.init import HeUniform

class TiramisuLayers():

    def __init__(self, img_shape, n_classes, growth_rate, weight_decay=1e-4):
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.growth_rate = growth_rate
        self.weight_decay = weight_decay
        self.border_mode = 'same'
        self.getChannel_idx()

    def getChannel_idx(self):
        """
        Keras DIM_ORDER
        :return: th or tf depending on the dim_ordering of Keras Backend
        """
        if K.image_dim_ordering() == 'th':
            self.bn_axis = 1
            self.concat_axis = 1
        else:
            self.bn_axis = -1
            self.concat_axis = -1


    def DenseBlock(self, x, nb_layers, nb_filter, dropout_rate=None, stack_input=True, block_id=""):
        """
        DenseBlock part of Tiramisu
        :param x: 
        :param nb_layers: 
        :param nb_filter: 
        :param dropout_rate: 
        :param stack_input: 
        :param block_id: 
        :return: output and number of layers
        """

        list_feat = []
        if stack_input:
            list_feat.append(x)

        for i in range(nb_layers):
            x = self.ConvolutionPart(x, nb_filter=self.growth_rate, dropout_rate=dropout_rate, dense_id='{}_l{}'.format(block_id, i))
            list_feat.append(x)

            if len(list_feat) > 1:
                x = merge(list_feat, mode='concat', concat_axis=self.concat_axis, name='{}_m{}'.format(block_id, i))

            nb_filter += self.growth_rate

        return x, nb_filter

    def TransitionDown(self, x, nb_filter, dropout_rate=None, td_id=""):
        """
        TransitionDown part of Tiramisu
        :param x: 
        :param nb_filter: 
        :param dropout_rate: 
        :param td_id: 
        :return: output
        """

        x = BatchNormalization(mode=0,
                               axis=self.bn_axis,
                               gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay),
                               name='{}_bn'.format(td_id))(x)

        x = Activation('relu', name='{}_relu'.format(td_id))(x)

        x = Convolution2D(nb_filter, 1, 1,
                          init='he_uniform',
                          border_mode=self.border_mode,
                          W_regularizer=l2(self.weight_decay),
                          b_regularizer=l2(self.weight_decay),
                          name='{}_conv'.format(td_id))(x)
        if dropout_rate:
            x = Dropout(dropout_rate, name='{}_drop'.format(td_id))(x)

        x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(td_id))(x)

        return x

    def TransitionUp(self, x, skip_connection, keep_filters, id=''):
        """
        TransitionUp of Tiramisu
        :param x: 
        :param skip_connection: 
        :param keep_filters: 
        :param id: 
        :return: output
        """

        deconv = Deconvolution2D(keep_filters, 3, 3, x._keras_shape,
                                 border_mode=self.border_mode,
                                 subsample=(2, 2),
                                 W_regularizer=l2(self.weight_decay),
                                 b_regularizer=l2(self.weight_decay),
                                 init=bilinear_init,
                                 name='{}_deconv'.format(id))(x)

        deconv = ZeroPadding2D({'bottom_pad': 1, 'right_pad': 1})(deconv)

        return merge([deconv, skip_connection], mode='concat', concat_axis=self.concat_axis, name='{}_merge'.format(id))

    def ConvolutionPart(self, x, nb_filter, dropout_rate=None, id=''):
        """
        ConvolutionFactory
        :param x: 
        :param nb_filter: 
        :param dropout_rate: 
        :param dense_id: 
        :return: output
        """

        x = BatchNormalization(mode=0,
                               axis=self.bn_axis,
                               gamma_regularizer=l2(self.weight_decay),
                               beta_regularizer=l2(self.weight_decay),
                               name='{}_bn'.format(id))(x)

        x = Activation('relu', name='{}_relu'.format(id))(x)

        x = Convolution2D(nb_filter, 3, 3,
                          init='he_uniform',
                          border_mode=self.border_mode,
                          W_regularizer=l2(self.weight_decay),
                          b_regularizer=l2(self.weight_decay),
                          name='{}_conv'.format(id))(x)
        if dropout_rate:
            x = Dropout(dropout_rate, name='{}_drop'.format(id))(x)

        return x

# def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
#     """
#     Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
#     """
#
#     l = NonlinearityLayer(BatchNormLayer(inputs))
#     l = Conv2DLayer(l, n_filters, filter_size, pad='same', W=HeUniform(gain='relu'), nonlinearity=linear,
#                     flip_filters=False)
#     if dropout_p != 0.0:
#         l = DropoutLayer(l, dropout_p)
#     return l
#
#
# def TransitionDown(inputs, n_filters, dropout_p=0.2):
#     """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
#
#     l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
#     l = Pool2DLayer(l, 2, mode='max')
#
#     return l
#     # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
#     # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution
#
#
# def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
#     """
#     Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """
#
#     # Upsample
#     l = ConcatLayer(block_to_upsample)
#     l = Deconv2DLayer(l, n_filters_keep, filter_size=3, stride=2,
#                       crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
#     # Concatenate with skip connection
#     l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])
#
#     return l
#     # Note : we also tried Subpixel Deconvolution without seeing any improvements.
#     # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution
#
#
# def SoftmaxLayer(inputs, n_classes):
#     """
#     Performs 1x1 convolution followed by softmax nonlinearity
#     The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
#     """
#
#     l = Conv2DLayer(inputs, n_classes, filter_size=1, nonlinearity=linear, W=HeUniform(gain='relu'), pad='same',
#                     flip_filters=False, stride=1)
#
#     # We perform the softmax nonlinearity in 2 steps :
#     #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
#     #     2. Apply softmax
#
#     l = DimshuffleLayer(l, (0, 2, 3, 1))
#     batch_size, n_rows, n_cols, _ = get_output(l).shape
#     l = ReshapeLayer(l, (batch_size * n_rows * n_cols, n_classes))
#     l = NonlinearityLayer(l, softmax)
#
#     return l
#
#     # Note : we also tried to apply deep supervision using intermediate outputs at lower resolutions but didn't see
#     # any improvements. Our guess is that FC-DenseNet naturally permits this multiscale approach