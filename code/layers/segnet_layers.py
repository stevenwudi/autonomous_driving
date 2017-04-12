from keras import backend as K
from keras.layers.core import Activation, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from layers.ourlayers import DePool2D

class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        # self.input = T.tensor4()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__, "poolsize":self.poolsize}

class AutoEncoder_Layers(object):
    """
    Encoding-Decoding Layers for SegNet Model (Segmentation problem)
    """

    def __init__(self, n_classes, w_decay, border_mode='valid'):
        self.n_classes = n_classes
        self.w_decay = w_decay
        self.kernel = 3
        self.filter_size = [64, 128, 256, 512]
        self.pad = 1
        self.pool_size = 2
        self.idx = self.getChannel_idx()
        self.border_mode = border_mode

    def getChannel_idx(self):
        """
        Keras DIM_ORDER
        :return: th or tf depending on the dim_ordering of Keras Backend
        """
        if K.image_dim_ordering() == 'th':
            return 1
        else:
            return 3

    def DeConvolution(self, layer, name='DePool2D', type=None):
        """
        Deconvolution Layer
        :param layer: Input Layer from the Model
        :param name: name for the layer of the network
        :param type: 'UnPool' or 'DePool' 
        :return: DeConvolution Layer for the output
        """

        if type=='UnPool2D':
            deconv2D = UnPooling2D(poolsize=(self.pool_size, self.pool_size))(layer)

        elif type=='DePool2D':
            deconv2D = DePool2D(pool2d_layer=Layer, size=(self.pool_size, self.pool_size), name=name)(layer)

        else:
            raise ValueError(type+' is not a valid DeConvolution Layer for this model. Use only: {Unpool2D, DePool2D}')

        return deconv2D

    def EncodingBlock(self, x, filter_size, block, num=1, max_pooling=True):
        """
        Encoding Block for SegNet
        :param x: 
        :param block: 
        :param num: 
        :param max_pooling: 
        :return: 
        """

        type = 'encoding'

        x = ZeroPadding2D(padding=(self.pad, self.pad), name='{}_block{}_zpad{}'.format(type,block,num))(x)
        x = Convolution2D(filter_size, self.kernel, self.kernel, W_regularizer=l2(self.w_decay),
                          b_regularizer=l2(self.w_decay), border_mode=self.border_mode, name='{}_block{}_conv{}'.format(type,block,num))(x)
        x = BatchNormalization(mode=0, axis=self.idx, name='{}_block{}_bnorm{}'.format(type,block,num))(x)
        x = Activation('relu', name='{}_block{}_act{}'.format(type,block,num))(x)

        if max_pooling:
            x = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), strides=(2, 2), name='{}_block{}_mpool{}'.format(type,block,num))(x)

        return x

    def DecodingBlock(self, x, filter_size, block, num=1):
        """
        Decoding Block for SegNet
        :param x: 
        :param block: 
        :param num: 
        :return: 
        """

        type = 'decoding'

        if block > 1:
            x = self.DeConvolution(x, '{}_block{}_deconv{}'.format(type, block, num), type='DePool2D') # 'UnPool2D or DePool2D'

        x = ZeroPadding2D(padding=(self.pad, self.pad), name='{}_block{}_zpad{}'.format(type, block, num))(x)
        x = Convolution2D(filter_size, self.kernel, self.kernel, W_regularizer=l2(self.w_decay),
                          b_regularizer=l2(self.w_decay), border_mode=self.border_mode,
                          name='{}_block{}_conv{}'.format(type, block, num))(x)
        x = BatchNormalization(mode=0, axis=self.idx, name='{}_block{}_bnorm{}'.format(type, block, num))(x)

        return x

    def EncodingLayers(self, input):
        """
        :return: Encoding Layers 
        """
        x = self.EncodingBlock(input, self.filter_size[0], block=1)
        x = self.EncodingBlock(x, self.filter_size[1], block=2)
        x = self.EncodingBlock(x, self.filter_size[2], block=3)
        x = self.EncodingBlock(x, self.filter_size[3], block=4, max_pooling=False)


        return x

    def DecodingLayers(self, input):
        """
        :return: Decoding Layers 
        """

        x = self.DecodingBlock(input, self.filter_size[3], block=1)
        x = self.DecodingBlock(x, self.filter_size[2], block=2)
        x = self.DecodingBlock(x, self.filter_size[1], block=3)
        x = self.DecodingBlock(x, self.filter_size[0], block=4)
        x = Convolution2D(self.n_classes, 1, 1, border_mode=self.border_mode, name='block5_LastConv2D')(x)

        return x