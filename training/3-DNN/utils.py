import h5py
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D, Attention
from tensorflow.keras.layers import ELU, BatchNormalization, Reshape, Concatenate, Dropout, Add, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn


def setup_tf():
    """
    Detects GPUs and (currently) sets automatic memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu,True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)
            
            
class SegmentGenerator(keras.utils.Sequence):
    def __init__(self, signals, labels, batch_size=128, shuffle=True, stride=200, window_size=1024, n_channels=20):
        super().__init__()
        self.signals = signals
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stride = stride
        self.window_size = window_size
        self.n_channels = n_channels

        key_array = []

        for i, array in enumerate(self.signals):
            n = (array.shape[0] - self.window_size)//self.stride
            for j in range(n):
                key_array.append([i, self.stride*j])

        self.key_array = np.asarray(key_array, dtype=np.uint32)

        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array)//self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index*self.batch_size, stop=(index+1)*self.batch_size)

        x, y = self.__data_generation__(keys)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        x = np.empty(shape=(self.batch_size, self.window_size, self.n_channels), dtype=np.float32)
        y = np.empty(shape=(self.batch_size, self.window_size))

        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            x[i, :, :] = self.signals[key[0]][key[1]:key[1]+self.window_size, :]
            y[i, :] = self.labels[key[0]][key[1]:key[1]+self.window_size]

        return x, y
    
    
class BiasedConv(Conv2D):
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
        super(Conv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=True,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)
    
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
    
        self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
        
        self.built = True
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
    # Check if the input_shape in call() is different from that in build().
    # If they are different, recreate the _convolution_op to avoid the stateful
    # behavior.
        call_input_shape = inputs.get_shape()
        outputs = inputs
    
        if self.data_format == 'channels_first':
            if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    
class AttentionPooling(object):
    def __init__(self,filters, channels=18):
        self.filters = filters
        self.channels = channels
        
    def __call__(self, inputs):
        query, value = inputs
        
        att_q = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', activation=None, use_bias=False)(query)
        att_k = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', activation=None, use_bias=False)(value)
        gate = BiasedConv(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='sigmoid',
                         kernel_initializer='zeros', bias_initializer='ones')(Add()([att_q, att_k]))
        att = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
                    padding='same', activation='sigmoid',
                    kernel_initializer='ones', bias_initializer='zeros')(gate)
        
        return AveragePooling2D(pool_size=(1, self.channels), padding='same')(Multiply()([att, value]))

def build_unet(window_size=4096, n_channels=18, n_filters=8):
    input_seq = Input(shape=(window_size, n_channels))

    x = Reshape(target_shape=(window_size, n_channels, 1))(input_seq)
    
    x = Conv2D(filters=n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl0 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl0)
    x = Conv2D(filters=2*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl1 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl1)
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl2 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl2)
    x = Conv2D(filters=4*n_filters, kernel_size=(7, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl3 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl3)
    x = Conv2D(filters=8*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl4 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl4)
    x = Conv2D(filters=8*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    lvl5 = x
    
    x = MaxPooling2D(pool_size=(1, 20), padding='same')(lvl5)
    x = Conv2D(filters=4*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(filters=4*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)
    
    out5 = Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)
    out5 = Reshape(target_shape=(window_size//1024,))(out5)
    
    up4 = UpSampling2D(size=(4, 1))(x)
    att4 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up4, lvl4])
    
    out4 = Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='sigmoid')(att4)
    out4 = Reshape(target_shape=(window_size//256,))(out4)
    
    x = Concatenate(axis=-1)([up4, att4])
    x = Conv2D(filters=4*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up3 = UpSampling2D(size=(4, 1))(x)
    att3 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up3, lvl3])
    
    out3 = Conv2D(filters=1, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='sigmoid')(att3)
    out3 = Reshape(target_shape=(window_size//64,))(out3)
    
    x = Concatenate(axis=-1)([up3, att3])
    x = Conv2D(filters=4*n_filters, kernel_size=(7, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up2 = UpSampling2D(size=(4, 1))(x)
    att2 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up2, lvl2])
    
    out2 = Conv2D(filters=1, kernel_size=(15, 1), strides=(1, 1), padding='same', activation='sigmoid')(att2)
    out2 = Reshape(target_shape=(window_size//16,))(out2)
    
    x = Concatenate(axis=-1)([up2, att2])
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    
    up1 = UpSampling2D(size=(4, 1))(x)
    att1 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up1, lvl1])
    
    out1 = Conv2D(filters=1, kernel_size=(15, 1), strides=(1, 1), padding='same', activation='sigmoid')(att1)
    out1 = Reshape(target_shape=(window_size//4,))(out1)
    
    x = Concatenate(axis=-1)([up1, att1])
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up0 = UpSampling2D(size=(4, 1))(x)
    att0 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up0, lvl0])
    x = Concatenate(axis=-1)([up0, att0])
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(filters=1, kernel_size=(15, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    output = Reshape(target_shape=(window_size,))(x)

    unet = Model(input_seq, output)
    unet_train = Model(input_seq, [output, out1, out2, out3, out4, out5])
    
    return unet, unet_train


def build_windowfree_unet(n_channels=18, n_filters=8):
    input_seq = Input(shape=(None, n_channels, 1))
    
    x = Conv2D(filters=n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(input_seq)
    x = BatchNormalization()(x)
    lvl0 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl0)
    x = Conv2D(filters=2*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl1 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl1)
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl2 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl2)
    x = Conv2D(filters=4*n_filters, kernel_size=(7, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl3 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl3)
    x = Conv2D(filters=8*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    lvl4 = ELU()(x)
    
    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl4)
    x = Conv2D(filters=8*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    lvl5 = x
    
    x = MaxPooling2D(pool_size=(1, 20), padding='same')(lvl5)
    x = Conv2D(filters=4*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(filters=4*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)
    
    up4 = UpSampling2D(size=(4, 1))(x)
    att4 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up4, lvl4])
    
    x = Concatenate(axis=-1)([up4, att4])
    x = Conv2D(filters=4*n_filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up3 = UpSampling2D(size=(4, 1))(x)
    att3 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up3, lvl3])
    
    x = Concatenate(axis=-1)([up3, att3])
    x = Conv2D(filters=4*n_filters, kernel_size=(7, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up2 = UpSampling2D(size=(4, 1))(x)
    att2 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up2, lvl2])
    
    x = Concatenate(axis=-1)([up2, att2])
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    
    up1 = UpSampling2D(size=(4, 1))(x)
    att1 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up1, lvl1])
    
    x = Concatenate(axis=-1)([up1, att1])
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up0 = UpSampling2D(size=(4, 1))(x)
    att0 = AttentionPooling(filters=4*n_filters, channels=n_channels)([up0, lvl0])
    x = Concatenate(axis=-1)([up0, att0])
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(filters=4*n_filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    output = Conv2D(filters=1, kernel_size=(15, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)
    
    unet = Model(input_seq, output)
    
    return unet

class SeizureState:
    def __init__(self, state):
        if state == 'seiz':
            self.state = 1
        elif state == 'bckg':
            self.state = 0
        else:
            raise ValueError('Invalid initial seizure state given')
    
    def print_state(self):
        if self.state:
            return 'seiz'
        else:
            return 'bckg'
    
    def change_state(self):
        self.state = (self.state+1)%2