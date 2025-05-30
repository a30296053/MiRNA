import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Max Blur Pooling
class MaxBlurPooling1D(layers.Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.avg_kernel = None
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        elif self.kernel_size == 5:
            bk = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = tf.keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size,), strides=(1,),
                       padding='SAME', pooling_type='MAX', data_format='NWC')
        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]

# Blur Pooling with 1D
class BlurPool1D(layers.Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        elif self.kernel_size == 5:
            bk = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = tf.keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), input_shape[2]

# SE Block: Channel-wise attention to recalibrate feature responses
def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    excitation = tf.keras.layers.Dense(units=int(num_filters/ratio))(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, num_filters])(excitation)
    scale = inputs * excitation
    return scale

# Different vision of models
def org_model(input_length=66, dout = 0.4):
    fils = 320
    ksize = 12
    inputs = tf.keras.Input(shape=(int(input_length),))
    x = tf.keras.layers.Embedding(input_dim=5, output_dim=5, input_length=input_length)(inputs)
    
    x = tf.keras.layers.Conv1D(filters=fils, kernel_size=ksize, activation='relu')(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dout)(x)
   
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(dout)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def model_v1(input_length=66, dout = 0.2):

    fils = 512
    lstm_fil = 32
    ksize = 11
    dout = 0.2
    
    inputs = tf.keras.Input(shape=(int(input_length),))  
    x = tf.keras.layers.Embedding(input_dim=5, output_dim=5, input_length=input_length)(inputs)
    
    x = tf.keras.layers.Conv1D(fils, ksize, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Dense(lstm_fil)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

def model_v2(input_length=66, dout = 0.2):

    fils = 512
    lstm_fil = 32
    ksize = 11
    dout = 0.2
    ratio = 4
    
    inputs = tf.keras.Input(shape=(int(input_length),))    
    x = tf.keras.layers.Embedding(input_dim=5, output_dim=5, input_length=input_length)(inputs)
    
    x = tf.keras.layers.Conv1D(fils, ksize, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Dropout(dout)(x)

    x = SE_Block(x, fils, ratio)
    
#    x = tf.keras.layers.Conv1D(fils, 1, strides=1, padding="same")(x)
#    x = tf.keras.layers.BatchNormalization()(x)
#    x = tf.keras.layers.Activation('relu')(x)    
#    x = tf.keras.layers.Dropout(dout)(x)

    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Dense(lstm_fil)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

# ATTbiLSTM
def model_v3(input_length=66, dout = 0.2):

    fils = 512
    lstm_fil = 32
    ksize = 11
    dout = 0.2
    ratio = 4
    
    inputs = tf.keras.Input(shape=(int(input_length),))   
    x = tf.keras.layers.Embedding(input_dim=5, output_dim=5, input_length=input_length)(inputs)
    
    x = tf.keras.layers.Conv1D(fils, ksize, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Dropout(dout)(x)

    x = SE_Block(x, fils, ratio)
    
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)

    x = SE_Block(x, lstm_fil*2, ratio)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Dense(lstm_fil)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

# ATTbiLSTM-MMP
def model_v4(input_length=66, dout = 0.2):

    fils = 512
    lstm_fil = 32
    ksize = 11
    
    ratio = 4
    
    inputs = tf.keras.Input(shape=(int(input_length),))   
    x = tf.keras.layers.Embedding(input_dim=5, output_dim=5, input_length=input_length)(inputs)
    
    x = tf.keras.layers.Conv1D(fils, ksize, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Dropout(dout)(x)

    x = SE_Block(x, fils, ratio)
    
    x1 = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x2 = tf.keras.layers.MaxPooling1D(pool_size=3)(x)
    x3 = tf.keras.layers.MaxPooling1D(pool_size=6)(x)
    x = tf.keras.layers.Concatenate(axis=1)([x1,x2,x3])
    
    x = tf.keras.layers.Dropout(dout)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)

    x = SE_Block(x, lstm_fil*2, ratio)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Dense(lstm_fil)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

def model_v5(input_length=66, dout = 0.2):

    fils = 512
    lstm_fil = 32
    ksize = 11
    dout = 0.2
    
    inputs = tf.keras.Input(shape=(int(input_length),))  
    x = tf.keras.layers.Embedding(input_dim=5, output_dim=5, input_length=input_length)(inputs)
    
    x = tf.keras.layers.Conv1D(fils, ksize, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Dense(lstm_fil)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

# ATTbiLSTM-MMP with different different parameters in Multi-kernel MaxPooling
def model_v6(input_length=66, dout = 0.2):

    fils = 512
    lstm_fil = 32
    ksize = 11
    
    ratio = 4
    
    inputs = tf.keras.Input(shape=(int(input_length),))   
    x = tf.keras.layers.Embedding(input_dim=5, output_dim=5, input_length=input_length)(inputs)
    
    x = tf.keras.layers.Conv1D(fils, ksize, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Dropout(dout)(x)

    x = SE_Block(x, fils, ratio)
    
    x1 = tf.keras.layers.MaxPooling1D(pool_size=1)(x)
    x2 = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x3 = tf.keras.layers.MaxPooling1D(pool_size=4)(x)

    x = tf.keras.layers.Concatenate(axis=1)([x1,x2,x3])
    
    x = tf.keras.layers.Dropout(dout)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)

    x = SE_Block(x, lstm_fil*2, ratio)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_fil))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dout)(x)
    
    x = tf.keras.layers.Dense(lstm_fil)(x)
    x = tf.keras.layers.Dropout(dout)(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model

model = model_v6()
model.summary()