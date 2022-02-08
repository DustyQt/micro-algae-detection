from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Add, BatchNormalization, Activation
from keras.activations import relu, softmax
class ResNet:
    def __init__(self):
        pass

    def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
        
        nb_filter1, nb_filter2, nb_filter3 = filters
        
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    
    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
        
        nb_filter1, nb_filter2, nb_filter3 = filters
        #Tensor flow is that tf
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
        else:
            bn_axis = 1
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
        shortcut = Conv2D(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    
                
        
