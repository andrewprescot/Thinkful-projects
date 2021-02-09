#!/usr/bin/env python
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

###############################################################################
'''

BASIC VGG Blocks (1,2, or 3 CNN, 1 dense, 1 output)

Simple CNN wirtten to simply evaluate effect of increasing number of CNN blocks

We can also specify 

'''

def CNN_BASIC_VGG11(input_shape  = (64, 64, 3), 
                    n_blocks = 1,
                    nClasses = 2, 
                    lr = 0.001, 
                    momentum = 0.9,
                    kernel_initializer = 'he_uniform',
                    padding = 'same',
                    activation_cnn = 'relu',
                    activation_dense = 'relu',
                    activation_out = 'softmax',
                    dropout = False,
                    batchnorm = False,
                    decay = 0,
                    nesterov=False,
                    optimizer = 'sgd'):
    
    model = Sequential()
    
    # LAYER 1 
    model.add(Conv2D(64, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001), input_shape=input_shape))
    
    if batchnorm:
        model.add(BatchNormalization())
        
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if dropout:
        model.add(Dropout(0.2))
        
    # LAYER 2          
    model.add(Conv2D(128, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))
        
    if batchnorm:
        model.add(BatchNormalization())
        
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if dropout:
        model.add(Dropout(0.2))
    
    # LAYER 3
        
    model.add(Conv2D(256, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(256, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))
        
    if batchnorm:
        model.add(BatchNormalization())
        
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if dropout:
        model.add(Dropout(0.2))

    # LAYER 4
    
    model.add(Conv2D(512, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(512, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))

    if batchnorm:
        model.add(BatchNormalization())
        
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if dropout:
        model.add(Dropout(0.2))
        
    # LAYER 5
    
    model.add(Conv2D(512, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))
    model.add(Conv2D(512, (3, 3), padding=padding, activation=activation_cnn, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))

    if batchnorm:
        model.add(BatchNormalization())
        
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if dropout:
        model.add(Dropout(0.2))


    model.add(Flatten())
    # First fully-connected layer
    model.add(Dense(4096, activation=activation_dense, kernel_regularizer=l2(0.001)))
    model.add(Dense(4096, activation=activation_dense, kernel_regularizer=l2(0.001)))
    model.add(Dense(1000, activation=activation_dense, kernel_regularizer=l2(0.001)))
    if dropout:
        model.add(Dropout(0.2))
 
    # Output
    model.add(Dense(nClasses, activation=activation_out))
    
    # Compile model
    if optimizer == 'sgd':
        opt = SGD(lr = lr, momentum = momentum, decay=decay, nesterov =nesterov)
    elif optimizer == 'adam':
        opt = Adam(lr = lr)
    
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model
