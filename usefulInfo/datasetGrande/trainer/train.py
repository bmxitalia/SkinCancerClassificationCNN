import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.applications.inception_v3 import InceptionV3
from keras import callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


def modelSGD():  ## accuracy validation: 0.6693, lo stesso cambiando il learning rate a 0.001
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(299, 299, 3))

    out = base_inception.output

    # first model
    out = GlobalAveragePooling2D()(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=out)
    #model.compile(SGD(lr=.0007, momentum=0.9, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

    # freezing of inception v3 feature extraction layers
    
    for layer in base_inception.layers:
        layer.trainable = False

    return model


def modelInceptionSGD():  ## accuracy validation: 0.6693
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(299, 299, 3))

    out = base_inception.output

    # first model
    out = GlobalAveragePooling2D()(out)
    out = Dropout(rate=0.5)(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=out)
    model.compile(SGD(lr=.001, momentum=0.9, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # freezing of inception v3 feature extraction layers

    for layer in base_inception.layers:
        layer.trainable = False

    return model


def modelInceptionAdam():  ## accuracy validation: 0.6693
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(299, 299, 3))

    out = base_inception.output

    # first model
    out = GlobalAveragePooling2D()(out)
    out = Dropout(rate=0.5)(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=out)
    model.compile(Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

    # freezing of inception v3 feature extraction layers

    for layer in base_inception.layers:
        layer.trainable = False

    return model


def modelInceptionRMS():  ## accuracy validation: 0.6693
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(299, 299, 3))

    out = base_inception.output

    # first model
    out = GlobalAveragePooling2D()(out)
    out = Dropout(rate=0.5)(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=out)
    model.compile(RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # freezing of inception v3 feature extraction layers

    for layer in base_inception.layers:
        layer.trainable = False

    return model


def modelSGDSoftmax():  ## accuracy validation: 0.6693
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(299, 299, 3))

    out = base_inception.output

    # first model
    out = GlobalAveragePooling2D()(out)
    out = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=out)
    model.compile(SGD(lr=.001, momentum=0.9, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # freezing of inception v3 feature extraction layers

    for layer in base_inception.layers:
        layer.trainable = False

    return model


def modelSoftmax():
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(299, 299, 3))

    out = base_inception.output

    # first model
    out = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=out)
    model.compile(RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # freezing of inception v3 feature extraction layers

    for layer in base_inception.layers:
        layer.trainable = False

    return model


def main():

    # get datasets
    
    x_train = np.load('xtrain.npy', allow_pickle=True)
    x_val = np.load('xval.npy', allow_pickle=True)
    y_train = np.load('ytrain.npy', allow_pickle=True)
    y_val = np.load('yval.npy', allow_pickle=True)
    y_train_ohe = np.load('ytraine.npy', allow_pickle=True)
    y_val_ohe = np.load('yvale.npy', allow_pickle=True)

    # make the keras model

    Model = modelInceptionSGD()
    ## Adding the callback for TensorBoard
    tensorboard = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    ## Adding callback for early stopping -> really insteresting
    early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max', restore_best_weights=True)

    # set balance for unbalanced data

    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

    class_weights = {0: 4.37305053, 1: 2.78174603, 2: 1.30224782, 3: 12.3633157, 4: 1.2855309, 5: 0.21338772, 6: 10.11544012}

    # fit of the model on training data

    Model.fit(x_train, y_train_ohe, validation_data=(x_val, y_val_ohe), epochs=20, verbose=1, callbacks=[tensorboard, early_stopping], batch_size=32, class_weight=class_weights) #batch size a 1 applica SGD
    
    # save of the model to be tested

    Model.save('modelSGD001.h5')


if __name__ == "__main__":
    main()