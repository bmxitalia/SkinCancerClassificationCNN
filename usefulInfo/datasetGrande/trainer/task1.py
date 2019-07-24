import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.applications.inception_v3 import InceptionV3
from keras import callbacks
import numpy as np
import pandas as pd
import scikitplot as skplt
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

DATASET_PATH = 'dataset/images/'
LABEL_PATH = 'dataset/HAM10000_metadata.csv'


def model():
    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False,
                                 input_shape=(299, 299, 3))

    out = base_inception.output

    # first model
    out = GlobalAveragePooling2D()(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(2048, activation='relu')(out)
    predictions = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=predictions)

    # freezing of inception v3 feature extraction layers

    for layer in base_inception.layers:
        layer.trainable = False

    return model

    '''

    # second model

    predictions = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=predictions)
    model.compile(RMSprop(lr=.001, rho=0.9, decay=0.9, epsilon=0.1), loss='categorical_crossentropy',
                  metrics=['accuracy'])


    # third model
    out = GlobalMaxPooling2D()(out)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.5)(out)
    predictions = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=predictions)
    model.compile(Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])


    # only if we want to freeze layers
    for layer in base_inception.layers:
        layer.trainable = False

    model.summary()

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, verbose=1)

    '''


def main():

    # get datasets

    x_train = np.load('xtrain.npy')
    x_test = np.load('xtest.npy')
    x_val = np.load('xval.npy')
    y_train = np.load('ytrain.npy')
    y_test = np.load('ytest.npy')
    y_val = np.load('yval.npy')

    # get one hot encoded vectors of img labels

    y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
    y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()
    y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).as_matrix()

    # make the keras model

    Model = model()
    Model.compile(SGD(lr=.0007, momentum=0.9, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    ## Adding the callback for TensorBoard
    tensorboard = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    ## Adding callback for early stopping -> really insteresting
    early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max', restore_best_weights=True)

    # fit of the model on training data

    Model.fit(x_train, y_train_ohe, validation_data=(x_val, y_val_ohe), epochs=200, verbose=1, callbacks=[tensorboard, early_stopping], batch_size=32) #batch size a 1 applica SGD
    
    # save of the model to be tested

    Model.save('model.h5')

    #Model.load_weights('model.h5')
    #y_pred = Model.predict(x_test)
    #labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
    #prediction = pd.DataFrame(y_pred, columns=labels_ohe_names.columns)
    #prediction = list(prediction.idxmax(axis=1))
    #compute_metrics(y_test, prediction, y_pred)
    #print(Model.evaluate(x_test, y_test_ohe))
    


def compute_metrics(y_test, prediction, y_pred):
    skplt.metrics.plot_precision_recall(y_test, y_pred)
    skplt.metrics.plot_roc(y_test, y_pred)
    plt.show()
    print("f1 macro average: ", f1_score(y_test, prediction, average='macro'))
    print("f1 micro average: ", f1_score(y_test, prediction, average='micro'))
    print("f1 score: ", f1_score(y_test, prediction, average=None))
    print("recall macro average: ", recall_score(y_test, prediction, average='macro'))
    print("recall micro average: ", recall_score(y_test, prediction, average='micro'))
    print("recall score: ", recall_score(y_test, prediction, average=None))
    print("precision macro average: ", precision_score(y_test, prediction, average='macro'))
    print("precision micro average: ", precision_score(y_test, prediction, average='micro'))
    print("precision score: ", precision_score(y_test, prediction, average=None))
    # volendo c'e' anche la f beta
    # manca AUC per multiclass che non e' supportata da scikit
    # Plot non-normalized confusion matrix
    skplt.metrics.plot_confusion_matrix(y_test, prediction, labels=['akiec','bcc','bkl','df','mel','nv','vasc'], title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    skplt.metrics.plot_confusion_matrix(y_test, prediction, labels=['akiec','bcc','bkl','df','mel','nv','vasc'], normalize=True, title='Normalized confusion matrix')
    plt.show()


if __name__ == "__main__":
    main()