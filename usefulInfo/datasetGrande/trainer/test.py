import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.applications.inception_v3 import InceptionV3
from keras import callbacks
import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

LABEL_PATH = "dataset/HAM10000_metadata.csv"

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
    

def main():

    data_labels = pd.read_csv(LABEL_PATH)
    data_labels['image_path'] = data_labels.apply(lambda row: (row["image_id"] + ".jpg"), axis=1)
    target_labels = data_labels['dx']

    # get test set

    x_test = np.load('xtest.npy', allow_pickle=True)
    y_test = np.load('ytest.npy', allow_pickle=True)

    # load weight of trained model

    Model = modelInceptionAdam()

    Model.load_weights('modelInceptionAdam.h5')

    # take predictions on test set

    y_pred = Model.predict(x_test)

    # re convert one hot encoded predictions in true labels

    labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
    prediction = pd.DataFrame(y_pred, columns=labels_ohe_names.columns)
    prediction = list(prediction.idxmax(axis=1))

    # show performance of the model

    compute_metrics(y_test, prediction, y_pred)
    


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
    print(confusion_matrix(y_test, prediction, labels=['akiec','bcc','bkl','df','mel','nv','vasc']))
    # volendo c'e' anche la f beta
    # manca AUC per multiclass che non e' supportata da scikit

    # Plot non-normalized confusion matrix
    skplt.metrics.plot_confusion_matrix(y_test, prediction, labels=['akiec','bcc','bkl','df','mel','nv','vasc'], title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    skplt.metrics.plot_confusion_matrix(y_test, prediction, labels=['akiec','bcc','bkl','df','mel','nv','vasc'], normalize=True, title='Normalized confusion matrix')
    plt.show()


if __name__ == "__main__":
    main()