import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.applications.inception_v3 import InceptionV3
from keras import callbacks
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

    # struttura rete ricercatori che comparano reti neurali
    out = GlobalAveragePooling2D()(out)
    out = Dense(2048, activation='relu')(out)
    out = Dense(2048, activation='relu')(out)
    predictions = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=predictions)

    for layer in base_inception.layers:
        layer.trainable = False
    # model.compile(SGD(lr=.0007, momentum=0.9, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    '''

    # struttura rete dei ricercatori dell'ospedale

    predictions = Dense(7, activation='softmax')(out)
    model = Model(inputs=base_inception.input, outputs=predictions)
    model.compile(RMSprop(lr=.001, rho=0.9, decay=0.9, epsilon=0.1), loss='categorical_crossentropy',
                  metrics=['accuracy'])


    # struttura rete codice trovato
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
    #with tf.device('/device:CPU:0'):
    # file_stream = file_io.FileIO(LABEL_PATH, mode='r')
    data_labels = pd.read_csv(LABEL_PATH)
    # aggiunta riga al pandas frame con il percorso dell'immagine
    data_labels['image_path'] = data_labels.apply(lambda row: (row["image_id"] + ".jpg"), axis=1)
    target_labels = data_labels['dx']

    # load dataset
    train_data = np.array([img_to_array(load_img(DATASET_PATH + img, target_size=(299, 299)))
                           for img in data_labels['image_path'].values.tolist()
                           ]).astype('float32')

    # create train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(train_data, target_labels,
                                                        test_size=0.3,
                                                        stratify=np.array(target_labels),
                                                        random_state=42)

    # create test and validation datasets
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
                                                    test_size=0.5,
                                                    stratify=np.array(y_test),
                                                    random_state=42)

    # get one hot encoded vectors of img labels

    y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
    y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()
    y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).as_matrix()

    np.save('xtrain', x_train)
    np.save('xtest', x_test)
    np.save('xval', x_val)
    np.save('ytrain', y_train)
    np.save('ytest', y_test)
    np.save('yval', y_val)

    '''
    # with tf.device('/device:GPU:0'):
    Model = model()
    Model.compile(SGD(lr=.0007, momentum=0.9, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    #Model.summary()
    ## Adding the callback for TensorBoard
    tensorboard = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    ## Adding callback for early stopping -> really insteresting
    early_stopping = callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max', restore_best_weights=True)
    Model.fit(x_train, y_train_ohe, validation_data=(x_val, y_val_ohe), epochs=200, verbose=1, callbacks=[tensorboard, early_stopping], batch_size=32) #batch size a 1 applica SGD
    #Model.load_weights('model.h5')
    #y_pred = Model.predict(x_test)
    #labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
    #prediction = pd.DataFrame(y_pred, columns=labels_ohe_names.columns)
    #prediction = list(prediction.idxmax(axis=1))
    #compute_metrics(y_test, prediction, y_pred)
    #print(Model.evaluate(x_test, y_test_ohe))
    Model.save('model.h5')
    '''

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