import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as keras
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.metrics import categorical_accuracy

def compute_metrics(y_test, prediction, y_pred):
	y_test = take_classes(y_test)
	prediction = take_classes(prediction)
	report = classification_report(y_test, prediction)
	print(report)

	skplt.metrics.plot_roc(y_test, y_pred)
	plt.savefig('roc.png')
	skplt.metrics.plot_precision_recall(y_test, y_pred)
	plt.savefig('roc1.png')
	skplt.metrics.plot_confusion_matrix(y_test, prediction, title='Confusion matrix, without normalization')
	plt.savefig('confusion.png')
	# Plot normalized confusion matrix
	skplt.metrics.plot_confusion_matrix(y_test, prediction, normalize=True, title='Normalized confusion matrix')
	plt.savefig('confusion-normalized.png')


def model1():
	x_train = np.load('xtrain.npy')
	x_test = np.load('xtest.npy')
	x_val = np.load('xval.npy')
	y_train = np.load('ytrain.npy')
	y_test = np.load('ytest.npy')
	y_val = np.load('yval.npy')
	y_train_o = np.load('ytraino.npy')
	y_test_o = np.load('ytesto.npy')


	# building model funziona bene ma overfitting dopo un po'

	input_shape = (75, 100, 3)
	num_classes = 7

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.10))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()

	model.load_weights('bestModel.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)


def model2():
	x_train = np.load('xtrain.npy')
	x_test = np.load('xtest.npy')
	x_val = np.load('xval.npy')
	y_train = np.load('ytrain.npy')
	y_test = np.load('ytest.npy')
	y_val = np.load('yval.npy')
	y_train_o = np.load('ytraino.npy')
	y_test_o = np.load('ytesto.npy')


	# building model funziona bene ma overfitting dopo un po'

	input_shape = (75, 100, 3)
	num_classes = 7

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.10))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.20))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()

	model.load_weights('bestModel2.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)


def model3():
	x_train = np.load('xtrain.npy')
	x_test = np.load('xtest.npy')
	x_val = np.load('xval.npy')
	y_train = np.load('ytrain.npy')
	y_test = np.load('ytest.npy')
	y_val = np.load('yval.npy')
	y_train_o = np.load('ytraino.npy')
	y_test_o = np.load('ytesto.npy')


	# building model funziona bene ma overfitting dopo un po'

	input_shape = (75, 100, 3)
	num_classes = 7

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.10))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.20))

	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.30))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()

	model.load_weights('bestModel.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)


def model4():
	x_train = np.load('xtrain.npy')
	x_test = np.load('xtest.npy')
	x_val = np.load('xval.npy')
	y_train = np.load('ytrain.npy')
	y_test = np.load('ytest.npy')
	y_val = np.load('yval.npy')
	y_train_o = np.load('ytraino.npy')
	y_test_o = np.load('ytesto.npy')


	# building model funziona bene ma overfitting dopo un po'

	input_shape = (75, 100, 3)
	num_classes = 7

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.10))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.20))

	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.30))

	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.40))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()

	model.load_weights('bestModel4.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)


def model5():
	x_train = np.load('xtrain.npy')
	x_test = np.load('xtest.npy')
	x_val = np.load('xval.npy')
	y_train = np.load('ytrain.npy')
	y_test = np.load('ytest.npy')
	y_val = np.load('yval.npy')
	y_train_o = np.load('ytraino.npy')
	y_test_o = np.load('ytesto.npy')


	# building model funziona bene ma overfitting dopo un po'

	input_shape = (75, 100, 3)
	num_classes = 7

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.10))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.20))

	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.30))

	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.40))

	model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='Same', input_shape=input_shape))
	model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='Same'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	model.add(Dropout(0.40))

	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.50))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()

	model.load_weights('bestModel.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)


def take_classes(y_test):
	classes_string = []
	for class_ in y_test:
		if class_ == 0:
			classes_string.append('akiec')
		if class_ == 1:
			classes_string.append('bcc')
		if class_ == 2:
			classes_string.append('bkl')
		if class_ == 3:
			classes_string.append('df')
		if class_ == 4:
			classes_string.append('nv')
		if class_ == 5:
			classes_string.append('mel')
		if class_ == 6:
			classes_string.append('vasc')
	return classes_string


if __name__ == "__main__":
	model1()
	model2()
	model3()
	model4()
	model5()