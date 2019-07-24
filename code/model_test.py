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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras import backend as keras
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.metrics import categorical_accuracy
from keras.applications.inception_v3 import InceptionV3

def compute_metrics_aug(y_test, prediction, y_pred):
	y_test = take_classes_aug(y_test)
	prediction = take_classes_aug(prediction)
	report = classification_report(y_test, prediction)
	print(report)
	
	skplt.metrics.plot_roc(y_test, y_pred)
	plt.savefig('roc.png')
	skplt.metrics.plot_precision_recall(y_test, y_pred, plot_micro=False)
	plt.savefig('precision-recall-curve.png')
	skplt.metrics.plot_precision_recall(y_test, y_pred, plot_micro=False)
	plt.legend(loc=3)
	plt.savefig('precision-recall-curve-legend.png')
	skplt.metrics.plot_confusion_matrix(y_test, prediction, title='Confusion matrix, without normalization')
	plt.savefig('confusion.png')
	# Plot normalized confusion matrix
	skplt.metrics.plot_confusion_matrix(y_test, prediction, normalize=True, title='Normalized confusion matrix')
	plt.savefig('confusion-normalized.png')


def compute_metrics(y_test, prediction, y_pred):
	y_test = take_classes(y_test)
	prediction = take_classes(prediction)
	report = classification_report(y_test, prediction)
	print(report)
	
	skplt.metrics.plot_roc(y_test, y_pred)
	plt.savefig('roc.png')
	skplt.metrics.plot_precision_recall(y_test, y_pred, plot_micro=False)
	plt.savefig('precision-recall-curve.png')
	skplt.metrics.plot_precision_recall(y_test, y_pred, plot_micro=False)
	plt.legend(loc=3)
	plt.savefig('precision-recall-curve-legend.png')
	skplt.metrics.plot_confusion_matrix(y_test, prediction, title='Confusion matrix, without normalization')
	plt.savefig('confusion.png')
	# Plot normalized confusion matrix
	skplt.metrics.plot_confusion_matrix(y_test, prediction, normalize=True, title='Normalized confusion matrix')
	plt.savefig('confusion-normalized.png')


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

def take_classes_aug(y_test):
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
			classes_string.append('mel')
		if class_ == 5:
			classes_string.append('nv')
		if class_ == 6:
			classes_string.append('vasc')
	return classes_string


def without_class_weight_model():
	x_train = np.load('xtrain.npy')
	x_test = np.load('xtest.npy')
	x_val = np.load('xval.npy')
	y_train = np.load('ytrain.npy')
	y_test = np.load('ytest.npy')
	y_val = np.load('yval.npy')
	y_train_o = np.load('ytraino.npy')
	y_test_o = np.load('ytesto.npy')

	# building model

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

	model.load_weights('without_class_weight_model.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)

	
def class_weight_model():
	x_train = np.load('xtrain.npy')
	x_test = np.load('xtest.npy')
	x_val = np.load('xval.npy')
	y_train = np.load('ytrain.npy')
	y_test = np.load('ytest.npy')
	y_val = np.load('yval.npy')
	y_train_o = np.load('ytraino.npy')
	y_test_o = np.load('ytesto.npy')

	# building model

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

	model.load_weights('class_weight_model.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)


def data_augmentation_model():
	x_train = np.load('xtraina.npy')
	x_test = np.load('xtesta.npy')
	x_val = np.load('xvala.npy')
	y_train = np.load('ytraina.npy')
	y_test = np.load('ytesta.npy')
	y_val = np.load('yvala.npy')
	y_train_o = np.load('ytrainoa.npy')
	y_test_o = np.load('ytestoa.npy')


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

	model.load_weights('bestModel.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics_aug(y_test_o, y_classes, y_pred)


def transfer_learning_model():
	x_train = np.load('xtraint.npy')
	x_test = np.load('xtestt.npy')
	x_val = np.load('xvalt.npy')
	y_train = np.load('ytraint.npy')
	y_test = np.load('ytestt.npy')
	y_val = np.load('yvalt.npy')
	y_train_o = np.load('ytrainot.npy')
	y_test_o = np.load('ytestot.npy')

	base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

	out = base_inception.output

	# first model
	out = GlobalAveragePooling2D()(out)
	out = Dense(2048, activation='relu')(out)
	out = Dense(2048, activation='relu')(out)
	out = Dense(7, activation='softmax')(out)
	model = Model(inputs=base_inception.input, outputs=out)
	for layer in base_inception.layers:
		layer.trainable = False

	model.load_weights('bestModel.h5')

	y_pred = model.predict(x_test)

	y_classes = [np.argmax(y, axis=None, out=None) for y in y_pred]

	compute_metrics(y_test_o, y_classes, y_pred)


if __name__ == "__main__":
	without_class_weight_model()
	class_weight_model()
	data_augmentation_model()
	transfer_learning_model()