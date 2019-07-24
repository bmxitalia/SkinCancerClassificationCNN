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
from keras import Model
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as keras
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet169
from sklearn.utils import class_weight
from tensorflow.keras.metrics import categorical_accuracy


def plot_model_history(model_history, name):
	fig, axs = plt.subplots(1,2,figsize=(15,5))

	# grafico accuracy
	axs[0].plot(range(1, len(model_history.history['categorical_accuracy'])+1), model_history.history['categorical_accuracy'])
	axs[0].plot(range(1, len(model_history.history['val_categorical_accuracy']) + 1), model_history.history['val_categorical_accuracy'])
	axs[0].set_title('Model Accuracy')
	axs[0].set_ylabel('Accuracy')
	axs[0].set_xlabel('Epoch')
	axs[0].set_xticks(np.arange(1, len(model_history.history['categorical_accuracy']) + 1), len(model_history.history['categorical_accuracy'])/10)
	axs[0].legend(['train', 'val'], loc='best')

	# grafico loss

	axs[1].plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'])
	axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
	axs[1].set_title('Model Loss')
	axs[1].set_ylabel('Loss')
	axs[1].set_xlabel('Epoch')
	axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss'])/10)
	axs[1].legend(['train', 'val'], loc='best')

	plt.savefig("accuracy-loss-"+name+".png", dpi=300)


def model001():
	x_train = np.load('xtrain.npy', allow_pickle=True)
	x_test = np.load('xtest.npy', allow_pickle=True)
	x_val = np.load('xval.npy', allow_pickle=True)
	y_train = np.load('ytrain.npy', allow_pickle=True)
	y_test = np.load('ytest.npy', allow_pickle=True)
	y_val = np.load('yval.npy', allow_pickle=True)
	y_train_o = np.load('ytraino.npy', allow_pickle=True)
	y_test_o = np.load('ytesto.npy', allow_pickle=True)


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

	optimizer = Adam(lr=0.001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
	early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
	checkpoint = ModelCheckpoint('bestModel4.h5', monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
	history = model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping, checkpoint, reduce_lr])

	model.save('model4.h5')

	loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
	loss_v, accuracy_v = model.evaluate(x_val, y_val, verbose=1)
	print("Validation: accuracy = ", accuracy_v, " loss = ", loss_v)
	print("Test: accuracy = ", accuracy, " loss = ", loss)
	plot_model_history(history, "001")


def model0001():
	x_train = np.load('xtrain.npy', allow_pickle=True)
	x_test = np.load('xtest.npy', allow_pickle=True)
	x_val = np.load('xval.npy', allow_pickle=True)
	y_train = np.load('ytrain.npy', allow_pickle=True)
	y_test = np.load('ytest.npy', allow_pickle=True)
	y_val = np.load('yval.npy', allow_pickle=True)
	y_train_o = np.load('ytraino.npy', allow_pickle=True)
	y_test_o = np.load('ytesto.npy', allow_pickle=True)


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

	optimizer = Adam(lr=0.0001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
	early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
	checkpoint = ModelCheckpoint('bestModel4.h5', monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
	history = model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping, checkpoint, reduce_lr])

	model.save('model4.h5')

	loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
	loss_v, accuracy_v = model.evaluate(x_val, y_val, verbose=1)
	print("Validation: accuracy = ", accuracy_v, " loss = ", loss_v)
	print("Test: accuracy = ", accuracy, " loss = ", loss)
	plot_model_history(history, "0001")


def model00001():
	x_train = np.load('xtrain.npy', allow_pickle=True)
	x_test = np.load('xtest.npy', allow_pickle=True)
	x_val = np.load('xval.npy', allow_pickle=True)
	y_train = np.load('ytrain.npy', allow_pickle=True)
	y_test = np.load('ytest.npy', allow_pickle=True)
	y_val = np.load('yval.npy', allow_pickle=True)
	y_train_o = np.load('ytraino.npy', allow_pickle=True)
	y_test_o = np.load('ytesto.npy', allow_pickle=True)


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

	optimizer = Adam(lr=0.00001)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[categorical_accuracy])
	early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=15, verbose=1, mode='max', restore_best_weights=True)
	reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
	checkpoint = ModelCheckpoint('bestModel4.h5', monitor='val_categorical_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
	history = model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping, checkpoint, reduce_lr])

	model.save('model4.h5')

	loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
	loss_v, accuracy_v = model.evaluate(x_val, y_val, verbose=1)
	print("Validation: accuracy = ", accuracy_v, " loss = ", loss_v)
	print("Test: accuracy = ", accuracy, " loss = ", loss)
	plot_model_history(history, "00001")


if __name__ == "__main__":
	model001()
	model0001()
	model00001()