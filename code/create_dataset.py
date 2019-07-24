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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from keras.applications.inception_v3 import InceptionV3
from sklearn.utils import class_weight


def normal_dataset():
	base_skin_dir = os.path.join('.', 'input')

	imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_skin_dir, '*.jpg'))}

	lesion_type_dict = {
		'nv': 'Melanocytic nevi',
		'mel': 'Melanoma',
		'bkl': 'Benign keratosis-like lesions',
		'bcc': 'Basal cell carcinoma',
		'akiec': 'Actinic keratoses',
		'vasc': 'Vascular lesions',
		'df': 'Dermatofibroma'
	}

	skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

	skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
	skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
	skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

	print(skin_df.head())

	skin_df.isnull().sum()

	skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))

	x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(skin_df['image'], skin_df['cell_type_idx'], test_size=0.20, random_state=1234)

	x_train = np.asarray(x_train_o.tolist())
	x_test = np.asarray(x_test_o.tolist())
	x_train_mean = np.mean(x_train)
	x_train_std = np.std(x_train)
	x_test_mean = np.mean(x_test)
	x_test_std = np.mean(x_test)
	x_train = (x_train - x_train_mean)/x_train_std
	x_test = (x_test - x_test_mean)/x_train_std

	y_train = to_categorical(y_train_o, num_classes = 7)
	y_test = to_categorical(y_test_o, num_classes = 7)

	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)

	x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
	x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
	x_val = x_val.reshape(x_val.shape[0], *(75, 100, 3))


	np.save('xtrain', x_train)
	np.save('xtest', x_test)
	np.save('xval', x_val)
	np.save('yval', y_val)
	np.save('ytrain', y_train)
	np.save('ytest', y_test)
	np.save('ytraino', y_train_o)
	np.save('ytesto', y_test_o)


def data_augmentation_dataset():
	train = pd.read_csv('train.csv')
	train = train.sample(frac=1).reset_index(drop=True) # to shuffle the trainig set
	test = pd.read_csv('test.csv')
	test = test.sample(frac=1).reset_index(drop=True)
	val = pd.read_csv('val.csv')
	val = val.sample(frac=1).reset_index(drop=True)

	train['cell_type_idx'] = pd.Categorical(train['dx']).codes
	test['cell_type_idx'] = pd.Categorical(test['dx']).codes
	val['cell_type_idx'] = pd.Categorical(val['dx']).codes

	train['image'] = train['image_id'].map(lambda x: np.asarray(Image.open("train_set/"+x).resize((100, 75))))
	test['image'] = test['image_id'].map(lambda x: np.asarray(Image.open("test_set/"+x).resize((100, 75))))
	val['image'] = val['image_id'].map(lambda x: np.asarray(Image.open("validation_set/"+x).resize((100, 75))))

	x_train_o = train['image']
	x_test_o = test['image']
	x_val_o = val['image']
	y_train_o = train['cell_type_idx']
	y_test_o = test['cell_type_idx']
	y_val_o = val['cell_type_idx']

	x_train = np.asarray(x_train_o.tolist())
	x_test = np.asarray(x_test_o.tolist())
	x_val = np.asarray(x_val_o.tolist())
	x_train_mean = np.mean(x_train)
	x_train_std = np.std(x_train)
	x_test_mean = np.mean(x_test)
	x_test_std = np.mean(x_test)
	x_val_mean = np.mean(x_val)
	x_val_std = np.std(x_val)
	x_train = (x_train - x_train_mean)/x_train_std
	x_test = (x_test - x_test_mean)/x_train_std
	x_val = (x_val - x_val_mean)/x_val_std

	y_train = to_categorical(y_train_o, num_classes = 7)
	y_test = to_categorical(y_test_o, num_classes = 7)
	y_val = to_categorical(y_val_o, num_classes = 7)

	x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
	x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
	x_val = x_val.reshape(x_val.shape[0], *(75, 100, 3))


	np.save('xtraina', x_train)
	np.save('xtesta', x_test)
	np.save('xvala', x_val)
	np.save('yvala', y_val)
	np.save('ytraina', y_train)
	np.save('ytesta', y_test)
	np.save('ytrainoa', y_train_o)
	np.save('ytestoa', y_test_o)


def transfer_learning_dataset():
	base_skin_dir = os.path.join('.', 'input')

	imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_skin_dir, '*.jpg'))}

	lesion_type_dict = {
		'nv': 'Melanocytic nevi',
		'mel': 'Melanoma',
		'bkl': 'Benign keratosis-like lesions',
		'bcc': 'Basal cell carcinoma',
		'akiec': 'Actinic keratoses',
		'vasc': 'Vascular lesions',
		'df': 'Dermatofibroma'
	}

	skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

	skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
	skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
	skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

	print(skin_df.head())

	skin_df.isnull().sum()

	skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((299, 299))))

	x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(skin_df['image'], skin_df['cell_type_idx'], test_size=0.20, random_state=1234)

	x_train = np.asarray(x_train_o.tolist())
	x_test = np.asarray(x_test_o.tolist())
	x_train_mean = np.mean(x_train)
	x_train_std = np.std(x_train)
	x_test_mean = np.mean(x_test)
	x_test_std = np.mean(x_test)
	x_train = (x_train - x_train_mean)/x_train_std
	x_test = (x_test - x_test_mean)/x_train_std

	y_train = to_categorical(y_train_o, num_classes = 7)
	y_test = to_categorical(y_test_o, num_classes = 7)

	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)

	x_train = x_train.reshape(x_train.shape[0], *(299, 299, 3))
	x_test = x_test.reshape(x_test.shape[0], *(299, 299, 3))
	x_val = x_val.reshape(x_val.shape[0], *(299, 299, 3))


	np.save('xtraint', x_train)
	np.save('xtestt', x_test)
	np.save('xvalt', x_val)
	np.save('yvalt', y_val)
	np.save('ytraint', y_train)
	np.save('ytestt', y_test)
	np.save('ytrainot', y_train_o)
	np.save('ytestot', y_test_o)


if __name__ == "__main__":
	normal_dataset()
	data_augmentation_dataset()
	transfer_learning_dataset()