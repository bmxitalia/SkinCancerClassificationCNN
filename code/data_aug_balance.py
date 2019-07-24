import csv
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def read_csv(filename):
    '''
    Read csv and create a dictionary with interest values
    '''
    metadata = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                metadata.update({row[1]: row[2]})
    return metadata


def preparing_directory(metadata):
    '''
    Create a directory for each class of metadata file
    '''
    values = list(set([x for x in metadata.values()]))
    for name in values:
        if not os.path.exists(name):
            os.mkdir(name)
        else:
            for file_ in os.listdir(name):
                os.remove(name + '/' + file_)


def subdivide_image(metadata):
    '''
    Subdivide images of directory list in directories, one for each class
    '''
    directory_list = ['training_set']
    for directory in directory_list:
        for file_ in os.listdir(directory):
            filename = os.path.splitext(file_)[0]
            class_ = metadata[filename]
            shutil.copyfile(directory+"/"+file_, class_+"/"+file_)


def train_split(test_size = 0.20):
    '''
    Split the whole dataset HAM10000 in training and test set.
    '''

    # This directory must contains all images of HAM10000
    base_skin_dir = 'Images'
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
    skin_df = pd.read_csv('HAM10000_metadata.csv')

    skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
    skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
    skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

    # Split dataset in training and test set
    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(skin_df['path'], skin_df['cell_type_idx'], test_size=test_size, random_state=1234)
    
    directory_list = ['training_set', 'test_set']
    for directory in directory_list:
        if not os.path.exists(directory):
            print("Creating {} directory".format(directory))
            os.mkdir(directory)
            print("{} created".format(directory))
        else:
            print("{} is already present, removing all images inside it".format(directory))
            for file_ in os.listdir(directory):
                os.remove(directory + '/' + file_)
            print("Files in {} removed".format(directory))
    
    # Copy images of training and test set in training and set directory 
    for element in x_train_o:
        image = element[len(base_skin_dir)+1:]
        shutil.copyfile(base_skin_dir + '/' + image, directory_list[0] + '/' + image)
    for element in x_test_o:
        image = element[len(base_skin_dir)+1:]
        shutil.copyfile(base_skin_dir + '/' + image, directory_list[1] + '/' + image)


def data_augmentation():
    class_list = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'vasc']
    for item in class_list:
        aug_dir = 'aug_dir'
        os.mkdir(aug_dir)
        # create a dir within the base dir to store images of the same class
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)

        # Choose a class
        img_class = item

        # list all images in that directory
        img_list = os.listdir(img_class)

        # Copy images from the class train dir to the img_dir e.g. class 'mel'
        for fname in img_list:
            # source path to image
            src = os.path.join(img_class, fname)
            # destination path to image
            dst = os.path.join(img_dir, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)

        # point to a dir containing the images and not to the images themselves
        path = aug_dir
        save_path = img_class
        datagen = ImageDataGenerator(rotation_range=45, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
        batch_size = 32
        aug_datagen = datagen.flow_from_directory(path, save_to_dir=save_path, save_format='jpg', target_size=(450,600), batch_size=batch_size)
        num_aug_images_wanted = 1000 # 6000
        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((num_aug_images_wanted-num_files)/batch_size))

        # run the generator and create about 6000 augmented images
        for i in range(0,num_batches):

            imgs, labels = next(aug_datagen)
            
        # delete temporary directory with the raw image files
        shutil.rmtree('aug_dir')


def count_element_dir():
    directory_list = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    for dir_ in directory_list:
        print(dir_ + " : " + str(len(os.listdir(dir_))))


def create_metadata():
    metadata = {}
    class_list = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    for class_ in class_list:
        for file_ in os.listdir(class_):
            filename_tuple = os.path.splitext(file_)
            filename = filename_tuple[0] + filename_tuple[1]
            metadata[filename] = class_

    with open('DA_metadata.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_id', 'dx'])
        for key, value in metadata.items():
            writer.writerow([key, value])


def test_metadata(ham_metadata):
    metadata = {}
    for file_ in os.listdir('train_test'):
        filename_tuple = os.path.splitext(file_)
        filename = filename_tuple[0] + filename_tuple[1]
        metadata[filename] = ham_metadata[filename_tuple[0]]

    with open('test.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_id', 'dx'])
        for key, value in metadata.items():
            writer.writerow([key, value])


def aggregate_images():
    base_dir = 'all_images'
    if not os.path.exists(base_dir):
        print("Creating {} directory".format(base_dir))
        os.mkdir(base_dir)
        print("{} created".format(base_dir))
    else:
        print("{} is already present, removing all images inside it".format(base_dir))
        for file_ in os.listdir(base_dir):
            os.remove(base_dir + '/' + file_)
        print("Files in {} removed".format(base_dir))
    
    directory_list = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    print("Copying images of class directories inside {}".format(base_dir))
    for directory in directory_list:
        for file_ in os.listdir(directory):
            shutil.copyfile(directory + '/' + file_, base_dir + '/' + file_)
    print("Done")


if __name__ == "__main__":
    metadata = read_csv('HAM10000_metadata.csv')
    train_split()
    preparing_directory(metadata)
    subdivide_image(metadata)
    data_augmentation()
    count_element_dir()
    create_metadata()
    aggregate_images()
    test_metadata(metadata)
    