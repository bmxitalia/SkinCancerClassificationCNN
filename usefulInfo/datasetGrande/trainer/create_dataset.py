import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

LABEL_PATH = "dataset/HAM10000_metadata.csv"
DATASET_PATH = "dataset/images/"

def main():
    data_labels = pd.read_csv(LABEL_PATH)
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

    # save numpy arrays: this is good to not compute all the images before every train
    np.save('xtrain', x_train)
    np.save('ytrain', y_train)
    np.save('ytraine', y_train_ohe)
    np.save('xtest', x_test)
    np.save('ytest', y_test)
    np.save('yteste', y_test_ohe)
    np.save('xval', x_val)
    np.save('yval', y_val)
    np.save('yvale', y_val_ohe)
    

if __name__ == "__main__":
    main()