import os, cv2
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras import backend as K
import numpy as np

K.set_image_data_format('channels_first')

def load_data(folder):
    # Define data path
    data_path = os.path.join('../data/Train/annotated_crops',folder)
    data_dir_list = os.listdir(data_path)
    num_classes = len(data_dir_list)
    num_channel=1


    # Load data from dir above
    img_data_list=[]

    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset- '+'{}'.format(dataset))
        for img in img_list:
            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize=cv2.resize(input_img,(128,128))
            img_data_list.append(input_img_resize)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    print (img_data.shape)

    # Add ONE channel
    if num_channel == 1:
        if K.image_data_format() == 'channels_first':
            img_data = np.expand_dims(img_data, axis=1)
            print (img_data.shape)
        else:
            img_data = np.expand_dims(img_data, axis=4)
            print (img_data.shape)
    else:
        if K.image_data_format() == 'channels_first':
            img_data = np.rollaxis(img_data, 3, 1)
            print (img_data.shape)

    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,), dtype='int64')
    names = []

    i = 0
    j = 0

    for dataset in data_dir_list:
        names.append(dataset)
        img_list = os.listdir(data_path + '/' + dataset)
        for img in img_list:
            labels[i] = j
            i += 1
        j += 1

    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)

    #Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=2)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # Defining the model
    input_shape = img_data[0].shape

    return num_classes, input_shape, X_train, y_train, X_test, y_test