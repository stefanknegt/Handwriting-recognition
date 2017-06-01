import os, cv2, numpy as np
from keras.utils import np_utils
from keras import backend as K

if K.backend()=='tensorflow':
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')

def load_data_internal(folder):
    data_path = os.path.join('../data/Train/annotated_crops', folder)
    num_classes, input_shape, X_train, y_train, X_test, y_test = load_data(data_path)
    return num_classes, input_shape, X_train, y_train, X_test, y_test

def load_data(data_path):
    # Define data path
    num_classes = 0
    data_dir_list = os.listdir(data_path)
    for i in range (0,len(data_dir_list)):

        if data_dir_list[i] == ".DS_Store":
            num_classes = len(data_dir_list)-1 #DS_Store screwes the count up so -1 for MAC only
            break

    if num_classes==0:
        num_classes = len(data_dir_list)
    num_channel=1

    # Load data from dir above
    img_data_list=[]

    for dataset in data_dir_list:
        if dataset == ".DS_Store":
            continue
        img_list=os.path.join(data_path, dataset)
        print ('Loaded the images of dataset- '+'{}'.format(dataset))
        for img in os.listdir(img_list):
            if dataset == ".DS_Store" or img == ".DS_Store" or img_list == ".DS_Store":
                continue
            img_path = os.path.join(img_list, img)
            input_img=cv2.imread(img_path, flags=0)
            img_data_list.append(input_img)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    print('Input dimensions of all data: ' + str(img_data.shape))

    # Add ONE channel
    if num_channel == 1:
        if K.image_data_format() == 'channels_first':
            img_data = np.expand_dims(img_data, axis=1)
            print('Input dimensions for model: ' + str(img_data.shape))
        else:
            img_data = np.expand_dims(img_data, axis=4)
            print('Input dimensions for model: ' + str(img_data.shape))
    else:
        if K.image_data_format() == 'channels_first':
            img_data = np.rollaxis(img_data, 3, 1)
            print('Input dimensions for model: ' + str(img_data.shape))


    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,), dtype='int64')
    names = []

    i = 0
    j = 0

    for dataset in data_dir_list:
        names.append(dataset)
        if dataset == ".DS_Store" or img == ".DS_Store" or img_list == ".DS_Store":
            continue
        img_list = os.listdir(data_path + '/' + dataset)
        for i in range(len(img_list)):
            labels[i] = j
        j += 1

    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)


    #Shuffle the dataset -- DOESN'T WORK WITH VERY LARGE SETS MEMORY ERROR
    #x,y = shuffle(img_data,Y, random_state=2)

    # Scary unison shuffle!
    rng_state = np.random.get_state()
    np.random.shuffle(img_data)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

    # Split the dataset -- DOESN't WORK WITH VERY LARGE SETS MEMORY ERROR
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # Hopeful that dis does work
    # test_proportion of 3 means 1/5 so 20% test and 80% train
    def split(matrix, target, test_proportion):
        ratio = matrix.shape[0] / test_proportion
        X_train = matrix[ratio:, :]
        X_test = matrix[:ratio, :]
        Y_train = target[ratio:, :]
        Y_test = target[:ratio, :]
        return X_train, X_test, Y_train, Y_test
    X_train, X_test, y_train, y_test = split(img_data, Y, 5)

    # Defining the model
    input_shape = X_train[0].shape

    return num_classes, input_shape, X_train, y_train, X_test, y_test
