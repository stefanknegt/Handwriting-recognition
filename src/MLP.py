import os, sys, cv2,  numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

PLOT = False
num_epoch = 50

def main(folder, fold):
    print('Training baseline MLP model')
    accuracy = np.zeros(fold)
    num_classes, input_shape, img_data, Y = load_shuffle_data_MLP(folder, verbose=False)
    for i in range(0, fold):
        X_train, y_train, X_test, y_test = k_fold(img_data, Y, fold, i)
        accuracy[i] = train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)
        del X_train, y_train, X_test, y_test
    mean = np.mean(accuracy)
    std = np.std(accuracy)
    print(accuracy)
    print('Mean accuracy: ' + str(mean))
    print('Standard deviation: ' + str(std))

def load_shuffle_data_MLP(folder, verbose):
    script_dir = os.path.dirname(__file__)
    test_path = os.path.join(script_dir, '../data/Train/annotated_crops')
    data_path = os.path.join(test_path, folder)
    data_dir_list = os.listdir(data_path)
    num_classes = 0

    for i in range(0, len(data_dir_list)):
        if data_dir_list[i] == ".DS_Store":
            num_classes = len(data_dir_list) - 1  # DS_Store screwes the count up so -1 for MAC only
            break
    if num_classes == 0:
        num_classes = len(data_dir_list)

    num_channel = 1

    # Load data from dir above
    img_data_list = []

    for dataset in data_dir_list:
        if dataset == ".DS_Store":
            continue
        img_list = os.path.join(data_path, dataset)
        if verbose:
            print ('Loaded the images of dataset- ' + '{}'.format(dataset))
        for img in os.listdir(img_list):
            if dataset == ".DS_Store" or img == ".DS_Store" or img_list == ".DS_Store":
                continue
            img_path = os.path.join(img_list, img)
            input_img = cv2.imread(img_path, flags=0)
            img_data_list.append(input_img)

    img_data = np.array(img_data_list)
    del img_data_list
    img_data = img_data.astype('float32')
    img_data /= 255
    print('Input dimensions of all data: ' + str(img_data.shape))
    img_data = img_data.reshape(img_data.shape[0], (img_data.shape[1]*img_data.shape[2]))
    print('Input dimensions of resized data: ' + str(img_data.shape))

    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,), dtype='int64')
    # names = []

    i = 0
    j = 0

    for dataset in data_dir_list:
        # names.append(dataset)
        if dataset == ".DS_Store" or img == ".DS_Store" or img_list == ".DS_Store":
            continue
        img_list = os.listdir(data_path + '/' + dataset)
        for k in range(len(img_list)):
            labels[i] = j
            i += 1
        j += 1

    del img_list
    del i
    del j

    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)
    del labels

    # Shuffle the dataset -- DOESN'T WORK WITH VERY LARGE SETS MEMORY ERROR
    # x,y = shuffle(img_data,Y, random_state=2)

    # Scary unison shuffle!
    rng_state = np.random.get_state()
    np.random.shuffle(img_data)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
    del rng_state

    input_shape = img_data[0].shape

    return num_classes, input_shape, img_data, Y

def k_fold(img_data, Y, fold, i):
    print('fold '+str(i)+' of '+str(fold)+' folds, splitting data!')
    ratio = img_data.shape[0] / fold
    X_train = np.append(img_data[0:i*ratio,:],img_data[ratio+i*ratio:img_data.shape[0],:]).reshape(img_data.shape[0]-ratio, img_data.shape[1])
    y_train = np.append(Y[0:i*ratio,:],Y[ratio+i*ratio:Y.shape[0],:]).reshape(Y.shape[0]-ratio, Y.shape[1])
    X_test = img_data[i*ratio:ratio+i*ratio, :]
    y_test = Y[i*ratio:ratio+i*ratio, :]
    return X_train, y_train, X_test, y_test

def MLP(num_classes, input_shape):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test):
    model = MLP(num_classes, input_shape)
    # Fit the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=100, verbose=2, callbacks=[early_stopping])

    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, batch_size=50, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    print("Baseline Error: %.2f%%" % (100-score[1]*100))

    i = 0
    while os.path.exists('MLP_' + str(X_train.shape[0]) + '_' + str(i) + '.h5'):
        i += 1
    model.save('MLP_' + str(X_train.shape[0]) + '_' + str(i) + '.h5')

    return score[1]

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))