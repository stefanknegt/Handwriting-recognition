import sys, os, cv2
import numpy as np
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from load_data import load_data_internal, load_data_external
if K.backend()=='tensorflow':
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')
PLOT = False
num_epoch = 10

def main(folder, fold):
    if fold is 0 or fold is 1:
        print('No clear fold given, run once with 80% Train, 20% Test')
        num_classes, input_shape, X_train, y_train, X_test, y_test = load_data_internal(folder)
        acc = train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test, 1)
    else:
        print('Training Baseline_model with '+str(fold)+' folds')
        accuracy = np.zeros(fold)
        num_classes, input_shape, img_data, Y = load_and_shuffle_internal(folder)
        for i in range(0, fold):
            X_train, y_train, X_test, y_test = k_fold(img_data, Y, fold, i)
            accuracy[i] = train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test, fold)
        mean = np.mean(accuracy)
        std = np.std(accuracy)
        print(accuracy)
        print('Mean accuracy: '+str(mean))
        print('Standard deviation: '+str(std))

def load_and_shuffle_internal(folder):
    data_path = os.path.join('../data/Train/annotated_crops', folder)
    num_classes, input_shape, img_data, Y = load_and_shuffle(data_path)
    return num_classes, input_shape, img_data, Y

def load_and_shuffle_external(folder):
    data_path = os.path.join('E:/Documenten/Studie/Master/HWR', folder)
    num_classes, input_shape, img_data, Y = load_and_shuffle(data_path)
    return num_classes, input_shape, img_data, Y

def load_and_shuffle(data_path):
    # Define data path
    data_dir_list = os.listdir(data_path)
    num_classes = len(data_dir_list)
    num_channel = 1

    invalid_ims = [0] * num_classes
    # Load data from dir above
    img_data_list = []
    j = 0
    for dataset in data_dir_list:
        img_list = os.path.join(data_path, dataset)
        print ('Loaded the images of dataset- ' + '{}'.format(dataset))
        for img in os.listdir(img_list):
            img_path = os.path.join(img_list, img)
            input_img = cv2.imread(img_path, flags=0)
            try:
                img_data_list.append(input_img)
            except cv2.error:
                invalid_ims[j] += 1
        j += 1

    img_data = np.array(img_data_list)
    if not '_bin_' in data_path:
        img_data = img_data.astype('float32')
        img_data /= 255

    # Add ONE channel
    if num_channel == 1:
        if K.image_data_format() == 'channels_first':
            img_data = np.expand_dims(img_data, axis=1)
            #print (img_data.shape)
        else:
            img_data = np.expand_dims(img_data, axis=4)
            #print (img_data.shape)
    else:
        if K.image_data_format() == 'channels_first':
            img_data = np.rollaxis(img_data, 3, 1)
            #print (img_data.shape)

    # Label creation
    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,), dtype='int64')
    names = []

    i = 0
    j = 0

    for dataset in data_dir_list:
        names.append(dataset)
        img_list = os.listdir(data_path + '/' + dataset)
        for i in range(len(img_list) - invalid_ims[j]):
            labels[i] = j
        j += 1

    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)

    # Scary unison shuffle!
    rng_state = np.random.get_state()
    np.random.shuffle(img_data)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

    input_shape = img_data[0].shape

    return num_classes, input_shape, img_data, Y

def k_fold(img_data, Y, fold, i):
    ratio = img_data.shape[0] / fold
    X_train = np.append(img_data[0:i*ratio,:],img_data[ratio+i*ratio:img_data.shape[0],:]).reshape(img_data.shape[0]-ratio, img_data.shape[1], img_data.shape[2], img_data.shape[3])
    y_train = np.append(Y[0:i*ratio,:],Y[ratio+i*ratio:Y.shape[0],:]).reshape(Y.shape[0]-ratio, Y.shape[1])
    X_test = img_data[i*ratio:ratio+i*ratio, :]
    y_test = Y[i*ratio:ratio+i*ratio, :]
    return X_train, y_train, X_test, y_test

# Define baseline CNN model
def baseline_model_CNN(num_classes, input_shape):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test, fold):
    model = baseline_model_CNN(num_classes, input_shape)
    print('Model Summary: ')
    model.summary()

    print('Start training...')
    # Fit the model
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=100, verbose=2)

    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, batch_size=50, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    print("Baseline Error: %.2f%%" % (100 - score[1] * 100))

    i = 0
    while os.path.exists(str(fold)+'_fold_'+str(X_train.shape[0])+'_'+str(i)+'.h5'):
        i+=1
    model.save(str(fold)+'_fold_'+str(X_train.shape[0])+'_'+str(i)+'.h5')

    if PLOT:
        import matplotlib.pyplot as plt
        # visualizing losses and accuracy
        train_loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        train_acc = hist.history['acc']
        val_acc = hist.history['val_acc']
        xc = range(num_epoch)

        plt.figure(1, figsize=(7, 5))
        plt.plot(xc, train_loss)
        plt.plot(xc, val_loss)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('train_loss vs val_loss (' + str(X_train.shape[0]) + ')')
        plt.grid(True)
        plt.legend(['train', 'val'])
        # print plt.style.available # use bmh, classic,ggplot for big pictures
        plt.style.use(['classic'])

        plt.figure(2, figsize=(7, 5))
        plt.plot(xc, train_acc)
        plt.plot(xc, val_acc)
        plt.xlabel('num of Epochs')
        plt.ylabel('accuracy')
        plt.title('train_acc vs val_acc (' + str(X_train.shape[0]) + ')')
        plt.grid(True)
        plt.legend(['train', 'val'], loc=4)
        plt.show()
        # print plt.style.available # use bmh, classic,ggplot for big pictures

    return score[1]

if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        print("Please give as argument 1, the folder, 128_over_9 is an example")
        print("Please give as argument 2, an int for the number of folds, 10 is used usually")
        print("So a correct call would be: python model_k_fold 128_over_9 10")
