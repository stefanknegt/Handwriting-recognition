import numpy as np
import os

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from load_data import load_data
from split_dataset import threshold



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

num_epoch = 10

def main():
    if not os.path.exists('../data/Train/annotated_crops/128_over_9'):
        threshold(10)
    if not os.path.exists('../data/Train/annotated_crops/128_over_99'):
        threshold(100)
    num_classes, input_shape, X_train, y_train, X_test, y_test = load_data('128_over_99')
    train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)
    num_classes, input_shape, X_train, y_train, X_test, y_test = load_data('128_over_9')
    train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)
    num_classes, input_shape, X_train, y_train, X_test, y_test = load_data('128')
    train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)


# define baseline model
def baseline_model_MLP(num_classes, num_pixels):
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

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

def train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test):
    model = baseline_model_CNN(num_classes, input_shape)
    model.summary()
    # Fit the model
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=200, verbose=2)

    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, batch_size=100, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    print("Baseline Error: %.2f%%" % (100-score[1]*100))

    model.save('baseline'+str(num_classes)+'.h5')

    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(num_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss ('+str(num_classes)+')')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc ('+str(num_classes)+')')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    plt.show()
    #print plt.style.available # use bmh, classic,ggplot for big pictures

if __name__ == '__main__':
    main()