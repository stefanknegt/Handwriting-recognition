import os, sys, numpy as np
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from load_data import load_data_internal
from keras.callbacks import EarlyStopping

PLOT = False
num_epoch = 50

def main(folder):
    print('Training baseline model')
    num_classes, input_shape, X_train, y_train, X_test, y_test = load_data_internal(folder, verbose = False)
    train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)


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

# CNN1
def CNN1(num_classes, input_shape):
    # create model
    model = Sequential()

    model.add(Conv2D(128, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#CNN2
def CNN2(num_classes, input_shape):
    # create model
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

#CNN3
def CNN2_no_dropout(num_classes, input_shape):
    # create model
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    #model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    #model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    #model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    #model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test):
    model = baseline_model_CNN(num_classes, input_shape)
    model.summary()
    # Fit the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=100, verbose=2, callbacks=[early_stopping])

    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, batch_size=50, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    print("Baseline Error: %.2f%%" % (100-score[1]*100))

    i = 0
    while os.path.exists('baseline_'+str(X_train.shape[0])+'_'+str(i)+'.h5'):
        i+=1
    model.save('baseline_'+str(X_train.shape[0])+'_'+str(i)+'.h5')

    if PLOT:
        import matplotlib.pyplot as plt
        # visualizing losses and accuracy
        train_loss=hist.history['loss']
        val_loss=hist.history['val_loss']
        train_acc=hist.history['acc']
        val_acc=hist.history['val_acc']
        xc=range(num_epoch)

        f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(7,5))
        ax1.plot(xc,train_loss)
        ax1.plot(xc,val_loss)
        f.xlabel('num of Epochs')
        ax1.ylabel('loss')
        ax1.title('train_loss vs val_loss ('+str(X_train.shape[0])+')')
        ax1.grid(True)
        ax1.legend(['train','val'])
        #print plt.style.available # use bmh, classic,ggplot for big pictures
        ax1.style.use(['classic'])

        ax2.plot(xc,train_acc)
        ax2.plot(xc,val_acc)
        ax2.ylabel('accuracy')
        ax2.title('train_acc vs val_acc ('+str(X_train.shape[0])+')')
        ax2.grid(True)
        ax2.legend(['train','val'],loc=4)
        f.savefig('baseline' + str(X_train.shape[0]) + '_' + str(i) + '_accuracy.png')
        plt.show()
        plt.close()
        #print plt.style.available # use bmh, classic,ggplot for big pictures

def evaluate_model(num_classes, X_test, y_test, model_str):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "../trained_models/baseline/baseline_"+model_str+".h5"
    abs_file_path = os.path.join(script_dir, rel_path)
    model = load_model(abs_file_path)
    model.summary()
    #scores = model.predict(X_test, batch_size=50, verbose=0)
    predictions = model.predict_classes(X_test, batch_size=50, verbose=0)
    pred = np_utils.to_categorical(predictions, num_classes)

    accuracy = (np.count_nonzero(pred!=y_test)/(predictions.shape[0]*2))*100
    print("Model accuracy: "+ str(accuracy)+ '%')


if __name__ == '__main__':
    main(sys.argv[1])
