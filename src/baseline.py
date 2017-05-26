import numpy as np
import os, cv2

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_data_format('channels_first')


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Define data path
data_path = '../data/Train/annotated_crops/128_over_99_train'
data_dir_list = os.listdir(data_path)
num_classes = len(data_dir_list)


img_rows=128
img_cols=128
num_channel=1
num_epoch=10

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

# define baseline model
def baseline_model_MLP():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def baseline_model_CNN():
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

# # build MLP model
# model = baseline_model_MLP()
# # Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# build CNN model
model = baseline_model_CNN()
model.summary()
# Fit the model
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, batch_size=200, verbose=2)

# Final evaluation of the model
score = model.evaluate(X_test, y_test, batch_size=100, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
print("Baseline Error: %.2f%%" % (100-score[1]*100))

model.save('baseline'+int(num_classes)+'.h5')

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
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures