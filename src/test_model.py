import cv2
import numpy as np
import os

from keras import backend as K
K.set_image_data_format('channels_first')

from keras.models import load_model

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../data/Train/annotated_crops/128/4e0a/4e0a_aczfkbqulzue.pgm"
abs_file_path = os.path.join(script_dir, rel_path)

input_img=cv2.imread(abs_file_path)
input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
input_img_resize=cv2.resize(input_img,(128,128))

img_data = np.array(input_img_resize)
img_data = img_data.astype('float32')
img_data /= 255
if K.image_data_format() == 'channels_first':
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.expand_dims(img_data, axis=0)
    print (img_data.shape)
else:
    img_data = np.expand_dims(img_data, axis=2)
    img_data = np.expand_dims(img_data, axis=2)
    print (img_data.shape)
test_image = img_data

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "baseline621.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)

print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
