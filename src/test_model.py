import cv2
import numpy as np

from keras import backend as K
K.set_image_data_format('channels_first')

from keras.models import load_model

input_img=cv2.imread('../data/Train/annotated_crops/128/4e0a/4e0a_aczfkbqulzue.pgm')
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



model = load_model('baseline621.h5')

print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
