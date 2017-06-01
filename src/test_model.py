import cv2, os, numpy as np
from keras.models import load_model
from keras import backend as K

if K.backend()=='tensorflow':
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')



# First, load model
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../trained_models/baseline/baseline_128bin.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)

#Then, load image to test
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../data/Train/annotated_crops/128/4e0a/4e0a_aczfkbqulzue.pgm"
abs_file_path = os.path.join(script_dir, rel_path)

input_img=cv2.imread(abs_file_path, flags=0)
input_img_resize=cv2.resize(input_img,(128,128))

img_data = np.array(input_img_resize)
img_data = img_data.astype('float32')
img_data /= 255
if K.image_data_format() == 'channels_first':
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.expand_dims(img_data, axis=0)
    print('Input dimensions for model: ' + str(img_data.shape))
else:
    img_data = np.expand_dims(img_data, axis=2)
    img_data = np.expand_dims(img_data, axis=2)
    print('Input dimensions for model: ' + str(img_data.shape))
test_image = img_data

print(model.predict(test_image))
print(model.predict_classes(test_image))

print(np.amax(model.predict(test_image)))
