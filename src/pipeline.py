import cv2
import numpy as np
import os

from keras import backend as K
K.set_image_data_format('channels_first')
from keras.models import load_model

from load_data import load_data_internal
from baseline import train_test_evaluate

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../trained_models/baseline/baseline_128.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)

num_classes, input_shape, X_train, y_train, X_test, y_test = load_data_internal('128') #Hier komt een aparte functie voor alleen test data
print(num_classes)
#train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)

score = model.evaluate(X_test, y_test, batch_size=50, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
print("Baseline Error: %.2f%%" % (100-score[1]*100))
