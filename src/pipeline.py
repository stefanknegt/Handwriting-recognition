import os, cv2, numpy as np
from keras.models import load_model
from load_data import load_data_internal

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../trained_models/baseline/baseline_128bin.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)
model.summary()

num_classes, input_shape, X_train, y_train, X_test, y_test = load_data_internal('128_bin') #Hier komt een aparte functie voor alleen test data
print('Shape of testing data: '+ str(X_test.shape))
print('Shape of testing labels: '+ str(y_test.shape))
#train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)

score = model.evaluate(X_test, y_test, batch_size=50, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
print("Baseline Error: %.2f%%" % (100-score[1]*100))
#model.predict(X_test, batch_size=32, verbose=0)

#if np.amax(model.predict(test_image)) < 0.99:
    #save current predicted class and entropy
    #run second model
    #if second model softmax > previous then return label otherwise return old label
    #maybe not fair since second model contains more classes?
