import os, cv2, numpy as np
from keras.models import load_model
from load_data import load_data_internal

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../trained_models/baseline/baseline_128_bin.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)
model.summary()

_, _, _, _, X_test, y_test = load_data_internal('128_bin', verbose = False) #Hier komt een aparte functie voor alleen test data
print('Shape of testing data: '+ str(X_test.shape))
print('Shape of testing labels: '+ str(y_test.shape))
#train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)

#score = model.evaluate(X_test, y_test, batch_size=50, verbose=0)
#print('Test Loss:', score[0])
#print('Test accuracy:', score[1])
#print("Baseline Error: %.2f%%\n" % (100-score[1]*100))

print("Predicted class: "+str(model.predict_classes(X_test[0:1])[0]) + " with certainty: " + str(np.amax(model.predict(X_test[0:1]))))

#if np.amax(model.predict(test_image)) < 0.99: ## ISN'T .99 ARBITRARY, WHY NOT TAKE 2.55*STDEV+MEAN? that's 99% certainty?
    #save current predicted class and entropy
    #run second model
    #if second model softmax > previous then return label otherwise return old label
    #maybe not fair since second model contains more classes?
