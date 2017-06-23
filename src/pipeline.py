import os, numpy as np
from keras.models import load_model
from load_data import load_test_data, load_data_internal
from matplotlib import pyplot

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "baseline_21584_0.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)
model.summary()

_, _, _, _, X_test, y_test = load_data_internal('128_bin', verbose = False) #Hier komt een aparte functie voor alleen test data
#X_test, y_test = load_test_data('test_128_extended_bin', verbose = False)
#print(num_classes)
print('Shape of testing data: '+ str(X_test.shape))
print('Shape of testing labels: '+ str(y_test.shape))
#train_test_evaluate(num_classes, input_shape, X_train, y_train, X_test, y_test)

score = model.evaluate(X_test, y_test, batch_size=50, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
print("Baseline Error: %.2f%%\n" % (100-score[1]*100))

predictions = model.predict(X_test, batch_size=50, verbose=0)
#pred = np_utils.to_categorical(predictions, num_classes)

entropies = []

for i in range(0,len(predictions)):
    predicted_class_model1 = np.argmax(predictions[i])
    entropy_model1 = predictions[i][predicted_class_model1]
    entropies.append(entropy_model1)
    if(entropy_model1 < 0.99):
        #entropies.append(entropy_model1)
        print(entropy_model1)
        #Call hier tweede model
        #if(entropy_model1 < entropy_model2): Dit is niet fair omdat tweede model meer classes heeft!
            #predicted_class = predicted_class_model2
        #else:
                #predicted_class = predicted_class_model1
    else:
        predicted_class = predicted_class_model1
        #print("Predicted with certainty by model 1, class is %d with %.2f percent certainty" % (predicted_class, entropy_model1))

#accuracy = (np.count_nonzero(pred!=y_test)/(predictions.shape[0]*2))*100
#print("Model accuracy: "+ str(accuracy)+ '%')
print('Mean entropy: ' + str(np.mean(entropies)))
print('std entropy: ' + str(np.std(entropies)))
pyplot.hist(entropies)
pyplot.axvline(np.mean(entropies), color='b', linestyle='dashed', linewidth=2)
pyplot.show()

def calc(mat1, mat2):
    acc = np.zeros(mat1.shape[0])
    for i in range(mat1.shape[0]):
        acc[i] = 1-(np.count_nonzero(mat1[i]!=mat2[i])/mat1.shape[1])
    return acc
total_accuracy = np.mean(calc(predictions, y_test))
print('Total accuracy is: '+ str(total_accuracy))