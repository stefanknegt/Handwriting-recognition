#!/usr/bin/env python
import sys, os, pickle, numpy as np
from scipy import misc
from keras.models import load_model
from preprocessing import process_for_classification
from keras import backend as K
K.set_image_data_format('channels_last')

DEBUG = True    # boolean to log debugging statements
PLOT = False    # boolean to draw image plots

# LOAD MODEL
script_dir = os.path.dirname(__file__)
rel_path = "red_extended_cnn1_nodropout.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)
if DEBUG:
    model.summary()

# LOAD LABELS
with open('names_red.txt', 'rb') as fp:
    names = pickle.load(fp)



def main(folder):
    path = os.path.relpath(folder)
    for filename in os.listdir(path):
        pat = os.path.join(path, filename)
        if '.pgm' in filename:
            process_line(pat, filename)
        else:
            print(filename + ' is not a .pgm file, ignored')

def process_line(path, filename):
    # LOAD IMAGE
    try:
        img = misc.imread(path)
        if DEBUG:
            print('Processing: '+str(path))
    except:
        print('Could not load image, is the path specified correctly? filename: ' + filename)
        return 0

    # PREPROCESS IMAGE / SEGMENT
    try:
        boxes, characters = process_for_classification(img)
    except:
        print('Could not segment image, filename: ' + filename)
        return 0

    # EXPAND FOUND CHARS FOR CLASSIFICATION
    try:
        characters = np.expand_dims(characters, axis=4)
        if DEBUG:
            print('Found ' + str(characters.shape[0]) + ' characters!')
            print('Test shape of line: ' + str(characters.shape))
    except:
        print('Could not expand character dimensions, filename: ' + filename)
        return 0

    # SINGLE MODEL CLASSIFICATION
    try:
        predictions = model.predict(characters)
        predicted_class_model = np.argmax(predictions, axis=1)

    except:
        print('Could not classify characters, filename: ' + filename)
        return 0

    try:
        if DEBUG:
            characters = np.squeeze(characters, axis=3)
            entropies = []
            for i in range(0, len(predictions)):
                predicted_class_model1 = np.argmax(predictions[i])
                entropy_model1 = predictions[i][predicted_class_model1]
                entropies.append(entropy_model1)
    except:
        print('Could nog calculate entropy, filename: ' + filename)
        return 0

    # WRITE RESULTS TO XML
    try:
        j = 0
        xml = path.replace(".pgm", ".xml")
        with open(xml, 'wb') as f:
            for box in boxes:
                # pad each value with zeros to length of four
                box = ['0' * (4 - len(str(i))) + str(i) for i in box]
                x = str(box[0])
                y = str(box[1])
                w = str(box[2])
                h = str(box[3])
                pred = names[predicted_class_model[j]]
                j += 1
                line = filename.replace('.pgm', '') + '-zone-DSTR-x=' + x + '-y=' + y + '-w=' + w + '-h=' + h + ' <utf> ' + pred + ' </utf>' + '\n'
                f.write(line)
                if DEBUG:
                    print(line)
                    act = entropies[j-1]
                    if PLOT:
                        import matplotlib.pyplot as plt
                        plt.imshow(characters[j-1], cmap=plt.cm.gray, vmin=0, vmax=1)
                        plt.title('Classified as "' + str(pred) + '" with activation ' + str(act))
                        plt.show()
    except:
        print('Could not write results to xml, filename: ' + filename)
        return 0

if __name__ == '__main__':
    if len(sys.argv)==2:
        main(sys.argv[1])
    else:
        sys.exit('Please specify a folder with test images as only argument')