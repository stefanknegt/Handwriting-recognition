#!/usr/bin/env python
import sys, os, pickle, numpy as np
from scipy import misc
from keras.models import load_model
from preprocessing import process_for_classification
from keras import backend as K
K.set_image_data_format('channels_last')

DEBUG = True
PLOT = False

def main(im_file='../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm', xml='.'):
    # LOAD MODEL
    try:
        script_dir = os.path.dirname(__file__)
        rel_path = "baseline_extended.h5"
        abs_file_path = os.path.join(script_dir, rel_path)
        model = load_model(abs_file_path)
        if DEBUG:
            model.summary()
    except:
        sys.exit('Could not load model, is the path specified correctly?')

    # LOAD LABELS
    try:
        with open('names.txt', 'rb') as fp:
            names = pickle.load(fp)
    except:
        sys.exit('Could not load labels file, is the path specified correctly?')

    # LOAD IMAGE
    try:
        path = os.path.relpath(im_file)
        img = misc.imread(path)
        if DEBUG:
            print('Processing: '+str(path))
    except:
        sys.exit('Could not load image, is the path specified correctly?')

    # PREPROCESS IMAGE / SEGMENT
    try:
        boxes, characters = process_for_classification(img)
    except:
        sys.exit('Could not segment image')

    # EXPAND FOUND CHARS FOR CLASSIFICATION
    try:
        if K.image_data_format() == 'channels_first':
            characters = np.expand_dims(characters, axis=1)
            if DEBUG:
                print('Found ' + str(characters.shape[0]) + ' characters!')
                print('Test shape of line: ' + str(characters.shape))
        else:
            characters = np.expand_dims(characters, axis=4)
            if DEBUG:
                print('Found ' + str(characters.shape[0]) + ' characters!')
                print('Test shape of line: ' + str(characters.shape))
    except:
        sys.exit('Could not expand character dimensions')

    # SINGLE MODEL CLASSIFICATION
    try:
        predictions = model.predict(characters)
        predicted_class_model = np.argmax(predictions, axis=1)

    except:
        sys.exit('Could not classify characters')

    try:
        if DEBUG:
            characters = np.squeeze(characters, axis=3)
            entropies = []
            for i in range(0, len(predictions)):
                predicted_class_model1 = np.argmax(predictions[i])
                entropy_model1 = predictions[i][predicted_class_model1]
                entropies.append(entropy_model1)
    except:
        sys.exit('Could nog calculate entropy')

    # WRITE RESULTS TO XML
    try:
        j = 0
        if xml=='.':
            xml = im_file.replace(".pgm", ".xml")
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
                line = im_file.replace('.pgm', '') + '-zone-DSTR-x=' + x + '-y=' + y + '-w=' + w + '-h=' + h + ' <utf> ' + pred + ' </utf>' + '\n'
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
        sys.exit('Could not write results to xml')

if __name__ == '__main__':
    if len(sys.argv)==3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv)==2:
        main(sys.argv[1])
    else:
        main()