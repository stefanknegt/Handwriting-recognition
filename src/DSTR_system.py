import sys, os, pickle, numpy as np
from scipy import misc
from keras.models import load_model
from preprocessing import process_for_classification

from keras import backend as K
if K.backend() == 'theano':
    K.set_image_data_format('channels_last')
else:
    K.set_image_data_format('channels_first')

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
rel_path = "../trained_models/extended/baseline_extended.h5"
abs_file_path = os.path.join(script_dir, rel_path)
model = load_model(abs_file_path)
model.summary()

with open('names.txt', 'rb') as fp:
    names = pickle.load(fp)

DEBUG = True

def main(folder='../data/Train/lines+xml/4'):
    # Reading in line-to-classify
    #rel_path = os.path.relpath('PATH_TO_TEST_LINES_FOLDER')
    top_path = os.path.relpath(folder)
    path = os.path.join(os.getcwd(), top_path)
    files = os.listdir(path)
    for file in files:  # 'for file in files' gets all files
        if '.xml' in file:
            continue
        else:
            process_line(path, file)

def process_line(path, im_file):
    '''function which finds bounding boxes for a line, using split by density and connected components methods
    from preprocessing.py'''
    if DEBUG:
        print('Processing: '+str(path)+'\\'+str(im_file))
    img = misc.imread(os.path.join(path, im_file))
    boxes, characters = process_for_classification(img)

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

    # SINGLE MODEL, FOR DOUBLE MODEL SEE PIPELINE
    predictions = model.predict(characters)
    predicted_class_model = np.argmax(predictions, axis=1)

    if DEBUG:
        characters = np.squeeze(characters, axis=3)
        entropies = []
        for i in range(0, len(predictions)):
            predicted_class_model1 = np.argmax(predictions[i])
            entropy_model1 = predictions[i][predicted_class_model1]
            entropies.append(entropy_model1)

    j = 0
    xml = im_file.replace(".pgm", "_DSTR.xml")
    #with open(os.path.join(path, xml), 'w') as f:
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

        if DEBUG:
            print(x, y, w, h)
            print(line)
            act = entropies[j-1]
            import matplotlib.pyplot as plt
            plt.imshow(characters[j-1], cmap=plt.cm.gray, vmin=0, vmax=1)
            plt.title('Classified as "' + str(pred) + '" with activation ' + str(act))
            plt.show()

        #f.write(line)



if __name__ == '__main__':
    if len(sys.argv)==2:
        main(sys.argv[1])
    else:
        main()