import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.optimize import curve_fit
from PIL import Image

DEBUG_3 = False

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def main():
    for filename in os.listdir('data'):
        if 'Wrd_' in filename:
            directory = filename[0:12]
            if not os.path.exists('data/'+directory):
                os.makedirs('data/'+directory)
            os.rename('data/'+filename, 'data/'+directory+'/'+filename)
        else:
            directory = filename[0:4]
            if not os.path.exists('data/'+directory):
                os.makedirs('data/'+directory)
            os.rename('data/'+filename, 'data/'+directory+'/'+filename)

def count():
    txt = ''
    total = 0
    for dir in os.listdir('annotated_crops/original'):
        txt = txt+dir
        count = 0
        pa = os.path.join('annotated_crops/original', dir)
        for filename in os.listdir(pa):
            pat = os.path.join(pa, filename)
            if os.path.isfile(pat):
                count += 1
                total += 1
        txt = txt + ', ' + str(count) + '\n'
        
    print(total)
    text_file = open("Occurences.txt", "w")
    text_file.write(txt)
    text_file.close()

def sizes():
    orig = 'annotated_crops/original'
    new = 'annotated_crops/128'
    sizes = np.zeros((27025,2), dtype=int)
    i=0
    for dir in os.listdir(orig):
        if not os.path.exists('annotated_crops/128/'+dir):
           os.makedirs('annotated_crops/128/'+dir)
        pa = os.path.join(orig, dir)
        pa2 = os.path.join(new, dir)
        for filename in os.listdir(pa):
            pat = os.path.join(pa, filename)
            if os.path.isfile(pat):
                image = misc.imread(pat)
                image = image.astype(int)
                sizes[i] = [image.shape[0], image.shape[1]]
                i += 1

                if image.shape[0]!=image.shape[1]:

                    if image.shape[0] >= image.shape[1]:
                        max_size = image.shape[0]
                    else:
                        max_size = image.shape[1]

                    if DEBUG_3:
                        fig = plt.figure()
                        fig.add_subplot(3, 1, 1)
                        plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)

                    diff0 = (max_size - image.shape[0])
                    if diff0 <= 0:
                        pass
                    else:
                        if diff0%2!=0:
                            diff0 += 1
                            a = np.ones((1, image.shape[0]), dtype=np.int)
                            a = a * 255
                            image = np.insert(image, 0, a, 1)
                        diff0 = int(diff0/ 2)
                        a = np.ones((diff0, image.shape[1]), dtype=np.int)
                        a = a*255
                        image = np.insert(image, 0, a, 0)
                        image = np.concatenate((image, a), axis=0)

                    diff1 = max_size - image.shape[1]
                    if diff1 <= 0:
                        pass
                    else:
                        if diff1%2!=0:
                            diff1 += 1
                            a = np.ones((1, image.shape[1]), dtype=np.int)
                            a = a * 255
                            image = np.insert(image, 0, a, 0)
                        diff1 = int(diff1/ 2)
                        a = np.ones((diff1, image.shape[0]), dtype=np.int)
                        a = a*255
                        image = np.insert(image, 0, a, 1)
                        a = np.ones((image.shape[0], diff1), dtype=np.int)
                        a = a*255
                        image = np.concatenate((image, a), axis=1)

                    if DEBUG_3:
                        fig.add_subplot(3,1,2)
                        plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)

                new_image = misc.imresize(image, (120,120), interp='nearest')
                a = np.ones((4, new_image.shape[0]), dtype=np.int)
                a = a * 255
                new_image = np.insert(new_image, 0, a, 1)
                a = np.ones((new_image.shape[0], 4), dtype=np.int)
                a = a * 255
                new_image = np.concatenate((new_image, a), axis=1)
                a = np.ones((4, new_image.shape[1]), dtype=np.int)
                a = a * 255
                new_image = np.insert(new_image, 0, a, 0)
                new_image = np.concatenate((new_image, a), axis=0)
                image = new_image

                if DEBUG_3:
                    fig.add_subplot(3,1,3)
                    plt.imshow(new_image, cmap=plt.cm.gray, vmin=0, vmax=1)
                    plt.show()

                im = Image.fromarray(image.astype(np.uint8))
                im.save(pa2 + '/' + filename)

    print('done')

if __name__ == '__main__':
    count()

