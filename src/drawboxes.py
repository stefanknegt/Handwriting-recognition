from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import math, os
import matplotlib.patches as patches
from preprocessing import *
x = '-x='
y = '-y='
w = '-w='
h = '-h='
utf = '<utf> '


def main(folder='../data/Train/lines+xml/2'):
    # Reading in line-to-classify
    # rel_path = os.path.relpath('PATH_TO_TEST_LINES_FOLDER')
    top_path = os.path.relpath(folder)
    path = os.path.join(os.getcwd(), top_path)
    files = os.listdir(path)
    for file in files:  # 'for file in files' gets all files
        if '.xml' in file:
            continue
        else:
            process_line(path, file)

def process_line(path, file):
    line_path = os.path.join(path, file)
    imagetry = line_path.replace('.pgm', '')
    line = misc.imread(line_path)

    otsu = binarize_otsu(line)
    test = remove_table_lines(otsu, 1, MIN_TABLE_SIZE_H)
    test = remove_table_lines(test, MIN_TABLE_SIZE_V, 1)
    test = remove_noise(test, NOISE_SIZE_TH)

    # Create figure and axes
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

    # Display the image
    ax1.imshow(line, cmap=plt.cm.gray)
    ax2.imshow(otsu, cmap=plt.cm.gray)
    ax3.imshow(test, cmap=plt.cm.gray)

    # fig1 = plt.figure()
    # plt.imshow(line, cmap=plt.cm.gray)

    with open(imagetry + '.xml', 'r') as original:
        lines = [line.rstrip('\n') for line in original]
        for line in lines:
            # Find coordinates of boundary box in original xml file and save to originalxml list
            xlo = line.find(x) + len(x)
            ylo = line.find(y) + len(y)
            hlo = line.find(h) + len(h)
            wlo = line.find(w) + len(w)
            utflo = line.find(utf) + len(utf)
            X = int(line[xlo:xlo + 4])
            Y = int(line[ylo:ylo + 4])
            W = int(line[wlo:wlo + 4])
            H = int(line[hlo:hlo + 4])
            UTF = line[utflo:utflo + 4]

            # Create a Rectangle patch
            rect = patches.Rectangle((X, Y), W, H, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax3.add_patch(rect)


    with open(imagetry + '_updated.xml', 'r') as updated:
        linesupdated = [lineupdated.rstrip('\n') for lineupdated in updated]

        for lineupdated in linesupdated:
            # Find x coordinates of boundary boxes in updated xml file
            try:
                x2lo = lineupdated.find(x) + len(x)
                w2lo = lineupdated.find(w) + len(w)
                y2lo = lineupdated.find(y) + len(y)
                h2lo = lineupdated.find(h) + len(h)
                X2 = int(lineupdated[x2lo:x2lo + 4])
                W2 = int(lineupdated[w2lo:w2lo + 4])
                Y2 = int(lineupdated[y2lo:y2lo + 4])
                H2 = int(lineupdated[h2lo:h2lo + 4])
            except ValueError:
                continue
            ar = float(float(W2)/float(H2))

            # Create a Rectangle patch
            if W2 < 30:
                rect = patches.Rectangle((X2, Y2), W2, H2, linewidth=1, edgecolor='b', facecolor='none')
            elif H2 < 25:
                rect = patches.Rectangle((X2, Y2), W2, H2, linewidth=1, edgecolor='c', facecolor='none')
            elif ar > 1.8:
                rect = patches.Rectangle((X2, Y2), W2, H2, linewidth=1, edgecolor='m', facecolor='none')
            elif ar < 0.5:
                rect = patches.Rectangle((X2, Y2), W2, H2, linewidth=1, edgecolor='0.75', facecolor='none')
            else:
                rect = patches.Rectangle((X2, Y2), W2, H2, linewidth=1, edgecolor='g', facecolor='none')

            # Add the patch to the Axes
            ax3.add_patch(rect)

    plt.title('added boxes, blue w<30, cyan h<25, magenta ar>1.8, grey ar<0.5')
    plt.show()

if __name__ == '__main__':
    main()
