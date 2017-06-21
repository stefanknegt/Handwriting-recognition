from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches

x = '-x='
y = '-y='
w = '-w='
h = '-h='
utf = '<utf> '

line = misc.imread(
    '../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571.pgm')  # Contains 3 bad IoU's

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(line, cmap=plt.cm.gray)


# fig1 = plt.figure()
# plt.imshow(line, cmap=plt.cm.gray)

with open('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571.xml', 'r') as original:
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
        ax.add_patch(rect)

with open('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571_updated.xml', 'r') as updated:
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

        # Create a Rectangle patch
        rect = patches.Rectangle((X2, Y2), W2, H2, linewidth=1, edgecolor='g', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

plt.show()


