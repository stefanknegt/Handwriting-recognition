import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc

"""face = misc.face()
misc.imsave('face.png', face) # First we need to create the PNG file

face = misc.imread('face.png')
print(type(face))

print(face)"""

threshold = 200

img = misc.imread('../data/Train/annotated_crops/4ea6/4ea6_AJGVLYWOIYHD.pgm')
img = img.astype(int)

print(type(np.unique(img)))

for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        if img[i][j] < threshold:
            img[i][j] = 0
        else:
            img[i][j] = 1

"""print(img)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()"""



countColumn = 0
countRow = 0
rows = np.zeros(shape=(img.shape[0],1))
cols = np.zeros(shape=(img.shape[1],1))
rowlines = []
collines = []

for i in range(0,img.shape[0]):
    for j in range(1,img.shape[1]):
        if((img[i][j] != img[i][j-1])):
            rows[i] += 1
            countRow += 1;
            rowlines.append((i,j))
    rows[i] = float(rows[i])/2

#print(countRow)
#print(rows)

for j in range(0,img.shape[1]):
    for i in range(1,img.shape[0]):
        if((img[i][j] != img[i-1][j])):
            cols[j] += 1
            countColumn += 1
            collines.append((i,j))
    cols[j] = float(cols[i])/2

#print(countColumn)
#print(cols)

plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
for val in rowlines:
    plt.plot([val, val], [0, img.shape[0]], 'r')
plt.show()

plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
for val in collines:
    plt.plot([1, img.shape[1]],[val, val], 'b')
plt.show()
