from scipy import misc, ndimage
import numpy as np
import os
import matplotlib.pyplot as plt

BINARY_TH = 200 # binary threshold

def binarize(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] <= BINARY_TH:
                img[i][j] = 0
            else:
                img[i][j] = 1

def remove_table_lines(img):
    img_inv = np.logical_not(img)
    str_e = np.ones((1, 100))
    eroded = ndimage.binary_erosion(img_inv, structure=str_e).astype(img_inv.dtype)
    recon_image = ndimage.binary_propagation(eroded, mask=img_inv)
    recon_image = np.logical_not(recon_image)
    res = np.logical_not(img - recon_image)
    return res

def main():
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-002-y1=280-y2=430.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-005-y1=701-y2=852.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-007-y1=984-y2=1129.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-008-y1=1120-y2=1268.pgm')
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-009-y1=1259-y2=1499.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0005-line-001-y1=0-y2=142.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0005-line-003-y1=269-y2=419.pgm')
    binarize(line)
    test = remove_table_lines(line)
    plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.show()

if __name__ == '__main__':
    main()
