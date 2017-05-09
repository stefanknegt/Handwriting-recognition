from scipy import misc, ndimage
import numpy as np
import os
import matplotlib.pyplot as plt
from edge_boxes_with_python.edge_boxes import get_windows

BINARY_TH = 200 # binary threshold
NOISE_SIZE_TH = 3 # threshold for what size counts as noise
# minimum nr of pixels lines have to be to count as table lines
MIN_TABLE_SIZE_H = 100
MIN_TABLE_SIZE_V = 100

def binarize(img):
    '''turns gray scale image into binary based on threshold'''
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] <= BINARY_TH:
                img[i][j] = 0
            else:
                img[i][j] = 1


def remove_table_lines(img, x, y):
    '''Removes horizontal or vertical table lines'''
    img_inv = np.logical_not(img)
    str_e = np.ones((x, y))
    eroded = ndimage.binary_erosion(img_inv, structure=str_e).astype(img_inv.dtype)
    recon_image = ndimage.binary_propagation(eroded, mask=img_inv)
    recon_image = np.logical_not(recon_image)
    res = np.logical_not(img - recon_image)
    return res


def remove_noise(img, threshold, inv=True):
    '''closing/opening by reconstruction'''
    if inv:
        img = np.logical_not(img)
    str_e = np.ones((threshold, threshold))
    eroded = ndimage.binary_erosion(img, structure=str_e).astype(img.dtype)
    recon_image = ndimage.binary_propagation(eroded, mask=img)
    if inv:
        recon_image = np.logical_not(recon_image)
    return recon_image


def main():
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm')
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-002-y1=280-y2=430.pgm') # character touches table line
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571.pgm')
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-005-y1=701-y2=852.pgm') # character touches table line
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-007-y1=984-y2=1129.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-008-y1=1120-y2=1268.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-009-y1=1259-y2=1499.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0005-line-001-y1=0-y2=142.pgm') # bad line
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0005-line-003-y1=269-y2=419.pgm')
    binarize(line)
    test = remove_table_lines(line, 1, MIN_TABLE_SIZE_H) # removes horizontal table lines
    test = remove_table_lines(test, MIN_TABLE_SIZE_V, 1) # removes vertical table lines
    test = remove_noise(test, NOISE_SIZE_TH)
    plt.imshow(line, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.show()
    plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.show()

if __name__ == '__main__':
    main()
