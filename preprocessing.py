from scipy import misc, ndimage
import numpy as np
import os
import matplotlib.pyplot as plt
from edge_boxes_with_python.edge_boxes import get_windows
from skimage import data
import math
from scipy.optimize import curve_fit
import scipy as sy

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import exposure

BINARY_TH = 200  # binary threshold
NOISE_SIZE_TH = 3  # threshold for what size counts as noise
# minimum nr of pixels lines have to be to count as table lines
MIN_TABLE_SIZE_H = 100
MIN_TABLE_SIZE_V = 100
SPLIT_TH = 0
OVERLAP_TH = 0.1
DEBUG = False

def binarize(img):
    '''turns gray scale image into binary based on threshold'''
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] <= BINARY_TH:
                img[i][j] = 0
            else:
                img[i][j] = 1
    return img


def binarize_otsu(img):
    "Turns greyscale image into binary using Otsu's method"
    val = filters.threshold_otsu(img)
    mask = img > val
    return mask

def rotate_lines(img):
    "Rotates lines to make the characters line up horizontally for further segmentation"
    hor = []
    vert = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] < 1:
                hor.append(j)
                vert.append(i)

    pixels = np.array([hor, vert])
    cov = np.cov(pixels)
    print(cov.tolist())
    w, v = np.linalg.eig(cov)
    angle = math.atan2(v[0,1], v[1,1])
    print(angle)
    rotated_img = ndimage.rotate(img, math.degrees(-angle), order=1)
    return rotated_img

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

def density_plot(img, axis):
    '''takes an image, and returns a histogram of the amount of pixels per column if axis==1 else per row'''
    hist = [0] * img.shape[axis]
    for i in range(0, img.shape[axis]):
        for j in range(0, img.shape[1 - axis]):
            if axis == 1:
                hist[i] += (1 - img[j][i])
            else:
                hist[i] += (1 - img[i][j])
    return hist

def split_by_density(img, hist, axis):
    '''split image based on axis density with a threshold. 0 for horizontal density, 1 for vertical'''
    img_lst = []
    lines = []
    image_flag = False
    white_space = []
    w = 0
    for i in range(0, len(hist)):
        if not image_flag and hist[i] > SPLIT_TH:
            im_start = i
            image_flag = True
            lines.append(i)
            #track nr of whitespaces previous to image segment
            white_space.append(w)
            w = 0
        elif image_flag and hist[i] <= SPLIT_TH:
            im_stop = i
            image_flag = False
            if axis == 1:
                img_lst.append(img[:, im_start:im_stop])
            else:
                img_lst.append(img[im_start:im_stop, :])
            lines.append(i)
        elif not image_flag:
            w += 1
    return img_lst, lines, white_space

class Component:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.size = stop - start
        self.label = None

def split_with_con_comp(img):
    hist = density_plot(img, 1)
    img_lst, lines, wh_sp = split_by_density(img, hist, 1)
    img_lst_new = []
    line_count = 0
    for i in range(0, len(img_lst)):
        image = img_lst[i]
        line_count += wh_sp[i]
        if image.shape[1] < image.shape[0]:
            img_lst_new.append(image)
        else:
            print('trying to split wide image')

            img_inv = np.logical_not(image)
            labels, n_labels = ndimage.label(img_inv)
            mask = image < image.mean()
            sizes = ndimage.sum(mask, labels, range(n_labels + 1))
            ordered_labels = np.argsort(sizes)
            sz = [int(x) for x in sizes.tolist()[1:]]
            print(sz)
            ordered_labels = [i[0] for i in sorted(enumerate(sz), key=lambda x:x[1], reverse=True)]
            print(ordered_labels)
            label_i = 1
            components = []
            for i in range(1, n_labels + 1):
                slice_x, slice_y = ndimage.find_objects(labels == ordered_labels[i - 1] + 1)[0]
                new_comp = Component(slice_y.start, slice_y.stop)
                if not components:
                    new_comp.label = label_i

                for comp in components:
                    if new_comp.start < comp.start:
                        if new_comp.stop > comp.start and (new_comp.stop - comp.start) / min(new_comp.size, comp.size) > OVERLAP_TH:
                            new_comp.label = comp.label
                            break
                    else:
                        if comp.stop > new_comp.start and (comp.stop - new_comp.start) / min(new_comp.size, comp.size) > OVERLAP_TH:
                            new_comp.label = comp.label
                            break
                if not new_comp.label:
                    label_i += 1
                    new_comp.label = label_i
                components.append(new_comp)
                print(new_comp.__dict__)

            min_max = [list([9999, 0])] * label_i
            for comp in components:
                # print(min_max)
                # print(comp.__dict__)
                if comp.start < min_max[comp.label - 1][0]:
                    min_max[comp.label - 1] = list([comp.start, min_max[comp.label - 1][1]])
                if comp.stop > min_max[comp.label - 1][1]:
                    min_max[comp.label - 1] = list([min_max[comp.label - 1][0], comp.stop])

            for slic in min_max:
                lines.append(line_count + slic[0])
                lines.append(line_count + slic[1])
                new_im = image[:, slic[0]:slic[1]]
                img_lst_new.append(new_im)
            # plt.imshow(image)
            # plt.show()
        line_count += image.shape[1]
    return img_lst_new, lines

def remove_whitespace_edges(img):
    for i in range(0,img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j]!=1:
                break
        if img[i][j] !=1:
            break
    img = np.delete(img,np.arange(i),0)
    ud_img = np.flipud(img)
    for i in range(0,ud_img.shape[0]):
        for j in range(0, ud_img.shape[1]):
            if ud_img[i][j]!=1:
                break
        if ud_img[i][j] !=1:
            break
    ud_img = np.delete(ud_img, np.arange(i), 0)
    img = np.flipud(ud_img)
    return img





def main():
    line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-002-y1=280-y2=430.pgm')  # character touches table line
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571.pgm')
    # line = misc.imread(
    #     'Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-005-y1=701-y2=852.pgm')  # character touches table line
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-007-y1=984-y2=1129.pgm')
    # # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-009-y1=1259-y2=1499.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-005-y1=701-y2=852.pgm') # character touches table line
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-007-y1=984-y2=1129.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-008-y1=1120-y2=1268.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-009-y1=1259-y2=1499.pgm')
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18637_0002-line-003-y1=343-y2=508.pgm') # background gray
    # line = misc.imread('Train/lines+xml/1/navis-Ming-Qing_18637_0022-line-009-y1=1226-y2=1386.pgm') # tilted
    if DEBUG:
        fig = plt.figure()
        a = fig.add_subplot(3, 1, 1)
        plt.imshow(line, cmap=plt.cm.gray)
        otsu = binarize_otsu(line)
        normal = binarize(line)
        a = fig.add_subplot(3, 1, 2)
        plt.imshow(normal, cmap=plt.cm.gray)
        a = fig.add_subplot(3, 1, 3)
        plt.imshow(otsu, cmap=plt.cm.gray)
        plt.show()
    otsu = binarize_otsu(line)
    t = rotate_lines(otsu)
    test = remove_table_lines(otsu, 1, MIN_TABLE_SIZE_H)  # removes horizontal table lines
    test_new = remove_table_lines(t, 1, MIN_TABLE_SIZE_H)  # removes horizontal table lines
    test = remove_table_lines(test, MIN_TABLE_SIZE_V, 1)  # removes vertical table lines
    test = remove_noise(test, NOISE_SIZE_TH)
    if DEBUG:
        fig = plt.figure()
        a = fig.add_subplot(3, 1, 1)
        plt.imshow(otsu, cmap=plt.cm.gray)
        a = fig.add_subplot(3, 1, 2)
        plt.imshow(t, cmap=plt.cm.gray)
        a = fig.add_subplot(3, 1, 3)
        plt.imshow(test, cmap=plt.cm.gray)
        plt.show()

    h_hist = density_plot(test, 0)
    # plt.plot(h_hist)
    plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.show()
    lines_list, lines, white_space = split_by_density(test, h_hist, 0)

    if True:
        char_list = []
        for lin in lines_list:
            #plt.imshow(lin, cmap=plt.cm.gray)
            #plt.show()
            im_list, lines = split_with_con_comp(lin)
            char_list.extend(im_list)

        # plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=1)
        # for val in lines_a:
        #     plt.plot([val, val], [0, line.shape[0]], 'b')
        # plt.show()
        # plt.imshow(test, cmap=plt.cm.gray, vmin=0, vmax=1)
        # for val in lines_b:
        #     plt.plot([val, val], [0, line.shape[0]], 'r')
        # plt.show()

    for image in char_list:
        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        h_hist = density_plot(image, 0)
        plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)
        new_img = remove_whitespace_edges(image)
        fig.add_subplot(2, 1, 2)
        plt.imshow(new_img, cmap=plt.cm.gray, vmin=0, vmax=1)

        plt.show()

if __name__ == '__main__':
    main()
