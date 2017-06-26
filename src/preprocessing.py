from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

BINARY_TH = 200  # binary threshold
NOISE_SIZE_TH = 3  # threshold for what size counts as noise
# minimum nr of pixels lines have to be to count as table lines
MIN_TABLE_SIZE_H = 100
MIN_TABLE_SIZE_V = 100
SPLIT_TH = 0
OVERLAP_TH = 0.1
DEBUG = True

def binarize_otsu(img):
    "Turns greyscale image into binary using Otsu's method"
    val = filters.threshold_otsu(img)
    mask = img > val
    return mask

def remove_table_lines(img, x, y):
    '''Removes horizontal or vertical table lines'''
    img_inv = np.logical_not(img)
    str_e = np.ones((x, y))
    # parameters voor het str_d element waarmee otsu net zo goed werkt voor het removen van de tablelines als global threshold van 254:
    if x > y:
        str_d = np.ones((x*10, y+7)) # dilate with a slightly larger structuring element than for the erosion to get rid of irregular table lines a bit better
    else:
        str_d = np.ones((x+7, y*10))
    eroded = ndimage.binary_erosion(img_inv, structure=str_e).astype(img_inv.dtype)
    # recon_image = ndimage.binary_propagation(eroded, mask=img_inv)
    recon_image = ndimage.binary_dilation(eroded, structure=str_d).astype(eroded.dtype)
    #recon_image = np.logical_not(recon_image)
    #res = np.logical_not(img - recon_image)
    res = np.logical_or(img, recon_image)
    return res
    
def remove_table_lines2(img, ref_img, x, y):
    '''Removes horizontal or vertical table lines'''
    bin_img = ref_img > 254
    img_inv = np.logical_not(bin_img)
    str_e = np.ones((x, y))
    str_d = np.ones((x+5, y+5)) # dilate with a slightly larger structuring element than for the erosion to get rid of irregular table lines a bit better
    eroded = ndimage.binary_erosion(img_inv, structure=str_e).astype(img_inv.dtype)
    # recon_image = ndimage.binary_propagation(eroded, mask=img_inv)
    recon_image = ndimage.binary_dilation(eroded, structure=str_d).astype(eroded.dtype)
    #recon_image = np.logical_not(recon_image)
    res = np.logical_or(img, recon_image)
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

def split_by_density(img, axis):
    '''split image based on axis density with a threshold. 0 for horizontal density, 1 for vertical'''
    hist = density_plot(img, axis)
    img_lst = []
    lines = []
    image_flag = False
    white_space = []
    w = 0
    for i in range(0, len(hist)):
        if not image_flag and hist[i] > SPLIT_TH:
            im_start = i
            image_flag = True
            #lines.append(i)
            #track nr of whitespaces previous to image segment
            white_space.append(w)
            w = 1
        elif image_flag and (hist[i] <= SPLIT_TH or i + 1 == len(hist)):
            im_stop = i
            image_flag = False
            if axis == 1:
                img_lst.append(img[:, im_start:im_stop])
            else:
                img_lst.append(img[im_start:im_stop, :])
            lines.append((im_start, im_stop))
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
    # hist = density_plot(img, 1)
    img_lst, lines, wh_sp = split_by_density(img, 1)
    img_lst_new = []
    lines_lst_new = []
    line_count = 0
    for i in range(0, len(img_lst)):
        image = img_lst[i]
        line_count += wh_sp[i]
        if image.shape[1] < image.shape[0]:
            img_lst_new.append(image)
            lines_lst_new.append(lines[i])
        else:
            #print('trying to split wide image')

            img_inv = np.logical_not(image)
            labels, n_labels = ndimage.label(img_inv)
            mask = image < image.mean()
            sizes = ndimage.sum(mask, labels, range(n_labels + 1))
            ordered_labels = np.argsort(sizes)
            sz = [int(x) for x in sizes.tolist()[1:]]
            # print(sz)
            ordered_labels = [i[0] for i in sorted(enumerate(sz), key=lambda x:x[1], reverse=True)]
            #print(ordered_labels)
            label_i = 1
            components = []
            for i in range(1, n_labels + 1):
                slice_x, slice_y = ndimage.find_objects(labels == ordered_labels[i - 1] + 1)[0]
                new_comp = Component(slice_y.start, slice_y.stop)
                #print('new comp : ' + str(new_comp.__dict__))
                if not components:
                    new_comp.label = label_i

                for comp in components:
                    #print('compare with comp : ' + str(comp.__dict__))
                    if new_comp.start < comp.start:
                        if new_comp.stop > comp.start:
                            #print((new_comp.stop - comp.start) / float(min(new_comp.size, comp.size)))
                            if (new_comp.stop - comp.start) / float(min(new_comp.size, comp.size)) > OVERLAP_TH:
                                new_comp.label = comp.label
                                break
                    else:
                        if comp.stop > new_comp.start:
                            #print((comp.stop - new_comp.start) / float(min(new_comp.size, comp.size)))
                            if (comp.stop - new_comp.start) / float(min(new_comp.size, comp.size)) > OVERLAP_TH:
                                new_comp.label = comp.label
                                break
                if not new_comp.label:
                    #print('fits with none, new label')
                    label_i += 1
                    new_comp.label = label_i
                components.append(new_comp)
                # print(new_comp.__dict__)

            min_max = [list([9999, 0])] * label_i
            for comp in components:
                if comp.start < min_max[comp.label - 1][0]:
                    min_max[comp.label - 1] = list([comp.start, min_max[comp.label - 1][1]])
                if comp.stop > min_max[comp.label - 1][1]:
                    min_max[comp.label - 1] = list([min_max[comp.label - 1][0], comp.stop])

            for slic in min_max:
                lines_lst_new.append((line_count + slic[0], line_count + slic[1]))
                # lines.append()
                new_im = image[:, slic[0]:slic[1]]
                img_lst_new.append(new_im)

        line_count += image.shape[1]
    return img_lst_new, lines_lst_new

def remove_whitespace_top_bottom(img):
    ''' This removes the whitespace from top and bottom of an img (np-array)'''
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

def update_top_bottom(img, top_bottom):
    '''This removes the whitespace from top and bottom of an img (np-array) and returns the updated top and bottom as a tuple'''
    for i in range(0,img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j]!=1:
                break
        if img[i][j] !=1:
            break
    img = np.delete(img,np.arange(i),0)
    top_offset = i
    ud_img = np.flipud(img)
    for i in range(0,ud_img.shape[0]):
        for j in range(0, ud_img.shape[1]):
            if ud_img[i][j]!=1:
                break
        if ud_img[i][j] !=1:
            break
    ud_img = np.delete(ud_img, np.arange(i), 0)
    bot_offset = i
    img = np.flipud(ud_img)
    top_bottom = (top_bottom[0] + top_offset, top_bottom[1] - bot_offset) # add offsets to top and bottom ycoords
    return img, top_bottom 

def sizes(image, rotate, output):
    if rotate:
        image = np.rot90(image, 3)
    edge = int((output/32.0)/2.0)
    if image.shape[0]!=image.shape[1]:
        if image.shape[0] >= image.shape[1]:
            max_size = image.shape[0]
        else:
            max_size = image.shape[1]

        diff0 = (max_size - image.shape[0])
        if diff0 <= 0:
            pass
        else:
            if diff0%2!=0:
                diff0 += 1
                a = np.ones((1, image.shape[0]), dtype=np.int)
                image = np.insert(image, 0, a, 1)
            diff0 = int(diff0/ 2.0)
            a = np.ones((diff0, image.shape[1]), dtype=np.int)
            image = np.insert(image, 0, a, 0)
            image = np.concatenate((image, a), axis=0)

        diff1 = max_size - image.shape[1]
        if diff1 <= 0:
            pass
        else:
            if diff1%2!=0:
                diff1 += 1
                a = np.ones((1, image.shape[1]), dtype=np.int)
                image = np.insert(image, 0, a, 0)
            diff1 = int(diff1/ 2.0)
            a = np.ones((diff1, image.shape[0]), dtype=np.int)
            image = np.insert(image, 0, a, 1)
            a = np.ones((image.shape[0], diff1), dtype=np.int)
            image = np.concatenate((image, a), axis=1)

    ## This is the final resizing and padding
    new_image = misc.imresize(image, (output-int(2*edge), output-int(2*edge)), interp='nearest')
    a = np.ones((edge, new_image.shape[0]), dtype=np.int)
    new_image = np.insert(new_image, 0, a, 1)
    a = np.ones((new_image.shape[0], edge), dtype=np.int)
    new_image = np.concatenate((new_image, a), axis=1)
    a = np.ones((edge, new_image.shape[1]), dtype=np.int)
    new_image = np.insert(new_image, 0, a, 0)
    new_image = np.concatenate((new_image, a), axis=0)


    return new_image

def combine_small(boxes, small_w = 35):
    def getKey(item):
        return item[0]
    boxes = sorted(boxes, key = getKey)
    boxes2 = []
    skip = False
    for i in range(len(boxes)):
        if skip:
            skip = False
            continue
        if boxes[i][2] <= small_w: # Width of a box is below 30
            if i == 0: # First char, check only with second char
                x2 = boxes[i][0] + boxes[i][2]
                x3 = boxes[i + 1][0]
                gap = x3 - x2
                dir = 'right'
                #print('first char is small')
            elif i >= len(boxes)-1: # Last char, check only with previous char
                x0 = boxes2[ - 1][0] + boxes2[ - 1][2]
                x1 = boxes[i][0]
                gap = x1 - x0
                dir = 'left'
                #print('last char is small')
            else: # Middle chars, check with previous and next char
                #print('small box found')
                x0 = boxes2[-1][0]+boxes2[-1][2]
                x1 = boxes[i][0]
                x2 = boxes[i][0]+boxes[i][2]
                x3 = boxes[i+1][0]
                gap1 = x1 - x0
                gap2 = x3 - x2
                if gap1 < gap2:
                    gap = gap1
                    dir = 'left'
                else:
                    gap = gap2
                    dir = 'right'

            if gap > boxes[i][2]: # gap is groter dan breedte character, staat op zichzelf
                boxes2.append(boxes[i])
                #print('small box is independent')
            else:
                if dir == 'left':
                     # remove latest entry to boxes2
                    x = boxes2[-1][0]
                    y = min(boxes2[-1][1], boxes[i][1])
                    w = boxes[i][0]+boxes[i][2]-boxes2[-1][0]
                    h = max(boxes[i][1]+boxes[i][3], boxes2[-1][1]+boxes2[-1][3]) - y
                    del boxes2[-1]
                    boxes2.append((x ,y ,w ,h ))
                    #print('previous box deleted, small box added to previous box')
                else:
                    x = boxes[i][0]
                    y = min(boxes[i][1], boxes[i+1][1])
                    w = boxes[i+1][0]+boxes[i+1][2]-boxes[i][0]
                    h = max(boxes[i+1][1]+boxes[i+1][3], boxes[i][1]+boxes[i][3]) - y
                    boxes2.append((x, y, w, h))
                    skip = True
                    #print('small box added to right box, next one skipped')
        else:
            boxes2.append(boxes[i])
    return boxes2

def process_for_classification(img):
    otsu = binarize_otsu(img)
    test = remove_table_lines(otsu, 1, MIN_TABLE_SIZE_H)
    test = remove_table_lines(test, MIN_TABLE_SIZE_V, 1)

    #test = remove_table_lines2(otsu, img, 1, MIN_TABLE_SIZE_H)
    #test = remove_table_lines2(test, img, MIN_TABLE_SIZE_V, 1)

    test = remove_noise(test, NOISE_SIZE_TH)

    if DEBUG:
        fig = plt.figure()
        a = fig.add_subplot(3, 1, 1)
        plt.imshow(img > 254, cmap=plt.cm.gray)
        a = fig.add_subplot(3, 1, 2)
        plt.imshow(otsu, cmap=plt.cm.gray)
        a = fig.add_subplot(3, 1, 3)
        plt.imshow(test, cmap=plt.cm.gray)
        plt.show()

    lines_list, top_bottom, white_space = split_by_density(test, 0)
    boxes = []

    if DEBUG:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.imshow(test, cmap=plt.cm.gray)

    for i in range(len(lines_list)):
        im_list, x_coords = split_with_con_comp(lines_list[i])
        for j in range(len(im_list)):
            #print("j is : " + str(j))
            im_list[j], im_top_bottom = update_top_bottom(im_list[j], top_bottom[i])
            left_right = x_coords[j]
            # add a 'box' with left top corner coordinate and width and height
            boxes.append((left_right[0], im_top_bottom[0], left_right[1] - left_right[0], im_top_bottom[1] - im_top_bottom[0]))

            if DEBUG:
                rect = patches.Rectangle((left_right[0], im_top_bottom[0]), left_right[1] - left_right[0], im_top_bottom[1] - im_top_bottom[0], linewidth=1, edgecolor='g', facecolor='none')
                ax1.add_patch(rect)

    boxes = combine_small(boxes)

    if DEBUG:
        ax2.imshow(test, cmap=plt.cm.gray)
        for box in boxes:
            rect = patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1, edgecolor='g', facecolor='none')
            ax2.add_patch(rect)
        plt.show()

    '''
    Extract images from boxes
    '''
    final_images = np.zeros((len(boxes), 128, 128))
    i = 0
    ymax = test.shape[0]
    print(test.shape)
    for box in boxes:
        y0 = box[1]
        y1 = box[1]+box[3]
        x0 = box[0]
        x1 = box[0]+box[2]
        image = test[y0:y1, x0:x1]
        final_img = sizes(image, True, 128)
        final_images[i] = final_img
        i += 1

        if DEBUG:
            fig = plt.figure()

            fig.add_subplot(2, 1, 1)
            plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)
            #h_hist = density_plot(image, 0)
            #x = np.arange(len(h_hist))
            #plt.plot(h_hist, x)

            fig.add_subplot(2, 1, 2)
            plt.imshow(final_img, cmap=plt.cm.gray, vmin=0, vmax=1)
            plt.show()


    return boxes, final_images

def main():
    #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571.pgm') # character touches table line
    #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm')
    #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0006-line-009-y1=1224-y2=1377.pgm')
    line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0006-line-009-y1=1224-y2=1377.pgm')

    process_for_classification(line)


if __name__ == '__main__':
    main()
