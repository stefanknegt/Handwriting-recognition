from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

NOISE_SIZE_TH = 3       # threshold for what size counts as noise
MIN_TABLE_SIZE_H = 100  # minimum nr of pixels horiz. lines have to be to count as table lines
MIN_TABLE_SIZE_V = 100  # minimum nr of pixels vert. lines have to be to count as table lines
SPLIT_TH = 0            # threshold black pixel density for pixel density splitting
OVERLAP_TH = 0.1        # threshold for percentage of overlap for splitting based on connected component overlap
DEBUG = False           # boolean to log debugging statements
PLOT = False            # boolean to draw image plots

def binarize_otsu(img):
    '''Turns greyscale image into binary using Otsu's method.
    
    parameters
    img:    a grayscale image
    
    returns
    mask:   an otsu thresholded binary image
    '''
    val = filters.threshold_otsu(img)
    mask = img > val
    return mask

def remove_table_lines(img, x, y):
    '''Removes horizontal or vertical table lines with morphological image operations.
    
    parameters 
    img:    an input image which potentially contains table lines
    x, y:   the dimensions of the erosion structuring element, one of these values will be the min. length
            an uninterrupted straight line has to have to be considered a table line, the other will be one
    
    returns
    res:    a version of the image with the identified table lines removed
    '''
    img_inv = np.logical_not(img)
    str_e = np.ones((x, y)) 
    if x > y:
        str_d = np.ones((x*10, y+7)) # dilate with a larger str. element than the one for the erosion
    else:
        str_d = np.ones((x+7, y*10))
    eroded = ndimage.binary_erosion(img_inv, structure=str_e).astype(img_inv.dtype)
    # here recon_image will be a black image with a white region around the location of each identified table line
    recon_image = ndimage.binary_dilation(eroded, structure=str_d).astype(eroded.dtype) 
    res = np.logical_or(img, recon_image) # black pixels within the table line region(s) are removed
    return res   

def remove_noise(img, threshold, inv=True):
    '''Removes small noise pixels using morphological closing or opening by reconstruction
    
    parameters
    img:        a noisy image
    threshold:  threshold size for what is seen as noise and what is seen as an object in the image
                (size of the structuring disk)
    inv:        a boolean if true a closing by reconstruction is performed else an opening by reconstruction
                (inv=True or False removes small black or white noise specs respectively)  

    returns
    recon_image    
    '''
    if inv:
        img = np.logical_not(img)
    str_e = np.ones((threshold, threshold))
    eroded = ndimage.binary_erosion(img, structure=str_e).astype(img.dtype)
    recon_image = ndimage.binary_propagation(eroded, mask=img)
    if inv:
        recon_image = np.logical_not(recon_image)
    return recon_image

def density_plot(img, axis):
    '''Makes a histogram of the amount of pixels per row or column of an image.
    
    parameters
    img:    the input image
    axis:   the axis for which to make the histogram, 0 for the horizontal axis 1 for the vertical
    
    returns
    hist:   a density histogram of the number of pixels per row/column on the axis specified by 'axis'
    '''
    hist = [0] * img.shape[axis]
    for i in range(0, img.shape[axis]):
        for j in range(0, img.shape[1 - axis]):
            if axis == 1:
                hist[i] += (1 - img[j][i])
            else:
                hist[i] += (1 - img[i][j])
    return hist

def split_by_density(img, axis):
    '''Splits an image of a line of characters into individual characters based on an axis pixel density, 
    
    parameters
    img:            the input image
    axis:           the axis for which to make the histogram, 0 for the horizontal axis 1 for the vertical
    
    returns 
    img_lst:        a list of character image segments (numpy arrays) 
    lines:          a list of begin and end coordinates of the image segments within the larger line image
    white_space:    a list of the number of white spaces between each of the image segments
    '''
    hist = density_plot(img, axis)
    img_lst = []
    lines = []
    image_flag = False
    white_space = []
    w = 0
    # move through the rows/cols of the image
    for i in range(0, len(hist)):
        if not image_flag and hist[i] > SPLIT_TH: # beginning of a new character image segment
            im_start = i
            image_flag = True
            #track nr of whitespaces previous to image segment
            white_space.append(w)
            w = 1
        elif image_flag and (hist[i] <= SPLIT_TH or i + 1 == len(hist)): # end of a character image segment
            im_stop = i
            image_flag = False
            if axis == 1:
                img_lst.append(img[:, im_start:im_stop])
            else:
                img_lst.append(img[im_start:im_stop, :])
            lines.append((im_start, im_stop))
        elif not image_flag: # outsite of an character image segment
            w += 1
    return img_lst, lines, white_space

class Component:
    '''A class for a connected component within an image segment
    
    properties
    start:  the x coordinate where the component starts
    stop:   the x coordinate where the component stops
    size:   the length of the component in the x dimension
    label:  the character label of the component, components with the same label belong to the same character
    '''
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.size = stop - start
        self.label = None

def split_with_con_comp(img):
    '''Splits an image of a line of characters into individual characters based on clusterings of overlapping connected components. 
    
    parameters
    img:            the input image
    
    returns
    img_lst_new:    a list of character image segments (numpy arrays) 
    lines_lst_new:  a list of begin and end coordinates of the image segments within the larger line image
    '''
    # Call on split_by_density for the initial split
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
        else: # the image segment is so wide that it is likely to contain multiple characters
            if DEBUG:
                print('trying to split wide image')

            # create a labeling of all connected components within the image segment and sort them by their size
            img_inv = np.logical_not(image)
            labels, n_labels = ndimage.label(img_inv)
            mask = image < image.mean()
            sizes = ndimage.sum(mask, labels, range(n_labels + 1))
            ordered_labels = np.argsort(sizes)
            sz = [int(x) for x in sizes.tolist()[1:]]
            ordered_labels = [i[0] for i in sorted(enumerate(sz), key=lambda x:x[1], reverse=True)]

            label_i = 1
            components = []
            for i in range(1, n_labels + 1): # go through all connected components starting with the largest
                slice_x, slice_y = ndimage.find_objects(labels == ordered_labels[i - 1] + 1)[0]
                new_comp = Component(slice_y.start, slice_y.stop)
                
                # assign a cluster label to the largest component
                if not components:
                    new_comp.label = label_i
                
                # compare each new component with all previous components (starting with the largest) and assign it the same class label as the first component it 
                # has enough overlap with
                for comp in components: 
                    if new_comp.start < comp.start:
                        if new_comp.stop > comp.start:
                            if DEBUG:
                                print((new_comp.stop - comp.start) / float(min(new_comp.size, comp.size)))
                            if (new_comp.stop - comp.start) / float(min(new_comp.size, comp.size)) > OVERLAP_TH:
                                new_comp.label = comp.label
                                break
                    else:
                        if comp.stop > new_comp.start:
                            if DEBUG:
                                print((comp.stop - new_comp.start) / float(min(new_comp.size, comp.size)))
                            if (comp.stop - new_comp.start) / float(min(new_comp.size, comp.size)) > OVERLAP_TH:
                                new_comp.label = comp.label
                                break
                # if the component doesn't have an overlap (as determined by the threshold) with any of the previous components assign it a new class label
                if not new_comp.label:
                    label_i += 1
                    new_comp.label = label_i
                components.append(new_comp)

            # find the beginning and ending of the characters as defined by their new class labels
            min_max = [list([9999, 0])] * label_i
            for comp in components:
                if comp.start < min_max[comp.label - 1][0]:
                    min_max[comp.label - 1] = list([comp.start, min_max[comp.label - 1][1]])
                if comp.stop > min_max[comp.label - 1][1]:
                    min_max[comp.label - 1] = list([min_max[comp.label - 1][0], comp.stop])

            for slic in min_max:
                lines_lst_new.append((line_count + slic[0], line_count + slic[1]))
                new_im = image[:, slic[0]:slic[1]]
                img_lst_new.append(new_im)

        line_count += image.shape[1]
    return img_lst_new, lines_lst_new

def update_top_bottom(img, top_bottom):
    '''Crops the whitespace from top and bottom of a character image
    
    parameters
    img:        a character input image
    top_bottom: the current top and bottom y coordinates of the character in a larger line image
    
    returns 
    img:        the cropped image
    top_bottom: the updated top and bottom y coordinates
    '''
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

def sizes(image, rotate):
    '''Rotates input, zeropads to square image and resizes result to be 128 by 128 pixels with 2 pixel whitespace border

    parameters
    image:      input image
    rotate:     boolean, true will rotate the image 90 degrees clockwise

    returns
    new_image:  the rotated, centered 128*128 pixels image


    '''
    if rotate:
        image = np.rot90(image, 3)
    edge = 2
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
    new_image = misc.imresize(image, (124, 124), interp='nearest')
    a = np.ones((edge, new_image.shape[0]), dtype=np.int)
    new_image = np.insert(new_image, 0, a, 1)
    a = np.ones((new_image.shape[0], edge), dtype=np.int)
    new_image = np.concatenate((new_image, a), axis=1)
    a = np.ones((edge, new_image.shape[1]), dtype=np.int)
    new_image = np.insert(new_image, 0, a, 0)
    new_image = np.concatenate((new_image, a), axis=0)

    return new_image

def combine_small(boxes, small_w = 35, small_h = 15):
    '''Combines or appends small loose character strokes that are oversegmented when an image is split by density
        also deletes boxes that are smaller than both thresholds
    
    parameters
    boxes:      a list of bounding boxes of characters in a line
    small_w:    width threshold
    small_h:    height threshold
    
    returns
    boxes2:     updated list of bounding boxes
    '''
    if DEBUG:
        print('Searching for small characters')
    def getKey(item):
        return item[0]
    boxes = sorted(boxes, key = getKey)
    boxes2 = []
    skip = False
    for i in range(len(boxes)):
        if skip:
            skip = False
            continue
        if boxes[i][2] <= small_w: # Width of a box is below the threshold
            if boxes[i][3] <= small_h:
                continue
            if len(boxes2)==0: # First char, check only with second char
                x2 = boxes[i][0] + boxes[i][2]
                x3 = boxes[i + 1][0]
                gap = x3 - x2
                dir = 'right'
                if DEBUG:
                    print('first char is small')
            elif i >= len(boxes)-1: # Last char, check only with previous char
                x0 = boxes2[-1][0] + boxes2[-1][2]
                x1 = boxes[i][0]
                gap = x1 - x0
                dir = 'left'
                if DEBUG:
                    print('last char is small')
            else: # Middle chars, check with previous and next char
                if DEBUG:
                    print('small box found')
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

            if gap > 8: # gap is groter dan 15, staat op zichzelf
                boxes2.append(boxes[i])
                if DEBUG:
                    print('small box is independent')
            else:
                if dir == 'left':
                     # remove latest entry to boxes2
                    x = boxes2[-1][0]
                    y = min(boxes2[-1][1], boxes[i][1])
                    w = boxes[i][0]+boxes[i][2]-boxes2[-1][0]
                    h = max(boxes[i][1]+boxes[i][3], boxes2[-1][1]+boxes2[-1][3]) - y
                    del boxes2[-1]
                    boxes2.append((x ,y ,w ,h ))
                    if DEBUG:
                         print('previous box deleted, small box added to previous box')
                else:
                    x = boxes[i][0]
                    y = min(boxes[i][1], boxes[i+1][1])
                    w = boxes[i+1][0]+boxes[i+1][2]-boxes[i][0]
                    h = max(boxes[i+1][1]+boxes[i+1][3], boxes[i][1]+boxes[i][3]) - y
                    boxes2.append((x, y, w, h))
                    skip = True
                    if DEBUG:
                        print('small box added to right box, next one skipped')
        else:
            boxes2.append(boxes[i])
    return boxes2

def process_for_classification(img):
    '''Preprocesses and segments an image of a line of characters
    
    parameters
    img:            the input image
    
    returns
    boxes:          a list of bounding boxes for the characters within the original image
    final_image:    a list of character images (numpy arrays)
    '''
    otsu = binarize_otsu(img)
    
    test = remove_noise(otsu, NOISE_SIZE_TH, inv=False)
    
    test = remove_table_lines(test, 1, MIN_TABLE_SIZE_H)
    test = remove_table_lines(test, MIN_TABLE_SIZE_V, 1)

    test = remove_noise(test, NOISE_SIZE_TH)


    if DEBUG and PLOT:
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

    if DEBUG and PLOT:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.imshow(test, cmap=plt.cm.gray)

    for i in range(len(lines_list)):
        im_list, x_coords = split_with_con_comp(lines_list[i])
        for j in range(len(im_list)):
            if DEBUG:
                print("j is : " + str(j))
            im_list[j], im_top_bottom = update_top_bottom(im_list[j], top_bottom[i])
            left_right = x_coords[j]
            # add a 'box' with left top corner coordinate and width and height
            x = left_right[0]-2
            y = im_top_bottom[0]-2
            w = left_right[1] - left_right[0]+2
            h = im_top_bottom[1] - im_top_bottom[0]+2
            if x <= 0:
                x = 0
            if y <= 0:
                y = 0
            if x+w >= test.shape[1]:
                w = test.shape[1]-x
            if y+h >= test.shape[0]:
                h = test.shape[0]-y
            boxes.append((x,y,w,h))

            if DEBUG and PLOT:
                rect = patches.Rectangle((left_right[0], im_top_bottom[0]), left_right[1] - left_right[0], im_top_bottom[1] - im_top_bottom[0], linewidth=1, edgecolor='g', facecolor='none')
                ax1.add_patch(rect)

    boxes = combine_small(boxes)

    if DEBUG and PLOT:
        ax2.imshow(test, cmap=plt.cm.gray)
        for box in boxes:
            rect = patches.Rectangle((box[0],box[1]), box[2], box[3], linewidth=1, edgecolor='g', facecolor='none')
            ax2.add_patch(rect)
        plt.show()

    
    # Extract images from boxes
    final_images = np.zeros((len(boxes), 128, 128))
    i = 0
    for box in boxes:
        y0 = box[1]
        y1 = box[1]+box[3]
        x0 = box[0]
        x1 = box[0]+box[2]
        image = test[y0:y1, x0:x1]
        final_img = sizes(image, True)
        final_images[i] = final_img
        i += 1

        if DEBUG and PLOT:
            fig = plt.figure()

            fig.add_subplot(2, 1, 1)
            plt.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)

            fig.add_subplot(2, 1, 2)
            plt.imshow(final_img, cmap=plt.cm.gray, vmin=0, vmax=1)
            plt.show()

    return boxes, final_images

def main():
    '''The preprocessing main can be used to call process_for_classification on a single example line
    
    Set PLOT to True
       
    make sure the path in misc.imread() points to an existing image
    
    examples:
       #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-003-y1=421-y2=571.pgm') # small last character
       #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0004-line-001-y1=0-y2=289.pgm')
       #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18341_0006-line-009-y1=1224-y2=1377.pgm')
       #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18641_0069-line-003-y1=258-y2=388.pgm')
       #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18637_0017-line-008-y1=1082-y2=1244.pgm')  # Still table lines left
       #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18641_0010-line-006-y1=997-y2=1321.pgm') # Only four chars
       #line = misc.imread('../data/Train/lines+xml/1/navis-Ming-Qing_18641_0028-line-005-y1=574-y2=716.pgm')
    '''
    
    line = misc.imread('../data/Train/lines+xml/7/navis-Ming-Qing_HarvYench_18_10_10084_0015-line-002-y1=509-y2=637.pgm')
    process_for_classification(line)


if __name__ == '__main__':
    main()
