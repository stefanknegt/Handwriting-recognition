from scipy import misc

from preprocessing import *


def generate_xml(im_file):
    img = misc.imread(im_file)
    img = preprocess_img(img)

    boxes = []

    lines_list, top_bottom, white_space = split_by_density(test, 0)

    for i in range(len(lines_list)):
        im_list, left_right = split_with_con_comp(lin)
        boxes.append(top_bottom[i][0], top_bottom[i][1], left_right[0], left_right[1])
