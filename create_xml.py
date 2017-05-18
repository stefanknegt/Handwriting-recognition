from scipy import misc

from preprocessing import *


def generate_xml(im_file):
    img = misc.imread(im_file)
    img = preprocess_img(img)
