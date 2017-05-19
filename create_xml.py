from scipy import misc
import os

from preprocessing import *


def update_xml_boxes(im_path, im_file):
    img = misc.imread(os.path.join(im_path, im_file))
    img = preprocess_img(img)

    boxes = []

    lines_list, top_bottom, white_space = split_by_density(img, 0)

    for i in range(len(lines_list)):
        im_list, x_coords = split_with_con_comp(lines_list[i])
        for left_right in x_coords:
            boxes.append((left_right[0], top_bottom[i][0], left_right[1] - left_right[0], top_bottom[i][1] - top_bottom[i][0]))

    xml = im_file.replace(".pgm", ".xml")

    with open(os.path.join(im_path, xml), 'a') as f:
        for box in boxes:

            box = ['0' * (4 - len(str(i))) + str(i) for i in box]

            x = str(box[0])
            y = str(box[1])
            w = str(box[2])
            h = str(box[3])
            # print(x, y, w, h)
            f.write('\n' + xml.replace('.xml', '') + '-zone-SEGMENT-x=' + x + '-y=' + y + '-w=' + w + '-h=' + h)

def main():
    for i in range(1, 2): # 'for i range(1,13)' gets all folders
        rel_path = os.path.relpath('Train/lines+xml/' + str(i) + '/')
        path = os.path.join(os.getcwd(), rel_path)
        files = os.listdir(path)
        print(path)
        for file in [files[0], files[1]]: # 'for file in files' gets all files
            if '.xml' in file:
                continue
            else:
                update_xml_boxes(rel_path, file)

if __name__ == '__main__':
    main()