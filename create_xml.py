from scipy import misc
import os

from preprocessing import *


def update_xml(im_file):
    img = misc.imread(im_file)
    img = preprocess_img(img)

    boxes = []

    lines_list, top_bottom, white_space = split_by_density(img, 0)

    for i in range(len(lines_list)):
        im_list, left_right = split_with_con_comp(lines_list[i])
        print(left_right)
        boxes.append(left_right[0], top_bottom[i][0], top_bottom[i][1] - top_bottom[i][0], left_right[1] - left_right[0])

    xml = im_file.replace(".pgm", ".xml")

    with open(xml, 'a') as f:
        for box in boxes:
            x = str(box[0])
            y = str(box[1])
            w = str(box[2])
            h = str(box[3])
            for i in [x, y, w, h]:
                if len(i) < 4:
                    zeros = [0] * (4 - len(i))
                    i = zeros + i
            print(x, y, w, h)
            # f.write(xml + '-zone-SEGMENT-x=' + x + '-y=' + y + '-w=' + w + '-h=' + h)

def main():
    for i in range(1, 2):
        rel_path = os.path.relpath('Train/lines+xml/' + str(i) + '/')
        path = os.path.join(os.getcwd(), rel_path)
        files = os.listdir(path)
        print(path)
        for file in [files[0], files[1]]:
            if '.xml' in file:
                continue
            else:
                update_xml(os.path.join(rel_path, file))

if __name__ == '__main__':
    main()