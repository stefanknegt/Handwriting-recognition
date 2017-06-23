from scipy import misc
import os
from preprocessing import process_for_classification

def update_xml_boxes(im_path, im_file):
    '''function which finds bounding boxes for a line, using split by density and connected components methods
    from preprocessing.py'''
    print(im_path)
    print(im_file)
    img = misc.imread(os.path.join(im_path, im_file))
    boxes, characters = process_for_classification(img)
    i = 0

    xml = im_file.replace(".pgm", "_updated.xml")

    with open(os.path.join(im_path, xml), 'w') as f:
        for box in boxes:
            '''
            
            '''
            # pad each value with zeros to length of four
            box = ['0' * (4 - len(str(i))) + str(i) for i in box]

            x = str(box[0])
            y = str(box[1])
            w = str(box[2])
            h = str(box[3])
            # print(x, y, w, h)
            f.write('\n' + xml.replace('.xml', '') + '-zone-SEGMENT-x=' + x + '-y=' + y + '-w=' + w + '-h=' + h)

def main():
    # loop over all files in all lines+xml folders
    for i in range(1, 13): # 'for i range(1,13)' gets all folders
        rel_path = os.path.relpath('../data/Train/lines+xml/' + str(i) + '/')
        path = os.path.join(os.getcwd(), rel_path)
        files = os.listdir(path)
        print(path)
        for file in files: # 'for file in files' gets all files
            if '.xml' in file:
                continue
            else:
                update_xml_boxes(rel_path, file)

if __name__ == '__main__':
    main()