from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from preprocessing import *
#from preprocessing import update_top_bottom

# To do: we willen waarschijnlijk de bounding boxes nog verbeteren met de code die Rogier geschreven heeft
# voor het verwijderen van whitespace, maar daar moeten we (Tim en Rogier) samen even voor zitten

def update_xml_boxes(im_path, im_file):
    '''function which finds bounding boxes for a line, using split by density and connected components methods
    from preprocessing.py'''
    print(im_path)
    print(im_file)
    img = misc.imread(os.path.join(im_path, im_file))
    img = preprocess_img(img)

    boxes = []

    # vertical boundaries are stored in top_bottom
    lines_list, top_bottom, white_space = split_by_density(img, 0)

    for i in range(len(lines_list)):
        #print("i is : " + str(i))
        # horizontal boundaries are stored in x_coords
        im_list, x_coords = split_with_con_comp(lines_list[i])
        print(x_coords)
        for j in range(len(im_list)):
            #print("j is : " + str(j))
            im_list[j], im_top_bottom = update_top_bottom(im_list[j], top_bottom[i])
            left_right = x_coords[j]
            # add a 'box' with left top corner coordinate and width and height
            
            boxes.append((left_right[0], im_top_bottom[0], left_right[1] - left_right[0], im_top_bottom[1] - im_top_bottom[0]))
            if True:
                fig,ax = plt.subplots(1)

                # Display the image
                ax.imshow(img, cmap=plt.cm.gray)
                rect = patches.Rectangle((left_right[0], im_top_bottom[0]), left_right[1] - left_right[0], im_top_bottom[1] - im_top_bottom[0], linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                plt.show()
                plt.imshow(im_list[j], cmap=plt.cm.gray)
                plt.show()

    xml = im_file.replace(".pgm", "_updated.xml")

    with open(os.path.join(im_path, xml), 'w') as f:
        for box in boxes:
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
    for i in range(1, 2): # 'for i range(1,13)' gets all folders
        rel_path = os.path.relpath('../data/Train/lines+xml/' + str(i) + '/')
        path = os.path.join(os.getcwd(), rel_path)
        files = os.listdir(path)
        print(path)
        for file in files: # 'for file in files' gets all files
            if '.xml' in file:
                continue
            else:
                update_xml_boxes(rel_path, file)
                break

if __name__ == '__main__':
    main()