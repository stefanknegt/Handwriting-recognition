import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

''''THIS FILE IS COMPELTELY DEPRECATED, USED FOR EXTENDING DATASET'''

def main():
    count = 0
    for dir in os.listdir('.'):
        if 'Chinese' in dir:
            suffix = dir.replace("Chinese-unicode-chars-", '')
            #print(suffix)
            for filename in os.listdir(dir):
                unicode = filename.replace('utf-', '')
                unicode = unicode.replace('.jpg', '')
                #print(unicode)
                pat = os.path.join(dir, filename)
                img = misc.imread(pat)
                image = misc.imresize(img, (128, 128), interp='nearest')
                im = Image.fromarray(image.astype(np.uint8))
                if not os.path.exists('Train/annotated_crops/128_extended/' + unicode):
                    os.makedirs('Train/annotated_crops/128_extended/' + unicode)
                im.save('Train/annotated_crops/128_extended/' + unicode + '/' + unicode + '_' + suffix + '.pgm')
                count += 1
            print('files processed: ', count)


def lowercase_rename( dir ):
    # renames all subforders of dir, not including dir itself
    def rename_all( root, items):
        for name in items:
            try:
                os.rename( os.path.join(root, name),
                                    os.path.join(root, name.lower()))
            except OSError:
                pass # can't rename it, so what

    # starts from the bottom so paths further up remain valid after renaming
    for root, dirs, files in os.walk( dir, topdown=False ):
        rename_all( root, dirs )
        rename_all( root, files)


if __name__ == '__main__':
    #lowercase_rename("Train/annotated_crops/128")
    main()