import sys
import re
import numpy as np
from PIL import Image
from random import choice
from string import ascii_uppercase

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def main():
    args = sys.argv[1:]
    for filename in args:
        if '.xml' in filename:
            quit()
        image = read_pgm(filename, byteorder='<')
        # pyplot.imshow(image, pyplot.cm.gray)
        # pyplot.show()
        #print('height = ' + str(len(image)))
        #print('width = ' + str(len(image[0])))
        #print('max = ' + str(max(image[0])))

        xml = filename.replace(".pgm", ".xml")
        lines = [line.rstrip('\n') for line in open(xml)]
        x = '-x='
        y = '-y='
        w = '-w='
        h = '-h='
        utf = '<utf> '
        wrd = 'Wrd_'
        for line in lines:
            xlo = line.find(x) + len(x)
            ylo = line.find(y) + len(y)
            hlo = line.find(h) + len(h)
            wlo = line.find(w) + len(w)
            utflo = line.find(utf) + len(utf)
            X = int(line[xlo:xlo + 4])
            Y = int(line[ylo:ylo + 4])
            W = int(line[wlo:wlo + 4])
            H = int(line[hlo:hlo + 4])
            UTF = line[utflo:utflo + 4]
            #print X, Y, W, H, UTF

            img = image[Y:Y + H, X:X + W]
            img = np.rot90(img, 3)
            im = Image.fromarray(img.astype(np.uint8))
            rand = ''.join(choice(ascii_uppercase) for i in range(12))

            if UTF == '-Min':
                wrdlo = line.find(wrd)
                WRD = line[wrdlo:wrdlo + 12]
                im.save(WRD + '_' + rand + '.pgm')
            else:
                im.save(UTF + '_' + rand + '.pgm')

if __name__ == '__main__':
    main()