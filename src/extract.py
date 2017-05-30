import sys, os
import re
import numpy as np
from scipy import misc
from preprocessing import binarize_otsu, remove_table_lines, remove_noise
from PIL import Image
from random import choice
from string import ascii_uppercase

NOISE_SIZE_TH = 3  # threshold for what size counts as noise
# minimum nr of pixels lines have to be to count as table lines
MIN_TABLE_SIZE_H = 100
MIN_TABLE_SIZE_V = 100

'''THIS FILE IS COMPLETELY DEPRECATED, USED FOR EXTRACTING IMAGES FROM XML'''

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
    pat1 = os.path.join('../data/Train','lines+xml')
    for updir in os.listdir(pat1):
        pat2 = os.path.join(pat1, updir)
        if int(updir)<4:
            for filename in os.listdir(pat2):
                if not '.xml' in filename:
                    path = os.path.join(pat2, filename)
                    extract(path, filename)
        else:
            for dir in os.listdir(pat2):
                pat3 = os.path.join(pat2, dir)
                for filename in os.listdir(pat3):
                    if not '.xml' in filename:
                        path = os.path.join(pat3, filename)
                        extract(path, filename)

def extract(fullpath, filename):
    dirpath = fullpath.replace('\\'+filename, '')
    line = misc.imread(fullpath)
    otsu = binarize_otsu(line)
    test = remove_table_lines(otsu, 1, MIN_TABLE_SIZE_H)  # removes horizontal table lines
    test = remove_table_lines(test, MIN_TABLE_SIZE_V, 1)  # removes vertical table lines
    test = remove_noise(test, NOISE_SIZE_TH)


'''
        image = read_pgm(filename, byteorder='<')




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
                if not os.path.exists()
                im.save(WRD + '_' + rand + '.pgm')
            else:
                im.save(UTF + '_' + rand + '.pgm')
'''

if __name__ == '__main__':
    main()