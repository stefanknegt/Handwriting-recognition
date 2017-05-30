import os
from scipy import misc
from preprocessing import binarize_otsu, remove_table_lines, remove_noise, sizes, remove_whitespace_top_bottom
from random import choice
from string import ascii_uppercase
import matplotlib.pyplot as plt

NOISE_SIZE_TH = 3  # threshold for what size counts as noise
# minimum nr of pixels lines have to be to count as table lines
MIN_TABLE_SIZE_H = 100
MIN_TABLE_SIZE_V = 100
DEBUG = False

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
    # File loading
    line = misc.imread(fullpath)
    otsu = binarize_otsu(line)
    test = remove_table_lines(otsu, 1, MIN_TABLE_SIZE_H)  # removes horizontal table lines
    test = remove_table_lines(test, MIN_TABLE_SIZE_V, 1)  # removes vertical table lines
    image = remove_noise(test, NOISE_SIZE_TH)

    xml = filename.replace(".pgm", ".xml")
    pat = os.path.join(dirpath, xml)
    lines = [line.rstrip('\n') for line in open(pat)]
    x = '-x='
    y = '-y='
    w = '-w='
    h = '-h='
    utf = '<utf> '
    wrd = 'Wrd_'
    for line in lines:
        print('Handling: ' + str(line))
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
        # print X, Y, W, H, UTF

        img = image[Y:Y + H, X:X + W]
        new_img = remove_whitespace_top_bottom(img)
        final_img = sizes(new_img, True, 128)

        if DEBUG:
            plt.imshow(final_img, cmap=plt.cm.gray, vmin=0, vmax=1)
            plt.show()

        #im = Image.fromarray(final_img.astype(np.uint8))
        rand = ''.join(choice(ascii_uppercase) for i in range(12))

        if UTF == '-Min':
            wrdlo = line.find(wrd)
            WRD = line[wrdlo:wrdlo + 12]
            path = os.path.join('../data/Train/annotated_crops/128_bin', WRD)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.imsave(path+'/'+WRD + '_' + rand + '.pgm', final_img, cmap =plt.cm.gray, vmin=0, vmax=1)
            #misc.imsave(path+'/'+WRD + '_' + rand + '.pgm', final_img)
        else:
            path = os.path.join('../data/Train/annotated_crops/128_bin', UTF)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.imsave(path + '/' + UTF + '_' + rand + '.pgm', final_img, cmap=plt.cm.gray, vmin=0, vmax=1)
            #misc.imsave(path + '/' + UTF + '_' + rand + '.pgm', final_img)

if __name__ == '__main__':
    main()