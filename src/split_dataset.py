import os

MINIMUM_INSTANCES = 100

def split():
    if not os.path.exists('../data/Train/temp'):
        os.makedirs('../data/Train/temp')
    if not os.path.exists('../data/Train/train'):
        os.makedirs('../data/Train/train')
    if not os.path.exists('../data/Train/test'):
        os.makedirs('../data/Train/test')

    for updir in os.listdir('../data/Train/128'):
        path = os.path.join('../data/Train/128', updir)
        leng = len([name for name in os.listdir(path) if os.path.isfile(path + '/'+ name)])
        if len
        print len([name for name in os.listdir(path) if os.path.isfile(path + '/'+ name)])
        count = 0
        #pa = os.path.join('annotated_crops/128_extended', dir)
        #for filename in os.listdir(pa):
        #    pat = os.path.join(pa, filename)
        #    if os.path.isfile(pat):
        #        count += 1
        #        total += 1
        #txt = txt + ', ' + str(count) + '\n'

if __name__ == '__main__':
    split()