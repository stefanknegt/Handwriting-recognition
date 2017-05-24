import os

MINIMUM_INSTANCES = 100

def split():
    if not os.path.exists('temp'):
        os.makedirs('temp')
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists('test'):
        os.makedirs('test')

    for updir in os.listdir('128'):
        path = os.path.join('128', updir)
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