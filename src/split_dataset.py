import os
from shutil import copyfile

MINIMUM_INSTANCES = 100
TRAIN_SIZE = 0.8
TEST_SIZE = 1-TRAIN_SIZE

def split():
    if not os.path.exists('../data/Train/annotated_crops/128_over_' + str(MINIMUM_INSTANCES-1) + '_train'):
        os.makedirs('../data/Train/annotated_crops/128_over_' + str(MINIMUM_INSTANCES-1) + '_train')

    updir = '../data/Train/annotated_crops/128'

    for dir in os.listdir(updir):
        path = os.path.join(updir, dir)
        leng = len([name for name in os.listdir(path) if os.path.isfile(path + '/'+ name)])
        if leng >= MINIMUM_INSTANCES:
            point = 0
            # Just for Roger
            #break_point = leng
            break_point = int(TRAIN_SIZE*leng)
            print(leng, break_point, leng-break_point)
            for filename in os.listdir(path):
                src = os.path.join(path, filename)
                if point <= break_point:
                    dst = '../data/Train/annotated_crops/128_over_' + str(MINIMUM_INSTANCES-1) + '_train/'
                    dst = os.path.join(dst, dir)
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                else:
                    dst = '../data/Train/annotated_crops/128_over_' + str(MINIMUM_INSTANCES-1) + '_test/'
                    dst = os.path.join(dst, dir)
                    if not os.path.exists(dst):
                        os.makedirs(dst)
                dst = os.path.join(dst, filename)
                copyfile(src, dst)
                point +=1

if __name__ == '__main__':
    #if os.path.exists('../data/Train/annotated_crops/128_train'):
    #    os.remove('../data/Train/annotated_crops/128_train')
    #if os.path.exists('../data/Train/annotated_crops/128_test'):
    #    os.remove('../data/Train/annotated_crops/128_test')
    split()