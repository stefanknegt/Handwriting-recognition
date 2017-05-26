import os
from shutil import copyfile

MINIMUM_INSTANCES = 10

def threshold():
    if not os.path.exists('../data/Train/annotated_crops/128_over_' + str(MINIMUM_INSTANCES-1)):
        os.makedirs('../data/Train/annotated_crops/128_over_' + str(MINIMUM_INSTANCES-1))

    updir = '../data/Train/annotated_crops/128'

    for dir in os.listdir(updir):
        path = os.path.join(updir, dir)
        leng = len([name for name in os.listdir(path) if os.path.isfile(path + '/'+ name)])
        if leng >= MINIMUM_INSTANCES:
            for filename in os.listdir(path):
                src = os.path.join(path, filename)
                dst = '../data/Train/annotated_crops/128_over_' + str(MINIMUM_INSTANCES-1)
                dst = os.path.join(dst, dir)
                if not os.path.exists(dst):
                    os.makedirs(dst)
                dst = os.path.join(dst, filename)
                copyfile(src, dst)

if __name__ == '__main__':
    threshold()