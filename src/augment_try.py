import os, cv2
from preprocessing import binarize_otsu
import matplotlib.pyplot as plt

def augment():
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    datagen = ImageDataGenerator(
        rotation_range=10,
        #width_shift_range=0.10,
        #height_shift_range=0.10,
        #shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest')

    TIMES = 10
    orig = '../data/Train/annotated_crops/128_bin'
    new = 'E:/Documenten/Studie/Master/HWR/128_bin_times_'+str(TIMES)
    if not os.path.exists(new):
        os.makedirs(new)
    for dir in os.listdir(orig):
        #print('Processing dir: '+ str(dir))
        path_orig = os.path.join(orig, dir)
        path_new = os.path.join(new, dir)
        if not os.path.exists(path_new):
            os.makedirs(path_new)
        for filename in os.listdir(path_orig):
            path_long = os.path.join(path_orig, filename)
            img = load_img(path_long)  # PIL image
            x = img_to_array(img)  # Numpy array (1,128,128)
            x = x.reshape((1,) + x.shape)  # Numpy array (1,1,128,128)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=path_new, save_prefix=dir, save_format='pgm'):
                i +=1
                if i>=TIMES:
                    break
    print("Done, you've got some new files!")

def to_gray():
    data_path = 'E:/Documenten/Studie/Master/HWR/128_times_10'
    for dataset in os.listdir(data_path):
        img_list = os.path.join(data_path, dataset)
        for img in os.listdir(img_list):
            print('handling_to_gray: ' + str(img))
            fullpath = os.path.join(img_list, img)
            input_img = cv2.imread(fullpath, flags=0)
            plt.imsave(fullpath, input_img)

def to_bin():
    data_path = 'E:/Documenten/Studie/Master/HWR/128_bin_times_10'
    for dataset in os.listdir(data_path):
        img_list = os.path.join(data_path, dataset)
        for img in os.listdir(img_list):
            print('handling_to_bin: ' + str(img))
            fullpath = os.path.join(img_list, img)
            input_img = cv2.imread(fullpath, flags=0)
            otsu = binarize_otsu(input_img)
            plt.imsave(fullpath, otsu, cmap=plt.cm.gray, vmin=0, vmax=1)

def extend_to_gray():
    data_path = '../data/Train/annotated_crops/128_extended'
    for dataset in os.listdir(data_path):
        img_list = os.path.join(data_path, dataset)
        for img in os.listdir(img_list):
            #print('handling_extend_to_gray: '+str(img))
            fullpath = os.path.join(img_list, img)
            input_img = cv2.imread(fullpath, flags=0)
            plt.imsave(fullpath, input_img)

def extend_to_bin():
    data_path = '../data/Train/annotated_crops/128_extended_bin'
    for dataset in os.listdir(data_path):
        img_list = os.path.join(data_path, dataset)
        for img in os.listdir(img_list):
            #print('handling_extend_to_bin: ' + str(img))
            fullpath = os.path.join(img_list, img)
            input_img = cv2.imread(fullpath, flags=0)
            otsu = binarize_otsu(input_img)
            plt.imsave(fullpath, otsu, cmap=plt.cm.gray, vmin=0, vmax=1)

def create_test_set(folder, NUM):
    orig = '../data/Train/annotated_crops/'+folder
    new = '../data/Train/annotated_crops/test_set_'+folder
    if not os.path.exists(new):
        os.makedirs(new)
    i = 0
    for dir in os.listdir(orig):
        if i < NUM:
            pat_or = os.path.join(orig, dir)
            pat_new = os.path.join(new, dir)
            if not os.path.exists(pat_new):
                os.makedirs(pat_new)
            once = True
            for filename in os.listdir(pat_or):
                if once:
                    file_orig = os.path.join(pat_or, filename)
                    file_new = os.path.join(pat_new, filename)
                    input_img = cv2.imread(file_orig, flags=0)
                    otsu = binarize_otsu(input_img)
                    plt.imsave(file_new, otsu, cmap=plt.cm.gray, vmin=0, vmax=1)
                    once=False
        else:
            break


if __name__ == '__main__':
    print('creating test sets')
    create_test_set('128_bin', 1000)
    create_test_set('128_extended_bin', 1000)