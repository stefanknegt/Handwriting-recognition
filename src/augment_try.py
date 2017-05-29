from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        rotation_range=10,
        #width_shift_range=0.10,
        #height_shift_range=0.10,
        #shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest')

TIMES = 10
orig = '../data/Train/annotated_crops/128'
new = 'E:/Documenten/Studie/Master/HWR/128_times_'+str(TIMES)
if not os.path.exists(new):
    os.makedirs(new)
for dir in os.listdir(orig):
    if 'Wrd' in str(dir):
        print('NOT PROCESSING: '+ str(dir))
    else:
        print('Processing dir: '+ str(dir))
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