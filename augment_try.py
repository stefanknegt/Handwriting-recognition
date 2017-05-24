from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=10,
        #width_shift_range=0.10,
        #height_shift_range=0.10,
        #shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest')

img = load_img('Train/annotated_crops/128/4e0a/4e0a_aczfkbqulzue.pgm')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (1, 128, 128)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 1, 128, 128)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='Train/preview', save_prefix='4e0a', save_format='jpeg'):
    i += 1
    if i > 25:
        break  # otherwise the generator would loop indefinitely