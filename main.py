from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

LOWER_LIST = sorted([
    "Denim", "Dresses", "Leggings", "Pants", "Shorts", "Skirts"
])
UPPER_LIST = sorted([
    "Blouses_Shirts", "Cardigans", "Graphic_Tees", "Jackets_Coats", "Sweaters", "Sweatshirts_Hoodies", "Tees_Tanks", "Rompers_Jumpsuits",
    "Jackets_Vests", "Shirts_Polos", "Suiting"
])


def process(filename):
    image = mpimg.imread(filename)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.show()


def predict(path):
    img = image.load_img(path, target_size=(128, 128))

    x = image.img_to_array(img)           # x.shape (128, 128, 3)
    x = np.expand_dims(x, axis=0)       # x.shape (1, 128, 128, 3)
    images = np.vstack([x])        # images.shape (1, 128, 128, 3)

    result = []
    # model_up_low = load_model('./model1.h5')
    # classes = model_up_low.predict(images)
    # label1 = ['lower', 'upper']
    # result.append(np.argmax(classes[0][0]))

    model_low = load_model('./model2.h5')
    next_class = model_low.predict(images)
    label = LOWER_LIST
    result.append(label[np.argmax(next_class[0])])

    model_up = load_model('./model3.h5')
    next_class = model_up.predict(images)
    label = UPPER_LIST
    result.append(label[np.argmax(next_class[0])])
    return result


if __name__ == '__main__':
    directory = './dataset/test0/'
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + '/' + file
            if filepath.endswith(".jpg"):
                print(filepath, 'predict =', predict(filepath))
