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
lower_upper_tags = {}


def generate_lower_upper_results():
    with open(f"dataset/Eval/list_eval_partition.txt", "r") as fi:
        lines = fi.readlines()
        content_lines = lines[2:]
        for line in content_lines:
            path, item_id, status = line.strip().split()
            path_parts = path.split("/")
            img_filename = f"{item_id}_{path_parts[-1]}"
            lower_upper_tags[img_filename] = [path_parts[2]]


def process(filename):
    image = mpimg.imread(filename)
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.show()


def test_lower(testpath, label, model_dir):
    model_low = load_model(model_dir)

    count_total_test = 0
    count_correct_test = 0

    for (dirpath, dirnames, filenames) in os.walk(testpath):
        for filename in filenames:
            path = os.path.join(dirpath, *dirnames, filename)
            img = image.load_img(path, target_size=(128, 128))

            x = image.img_to_array(img)           # x.shape (128, 128, 3)
            x = np.expand_dims(x, axis=0)       # x.shape (1, 128, 128, 3)
            images = np.vstack([x])        # images.shape (1, 128, 128, 3)
            next_class = model_low.predict(images)
            print(filename, label[np.argmax(next_class[0])], lower_upper_tags[filename])
            count_total_test += 1
            count_correct_test += label[np.argmax(next_class[0])] in lower_upper_tags[filename]

    print(count_correct_test, count_total_test, count_correct_test/count_total_test)


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
    generate_lower_upper_results()
    # test_lower("dataset_lower/gallery/", LOWER_LIST, './model2.h5')
    test_lower("dataset_upper/gallery/", UPPER_LIST, './model3.h5')
    # directory = './dataset/test/'
    # for subdir, dirs, files in os.walk(directory):
    #     for file in files:
    #         filepath = subdir + '/' + file
    #         if filepath.endswith(".jpg"):
    #             print(filepath, 'predict =', predict(filepath))
