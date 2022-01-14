# Importing the Keras libraries and packages
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras import optimizers

# Importing other necessary libraries
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import h5py, os, itertools, heapq

from plot_model import plot_confusion_matrix

# Declaring shape of input images and number of categories to classify
classes = ['lower', 'upper', 'lower_upper']
input_shape = (128, 128, 3)

model = Sequential()

# convolution layer 1, 2
model.add(Convolution2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# convolution layer 3, 4
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# convolution layer 5, 6
model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='sigmoid'))

# model.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.01))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

# Viewing model_configuration
model.summary()

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 40,
    zoom_range = 0.2,
    horizontal_flip = True
)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    './dataset/train_',
    target_size = (128, 128),
    batch_size = 32
)
test_set = test_datagen.flow_from_directory(
    './dataset/query',
    target_size = (128, 128),
    batch_size = 32
)

# Setting callbacks parameters
checkpointer = ModelCheckpoint(filepath='model_upper_lower.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
filename='model1_upper_lower.csv'
csv_log = CSVLogger(filename, separator=',', append=False)

# Training the model
hist = model.fit(
    training_set,
    steps_per_epoch = (5315//32),
    epochs = 150,
    validation_data = test_set,
    validation_steps = (592//32),
    workers = 4,
    callbacks = [csv_log, checkpointer]
)
# plot_loss_accuracy_curves(hist)
model.save('model1.h5')

test_path = 'dataset/test'
test_batches = ImageDataGenerator().flow_from_directory(
    test_path, target_size = (128, 128),
    classes=classes,
    batch_size = 180
)

test_imgs, test_labels = next(test_batches)
batch_pred = model.predict_generator(test_batches, steps=1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(batch_pred, axis=1))
np.set_printoptions(precision=1)
print(cnf_matrix)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=classes,
    title='Confusion matrix, without normalization'
)
print(classification_report(np.argmax(test_labels, axis=1), np.argmax(batch_pred, axis=1), target_names=classes))

print(training_set.class_indices)
print(test_batches.class_indices)

'''
# img = image.load_img('./dataset/test/lower/Acid_Wash_-_Skinny_Jeans_img_00000019.jpg', target_size=(128, 128))
img = image.load_img('./dataset/train/upper/Abstract_Print_Peasant_Blouse_img_00000060.jpg', target_size=(128, 128))

label=['lower', 'upper']

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = model.predict_classes(images)
plt.figure(figsize=(5, 5))
plt.imshow(img)
# plots(images)  
print('Predict below item is a: ', label[classes[0][0]])

from keras.models import load_model
loaded_model = load_model('./model_upper_lower.118-0.07.hdf5')

img = image.load_img('./dataset/upper/gallery/id_00000001_02_1_front.jpg', target_size=(128, 128))

label = ['lower', 'upper']

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = loaded_model.predict_classes(images)
plt.figure(figsize=(5, 5))
plt.imshow(img)
# plots(images)  
print('Predict below item is a : ', label[classes[0][0]])
'''