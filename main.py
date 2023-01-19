import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf

warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
from keras import backend as K

K.image_data_format()
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from time import time

PATH = os.getcwd()
# Data pathi verilir.
data_path = 'dataset'
data_dir_list = os.listdir(data_path)
print("Sınıflar : ", data_dir_list),
img_rows = 128
img_cols = 128
num_channel = 1
num_epoch = 100
# Sınıf sayısı tanımlanır.
num_classes = 7
img_data_list = []
print("Veri seti yükleniyor...")

for dataset in data_dir_list:
    img_list = os.listdir(data_path + "/" + dataset)
    print('Loaded the images of dataset - ' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (128, 128))  # resimleri gri tonlamaya çevirir ve boyutlandırır.
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)  # nesnenin giriş şekli belirlenir. 1803 resim en=128, boy=128

if num_channel == 1:
    if K.image_data_format() == 'th':
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis=3)
        print(img_data.shape)
else:
    if K.image_data_format() == 'th':
        img_data = np.rollaxis(img_data, 3, 1)
        print(img_data.shape)

num_classes = 7
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:698] = 0
labels[365:567] = 1
labels[567:987] = 2
labels[987:1189] = 3
labels[1189:1399] = 4
labels[1399:1601] = 5
labels[1601:1803] = 6

names = ['dew', 'frogsmog', 'frost', 'dogs', 'flowers', 'horses', 'human']

Y = np_utils.to_categorical(labels,
                            num_classes)  # veri kümesi karıştırılıyor ve eğitim ve test veri kümesi oluşturuluyor
x, y = shuffle(img_data, Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print("X_train shape = {}".format(X_train.shape))
print("X_test shape = {}".format(X_test.shape))
image = X_train[1204, :].reshape((128, 128))  # 1204. resim gösterilir.
plt.imshow(image)
plt.show()

# Giriş şeklinin başlatılması
input_shape = img_data[0].shape

# CNN modelini tasarlandı
cnn_model = Sequential([
    Convolution2D(32, 3, 3, padding='same', activation='relu', input_shape=input_shape),
    Convolution2D(32, 3, 3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
start_time = time()
# modeli derleme
cnn_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])

cnn_model.summary()

layer_names = [layer.name for layer in cnn_model.layers]
print("Katman isimleri : l", layer_names)
hist = cnn_model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(num_epoch)

# loss grafiği
plt.figure(1, figsize=(10, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train Loss', 'Validation Loss'])
plt.style.use(['classic'])
plt.show()

# accuracy grafiği
plt.figure(2, figsize=(10, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc=4)
plt.style.use(['classic'])
plt.show()

score = cnn_model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

test_image = X_test[0:1]
print(test_image.shape)
print(cnn_model.predict(test_image))
print(np.argmax(cnn_model.predict(test_image), axis=1))
print(y_test[0:1])

image = test_image.reshape((128, 128))
plt.imshow(image)
plt.show()

# Yeni bir resim ile test
test_img = cv2.imread("data/cats/cat.41.jpg")
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_img = cv2.resize(test_img, (128, 128))
test_img = np.array(test_img)
test_img = test_img.astype('float32')
test_img /= 255
print("Test image Shape : ", test_img.shape)

image = test_img.reshape((128, 128))
plt.imshow(image)
plt.show()

if num_channel == 1:
    if K.image_data_format == 'th':
        test_img = np.expand_dims(test_img, axis=0)
        test_img = np.expand_dims(test_img, axis=0)
        print("1. dongu : ", test_img.shape)
    else:
        test_img = np.expand_dims(test_img, axis=2)  # 3
        test_img = np.expand_dims(test_img, axis=0)
        print("2. dongu : ", test_img.shape)

else:
    if K.image_data_format() == 'th':
        test_img = np.rollaxis(test_img, 2, 0)
        test_img = np.expand_dims(test_img, axis=0)
        print("3. dongu : ", test_img.shape)
    else:
        test_img = np.expand_dims(test_img, axis=0)
        print("4. dongu : ", test_img.shape)

# Test görüntüsünü tahmin etme
print("model cnn", (cnn_model.predict(test_img)))
print("np argmax ", np.argmax(cnn_model.predict(test_img), axis=1))

layer_num = 3
filter_num = 0

test_image = x[0]
test_image_show = test_image[:, :, 0]
plt.axis('off')
test_image = np.expand_dims(test_image, axis=0)

feature_map_model = tf.keras.models.Model(inputs=cnn_model.input, outputs=cnn_model.layers[layer_num].output)
feature_maps = feature_map_model.predict(test_image)

print("feature maps shape = {}".format(feature_maps.shape))

if K.image_data_format() == 'th':
    feature_maps = np.rollaxis((np.rollaxis(feature_maps, 2, 0)), 2, 0)
print(feature_maps.shape)

for layer_name, feature_map in zip(layer_names, feature_maps):
    print(f"The shape of the {layer_name} is =======> {feature_map.shape}")

dx, dy = 0.05, 0.05
x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)
extent = np.min(x), np.max(x), np.min(y), np.max(y)
fig = plt.figure(figsize=(16, 16))
Z1 = np.add.outer(range(8), range(8)) % 2
plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest', extent=extent)
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num) + '.jpg')
num_of_featuremaps = feature_maps.shape[2]
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num = int(np.ceil(np.sqrt(num_of_featuremaps)))
print("subplot_num = {}".format(subplot_num))
for i in range(int(num_of_featuremaps)):
    plt.subplot(subplot_num, subplot_num, i + 1)
    plt.axis('off')
    plt.imshow(feature_maps[0, :, :, i], interpolation="nearest", cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num) + '.jpg')
plt.show()

# Hata matrisini ekrana çiz (confusion matrix)
Y_pred = cnn_model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
target_names = ['Class 0 (flowers)', 'Class 1 (cars)', 'Class 2 (cats)', 'Class 3 (horses)',
                'Class 4 (human)', 'Class 5 (bike)', 'Class 6 (dogs)']
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))

print('Confusion Matrix \n')
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix with Normalization")
    else:
        print('Confusion matrix without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))
np.set_printoptions(precision=2)
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion Matrix without Normalisation')
plt.show()

plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='Normalized Confusion Matrix')
plt.figure()
plt.show()
goal_time = time() - start_time
print("Öğrenme", goal_time, "saniye sürmüştür.")
