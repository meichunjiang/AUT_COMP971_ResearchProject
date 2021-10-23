# from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from tensorflow.keras.models import Model
from keras.models import Sequential

model = ResNet50(weights='imagenet')

train, test = tf.keras.datasets.mnist.load_data()

# (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
# (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

print (train_data[0].shape)     # shape is (28,28)

# 利用ResNet50网络进行ImageNet分类

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)



preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


# 输入层
inputs = Input(shape=(224, 224, 3))

# 卷积层和最大池化层
conv1 = Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
conv2 = Conv2D(64, (3,3), padding='same', activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=2)(conv2)

conv3 = Conv2D(128, (3,3), padding='same', activation='relu')(pool1)
conv4 = Conv2D(128, (3,3), padding='same', activation='relu')(conv3)
pool2 = MaxPooling2D(pool_size=2)(conv4)

conv5 = Conv2D(256, (3,3), padding='same', activation='relu')(pool2)
conv6 = Conv2D(256, (3,3), padding='same', activation='relu')(conv5)
conv7 = Conv2D(256, (3,3), padding='same', activation='relu')(conv6)
pool3 = MaxPooling2D(pool_size=2)(conv7)

conv8 = Conv2D(512, (3,3), padding='same', activation='relu')(pool3)
conv9 = Conv2D(512, (3,3), padding='same', activation='relu')(conv8)
conv10 = Conv2D(512, (3,3), padding='same', activation='relu')(conv9)
pool4 = MaxPooling2D(pool_size=2)(conv10)

conv11 = Conv2D(512, (3,3), padding='same', activation='relu')(pool4)
conv12 = Conv2D(512, (3,3), padding='same', activation='relu')(conv11)
conv13 = Conv2D(512, (3,3), padding='same', activation='relu')(conv12)
pool5 = MaxPooling2D(pool_size=2)(conv13)

# 扁平层
flat = Flatten()(pool5)

# 全联接层
fc1 = Dense(4096, activation='relu')(flat)
fc2 = Dense(4096, activation='relu')(fc1)

# 输出层
outputs = Dense(1000, activation='softmax')(fc2)

my_VGG16_model = Model(inputs=inputs, outputs=outputs)
my_VGG16_model.summary()

model = Sequential()
# conv1
model.add(Conv2D(96, (11, 11), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# conv2
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# conv3
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
# conv4
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
# conv5
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# fc5
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.8))
# fc6
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.8))
# fc7
model.add(Dense(10, activation='softmax'))

model.summary()