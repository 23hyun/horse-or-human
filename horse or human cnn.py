import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import os
import PIL
import shutil

model = tf.keras.models.Sequential([
    # The first convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3), strides = (1,2), padding = 'same'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',strides = (1,2), padding = 'same'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),
    tf.keras.layers.MaxPool2D(2, 2),
    # Flatten
    tf.keras.layers.Flatten(),
    # 512 Neuron (Hidden layer)
    tf.keras.layers.Dense(512, activation='relu'),
    # 1 Output neuron
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
#모델 컴파일 하기(손실함수와 옵티마이저 설정)
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.001),
            metrics=['accuracy'])

#이미지 데이터 전처리 하기
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
  'C:/Users/user/Desktop/3/train',
  target_size=(300, 300),
  batch_size=32,
  class_mode='binary')

test_generator = test_datagen.flow_from_directory(
  'C:/Users/user/Desktop/3/test',
  target_size=(300, 300),
  batch_size=32,
  class_mode='binary')


#모델 훈련하기
history = model.fit(
  train_generator,
  steps_per_epoch=8,
  epochs=100,
  validation_steps=5,
  verbose=2
)
model.evaluate(train_generator)
model.evaluate(test_generator)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image

# 테스트 이미지 가져오기
sample_images = ['C:/Users/user/Desktop/3test/{}.jpg'.format(i) for i in range(1, 266)]

for fn in sample_images:
    # matplotlib을 이용하여 이미지 출력
    plt.imshow(mpimg.imread(fn))
    plt.show()

    # Keras에 이미지를 300*300 크기로 불러오기
    img = tf.keras.utils.load_img(fn, target_size=(300, 300))
    # 이미지를 2D 배열로 변환
    x = tf.keras.utils.img_to_array(img)
    print("2D 배열 shape : ", x.shape)
    # 모델의 input_shape가 (300, 300, 3)이므로 이 모양으로 변환
    x = np.expand_dims(x, axis=0)
    print("3D 배열 shape : ", x.shape)

    classes = model.predict(x)

    print("모델 출력 : ", classes[0][0])
    if (classes[0][0] > 0.5):
        print(fn + "는 사람입니다.")
    else:
        print(fn + "는 말입니다.")
