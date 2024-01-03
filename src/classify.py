import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src import constant

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Model:
    def __init__(self):
        self.model = None

    def buildModel(self):
        with tf.device('/device:GPU:0'):
            self.model = Sequential()

            self.model.add(
                Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)))
            self.model.add(MaxPooling2D(pool_size=2))

            self.model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
            self.model.add(MaxPooling2D(pool_size=2))

            self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
            self.model.add(MaxPooling2D(pool_size=2))

            self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
            self.model.add(MaxPooling2D(pool_size=2))

            self.model.add(Flatten())  # 扁平化参数
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(9, activation='softmax'))
            self.model.summary()

    def trainModel(self):
        # 优化器, 主要有Adam、sgd、rmsprop等方式。
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        with tf.device('/device:GPU:0'):

            self.model.compile(loss='categorical_crossentropy',
                               optimizer=tf.compat.v1.train.AdamOptimizer(),  # 使用 TensorFlow 的 Adam 优化器
                               metrics=['accuracy'])
            # 自动扩充训练样本

            train_datagen = ImageDataGenerator(
                rescale=1. / 255,  # 数据缩放，把像素点的值除以255，使之在0到1之间
                shear_range=0.1,  # 错切变换角度
                zoom_range=0.1,  # 随机缩放范围
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                validation_split=0.1
            )
            # 生成验证集

            val_datagen = ImageDataGenerator(
                rescale=1. / 255, validation_split=0.1)

            # 以文件分类名划分label
            train_generator = train_datagen.flow_from_directory(
                '../dataset-resized',
                # 整数元组 (height, width)，默认：(300, 300)。 所有的图像将被调整到的尺寸。
                target_size=(300, 300),
                # 一批数据的大小
                batch_size=32,
                # "categorical", "binary", "sparse", "input" 或 None 之一。
                # 默认："categorical",返回one-hot 编码标签。
                class_mode='categorical',
                subset='training',
                seed=0)
            val_generator = val_datagen.flow_from_directory(
                '../dataset-resized',
                target_size=(300, 300),
                batch_size=32,
                class_mode='categorical',
                subset='validation',
                seed=0)
            # 编译模型
            try:
                history_fit = self.model.fit(train_generator,
                                             epochs=50,
                                             steps_per_epoch=9032 // 32,  # 训练集
                                             validation_data=val_generator,
                                             validation_steps=833 // 32)  # 测试集
                with open("../model/history_fit.json", "w") as json_file:
                    json_file.write(str(history_fit))

                acc = history_fit.history['accuracy']
                val_acc = history_fit.history['val_accuracy']
                loss = history_fit.history['loss']
                val_loss = history_fit.history['val_loss']

                epochs = range(1, len(acc) + 1)
                plt.figure("acc")
                plt.plot(epochs, acc, 'r-', label='Training acc')
                plt.plot(epochs, val_acc, 'b', label='validation acc')
                plt.title('The comparison of train_acc and val_acc')
                plt.legend()
                plt.show()

                plt.figure("loss")
                plt.plot(epochs, loss, 'r-', label='Training loss')
                plt.plot(epochs, val_loss, 'b', label='validation loss')
                plt.title('The comparison of train_loss and val_loss')
                plt.legend()
                plt.show()
            except StopIteration:
                pass

    def saveModel(self):
        model_json = self.model.to_json()
        with open('../model/model_json.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights('../model/model_weight.h5')
        self.model.save('../model/model.h5')
        print('model saved')

    def loadModel(self):
        json_file = open('../model/model_json.json')  # 加载模型结构文件
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)  # 结构文件转化为模型
        # 加载权重
        model.load_weights('../model/model_weight.h5')  # h5文件保存模型的权重数据
        return model

    def retrainModel(self, epochs=50):
        # 加载已存在的模型
        if os.path.exists("../model/model.h5"):
            self.model = self.loadModel()
            print('已加载现有模型以进行再训练')
        else:
            print('未找到现有模型。正在构建新模型。')
            self.buildModel()

        # 继续训练模型
        if self.model is not None:  # 确保成功加载了模型
            with tf.device('/device:GPU:0'):
                self.model.compile(loss='categorical_crossentropy',
                                   optimizer=tf.compat.v1.train.AdamOptimizer(),
                                   metrics=['accuracy'])
                train_datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    shear_range=0.1,
                    zoom_range=0.1,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    validation_split=0.1
                )

                val_datagen = ImageDataGenerator(
                    rescale=1. / 255, validation_split=0.1)

                train_generator = train_datagen.flow_from_directory(
                    '../dataset-resized',
                    target_size=(300, 300),
                    batch_size=32,
                    class_mode='categorical',
                    subset='training',
                    seed=0)
                val_generator = val_datagen.flow_from_directory(
                    '../dataset-resized',
                    target_size=(300, 300),
                    batch_size=32,
                    class_mode='categorical',
                    subset='validation',
                    seed=0)

                try:
                    history_fit = self.model.fit(train_generator,
                                                 epochs=epochs,
                                                 steps_per_epoch=9032 // 32,
                                                 validation_data=val_generator,
                                                 validation_steps=833 // 32)
                    # 在重新训练后保存更新后的模型
                    self.saveModel()
                except StopIteration:
                    pass


def generate_result(result):
    for i in range(9):
        if result[0][i] == 1:
            return constant.labels[i]


if __name__ == '__main__':
    model = Model()
    if os.path.exists("../model/model.h5"):
        print('Model loaded')
        model.retrainModel(epochs=50)
    else:
        model.buildModel()
        print('Model built')
        model.trainModel()
        print('Model trained')
        model.saveModel()
        print('Model saved')