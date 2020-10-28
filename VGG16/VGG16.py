import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class Model(tf.keras.Model):
    # structure of the model
    def __init__(self, input_shape=(1, 224, 224, 3), name='vgg16'):
        super(Model, self).__init__(name = name)
        self.conv1_1 = Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu')
        self.conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')
        self.conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu')
        self.conv3_3 = Conv2D(256, (3, 3), padding='same', activation='relu')
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv4_3 = Conv2D(512, (3, 3), padding='same', activation='relu')
        self.maxpool4 = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv5_2 = Conv2D(512, (3, 3), padding='same', activation='relu')
        self.conv5_3 = Conv2D(512, (3, 3), padding='same', activation='relu')
        self.maxpool5 = MaxPooling2D(pool_size=(2, 2), strides=2)

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.maxpool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.maxpool5(x)
        return x