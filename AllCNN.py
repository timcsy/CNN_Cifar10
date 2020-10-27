import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D

class AllCNN(tf.keras.Model):
    # structure of the model
    def __init__(self, input_shape=(None, 32, 32, 3), name='All-CNN-C'):
        super(AllCNN, self).__init__(name=name)
        self.conv1_1 = Conv2D(96, (3, 3), input_shape=input_shape, padding='same', activation='relu')
        self.conv1_2 = Conv2D(96, (3, 3), padding='same', activation='relu')
        self.conv2_1 = Conv2D(96, (3, 3), padding='same', activation='relu', strides=2)
        self.conv3_1 = Conv2D(192, (3, 3), padding='same', activation='relu')
        self.conv3_2 = Conv2D(192, (3, 3), padding='same', activation='relu')
        self.conv4_1 = Conv2D(192, (3, 3), padding='same', activation='relu', strides=2)
        self.conv5_1 = Conv2D(192, (3, 3), padding='valid', activation='relu')
        self.conv5_2 = Conv2D(192, (1, 1), padding='same', activation='relu')
        self.conv5_3 = Conv2D(10, (1, 1), padding='same', activation='relu')
        self.gav = GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = tf.nn.dropout(x, 0.5)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = tf.nn.dropout(x, 0.5)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.gav(x)
        x = tf.nn.softmax(x)
        return x
    
    def model(self, shape=(None, 32, 32, 3)):
        x = tf.keras.Input(shape=shape[1:])
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)

    def summary(self, shape=(None, 32, 32, 3)):
        return self.model(shape).summary()