import tensorflow as tf
import numpy as np
import os
import random
import time
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None' 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# model path
model_path = 'model/VGG16.h5'

# data
cifar10 = tf.keras.datasets.cifar10
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
# normalized  the data
x_train_n, x_test_n = x_train / 255.0, x_test / 255.0

# label names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# training hyperparameters
batch_size = 32
learning_rate = 0.001
optimizer = 'Adam'
epochs = 40

# structure of the model
model = tf.keras.models.Sequential([
    Conv2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'), # conv1_1
    Conv2D(64, (3, 3), padding='same', activation='relu'), # conv1_2
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool1
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'), # conv2_1
    Conv2D(128, (3, 3), padding='same', activation='relu'), # conv2_2
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool2
    BatchNormalization(),
    Conv2D(256, (3, 3), padding='same', activation='relu'), # conv3_1
    Conv2D(256, (3, 3), padding='same', activation='relu'), # conv3_2
    Conv2D(256, (3, 3), padding='same', activation='relu'), # conv3_3
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool3
    BatchNormalization(),
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv4_1
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv4_2
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv4_3
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool4
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv5_1
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv5_2
    Conv2D(512, (3, 3), padding='same', activation='relu'), # conv5_3
    MaxPooling2D(pool_size=(2, 2), strides=2), # maxpool5
    BatchNormalization(),
    Flatten(), # flatten
    Dense(4096, activation='relu'), # fc1
    Dense(4096, activation='relu'), # fc2
    Dense(10, activation='softmax'), # output (with softmax) 
], name='VGG16')

def show_10_images():
    global x_train, y_train
    plt.figure('Cifar10 Images')
    n = 0
    while n < 10:
        # random number
        i = random.randint(0, len(x_train) - 1)
        plt.subplot(2, 5, n + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i])
        plt.xlabel(class_names[y_train[i][0]])
        n += 1
    plt.show()

def show_param():
    global batch_size, learning_rate, optimizer
    print('hyperparameters:')
    print('batch size:', batch_size)
    print('learning rate:', learning_rate)
    print('optimizer:', optimizer)

def show_summary():
    model.summary()

def train():
    global model, x_train_n, y_train, x_test_n, y_test, batch_size, learning_rate, momentum, optimizer, epochs

    # choose a optimizer
    if optimizer.upper() == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, momentum=momentum)
    elif optimizer.upper() == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    
    # compile the model, prepare to train
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # from_logits=False: output layer is already softmax
        metrics=['accuracy']
    )

    # training
    history_training = model.fit(x_train_n, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test_n, y_test))

    # save model
    model.save_weights(model_path)

    # plot
    training_accuracy = np.array(history_training.history['accuracy']) * 100
    training_loss = history_training.history['loss']
    testing_accuracy = np.array(history_training.history['val_accuracy']) * 100

    plt.figure('Accuracy and Loss')
    plt.subplot(2, 1, 1)
    plt.title('Accuracy')
    plt.plot(training_accuracy, label='Training')
    plt.plot(testing_accuracy, label = 'Testing')
    plt.ylabel('%')
    plt.legend(loc='lower right')
    plt.subplot(2, 1, 2)
    plt.plot(training_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    t = time.localtime()
    plt.savefig('images/history_' + time.strftime("%Y%m%d_%H%M%S", t) + '.png')

def predict(index):
    global model, x_test, x_test_n, y_test, class_names
    if not 0 < index < len(x_test):
        index = 0
    
    # load model
    if os.path.exists(model_path):
        model.build(input_shape=(None, 32, 32, 3))
        model.load_weights(model_path)
    
    # prediction (with softmax)
    prediction = model.predict(x_test_n[index:index+1])[0]

    # Inference
    inference = int(tf.math.argmax(prediction))
    print('Inference: ', class_names[inference])
    print('Answer: ', class_names[y_test[index][0]])

    # plot
    # new label for plot
    label = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[index])
    plt.xlabel('Answer: ' + label[y_test[index][0]])
    plt.subplot(1, 2, 2)
    plt.bar(label, prediction)
    plt.xlabel('Inference: ' + label[inference])
    plt.show()