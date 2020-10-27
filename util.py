import tensorflow as tf
import VGG16
import numpy as np
import os
import random
import time
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None' 
import matplotlib.pyplot as plt

# model path
model_path = 'model.h5'

# data
cifar10 = tf.keras.datasets.cifar10
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
# Eliminate an unused dimension
y_train, y_test = np.reshape(y_train, y_train.shape[:1]), np.reshape(y_test, y_test.shape[:1])

# label names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# training hyperparameters
batch_size = 32
learning_rate = 0.001
optimizer = 'Adam'
epochs = 20

# structure of the model
model = tf.keras.models.Sequential([
    # tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(32, 32, 3)),
    VGG16.Model(input_shape=(32, 32, 3)),
    tf.keras.layers.Flatten(), # flatten
    tf.keras.layers.Dense(4096, activation='relu'), # FC1
    tf.keras.layers.Dense(4096, activation='relu'), # FC2
    tf.keras.layers.Dense(10), # output layer (cannot use softmax here, or hard to BP)
])
model.build((1, 32, 32, 3))

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
        plt.xlabel(class_names[y_train[i]])
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
    global model, x_train, y_train, x_test, y_test, batch_size, learning_rate, optimizer, epochs

    # choose a optimizer
    if optimizer.upper() == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.upper() == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # compile the model, prepare to train
    model.compile(
        optimizer=opt,
        #optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # normalized  the data
    x_train_n, x_test_n = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train, y_test

    # resize the image
    x_train_n = tf.image.resize(x_train_n, [32, 32])
    x_test_n = tf.image.resize(x_test_n, [32, 32])

    # training
    history_training = model.fit(x_train_n, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test_n, y_test))

    # save model
    model.save(model_path)

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
    plt.savefig('history_' + time.strftime("%Y%m%d_%H%M%S", t) + '.png')

def predict(index):
    global model, x_test, y_test, class_names
    if not 0 < index < len(x_test):
        index = 0
    
    # load model
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    
    # prediction (with softmax)
    prediction = tf.nn.softmax(model.predict(x_test[index:index+1]))[0]

    # Inference
    inference = int(tf.math.argmax(prediction))
    print('Inference: ', class_names[inference])
    print('Answer: ', class_names[y_test[index]])

    # plot
    # new label for plot
    label = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[index])
    plt.xlabel('Answer: ' + label[y_test[index]])
    plt.subplot(1, 2, 2)
    plt.bar(label, prediction)
    plt.xlabel('Inference: ' + label[inference])
    plt.show()