
import tensorflow as tf
from tensorflow import keras
tf.keras.preprocessing.image.ImageDataGenerator
import numpy as np
import cv2 as cv

import PIL.Image
import pathlib
import matplotlib.pyplot as plt

letter_dict = {'A' : 1,'B' : 2, 'C' :3 ,'D': 4,'E':5 }
TOTAL_EPOC = 30

def prepare_mnist_model():
    print("prepare model for handwritten digit recognition")
    digits_mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()
    
    print("================================================")
    print(" MNIST data loaded and split into train and test sets\n")
    print(" Train Set : shape = ",train_images.shape)
    print(" Test  Set : shape = ",test_images.shape)
    print("================================================")
    return train_images,train_labels,test_images,test_labels
  
def plot_digit(digit):
    plt.figure()
    plt.imshow(digit)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def plot_digits(digits,labels):
    plt.figure(figsize=(10,10))
    total = 25
    if (len(digits)< 25):
       total = len(digits)

    for i in range(total):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(digits[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
     
    plt.show()        

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
       color = 'blue'
    else:
       color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(true_label,
                                   100*np.max(predictions_array),
                                   true_label),
                                   color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

from __future__ import print_function
import datetime
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D
from keras import backend as K
now = datetime.datetime.now
batch_size = 128
num_classes = 5
epochs = 5
img_rows, img_cols = 28, 28
filters = 128
pool_size = 2
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def build_model(train_images,train_labels,test_images,test_labels):

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    feature_layers = [
        Conv2D(filters, kernel_size,
             padding='valid',
           input_shape=input_shape),
        Activation('relu'),
        Conv2D(filters, kernel_size),
        Activation('relu'),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.25),
        Flatten(),
    ]

    classification_layers = [
         Dense(128),
         Activation('relu'),
         Dropout(0.5),
         Dense(num_classes),
         Activation('softmax')
    ]



#    model = tf.keras.Sequential([
#              tf.keras.layers.Flatten(input_shape=(28, 28)),
 #             tf.keras.layers.Dense(128, activation='relu'),
  #            tf.keras.layers.Dense(10)
   #         ])
    model = Sequential(feature_layers + classification_layers)
    #model.load_weights('mnist_model')
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history= model.fit(train_images, train_labels, validation_split=0.2,epochs=TOTAL_EPOC)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    return model,predictions,history

def plot_predictions(test_images,test_labels,predictions):

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
    plt.show()

def load_hw_letters(path):
    data_dir = pathlib.Path(path)
    images = [ cv.imread(str(images),0) for images in list(data_dir.glob('*')) ]
    labels = [ str(images)[len(path)+1] for images in list(data_dir.glob('*')) ]  
    #images = []
    #labels = []     

    #for image in list(data_dir.glob('*')):
    #    letter_c = str(image)
    #    image = load_img(letter_c, target_size=(28, 28))
     #   image = img_to_array(image)        
      #  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
       # image = preprocess_input(image)
        #images.append(image)
        #labels.append(letter_c[len(path)+1])
    return images,labels 

def model_performance(title,history,train,train_labels,test,test_labels):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle(title, fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,TOTAL_EPOC))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, TOTAL_EPOC, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, TOTAL_EPOC, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")

def transfer_lightweight_model(base_model,derived_model,hw_train_test):

    base_model=mnist_model
    base_model.trainable = False
    base_model.summary()

    inputs = tf.keras.Input(shape=(150, 150,3))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(),
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.fit(hw_train_test[0]/255.0, epochs=30, validation_data=hw_train_test[2]/255.0)
    
    history= model.fit(hw_train_test[0]/255.0, hw_train_test[1], validation_split=0.2,epochs=TOTAL_EPOC)


def main():
    print("Welcome Tensorflow version ",tf.__version__)
    train_images,train_labels,test_images,test_labels = prepare_mnist_model()
    plot_digit(train_images[3])
    plot_digits(train_images,train_labels)
    mnist_model,predictions,history = build_model(train_images,train_labels,test_images,test_labels)
    model_performance("MNIST Model Performance",history,train_images,train_labels,test_images,test_labels)
    mnist_model.save_weights('mnist_model')
    history.history.keys()
    mnist_model.summary()
    plot_predictions(test_images,test_labels,predictions)
    
    train_hw_images,train_hw_labels = load_hw_letters('s_data/s_train')
    test_hw_images,test_hw_labels   = load_hw_letters('s_data/s_test')
    
   
    train_hw_images = np.array(train_hw_images)
    test_hw_images  = np.array(test_hw_images)
    
    train_hw_lab_num = np.array([ letter_dict[l] for l in train_hw_labels])
    test_hw_lab_num = np.array([ letter_dict[l] for l in test_hw_labels])
    
    print('Letters : Train set')
    plot_digits(train_hw_images-255,train_hw_labels)
    print('Letters : Test set')
    plot_digits(test_hw_images-255,test_hw_labels)

    letters_model,l_predictions,history = build_model(train_hw_images-255,train_hw_lab_num,test_hw_images-255,test_hw_lab_num)
    letters_model.summary()
    plot_predictions(test_hw_images,test_hw_lab_num,l_predictions)
    letters_model.summary()
    model_performance("HW Letters Model Performance",history,train_hw_images,train_hw_lab_num,test_hw_images,test_hw_lab_num)
    hw_train_test = [train_hw_images,train_hw_lab_num,test_hw_images,test_hw_lab_num]
    transfer_model(mnist_model,letters_model,hw_train_test)
    
    mnist_model.trainable = False
    
    mnist_model.load_weights('mnist_model')
mnist_model.summary()
    letters_model

    image = train_hw_images[0]
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    yhat  = mnist_model.predict(train_hw_images[0])

# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
# load an image from file
image = load_img('/Users/sachin/Downloads/dog.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array
image = img_to_array(image)
image.shape
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)

if __name__ == "__main__":
    main()
