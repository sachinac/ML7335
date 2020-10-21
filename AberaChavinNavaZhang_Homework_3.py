from __future__ import print_function
import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import pandas as pd

print('='*80)
print('Group Member Names:')        
print('='*80)
print('Tazeb Abera')
print('Sachin Chavin')
print('Yang Zhang')
print('Christian Nava')
print("\n")

now = datetime.datetime.now
batch_size = 128
num_classes = 5
epochs = 5
img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]
x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5



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

# create complete model
model = Sequential(feature_layers + classification_layers)

# train model for 5-digit classification [0..4]
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

# transfer: train dense layers for new classification task [5..9]
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

x_train.shape
x_test.shape
y_train
y_test.shape
input_shape
X_train_plot = x_train.reshape(-1, 28, 28)

import matplotlib.pyplot as plt
import seaborn as sns

# Check example images from MNIST dataset
def Show_example_digits(mono = 'gray'):
    fig = plt.figure(figsize = (16, 16))
    for idx in range(15):
        plt.subplot(5, 5,idx+1)
        plt.imshow(X_train_plot[idx], cmap = mono)
        plt.title("Digit {}".format(y_train[idx]))
        
    plt.tight_layout()

print('='*80)
print('Example Images from MNIST dataset:')        
print('='*80)
print("\n")
Show_example_digits()

# Each image is a 28x28 matrix where each value in the matrix corresponds 
# to the grayscale value from 0 to 255.
def plot_digit(digit, dem = 28, font_size = 12):
    max_ax = font_size * dem
    
    fig = plt.figure(figsize=(13, 13))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')
    black = '#000000'
    
    for idx in range(dem):
        for jdx in range(dem):

            t = plt.text(idx * font_size, max_ax - jdx*font_size, digit[jdx][idx], fontsize = font_size, color = black)
            c = digit[jdx][idx] / 255.
            t.set_bbox(dict(facecolor=(c, c, c), alpha = 0.5, edgecolor = 'black'))
            
    plt.show()

np.random.seed(0) 
import random

################################################
###   Sample Digit Image and Matrix Values   ###
################################################

print('='*80)
print('Sample digit matrix values:')        
print('='*80)
print("\n")
plot_digit(X_train_plot[1,:,:])

# Here, we plot of the distribution of each digit
new_series = pd.Series(y_train)
new_series.value_counts().index
new_series

digit_range = np.arange(10)

val = new_series.value_counts().index
cnt = new_series.value_counts().values
mycolors = ['red', 'blue', 'green', 'orange', 'brown', 'grey', 'pink', 'olive', 'deeppink', 'steelblue']

print('='*80)
print('Plot of the distribution of each digit:')        
print('='*80)
print("\n")

# Plot of the distribution of each digit
plt.figure(figsize = (15, 7))
plt.title("The number of digits in the data", fontsize = 20)
plt.xticks(range(10))
plt.bar(val, cnt, color = mycolors);


model.summary()


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train[0].shape


train=x_train
test=x_test

# Define the training model
num_classes = 10
def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))
    y_pred = model.predict(x_test)
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return y_pred


(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

# create complete model
model = Sequential(feature_layers + classification_layers)


yy = train_model(model,(x_train, y_train),(x_test, y_test), num_classes)


def draw_output(idx_nums):
    plt.figure(figsize = (20, 20))
    plt.xticks( range(10) )
    x = np.ceil(np.sqrt(len(idx_nums)))
    cnt = 1
    for ph in idx_nums:
        plt.subplot(x, x, cnt)
        curr_photo = y_test[ph]
        
        plt.xlim(0, 10)
        plt.title("Digit: {0}\n idx: {1} ".format(np.argmax(y_test[ph]), ph), fontsize = 10) 
        plt.bar(range(10), yy[ph])
        
        cnt += 1

yy[:,1]
y_test

################################################
###     Predict the error for each digit     ###
################################################

print('='*80)
print('The chart below shows the predicted error for each digit')        
print('='*80)
print("\n")

cnt_error = []
for idx, (a, b) in enumerate(zip(y_test, yy)):
    #if np.argmax(a) == np.argmax(b): continue
    #cnt_error.append( (np.argmax(a)) )
    if a == np.argmax(b): continue
    cnt_error.append( (a) )

cnt_error = np.unique(cnt_error, return_counts = True)
sns.set_style("darkgrid")
plt.figure(figsize = (15, 7))
bar_plot = sns.barplot(cnt_error[0], cnt_error[1], palette="muted")
plt.show()


################################################
###      Import Our Handwritten Dataset      ###
################################################

print('='*80)
print('Import our handwritten dataset')        
print('='*80)
print('The dataset consists of 10 handwritten samples for the letters A-E')        
print('for a total of 50 observations.')        

print("\n")

#import cv2
from keras.preprocessing import image




################################################
###           Create Training Set            ###
################################################
from PIL import Image
import urllib.request
import io

train=pd.read_csv("https://raw.githubusercontent.com/sachinac/ML7335/main/train.csv", header=None)
train_path="https://github.com/sachinac/ML7335/tree/main/train"


train_img=[]
for i in range(len(train)):

    f = io.BytesIO(urllib.request.urlopen(train_path+train[0][i]).read())

    temp_img = Image.open(f)

    temp_img=image.img_to_array(temp_img)

    train_img.append(temp_img)

train_img=np.array(train_img)
train_img.shape

################################################
###             Create Test Set              ###
################################################

test=pd.read_csv("https://raw.githubusercontent.com/sachinac/ML7335/main/test.csv",header=None)
test_path="https://raw.githubusercontent.com/sachinac/ML7335/main/test"

test_img=[]
for i in range(len(test)):

    f = io.BytesIO(urllib.request.urlopen(test_path+test[0][i]).read())

    temp_img = Image.open(f)

    temp_img=image.img_to_array(temp_img)

    test_img.append(temp_img)

# Compress from 3 channels to 1 channel.
train_img = train_img[:,:,:,0]
train_img.shape

test_img = test_img[:,:,:,0]
test_img.shape

# Manually set up labels
train_label_all = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4]
test_label_all = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]


print('='*80)
print('Below is a sample of images from our handwritten dataset:')        
print('='*80)
print("\n")


def Show_example_digits(mono = 'gray'):
    fig = plt.figure(figsize = (16, 16))
    for idx in range(35):
        plt.subplot(5, 7,idx+1)
        plt.imshow(train_img[idx], cmap = mono)
        #plt.imshow(X_image_plot[idx], cmap = mono)
        plt.title("Digit {}".format(train_label_all[idx]))
        
    plt.tight_layout()
    
Show_example_digits()

# Convert training labels from list to array
train_label_all = np.asarray(train_label_all)
train_label_all.shape

# Convert test labels from list to array
test_label_all = np.asarray(test_label_all)
test_label_all.shape

# check input shape
#input_shape


##########################
##      Train Model     ##
##########################

train_model(model,
            (train_img, train_label_all),
            (test_img, test_label_all), num_classes)

# train model for 5-digit classification [0..4]
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

train_model(model,
            (train_img, train_label_all),
            (test_img, test_label_all), num_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = True

train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False
    
train_model(model,
            (train_img, train_label_all),
            (test_img, test_label_all), num_classes)


# Predition is made for the test_img (a total of 15)

y_pred = model.predict(test_img.reshape(15,28,28,1))


#################
##   Results   ##
#################

print('='*80)
print('How many images did our model correctly classify?')        
print('='*80)
print('The plot below shows how many handwritten letters for each class were classified')        
print('incorrectly. The plot shows that for letters A, C, and D, the model correctly')        
print('identified 2 out of the 3 images and did not correctly identify any of the the')        
print('images for letters B or E.')        
print("\n")

# Error shows the accuracy rate of A, C and D are 66.7%, B and E are 0%
cnt_error = []
for idx, (a, b) in enumerate(zip(test_label_all,y_pred)):
    #if np.argmax(a) == np.argmax(b): continue
    #cnt_error.append( (np.argmax(a)) )
    if a == np.argmax(b): continue
    cnt_error.append( (a) )

cnt_error = np.unique(cnt_error, return_counts = True)
sns.set_style("darkgrid")
plt.figure(figsize = (15, 7))
bar_plot = sns.barplot(cnt_error[0], cnt_error[1], palette="muted")
plt.show()


print('='*80)
print('Our model incorrectly predicted the following images:')        
print('='*80)
print("\n")

# A's incorrectly predicted
cnt_ind = 1
list_idx = []
fig = plt.figure(figsize=(14, 14))
for idx, (a, b) in enumerate(zip(test_label_all,y_pred)):
    if a == np.argmax(b): continue
    if (a == 0):    
        plt.subplot(5, 5, cnt_ind)
        plt.imshow(test_img[idx], cmap='gray', interpolation='none')
        plt.title('y_true={0}\ny_pred={1}\n ind = {2}'.format(a, np.argmax(b), idx))
        plt.tight_layout()
        list_idx.append(idx)
        cnt_ind += 1

# B's incorrectly predicted
cnt_ind = 1
list_idx = []
fig = plt.figure(figsize=(14, 14))
for idx, (a, b) in enumerate(zip(test_label_all,y_pred)):
    if a == np.argmax(b): continue
    if (a == 1):    
        plt.subplot(5, 5, cnt_ind)
        plt.imshow(test_img[idx], cmap='gray', interpolation='none')
        plt.title('y_true={0}\ny_pred={1}\n ind = {2}'.format(a, np.argmax(b), idx))
        plt.tight_layout()
        list_idx.append(idx)
        cnt_ind += 1

# C's incorrectly predicted
cnt_ind = 1
list_idx = []
fig = plt.figure(figsize=(14, 14))
for idx, (a, b) in enumerate(zip(test_label_all,y_pred)):
    if a == np.argmax(b): continue
    if (a == 2):    
        plt.subplot(5, 5, cnt_ind)
        plt.imshow(test_img[idx], cmap='gray', interpolation='none')
        plt.title('y_true={0}\ny_pred={1}\n ind = {2}'.format(a, np.argmax(b), idx))
        plt.tight_layout()
        list_idx.append(idx)
        cnt_ind += 1

# D's incorrectly predicted
cnt_ind = 1
list_idx = []
fig = plt.figure(figsize=(14, 14))
for idx, (a, b) in enumerate(zip(test_label_all,y_pred)):
    if a == np.argmax(b): continue
    if (a == 3):    
        plt.subplot(5, 5, cnt_ind)
        plt.imshow(test_img[idx], cmap='gray', interpolation='none')
        plt.title('y_true={0}\ny_pred={1}\n ind = {2}'.format(a, np.argmax(b), idx))
        plt.tight_layout()
        list_idx.append(idx)
        cnt_ind += 1

# E's incorrectly predicted
cnt_ind = 1
list_idx = []
fig = plt.figure(figsize=(14, 14))
for idx, (a, b) in enumerate(zip(test_label_all,y_pred)):
    if a == np.argmax(b): continue
    if (a == 4):    
        plt.subplot(5, 5, cnt_ind)
        plt.imshow(test_img[idx], cmap='gray', interpolation='none')
        plt.title('y_true={0}\ny_pred={1}\n ind = {2}'.format(a, np.argmax(b), idx))
        plt.tight_layout()
        list_idx.append(idx)
        cnt_ind += 1




####################
##   Refinement   ##
####################

print('='*80)
print('Can we increase the accuracy of the model?')        
print('='*80)
print('We increased the epochs from 5 to 20 to see if that would affect the acccuracy.')        
print('After increasing the epochs to 20, the plot below shows that the model now')        
print('correctly identified all images of the letter A and did not correctly identify')        
print('any of the images for letter B. It also only correctly identified 1 of the images')        
print('for both the letters C and E and two images for the letter D.')        
print("\n")


epochs = 20

# Run model again.
# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = True

train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False
    
train_model(model,
            (train_img, train_label_all),
            (test_img, test_label_all), num_classes)


train_model(model,
            (train_img, train_label_all),
            (test_img, test_label_all), num_classes)

y_pred = model.predict(test_img.reshape(15,28,28,1))

y_pred



cnt_error = []
for idx, (a, b) in enumerate(zip(test_label_all,y_pred)):
    #if np.argmax(a) == np.argmax(b): continue
    #cnt_error.append( (np.argmax(a)) )
    if a == np.argmax(b): continue
    cnt_error.append( (a) )

cnt_error = np.unique(cnt_error, return_counts = True)
sns.set_style("darkgrid")
plt.figure(figsize = (15, 7))
bar_plot = sns.barplot(cnt_error[0], cnt_error[1], palette="muted")
plt.show()



print('='*80)
print('What are the main takeaways from this experiment?')        
print('='*80)
print('We believe our dataset is probably too small and/or there is too much variation')        
print('in the sample images. An alternative would be to try LO-shot learning where we')        
print('could use fewer images that capture more information, that is, we could tell the')        
print('the model that, for example, the image for a letter B is 80% B and 20% D,')        
print('which could help in improving the accuracy of the model.')        
print("\n")