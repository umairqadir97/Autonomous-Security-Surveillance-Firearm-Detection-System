# -*- coding: utf-8 -*-


#importing Keras, Library for deep learning 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array

import numpy as np
import cv2
# Image manipulations and arranging data
import os
from PIL import Image
import theano
theano.config.optimizer="None"
#Sklearn to modify the data

from sklearn.cross_validation import train_test_split
# os.chdir("provide path")

# input image dimensions
m,n = 50,50

path1="test/"
path2="train/"

classes=os.listdir(path2)
x=[]
y=[]
for fol in classes:
    print(fol)
    imgfiles=os.listdir(path2 + '/' + fol);
    for img in imgfiles:
        im=Image.open(path2+'/'+fol+'/'+img);
        im=im.convert(mode='RGB')
        imrs=im.resize((m,n))
        imrs=img_to_array(imrs)/255;
        imrs=imrs.transpose(2,0,1);
        imrs=imrs.reshape(3,m,n);
        x.append(imrs)
        y.append(fol)
        
x=np.array(x);
y=np.array(y);

batch_size=32
nb_classes=len(classes)
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

model= Sequential()
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool), dim_ordering="th"));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

nb_epoch=1
batch_size=5
history = model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
model.save('firearm_model.h5')

webcam = cv2.VideoCapture(0)
#cv2.namedWindow("Gun Detection", cv2.WINDOW_AUTOSIZE)
files=os.listdir(path1);

for i in range(15):
    #ret, img = webcam.read()
    img=files[i]
    imrs = im.resize((m,n))
    imrs=img_to_array(imrs)/255;
    imrs=imrs.transpose(2,0,1);
    imrs=imrs.reshape(3,m,n);
    x=[]
    x.append(imrs)
    x=np.array(x);
    predictions = model.predict(x)
    print("printing model predictions:   ",predictions)
    print("predictions tpw:  ",  predictions.shape) 
    cv2.putText(img, 'Weapon detection Confidence:  ' + str(predictions),(5, 100), cv2.FONT_HERSHEY_PLAIN, 1, (66, 53, 243), 2)

    cv2.putText(img, "ESC to exit", (5, img.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
    cv2.imshow("Gun Detection", img) # live feed in external
    if cv2.waitKey(40) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
    cv2.waitKey(1000)
webcam.release()
print("printing model summary:  ", model.summary() )