# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal


seed = 128
rng = np.random.RandomState(seed)

root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'train')
preview_dir = os.path.join(root_dir,'preview')
train = pd.read_csv(os.path.join(root_dir, 'labels', 'preview.csv'))
test_labels =  pd.read_csv(os.path.join(root_dir, 'labels', 'test_final_1000.csv'))
cv_labels = pd.read_csv(os.path.join(root_dir,'labels','cv_1000.csv'))

size = 512,512
basewidth = 512
filename = train.iloc[:,0]
labels = train.iloc[:,1]
categorical_labels = to_categorical(labels, num_classes=5)
test_level = test_labels.iloc[:,1]
categorical_test_level=to_categorical(test_level,num_classes=5)
cv_level = cv_labels.iloc[:,1]
cv_categorical = to_categorical(cv_level,num_classes=5)


temp = []
ext = '.jpeg'
for img_name in train.image:
    image_path = os.path.join(root_dir,'train','preview', img_name+'.jpeg')
    #img = imread(image_path, flatten=True)
    #img = img.astype('float32')
    img=Image.open(image_path)    
    img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
    #img.thumbnail(size, Image.ANTIALIAS)
    #img.save(img_name+ext, 'JPEG')    
    #img.save(img_name+ext)
    #img.thumbnail(size, Image.ANTIALIAS)
    temp.append(img)
    

train_x = np.stack(temp)
train_x= train_x.astype('float32')

print(train_x.shape)


#for cv data set set
cv_set = []
ext = '.jpeg'
for img_name in cv_labels.image:
    image_path = os.path.join(root_dir, 'train','cv_1000', img_name+'.jpeg')
    #img = imread(image_path, flatten=True)
    #img = img.astype('float32')
    img=Image.open(image_path)    
    img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
    #img.thumbnail(size, Image.ANTIALIAS)
    #img.save(r'D:\data\preview'+img_name+ext, 'JPEG')
    #img.save(img_name+ext)
    
    test_set.append(img)
    

cv_x = np.stack(cv_set)
cv_x= test_x.astype('float32')


#for test set
test_set = []
ext = '.jpeg'
for img_name in test_labels.image:
    image_path = os.path.join(root_dir, 'train','test_final_1000', img_name+'.jpeg')
    #img = imread(image_path, flatten=True)
    #img = img.astype('float32')
    img=Image.open(image_path)    
    img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
    #img.thumbnail(size, Image.ANTIALIAS)
    #img.save(r'D:\data\preview'+img_name+ext, 'JPEG')
    #img.save(img_name+ext)
    
    test_set.append(img)
    

test_x = np.stack(test_set)
test_x= test_x.astype('float32')




#datagen = ImageDataGenerator(rotation_range=360,rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
# fit parameters from data



import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal
import keras


output_size = 512 #could be 128,256 or 512
batch_size = 64  
input_height, input_width = (output_size, output_size) 
num_channels = 3
num_classes =5 
input_shape = (512,512,3)


model = Sequential()

#could use orthogonal as initializer as well and have different init biases
#golot is Xavier initialization

model.add(Conv2D(filters =32,kernel_size=(2,2),strides=(2,2),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),input_shape=input_shape,bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))


model.add(Conv2D(filters =32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Conv2D(filters =64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =64 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Conv2D(filters =128 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =128 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =128 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =128 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Conv2D(filters =256 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =256 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =256 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Conv2D(filters =256 ,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',
                 kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Dropout(0.5, noise_shape=None, seed=7))

model.add(Flatten())

model.add(Dense(units=1024,activation='relu',kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Dense(units=512,activation='relu',kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Dense(units=256,activation='relu',kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Dense(units=128,activation='relu',kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))

model.add(Dense(units=64,activation='relu',kernel_initializer=glorot_normal(seed=7),bias_initializer=keras.initializers.Constant(value=0.1)))


model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

epochs = 5
model.fit(train_x,categorical_labels,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(cv_x,cv_categorical))

#history = model.fit(train_x,categorical_labels,batch_size=batch_size,epochs=epochs,verbose=1)

score = model.evaluate(test_x ,categorical_test_level, verbose=0)

print("done")
