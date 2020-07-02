from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input,Activation,Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle 
import numpy as np

# Data Loading
with open('images.pkl','rb') as f:
  images = pickle.load(f)

with open('labels.pkl','rb') as f:
  labels = pickle.load(f)
  
X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size = 0.3,random_state = 10)

datagen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])
datagen.fit(X_train)

# Model Building
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(224,224,1),padding='valid',activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPool2D(3))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPool2D(3))
model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPool2D(3))
model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(2000))
model.add(Activation('relu'))
model.add(Dense(14,activation='relu'))

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
                                            
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['acc'])

epochs = 10 
batch_size = 10

model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs,verbose = 1, steps_per_epoch=150000,
                              callbacks=[learning_rate_reduction])

model.save("cnn_model.h5")
print(model.evaluate(X_test,y_test))
